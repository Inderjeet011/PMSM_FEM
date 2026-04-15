#!/usr/bin/env python3
"""
2D PMSM mesh: Gmsh OCC geometry → conforming triangle mesh → optional DOLFINx XDMF.

Pipeline:
  1. Concentric rings + PM sectors + coil sectors (overlapping CAD).
  2. Boolean fragment so shared edges match (conforming FE interfaces).
  3. Tag surfaces by expected area + radial/angular checks (TEAM-30 style).
  4. Background mesh fields: coarsen away from motor, refine air gap / PM / rotor.
  5. Write ``mesh.msh``; if ``MPI.size == 1``, also write ``mesh.xdmf`` + ``.h5``.
"""

import gmsh
import math
import numpy as np
from mpi4py import MPI
from pathlib import Path
from dolfinx.io import XDMFFile

# DOLFINx moved gmsh I/O between ``dolfinx.io.gmshio`` and ``dolfinx.io.gmsh``.
try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio


# --- Configuration (edit geometry here; tags are fixed for the 2D solver) ---

class MotorGeometry:
    """Fixed radial layout and winding layout. All lengths are metres."""

    # Radial dimensions (m)
    r_shaft = 0.001          # Shaft radius: 1 mm
    r_pm_in = 0.034          # PM inner radius: thin magnets near rotor surface
    r_pm_out = 0.036         # PM outer radius
    r_rotor = 0.038          # Rotor outer steel radius: 38 mm
    r_airgap_mid = 0.0385    # Air gap middle: 38.5 mm (split point)
    r_airgap_out = 0.039     # Air gap outer: 39 mm (1 mm total gap)
    r_stator_in = r_airgap_out  # same as outer air-gap radius
    r_coil_in = 0.0395       # inner radius of stator coil sectors
    r_stator_out = 0.057     # Stator outer: 57 mm
    r_outer_air = 0.090      # Outer air boundary: 90 mm
    
    # Magnets: ``n_magnets`` poles around the rotor; arc length fraction of each pole pitch.
    pole_pairs = 5
    n_magnets = 2 * pole_pairs  # 10 magnets
    magnet_coverage = 0.72       # angular fraction of each pole occupied by PM material
    
    # Six stator slots (3-phase × 2 layers); angles in degrees, CCW from +x.
    n_coils = 6
    coil_angles_deg = [30 + i * 60 for i in range(n_coils)]  # e.g. 30°, 90°, …
    coil_half_angle_deg = 15.0  # half-width of each slot (full sector = 2× this)
    
    # Target edge length scale for the coarsest part of field 2 (before Min).
    mesh_resolution = 0.002  # 2.0 mm base resolution


class DomainMarkers:
    """Gmsh physical group IDs (2D surfaces). Must match ``DomainTags`` in ``solve.py``."""

    OUTER_AIR = 1
    ROTOR = 2
    PM_N = 3
    PM_S = 4
    AIRGAP_INNER = 5
    AIRGAP_OUTER = 6
    STATOR = 7
    COIL_BASE = 8  # physical groups COIL_BASE … COIL_BASE + n_coils - 1
    EXTERIOR = 100  # outermost circle (1D curve group) for Az = 0 in ``solve.py``


class PMMotorMeshGenerator:
    """Builds OCC geometry, classifies domains, and drives Gmsh meshing + optional XDMF."""

    def __init__(self, geometry=None):
        self.geom = geometry or MotorGeometry()
        self.markers = DomainMarkers()

        # One magnet sector spans ``mag_span`` rad; ``theta_pole`` is angular pitch of poles.
        self.theta_pole = 2 * math.pi / self.geom.n_magnets
        self.mag_span = self.geom.magnet_coverage * self.theta_pole

        # Arc centre for PM/coil sector edges (origin); set when rank 0 builds geometry.
        self.center_point = None
        self.surfaces = []  # reserved / unused; kept for API stability
        
    def print_info(self):
        """Human-readable summary of radii, markers, and mesh size (rank 0 log)."""
        print("=" * 70)
        print(" PERMANENT MAGNET MOTOR MESH GENERATOR")
        print(" Based on TEAM 30 benchmark techniques")
        print("=" * 70)
        
        print(f"\n📐 Geometry:")
        print(f"   Rotor steel:   {self.geom.r_shaft*1000:.1f} - {self.geom.r_rotor*1000:.1f} mm")
        print(f"   PM (embedded): {self.geom.r_pm_in*1000:.1f} - {self.geom.r_pm_out*1000:.1f} mm")
        print(f"   Air gap:       {self.geom.r_rotor*1000:.1f} - {self.geom.r_airgap_out*1000:.1f} mm")
        print(f"     (split at {self.geom.r_airgap_mid*1000:.1f} mm)")
        print(f"   Stator:        {self.geom.r_stator_in*1000:.1f} - {self.geom.r_stator_out*1000:.1f} mm")
        print(f"   Coil sectors:  {self.geom.r_coil_in*1000:.1f} - {self.geom.r_stator_out*1000:.1f} mm")
        print(f"   Magnets:       {self.geom.n_magnets} (N-S alternating)")
        print(f"   Coils:         {self.geom.n_coils} slots")
        print(f"   Mesh res:      {self.geom.mesh_resolution*1000:.2f} mm")
        
        print(f"\n🏷️  Domain markers:")
        print(f"   Rotor: {self.markers.ROTOR}, PM_N: {self.markers.PM_N}, PM_S: {self.markers.PM_S}")
        print(f"   AirGap: [{self.markers.AIRGAP_INNER}, {self.markers.AIRGAP_OUTER}] (split!)")
        print(f"   Stator: {self.markers.STATOR}, Coils: {self.markers.COIL_BASE}-{self.markers.COIL_BASE+self.geom.n_coils-1}")
    
    def create_circular_layers(self):
        """Define axisymmetric rings: each ``addCircle`` is one closed boundary curve."""
        print(f"\n🔧 Creating circular layers...")
        
        g = self.geom

        # Circle curve tags (single loop each); used as hole boundaries for annular surfaces.
        circles = {
            'outer_air': gmsh.model.occ.addCircle(0, 0, 0, g.r_outer_air),
            'stator': gmsh.model.occ.addCircle(0, 0, 0, g.r_stator_out),
            'airgap_out': gmsh.model.occ.addCircle(0, 0, 0, g.r_airgap_out),
            'airgap_mid': gmsh.model.occ.addCircle(0, 0, 0, g.r_airgap_mid),
            'pm_out': gmsh.model.occ.addCircle(0, 0, 0, g.r_pm_out),
            'pm_in': gmsh.model.occ.addCircle(0, 0, 0, g.r_pm_in),
            'rotor': gmsh.model.occ.addCircle(0, 0, 0, g.r_rotor),
            'shaft': gmsh.model.occ.addCircle(0, 0, 0, g.r_shaft),
        }
        
        # One loop per circle (winding +1); later surfaces use [outer, inner] as hole pairs.
        loops = {k: gmsh.model.occ.addCurveLoop([v]) for k, v in circles.items()}
        
        # Annuli: outer_air = beyond stator; rotor ring excludes shaft hole; air gap split in two.
        rings = {
            'outer_air': gmsh.model.occ.addPlaneSurface([loops['outer_air'], loops['stator']]),
            'stator': gmsh.model.occ.addPlaneSurface([loops['stator'], loops['airgap_out']]),
            'airgap_outer': gmsh.model.occ.addPlaneSurface([loops['airgap_out'], loops['airgap_mid']]),
            'airgap_inner': gmsh.model.occ.addPlaneSurface([loops['airgap_mid'], loops['rotor']]),
            'rotor': gmsh.model.occ.addPlaneSurface([loops['rotor'], loops['shaft']]),
        }
        
        gmsh.model.occ.synchronize()
        
        print(f"   ✅ Circular layers created")
        print(f"   ✅ Air gap: SPLIT into inner/outer for robust meshing")
        
        return circles, rings
    
    def create_pm_sectors(self):
        """One quadrilateral surface per magnet; arcs use ``center_point`` as circle centre."""
        print(f"\n🧲 Creating {self.geom.n_magnets} PM sectors...")
        
        pm_surfaces = []
        
        for k in range(self.geom.n_magnets):
            theta_center = k * self.theta_pole
            theta_start = theta_center - self.mag_span / 2
            theta_end = theta_center + self.mag_span / 2

            # Quadrilateral: radial lines at ends, circle arcs on inner/outer PM radii.
            p1 = gmsh.model.occ.addPoint(
                self.geom.r_pm_in * math.cos(theta_start),
                self.geom.r_pm_in * math.sin(theta_start), 0
            )
            p2 = gmsh.model.occ.addPoint(
                self.geom.r_pm_out * math.cos(theta_start),
                self.geom.r_pm_out * math.sin(theta_start), 0
            )
            p3 = gmsh.model.occ.addPoint(
                self.geom.r_pm_out * math.cos(theta_end),
                self.geom.r_pm_out * math.sin(theta_end), 0
            )
            p4 = gmsh.model.occ.addPoint(
                self.geom.r_pm_in * math.cos(theta_end),
                self.geom.r_pm_in * math.sin(theta_end), 0
            )
            
            # Create edges
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addCircleArc(p2, self.center_point, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addCircleArc(p4, self.center_point, p1)
            
            # Create surface
            loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            surf = gmsh.model.occ.addPlaneSurface([loop])
            pm_surfaces.append((2, surf))
        
        print(f"   ✅ {len(pm_surfaces)} PM sectors created")
        return pm_surfaces
    
    def create_coil_slots(self):
        """Six stator coil sectors; same pattern as PM sectors, different radii."""
        print(f"\n⚡ Creating {self.geom.n_coils} coil slots...")
        
        coil_surfaces = []
        
        for angle_deg in self.geom.coil_angles_deg:
            theta_c = math.radians(angle_deg)
            theta_start = theta_c - math.radians(self.geom.coil_half_angle_deg)
            theta_end = theta_c + math.radians(self.geom.coil_half_angle_deg)
            
            # Create sector with 4 corner points
            p1 = gmsh.model.occ.addPoint(
                self.geom.r_coil_in * math.cos(theta_start),
                self.geom.r_coil_in * math.sin(theta_start), 0
            )
            p2 = gmsh.model.occ.addPoint(
                self.geom.r_stator_out * math.cos(theta_start),
                self.geom.r_stator_out * math.sin(theta_start), 0
            )
            p3 = gmsh.model.occ.addPoint(
                self.geom.r_stator_out * math.cos(theta_end),
                self.geom.r_stator_out * math.sin(theta_end), 0
            )
            p4 = gmsh.model.occ.addPoint(
                self.geom.r_coil_in * math.cos(theta_end),
                self.geom.r_coil_in * math.sin(theta_end), 0
            )
            
            # Create edges
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addCircleArc(p2, self.center_point, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addCircleArc(p4, self.center_point, p1)
            
            # Create surface
            loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            surf = gmsh.model.occ.addPlaneSurface([loop])
            coil_surfaces.append((2, surf))
        
        print(f"   ✅ {len(coil_surfaces)} coil slots created")
        return coil_surfaces
    
    def fragment_domains(self, rings, pm_surfaces, coil_surfaces):
        """Boolean fragment: cuts overlapping surfaces so adjacent regions share edges."""
        print(f"\n🔀 Fragmenting all domains for conforming mesh...")
        
        # Order lists rotor → PMs → air gaps → coils → stator → outer air (all dim-2 OCC tags).
        all_domains = (
            [(2, rings['rotor'])] +
            pm_surfaces +
            [(2, rings['airgap_inner'])] +
            [(2, rings['airgap_outer'])] +
            coil_surfaces +
            [(2, rings['stator'])] +
            [(2, rings['outer_air'])]
        )
        
        # Second return value is a map of old→new entities; not needed here.
        surfaces, _ = gmsh.model.occ.fragment(all_domains, [])
        gmsh.model.occ.synchronize()
        
        print(f"   ✅ Fragmented into {len(surfaces)} surfaces")
        return surfaces
    
    def classify_domains_by_area(self, surfaces):
        """
        After fragmenting, many small surfaces appear. We assign tags by:
          (1) radial + angular position for coils and PMs (avoids area clashes with rings);
          (2) else compare surface ``mass`` (area) to precomputed expectations with ``tol``.
        """
        print(f"\n🔍 Classifying domains by area...")
        
        g = self.geom
        
        # Analytical areas for ring domains; PM/coil pieces use geometry formulas below.
        area_outer_air = math.pi * (g.r_outer_air**2 - g.r_stator_out**2)
        area_rotor = math.pi * (g.r_rotor**2 - g.r_shaft**2) - math.pi * (g.r_pm_out**2 - g.r_pm_in**2) * g.magnet_coverage
        area_pm_single = math.pi * (g.r_pm_out**2 - g.r_pm_in**2) * g.magnet_coverage / g.n_magnets
        area_airgap_inner = math.pi * (g.r_airgap_mid**2 - g.r_rotor**2)
        area_airgap_outer = math.pi * (g.r_airgap_out**2 - g.r_airgap_mid**2)
        area_coil_single = math.pi * (g.r_stator_out**2 - g.r_coil_in**2) * (2*g.coil_half_angle_deg) / 360
        area_stator = math.pi * (g.r_stator_out**2 - g.r_stator_in**2) - g.n_coils * area_coil_single
        
        # Debug print only: expected areas (mm²). Classification uses scalars below, not dict order.
        area_map = {
            area_outer_air: ("OuterAir", self.markers.OUTER_AIR),
            area_rotor: ("Rotor", self.markers.ROTOR),
            area_pm_single: ("PM", None),  # N/S classified by angle
            area_coil_single: ("Coil", None),  # Assigned by angle
            area_airgap_inner: ("AirGap_inner", self.markers.AIRGAP_INNER),
            area_airgap_outer: ("AirGap_outer", self.markers.AIRGAP_OUTER),
            area_stator: ("Stator", self.markers.STATOR),
        }
        
        print(f"\n   Expected areas:")
        for area, (name, _) in area_map.items():
            print(f"     {name:20s}: {area*1e6:.2f} mm²")
        
        # PM pole centres (rad); coil slot centres; small slack on coil angular half-width.
        pm_angles = np.array([k * self.theta_pole for k in range(g.n_magnets)])
        coil_angle_centers = np.array([math.radians(a) for a in g.coil_angles_deg])
        coil_half_span = math.radians(g.coil_half_angle_deg) * 1.05
        
        # Gmsh surface tags (dim 2) per category; airgap stores (tag, INNER|OUTER marker).
        classified = {
            'outer_air': [],
            'rotor': [],
            'pm_n': [],
            'pm_s': [],
            'airgap': [],
            'stator': [],
            'coils': {i: [] for i in range(g.n_coils)}
        }
        
        # Relative area tolerance for ring domains (fragmentation slightly changes areas).
        tol = 0.25
        
        for surf in surfaces:
            mass = gmsh.model.occ.get_mass(surf[0], surf[1])
            com = gmsh.model.occ.get_center_of_mass(surf[0], surf[1])
            r_com = math.sqrt(com[0]**2 + com[1]**2)
            theta_com = math.atan2(com[1], com[0])
            if theta_com < 0:
                theta_com += 2 * math.pi
            
            matched = False

            # First classify sector-like domains (coils and PMs) by radial+angular location.
            # This avoids confusion with similarly-sized air-gap fragments.
            if g.r_coil_in <= r_com <= g.r_stator_out:
                diffs = np.abs(coil_angle_centers - theta_com)
                diffs = np.minimum(diffs, 2 * math.pi - diffs)
                closest_idx = np.argmin(diffs)
                if diffs[closest_idx] <= coil_half_span:
                    classified['coils'][closest_idx].append(surf[1])
                    matched = True
            elif g.r_pm_in <= r_com <= g.r_pm_out:
                diffs = np.abs(pm_angles - theta_com)
                diffs = np.minimum(diffs, 2 * math.pi - diffs)
                closest_idx = np.argmin(diffs)
                # Alternate poles: even index → N, odd → S (magnet order around the rotor).
                if closest_idx % 2 == 0:
                    classified['pm_n'].append(surf[1])
                else:
                    classified['pm_s'].append(surf[1])
                matched = True

            # Then classify ring-like domains by area (centroid can be near origin).
            if matched:
                continue

            ring_candidates = [
                (area_outer_air, "OuterAir", self.markers.OUTER_AIR),
                (area_rotor, "Rotor", self.markers.ROTOR),
                (area_airgap_inner, "AirGap_inner", self.markers.AIRGAP_INNER),
                (area_airgap_outer, "AirGap_outer", self.markers.AIRGAP_OUTER),
                (area_stator, "Stator", self.markers.STATOR),
            ]
            for expected_area, domain_name, marker in ring_candidates:
                if abs(mass - expected_area) / max(expected_area, 1e-18) < tol:
                    if domain_name == "OuterAir":
                        classified['outer_air'].append(surf[1])
                    elif domain_name == "Rotor":
                        classified['rotor'].append(surf[1])
                    elif "AirGap" in domain_name:
                        classified['airgap'].append((surf[1], marker))
                    elif domain_name == "Stator":
                        classified['stator'].append(surf[1])
                    matched = True
                    break
        
        # Print classification results
        print(f"\n   Classification results:")
        print(f"     Outer Air: {len(classified['outer_air'])} surfaces")
        print(f"     Rotor:     {len(classified['rotor'])} surfaces")
        print(f"     PM North:  {len(classified['pm_n'])} surfaces")
        print(f"     PM South:  {len(classified['pm_s'])} surfaces")
        print(f"     Air gap:   {len(classified['airgap'])} surfaces")
        print(f"     Stator:    {len(classified['stator'])} surfaces")
        total_coils = sum(len(surfs) for surfs in classified['coils'].values())
        print(f"     Coils:     {total_coils} surfaces ({g.n_coils} slots)")
        
        return classified
    
    def create_physical_groups(self, classified, outer_air_circle):
        """Attach integer physical groups (volumes unused in 2D: dim=2 surfaces, dim=1 exterior arc)."""
        print(f"\n🏷️  Creating physical groups...")
        
        m = self.markers
        
        # Surface groups (tag 2 = 2D elements in Gmsh).
        if classified['outer_air']:
            gmsh.model.addPhysicalGroup(2, classified['outer_air'], m.OUTER_AIR)
            gmsh.model.setPhysicalName(2, m.OUTER_AIR, "OUTER_AIR")
        
        # Rotor
        if classified['rotor']:
            gmsh.model.addPhysicalGroup(2, classified['rotor'], m.ROTOR)
            gmsh.model.setPhysicalName(2, m.ROTOR, "ROTOR")
        
        # Permanent magnets
        if classified['pm_n']:
            gmsh.model.addPhysicalGroup(2, classified['pm_n'], m.PM_N)
            gmsh.model.setPhysicalName(2, m.PM_N, "PM_N")
        
        if classified['pm_s']:
            gmsh.model.addPhysicalGroup(2, classified['pm_s'], m.PM_S)
            gmsh.model.setPhysicalName(2, m.PM_S, "PM_S")
        
        # Air gap: classifier stored marker so inner/outer split survives tagging.
        if classified['airgap']:
            airgap_inner_list = [s for s, marker in classified['airgap'] if marker == m.AIRGAP_INNER]
            airgap_outer_list = [s for s, marker in classified['airgap'] if marker == m.AIRGAP_OUTER]
            
            if airgap_inner_list:
                gmsh.model.addPhysicalGroup(2, airgap_inner_list, m.AIRGAP_INNER)
                gmsh.model.setPhysicalName(2, m.AIRGAP_INNER, "AIR_GAP")
            
            if airgap_outer_list:
                gmsh.model.addPhysicalGroup(2, airgap_outer_list, m.AIRGAP_OUTER)
                gmsh.model.setPhysicalName(2, m.AIRGAP_OUTER, "AIR_GAP_OUTER")
        
        # Stator
        if classified['stator']:
            gmsh.model.addPhysicalGroup(2, classified['stator'], m.STATOR)
            gmsh.model.setPhysicalName(2, m.STATOR, "STATOR")
        
        # Coils
        for i, surfs in classified['coils'].items():
            if surfs:
                gmsh.model.addPhysicalGroup(2, surfs, m.COIL_BASE + i)
                gmsh.model.setPhysicalName(2, m.COIL_BASE + i, f"COIL_{i}")
        
        # Dim-1 group on the outer air circle (Gmsh curve id ``outer_air_circle``).
        gmsh.model.occ.synchronize()
        if classified['outer_air']:
            exterior_curve = gmsh.model.getBoundary(
                [(2, classified['outer_air'][0])], oriented=False
            )
            # ``outer_air_circle`` is the circle curve id from ``circles['outer_air']``.
            exterior_lines = [abs(tag) for dim, tag in exterior_curve if abs(tag) == outer_air_circle]
            if exterior_lines:
                gmsh.model.addPhysicalGroup(1, [outer_air_circle], m.EXTERIOR)
                gmsh.model.setPhysicalName(1, m.EXTERIOR, "EXTERIOR")
        
        print(f"   ✅ Physical groups created")
    
    def setup_mesh_refinement(self):
        """Background size field: combine several threshold fields with ``Min`` (finest wins)."""
        print(f"\n⚙️  Setting up mesh refinement...")
        
        g = self.geom
        res = g.mesh_resolution
        
        # Fields 1–2: distance from origin → threshold → larger cells far from the motor.
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [self.center_point])
        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 10 * res)
        gmsh.model.mesh.field.setNumber(2, "DistMin", g.r_airgap_out)
        gmsh.model.mesh.field.setNumber(2, "DistMax", g.r_stator_out * 1.5)
        
        # Fields 4–5: distance to mid-air-gap circle → small ``Lc`` inside the gap band.
        r_airgap_center = (g.r_rotor + g.r_airgap_out) / 2
        r_airgap_halfwidth = (g.r_airgap_out - g.r_rotor) / 2
        
        gmsh.model.mesh.field.add("MathEval", 4)
        gmsh.model.mesh.field.setString(4, "F", 
            f"abs(sqrt(x^2 + y^2) - {r_airgap_center})")
        
        gmsh.model.mesh.field.add("Threshold", 5)
        gmsh.model.mesh.field.setNumber(5, "IField", 4)
        gmsh.model.mesh.field.setNumber(5, "LcMin", res * 0.3)
        gmsh.model.mesh.field.setNumber(5, "LcMax", res * 2)
        gmsh.model.mesh.field.setNumber(5, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(5, "DistMax", r_airgap_halfwidth * 1.5)
        
        # Fields 6–7: narrow band around PM outer radius for sharp material change.
        gmsh.model.mesh.field.add("MathEval", 6)
        gmsh.model.mesh.field.setString(6, "F", f"abs(sqrt(x^2 + y^2) - {g.r_pm_out})")
        
        gmsh.model.mesh.field.add("Threshold", 7)
        gmsh.model.mesh.field.setNumber(7, "IField", 6)
        gmsh.model.mesh.field.setNumber(7, "LcMin", res * 0.25)
        gmsh.model.mesh.field.setNumber(7, "LcMax", res * 0.5)
        gmsh.model.mesh.field.setNumber(7, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(7, "DistMax", 0.001)
        
        # Fields 8–9: band on rotor outer radius (motion / conductivity interface in the solver).
        gmsh.model.mesh.field.add("MathEval", 8)
        gmsh.model.mesh.field.setString(8, "F", f"abs(sqrt(x^2 + y^2) - {g.r_rotor})")
        
        gmsh.model.mesh.field.add("Threshold", 9)
        gmsh.model.mesh.field.setNumber(9, "IField", 8)
        gmsh.model.mesh.field.setNumber(9, "LcMin", res * 0.3)
        gmsh.model.mesh.field.setNumber(9, "LcMax", res * 2)
        gmsh.model.mesh.field.setNumber(9, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(9, "DistMax", 0.002)
        
        # Field 3: pointwise minimum of fields 2,5,7,9 → one global sizing field.
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2, 5, 7, 9])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)
        
        print(f"   ✅ Air gap: Uniform fine mesh ({res*0.3*1000:.2f} mm)")
        print(f"   ✅ PM boundaries: Extra fine ({res*0.25*1000:.2f} mm)")
    
    def generate_mesh(self, output_file=None):
        """Run Gmsh on rank 0; optional DOLFINx export on single process after ``finalize``."""
        if output_file is None:
            output_file = Path(__file__).resolve().parent / "mesh.msh"
        output_path = Path(output_file).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.print_info()
        
        gmsh.initialize()
        gmsh.model.add("pm_motor")
        
        # Geometry + meshing are sequential OCC; keep on one MPI rank to avoid duplicate tags.
        if MPI.COMM_WORLD.rank == 0:
            self.center_point = gmsh.model.occ.addPoint(0, 0, 0)
            
            circles, rings = self.create_circular_layers()
            pm_surfaces = self.create_pm_sectors()
            coil_surfaces = self.create_coil_slots()
            
            surfaces = self.fragment_domains(rings, pm_surfaces, coil_surfaces)
            classified = self.classify_domains_by_area(surfaces)
            
            self.create_physical_groups(classified, circles['outer_air'])
            self.setup_mesh_refinement()
            
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.mesh.generate(2)  # 2D triangular mesh in the z=0 plane
            
            nodes = len(gmsh.model.mesh.getNodes()[0])
            elements = len(gmsh.model.mesh.getElements()[1][0])  # first family: triangles
            
            print(f"\n✅ Mesh generated:")
            print(f"   Nodes:    {nodes}")
            print(f"   Elements: {elements}")
            
            gmsh.write(str(output_path))
            print(f"   💾 Saved: {output_path}")
        
        # All ranks must finalize Gmsh even if only rank 0 built the model.
        MPI.COMM_WORLD.Barrier()
        gmsh.finalize()
        
        print("\n" + "=" * 70)
        print(" ✅ MESH GENERATION COMPLETE")
        print(" Key techniques applied:")
        print("   - Air gap split into 2 regions")
        print("   - Area-based classification")
        print("   - Fragment-based conforming mesh")
        print("   - Distance field refinement")
        print("=" * 70)

        # DOLFINx path: re-read ``.msh`` and write XDMF+H5 (same basename). Multi-rank skipped:
        # gmshio expects a consistent parallel mesh read; this script is single-rank for 2D.
        if MPI.COMM_WORLD.size == 1:
            msh_path = output_path
            xdmf_path = output_path.with_suffix(".xdmf")
            mesh_data = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, rank=0, gdim=2)
            mesh = mesh_data[0]
            ct = mesh_data[1] if len(mesh_data) > 1 else None
            ft = mesh_data[2] if len(mesh_data) > 2 else None
            with XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "w") as xdmf:
                xdmf.write_mesh(mesh)
                if ct is not None:
                    # Naming optional; helps ParaView / downstream scripts.
                    try:
                        ct.name = "cell_tags"
                    except Exception:
                        pass
                    xdmf.write_meshtags(ct, mesh.geometry)
                if ft is not None:
                    try:
                        ft.name = "facet_tags"
                    except Exception:
                        pass
                    xdmf.write_meshtags(ft, mesh.geometry)
            if MPI.COMM_WORLD.rank == 0:
                print(f"   💾 Saved: {xdmf_path} (+ {xdmf_path.with_suffix('.h5')})")


if __name__ == "__main__":
    generator = PMMotorMeshGenerator()
    generator.generate_mesh()
