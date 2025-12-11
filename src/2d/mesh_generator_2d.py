#!/usr/bin/env python3
"""
Permanent Magnet Motor Mesh Generator
=====================================
- Area-based domain classification
- Air gap splitting for robust meshing
- Fragment-based conforming mesh
- Distance field refinement
"""

import gmsh
import math
import numpy as np
from mpi4py import MPI





# ============================================================================
# CONFIGURATION
# ============================================================================

class MotorGeometry:
    """Geometric parameters for PM motor"""
    # Radial dimensions (in meters)
    r_shaft = 0.001          # Shaft radius: 1 mm
    r_rotor = 0.030          # Rotor outer: 30 mm
    r_pm_out = 0.038         # PM outer: 38 mm (PM thickness = 8 mm)
    r_airgap_mid = 0.039     # Air gap middle: 39 mm (split point)
    r_airgap_out = 0.040     # Air gap outer: 40 mm (2 mm total gap)
    r_stator_in = r_airgap_out
    r_coil_out = 0.050       # Coil depth: 50 mm
    r_stator_out = 0.057     # Stator outer: 57 mm
    r_outer_air = 0.090      # Outer air boundary: 90 mm
    
    # Magnet configuration
    pole_pairs = 4
    n_magnets = 2 * pole_pairs  # 8 magnets
    magnet_coverage = 0.8        # 80% coverage per pole
    
    # Coil configuration
    n_coils = 6
    coil_angles_deg = [i * 60 for i in range(n_coils)]  # 0°, 60°, 120°, 180°, 240°, 300°
    coil_half_angle_deg = 7.5  # ±7.5° = 15° slot width
    
    # Mesh resolution
    mesh_resolution = 0.002  # 2.0 mm base resolution


class DomainMarkers:
    """Physical domain markers for FEM"""
    OUTER_AIR = 1
    ROTOR = 2
    PM_N = 3
    PM_S = 4
    AIRGAP_INNER = 5
    AIRGAP_OUTER = 6
    STATOR = 7
    COIL_BASE = 8  # Coils: 8, 9, 10, 11, 12, 13
    EXTERIOR = 100  # Outer boundary for BC





# ============================================================================
# MESH GENERATOR CLASS
# ============================================================================

class PMMotorMeshGenerator:
    """Professional PM motor mesh generator"""
    
    def __init__(self, geometry=None):
        self.geom = geometry or MotorGeometry()
        self.markers = DomainMarkers()
        
        # Derived parameters
        self.theta_pole = 2 * math.pi / self.geom.n_magnets
        self.mag_span = self.geom.magnet_coverage * self.theta_pole
        
        # Storage for mesh entities
        self.center_point = None
        self.surfaces = []
        
    def print_info(self):
        """Print geometry configuration"""
        print("=" * 70)
        print(" PERMANENT MAGNET MOTOR MESH GENERATOR")
        print(" Based on TEAM 30 benchmark techniques")
        print("=" * 70)
        
        print(f"\n📐 Geometry:")
        print(f"   Rotor:         {self.geom.r_shaft*1000:.1f} - {self.geom.r_rotor*1000:.1f} mm")
        print(f"   PM:            {self.geom.r_rotor*1000:.1f} - {self.geom.r_pm_out*1000:.1f} mm")
        print(f"   Air gap:       {self.geom.r_pm_out*1000:.1f} - {self.geom.r_airgap_out*1000:.1f} mm")
        print(f"     (split at {self.geom.r_airgap_mid*1000:.1f} mm)")
        print(f"   Stator:        {self.geom.r_stator_in*1000:.1f} - {self.geom.r_stator_out*1000:.1f} mm")
        print(f"   Magnets:       {self.geom.n_magnets} (N-S alternating)")
        print(f"   Coils:         {self.geom.n_coils} slots")
        print(f"   Mesh res:      {self.geom.mesh_resolution*1000:.2f} mm")
        
        print(f"\n🏷️  Domain markers:")
        print(f"   Rotor: {self.markers.ROTOR}, PM_N: {self.markers.PM_N}, PM_S: {self.markers.PM_S}")
        print(f"   AirGap: [{self.markers.AIRGAP_INNER}, {self.markers.AIRGAP_OUTER}] (split!)")
        print(f"   Stator: {self.markers.STATOR}, Coils: {self.markers.COIL_BASE}-{self.markers.COIL_BASE+self.geom.n_coils-1}")
    
    def create_circular_layers(self):
        """Create concentric circular layers"""
        print(f"\n🔧 Creating circular layers...")
        
        g = self.geom
        
        # Create circles from outer to inner
        circles = {
            'outer_air': gmsh.model.occ.addCircle(0, 0, 0, g.r_outer_air),
            'stator': gmsh.model.occ.addCircle(0, 0, 0, g.r_stator_out),
            'coil': gmsh.model.occ.addCircle(0, 0, 0, g.r_coil_out),
            'airgap_out': gmsh.model.occ.addCircle(0, 0, 0, g.r_airgap_out),
            'airgap_mid': gmsh.model.occ.addCircle(0, 0, 0, g.r_airgap_mid),
            'pm': gmsh.model.occ.addCircle(0, 0, 0, g.r_pm_out),
            'rotor': gmsh.model.occ.addCircle(0, 0, 0, g.r_rotor),
            'shaft': gmsh.model.occ.addCircle(0, 0, 0, g.r_shaft),
        }
        
        # Create curve loops
        loops = {k: gmsh.model.occ.addCurveLoop([v]) for k, v in circles.items()}
        
        # Create ring surfaces
        rings = {
            'outer_air': gmsh.model.occ.addPlaneSurface([loops['outer_air'], loops['stator']]),
            'stator': gmsh.model.occ.addPlaneSurface([loops['stator'], loops['coil']]),
            'coil_region': gmsh.model.occ.addPlaneSurface([loops['coil'], loops['airgap_out']]),
            'airgap_outer': gmsh.model.occ.addPlaneSurface([loops['airgap_out'], loops['airgap_mid']]),
            'airgap_inner': gmsh.model.occ.addPlaneSurface([loops['airgap_mid'], loops['pm']]),
            'rotor': gmsh.model.occ.addPlaneSurface([loops['rotor'], loops['shaft']]),
        }
        
        gmsh.model.occ.synchronize()
        
        print(f"   ✅ Circular layers created")
        print(f"   ✅ Air gap: SPLIT into inner/outer for robust meshing")
        
        return circles, rings
    
    def create_pm_sectors(self):
        """Create permanent magnet sectors"""
        print(f"\n🧲 Creating {self.geom.n_magnets} PM sectors...")
        
        pm_surfaces = []
        
        for k in range(self.geom.n_magnets):
            theta_center = k * self.theta_pole
            theta_start = theta_center - self.mag_span / 2
            theta_end = theta_center + self.mag_span / 2
            
            # Create sector with 4 corner points
            p1 = gmsh.model.occ.addPoint(
                self.geom.r_rotor * math.cos(theta_start),
                self.geom.r_rotor * math.sin(theta_start), 0
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
                self.geom.r_rotor * math.cos(theta_end),
                self.geom.r_rotor * math.sin(theta_end), 0
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
        """Create coil slot sectors"""
        print(f"\n⚡ Creating {self.geom.n_coils} coil slots...")
        
        coil_surfaces = []
        
        for angle_deg in self.geom.coil_angles_deg:
            theta_c = math.radians(angle_deg)
            theta_start = theta_c - math.radians(self.geom.coil_half_angle_deg)
            theta_end = theta_c + math.radians(self.geom.coil_half_angle_deg)
            
            # Create sector with 4 corner points
            p1 = gmsh.model.occ.addPoint(
                self.geom.r_stator_in * math.cos(theta_start),
                self.geom.r_stator_in * math.sin(theta_start), 0
            )
            p2 = gmsh.model.occ.addPoint(
                self.geom.r_coil_out * math.cos(theta_start),
                self.geom.r_coil_out * math.sin(theta_start), 0
            )
            p3 = gmsh.model.occ.addPoint(
                self.geom.r_coil_out * math.cos(theta_end),
                self.geom.r_coil_out * math.sin(theta_end), 0
            )
            p4 = gmsh.model.occ.addPoint(
                self.geom.r_stator_in * math.cos(theta_end),
                self.geom.r_stator_in * math.sin(theta_end), 0
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
        """Fragment all domains for conforming mesh"""
        print(f"\n🔀 Fragmenting all domains for conforming mesh...")
        
        all_domains = (
            [(2, rings['rotor'])] +
            pm_surfaces +
            [(2, rings['airgap_inner'])] +
            [(2, rings['airgap_outer'])] +
            [(2, rings['coil_region'])] +
            coil_surfaces +
            [(2, rings['stator'])] +
            [(2, rings['outer_air'])]
        )
        
        surfaces, _ = gmsh.model.occ.fragment(all_domains, [])
        gmsh.model.occ.synchronize()
        
        print(f"   ✅ Fragmented into {len(surfaces)} surfaces")
        return surfaces
    
    def classify_domains_by_area(self, surfaces):
        """Classify fragmented domains by area (TEAM 30 technique)"""
        print(f"\n🔍 Classifying domains by area...")
        
        g = self.geom
        
        # Calculate expected areas for each domain type
        area_outer_air = math.pi * (g.r_outer_air**2 - g.r_stator_out**2)
        area_rotor = math.pi * (g.r_rotor**2 - g.r_shaft**2)
        area_pm_single = math.pi * (g.r_pm_out**2 - g.r_rotor**2) * g.magnet_coverage / g.n_magnets
        area_airgap_inner = math.pi * (g.r_airgap_mid**2 - g.r_pm_out**2)
        area_airgap_outer = math.pi * (g.r_airgap_out**2 - g.r_airgap_mid**2)
        area_stator = math.pi * (g.r_stator_out**2 - g.r_coil_out**2)
        area_coil_single = math.pi * (g.r_coil_out**2 - g.r_stator_in**2) * (2*g.coil_half_angle_deg) / 360
        
        # Build area-to-domain map (order matters - check specific areas first)
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
        
        # Precompute PM and coil angle centers
        pm_angles = np.array([k * self.theta_pole for k in range(g.n_magnets)])
        coil_angle_centers = np.array([math.radians(a) for a in g.coil_angles_deg])
        
        # Storage for classified surfaces
        classified = {
            'outer_air': [],
            'rotor': [],
            'pm_n': [],
            'pm_s': [],
            'airgap': [],
            'stator': [],
            'coils': {i: [] for i in range(g.n_coils)}
        }
        
        # Classify each surface
        tol = 0.25  # 25% tolerance for area matching
        
        for surf in surfaces:
            mass = gmsh.model.occ.get_mass(surf[0], surf[1])
            com = gmsh.model.occ.get_center_of_mass(surf[0], surf[1])
            r_com = math.sqrt(com[0]**2 + com[1]**2)
            theta_com = math.atan2(com[1], com[0])
            if theta_com < 0:
                theta_com += 2 * math.pi
            
            matched = False
            
            # Try to match to expected areas
            for expected_area, (domain_name, marker) in area_map.items():
                if abs(mass - expected_area) / expected_area < tol:
                    if domain_name == "OuterAir":
                        classified['outer_air'].append(surf[1])
                        matched = True
                        break
                    elif domain_name == "Rotor":
                        classified['rotor'].append(surf[1])
                        matched = True
                        break
                    elif domain_name == "PM":
                        # Classify N/S by angle
                        diffs = np.abs(pm_angles - theta_com)
                        diffs = np.minimum(diffs, 2*math.pi - diffs)
                        closest_idx = np.argmin(diffs)
                        if closest_idx % 2 == 0:
                            classified['pm_n'].append(surf[1])
                        else:
                            classified['pm_s'].append(surf[1])
                        matched = True
                        break
                    elif "AirGap" in domain_name:
                        classified['airgap'].append((surf[1], marker))
                        matched = True
                        break
                    elif domain_name == "Stator":
                        classified['stator'].append(surf[1])
                        matched = True
                        break
                    elif domain_name == "Coil":
                        # Assign to closest coil by angle
                        diffs = np.abs(coil_angle_centers - theta_com)
                        diffs = np.minimum(diffs, 2*math.pi - diffs)
                        closest_idx = np.argmin(diffs)
                        classified['coils'][closest_idx].append(surf[1])
                        matched = True
                        break
            
            # Fallback: classify by radial position
            if not matched:
                if r_com > g.r_stator_out * 1.1:
                    classified['outer_air'].append(surf[1])
                elif r_com < g.r_rotor * 1.1:
                    classified['rotor'].append(surf[1])
                elif r_com < g.r_pm_out * 1.1:
                    # Small PM fragment
                    diffs = np.abs(pm_angles - theta_com)
                    diffs = np.minimum(diffs, 2*math.pi - diffs)
                    closest_idx = np.argmin(diffs)
                    if closest_idx % 2 == 0:
                        classified['pm_n'].append(surf[1])
                    else:
                        classified['pm_s'].append(surf[1])
                elif r_com < g.r_airgap_out * 1.1:
                    # Air gap fragment
                    if r_com < g.r_airgap_mid:
                        classified['airgap'].append((surf[1], self.markers.AIRGAP_INNER))
                    else:
                        classified['airgap'].append((surf[1], self.markers.AIRGAP_OUTER))
                else:
                    classified['stator'].append(surf[1])
        
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
        """Create physical groups for FEM"""
        print(f"\n🏷️  Creating physical groups...")
        
        m = self.markers
        
        # Outer air
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
        
        # Air gap (split into inner/outer)
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
        
        # Exterior boundary for BC
        gmsh.model.occ.synchronize()
        if classified['outer_air']:
            exterior_curve = gmsh.model.getBoundary(
                [(2, classified['outer_air'][0])], oriented=False
            )
            exterior_lines = [abs(tag) for dim, tag in exterior_curve if abs(tag) == outer_air_circle]
            if exterior_lines:
                gmsh.model.addPhysicalGroup(1, [outer_air_circle], m.EXTERIOR)
                gmsh.model.setPhysicalName(1, m.EXTERIOR, "EXTERIOR")
        
        print(f"   ✅ Physical groups created")
    
    def setup_mesh_refinement(self):
        """Setup distance field refinement (TEAM 30 technique)"""
        print(f"\n⚙️  Setting up mesh refinement...")
        
        g = self.geom
        res = g.mesh_resolution
        
        # Distance field from center
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [self.center_point])
        
        # Threshold field - coarse far from center
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 10 * res)
        gmsh.model.mesh.field.setNumber(2, "DistMin", g.r_airgap_out)
        gmsh.model.mesh.field.setNumber(2, "DistMax", g.r_stator_out * 1.5)
        
        # Air gap region refinement (CRITICAL!)
        r_airgap_center = (g.r_pm_out + g.r_airgap_out) / 2
        r_airgap_halfwidth = (g.r_airgap_out - g.r_pm_out) / 2
        
        gmsh.model.mesh.field.add("MathEval", 4)
        gmsh.model.mesh.field.setString(4, "F", 
            f"abs(sqrt(x^2 + y^2) - {r_airgap_center})")
        
        gmsh.model.mesh.field.add("Threshold", 5)
        gmsh.model.mesh.field.setNumber(5, "IField", 4)
        gmsh.model.mesh.field.setNumber(5, "LcMin", res * 0.3)
        gmsh.model.mesh.field.setNumber(5, "LcMax", res * 2)
        gmsh.model.mesh.field.setNumber(5, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(5, "DistMax", r_airgap_halfwidth * 1.5)
        
        # PM outer boundary refinement
        gmsh.model.mesh.field.add("MathEval", 6)
        gmsh.model.mesh.field.setString(6, "F", f"abs(sqrt(x^2 + y^2) - {g.r_pm_out})")
        
        gmsh.model.mesh.field.add("Threshold", 7)
        gmsh.model.mesh.field.setNumber(7, "IField", 6)
        gmsh.model.mesh.field.setNumber(7, "LcMin", res * 0.25)
        gmsh.model.mesh.field.setNumber(7, "LcMax", res * 0.5)
        gmsh.model.mesh.field.setNumber(7, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(7, "DistMax", 0.001)
        
        # PM inner boundary refinement
        gmsh.model.mesh.field.add("MathEval", 8)
        gmsh.model.mesh.field.setString(8, "F", f"abs(sqrt(x^2 + y^2) - {g.r_rotor})")
        
        gmsh.model.mesh.field.add("Threshold", 9)
        gmsh.model.mesh.field.setNumber(9, "IField", 8)
        gmsh.model.mesh.field.setNumber(9, "LcMin", res * 0.3)
        gmsh.model.mesh.field.setNumber(9, "LcMax", res * 2)
        gmsh.model.mesh.field.setNumber(9, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(9, "DistMax", 0.002)
        
        # Combine all fields (minimum = finest)
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2, 5, 7, 9])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)
        
        print(f"   ✅ Air gap: Uniform fine mesh ({res*0.3*1000:.2f} mm)")
        print(f"   ✅ PM boundaries: Extra fine ({res*0.25*1000:.2f} mm)")
    
    def generate_mesh(self, output_file="../../meshes/2d/motor.msh"):
        """Main mesh generation workflow"""
        self.print_info()
        
        # Initialize Gmsh
        gmsh.initialize()
        gmsh.model.add("pm_motor")
        
        if MPI.COMM_WORLD.rank == 0:
            # Create center point
            self.center_point = gmsh.model.occ.addPoint(0, 0, 0)
            
            # Create geometry
            circles, rings = self.create_circular_layers()
            pm_surfaces = self.create_pm_sectors()
            coil_surfaces = self.create_coil_slots()
            
            # Fragment and classify
            surfaces = self.fragment_domains(rings, pm_surfaces, coil_surfaces)
            classified = self.classify_domains_by_area(surfaces)
            
            # Create physical groups
            self.create_physical_groups(classified, circles['outer_air'])
            
            # Setup mesh refinement
            self.setup_mesh_refinement()
            
            # Generate mesh
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.mesh.generate(2)
            
            # Get statistics
            nodes = len(gmsh.model.mesh.getNodes()[0])
            elements = len(gmsh.model.mesh.getElements()[1][0])
            
            print(f"\n✅ Mesh generated:")
            print(f"   Nodes:    {nodes}")
            print(f"   Elements: {elements}")
            
            # Save mesh
            gmsh.write(output_file)
            print(f"   💾 Saved: {output_file}")
        
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






# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    generator = PMMotorMeshGenerator()
    generator.generate_mesh(output_file="../../meshes/2d/motor.msh")

