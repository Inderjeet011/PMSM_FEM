# Generate the three phase PMSM model, with a given minimal resolution, encapsulated in a LxL box.

import argparse
from pathlib import Path
from typing import Dict, Union
import numpy as np
from mpi4py import MPI
import gmsh
import dolfinx

__all__ = ["model_parameters", "mesh_parameters", "surface_map"]

# ---------------------------------------------------------------------------
# Model and mesh parameters
# ---------------------------------------------------------------------------

model_parameters = {
    "mu_0": 1.25663753e-6,  # [H/m]
    "freq": 60,  # Hz
    "J": 3.1e6 * np.sqrt(2),  # [A/m^2]
    "mu_r": {
        "Cu": 1,
        "Stator": 30,
        "Rotor": 30,
        "Al": 1,
        "Air": 1,
        "AirGap": 1,
        "PM": 1.04457,
    },
    "sigma": {
        "Rotor": 1.6e6,
        "Al": 3.72e7,
        "Stator": 0,
        "Cu": 0,
        "Air": 0,
        "AirGap": 0,
        "PM": 6.25e5,
    },
    "densities": {
        "Rotor": 7850,
        "Al": 2700,
        "Stator": 0,
        "Air": 0,
        "Cu": 0,
        "AirGap": 0,
        "PM": 7500,
    },
}

# Facet markers
surface_map: Dict[str, Union[int, str]] = {"Exterior": 1, "MidAir": 2, "restriction": "+"}

# Volume markers (3D physical groups)
_domain_map_three: Dict[str, tuple[int, ...]] = {
    "Air": (1,),
    "AirGap": (2, 3),
    "Al": (4,),
    "Rotor": (5,),
    "Stator": (6,),
    "Cu": (7, 8, 9, 10, 11, 12),
    "PM": (13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
}

# Currents mapping to Cu markers (you already had this)
_currents_three: Dict[int, Dict[str, float]] = {
    7: {"alpha": 1, "beta": 0},
    8: {"alpha": -1, "beta": 2 * np.pi / 3},
    9: {"alpha": 1, "beta": 4 * np.pi / 3},
    10: {"alpha": -1, "beta": 0},
    11: {"alpha": 1, "beta": 2 * np.pi / 3},
    12: {"alpha": -1, "beta": 4 * np.pi / 3},
}

# Radii (kept from your script)
mesh_parameters: Dict[str, float] = {
    "r1": 0.017,  # shaft
    "r2": 0.04,
    "r3": 0.042,  # rotor outer
    "r4": 0.062,  # stator inner
    "r5": 0.075,  # stator outer
    "r6": 0.036,  # PM inner radius
    "r7": 0.038,  # PM outer radius
    # NOTE: We model the true air-gap as the thin annulus [r3, r3 + air_gap]
    # The slot/air region [r3 + air_gap, r4] contains copper windings
    "air_gap": 0.002,  # 2mm physical air-gap
}


# ---------------------------------------------------------------------------
# Helper functions for 2D geometry
# ---------------------------------------------------------------------------

def _add_copper_segment(angle: float, center: int) -> int:
    """
    Add a copper segment at a given angle (2D surface at z=0).
    Copper slots live outside the true air-gap annulus to preserve their
    thickness/appearance while allowing a small physical air-gap.
    """
    r3 = mesh_parameters["r3"]
    r_gap = r3 + mesh_parameters["air_gap"]
    r4 = mesh_parameters["r4"]

    # Angular half-width of the slot = 22.5 deg (π/8) – same as original
    dphi = np.pi / 8

    # Create explicit points for inner and outer arcs
    # Inner radius is now r_gap (not r3) to preserve slot thickness
    p_i0 = gmsh.model.occ.addPoint(r_gap * np.cos(angle - dphi), r_gap * np.sin(angle - dphi), 0.0)
    p_i1 = gmsh.model.occ.addPoint(r_gap * np.cos(angle + dphi), r_gap * np.sin(angle + dphi), 0.0)
    p_o0 = gmsh.model.occ.addPoint(r4 * np.cos(angle - dphi), r4 * np.sin(angle - dphi), 0.0)
    p_o1 = gmsh.model.occ.addPoint(r4 * np.cos(angle + dphi), r4 * np.sin(angle + dphi), 0.0)

    # Create arcs using addCircleArc with shared center point
    arc_inner = gmsh.model.occ.addCircleArc(p_i0, center, p_i1)
    arc_outer = gmsh.model.occ.addCircleArc(p_o0, center, p_o1)

    # Create side lines
    side1 = gmsh.model.occ.addLine(p_i0, p_o0)
    side2 = gmsh.model.occ.addLine(p_i1, p_o1)

    # Create curve loop and surface
    loop = gmsh.model.occ.addCurveLoop([arc_inner, side2, arc_outer, side1])
    copper_segment = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    return copper_segment


def _add_permanent_magnets(angle: float, center: int) -> int:
    """
    Add a permanent magnet at a given angle (2D surface at z=0).
    Magnets lie in annular region [r6, r7].
    """
    r6 = mesh_parameters["r6"]
    r7 = mesh_parameters["r7"]

    dphi = np.pi / 12  # 15 deg half-width => 30 deg magnet

    # Create explicit points for inner and outer arcs
    p_i0 = gmsh.model.occ.addPoint(r6 * np.cos(angle - dphi), r6 * np.sin(angle - dphi), 0.0)
    p_i1 = gmsh.model.occ.addPoint(r6 * np.cos(angle + dphi), r6 * np.sin(angle + dphi), 0.0)
    p_o0 = gmsh.model.occ.addPoint(r7 * np.cos(angle - dphi), r7 * np.sin(angle - dphi), 0.0)
    p_o1 = gmsh.model.occ.addPoint(r7 * np.cos(angle + dphi), r7 * np.sin(angle + dphi), 0.0)

    # Create arcs using addCircleArc with shared center point
    arc_inner = gmsh.model.occ.addCircleArc(p_i0, center, p_i1)
    arc_outer = gmsh.model.occ.addCircleArc(p_o0, center, p_o1)

    # Create side lines
    side1 = gmsh.model.occ.addLine(p_i0, p_o0)
    side2 = gmsh.model.occ.addLine(p_i1, p_o1)

    # Create curve loop and surface
    loop = gmsh.model.occ.addCurveLoop([arc_inner, side2, arc_outer, side1])
    pm_segment = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    return pm_segment


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in [0, 2π)."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


# ---------------------------------------------------------------------------
# Main mesh generator
# ---------------------------------------------------------------------------

def generate_PMSM_mesh(
    filename: Path, single: bool, res: np.float64, L: np.float64, depth: np.float64
):
    """
    Generate PMSM 3D mesh with motor centered in an air box.
    Air box covers motor from all directions (x, y, z).
    Returns: (mesh, cell_tags, facet_tags)
    """

    comm = MPI.COMM_WORLD
    rank = comm.rank
    root = 0
    gdim = 3

    domain_map = _domain_map_three

    # Extract geometry parameters (needed by all ranks for retagging)
    r1 = mesh_parameters["r1"]
    r2 = mesh_parameters["r2"]
    r3 = mesh_parameters["r3"]
    r4 = mesh_parameters["r4"]
    r5 = mesh_parameters["r5"]
    r6 = mesh_parameters["r6"]
    r7 = mesh_parameters["r7"]
    air_gap = mesh_parameters["air_gap"]
    r_gap = r3 + air_gap  # outer radius of true air-gap annulus
    r_mid_gap = r3 + 0.5 * air_gap  # Mid-radius for splitting the true air-gap

    # Slot and PM angles (needed for retagging)
    spacing = (np.pi / 4.0) + (np.pi / 4.0) / 3.0  # = π/3 => 60 deg
    slot_angles = np.asarray([i * spacing for i in range(6)], dtype=np.float64)
    pm_count = 10
    pm_spacing = 2.0 * np.pi / pm_count
    pm_angles = np.asarray([i * pm_spacing for i in range(pm_count)], dtype=np.float64)

    # Motor z-range (needed for retagging)
    xy_buffer_factor = 6.0  # ~6x stator radius
    z_buffer_factor = 3.0   # ~3x motor depth above and below
    air_box_size_xy = 2.0 * xy_buffer_factor * r5
    air_box_size_z = depth * (1.0 + 2.0 * z_buffer_factor)
    air_box_z_min = 0.0
    box_center_z = air_box_z_min + air_box_size_z / 2.0
    motor_center_z = box_center_z
    shift_z = motor_center_z - depth / 2.0
    motor_z_start = shift_z
    motor_z_end = shift_z + depth

    # Tolerance for classification (needed for retagging)
    tol = 1e-3 * r5

    # Geometry is built only on rank 0 using Gmsh
    if rank == root:
        gmsh.initialize()
        gmsh.model.add("PMSM_3D_IPM")

        # ------------------------------------------------------------
        # Step 1: Air box dimensions
        # ------------------------------------------------------------
        air_box_x_min = -air_box_size_xy / 2.0
        air_box_y_min = -air_box_size_xy / 2.0

        air_box_x_max = air_box_x_min + air_box_size_xy
        air_box_y_max = air_box_y_min + air_box_size_xy
        air_box_z_max = air_box_z_min + air_box_size_z

        print("\n=== MESH CONFIGURATION ===")
        print(f"Air box: x=[{air_box_x_min:.4f}, {air_box_x_max:.4f}], "
              f"y=[{air_box_y_min:.4f}, {air_box_y_max:.4f}], "
              f"z=[{air_box_z_min:.4f}, {air_box_z_max:.4f}]")
        print(f"Motor z-range after translation: [{motor_z_start:.4f}, {motor_z_end:.4f}]")
        print(f"Motor center z: {motor_center_z:.4f}")

        # ------------------------------------------------------------
        # Step 2: 2D motor geometry at z=0
        # ------------------------------------------------------------

        # Center point (reused for all arcs)
        center_point = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)

        # Center line for mesh refinement (later translated)
        cf = center_point  # Reuse center point
        cb = gmsh.model.occ.addPoint(0.0, 0.0, depth)
        cline = gmsh.model.occ.addLine(cf, cb)

        # Base circles
        c_r1 = gmsh.model.occ.addCircle(0, 0, 0, r1)
        c_r2 = gmsh.model.occ.addCircle(0, 0, 0, r2)
        c_r3 = gmsh.model.occ.addCircle(0, 0, 0, r3)
        c_rgap = gmsh.model.occ.addCircle(0, 0, 0, r_gap)
        c_r4 = gmsh.model.occ.addCircle(0, 0, 0, r4)
        c_r5 = gmsh.model.occ.addCircle(0, 0, 0, r5)

        gmsh.model.occ.synchronize()

        # Curve loops and base surfaces:
        # Shaft (Al)
        loop_r1 = gmsh.model.occ.addCurveLoop([c_r1])
        shaft_surf = gmsh.model.occ.addPlaneSurface([loop_r1])  # r in [0, r1]

        # Rotor annulus (Rotor) between r1 and r3
        loop_r3 = gmsh.model.occ.addCurveLoop([c_r3])
        rotor_surf = gmsh.model.occ.addPlaneSurface([loop_r3, loop_r1])  # [r1, r3]

        # Stator (Stator) between r4 and r5
        loop_r4 = gmsh.model.occ.addCurveLoop([c_r4])
        loop_r5 = gmsh.model.occ.addCurveLoop([c_r5])
        stator_surf = gmsh.model.occ.addPlaneSurface([loop_r5, loop_r4])  # [r4, r5]

        # True air-gap annulus [r3, r_gap] (physical air-gap)
        loop_rgap = gmsh.model.occ.addCurveLoop([c_rgap])
        airgap_surf = gmsh.model.occ.addPlaneSurface([loop_rgap, loop_r3])

        # Slot/air region up to stator inner radius [r_gap, r4]
        slot_air_surf = gmsh.model.occ.addPlaneSurface([loop_r4, loop_rgap])

        gmsh.model.occ.synchronize()

        # ------------------------------------------------------------
        # Copper slots: 6 slots, three-phase winding
        # ------------------------------------------------------------
        copper_surfaces = []
        for ang in slot_angles:
            copper_surfaces.append(_add_copper_segment(ang, center_point))

        # ------------------------------------------------------------
        # Permanent magnets: interior PM (IPM) in [r6, r7]
        # ------------------------------------------------------------

        pm_surfaces = []
        for ang in pm_angles:
            pm_surfaces.append(_add_permanent_magnets(ang, center_point))

        gmsh.model.occ.synchronize()

        # Collect all 2D surfaces to extrude
        domains_2d = []
        # Shaft (Al)
        domains_2d.append((2, shaft_surf))
        # Rotor iron
        domains_2d.append((2, rotor_surf))
        # True airgap ring (thin)
        domains_2d.append((2, airgap_surf))
        # Slot/air ring up to stator (this will be cut by Cu segments)
        domains_2d.append((2, slot_air_surf))
        # Stator ring
        domains_2d.append((2, stator_surf))
        # Copper
        for s in copper_surfaces:
            domains_2d.append((2, s))
        # PM
        for s in pm_surfaces:
            domains_2d.append((2, s))

        gmsh.model.occ.synchronize()

        # ------------------------------------------------------------
        # Step 3: Extrude 2D motor surfaces to 3D
        # ------------------------------------------------------------
        if not domains_2d:
            raise RuntimeError("No valid 2D domains to extrude.")

        extruded = gmsh.model.occ.extrude(domains_2d, 0.0, 0.0, depth)
        gmsh.model.occ.synchronize()

        # Extract 3D motor volumes
        motor_volumes = [e for e in extruded if e[0] == 3]
        if not motor_volumes:
            raise RuntimeError("No 3D volumes created during extrusion.")

        # Translate motor volumes and refinement line to center z
        gmsh.model.occ.translate(motor_volumes, 0.0, 0.0, shift_z)
        gmsh.model.occ.translate([(1, cline)], 0.0, 0.0, shift_z)
        gmsh.model.occ.synchronize()

        # ------------------------------------------------------------
        # Step 4: Create outer air box and fragment
        # ------------------------------------------------------------
        air_box = gmsh.model.occ.addBox(
            air_box_x_min,
            air_box_y_min,
            air_box_z_min,
            air_box_size_xy,
            air_box_size_xy,
            air_box_size_z,
        )

        volumes, _ = gmsh.model.occ.fragment([(3, air_box)], motor_volumes)
        gmsh.model.occ.synchronize()
        
        # Debug: print total volumes
        num_volumes = sum(1 for dim, tag in volumes if dim == 3)
        print(f"\nTotal 3D volumes after fragment: {num_volumes}")

        # ------------------------------------------------------------
        # Step 5: Tag volumes by center-of-mass (robust classification)
        # ------------------------------------------------------------

        def classify_volume(tag3: int):
            x, y, z = gmsh.model.occ.getCenterOfMass(3, tag3)
            r = np.hypot(x, y)
            theta = np.arctan2(y, x)
            if theta < 0:
                theta += 2.0 * np.pi

            # Outer air box - check this FIRST before other conditions
            if z < motor_z_start - tol or z > motor_z_end + tol:
                return "Air", domain_map["Air"][0]
            if r > r5 + tol:
                return "Air", domain_map["Air"][0]

            # Check Cu slots first (overrides other bands)
            slot_half = np.pi / 8.0
            # Cu exists only in [r_gap, r4] now
            if r_gap - tol <= r <= r4 + tol:
                for i, ang in enumerate(slot_angles):
                    if _angle_diff(theta, ang) <= slot_half:
                        # assign phase-wise Cu marker
                        cu_marker = domain_map["Cu"][i % len(domain_map["Cu"])]
                        return "Cu", cu_marker

            # Check PM pockets (must be before rotor check)
            pm_half = np.pi / 12.0
            if r6 - tol <= r <= r7 + tol:
                for i, ang in enumerate(pm_angles):
                    if _angle_diff(theta, ang) <= pm_half:
                        pm_marker = domain_map["PM"][i % len(domain_map["PM"])]
                        return "PM", pm_marker

            # Radial bands for other regions
            if r <= r1 + tol:
                return "Al", domain_map["Al"][0]
            # Rotor: from r1 to r3, excluding PM region (r6-r7 already handled above)
            if r1 < r <= r3 + tol:
                return "Rotor", domain_map["Rotor"][0]
            # True air gap: from r3 to r_gap
            if r3 - tol <= r <= r_gap + tol:
                # air gap annulus - split into two regions
                if r <= r_mid_gap:
                    return "AirGap", domain_map["AirGap"][0]
                else:
                    return "AirGap", domain_map["AirGap"][1]
            # Remaining slot/air region between air-gap and stator inner radius
            if r_gap - tol < r <= r4 + tol:
                return "Air", domain_map["Air"][0]
            # Stator: from r4 to r5
            if r4 - tol <= r <= r5 + tol:
                return "Stator", domain_map["Stator"][0]

            # Anything else inside box but not covered => treat as Air
            return "Air", domain_map["Air"][0]

        # Add physical groups - collect volumes by marker first
        volumes_by_marker = {}
        classification_counts = {}
        sample_centers = {}  # For debugging
        all_centers = []  # For debugging - store all volume centers
        for dim, tag in volumes:
            if dim != 3:
                continue
            x, y, z = gmsh.model.occ.getCenterOfMass(3, tag)
            r = np.hypot(x, y)
            all_centers.append((r, z, tag))
            name, marker = classify_volume(tag)
            if marker not in volumes_by_marker:
                volumes_by_marker[marker] = []
            volumes_by_marker[marker].append(tag)
            # Count classifications for debugging
            if name not in classification_counts:
                classification_counts[name] = {"count": 0, "marker": marker}
            classification_counts[name]["count"] += 1
            # Store sample center for each material type
            if name not in sample_centers:
                sample_centers[name] = (r, x, y, z)
        
        # Debug: print volume distribution by radial position
        print("\n=== VOLUME DISTRIBUTION BY RADIUS ===")
        all_centers.sort(key=lambda x: x[0])  # Sort by radius
        for r, z, tag in all_centers:
            if r1 - 0.001 <= r <= r1 + 0.001:
                region = "r1 (shaft)"
            elif r1 < r <= r6:
                region = "r1-r6 (rotor inner)"
            elif r6 <= r <= r7:
                region = "r6-r7 (PM region)"
            elif r7 < r <= r3:
                region = "r7-r3 (rotor outer)"
            elif r3 <= r <= r4:
                region = "r3-r4 (airgap)"
            elif r4 <= r <= r5:
                region = "r4-r5 (stator)"
            elif r > r5:
                region = ">r5 (air)"
            else:
                region = "other"
            print(f"  Vol {tag:3d}: r={r:.4f}, z={z:.4f}, region={region}")
        
        # Print classification summary
        print("\n=== VOLUME CLASSIFICATION SUMMARY ===")
        for name in sorted(classification_counts.keys()):
            count = classification_counts[name]["count"]
            marker = classification_counts[name]["marker"]
            r, x, y, z = sample_centers[name]
            print(f"  {name:10s} (tag {marker:2d}): {count:4d} volumes, sample r={r:.4f}, z={z:.4f}")
        
        # Check for missing expected tags
        expected_tags = set()
        for tags in domain_map.values():
            expected_tags.update(tags)
        found_tags = set(volumes_by_marker.keys())
        missing_tags = expected_tags - found_tags
        if missing_tags:
            print(f"\n⚠️  WARNING: Missing expected tags: {sorted(missing_tags)}")
            print(f"   Found tags: {sorted(found_tags)}")
            print(f"   Expected tags: {sorted(expected_tags)}")
            print(f"\n   Radial boundaries: r1={r1:.4f}, r2={r2:.4f}, r3={r3:.4f}, r4={r4:.4f}, r5={r5:.4f}")
            print(f"   PM region: r6={r6:.4f}, r7={r7:.4f}")
            print(f"   Motor z-range: [{motor_z_start:.4f}, {motor_z_end:.4f}]")
        
        # Add physical groups with all volumes sharing the same marker
        for marker, volume_tags in volumes_by_marker.items():
            gmsh.model.addPhysicalGroup(3, volume_tags, marker)

        # ------------------------------------------------------------
        # Step 6: Tag boundaries (MidAir & Exterior)
        # ------------------------------------------------------------
        # Get all faces bounding all volumes
        surf_entities = gmsh.model.getBoundary(
            volumes, combined=False, oriented=False
        )
        surf_tags = sorted(set(s[1] for s in surf_entities))

        # MidAir: cylindrical surface around mid of air gap
        r_mid = r_mid_gap
        tol_mid = 0.02 * r_mid

        midair_faces = []
        exterior_faces = []

        for s_tag in surf_tags:
            # Center of mass of surface
            x, y, z = gmsh.model.occ.getCenterOfMass(2, s_tag)
            r_surf = np.hypot(x, y)

            # Bounding box for detecting outer air-box faces
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s_tag)

            # MidAir: cylindrical band approximately at r_mid and in motor z-range
            if abs(r_surf - r_mid) < tol_mid and (motor_z_start - tol) <= z <= (motor_z_end + tol):
                midair_faces.append(s_tag)

            # Exterior: faces lying on the outer box boundary
            eps_box = 1e-6 * max(
                abs(air_box_x_min),
                abs(air_box_x_max),
                abs(air_box_y_min),
                abs(air_box_y_max),
                abs(air_box_z_min),
                abs(air_box_z_max),
                1.0,
            )
            on_x_min = abs(xmin - air_box_x_min) < eps_box
            on_x_max = abs(xmax - air_box_x_max) < eps_box
            on_y_min = abs(ymin - air_box_y_min) < eps_box
            on_y_max = abs(ymax - air_box_y_max) < eps_box
            on_z_min = abs(zmin - air_box_z_min) < eps_box
            on_z_max = abs(zmax - air_box_z_max) < eps_box

            if on_x_min or on_x_max or on_y_min or on_y_max or on_z_min or on_z_max:
                exterior_faces.append(s_tag)

        if midair_faces:
            gmsh.model.addPhysicalGroup(2, midair_faces, surface_map["MidAir"])
        if exterior_faces:
            gmsh.model.addPhysicalGroup(2, exterior_faces, surface_map["Exterior"])

        # ------------------------------------------------------------
        # Step 7: Mesh generation
        # ------------------------------------------------------------
        res_base = float(res)

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", [cline])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res_base)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 10.0 * res_base)
        gmsh.model.mesh.field.setNumber(2, "DistMin", r5)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 6.0 * r5)

        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)

        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.optimize("Netgen")

        msh_file = str(filename.with_suffix(".msh"))
        gmsh.write(msh_file)
        gmsh.finalize()

    # Synchronize all ranks before reading mesh
    comm.barrier()

    # ------------------------------------------------------------
    # Step 8: Read MSH into DOLFINx and retag cells if needed
    # ------------------------------------------------------------
    # In newer DOLFINx (e.g. 0.10+), gmsh I/O lives in dolfinx.io.gmsh.
    # Avoid shadowing the top-level gmsh module used for geometry creation.
    from dolfinx.io import gmsh as gmshio, XDMFFile
    import dolfinx.mesh as dmesh

    result = gmshio.read_from_msh(
        str(filename.with_suffix(".msh")), comm, 0, gdim=3
    )
    mesh = result[0]
    ct = result[1] if len(result) > 1 else None
    ft = result[2] if len(result) > 2 else None

    # Always retag cells based on center coordinates for accurate classification
    # This ensures the 2mm air-gap is correctly identified regardless of Gmsh classification
    if rank == root:
        print("\n🔄 Retagging all cells by center coordinates for accurate domain classification...")
    
    # Get cell centers (same approach as load_mesh.py)
    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    # dofmap is a 2D array: shape (num_cells, num_vertices_per_cell)
    centers = coords[dofmap].mean(axis=1)
    radii = np.linalg.norm(centers[:, :2], axis=1)
    angles = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)
    z_coords = centers[:, 2]
    
    # Classification function for cells
    def classify_cell(r, theta, z):
        # Outer air box
        if z < motor_z_start - tol or z > motor_z_end + tol:
            return domain_map["Air"][0]
        if r > r5 + tol:
            return domain_map["Air"][0]
        
        # Check Cu slots first (overrides other bands in [r_gap, r4])
        slot_half = np.pi / 8.0
        if r_gap - tol <= r <= r4 + tol:
            for i, ang in enumerate(slot_angles):
                if _angle_diff(theta, ang) <= slot_half:
                    return domain_map["Cu"][i % len(domain_map["Cu"])]
        
        # Check PM pockets (must be before rotor check)
        pm_half = np.pi / 12.0
        if r6 - tol <= r <= r7 + tol:
            for i, ang in enumerate(pm_angles):
                if _angle_diff(theta, ang) <= pm_half:
                    return domain_map["PM"][i % len(domain_map["PM"])]
        
        # Radial bands
        if r <= r1 + tol:
            return domain_map["Al"][0]
        if r1 < r <= r3 + tol:
            return domain_map["Rotor"][0]
        # True air-gap: ONLY in [r3, r_gap] (2mm annulus)
        if r3 - tol <= r <= r_gap + tol:
            if r <= r_mid_gap:
                return domain_map["AirGap"][0]
            else:
                return domain_map["AirGap"][1]
        # Slot/air region between air-gap and stator inner radius
        if r_gap - tol < r <= r4 + tol:
            return domain_map["Air"][0]
        # Stator: from r4 to r5
        if r4 - tol <= r <= r5 + tol:
            return domain_map["Stator"][0]
        
        return domain_map["Air"][0]
    
    # Retag all cells on all ranks
    n_cells = mesh.topology.index_map(3).size_local
    new_tags = np.empty(n_cells, dtype=np.int32)
    for i in range(n_cells):
        new_tags[i] = classify_cell(radii[i], angles[i], z_coords[i])
    
    # Create new cell tags - this REPLACES the old ct (or creates it if None)
    cell_indices = np.arange(n_cells, dtype=np.int32)
    ct = dmesh.meshtags(mesh, mesh.topology.dim, cell_indices, new_tags)
    
    # Validate AirGap tagging (aggregate across all ranks)
    airgap_mask = np.isin(new_tags, domain_map["AirGap"])
    airgap_cell_indices = ct.indices[airgap_mask]
    if len(airgap_cell_indices) > 0:
        # Get cell centers for AirGap cells
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        c2v = mesh.topology.connectivity(mesh.topology.dim, 0)
        airgap_centers = np.array([centers[int(idx)] for idx in airgap_cell_indices if int(idx) < len(centers)])
        airgap_radii = np.linalg.norm(airgap_centers[:, :2], axis=1) if len(airgap_centers) > 0 else np.array([])
    else:
        airgap_radii = np.array([])
    
    # Gather statistics from all ranks
    n_airgap_local = len(airgap_radii)
    n_airgap_total = comm.allreduce(n_airgap_local, op=MPI.SUM)
    
    if n_airgap_total > 0:
        # Find global min/max
        airgap_min_local = float(airgap_radii.min()) if len(airgap_radii) > 0 else np.inf
        airgap_max_local = float(airgap_radii.max()) if len(airgap_radii) > 0 else -np.inf
        airgap_min_global = comm.allreduce(airgap_min_local, op=MPI.MIN)
        airgap_max_global = comm.allreduce(airgap_max_local, op=MPI.MAX)
        
        # Count cells in correct range
        in_range_local = np.sum((airgap_radii >= r3 - tol) & (airgap_radii <= r_gap + tol)) if len(airgap_radii) > 0 else 0
        in_range_total = comm.allreduce(in_range_local, op=MPI.SUM)
        
        if rank == root:
            unique_tags = np.unique(new_tags)
            print(f"✅ Retagged {comm.allreduce(n_cells, op=MPI.SUM)} cells total. Unique tags: {sorted(unique_tags)}")
            print(f"\n📊 AirGap validation (all ranks):")
            print(f"   Total AirGap cells: {n_airgap_total}")
            print(f"   Radius range: [{airgap_min_global:.6f}, {airgap_max_global:.6f}] m")
            print(f"   Expected range: [{r3:.6f}, {r_gap:.6f}] m")
            print(f"   Cells in correct range: {in_range_total}/{n_airgap_total} ({100*in_range_total/n_airgap_total:.1f}%)")
            
            # Measure actual gap from rotor
            rotor_mask = (new_tags == domain_map["Rotor"][0])
            if np.any(rotor_mask):
                rotor_radii = radii[rotor_mask]
                rotor_max_local = float(rotor_radii.max())
                rotor_max_global = comm.allreduce(rotor_max_local, op=MPI.MAX)
                gap_measured = airgap_min_global - rotor_max_global
                print(f"   Rotor max radius: {rotor_max_global:.6f} m")
                print(f"   Measured air-gap: {gap_measured:.6f} m (target: {air_gap:.6f} m)")
    else:
        if rank == root:
            unique_tags = np.unique(new_tags)
            print(f"✅ Retagged {comm.allreduce(n_cells, op=MPI.SUM)} cells total. Unique tags: {sorted(unique_tags)}")
            print(f"\n⚠️  No AirGap cells found after retagging!")

    # Write XDMF with function field for ParaView visualization
    from dolfinx import fem

    with XDMFFile(comm, filename.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        
        if ct is not None:
            # Ensure a stable, readable name in XDMF (and avoid clobbering facet tags)
            try:
                ct.name = "cell_tags"
            except Exception:
                pass
            # Write meshtags (default name will be used)
            xdmf.write_meshtags(ct, mesh.geometry)
            
            # Create a function field for easier ParaView visualization
            DG0 = fem.functionspace(mesh, ("DG", 0))
            cell_tag_function = fem.Function(DG0)
            cell_tag_function.name = "CellTags"
            
            # Map cell tags to function - initialize all to 0 first
            cell_tag_function.x.array[:] = 0.0
            
            # Map tagged cells
            cell_to_tag = {int(i): int(v) for i, v in zip(ct.indices, ct.values)}
            for cell_idx, tag in cell_to_tag.items():
                if cell_idx < cell_tag_function.x.array.size:
                    cell_tag_function.x.array[cell_idx] = float(tag)
            
            # Write function field for ParaView
            xdmf.write_function(cell_tag_function, 0.0)
        
        if ft is not None:
            # Ensure a stable, readable name in XDMF (and avoid clobbering cell tags)
            try:
                ft.name = "facet_tags"
            except Exception:
                pass
            xdmf.write_meshtags(ft, mesh.geometry)

    if rank == root:
        print(f"\n✅ Mesh generated: {filename.with_suffix('.xdmf')}")

    return mesh, ct, ft


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gmsh script to generate 3D IPM PMSM mesh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--res",
        default=0.01,
        type=np.float64,
        dest="res",
        help="Base mesh resolution near motor",
    )
    parser.add_argument(
        "--L",
        default=1.0,
        type=np.float64,
        dest="L",
        help="(Unused) legacy parameter for box size",
    )
    parser.add_argument(
        "--depth",
        default=0.057,
        type=np.float64,
        dest="depth",
        help="Axial depth of the motor (extrusion length)",
    )

    args = parser.parse_args()
    res = args.res
    depth = args.depth

    folder = Path("../../meshes/3d")
    folder.mkdir(parents=True, exist_ok=True)
    fname = folder / "pmesh3D_ipm"

    generate_PMSM_mesh(fname, False, res, args.L, depth)
