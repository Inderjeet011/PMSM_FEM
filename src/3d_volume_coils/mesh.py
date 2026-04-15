"""
3D Gmsh mesh for the **volume-coil** interior permanent-magnet motor.

Constructs 2D motor slices (shaft, rotor, air-gap, stator, slot copper, PMs),
extrudes to 3D with optional coil height extension, embeds the stack in a
bounding air box, fragments and tags volumes/surfaces, then exports ``mesh.msh``
and DOLFINx ``mesh.xdmf`` / ``mesh.h5``. Cell markers are recomputed from cell
center coordinates so material tags stay consistent after boolean operations.

Also exports ``model_parameters``, ``mesh_parameters``, and ``surface_map`` for
the solver and ``load_mesh``.

CLI: ``python mesh.py`` (see ``--help`` for resolution and box grading options).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from mpi4py import MPI
import gmsh

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
        "Air": 1,
        "AirGap": 1,
        "PM": 1.04457,
    },
    "sigma": {
        "Rotor": 1.6e6,
        "Stator": 0,
        "Cu": 0,
        "Air": 0,
        "AirGap": 0,
        "PM": 6.25e5,
    },
}

# Facet markers
surface_map: Dict[str, Union[int, str]] = {"Exterior": 1, "MidAir": 2, "restriction": "+"}

# Volume markers (3D physical groups)
_domain_map_three: Dict[str, tuple[int, ...]] = {
    "Air": (1,),
    "AirGap": (2, 3),
    # Shaft and rotor iron share one tag (rotor assembly)
    "Rotor": (5,),
    "Stator": (6,),
    "Cu": (7, 8, 9, 10, 11, 12),
    "PM": (13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
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
    filename: Path,
    res: np.float64,
    depth: np.float64,
    lc_max_ratio: float = 25.0,
    dist_max_ratio: float = 8.0,
    optimize: str = "Netgen",
):
    """
    Generate PMSM 3D mesh with motor centered in an air box.
    Mesh is finer near the motor (distance < r5) and coarser as distance increases.
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
    coil_extension_height = 0.08 * depth  # smaller symmetric coil overhang
    air_box_size_xy = 2.0 * xy_buffer_factor * r5
    air_box_size_z = depth * (1.0 + 2.0 * z_buffer_factor)
    air_box_z_min = 0.0
    box_center_z = air_box_z_min + air_box_size_z / 2.0
    motor_center_z = box_center_z
    shift_z = motor_center_z - depth / 2.0
    motor_z_start = shift_z
    motor_z_end = shift_z + depth
    coil_z_start = motor_z_start - coil_extension_height
    coil_z_end = motor_z_end + coil_extension_height

    # Tolerance for classification (needed for retagging)
    tol = 1e-3 * r5

    def classify_marker(r: float, theta: float, z: float) -> int:
        """Gmsh volume tagging and DOLFINx retagging use the same rules (theta in [0, 2π))."""
        slot_half = np.pi / 8.0
        if (
            coil_z_start - tol <= z <= coil_z_end + tol
            and r_gap - tol <= r <= r4 + tol
        ):
            for i, ang in enumerate(slot_angles):
                if _angle_diff(theta, ang) <= slot_half:
                    return domain_map["Cu"][i % len(domain_map["Cu"])]
        if z < motor_z_start - tol or z > motor_z_end + tol:
            return domain_map["Air"][0]
        if r > r5 + tol:
            return domain_map["Air"][0]
        pm_half = np.pi / 12.0
        if r6 - tol <= r <= r7 + tol:
            for i, ang in enumerate(pm_angles):
                if _angle_diff(theta, ang) <= pm_half:
                    return domain_map["PM"][i % len(domain_map["PM"])]
        # Inner shaft and rotor annulus share one rotor-assembly tag (marker 5)
        if r <= r1 + tol:
            return domain_map["Rotor"][0]
        if r1 < r <= r3 + tol:
            return domain_map["Rotor"][0]
        if r3 - tol <= r <= r_gap + tol:
            return (
                domain_map["AirGap"][0]
                if r <= r_mid_gap
                else domain_map["AirGap"][1]
            )
        if r_gap - tol < r <= r4 + tol:
            return domain_map["Air"][0]
        if r4 - tol <= r <= r5 + tol:
            return domain_map["Stator"][0]
        return domain_map["Air"][0]

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
        print(f"Coil z-range after translation:  [{coil_z_start:.4f}, {coil_z_end:.4f}]")
        print(f"Motor center z: {motor_center_z:.4f}")

        # ------------------------------------------------------------
        # Step 2: 2D motor geometry at z=0
        # ------------------------------------------------------------

        # Center point (reused for all arcs)
        center_point = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)

        # Center line for mesh refinement (later translated)
        cf = center_point  # Reuse center point
        cb = gmsh.model.occ.addPoint(0.0, 0.0, depth + 2.0 * coil_extension_height)
        cline = gmsh.model.occ.addLine(cf, cb)

        # Base circles
        c_r1 = gmsh.model.occ.addCircle(0, 0, 0, r1)
        c_r3 = gmsh.model.occ.addCircle(0, 0, 0, r3)
        c_rgap = gmsh.model.occ.addCircle(0, 0, 0, r_gap)
        c_r4 = gmsh.model.occ.addCircle(0, 0, 0, r4)
        c_r5 = gmsh.model.occ.addCircle(0, 0, 0, r5)

        gmsh.model.occ.synchronize()

        # Curve loops and base surfaces:
        # Inner shaft (same physical tag as rotor assembly after classification)
        loop_r1 = gmsh.model.occ.addCurveLoop([c_r1])
        shaft_surf = gmsh.model.occ.addPlaneSurface([loop_r1])  # r in [0, r1]

        # Rotor annulus between r1 and r3
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

        # Make 2D regions disjoint before extrusion so copper can be
        # extruded separately without overlapping the slot-air region.
        rotor_cut, _ = gmsh.model.occ.cut(
            [(2, rotor_surf)],
            [(2, s) for s in pm_surfaces],
            removeObject=True,
            removeTool=False,
        )
        slot_air_cut, _ = gmsh.model.occ.cut(
            [(2, slot_air_surf)],
            [(2, s) for s in copper_surfaces],
            removeObject=True,
            removeTool=False,
        )

        gmsh.model.occ.synchronize()

        rotor_surfaces = [e for e in rotor_cut if e[0] == 2]
        slot_air_surfaces = [e for e in slot_air_cut if e[0] == 2]
        if not rotor_surfaces:
            rotor_surfaces = [(2, rotor_surf)]
        if not slot_air_surfaces:
            slot_air_surfaces = [(2, slot_air_surf)]

        domains_2d_non_cu = []
        domains_2d_cu = []
        domains_2d_non_cu.append((2, shaft_surf))
        domains_2d_non_cu.extend(rotor_surfaces)
        # True airgap ring (thin)
        domains_2d_non_cu.append((2, airgap_surf))
        # Slot/air ring up to stator with copper removed
        domains_2d_non_cu.extend(slot_air_surfaces)
        # Stator ring
        domains_2d_non_cu.append((2, stator_surf))
        # Copper
        for s in copper_surfaces:
            domains_2d_cu.append((2, s))
        # PM
        for s in pm_surfaces:
            domains_2d_non_cu.append((2, s))

        gmsh.model.occ.synchronize()

        # ------------------------------------------------------------
        # Step 3: Extrude 2D motor surfaces to 3D
        # ------------------------------------------------------------
        if not domains_2d_non_cu and not domains_2d_cu:
            raise RuntimeError("No valid 2D domains to extrude.")

        extruded_non_cu = []
        extruded_cu = []
        if domains_2d_non_cu:
            extruded_non_cu = gmsh.model.occ.extrude(domains_2d_non_cu, 0.0, 0.0, depth)
        if domains_2d_cu:
            extruded_cu = gmsh.model.occ.extrude(
                domains_2d_cu, 0.0, 0.0, depth + 2.0 * coil_extension_height
            )
        gmsh.model.occ.synchronize()

        extruded = list(extruded_non_cu) + list(extruded_cu)

        # Extract 3D motor volumes
        motor_volumes = [e for e in extruded if e[0] == 3]
        if not motor_volumes:
            raise RuntimeError("No 3D volumes created during extrusion.")

        # Translate motor volumes and refinement line to center z
        non_cu_volumes = [e for e in extruded_non_cu if e[0] == 3]
        cu_volumes = [e for e in extruded_cu if e[0] == 3]
        if non_cu_volumes:
            gmsh.model.occ.translate(non_cu_volumes, 0.0, 0.0, shift_z)
        if cu_volumes:
            gmsh.model.occ.translate(cu_volumes, 0.0, 0.0, shift_z - coil_extension_height)
        gmsh.model.occ.translate([(1, cline)], 0.0, 0.0, shift_z - coil_extension_height)
        gmsh.model.occ.synchronize()

        motor_volumes = list(non_cu_volumes) + list(cu_volumes)

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

        # ------------------------------------------------------------
        # Step 5: Tag volumes by center-of-mass (same logic as DOLFINx retag below)
        # ------------------------------------------------------------
        volumes_by_marker: Dict[int, List[int]] = {}
        for dim, tag in volumes:
            if dim != 3:
                continue
            x, y, z = gmsh.model.occ.getCenterOfMass(3, tag)
            r = np.hypot(x, y)
            theta = float(np.arctan2(y, x))
            if theta < 0:
                theta += 2.0 * np.pi
            marker = classify_marker(r, theta, z)
            volumes_by_marker.setdefault(marker, []).append(tag)

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
        # Step 7: Mesh generation (finer near motor, coarser away)
        # ------------------------------------------------------------
        res_base = float(res)
        lc_max = float(lc_max_ratio) * res_base
        dist_max = float(dist_max_ratio) * r5
        print(
            f"\nMesh grading: LcMin={res_base:.4e} (near motor r<{r5:.4f}), "
            f"LcMax={lc_max:.4e} (far r>{dist_max:.4f})"
        )

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", [cline])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res_base)
        gmsh.model.mesh.field.setNumber(2, "LcMax", lc_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", r5)
        gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max)

        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)

        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.mesh.generate(gdim)
        if optimize and optimize.lower() not in ("none", "off", ""):
            print(f"\nMesh optimization ({optimize})...")
            gmsh.model.mesh.optimize(optimize)

        msh_file = str(filename.with_suffix(".msh"))
        gmsh.write(msh_file)
        gmsh.finalize()

    # Synchronize all ranks before reading mesh
    comm.barrier()

    # ------------------------------------------------------------
    # Step 8: Read MSH into DOLFINx and retag cells (same rules as Gmsh volume tagging)
    # ------------------------------------------------------------
    # In newer DOLFINx (e.g. 0.10+), gmsh I/O lives in dolfinx.io.gmsh.
    from dolfinx.io import gmsh as gmshio, XDMFFile
    import dolfinx.mesh as dmesh

    result = gmshio.read_from_msh(
        str(filename.with_suffix(".msh")), comm, 0, gdim=3
    )
    mesh = result[0]
    ft = result[2] if len(result) > 2 else None

    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    centers = coords[dofmap].mean(axis=1)
    radii = np.linalg.norm(centers[:, :2], axis=1)
    angles = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)
    z_coords = centers[:, 2]

    n_cells = mesh.topology.index_map(3).size_local
    new_tags = np.empty(n_cells, dtype=np.int32)
    for i in range(n_cells):
        new_tags[i] = classify_marker(radii[i], angles[i], z_coords[i])

    cell_indices = np.arange(n_cells, dtype=np.int32)
    ct = dmesh.meshtags(mesh, mesh.topology.dim, cell_indices, new_tags)

    if rank == root:
        print(
            f"Retagged {mesh.topology.index_map(3).size_global} cells "
            f"(markers from cell-center coordinates)."
        )

    with XDMFFile(comm, filename.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        if ct is not None:
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

    if rank == root:
        print(f"Mesh generated: {filename.with_suffix('.xdmf')}")

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
        default=0.005,
        type=np.float64,
        dest="res",
        help="Base mesh resolution near motor (2mm=0.002). Finer near motor, coarser with distance.",
    )
    parser.add_argument(
        "--depth",
        default=0.057,
        type=np.float64,
        dest="depth",
        help="Axial depth of the motor (extrusion length)",
    )
    parser.add_argument(
        "--lc-max-ratio",
        default=40.0,
        type=np.float64,
        dest="lc_max_ratio",
        help="LcMax/LcMin ratio: element size increases with distance from motor (e.g. 40 = 40x coarser far field)",
    )
    parser.add_argument(
        "--dist-max-ratio",
        default=8.0,
        type=np.float64,
        dest="dist_max_ratio",
        help="DistMax/r5 ratio: distance beyond which mesh is coarsest (e.g. 8 = coarse beyond 8*r5)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        dest="no_optimize",
        help="Skip Netgen mesh optimization (much faster, but lower quality elements)",
    )

    args = parser.parse_args()
    res = args.res
    depth = args.depth

    folder = Path(__file__).resolve().parent
    folder.mkdir(parents=True, exist_ok=True)
    fname = folder / "mesh"

    opt = "" if args.no_optimize else "Netgen"
    generate_PMSM_mesh(
        fname,
        res,
        depth,
        lc_max_ratio=args.lc_max_ratio,
        dist_max_ratio=args.dist_max_ratio,
        optimize=opt,
    )