# Generate the three phase PMSM model, with a given minimal resolution, encapsilated in a LxL box.

import argparse
from pathlib import Path
from typing import Dict, Union
from datetime import datetime
import dolfinx
import gmsh
import numpy as np
from mpi4py import MPI

__all__ = ["model_parameters", "mesh_parameters", "surface_map"]

# Model parameters for the PMSM 2D Model
model_parameters = {
    "mu_0": 1.25663753e-6,  # Relative permability of air [H/m]=[kg m/(s^2 A^2)]
    "freq": 60,  # Frequency of excitation,
    "J": 3.1e6 * np.sqrt(2),  # [A/m^2] Current density of copper winding
    "mu_r": {"Cu": 1, "Stator": 30, "Rotor": 30, "Al": 1, "Air": 1, "AirGap": 1, "PM": 1.04457},  # Relative permability
    "sigma": {"Rotor": 1.6e6, "Al": 3.72e7, "Stator": 0, "Cu": 0, "Air": 0, "AirGap": 0, "PM": 6.25e5},  # Conductivity 6.
    "densities": {"Rotor": 7850, "Al": 2700, "Stator": 0, "Air": 0, "Cu": 0, "AirGap": 0, "PM": 7500}  # [kg/m^3]
}
# Marker for facets, and restriction to use in surface integral of airgap
surface_map: Dict[str, Union[int, str]] = {"Exterior": 1, "MidAir": 2, "restriction": "+"}

# Copper wires is ordered in counter clock-wise order from angle = 0, 2*np.pi/num_segments...
_domain_map_three: Dict[str, tuple[int, ...]] = {"Air": (1,), "AirGap": (2, 3), "Al": (4,), "Rotor": (5, ), 
                                                 "Stator": (6, ), "Cu": (7, 8, 9, 10, 11, 12),
                                                 "PM": (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)}

# Currents mapping to the domain marker sof the copper
_currents_three: Dict[int, Dict[str, float]] = {7: {"alpha": 1, "beta": 0}, 8: {"alpha": -1, "beta": 2 * np.pi / 3},
                                                9: {"alpha": 1, "beta": 4 * np.pi / 3}, 10: {"alpha": -1, "beta": 0},
                                                11: {"alpha": 1, "beta": 2 * np.pi / 3},
                                                12: {"alpha": -1, "beta": 4 * np.pi / 3}}


# The different radiuses used in domain specifications
mesh_parameters: Dict[str, float] = {"r1": 0.017, "r2": 0.04, "r3": 0.042, "r4": 0.062, "r5": 0.075, "r6": 0.036, "r7": 0.038}
# mesh_parameters: Dict[str, float] = {"r1": 0.02, "r2": 0.03, "r3": 0.032, "r4": 0.052, "r5": 0.057, "r6": 0.026}



def _add_copper_segment(start_angle=0):
    """
    Helper function
    Add a 45 degree copper segement, r in (r3, r4) with midline at "start_angle".
    """
    copper_arch_inner = gmsh.model.occ.addCircle(
        0, 0, 0, mesh_parameters["r3"], angle1=start_angle - np.pi / 8, angle2=start_angle + np.pi / 8)
    copper_arch_outer = gmsh.model.occ.addCircle(
        0, 0, 0, mesh_parameters["r4"], angle1=start_angle - np.pi / 8, angle2=start_angle + np.pi / 8)
    gmsh.model.occ.synchronize()
    nodes_inner = gmsh.model.getBoundary([(1, copper_arch_inner)])
    nodes_outer = gmsh.model.getBoundary([(1, copper_arch_outer)])
    l0 = gmsh.model.occ.addLine(nodes_inner[0][1], nodes_outer[0][1])
    l1 = gmsh.model.occ.addLine(nodes_inner[1][1], nodes_outer[1][1])
    c_l = gmsh.model.occ.addCurveLoop([copper_arch_inner, l1, copper_arch_outer, l0])

    copper_segment = gmsh.model.occ.addPlaneSurface([c_l])
    gmsh.model.occ.synchronize()
    return copper_segment

def _add_permanent_magnets(start_angle=0):
    """
    Helper function
    Add a 45 degree copper segement, r in (r3, r4) with midline at "start_angle".
    """
    copper_arch_inner = gmsh.model.occ.addCircle(
        0, 0, 0, mesh_parameters["r6"], angle1=start_angle - np.pi / 12, angle2=start_angle + np.pi / 12) # 30 deg + 6 deg arc length
    copper_arch_outer = gmsh.model.occ.addCircle(
        0, 0, 0, mesh_parameters["r7"], angle1=start_angle - np.pi / 12, angle2=start_angle + np.pi / 12)
    gmsh.model.occ.synchronize()
    nodes_inner = gmsh.model.getBoundary([(1, copper_arch_inner)])
    nodes_outer = gmsh.model.getBoundary([(1, copper_arch_outer)])
    l0 = gmsh.model.occ.addLine(nodes_inner[0][1], nodes_outer[0][1])
    l1 = gmsh.model.occ.addLine(nodes_inner[1][1], nodes_outer[1][1])
    c_l = gmsh.model.occ.addCurveLoop([copper_arch_inner, l1, copper_arch_outer, l0])

    copper_segment = gmsh.model.occ.addPlaneSurface([c_l])
    gmsh.model.occ.synchronize()
    return copper_segment

def generate_PMSM_mesh(filename: Path, single: bool, res: np.float64, L: np.float64, depth: np.float64):
    """
    Generate the three phase PMSM model, with a given minimal resolution, encapsilated in
    a LxL box.
    All domains are marked, while only the exterior facets and the mid air gap facets are marked
      """
    spacing = (np.pi / 4) + (np.pi / 4) / 3
    angles = np.asarray([i * spacing for i in range(6)], dtype=np.float64)
    domain_map = _domain_map_three
    assert len(domain_map["Cu"]) == len(angles)  

    gmsh.initialize()
    # Generate three phase induction motor
    rank = MPI.COMM_WORLD.rank
    root = 0
    gdim = 3  # Geometric dimension of the mesh
    if rank == root:

        # Center line for mesh resolution
        cf = gmsh.model.occ.addPoint(0, 0, 0)
        cb = gmsh.model.occ.addPoint(0, 0, depth)
        cline = gmsh.model.occ.addLine(cf, cb)

        # Calculate air box size based on motor radius (not fixed L parameter)
        r5 = mesh_parameters["r5"]
        outer_air_factor = 4.0  # 3.0–5.0 is reasonable; start with 4.0
        L_actual = 2.0 * outer_air_factor * r5
        
        # Define the different circular layers
        strator_steel = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r5"])
        air_2 = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r4"])        # stator bdry
        air = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r3"])          # air bdry
        air_mid = gmsh.model.occ.addCircle(0, 0, 0, 0.5 * (mesh_parameters["r2"] + mesh_parameters["r3"]))  # mid_air
        aluminium = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r2"])    # al boundary 40 mm
        rotor_steel = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r1"])  # 17 mm
        pmsm1 = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r6"])        # pm bdry 37 mm
        pmsm2 = gmsh.model.occ.addCircle(0, 0, 0, mesh_parameters["r7"])        # pm bdry 39 mm

        # Create out strator steel
        steel_loop = gmsh.model.occ.addCurveLoop([strator_steel])
        air_2_loop = gmsh.model.occ.addCurveLoop([air_2])
        strator_steel = gmsh.model.occ.addPlaneSurface([steel_loop, air_2_loop])

        # Create air layer
        air_loop = gmsh.model.occ.addCurveLoop([air])
        air = gmsh.model.occ.addPlaneSurface([air_2_loop, air_loop])

        domains = [(2, _add_copper_segment(angle)) for angle in angles]

        # air_3_loop = gmsh.model.occ.addCurveLoop([aluminium])        

        # Add second air segment (in two pieces)
        air_mid_loop = gmsh.model.occ.addCurveLoop([air_mid])
        al_loop = gmsh.model.occ.addCurveLoop([aluminium])  # outer
        al_2_loop = gmsh.model.occ.addCurveLoop([pmsm1])
        al_3_loop = gmsh.model.occ.addCurveLoop([pmsm2])

        air_surf1 = gmsh.model.occ.addPlaneSurface([air_loop, air_mid_loop])
        air_surf2 = gmsh.model.occ.addPlaneSurface([air_mid_loop, al_loop])

        # Add aluminium segement
        rotor_loop = gmsh.model.occ.addCurveLoop([rotor_steel])
        aluminium_surf1 = gmsh.model.occ.addPlaneSurface([al_2_loop, rotor_loop])    # rotor - Al gap   20 mm
        aluminium_surf2 = gmsh.model.occ.addPlaneSurface([al_3_loop, al_2_loop])    # PM gap    2 mm
        aluminium_surf3 = gmsh.model.occ.addPlaneSurface([al_loop, al_3_loop])      # Pm-bdry gap 1 mm

        # Creating PMs
        pm_spacing = (np.pi / 6) + (np.pi / 30)
        pm_angles = np.asarray([i * pm_spacing for i in range(10)], dtype=np.float64)
        magnets = [(2, _add_permanent_magnets(angle)) for angle in pm_angles]
        domains.extend(magnets)

        # Add steel rotor
        rotor_disk = gmsh.model.occ.addPlaneSurface([rotor_loop])
        gmsh.model.occ.synchronize()
        domains.extend([(2, strator_steel), (2, rotor_disk), (2, air),
                        (2, air_surf1), (2, air_surf2), (2, aluminium_surf1), (2, aluminium_surf2), (2, aluminium_surf3)])

        gmsh.model.occ.synchronize()
        
        domains = gmsh.model.occ.extrude(domains, 0, 0, depth)
        domains_3D = []
        for domain in domains:
            if domain[0] == 3:
                domains_3D.append(domain)
        # air_box = gmsh.model.occ.addBox(
        #     -L / 2, -L / 2, -5 * depth, L, L, 10 * depth
        # )
        # Calculate air box size based on motor radius (not fixed L)
        r5 = mesh_parameters["r5"]
        outer_air_factor = 4.0  # 3.0–5.0 is reasonable; start with 4.0
        L_actual = 2.0 * outer_air_factor * r5
        
        # Air box enclosing whole machine
        air_box = gmsh.model.occ.addBox(
            -L_actual / 2, -L_actual / 2, 0.0,
            L_actual, L_actual, depth
        )
        # Cutting box (removes one half)
        # cutting_box = gmsh.model.occ.addBox(0, -L/2, -5 * depth, L/2, L, 10 * depth)

        # # Perform Boolean cut to remove one half
        # half_box = gmsh.model.occ.cut([(3, air_box)], [(3, cutting_box)])
        volumes, _ = gmsh.model.occ.fragment([(3, air_box)], domains_3D)

        gmsh.model.occ.synchronize()

        # Helpers for assigning domain markers based on area of domain
        rs = [mesh_parameters[f"r{i}"] for i in range(1, 8)]
        r_mid = 0.5 * (rs[1] + rs[2])  # Radius for middle of air gap
        area_helper = (rs[3]**2 - rs[2]**2) * np.pi  # Helper function to determine area of copper and air
        area_helper1 = (rs[6]**2 - rs[5]**2) * np.pi  # Helper function to determine area of PM and Al
        frac_cu = 45 / 360
        frac_air = (360 - len(angles) * 45) / (360 * len(angles))
        frac_pm = 30 / 360
        frac_al = 60  / 3600
        _area_to_domain_map: Dict[float, str] = {depth * rs[0]**2 * np.pi: "Rotor",
                                                 depth * (rs[5]**2 - rs[0]**2) * np.pi: "Al",                           # change
                                                 depth * (r_mid**2 - rs[1]**2) * np.pi: "AirGap1",
                                                 depth * (rs[2]**2 - r_mid**2) * np.pi: "AirGap0",
                                                 depth * area_helper * frac_cu: "Cu",
                                                 depth * area_helper * frac_air: "Air",
                                                 depth * (rs[4]**2 - rs[3]**2) * np.pi: "Stator",
                                                 float(L_actual**2 * depth - depth * np.pi * rs[4]**2): "Air",
                                                 depth * area_helper1 * frac_pm: "PM",
                                                 depth * area_helper1 * frac_al: "Al",
                                                 depth * (rs[1]**2 - rs[6]**2) * np.pi: "Al"}

        # Helper for assigning current wire tag to copper windings
        cu_points = np.asarray([[np.cos(angle), np.sin(angle)] for angle in angles])
        pm_points = np.asarray([[np.cos(angle), np.sin(angle)] for angle in pm_angles])

        # Assign physical surfaces based on the mass of the segment
        # For copper wires order them counter clockwise
        other_air_markers = []
        other_al_markers = []
        assigned_volumes = set()  # Track which volumes have been assigned

        for volume in volumes:
            
            mass = gmsh.model.occ.get_mass(volume[0], volume[1])
            found_domain = False
            # Use relative tolerance for mass matching (1% tolerance)
            for _mass in _area_to_domain_map.keys():
                if np.isclose(mass, _mass, rtol=0.01):
                    domain_type = _area_to_domain_map[_mass]
                    print(domain_type)
                    if domain_type == "Cu":
                        com = gmsh.model.occ.get_center_of_mass(volume[0], volume[1])
                        point = np.array([com[0], com[1]]) / np.sqrt(com[0]**2 + com[1]**2)
                        index = np.flatnonzero(np.isclose(cu_points, point).all(axis=1))[0]
                        marker = domain_map[domain_type][index]
                        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], marker)
                        assigned_volumes.add(volume[1])
                        found_domain = True
                        break
                    elif domain_type == "PM":
                        com = gmsh.model.occ.get_center_of_mass(volume[0], volume[1])
                        point = np.array([com[0], com[1]]) / np.sqrt(com[0]**2 + com[1]**2)
                        index = np.flatnonzero(np.isclose(pm_points, point).all(axis=1))[0]
                        marker = domain_map[domain_type][index]
                        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], marker)
                        assigned_volumes.add(volume[1])
                        found_domain = True
                        break
                    elif domain_type == "AirGap0":
                        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], domain_map["AirGap"][0])
                        assigned_volumes.add(volume[1])
                        found_domain = True
                        break
                    elif domain_type == "AirGap1":
                        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], domain_map["AirGap"][1])
                        assigned_volumes.add(volume[1])
                        found_domain = True
                        break

                    elif domain_type == "Air":
                        other_air_markers.append(volume[1])
                        found_domain = True
                        break
                    elif domain_type == "Al":
                        other_al_markers.append(volume[1])
                        found_domain = True
                        break
                    else:
                        marker = domain_map[domain_type][0]
                        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], marker)
                        assigned_volumes.add(volume[1])
                        found_domain = True
                        break
            if not found_domain:
                # Try to find closest match
                closest_mass = min(_area_to_domain_map.keys(), key=lambda x: abs(x - mass))
                relative_error = abs(mass - closest_mass) / max(abs(closest_mass), 1e-10)
                if relative_error < 0.15:  # 15% tolerance as fallback
                    print(f"⚠️  Warning: Domain {volume[1]} mass {mass:.6e} close to {closest_mass:.6e} (error: {relative_error*100:.2f}%), using closest match")
                    domain_type = _area_to_domain_map[closest_mass]
                    if domain_type == "Air":
                        other_air_markers.append(volume[1])
                    elif domain_type == "Al":
                        other_al_markers.append(volume[1])
                    else:
                        marker = domain_map[domain_type][0] if domain_type in domain_map else domain_map["Air"][0]
                        if marker == domain_map["Air"][0]:
                            other_air_markers.append(volume[1])
                        else:
                            gmsh.model.addPhysicalGroup(volume[0], [volume[1]], marker)
                            assigned_volumes.add(volume[1])
                else:
                    # Default to Air for unmatched volumes (likely outer air box fragments)
                    print(f"⚠️  Warning: Domain {volume[1]} mass {mass:.6e} doesn't match any expected domain (closest: {closest_mass:.6e}, error: {relative_error*100:.2f}%). Assigning to Air.")
                    other_air_markers.append(volume[1])

        # Assign air domains (only if there are any)
        if other_air_markers:
            gmsh.model.addPhysicalGroup(volume[0], other_air_markers, domain_map["Air"][0])

        # Assign Al domains (only if there are any)
        if other_al_markers:
            gmsh.model.addPhysicalGroup(volume[0], other_al_markers, domain_map["Al"][0])

        # Mark air gap boundary and exterior box
        surfaces = gmsh.model.getBoundary(volumes, combined=False, oriented=False)
        surfaces_filtered = set([surface[1] for surface in surfaces])
        air_gap_circumference = 2 * r_mid * np.pi * depth

        for surface in surfaces_filtered:
            length = gmsh.model.occ.get_mass(gdim - 1, surface)
            if np.isclose(length - air_gap_circumference, 0):
                gmsh.model.addPhysicalGroup(gdim - 1, [surface], surface_map["MidAir"])
        lines = gmsh.model.getBoundary(surfaces, combined=True, oriented=False)
        gmsh.model.addPhysicalGroup(gdim - 1, [surface[1] for surface in surfaces], surface_map["Exterior"])

        # Generate mesh with improved resolution settings
        res_base = res  # base resolution near machine
        
        # Field 1: Distance field (computes distances from center line)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", [cline])
        
        # Field 2: Threshold field (distance-based transition)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res_base)  # fine near machine
        gmsh.model.mesh.field.setNumber(2, "LcMax", 10 * res_base)  # coarse outer air, but not insane
        gmsh.model.mesh.field.setNumber(2, "DistMin", r5)  # start coarsening after stator
        gmsh.model.mesh.field.setNumber(2, "DistMax", outer_air_factor * r5)  # coarsen up to outer box
        
        # Field 3: Min field (combines all fields)
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)

        # gmsh.option.setNumber("Mesh.Algorithm", 7)
        gmsh.option.setNumber("General.Terminal", 1)            # Generates triangular elements
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)         # Generates quadrilateral elements
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write(str(filename.with_suffix(".msh")))

    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMSH scripts to generate PMSM engines for",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--res", default=0.01, type=np.float64, dest="res",
                        help="Mesh resolution")
    parser.add_argument("--L", default=1, type=np.float64, dest="L",
                        help="Size of surround box with air")
    parser.add_argument("--depth", default=0.057, type=np.float64, dest="depth",
                        help="Height of surround box with air")

    args = parser.parse_args()
    L = args.L
    res = args.res
    depth = args.depth

    folder = Path("../../meshes/3d")
    folder.mkdir(parents=True, exist_ok=True)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%b_%d_%H_%M_%S")
    fname = folder / f"pmesh3D_test1"    #
    generate_PMSM_mesh(fname, False, res, L, depth)

    result = dolfinx.io.gmshio.read_from_msh(
        str(fname.with_suffix(".msh")), MPI.COMM_WORLD, 0, gdim=3)
    mesh = result[0]
    cell_markers = result[1] if len(result) > 1 else None
    facet_markers = result[2] if len(result) > 2 else None

    # Retag cells based on geometry if tags are incomplete
    if cell_markers is not None:
        current_tags = set(np.unique(cell_markers.values))
        required_tags = {1, 2, 3, 4, 5, 6}  # Basic domains
        required_tags.update(range(7, 13))  # Coils (7-12)
        required_tags.update(range(13, 23))  # Magnets (13-22)
        
        if not required_tags.issubset(current_tags):
            print("\n🔄 Retagging cells based on geometry (GMSH tags incomplete)...")
            
            # Get cell centers
            coords = mesh.geometry.x
            dofmap = mesh.geometry.dofmap
            centers = coords[dofmap].mean(axis=1)
            radii = np.linalg.norm(centers[:, :2], axis=1)
            angles = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)
            
            # Domain radii
            r1 = mesh_parameters["r1"]
            r2 = mesh_parameters["r2"]
            r3 = mesh_parameters["r3"]
            r4 = mesh_parameters["r4"]
            r5 = mesh_parameters["r5"]
            r6 = mesh_parameters["r6"]
            r7 = mesh_parameters["r7"]
            r_mid = 0.5 * (r2 + r3)
            
            # Coil and PM angles
            coil_spacing = (np.pi / 4) + (np.pi / 4) / 3
            coil_centers = np.asarray([i * coil_spacing for i in range(6)])
            pm_spacing = (np.pi / 6) + (np.pi / 30)
            pm_centers = np.asarray([i * pm_spacing for i in range(10)])
            
            coil_half = np.pi / 8 + np.deg2rad(2.0)
            pm_half = np.pi / 12 + np.deg2rad(2.0)
            radial_tol = 5e-4
            
            def _nearest(theta, centers_array):
                diffs = np.arctan2(np.sin(theta - centers_array), np.cos(theta - centers_array))
                idx = int(np.argmin(np.abs(diffs)))
                return idx, float(abs(diffs[idx]))
            
            # Retag all cells
            new_tags = np.empty_like(cell_markers.values)
            for cell in range(len(new_tags)):
                r = radii[cell]
                theta = angles[cell]
                tag = 1  # Default to Air
                
                if r <= r1 + radial_tol:
                    tag = 5  # Rotor
                elif r <= r6 - radial_tol:
                    tag = 4  # Al
                elif r <= r7 + radial_tol:
                    idx, delta = _nearest(theta, pm_centers)
                    if delta <= pm_half:
                        tag = 13 + idx  # PM (13-22)
                    else:
                        tag = 4  # Al
                elif r <= r2 - radial_tol:
                    tag = 4  # Al
                elif r <= r_mid + radial_tol:
                    tag = 2  # AirGap0
                elif r <= r3 + radial_tol:
                    tag = 3  # AirGap1
                elif r <= r4 + radial_tol:
                    idx, delta = _nearest(theta, coil_centers)
                    if delta <= coil_half:
                        tag = 7 + idx  # Coil (7-12)
                    else:
                        tag = 1  # Air
                elif r <= r5 + radial_tol:
                    tag = 6  # Stator
                else:
                    tag = 1  # Air
                
                new_tags[cell] = tag
            
            # Create new meshtags
            cell_indices = np.arange(len(new_tags), dtype=np.int32)
            cell_markers = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cell_indices, new_tags)
            unique_tags = sorted(set(new_tags.tolist()))
            print(f"   ✅ Retagged mesh. Unique cell tags: {unique_tags}")
            print(f"   Total cells: {len(new_tags)}")
            for tag in unique_tags:
                count = np.sum(new_tags == tag)
                print(f"      Tag {tag}: {count} cells ({100*count/len(new_tags):.1f}%)")

    cell_markers.name = "Cell_markers"
    if facet_markers is not None:
        facet_markers.name = "Facet_markers"
    
    # Create cell tag function for ParaView visualization
    print("\n🎨 Creating cell tag function for ParaView...")
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    cell_tag_function = dolfinx.fem.Function(DG0, name="CellTags")
    
    if cell_markers is not None:
        # Map cell markers to function values
        cell_to_tag = {int(i): int(v) for i, v in zip(cell_markers.indices, cell_markers.values)}
        print(f"   Found {len(cell_to_tag)} tagged cells")
        
        for cell_idx, tag in cell_to_tag.items():
            if cell_idx < cell_tag_function.x.array.size:
                cell_tag_function.x.array[cell_idx] = float(tag)
        
        # Get unique tags for reporting
        unique_tags = sorted(set(cell_to_tag.values()))
        print(f"   Unique domain tags: {unique_tags}")
    
    # Write XDMF with mesh, meshtags, and cell tag function
    print(f"\n💾 Writing XDMF file: {fname.with_suffix('.xdmf')}")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        
        if cell_markers is not None:
            xdmf.write_meshtags(cell_markers, mesh.geometry)
            xdmf.write_function(cell_tag_function, 0.0)
        
        if facet_markers is not None:
            xdmf.write_meshtags(facet_markers, mesh.geometry)
    
    print("✅ XDMF file written with cell tag function")
    print("\n📋 ParaView Instructions:")
    print("   1. Open the XDMF file in ParaView")
    print("   2. Click 'Apply'")
    print("   3. In the 'Coloring' dropdown, select 'CellTags'")
    print("   4. Use 'Discrete' color map for distinct domain colors")
    print("   5. Adjust opacity if needed")