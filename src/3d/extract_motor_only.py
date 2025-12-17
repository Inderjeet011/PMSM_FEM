#!/usr/bin/env python3
"""
Extract motor-only mesh and data (exclude airbox completely).

This creates a new XDMF/H5 file containing ONLY motor cells - no airbox at all.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import h5py
from dolfinx import fem, io, mesh as dmesh
from mpi4py import MPI
from load_mesh import AIR


def extract_motor_mesh(input_path, output_path, *, rewrite_xdmf: bool = False, write_pv_safe: bool = False):
    """Extract motor-only mesh and data."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_h5 = input_path.with_suffix('.h5')
    output_h5 = output_path.with_suffix('.h5')
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_h5}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(" EXTRACTING MOTOR-ONLY MESH (NO AIRBOX)")
    print("=" * 70)
    print(f"\n📖 Reading: {input_path}")
    
    # Read mesh and tags
    with io.XDMFFile(MPI.COMM_WORLD, str(input_path), "r") as xdmf_in:
        mesh = xdmf_in.read_mesh()
        
        cell_tags = None
        for name in ["Cell_markers", "mesh_tags"]:
            try:
                cell_tags = xdmf_in.read_meshtags(mesh, name=name)
                break
            except:
                continue
        
        if cell_tags is None:
            raise ValueError("Could not find cell tags in input file")
    
    # Find motor cells (exclude Air tag 1)
    print("\n🔍 Identifying motor cells...")
    cell_to_tag = {int(i): int(v) for i, v in zip(cell_tags.indices, cell_tags.values)}
    motor_cells = []
    
    for cell_idx in range(mesh.topology.index_map(mesh.topology.dim).size_local):
        tag = cell_to_tag.get(cell_idx, 1)
        if tag != AIR[0]:  # Motor cell (not Air)
            motor_cells.append(cell_idx)
    
    motor_cells = np.array(motor_cells, dtype=np.int32)
    
    if mesh.comm.rank == 0:
        total_cells = mesh.topology.index_map(mesh.topology.dim).size_global
        motor_count = len(motor_cells)
        print(f"   Total cells: {total_cells}")
        print(f"   Motor cells: {motor_count}")
        print(f"   Airbox cells excluded: {total_cells - motor_count}")
    
    if len(motor_cells) == 0:
        raise ValueError("No motor cells found!")
    
    # Create submesh with only motor cells
    print("\n🔧 Creating motor-only submesh...")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    
    # Get all vertices used by motor cells
    c2v = mesh.topology.connectivity(tdim, 0)
    motor_vertices = set()
    for cell in motor_cells:
        vertices = c2v.links(cell)
        motor_vertices.update(vertices)
    
    motor_vertices = np.array(sorted(motor_vertices), dtype=np.int32)
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(motor_vertices)}
    
    # Get coordinates for motor vertices
    coords = mesh.geometry.x
    motor_coords = coords[motor_vertices].copy()  # Ensure contiguous array
    
    # Build new connectivity - need proper array format
    new_cells_list = []
    new_cell_tags = []
    
    for cell in motor_cells:
        vertices = c2v.links(cell)
        new_vertices = np.array([vertex_map[v] for v in vertices], dtype=np.int32)
        new_cells_list.append(new_vertices)
        tag = cell_to_tag.get(cell, 1)
        new_cell_tags.append(tag)
    
    # Convert to proper array format for mesh creation
    # For tetrahedra, each cell has 4 vertices
    if len(new_cells_list) > 0:
        new_cells = np.vstack(new_cells_list).astype(np.int32)
    else:
        raise ValueError("No cells to create mesh")
    
    # Create new mesh - use UFL mesh from original
    import ufl
    try:
        motor_mesh = dmesh.create_mesh(
            mesh.comm,
            new_cells,
            mesh.ufl_domain(),  # Use UFL domain from original mesh
            motor_coords
        )
    except Exception as e:
        if mesh.comm.rank == 0:
            print(f"   Error creating mesh: {e}")
            print(f"   Trying alternative method...")
        # Alternative: use basix element
        import basix.ufl
        cell_type = mesh.basix_cell()
        element = basix.ufl.element("Lagrange", cell_type, 1, shape=(3,))
        motor_mesh = dmesh.create_mesh(
            mesh.comm,
            new_cells,
            element,
            motor_coords
        )
    
    # Create new cell tags
    new_cell_indices = np.arange(len(new_cells), dtype=np.int32)
    motor_ct = dmesh.meshtags(motor_mesh, tdim, new_cell_indices, 
                              np.array(new_cell_tags, dtype=np.int32))
    
    if mesh.comm.rank == 0:
        print(f"   ✅ Submesh created: {motor_mesh.topology.index_map(tdim).size_global} cells")
        print(f"   ✅ Vertices: {motor_mesh.geometry.x.shape[0]}")
    
    def _time_from_h5_key(key: str) -> float:
        # dolfinx XDMF/HDF5 keys are often like "0_00083333333333333328"
        # meaning time = 0.00083333333333333328.
        parts = key.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return float(parts[0] + "." + parts[1])
        # Fallback: try to parse directly
        return float(key)

    def _write_paraview_friendly_xdmf(
        xdmf_path: Path,
        h5_path: Path,
        function_configs: dict,
        *,
        include_cell_vectors: bool = True,
    ):
        """
        Write a simple, ParaView-friendly XDMF:
        - One Temporal Collection grid named "TimeSeries"
        - Each timestep grid contains all attributes (A, V, B, B_Magnitude, B_dg)
        - No xi:include / xpointer (these sometimes crash ParaView builds)
        """
        import h5py
        import xml.etree.ElementTree as ET

        h5_path = Path(h5_path)
        with h5py.File(str(h5_path), "r") as f:
            ncell = int(f["Mesh/mesh/topology"].shape[0])
            nnode = int(f["Mesh/mesh/geometry"].shape[0])

            # Collect available functions and a common set of timestep keys
            funcs = list(f["Function"].keys())
            # Choose timestep keys from the first available function
            first = funcs[0]
            keys = sorted(list(f[f"Function/{first}"].keys()), key=_time_from_h5_key)

        # ParaView's XDMF support can vary; "2.0" is typically the most compatible.
        Xdmf = ET.Element("Xdmf", Version="2.0")
        Domain = ET.SubElement(Xdmf, "Domain")

        # Base mesh grid
        mesh_grid = ET.SubElement(Domain, "Grid", Name="mesh", GridType="Uniform")
        topo = ET.SubElement(mesh_grid, "Topology", TopologyType="Tetrahedron", NumberOfElements=str(ncell), NodesPerElement="4")
        topo_di = ET.SubElement(topo, "DataItem", Dimensions=f"{ncell} 4", NumberType="Int", Format="HDF")
        topo_di.text = f"{h5_path.name}:/Mesh/mesh/topology"
        geom = ET.SubElement(mesh_grid, "Geometry", GeometryType="XYZ")
        geom_di = ET.SubElement(geom, "DataItem", Dimensions=f"{nnode} 3", Format="HDF")
        geom_di.text = f"{h5_path.name}:/Mesh/mesh/geometry"

        # Cell tags (optional, but useful)
        tags_grid = ET.SubElement(Domain, "Grid", Name="mesh_tags", GridType="Uniform")
        tags_topo = ET.SubElement(tags_grid, "Topology", TopologyType="Tetrahedron", NumberOfElements=str(ncell), NodesPerElement="4")
        tags_topo_di = ET.SubElement(tags_topo, "DataItem", Dimensions=f"{ncell} 4", NumberType="Int", Format="HDF")
        tags_topo_di.text = f"{h5_path.name}:/MeshTags/mesh_tags/topology"
        tags_geom = ET.SubElement(tags_grid, "Geometry", GeometryType="XYZ")
        tags_geom_di = ET.SubElement(tags_geom, "DataItem", Dimensions=f"{nnode} 3", Format="HDF")
        tags_geom_di.text = f"{h5_path.name}:/Mesh/mesh/geometry"
        tags_attr = ET.SubElement(tags_grid, "Attribute", Name="mesh_tags", AttributeType="Scalar", Center="Cell")
        tags_attr_di = ET.SubElement(tags_attr, "DataItem", Dimensions=f"{ncell} 1", Format="HDF")
        tags_attr_di.text = f"{h5_path.name}:/MeshTags/mesh_tags/Values"

        # Single time-series grid with all attributes per timestep
        ts = ET.SubElement(Domain, "Grid", Name="TimeSeries", GridType="Collection", CollectionType="Temporal")
        for key in keys:
            t = _time_from_h5_key(key)
            g = ET.SubElement(ts, "Grid", Name="timestep", GridType="Uniform")
            # topology+geometry (repeat explicitly; no xi:include)
            ttopo = ET.SubElement(g, "Topology", TopologyType="Tetrahedron", NumberOfElements=str(ncell), NodesPerElement="4")
            ttopo_di = ET.SubElement(ttopo, "DataItem", Dimensions=f"{ncell} 4", NumberType="Int", Format="HDF")
            ttopo_di.text = f"{h5_path.name}:/Mesh/mesh/topology"
            tgeom = ET.SubElement(g, "Geometry", GeometryType="XYZ")
            tgeom_di = ET.SubElement(tgeom, "DataItem", Dimensions=f"{nnode} 3", Format="HDF")
            tgeom_di.text = f"{h5_path.name}:/Mesh/mesh/geometry"

            ET.SubElement(g, "Time", Value=str(t))

            # Write each known function if present
            for fname, space_config in function_configs.items():
                # Determine center and dims from space_config
                if len(space_config) >= 2 and space_config[0] == "DG" and space_config[1] == 0:
                    if not include_cell_vectors:
                        # Some ParaView builds crash on cell-centered vector attributes.
                        # Keep node fields (A, V, B, B_Magnitude) for visualization.
                        continue
                    center = "Cell"
                    n = ncell
                else:
                    center = "Node"
                    n = nnode

                # Determine vector vs scalar
                shape = space_config[2] if len(space_config) >= 3 else ()
                is_vec = isinstance(shape, tuple) and len(shape) == 1 and shape[0] == 3 or shape == (3,)
                attr_type = "Vector" if is_vec else "Scalar"
                ncomp = 3 if is_vec else 1

                attr = ET.SubElement(g, "Attribute", Name=fname, AttributeType=attr_type, Center=center)
                di = ET.SubElement(attr, "DataItem", Dimensions=f"{n} {ncomp}", Format="HDF")
                di.text = f"{h5_path.name}:/Function/{fname}/{key}"

        # Pretty-print to avoid extremely long single-line XML (some readers dislike it)
        try:
            ET.indent(Xdmf, space="  ")  # Python 3.9+
        except Exception:
            pass
        xml = ET.tostring(Xdmf, encoding="utf-8", xml_declaration=False)

        # Write a single XML declaration (avoid duplicate declarations that can crash readers)
        header = b'<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
        xdmf_path.write_bytes(header + xml + b"\n")
    
    # Read function data and extract motor cells only
    print("\n📖 Reading and extracting function data...")
    function_configs = {
        "A": ("Lagrange", 1, (3,)),
        "V": ("Lagrange", 1,),
        "B": ("Lagrange", 1, (3,)),
        "B_dg": ("DG", 0, (3,)),
        "B_Magnitude": ("Lagrange", 1,),
        "B_vis": ("Lagrange", 1, (3,)),
        "B_vis_mag": ("Lagrange", 1,),
    }
    
    # Map old cell indices to new cell indices
    old_to_new_cell = {old_idx: new_idx for new_idx, old_idx in enumerate(motor_cells)}
    
    # Read all function data and extract motor portion
    motor_function_data = {}
    with h5py.File(str(input_h5), 'r') as f_in:
        # Prefer discovering functions directly from the HDF5 (reliable).
        function_names = []
        if "Function" in f_in:
            function_names = sorted(list(f_in["Function"].keys()))
        if mesh.comm.rank == 0:
            print(f"\n📊 Found functions in HDF5: {function_names}")

        for func_name in function_names:
            if func_name not in function_configs:
                continue
            motor_function_data[func_name] = {}
            for timestep_key in f_in[f'Function/{func_name}'].keys():
                data = f_in[f'Function/{func_name}/{timestep_key}'][:]
                space_config = function_configs[func_name]
                # DG0 data is cell-centered (even for vector-valued DG0 like B_dg).
                if len(space_config) >= 2 and space_config[0] == "DG" and space_config[1] == 0:
                    motor_data = data[motor_cells]
                else:
                    motor_data = data[motor_vertices]
                motor_function_data[func_name][timestep_key] = motor_data

    # Wait so rank 0 prints don't interleave badly
    mesh.comm.Barrier()
    
    # Write XDMF/HDF5 via dolfinx (do NOT pre-create the HDF5 with h5py, because
    # XDMFFile("w") will truncate/overwrite the .h5 anyway).
    print(f"💾 Writing motor-only XDMF/HDF5 file...")
    with io.XDMFFile(MPI.COMM_WORLD, str(output_path), "w") as xdmf_out:
        xdmf_out.write_mesh(motor_mesh)
        xdmf_out.write_meshtags(motor_ct, motor_mesh.geometry)
        
        # Write all functions at their native timesteps (parsed from the HDF5 keys)
        for func_name, series in motor_function_data.items():
            if func_name not in function_configs:
                continue
            space_config = function_configs[func_name]
            func_space = fem.functionspace(motor_mesh, space_config)
            func = fem.Function(func_space, name=func_name)

            keys_sorted = sorted(series.keys(), key=_time_from_h5_key)
            for h5_key in keys_sorted:
                t = _time_from_h5_key(h5_key)
                data = series[h5_key]
                flat = data.reshape(-1)
                n = min(len(func.x.array), len(flat))
                func.x.array[:] = 0.0
                func.x.array[:n] = flat[:n]
                func.x.scatter_forward()
                xdmf_out.write_function(func, t)

    # Optional: rewrite XDMF for ParaView compatibility and/or write a _pv_safe variant.
    # For "no format change", keep rewrite_xdmf=False and write_pv_safe=False.
    if mesh.comm.rank == 0 and (rewrite_xdmf or write_pv_safe):
        out_path = Path(output_path)
        h5_path = out_path.with_suffix(".h5")
        if rewrite_xdmf:
            _write_paraview_friendly_xdmf(out_path, h5_path, function_configs, include_cell_vectors=True)
        if write_pv_safe:
            pv_safe = out_path.with_name(out_path.stem + "_pv_safe.xdmf")
            _write_paraview_friendly_xdmf(pv_safe, h5_path, function_configs, include_cell_vectors=False)
    
    if mesh.comm.rank == 0:
        print(f"\n✅ Complete! Motor-only file ready: {output_path}")
        print(f"   Contains only {len(motor_cells)} motor cells (no airbox)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract motor-only mesh and data (exclude airbox)"
    )
    parser.add_argument(
        "--input", "-i",
        default="../../results/3d/av_solver.xdmf",
        help="Input XDMF results file"
    )
    parser.add_argument(
        "--output", "-o",
        default="../../results/3d/av_solver_motor_only.xdmf",
        help="Output XDMF file (motor only)"
    )
    
    args = parser.parse_args()
    
    try:
        extract_motor_mesh(args.input, args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

