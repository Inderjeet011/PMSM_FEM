#!/usr/bin/env python3
"""
Create a better XDMF file with proper domain coloring for Paraview
This ensures all domains are properly tagged and can be colored in Paraview
"""

import sys
from pathlib import Path
from dolfinx import fem, io
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np

def enhance_xdmf_with_colors(mesh_file, output_xdmf=None):
    """Enhance XDMF file with proper cell tag coloring"""
    
    if output_xdmf is None:
        output_xdmf = Path(mesh_file).with_suffix('.xdmf')
    
    print(f"Loading mesh: {mesh_file}")
    result = gmshio.read_from_msh(str(mesh_file), MPI.COMM_WORLD, rank=0, gdim=3)
    mesh = result[0]
    cell_markers = result[1] if len(result) > 1 else None
    facet_markers = result[2] if len(result) > 2 else None
    
    print(f"Mesh: {mesh.topology.index_map(mesh.topology.dim).size_local} cells")
    
    # Create DG0 function space for cell tags
    DG0 = fem.functionspace(mesh, ("DG", 0))
    cell_tag_function = fem.Function(DG0)
    
    if cell_markers is not None:
        cell_to_tag = {int(i): int(v) for i, v in zip(cell_markers.indices, cell_markers.values)}
        print(f"Found {len(cell_to_tag)} tagged cells")
        
        # Map tags to function
        for cell_idx, tag in cell_to_tag.items():
            if cell_idx < cell_tag_function.x.array.size:
                cell_tag_function.x.array[cell_idx] = float(tag)
        
        cell_tag_function.name = "CellTags"
        
        # Get unique tags
        unique_tags = sorted(set(cell_to_tag.values()))
        print(f"Unique domain tags: {unique_tags}")
    
    # Write enhanced XDMF
    print(f"Writing enhanced XDMF: {output_xdmf}")
    with io.XDMFFile(MPI.COMM_WORLD, str(output_xdmf), "w") as xdmf:
        xdmf.write_mesh(mesh)
        
        if cell_markers is not None:
            xdmf.write_meshtags(cell_markers, mesh.geometry)
            xdmf.write_function(cell_tag_function, 0.0)
        
        if facet_markers is not None:
            xdmf.write_meshtags(facet_markers, mesh.geometry)
    
    print(f"✅ Enhanced XDMF written: {output_xdmf}")
    print("\n📋 Paraview Instructions:")
    print("1. Open the XDMF file in Paraview")
    print("2. Click 'Apply'")
    print("3. In the 'Coloring' dropdown, select 'CellTags'")
    print("4. Use 'Discrete' color map for distinct domain colors")
    print("5. Adjust opacity if needed")
    
    return output_xdmf

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhance XDMF with proper domain coloring")
    parser.add_argument("--mesh", default="../../meshes/3d/pmesh3D_test1.msh",
                       help="Input mesh file (.msh)")
    parser.add_argument("--out", default=None,
                       help="Output XDMF file (default: same as mesh with .xdmf)")
    args = parser.parse_args()
    
    mesh_path = Path(args.mesh)
    if not mesh_path.is_absolute():
        mesh_path = Path.cwd() / mesh_path
    
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
    else:
        out_path = mesh_path.with_suffix('.xdmf')
    
    enhance_xdmf_with_colors(str(mesh_path), str(out_path))

