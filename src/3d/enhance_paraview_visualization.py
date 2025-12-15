#!/usr/bin/env python3
"""
Enhance XDMF file for better ParaView visualization
Creates separate function fields and filters for better visibility
"""

import sys
from pathlib import Path
from dolfinx import fem, io
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np

def enhance_xdmf_for_paraview(xdmf_file, output_xdmf=None):
    """Enhance XDMF file with better visualization fields"""
    
    if output_xdmf is None:
        output_xdmf = Path(xdmf_file).parent / (Path(xdmf_file).stem + "_enhanced.xdmf")
    
    print(f"Loading mesh from XDMF: {xdmf_file}")
    
    # Read from XDMF (which has retagged cells)
    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_file), "r") as xdmf:
        mesh = xdmf.read_mesh()
        try:
            cell_markers = xdmf.read_meshtags(mesh, name="mesh_tags")
        except:
            cell_markers = None
        try:
            facet_markers = xdmf.read_meshtags(mesh, name="facet_tags")
        except:
            facet_markers = None
    
    print(f"Mesh: {mesh.topology.index_map(mesh.topology.dim).size_local} cells")
    
    if cell_markers is None:
        print("❌ No cell markers found!")
        return None
    
    # Create DG0 function space
    DG0 = fem.functionspace(mesh, ("DG", 0))
    
    # Create cell tags function
    cell_tag_function = fem.Function(DG0)
    cell_tag_function.name = "CellTags"
    cell_tag_function.x.array[:] = 0.0
    
    # Map all cells (not just tagged ones)
    cell_to_tag = {int(i): int(v) for i, v in zip(cell_markers.indices, cell_markers.values)}
    n_cells = mesh.topology.index_map(3).size_local
    
    # Get cell centers for retagging if needed
    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    centers = coords[dofmap].mean(axis=1)
    radii = np.linalg.norm(centers[:, :2], axis=1)
    
    # Tag all cells
    for cell_idx in range(n_cells):
        if cell_idx in cell_to_tag:
            cell_tag_function.x.array[cell_idx] = float(cell_to_tag[cell_idx])
        else:
            # Untagged cells - classify by radius
            r = radii[cell_idx]
            if r < 0.075:  # Inside motor region
                cell_tag_function.x.array[cell_idx] = 0.0  # Unknown
            else:
                cell_tag_function.x.array[cell_idx] = 1.0  # Air
    
    # Create component visibility function (0 = Air, 1 = Motor components)
    visibility_function = fem.Function(DG0)
    visibility_function.name = "MotorComponents"
    
    for cell_idx in range(n_cells):
        tag = int(cell_tag_function.x.array[cell_idx])
        # Tag 1 is Air, everything else is motor component
        visibility_function.x.array[cell_idx] = 0.0 if tag == 1 else 1.0
    
    # Create material type function (simplified categories)
    material_function = fem.Function(DG0)
    material_function.name = "MaterialType"
    
    material_map = {
        1: 0,   # Air
        2: 1,   # AirGap
        3: 1,   # AirGap
        4: 2,   # Aluminum
        5: 3,   # Rotor
        6: 4,   # Stator
    }
    # Copper (7-12) -> 5
    # PM (13-22) -> 6
    
    for cell_idx in range(n_cells):
        tag = int(cell_tag_function.x.array[cell_idx])
        if tag in material_map:
            material_function.x.array[cell_idx] = float(material_map[tag])
        elif 7 <= tag <= 12:
            material_function.x.array[cell_idx] = 5.0  # Copper
        elif 13 <= tag <= 22:
            material_function.x.array[cell_idx] = 6.0  # PM
        else:
            material_function.x.array[cell_idx] = 0.0  # Unknown/Air
    
    # Write enhanced XDMF
    print(f"Writing enhanced XDMF: {output_xdmf}")
    with io.XDMFFile(MPI.COMM_WORLD, str(output_xdmf), "w") as xdmf:
        xdmf.write_mesh(mesh)
        
        # Write meshtags
        xdmf.write_meshtags(cell_markers, mesh.geometry)
        if facet_markers is not None:
            xdmf.write_meshtags(facet_markers, mesh.geometry)
        
        # Write function fields
        xdmf.write_function(cell_tag_function, 0.0)
        xdmf.write_function(visibility_function, 0.0)
        xdmf.write_function(material_function, 0.0)
    
    unique_tags = sorted(set(cell_to_tag.values()))
    print(f"✅ Enhanced XDMF written: {output_xdmf}")
    print(f"   Unique tags: {unique_tags}")
    print()
    print("=" * 70)
    print("📋 PARAVIEW VISUALIZATION INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Method 1 - View Motor Components Only (Recommended):")
    print("  1. Open the XDMF file in ParaView")
    print("  2. Click 'Apply'")
    print("  3. Add a 'Threshold' filter:")
    print("     - Input: pmesh3D_ipm")
    print("     - Scalar: MotorComponents")
    print("     - Lower threshold: 0.5")
    print("     - Upper threshold: 1.0")
    print("  4. Click 'Apply' on Threshold")
    print("  5. In Coloring, select 'CellTags'")
    print("  6. Set color map to 'Discrete'")
    print()
    print("Method 2 - View All with Material Types:")
    print("  1. Open the XDMF file")
    print("  2. Click 'Apply'")
    print("  3. In Coloring, select 'MaterialType'")
    print("  4. Set color map to 'Discrete'")
    print("  5. Adjust opacity: Edit → Opacity → 0.1 for Air (MaterialType=0)")
    print()
    print("Method 3 - View Specific Components:")
    print("  1. Use 'MaterialType' field")
    print("  2. Add 'Threshold' filter to show only specific materials:")
    print("     - MaterialType = 3 → Rotor")
    print("     - MaterialType = 4 → Stator")
    print("     - MaterialType = 6 → Permanent Magnets")
    print("     - MaterialType = 5 → Copper Windings")
    print()
    print("=" * 70)
    
    return output_xdmf

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance XDMF for ParaView visualization")
    parser.add_argument("--xdmf", default="../../meshes/3d/pmesh3D_ipm.xdmf", 
                       help="Input XDMF file")
    parser.add_argument("--output", default="../../meshes/3d/pmesh3D_ipm_enhanced.xdmf",
                       help="Output XDMF file")
    
    args = parser.parse_args()
    enhance_xdmf_for_paraview(args.xdmf, args.output)

