#!/usr/bin/env python3
"""
3D Mesh plotting utility for FEniCS/DOLFINx 3D meshes created via Gmsh.

Usage examples:
  python3 plot_mesh_3d.py --mesh ../../meshes/3d/pmesh3D_test1.msh --out ../../results/3d/mesh_preview.png
  python3 plot_mesh_3d.py --mesh ../../meshes/3d/pmesh3D_test1.msh --show --slice
  python3 plot_mesh_3d.py --mesh ../../meshes/3d/pmesh3D_test1.msh --show --wireframe

Features:
- Colors cells by cell (physical) tags if available
- Multiple visualization modes: surface, wireframe, slice
- Saves to PNG or displays interactively
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

try:
    import pyvista as pv
    pv.set_plot_theme("document")
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. Install with: pip install pyvista")
    print("Falling back to matplotlib 3D visualization...")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_mesh_with_tags(mesh_path: str, gdim: int = 3):
    """Load mesh and cell/facet tags from Gmsh file"""
    result = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, rank=0, gdim=gdim)
    mesh = result[0]
    ct = result[1] if len(result) > 1 else None
    ft = result[2] if len(result) > 2 else None
    return mesh, ct, ft


def get_domain_colors_3d() -> Dict[int, Tuple[float, float, float]]:
    """Color mapping for 3D PMSM domains based on mesh_3D.py domain_map"""
    return {
        1: (0.94, 0.96, 0.97),   # Air - light blue-gray
        2: (0.95, 0.97, 0.99),   # AirGap - very light blue
        3: (0.95, 0.97, 0.99),   # AirGap - very light blue
        4: (0.85, 0.90, 0.95),   # Al - light aluminum
        5: (0.70, 0.72, 0.75),   # Rotor - steel gray
        6: (0.55, 0.57, 0.60),   # Stator - darker iron
        7: (0.98, 0.65, 0.15),   # Cu - orange (coil)
        8: (0.98, 0.65, 0.15),   # Cu
        9: (0.98, 0.65, 0.15),   # Cu
        10: (0.98, 0.65, 0.15),  # Cu
        11: (0.98, 0.65, 0.15),  # Cu
        12: (0.98, 0.65, 0.15),  # Cu
        13: (0.85, 0.20, 0.20),  # PM - red
        14: (0.20, 0.35, 0.85),  # PM - blue
        15: (0.85, 0.20, 0.20),  # PM - red
        16: (0.20, 0.35, 0.85),  # PM - blue
        17: (0.85, 0.20, 0.20),  # PM - red
        18: (0.20, 0.35, 0.85),  # PM - blue
        19: (0.85, 0.20, 0.20),  # PM - red
        20: (0.20, 0.35, 0.85),  # PM - blue
        21: (0.85, 0.20, 0.20),  # PM - red
        22: (0.20, 0.35, 0.85),  # PM - blue
    }


def plot_mesh_pyvista(mesh, ct, out_path: Optional[str], show: bool, 
                      wireframe: bool, slice_view: bool, opacity: float):
    """Plot 3D mesh using PyVista"""
    # Convert DOLFINx mesh to PyVista
    import dolfinx.plot
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    
    # Add cell tags as data
    if ct is not None:
        cell_tags = np.zeros(grid.n_cells, dtype=int)
        cell_to_tag = {int(i): int(v) for i, v in zip(ct.indices, ct.values)}
        for i in range(grid.n_cells):
            cell_tags[i] = cell_to_tag.get(i, 0)
        grid.cell_data["CellTags"] = cell_tags
        
        # Color by tags
        domain_colors = get_domain_colors_3d()
        colors = np.array([domain_colors.get(tag, (0.9, 0.9, 0.95)) for tag in cell_tags])
        grid.cell_data["colors"] = colors
        
        # Create opacity array: make Air (tag 1) very transparent, others fully opaque
        opacities = np.array([0.05 if tag == 1 else opacity for tag in cell_tags])
        grid.cell_data["opacities"] = opacities
    
    # Create plotter with better visual settings
    plotter = pv.Plotter(off_screen=not show, lighting='three lights')
    plotter.set_background('white')
    
    if slice_view:
        # Add slice view with custom colors
        if ct is not None and "colors" in grid.cell_data:
            plotter.add_mesh(grid.slice('z'), scalars="colors", rgb=True, 
                           show_edges=wireframe, opacity=opacity, smooth_shading=True)
            plotter.add_mesh(grid.slice('x'), scalars="colors", rgb=True, 
                           show_edges=wireframe, opacity=opacity, smooth_shading=True)
            plotter.add_mesh(grid.slice('y'), scalars="colors", rgb=True, 
                           show_edges=wireframe, opacity=opacity, smooth_shading=True)
        else:
            plotter.add_mesh(grid.slice('z'), show_edges=wireframe, opacity=opacity, smooth_shading=True)
            plotter.add_mesh(grid.slice('x'), show_edges=wireframe, opacity=opacity, smooth_shading=True)
            plotter.add_mesh(grid.slice('y'), show_edges=wireframe, opacity=opacity, smooth_shading=True)
    else:
        # Full mesh - separate Air from motor components for better visualization
        if ct is not None and "colors" in grid.cell_data:
            # Filter out Air (tag 1) cells for main motor visualization
            motor_mask = cell_tags != 1
            if np.any(motor_mask):
                motor_grid = grid.extract_cells(np.where(motor_mask)[0])
                plotter.add_mesh(motor_grid, scalars="colors", rgb=True, show_edges=wireframe, 
                               opacity=opacity, smooth_shading=True)
            
            # Add Air box with very low opacity so it doesn't obscure the motor
            air_mask = cell_tags == 1
            if np.any(air_mask):
                air_grid = grid.extract_cells(np.where(air_mask)[0])
                plotter.add_mesh(air_grid, scalars="colors", rgb=True, show_edges=False, 
                               opacity=0.05, smooth_shading=True)
        else:
            plotter.add_mesh(grid, show_edges=wireframe, opacity=opacity, 
                           color=(0.9, 0.9, 0.95), smooth_shading=True)
    
    # Focus camera on motor center (assume motor is at origin)
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.5)  # Zoom in to focus on motor
    
    # Add axes and title with better styling
    plotter.add_axes(line_width=3, labels_off=False)
    plotter.add_text("3D PMSM Mesh", font_size=14, position='upper_left')
    
    if out_path:
        plotter.screenshot(out_path, window_size=[1920, 1080])
        print(f"✅ Saved mesh plot: {out_path}")
    
    if show:
        plotter.show()
    else:
        plotter.close()


def plot_mesh_matplotlib(mesh, ct, out_path: Optional[str], show: bool):
    """Fallback 3D visualization using matplotlib (basic wireframe)"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates
    coords = mesh.geometry.x
    tdim = mesh.topology.dim
    
    # Build connectivity
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    
    # Plot a subset of cells (for performance)
    n_cells = mesh.topology.index_map(tdim).size_local
    step = max(1, n_cells // 1000)  # Plot max 1000 cells
    
    domain_colors = get_domain_colors_3d()
    cell_to_tag = {int(i): int(v) for i, v in zip(ct.indices, ct.values)} if ct else {}
    
    for c in range(0, n_cells, step):
        vs = c2v.links(c)
        if len(vs) >= 3:
            pts = coords[vs]
            # Create triangle faces
            if len(vs) == 4:  # Tetrahedron
                faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            else:
                continue
            
            tag = cell_to_tag.get(c, 1)
            color = domain_colors.get(tag, (0.9, 0.9, 0.95))
            
            for face in faces:
                if all(i < len(vs) for i in face):
                    tri = pts[face]
                    ax.plot_trisurf(tri[:, 0], tri[:, 1], tri[:, 2], 
                                   color=color, alpha=0.6, edgecolor='black', linewidth=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D PMSM Mesh (matplotlib - simplified)")
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved mesh plot: {out_path}")
    
    if show:
        plt.show()
    plt.close(fig)


def plot_mesh_3d(mesh_path: str, out_path: Optional[str], show: bool, 
                wireframe: bool, slice_view: bool, opacity: float, gdim: int = 3):
    """Main plotting function"""
    print(f"Loading mesh: {mesh_path}")
    mesh, ct, ft = load_mesh_with_tags(mesh_path, gdim=gdim)
    
    print(f"Mesh info: {mesh.topology.index_map(mesh.topology.dim).size_local} cells, "
          f"{mesh.geometry.x.shape[0]} vertices")
    
    if ct:
        unique_tags = sorted(set(int(v) for v in ct.values))
        print(f"Cell tags found: {unique_tags}")
    
    if PYVISTA_AVAILABLE:
        plot_mesh_pyvista(mesh, ct, out_path, show, wireframe, slice_view, opacity)
    else:
        print("Using matplotlib fallback (limited functionality)")
        plot_mesh_matplotlib(mesh, ct, out_path, show)


def main():
    parser = argparse.ArgumentParser(description="Plot a 3D mesh (DOLFINx/Gmsh)")
    parser.add_argument("--mesh", 
                       default="../../meshes/3d/pmesh3D_test1.msh",
                       help="Path to .msh file")
    parser.add_argument("--out", 
                       default="../../results/3d/mesh_preview_3d.png",
                       help="Path to save output image (PNG)")
    parser.add_argument("--show", action="store_true", 
                       help="Display interactively")
    parser.add_argument("--wireframe", action="store_true", 
                       help="Show wireframe edges")
    parser.add_argument("--slice", action="store_true", 
                       help="Show slice views (x, y, z planes)")
    parser.add_argument("--opacity", type=float, default=0.8,
                       help="Opacity for mesh (0.0-1.0)")
    parser.add_argument("--gdim", type=int, default=3,
                       help="Geometric dimension (default: 3)")
    args = parser.parse_args()
    
    # Resolve paths - if relative, resolve from current working directory or script location
    if os.path.isabs(args.mesh):
        mesh_path = Path(args.mesh)
    else:
        # Try relative to current directory first, then script directory
        cwd_path = Path.cwd() / args.mesh
        if cwd_path.exists():
            mesh_path = cwd_path.resolve()
        else:
            script_dir = Path(__file__).parent
            mesh_path = (script_dir / args.mesh).resolve()
    
    if args.out:
        if os.path.isabs(args.out):
            out_path = Path(args.out)
        else:
            # Try relative to current directory first
            cwd_path = Path.cwd() / args.out
            if cwd_path.parent.exists():
                out_path = cwd_path.resolve()
            else:
                script_dir = Path(__file__).parent
                out_path = (script_dir / args.out).resolve()
    else:
        out_path = None
    
    if not mesh_path.exists():
        print(f"❌ Mesh file not found: {mesh_path}")
        sys.exit(1)
    
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_mesh_3d(str(mesh_path), str(out_path) if out_path else None, 
                args.show, args.wireframe, args.slice, args.opacity, args.gdim)


if __name__ == "__main__":
    main()

