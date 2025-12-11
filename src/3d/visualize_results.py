#!/usr/bin/env python3
"""
Visualize 3D solver results and generate PNG images
Reads directly from HDF5 file
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dolfinx import io, fem, geometry
from mpi4py import MPI
import h5py
import sys

def visualize_xdmf(xdmf_path: Path, output_dir: Path):
    """Read XDMF file and generate PNG visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 Reading XDMF file: {xdmf_path}")
    
    # Read mesh from XDMF
    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "r") as xdmf:
        mesh = xdmf.read_mesh()
        
        if mesh.comm.rank != 0:
            return
        
        print(f"✅ Mesh loaded: {mesh.topology.index_map(3).size_global} cells")
        
        # Get mesh coordinates
        coords = mesh.geometry.x
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        
        print(f"   Domain: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}], z=[{z_min:.3f}, {z_max:.3f}]")
    
    # Read data from HDF5
    h5_path = xdmf_path.with_suffix('.h5')
    if not h5_path.exists():
        print(f"❌ HDF5 file not found: {h5_path}")
        return
    
    print(f"📖 Reading HDF5 file: {h5_path}")
    
    with h5py.File(h5_path, 'r') as h5:
        # Find available timesteps
        timesteps = []
        if 'Function/V' in h5:
            for key in sorted(h5['Function/V'].keys()):
                timesteps.append(key)
        
        if not timesteps:
            print("   ⚠️  No timesteps found")
            return
        
        print(f"   Found {len(timesteps)} timestep(s)")
        
        # Create function spaces
        # A is stored as Lagrange vector (interpolated from Nédélec)
        A_space = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
        V_space = fem.functionspace(mesh, ("CG", 1))
        B_space = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
        B_mag_space = fem.functionspace(mesh, ("CG", 1))
        
        # Create functions
        A = fem.Function(A_space, name="A")
        V = fem.Function(V_space, name="V")
        B = fem.Function(B_space, name="B")
        B_mag = fem.Function(B_mag_space, name="B_Magnitude")
        
        # Process each timestep
        for step, time_key in enumerate(timesteps):
            print(f"\n📊 Processing timestep {step+1}/{len(timesteps)}...")
            
            try:
                # Load data from HDF5
                # Try to load A field first (main field) - A is stored as Lagrange (interpolated)
                if f'Function/A/{time_key}' in h5:
                    A_data = h5[f'Function/A/{time_key}'][:]
                    # A is interpolated to Lagrange space (3 components per node)
                    if A_data.size == A.x.array.size:
                        A.x.array[:] = A_data.flatten()
                        A.x.scatter_forward()
                        A_mag = np.linalg.norm(A.x.array.reshape(-1, 3), axis=1)
                        print(f"      A field: ||A|| = {np.linalg.norm(A.x.array):.6e}, max|A| = {np.max(A_mag):.6e}")
                    else:
                        print(f"      ⚠️  A data size mismatch: {A_data.size} vs {A.x.array.size}")
                else:
                    print(f"      ⚠️  A field not found in HDF5")
                
                if f'Function/V/{time_key}' in h5:
                    V_data = h5[f'Function/V/{time_key}'][:]
                    if V_data.size == V.x.array.size:
                        V.x.array[:] = V_data.flatten()
                        V.x.scatter_forward()
                        print(f"      V field: ||V|| = {np.linalg.norm(V.x.array):.6e}")
                
                if f'Function/B/{time_key}' in h5:
                    B_data = h5[f'Function/B/{time_key}'][:]
                    if B_data.size == B.x.array.size:
                        B.x.array[:] = B_data.flatten()
                        B.x.scatter_forward()
                        B_mag_from_B = np.linalg.norm(B.x.array.reshape(-1, 3), axis=1)
                        print(f"      B field: max|B| = {np.max(B_mag_from_B):.6e} T")
                
                if f'Function/B_Magnitude/{time_key}' in h5:
                    B_mag_data = h5[f'Function/B_Magnitude/{time_key}'][:]
                    if B_mag_data.size == B_mag.x.array.size:
                        B_mag.x.array[:] = B_mag_data.flatten()
                        B_mag.x.scatter_forward()
                        print(f"      B_Magnitude: max = {np.max(B_mag.x.array):.6e} T")
                
                # Get z-mid slice
                z_mid = 0.5 * (z_min + z_max)
                
                # Create visualizations
                print("   Generating PNG images...")
                
                # 1. B Magnitude - XY slice at z=z_mid
                fig, ax = plt.subplots(figsize=(10, 10))
                plot_B_magnitude_xy_slice(mesh, B_mag, z_mid, ax)
                plt.tight_layout()
                png_path = output_dir / f"B_magnitude_t{step:03d}_zmid.png"
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"      ✅ {png_path.name}")
                
                # 2. B Vector field - XY slice
                fig, ax = plt.subplots(figsize=(10, 10))
                plot_B_vector_xy_slice(mesh, B, z_mid, ax)
                plt.tight_layout()
                png_path = output_dir / f"B_vector_t{step:03d}_zmid.png"
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"      ✅ {png_path.name}")
                
                # 3. V potential - XY slice
                fig, ax = plt.subplots(figsize=(10, 10))
                plot_V_xy_slice(mesh, V, z_mid, ax)
                plt.tight_layout()
                png_path = output_dir / f"V_potential_t{step:03d}_zmid.png"
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"      ✅ {png_path.name}")
                
            except Exception as e:
                print(f"   ⚠️  Error processing timestep {step}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ All visualizations saved to: {output_dir}")

def plot_mesh_outline_xy(mesh, ct, z_slice, ax):
    """Plot mesh outline showing motor geometry."""
    from dolfinx import geometry
    
    # Sample points along circles to show motor geometry
    r_values = [0.017, 0.030, 0.037, 0.040, 0.050, 0.057, 0.075]  # Motor radii
    theta = np.linspace(0, 2*np.pi, 100)
    
    for r in r_values:
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        z_circle = np.full_like(x_circle, z_slice)
        points = np.column_stack([x_circle, y_circle, z_circle])
        
        # Check if points are in mesh
        bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
        cells = geometry.compute_collisions_points(bb_tree, points)
        in_mesh = np.array([len(cells.links(i)) > 0 for i in range(len(points))])
        
        if np.any(in_mesh):
            ax.plot(x_circle[in_mesh], y_circle[in_mesh], 'k-', linewidth=0.5, alpha=0.3)

def plot_A_magnitude_xy_slice(mesh, A, z_slice, ax, ct=None):
    """Plot A field magnitude on XY slice. A is in Lagrange space (3 components)."""
    from dolfinx import geometry
    
    # Create sample points
    x_range = np.linspace(-0.3, 0.3, 150)
    y_range = np.linspace(-0.3, 0.3, 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, z_slice)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate A at points (A is Lagrange vector, 3 components)
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cells = geometry.compute_collisions_points(bb_tree, points)
    cell_candidates = geometry.compute_colliding_cells(mesh, cells, points)
    
    A_values = np.zeros(len(points))
    for i, point in enumerate(points):
        if len(cell_candidates.links(i)) > 0:
            cell = cell_candidates.links(i)[0]
            try:
                A_val = A.eval(point, np.array([cell]))
                if isinstance(A_val, np.ndarray):
                    if A_val.ndim == 1 and len(A_val) == 3:
                        A_mag = np.linalg.norm(A_val)
                    elif A_val.ndim == 2:
                        A_mag = np.linalg.norm(A_val[0])
                    else:
                        A_mag = abs(A_val[0]) if len(A_val) > 0 else 0.0
                else:
                    A_mag = abs(A_val)
                A_values[i] = A_mag
            except Exception as e:
                A_values[i] = 0.0
    
    A_values = A_values.reshape(X.shape)
    
    # Plot with mesh outline
    if ct is not None:
        plot_mesh_outline_xy(mesh, ct, z_slice, ax)
    
    im = ax.contourf(X, Y, A_values, levels=50, cmap='plasma', extend='both', alpha=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'A Field Magnitude (Wb/m) at z={z_slice:.4f} m')
    plt.colorbar(im, ax=ax, label='|A| (Wb/m)')

def plot_B_magnitude_xy_slice(mesh, B_mag, z_slice, ax, ct=None):
    """Plot B magnitude on XY slice."""
    # Create sample points in XY plane at z=z_slice
    x_range = np.linspace(-0.3, 0.3, 100)
    y_range = np.linspace(-0.3, 0.3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, z_slice)
    
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate B_mag at points
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cells = geometry.compute_collisions_points(bb_tree, points)
    cell_candidates = geometry.compute_colliding_cells(mesh, cells, points)
    
    B_values = np.zeros(len(points))
    for i, point in enumerate(points):
        if len(cell_candidates.links(i)) > 0:
            cell = cell_candidates.links(i)[0]
            try:
                val = B_mag.eval(point, np.array([cell]))
                B_values[i] = val[0] if isinstance(val, np.ndarray) else val
            except:
                B_values[i] = 0.0
    
    B_values = B_values.reshape(X.shape)
    
    # Clip outliers for visualization
    B_values = np.clip(B_values, 0.0, 10.0)
    
    # Plot with mesh outline
    if ct is not None:
        plot_mesh_outline_xy(mesh, ct, z_slice, ax)
    
    im = ax.contourf(X, Y, B_values, levels=50, cmap='viridis', extend='both', alpha=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'B Magnitude (T) at z={z_slice:.4f} m')
    plt.colorbar(im, ax=ax, label='|B| (T)')

def plot_B_vector_xy_slice(mesh, B, z_slice, ax, ct=None):
    """Plot B vector field on XY slice."""
    # Create sample points (coarser for vectors)
    x_range = np.linspace(-0.3, 0.3, 30)
    y_range = np.linspace(-0.3, 0.3, 30)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, z_slice)
    
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate B at points
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cells = geometry.compute_collisions_points(bb_tree, points)
    cell_candidates = geometry.compute_colliding_cells(mesh, cells, points)
    
    Bx = np.zeros(len(points))
    By = np.zeros(len(points))
    
    for i, point in enumerate(points):
        if len(cell_candidates.links(i)) > 0:
            cell = cell_candidates.links(i)[0]
            try:
                B_val = B.eval(point, np.array([cell]))
                if isinstance(B_val, np.ndarray):
                    if B_val.ndim == 1:
                        Bx[i], By[i] = B_val[0], B_val[1]
                    else:
                        Bx[i], By[i] = B_val[0, 0], B_val[0, 1]
            except:
                pass
    
    Bx = Bx.reshape(X.shape)
    By = By.reshape(Y.shape)
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # Clip for visualization
    B_mag = np.clip(B_mag, 0.0, 10.0)
    
    # Plot with mesh outline
    if ct is not None:
        plot_mesh_outline_xy(mesh, ct, z_slice, ax)
    
    ax.quiver(X, Y, Bx, By, B_mag, cmap='plasma', scale=1e6, width=0.003, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'B Vector Field (XY components) at z={z_slice:.4f} m')
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)

def plot_V_xy_slice(mesh, V, z_slice, ax, ct=None):
    """Plot V potential on XY slice."""
    # Create sample points
    x_range = np.linspace(-0.3, 0.3, 100)
    y_range = np.linspace(-0.3, 0.3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, z_slice)
    
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate V at points
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cells = geometry.compute_collisions_points(bb_tree, points)
    cell_candidates = geometry.compute_colliding_cells(mesh, cells, points)
    
    V_values = np.zeros(len(points))
    for i, point in enumerate(points):
        if len(cell_candidates.links(i)) > 0:
            cell = cell_candidates.links(i)[0]
            try:
                val = V.eval(point, np.array([cell]))
                V_values[i] = val[0] if isinstance(val, np.ndarray) else val
            except:
                V_values[i] = 0.0
    
    V_values = V_values.reshape(X.shape)
    
    # Plot with mesh outline
    if ct is not None:
        plot_mesh_outline_xy(mesh, ct, z_slice, ax)
    
    im = ax.contourf(X, Y, V_values, levels=50, cmap='coolwarm', extend='both', alpha=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'V Potential (V) at z={z_slice:.4f} m')
    plt.colorbar(im, ax=ax, label='V (V)')

if __name__ == "__main__":
    xdmf_path = Path("../../results/3d/av_solver.xdmf")
    output_dir = Path("../../results/3d/png_images")
    
    if len(sys.argv) > 1:
        xdmf_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    
    visualize_xdmf(xdmf_path, output_dir)
