#!/usr/bin/env python3
"""
Magnetic Flux Density Visualization for TEAM 30
→ Computes B = curl(Az) = (∂Az/∂y, -∂Az/∂x) in 2D
"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from dolfinx import fem, io
import h5py

# ==============================================================
# Load mesh
# ==============================================================
with io.XDMFFile(MPI.COMM_WORLD, "meshes/three_phase3D.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")

tdim = mesh.topology.dim
print(f"✅ Loaded mesh with {mesh.topology.index_map(tdim).size_global} cells")

# ==============================================================
# Helper: load Az
# ==============================================================
def load_Az(filename):
    """Load Az from XDMF file"""
    Az_space = fem.functionspace(mesh, ("Lagrange", 1))
    Az = fem.Function(Az_space)
    
    with h5py.File(filename.replace(".xdmf", ".h5"), "r") as h5:
        # Try to find the data in the HDF5 file
        if "/Function/f_0/0" in h5:
            data = h5["/Function/f_0/0"][:]
        elif "/Function/f_0/0.0000000000000000e+00" in h5:
            data = h5["/Function/f_0/0.0000000000000000e+00"][:]
        elif "/Function/f" in h5:
            data = h5["/Function/f"][:]
        else:
            # List available datasets
            print("Available datasets in HDF5:")
            def print_structure(name, obj):
                print(name)
            h5.visititems(print_structure)
            raise ValueError("Could not find Az data in HDF5 file")
        
        Az.x.array[:] = data.flatten()[:len(Az.x.array)]
    return Az

# ==============================================================
# Compute B field (cell-center averaging)
# ==============================================================
def compute_B(Az):
    """Compute magnetic flux density B = curl(Az) = (∂Az/∂y, -∂Az/∂x)"""
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    x = mesh.geometry.x
    ncell = mesh.topology.index_map(tdim).size_local
    Bx = np.zeros(ncell)
    By = np.zeros(ncell)
    centers = np.zeros((ncell, 2))
    
    for c in range(ncell):
        verts = c2v.links(c)
        xy = x[verts, :2]
        Azv = Az.x.array[verts]
        centers[c] = xy.mean(0)
        
        # Compute gradient using least squares
        A = np.column_stack([xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1]])
        b = Azv - Azv[0]
        grad = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # B = curl(Az) = (∂Az/∂y, -∂Az/∂x)
        Bx[c], By[c] = grad[1], -grad[0]
    
    return centers, np.sqrt(Bx**2 + By**2), Bx, By

# ==============================================================
# Load and compute B
# ==============================================================
try:
    print("\n📖 Loading Az from output/team30_Az.xdmf...")
    Az = load_Az("output/team30_Az.xdmf")
    print(f"✅ Loaded Az with ||Az|| = {np.linalg.norm(Az.x.array):.4e}")
    
    centers, Bmag, Bx, By = compute_B(Az)
    print(f"✅ Computed B field: |B|_max = {np.max(Bmag):.4e} T")
    
    # ==============================================================
    # Plot
    # ==============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Plot |B|
    triang = tri.Triangulation(centers[:, 0], centers[:, 1])
    vmax = max(np.max(Bmag), 1e-6)  # Avoid zero vmax
    
    im1 = ax1.tricontourf(triang, Bmag, levels=100, cmap='jet', vmin=0, vmax=vmax)
    ax1.set_aspect('equal')
    ax1.set_title('|B| (T)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    plt.colorbar(im1, ax=ax1, label='|B| (T)')
    
    # Plot Az
    x_coords = mesh.geometry.x[:, 0]
    y_coords = mesh.geometry.x[:, 1]
    triang2 = tri.Triangulation(x_coords, y_coords)
    
    Az_array = Az.x.array
    vmax_az = max(abs(np.max(Az_array)), abs(np.min(Az_array)), 1e-10)
    
    im2 = ax2.tricontourf(triang2, Az_array, levels=100, cmap='RdBu_r', 
                          vmin=-vmax_az, vmax=vmax_az)
    ax2.set_aspect('equal')
    ax2.set_title('Az (Wb/m)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    plt.colorbar(im2, ax=ax2, label='Az (Wb/m)')
    
    plt.tight_layout()
    plt.savefig("output/team30_B_field.png", dpi=300, bbox_inches='tight')
    print("\n✅ Saved: output/team30_B_field.png")
    plt.show()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

