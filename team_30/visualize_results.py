#!/usr/bin/env python3
"""
Visualize TEAM 30 simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from dolfinx import io, fem
from mpi4py import MPI

# Load mesh
with io.XDMFFile(MPI.COMM_WORLD, "meshes/three_phase3D.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()

print("📖 Loading results from TEAM30_0.0_three/...")

# Create function space to read into
degree = 1
Az_space = fem.functionspace(mesh, ("Lagrange", degree))
Az = fem.Function(Az_space)

# Read Az from BP file using adios2
import adios2

with adios2.open("TEAM30_0.0_three/Az.bp", "r", MPI.COMM_WORLD) as fh:
    # Read the last step
    for step in fh:
        Az_data = step.read("Az")
        Az.x.array[:len(Az_data)] = Az_data

print(f"✅ Loaded Az: ||Az|| = {np.linalg.norm(Az.x.array):.4e}")

# Compute B field from Az
def compute_B(Az):
    """Compute B = curl(Az) = (∂Az/∂y, -∂Az/∂x)"""
    mesh = Az.function_space.mesh
    tdim = mesh.topology.dim
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
        
        # Gradient using least squares
        A_mat = np.column_stack([xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1]])
        b_vec = Azv - Azv[0]
        grad = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        
        # B = curl(Az) = (∂Az/∂y, -∂Az/∂x)
        Bx[c], By[c] = grad[1], -grad[0]
    
    return centers, np.sqrt(Bx**2 + By**2), Bx, By

centers, Bmag, Bx, By = compute_B(Az)
print(f"✅ Computed B: |B|_max = {np.max(Bmag):.4e} T")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# Plot |B|
triang_cells = tri.Triangulation(centers[:, 0], centers[:, 1])
vmax = max(np.max(Bmag), 1e-6)

im1 = ax1.tricontourf(triang_cells, Bmag, levels=100, cmap='jet', vmin=0, vmax=vmax)
ax1.set_aspect('equal')
ax1.set_title('Magnetic Flux Density |B| (T)', fontsize=14, fontweight='bold')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('|B| (T)', fontsize=11)

# Plot Az
x_coords = mesh.geometry.x[:, 0]
y_coords = mesh.geometry.x[:, 1]
triang_nodes = tri.Triangulation(x_coords, y_coords)

Az_array = Az.x.array
vmax_az = max(abs(np.max(Az_array)), abs(np.min(Az_array)), 1e-10)

im2 = ax2.tricontourf(triang_nodes, Az_array, levels=100, cmap='RdBu_r', 
                       vmin=-vmax_az, vmax=vmax_az)
ax2.set_aspect('equal')
ax2.set_title('Magnetic Vector Potential Az (Wb/m)', fontsize=14, fontweight='bold')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Az (Wb/m)', fontsize=11)

plt.suptitle('TEAM 30 - Three-Phase Induction Motor', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("TEAM30_magnetic_field.png", dpi=300, bbox_inches='tight')
print("\n✅ Saved: TEAM30_magnetic_field.png")
plt.show()

