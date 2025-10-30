#!/usr/bin/env python3
"""
Compute magnetic flux density B from Az and save to XDMF
B = curl(Az) = (dAz/dy, -dAz/dx) in 2D
"""

import numpy as np
from dolfinx import io, fem
from mpi4py import MPI
import h5py
from pathlib import Path

print("="*70)
print(" Computing Magnetic Flux Density B = curl(Az)")
print("="*70)

# Load mesh
print("\n📖 Loading mesh...")
with io.XDMFFile(MPI.COMM_WORLD, "meshes/three_phase.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()

print(f"✅ Mesh: {mesh.topology.index_map(2).size_global} cells")

# Create function spaces
degree = 1
V_scalar = fem.functionspace(mesh, ("Lagrange", degree))
V_vector = fem.functionspace(mesh, ("DG", 1, (2,)))  # 2D vector field

Az = fem.Function(V_scalar, name="Az")
B = fem.Function(V_vector, name="B")

# Open XDMF file to read Az
print("\n📖 Reading Az from XDMF_results/Az.xdmf...")
with io.XDMFFile(MPI.COMM_WORLD, "XDMF_results/Az.xdmf", "r") as xdmf_in:
    # Get time values from HDF5
    with h5py.File("XDMF_results/Az.h5", "r") as h5:
        times = sorted([float(k) for k in h5["/Function/Az"].keys()])
    
    print(f"✅ Found {len(times)} timesteps")
    
    # Create output file for B
    outdir = Path("XDMF_results")
    with io.XDMFFile(mesh.comm, str(outdir / "B.xdmf"), "w") as xdmf_out:
        xdmf_out.write_mesh(mesh)
        
        print("\n🔧 Computing B = curl(Az) for each timestep...")
        for i, t in enumerate(times):
            # Read Az at this time
            Az.x.array[:] = 0
            xdmf_in.read_function(Az, str(t))
            
            # Compute B = curl(Az) = (dAz/dy, -dAz/dx)
            # This is done cell-wise
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim, 0)
            c2v = mesh.topology.connectivity(tdim, 0)
            x = mesh.geometry.x
            ncell = mesh.topology.index_map(tdim).size_local
            
            B_array = B.x.array.reshape((ncell, 2))
            
            for c in range(ncell):
                verts = c2v.links(c)
                xy = x[verts, :2]
                Azv = Az.x.array[verts]
                
                # Compute gradient using least squares
                A_mat = np.column_stack([xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1]])
                b_vec = Azv - Azv[0]
                
                if np.linalg.matrix_rank(A_mat) == 2:
                    grad = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    # B = curl(Az) = (dAz/dy, -dAz/dx)
                    B_array[c, 0] = grad[1]   # Bx = dAz/dy
                    B_array[c, 1] = -grad[0]  # By = -dAz/dx
                else:
                    B_array[c, :] = 0
            
            B.x.scatter_forward()
            
            # Save to XDMF
            xdmf_out.write_function(B, t)
            
            if i % 5 == 0 or i == len(times) - 1:
                B_mag = np.sqrt(B_array[:, 0]**2 + B_array[:, 1]**2)
                print(f"  Step {i+1}/{len(times)}: t={t:.6f}s, |B|_max = {np.max(B_mag):.4e} T")

print("\n✅ Done!")
print("="*70)
print(f" Magnetic flux density saved to: XDMF_results/B.xdmf")
print("="*70)
print("\n📋 To visualize B in ParaView:")
print("  1. Copy XDMF_results/B.xdmf to host (docker cp ...)")
print("  2. Open B.xdmf in ParaView")
print("  3. Select 'B' or 'B_0', 'B_1' from coloring")
print("  4. Use Filters → Calculator to compute magnitude:")
print("     Formula: sqrt(B_0^2 + B_1^2)")
print("="*70)

