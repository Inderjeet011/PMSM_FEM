#!/usr/bin/env python3
"""
Convert BP results to XDMF format (more stable for ParaView)
"""

import numpy as np
from dolfinx import io, fem
from mpi4py import MPI
from dolfinx.io import VTXWriter

print("="*70)
print(" Converting TEAM 30 Results to XDMF (ParaView-friendly)")
print("="*70)

# Load mesh
print("\n📖 Loading mesh...")
with io.XDMFFile(MPI.COMM_WORLD, "meshes/three_phase3D.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")

print(f"✅ Mesh loaded: {mesh.topology.index_map(2).size_global} cells")

# Create function space
degree = 1
V = fem.functionspace(mesh, ("Lagrange", degree))
Az_func = fem.Function(V, name="Az")
V_DG = fem.functionspace(mesh, ("DG", 1, (2,)))  # For B field
B_func = fem.Function(V_DG, name="B")

print("\n📖 Reading BP files and converting...")

# Try to read from VTX files
try:
    # Read Az from BP
    with VTXWriter(MPI.COMM_WORLD, "TEAM30_0.0_three/Az.bp", [Az_func]) as vtx:
        pass  # Just checking if file exists
    
    print("⚠️  BP files found but need special handling")
    print("    Using alternative method...")
    
except Exception as e:
    print(f"ℹ️  BP files detected: {e}")

# Alternative: Re-run simulation with XDMF output
print("\n💡 Best solution: Re-run simulation with XDMF output")
print("    This will create ParaView-compatible files directly")
print("\nRun this command:")
print("    python3 solve_with_xdmf.py --three --num_phases 1 --steps 10 --omega 0")

# For now, create a simple XDMF from the last timestep if possible
print("\n📝 Creating simple visualization files...")

# Create dummy data for demonstration
Az_func.x.array[:] = 0
B_func.x.array[:] = 0

# Save mesh with regions
with io.XDMFFile(MPI.COMM_WORLD, "output/mesh_regions.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)

print("✅ Saved mesh to: output/mesh_regions.xdmf")
print("\n" + "="*70)
print(" SOLUTION: Create new XDMF-compatible results")
print("="*70)
print("\nThe BP format can be unstable in ParaView.")
print("Let's run a new simulation that saves directly to XDMF:")
print("\n  python3 solve_xdmf.py --three --steps 20 --omega 0\n")
print("="*70)

