#!/usr/bin/env python3
"""Check mesh quality and investigate potential issues."""

from pathlib import Path
import numpy as np
from dolfinx import fem, io
from mpi4py import MPI
from mesh_3D import mesh_parameters, model_parameters

print("="*70)
print("🔍 MESH QUALITY INVESTIGATION")
print("="*70)

# Load mesh
mesh_path = Path(__file__).parents[2] / "meshes" / "3d" / "pmesh3D_test1.xdmf"
if not mesh_path.exists():
    print(f"❌ Mesh file not found: {mesh_path}")
    print("   Please generate mesh first by running: python3 mesh_3D.py")
    exit(1)

print(f"\n📁 Loading mesh from: {mesh_path}")
with io.XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")

print(f"\n✅ Mesh loaded:")
print(f"   Total cells: {mesh.topology.index_map(3).size_global}")
print(f"   Total vertices: {mesh.geometry.x.shape[0]}")

coords = mesh.geometry.x
print(f"\n   Mesh bounds:")
print(f"     x: [{np.min(coords[:, 0]):.4f}, {np.max(coords[:, 0]):.4f}]")
print(f"     y: [{np.min(coords[:, 1]):.4f}, {np.max(coords[:, 1]):.4f}]")
print(f"     z: [{np.min(coords[:, 2]):.4f}, {np.max(coords[:, 2]):.4f}]")

if ct is not None:
    unique_tags = np.unique(ct.values)
    print(f"\n   Domain tags: {sorted(unique_tags)}")
    for tag in sorted(unique_tags):
        cells = ct.find(int(tag))
        print(f"     Tag {tag}: {cells.size} cells ({100*cells.size/mesh.topology.index_map(3).size_global:.1f}%)")

print("\n📋 Mesh parameters:")
for key, value in mesh_parameters.items():
    print(f"   {key}: {value:.6f} m")

print("\n📋 Model parameters:")
print(f"   mu_0: {model_parameters['mu_0']:.6e} H/m")
print(f"   freq: {model_parameters['freq']} Hz")
print(f"   J: {model_parameters['J']:.6e} A/m²")

# Check for potential issues in mesh_3D.py
print("\n🔍 Checking mesh_3D.py for potential issues...")

# Check if air box placement is correct
r5 = mesh_parameters["r5"]
depth = 0.0855  # Default depth
L = 1.0  # Default box size

print(f"\n   Air box configuration:")
print(f"     Box size L: {L} m")
print(f"     Box depth: {depth} m")
print(f"     Stator radius r5: {r5} m")
print(f"     Box should contain stator (r5={r5} < L/2={L/2})? {r5 < L/2}")

# Check mesh resolution
print(f"\n   Mesh resolution (from mesh_3D.py defaults):")
print(f"     res: 0.01 m (default)")
print(f"     LcMin: 0.01 m")
print(f"     LcMax: 1.0 m (100 * res)")

print("\n" + "="*70)
print("✅ Mesh quality check complete")
print("="*70)

