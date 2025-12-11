#!/usr/bin/env python3
"""Check mesh quality and investigate potential issues."""

import sys
from pathlib import Path

# Add current directory to path  
sys.path.insert(0, str(Path(__file__).parent))

from dolfinx import geometry
import numpy as np

# Direct import - the module name has numbers but Python can handle it
# We'll import it as a module
import importlib
import os

# Change to the directory and import
os.chdir(Path(__file__).parent)
spec = importlib.util.spec_from_file_location("solver_3d", "3d_av_solver.py")
import importlib.util
solver_module = importlib.util.module_from_spec(spec)
sys.modules["solver_3d"] = solver_module  # Register it
spec.loader.exec_module(solver_module)

MaxwellAVSolver3D = solver_module.MaxwellAVSolver3D

print("="*70)
print("🔍 MESH QUALITY INVESTIGATION")
print("="*70)

# Create solver and setup
solver = MaxwellAVSolver3D()
print("\n1. Setting up solver...")
solver.setup()

print("\n2. Checking mesh quality near boundary...")
solver._check_mesh_quality_near_boundary()

print("\n3. Basic mesh statistics...")
mesh = solver.mesh
print(f"   Total cells: {mesh.topology.index_map(3).size_global}")
print(f"   Total vertices: {mesh.geometry.x.shape[0]}")
print(f"   Mesh bounds:")
coords = mesh.geometry.x
print(f"     x: [{np.min(coords[:, 0]):.4f}, {np.max(coords[:, 0]):.4f}]")
print(f"     y: [{np.min(coords[:, 1]):.4f}, {np.max(coords[:, 1]):.4f}]")
print(f"     z: [{np.min(coords[:, 2]):.4f}, {np.max(coords[:, 2]):.4f}]")

print("\n4. Checking domain tags...")
if solver.ct is not None:
    unique_tags = np.unique(solver.ct.values)
    print(f"   Unique domain tags: {sorted(unique_tags)}")
    for tag in sorted(unique_tags):
        cells = solver.ct.find(int(tag))
        print(f"     Tag {tag}: {cells.size} cells")

print("\n5. Checking for potential mesh issues...")
tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local

# Check cell aspect ratios (rough estimate)
aspect_ratios = []
small_cells = 0
for i in range(min(100, num_cells)):  # Sample first 100 cells
    cell_entities = mesh.topology.connectivity(tdim, 0)
    if i < len(cell_entities.array):
        vertices = cell_entities.links(i)
        cell_coords = coords[vertices]
        ranges = np.max(cell_coords, axis=0) - np.min(cell_coords, axis=0)
        max_range = np.max(ranges)
        min_range = np.min(ranges[ranges > 1e-12])
        if min_range > 0:
            aspect = max_range / min_range
            aspect_ratios.append(aspect)
        if max_range < 1e-6:
            small_cells += 1

if aspect_ratios:
    print(f"   Cell aspect ratio (sample):")
    print(f"     Max: {np.max(aspect_ratios):.2f}")
    print(f"     Mean: {np.mean(aspect_ratios):.2f}")
    print(f"     Median: {np.median(aspect_ratios):.2f}")

print(f"\n   Very small cells (< 1e-6 m): {small_cells} in sample")

print("\n6. Checking mesh_3D.py parameters...")
from mesh_3D import mesh_parameters, model_parameters
print(f"   Mesh parameters:")
for key, value in mesh_parameters.items():
    print(f"     {key}: {value:.6f} m")

print(f"\n   Model parameters:")
print(f"     mu_0: {model_parameters['mu_0']:.6e} H/m")
print(f"     freq: {model_parameters['freq']} Hz")
print(f"     J: {model_parameters['J']:.6e} A/m²")

print("\n" + "="*70)
print("✅ Mesh quality check complete")
print("="*70)

