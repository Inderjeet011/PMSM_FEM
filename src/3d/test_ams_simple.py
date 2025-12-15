#!/usr/bin/env python3
"""Simple test case to verify AMS setup works."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.cpp.fem.petsc import discrete_gradient
from petsc4py import PETSc
import ufl
import basix.ufl
from mpi4py import MPI
import time

# Create a simple 3D unit cube mesh
from dolfinx import mesh
mesh = mesh.create_box(MPI.COMM_WORLD, [[0, 0, 0], [1, 1, 1]], [4, 4, 4], cell_type=mesh.CellType.tetrahedron)

print(f"[TEST] Mesh created: {mesh.topology.index_map(3).size_global} cells")

# Create function spaces
nedelec = basix.ufl.element("N1curl", mesh.basix_cell(), 1)
lagrange = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
A_space = fem.functionspace(mesh, nedelec)
V_space_ams = fem.functionspace(mesh, ("Lagrange", 1))

print(f"[TEST] A_space DOFs: {A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs}")
print(f"[TEST] V_space_ams DOFs: {V_space_ams.dofmap.index_map.size_global}")

# Build discrete gradient
print("[TEST] Building discrete gradient...")
t0 = time.time()
G = discrete_gradient(V_space_ams._cpp_object, A_space._cpp_object)
G.assemble()
t1 = time.time()
print(f"[TEST] Discrete gradient built in {t1-t0:.3f} seconds")

# Build coordinate vectors
print("[TEST] Building coordinate vectors...")
t0 = time.time()
vertex_coord_vecs = []
xcoord = ufl.SpatialCoordinate(mesh)

for dim in range(3):
    coord_func = fem.Function(V_space_ams)
    coord_expr = fem.Expression(xcoord[dim], V_space_ams.element.interpolation_points)
    coord_func.interpolate(coord_expr)
    coord_func.x.scatter_forward()
    vertex_coord_vecs.append(coord_func.x.petsc_vec)
    print(f"[TEST] Coordinate vector {dim}: size={coord_func.x.petsc_vec.getSize()}")
t1 = time.time()
print(f"[TEST] Coordinate vectors built in {t1-t0:.3f} seconds")

# Create a simple SPD matrix (curl-curl + mass)
print("[TEST] Creating test matrix...")
A = ufl.TrialFunction(A_space)
v = ufl.TestFunction(A_space)
dx = ufl.dx(domain=mesh)

# Simple curl-curl + mass form
a = ufl.inner(ufl.curl(A), ufl.curl(v)) * dx + 1e-6 * ufl.inner(A, v) * dx
a_form = fem.form(a)

A00_spd = petsc.assemble_matrix(a_form, bcs=None)
A00_spd.assemble()
A00_spd.setOption(PETSc.Mat.Option.SPD, True)
print(f"[TEST] Test matrix created: size={A00_spd.getSize()}, norm={A00_spd.norm(PETSc.NormType.NORM_FROBENIUS):.6e}")

# Create KSP and configure AMS
print("[TEST] Configuring AMS...")
t0 = time.time()
ksp = PETSc.KSP().create(comm=mesh.comm)
ksp.setOperators(A00_spd, A00_spd)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

# Set discrete gradient
print("[TEST] Setting discrete gradient...")
pc.setHYPREDiscreteGradient(G)

# Compute edge constants
print("[TEST] Computing edge constant vectors...")
edge_const_vecs = []
A_space_map = A_space.dofmap.index_map
bs = A_space.dofmap.index_map_bs
edge_size_local = A_space_map.size_local * bs
edge_size_global = A_space_map.size_global * bs

for dim in range(3):
    edge_vec = PETSc.Vec().create(comm=mesh.comm)
    edge_vec.setSizes((edge_size_local, edge_size_global))
    edge_vec.setUp()
    G.mult(vertex_coord_vecs[dim], edge_vec)
    edge_const_vecs.append(edge_vec)
    print(f"[TEST] Edge constant vector {dim}: size={edge_vec.getSize()}")

# Set edge constant vectors
pc.setHYPRESetEdgeConstantVectors(edge_const_vecs[0], edge_const_vecs[1], edge_const_vecs[2])
print("[TEST] Edge constant vectors set")

# Set projection frequency
PETSc.Options().setValue("-pc_hypre_ams_project_frequency", "1")

# Call setFromOptions
print("[TEST] Calling setFromOptions()...")
t1 = time.time()
pc.setFromOptions()
t2 = time.time()
print(f"[TEST] setFromOptions() completed in {t2-t1:.3f} seconds")

# Try to set up the preconditioner
print("[TEST] Attempting pc.setUp()...")
t0 = time.time()
try:
    pc.setUp()
    t1 = time.time()
    print(f"[TEST] ✅ SUCCESS: pc.setUp() completed in {t1-t0:.3f} seconds")
    print("[TEST] ✅ AMS setup verified successfully!")
except Exception as e:
    t1 = time.time()
    print(f"[TEST] ❌ FAILED: pc.setUp() failed after {t1-t0:.3f} seconds")
    print(f"[TEST] Error: {e}")
    import traceback
    traceback.print_exc()

print("[TEST] Test completed")

