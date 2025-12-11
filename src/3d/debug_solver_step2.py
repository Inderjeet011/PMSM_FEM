#!/usr/bin/env python3
"""
Step 3: Verify Matrix and Solve
"""

from setup import SimulationConfig3D, DomainTags3D, load_mesh, maybe_retag_cells, setup_materials, setup_boundary_conditions
from solver import setup_sources, initialise_magnetisation, update_currents, build_forms, assemble_system_matrix, configure_solver
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import basix.ufl
import numpy as np
import ufl

print("="*70)
print("STEP 3: Verify Matrix and Solve")
print("="*70)

config = SimulationConfig3D()
mesh, ct, ft = load_mesh(config.mesh_path)
ct = maybe_retag_cells(mesh, ct)

# Setup everything
sigma, nu, density = setup_materials(mesh, ct, config)
J_z, M_vec = setup_sources(mesh, ct)
initialise_magnetisation(mesh, ct, M_vec, config)
update_currents(mesh, ct, J_z, config, t=0.0)

nedelec = basix.ufl.element("N1curl", mesh.basix_cell(), config.degree_A)
lagrange = basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_V)
A_space = fem.functionspace(mesh, nedelec)
V_space = fem.functionspace(mesh, lagrange)
A_prev = fem.Function(A_space, name="A_prev")

bc_A, bc_V, block_bcs = setup_boundary_conditions(mesh, ft, A_space, V_space)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dx_conductors = dx(DomainTags3D.conducting()[0])
for marker in DomainTags3D.conducting()[1:]:
    dx_conductors += dx(marker)
dx_magnets = dx(DomainTags3D.MAGNETS[0])
for marker in DomainTags3D.MAGNETS[1:]:
    dx_magnets += dx(marker)

a_blocks, L_blocks = build_forms(
    mesh, A_space, V_space, sigma, nu, J_z, M_vec, A_prev,
    dx, dx_conductors, dx_magnets, config
)

print(f"\n[3.1] Assembling system matrix...")
mat_blocks, mat_nest, A00_standalone = assemble_system_matrix(mesh, a_blocks, block_bcs)

print(f"\n[3.2] Assembling RHS...")
rhs = petsc.create_vector(L_blocks[0])
petsc.assemble_vector(rhs, L_blocks[0])
rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

rhs_norm_before = rhs.norm()
print(f"   ||RHS|| before BCs: {rhs_norm_before:.6e}")

# DO NOT apply BCs to RHS for Nédélec elements (A block)
# The matrix was assembled with BCs, so it will enforce zero solution
# Zeroing RHS boundary DOFs would lose all contributions
print(f"   ||RHS|| (preserved): {rhs_norm_before:.6e}")
print(f"   ✅ RHS preserved - matrix will enforce BCs")

print(f"\n[3.3] Configuring solver...")
ksp = configure_solver(mesh, mat_nest, mat_blocks)

print(f"\n[3.4] Solving...")
# Create solution vector matching the nested structure
sol_A = petsc.create_vector(L_blocks[0])
sol_V = petsc.create_vector(L_blocks[1])
sol = PETSc.Vec().createNest([sol_A, sol_V], comm=mesh.comm)

# Create nested RHS
rhs_V = petsc.create_vector(L_blocks[1])
petsc.assemble_vector(rhs_V, L_blocks[1])
rhs_V.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# Apply BCs to V block (Lagrange - safe)
petsc.set_bc(rhs_V, block_bcs[1])
rhs_nest = PETSc.Vec().createNest([rhs, rhs_V], comm=mesh.comm)

ksp.solve(rhs_nest, sol)

iterations = ksp.getIterationNumber()
residual = ksp.getResidualNorm()
print(f"   Iterations: {iterations}")
print(f"   Residual: {residual:.2e}")

A_sol = fem.Function(A_space, name="A")
sol_A_block = sol.getNestSubVecs()[0]
with sol_A_block.localForm() as src:
    A_sol.x.array[:] = src.array_r[:A_sol.x.array.size]
A_sol.x.scatter_forward()

A_norm = np.linalg.norm(A_sol.x.array)
A_max = np.max(np.abs(A_sol.x.array))
print(f"\n[3.5] Solution check:")
print(f"   ||A|| = {A_norm:.6e} Wb/m")
print(f"   max|A| = {A_max:.6e} Wb/m")

if A_norm < 1e-10:
    print("   ⚠️  PROBLEM: A field is zero!")
    print("   Root cause: RHS was zeroed out by boundary conditions")
else:
    print("   ✅ A field is non-zero")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("The issue is that 100% of RHS contributions are in boundary DOFs.")
print("When we apply zero BCs, we zero out the boundary DOFs, losing all RHS.")
print("\nSOLUTION: Need to redistribute boundary contributions to interior DOFs")
print("before applying BCs, or use a different BC application strategy.")

