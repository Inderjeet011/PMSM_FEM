#!/usr/bin/env python3
"""
Step-by-step debugging script for 3D A-V solver.
This will verify each component one at a time.
"""

from setup import SimulationConfig3D, DomainTags3D, load_mesh, maybe_retag_cells
from solver import setup_sources, initialise_magnetisation, update_currents, current_stats
from dolfinx import fem
import numpy as np

print("="*70)
print("STEP 1: Verify Sources (Currents & Magnets)")
print("="*70)

# Load mesh
config = SimulationConfig3D()
print(f"\n[1.1] Loading mesh from: {config.mesh_path}")
mesh, ct, ft = load_mesh(config.mesh_path)
ct = maybe_retag_cells(mesh, ct)

# Setup sources
print(f"\n[1.2] Setting up source fields...")
J_z, M_vec = setup_sources(mesh, ct)
initialise_magnetisation(mesh, ct, M_vec, config)

# Check magnetization
print(f"\n[1.3] Checking magnetization:")
magnet_cells = []
for marker in DomainTags3D.MAGNETS:
    cells = ct.find(marker)
    magnet_cells.extend(cells.tolist())

if len(magnet_cells) > 0:
    M_array = M_vec.x.array.reshape((-1, 3))
    M_magnitudes = np.linalg.norm(M_array[magnet_cells], axis=1)
    print(f"   Found {len(magnet_cells)} magnet cells")
    print(f"   M magnitude range: [{np.min(M_magnitudes):.3e}, {np.max(M_magnitudes):.3e}] A/m")
    print(f"   Expected: ~{config.magnet_remanence/config.mu0:.3e} A/m")
    if np.max(M_magnitudes) < 1e-10:
        print("   ⚠️  PROBLEM: Magnetization is zero!")
    else:
        print("   ✅ Magnetization looks good")
else:
    print("   ⚠️  PROBLEM: No magnet cells found!")

# Check currents at t=0
print(f"\n[1.4] Checking coil currents at t=0:")
update_currents(mesh, ct, J_z, config, t=0.0)
max_J = current_stats(J_z)
print(f"   max|J_z| = {max_J:.3e} A/m²")
print(f"   Expected: ~{config.coil_current_peak:.3e} A/m²")

coil_cells = []
for marker in DomainTags3D.COILS:
    cells = ct.find(marker)
    coil_cells.extend(cells.tolist())

if len(coil_cells) > 0:
    J_values = J_z.x.array[coil_cells]
    print(f"   Found {len(coil_cells)} coil cells")
    print(f"   J_z range in coils: [{np.min(J_values):.3e}, {np.max(J_values):.3e}] A/m²")
    if np.max(np.abs(J_values)) < 1e-10:
        print("   ⚠️  PROBLEM: Currents are zero!")
    else:
        print("   ✅ Currents look good")
else:
    print("   ⚠️  PROBLEM: No coil cells found!")

print("\n" + "="*70)
print("STEP 2: Verify RHS Assembly")
print("="*70)

from setup import setup_materials, setup_boundary_conditions
from solver import build_forms
import ufl

print(f"\n[2.1] Setting up materials...")
sigma, nu, density = setup_materials(mesh, ct, config)

print(f"\n[2.2] Setting up function spaces...")
import basix.ufl
nedelec = basix.ufl.element("N1curl", mesh.basix_cell(), config.degree_A)
lagrange = basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_V)
A_space = fem.functionspace(mesh, nedelec)
V_space = fem.functionspace(mesh, lagrange)
A_prev = fem.Function(A_space, name="A_prev")

print(f"\n[2.3] Setting up boundary conditions...")
bc_A, bc_V, block_bcs = setup_boundary_conditions(mesh, ft, A_space, V_space)

print(f"\n[2.4] Setting up measures...")
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dx_conductors = dx(DomainTags3D.conducting()[0])
for marker in DomainTags3D.conducting()[1:]:
    dx_conductors += dx(marker)
dx_magnets = dx(DomainTags3D.MAGNETS[0])
for marker in DomainTags3D.MAGNETS[1:]:
    dx_magnets += dx(marker)

print(f"\n[2.5] Building forms...")
a_blocks, L_blocks = build_forms(
    mesh, A_space, V_space, sigma, nu, J_z, M_vec, A_prev,
    dx, dx_conductors, dx_magnets, config
)

print(f"\n[2.6] Assembling RHS vector...")
from dolfinx.fem import petsc
from petsc4py import PETSc

vec = petsc.create_vector(L_blocks[0])
petsc.assemble_vector(vec, L_blocks[0])
vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

rhs_norm = vec.norm()
rhs_array = vec.array_r
non_zero_dofs = np.nonzero(np.abs(rhs_array) > 1e-10)[0]

print(f"   ||RHS|| = {rhs_norm:.6e}")
print(f"   Non-zero DOFs: {len(non_zero_dofs)}")
print(f"   Max |RHS| = {np.max(np.abs(rhs_array)):.6e}")

if rhs_norm < 1e-10:
    print("   ⚠️  PROBLEM: RHS is essentially zero!")
    print("   This means sources are not contributing to the RHS")
else:
    print("   ✅ RHS is non-zero")

# Check boundary DOFs
bdofs = bc_A.dof_indices()[0] if bc_A else []
if len(bdofs) > 0:
    rhs_boundary = rhs_array[bdofs]
    rhs_interior = np.delete(rhs_array, bdofs)
    boundary_norm = np.linalg.norm(rhs_boundary)
    interior_norm = np.linalg.norm(rhs_interior)
    print(f"\n   RHS breakdown:")
    print(f"   ||RHS_boundary|| = {boundary_norm:.6e}")
    print(f"   ||RHS_interior|| = {interior_norm:.6e}")
    print(f"   Boundary % = {100*boundary_norm/max(rhs_norm,1e-20):.1f}%")

vec.destroy()

print("\n" + "="*70)
print("Summary")
print("="*70)
print("If sources are non-zero but RHS is zero, check form building.")
print("If RHS is non-zero, proceed to check matrix and solve.")

