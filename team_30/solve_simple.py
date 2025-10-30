#!/usr/bin/env python3
"""
Simplified 2D AV formulation solver for TEAM 30
Using direct solver on mixed function space
"""

from mpi4py import MPI
import numpy as np
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from ufl import Measure, SpatialCoordinate, TestFunction, TrialFunction, curl, div, grad, inner

from mesh import domain_parameters, model_parameters

print("="*70)
print(" TEAM 30 - 2D A-V Mixed Formulation")
print("="*70)

# -- Parameters -- #
freq = model_parameters["freq"]
mu_0 = model_parameters["mu_0"]
omega_J = 2 * np.pi * freq

single_phase = False
mesh_dir = "meshes"
ext = "single" if single_phase else "three"
fname = f"{mesh_dir}/{ext}_phase3D"

domains, currents = domain_parameters(single_phase)
degree = 1

# -- Load Mesh -- #
print("\n📖 Loading mesh...")
with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")

print(f"✅ {ct.values.size} cells, dim={tdim}")

# -- Function Spaces -- #
# Use simple CG elements for 2D (Az is scalar in 2D)
import basix.ufl
Az_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree)
V_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree)
W = fem.functionspace(mesh, basix.ufl.mixed_element([Az_el, V_el]))

print(f"✅ Mixed space with {W.dofmap.index_map.size_global * W.dofmap.index_map_bs} DOFs")

# -- Material properties -- #
DG0 = fem.functionspace(mesh, ("DG", 0))
mu_R = fem.Function(DG0)
sigma = fem.Function(DG0)

for material, domain in domains.items():
    for marker in domain:
        cells = ct.find(marker)
        mu_R.x.array[cells] = model_parameters["mu_r"][material]
        sigma.x.array[cells] = model_parameters["sigma"][material]

# Define regions
Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

dx = Measure("dx", domain=mesh, subdomain_data=ct)

# -- Weak Form -- #
(Az, V_pot) = TrialFunction(W)
(v, q) = TestFunction(W)

# Bilinear form
a = (1 / (mu_0 * mu_R)) * inner(grad(Az), grad(v)) * dx
a += sigma * mu_0 * inner(grad(V_pot), grad(v)) * dx(Omega_c + Omega_n)
a += sigma * mu_0 * inner(grad(Az), grad(q)) * dx(Omega_c + Omega_n)
a += sigma * mu_0 * inner(grad(V_pot), grad(q)) * dx(Omega_c + Omega_n)

# Linear form (no source for now since we don't have proper 3D structure)
from petsc4py import PETSc
L = fem.Constant(mesh, PETSc.ScalarType(0)) * v * dx

# -- Boundary Conditions -- #
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary

def boundary_marker(x):
    return np.full(x.shape[1], True)

mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)

# Get sub-spaces for BCs
W0, _ = W.sub(0).collapse()
W1, _ = W.sub(1).collapse()

# BC on Az
bdofs_Az = locate_dofs_topological((W.sub(0), W0), entity_dim=tdim - 1, entities=boundary_facets)
zero_Az = fem.Function(W0)
zero_Az.x.array[:] = 0
bc_Az = fem.dirichletbc(zero_Az, bdofs_Az, W.sub(0))

# BC on V
bdofs_V = locate_dofs_topological((W.sub(1), W1), entity_dim=tdim - 1, entities=boundary_facets)
zero_V = fem.Function(W1)
zero_V.x.array[:] = 0
bc_V = fem.dirichletbc(zero_V, bdofs_V, W.sub(1))

bcs = [bc_Az, bc_V]

# -- Solve -- #
print("\n🔧 Solving linear system...")
problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options={
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="team30"
)

w_sol = problem.solve()
Az_sol, V_sol = w_sol.split()

print(f"✅ ||Az|| = {np.linalg.norm(Az_sol.x.array):.4e}")
print(f"✅ ||V||  = {np.linalg.norm(V_sol.x.array):.4e}")

# -- Save for Visualization -- #
print("\n💾 Saving results...")
io.XDMFFile(mesh.comm, "output/team30_Az.xdmf", "w").write_mesh(mesh)
with io.XDMFFile(mesh.comm, "output/team30_Az.xdmf", "a") as xdmf:
    xdmf.write_function(Az_sol, 0.0)

with io.XDMFFile(mesh.comm, "output/team30_V.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(V_sol, 0.0)

print("✅ Done! Results saved to output/team30_Az.xdmf and output/team30_V.xdmf")
print("="*70)

