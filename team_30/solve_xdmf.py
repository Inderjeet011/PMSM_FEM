#!/usr/bin/env python3
"""
TEAM 30 Solver - Modified to save XDMF output (ParaView-friendly)
Based on solve.py but with XDMF output instead of BP
"""

import argparse
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.fem.petsc as _petsc
from dolfinx import default_scalar_type, fem, io
import basix.ufl
import ufl
import tqdm

from mesh import domain_parameters, model_parameters, surface_map
from util import DerivedQuantities2D, update_current_density

print("="*70)
print(" TEAM 30 Solver - XDMF Output (ParaView Compatible)")
print("="*70)

# Parse arguments
parser = argparse.ArgumentParser(description="TEAM 30 - XDMF output")
parser.add_argument("--three", action="store_true", default=False)
parser.add_argument("--num_phases", type=int, default=1)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--omega", type=float, default=0.0)
args = parser.parse_args()

single_phase = not args.three
num_phases = args.num_phases
steps_per_phase = args.steps
omega_u = args.omega
degree = 1

# Parameters
freq = model_parameters["freq"]
T = num_phases * 1 / freq
dt_ = 1 / steps_per_phase * 1 / freq
mu_0 = model_parameters["mu_0"]
omega_J = 2 * np.pi * freq

ext = "single" if single_phase else "three"
fname = Path("meshes") / f"{ext}_phase"
domains, currents = domain_parameters(single_phase)

# Load mesh
print(f"\n📖 Loading mesh: {fname}.xdmf")
with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")

print(f"✅ Mesh: {mesh.topology.index_map(tdim).size_global} cells")

# Material properties
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

# Create submesh for conductive domain
sub_cells = np.hstack([ct.find(tag) for tag in Omega_c])
sort_order = np.argsort(sub_cells)
conductive_domain, entity_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, sub_cells[sort_order]
)

# Function spaces
cell = mesh.ufl_cell()
FE = basix.ufl.element("Lagrange", str(cell), degree)
V = dolfinx.fem.functionspace(mesh, FE)
Q = dolfinx.fem.functionspace(conductive_domain, FE)

print(f"✅ DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs} (V) + {Q.dofmap.index_map.size_global * Q.dofmap.index_map_bs} (Q)")

# Trial/test functions
Az = ufl.TrialFunction(V)
vz = ufl.TestFunction(V)
v = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)
Azn = dolfinx.fem.Function(V)
J0z = fem.Function(DG0)

# Measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=surface_map["Exterior"])
n = ufl.FacetNormal(mesh)
dt = fem.Constant(mesh, dt_)
x = ufl.SpatialCoordinate(mesh)
omega = fem.Constant(mesh, default_scalar_type(omega_u))

# Weak forms
a_00 = dt / mu_R * ufl.inner(ufl.grad(Az), ufl.grad(vz)) * dx(Omega_n + Omega_c)
a_00 += dt / mu_R * vz * (n[0] * Az.dx(0) - n[1] * Az.dx(1)) * ds
a_00 += mu_0 * sigma * Az * vz * dx(Omega_c)
u = omega * ufl.as_vector((-x[1], x[0]))
a_00 += dt * mu_0 * sigma * ufl.dot(u, ufl.grad(Az)) * vz * dx(Omega_c)
a_11 = dt * mu_0 * sigma * (v.dx(0) * q.dx(0) + v.dx(1) * q.dx(1)) * dx(Omega_c)

L_0 = mu_0 * sigma * Azn * vz * dx(Omega_c)
L_0 += dt * mu_0 * J0z * vz * dx(Omega_n)

L = [
    dolfinx.fem.form(L_0),
    fem.form(fem.Constant(conductive_domain, default_scalar_type(0)) * q * ufl.dx(domain=conductive_domain)),
]

# Boundary conditions
mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
zeroV = fem.Function(V)
zeroV.x.array[:] = 0
bc_V = fem.dirichletbc(zeroV, bndry_dofs)

conductive_domain.topology.create_connectivity(conductive_domain.topology.dim - 1, conductive_domain.topology.dim)
conductive_domain_facets = dolfinx.mesh.exterior_facet_indices(conductive_domain.topology)
q_boundary = fem.locate_dofs_topological(Q, tdim - 1, conductive_domain_facets)
zeroQ = fem.Function(Q)
bc_p = fem.dirichletbc(zeroQ, q_boundary)
bcs = [bc_V, bc_p]

# Assemble system
a = [
    [dolfinx.fem.form(a_00), None],
    [None, dolfinx.fem.form(a_11, entity_maps=[entity_map])],
]
A = fem.petsc.assemble_matrix(a, bcs=bcs)
A.assemble()
b = fem.petsc.create_vector(L)
fem.petsc.assemble_vector(b, L)

bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs)
fem.petsc.apply_lifting(b, a, bcs=bcs0)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, bcs0)

# Solver
solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setTolerances(atol=1e-9, rtol=1e-9)
solver.setType("gmres")
solver.getPC().setType("lu")
solver.getPC().setFactorSolverType("mumps")

# Output functions
solution_vector = A.createVecLeft()
Az_out = dolfinx.fem.Function(V, name="Az")
V_out = dolfinx.fem.Function(Q, name="V")

# Create output directory
outdir = Path("XDMF_results")
outdir.mkdir(exist_ok=True)

# XDMF output files (ParaView compatible!)
print(f"\n💾 Saving to: {outdir}/")
Az_file = io.XDMFFile(mesh.comm, str(outdir / "Az.xdmf"), "w")
Az_file.write_mesh(mesh)

print("\n🔧 Starting time-stepping...")
num_steps = int(T / float(dt.value))

t = 0.0
update_current_density(J0z, omega_J, t, ct, currents)

if MPI.COMM_WORLD.rank == 0:
    progressbar = tqdm.tqdm(desc="Solving", total=num_steps)

for i in range(num_steps):
    if MPI.COMM_WORLD.rank == 0:
        progressbar.update(1)
    
    t += float(dt.value)
    update_current_density(J0z, omega_J, t, ct, currents)
    
    with b.localForm() as loc_b:
        loc_b.set(0)
    _petsc.assemble_vector(b, L)
    _petsc.apply_lifting(b, a, bcs=bcs0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    _petsc.set_bc(b, bcs0)
    
    solver.solve(b, solution_vector)
    
    offset_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    Az_out.x.array[:offset_V] = solution_vector.array_r[:offset_V]
    Az_out.x.scatter_forward()
    
    # Save to XDMF (ParaView can read this!)
    Az_file.write_function(Az_out, t)
    
    Azn.x.array[:offset_V] = solution_vector.array_r[:offset_V]
    Azn.x.scatter_forward()

if MPI.COMM_WORLD.rank == 0:
    progressbar.close()

Az_file.close()
b.destroy()

print("\n✅ Done!")
print("="*70)
print(f" Results saved to: {outdir}/Az.xdmf")
print("="*70)
print("\n📋 To visualize in ParaView:")
print("  1. Copy XDMF_results/ folder from Docker to host")
print("  2. Open ParaView")
print("  3. File → Open → Az.xdmf")
print("  4. Click Apply")
print("  5. Select 'Az' from coloring dropdown")
print("  6. Click Play to animate")
print("="*70)

