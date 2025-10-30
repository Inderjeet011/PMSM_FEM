# Copyright (C) 2021-2022 Jørgen S. Dokken and Igor A. Baratta
#
# SPDX-License-Identifier:    MIT

import argparse
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Callable, Optional, TextIO, Union

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx.fem.petsc as _petsc
import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import ufl
from dolfinx import default_scalar_type, fem, io
from dolfinx.io import VTXWriter

from mesh import domain_parameters, model_parameters, surface_map
from util import DerivedQuantities2D, MagneticField2D, update_current_density


def solve_team30(
    single_phase: bool,
    num_phases: int,
    omega_u: float,
    degree: int,
    form_compiler_options: dict = {},
    jit_parameters: dict = {},
    apply_torque: bool = False,
    T_ext: Callable[[float], float] = lambda t: 0,
    outdir: Path = Path("results"),
    steps_per_phase: int = 100,
    outfile: Optional[Union[TextIOWrapper, TextIO]] = sys.stdout,
    plot: bool = False,
    progress: bool = False,
    mesh_dir: Path = Path("meshes"),
    save_output: bool = False,
):
    """
    Solve the TEAM 30 problem for a single or three phase engine.
    """

    freq = model_parameters["freq"]
    T = num_phases * 1 / freq
    dt_ = 1 / steps_per_phase * 1 / freq
    mu_0 = model_parameters["mu_0"]
    omega_J = 2 * np.pi * freq

    ext = "single" if single_phase else "three"
    fname = mesh_dir / f"{ext}_phase"

    domains, currents = domain_parameters(single_phase)

    # Read mesh and cell markers
    with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        ct = xdmf.read_meshtags(mesh, name="Cell_markers")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        ft = xdmf.read_meshtags(mesh, name="Facet_markers")

    # Create DG0 function spaces
    DG0 = fem.functionspace(mesh, ("DG", 0))
    mu_R = fem.Function(DG0)
    sigma = fem.Function(DG0)
    density = fem.Function(DG0)

    for material, domain in domains.items():
        for marker in domain:
            cells = ct.find(marker)
            mu_R.x.array[cells] = model_parameters["mu_r"][material]
            sigma.x.array[cells] = model_parameters["sigma"][material]
            density.x.array[cells] = model_parameters["densities"][material]

    # Split domain into conductive and non-conductive parts
    Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
    Omega_c = domains["Rotor"] + domains["Al"]

    sub_cells = np.hstack([ct.find(tag) for tag in Omega_c])
    sort_order = np.argsort(sub_cells)
    conductive_domain, entity_map, _, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, sub_cells[sort_order]
    )

    # Create parent-to-sub mapping
    parent_cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = parent_cell_map.size_local + parent_cell_map.num_ghosts
    parent_to_sub = np.full(num_cells, -1, dtype=np.int32)
    
    # Get the mapping array from EntityMap
    sub_cells_sorted = sub_cells[sort_order]
    for i, parent_cell in enumerate(sub_cells_sorted):
        parent_to_sub[parent_cell] = i

    # Define problem function spaces
    cell = mesh.ufl_cell()
    FE = basix.ufl.element("Lagrange", str(cell), degree)
    V = dolfinx.fem.functionspace(mesh, FE)
    Q = dolfinx.fem.functionspace(conductive_domain, FE)

    # Define test, trial, and previous-step functions
    Az = ufl.TrialFunction(V)
    vz = ufl.TestFunction(V)
    v = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    Azn = dolfinx.fem.Function(V)
    J0z = fem.Function(DG0)

    # Integration measures
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=surface_map["Exterior"])
    n = ufl.FacetNormal(mesh)
    dt = fem.Constant(mesh, dt_)
    x = ufl.SpatialCoordinate(mesh)
    omega = fem.Constant(mesh, default_scalar_type(omega_u))

    # Variational forms
    a_00 = dt / mu_R * ufl.inner(ufl.grad(Az), ufl.grad(vz)) * dx(Omega_n + Omega_c)
    a_00 += dt / mu_R * vz * (n[0] * Az.dx(0) - n[1] * Az.dx(1)) * ds
    a_00 += mu_0 * sigma * Az * vz * dx(Omega_c)

    # Motion voltage term
    u = omega * ufl.as_vector((-x[1], x[0]))
    a_00 += dt * mu_0 * sigma * ufl.dot(u, ufl.grad(Az)) * vz * dx(Omega_c)

    a_11 = dt * mu_0 * sigma * (v.dx(0) * q.dx(0) + v.dx(1) * q.dx(1)) * dx(Omega_c)

    L_0 = mu_0 * sigma * Azn * vz * dx(Omega_c)
    L_0 += dt * mu_0 * J0z * vz * dx(Omega_n)

    L = [
        dolfinx.fem.form(
            L_0,
            form_compiler_options=form_compiler_options,
            jit_options=jit_parameters,
        ),
        fem.form(
            fem.Constant(conductive_domain, default_scalar_type(0)) * q * ufl.dx(domain=conductive_domain),
            form_compiler_options=form_compiler_options,
            jit_options=jit_parameters,
        ),
    ]

    # Boundary conditions
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
    zeroV = fem.Function(V)
    zeroV.x.array[:] = 0
    bc_V = fem.dirichletbc(zeroV, bndry_dofs)

    conductive_domain.topology.create_connectivity(
        conductive_domain.topology.dim - 1, conductive_domain.topology.dim
    )
    conductive_domain_facets = dolfinx.mesh.exterior_facet_indices(conductive_domain.topology)
    q_boundary = fem.locate_dofs_topological(Q, tdim - 1, conductive_domain_facets)
    zeroQ = fem.Function(Q)
    bc_p = fem.dirichletbc(zeroQ, q_boundary)
    bcs = [bc_V, bc_p]

    # Matrix assembly
    a = [
        [dolfinx.fem.form(a_00, form_compiler_options=form_compiler_options, jit_options=jit_parameters), None],
        [None, dolfinx.fem.form(a_11, entity_maps=[entity_map], form_compiler_options=form_compiler_options, jit_options=jit_parameters)],
    ]
    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = fem.petsc.create_vector(L)
    fem.petsc.assemble_vector(b, L)
    
    # Convert bcs to block structure
    bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs)
    fem.petsc.apply_lifting(b, a, bcs=bcs0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs0)

    # PETSc solver
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setTolerances(atol=1e-9, rtol=1e-9)
    solver.setType("cg")
    solver.getPC().setType("fieldsplit")
    solver.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0] * Q.dofmap.index_map_bs
    offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs

    is_u = PETSc.IS().createStride(V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF)
    is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)
    solver.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    ksp_u, ksp_p = solver.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("lu")
    ksp_u.getPC().setFactorSolverType("mumps")

    ksp_p.setType("preonly")
    ksp_p.getPC().setType("lu")
    ksp_p.getPC().setFactorSolverType("mumps")

    solver.getPC().setUp()

    # Output functions
    solution_vector = A.createVecLeft()
    Az_out = dolfinx.fem.Function(V)
    V_out = dolfinx.fem.Function(Q)
    post_B = MagneticField2D(Az_out)
    derived = DerivedQuantities2D(Az_out, Azn, u, sigma, domains, ct, ft)
    Az_out.name = "Az"
    post_B.B.name = "B"

    if save_output:
        Az_vtx = VTXWriter(mesh.comm, str(outdir / "Az.bp"), [Az_out], engine="BP4")
        B_vtx = VTXWriter(mesh.comm, str(outdir / "B.bp"), [post_B.B], engine="BP4")
        V_vtx = VTXWriter(mesh.comm, str(outdir / "V.bp"), [V_out], engine="BP4")

    # Initialize variables
    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    Depth = 1
    I_rotor = mesh.comm.allreduce(fem.assemble_scalar(fem.form(Depth * r**2 * density * dx(Omega_c))))

    num_steps = int(T / float(dt.value))
    torques = np.zeros(num_steps + 1)
    torques_vol = np.zeros(num_steps + 1)
    times = np.zeros(num_steps + 1)
    omegas = np.zeros(num_steps + 1)
    pec_tot = np.zeros(num_steps + 1)
    pec_steel = np.zeros(num_steps + 1)
    VA = np.zeros(num_steps + 1)
    VmA = np.zeros(num_steps + 1)
    omegas[0] = omega_u

    # Time stepping
    t = 0.0
    update_current_density(J0z, omega_J, t, ct, currents)

    if MPI.COMM_WORLD.rank == 0 and progress:
        progressbar = tqdm.tqdm(desc="Solving time-dependent problem", total=int(T / float(dt.value)))

    for i in range(num_steps):
        if MPI.COMM_WORLD.rank == 0 and progress:
            progressbar.update(1)

        t += float(dt.value)
        update_current_density(J0z, omega_J, t, ct, currents)

        if apply_torque:
            A.zeroEntries()
            _petsc.assemble_matrix(A, a, bcs=bcs)
            A.assemble()

        with b.localForm() as loc_b:
            loc_b.set(0)
        _petsc.assemble_vector(b, L)
        _petsc.apply_lifting(b, a, bcs=bcs0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        _petsc.set_bc(b, bcs0)

        solver.solve(b, solution_vector)

        offset_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        offset_Q = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
        Az_out.x.array[:offset_V] = solution_vector.array_r[:offset_V]
        Az_out.x.scatter_forward()
        V_out.x.array[:offset_Q] = solution_vector.array_r[offset_V : offset_V + offset_Q]

        loss_al, loss_steel = derived.compute_loss(float(dt.value))
        pec_tot[i + 1] = float(dt.value) * (loss_al + loss_steel)
        pec_steel[i + 1] = float(dt.value) * loss_steel
        torques[i + 1] = derived.torque_surface()
        torques_vol[i + 1] = derived.torque_volume()
        vA, vmA = derived.compute_voltage(float(dt.value))
        VA[i + 1], VmA[i + 1] = vA, vmA
        times[i + 1] = t

        Azn.x.array[:offset_V] = solution_vector.array_r[:offset_V]
        Azn.x.scatter_forward()

        if apply_torque:
            omega.value += float(dt.value) * (derived.torque_volume() - T_ext(t)) / I_rotor
        omegas[i + 1] = float(omega.value)

        if save_output:
            post_B.interpolate()
            Az_vtx.write(t)
            B_vtx.write(t)
            V_vtx.write(t)

    b.destroy()
    if save_output:
        Az_vtx.close()
        B_vtx.close()
        V_vtx.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TEAM 30 problem solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--single", action="store_true", help="Solve single phase", default=False)
    parser.add_argument("--three", action="store_true", help="Solve three phase", default=False)
    parser.add_argument("--apply-torque", dest="apply_torque", action="store_true", default=False)
    parser.add_argument("--num_phases", type=int, default=1, help="Number of phases")
    parser.add_argument("--omega", dest="omegaU", type=np.float64, default=0, help="Angular speed [rad/s]")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial degree")
    parser.add_argument("--steps", type=int, default=100, help="Steps per phase")
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=False)
    parser.add_argument("--output", action="store_true", help="Save output", default=False)
    args = parser.parse_args()

    def T_ext(t):
        return 0

    if args.single:
        outdir = Path(f"TEAM30_{args.omegaU}_single")
        outdir.mkdir(exist_ok=True)
        solve_team30(
            True, args.num_phases, args.omegaU, args.degree,
            apply_torque=args.apply_torque, T_ext=T_ext,
            outdir=outdir, steps_per_phase=args.steps,
            plot=args.plot, progress=args.progress, save_output=args.output
        )
    if args.three:
        outdir = Path(f"TEAM30_{args.omegaU}_three")
        outdir.mkdir(exist_ok=True)
        solve_team30(
            False, args.num_phases, args.omegaU, args.degree,
            apply_torque=args.apply_torque, T_ext=T_ext,
            outdir=outdir, steps_per_phase=args.steps,
            plot=args.plot, progress=args.progress, save_output=args.output
        )
