#!/usr/bin/env python3
"""
Proof-of-concept: 3D A-V solver with submesh approach.

A (vector potential) lives on the full parent mesh.
V (scalar potential) lives exclusively on the conductor submesh.

Run:
  cd src/3d_submesh
  python main_submesh.py
"""

import basix.ufl
from dolfinx import fem, io
from mpi4py import MPI
import ufl
import numpy as np
from pathlib import Path

from load_mesh_submesh import (
    load_mesh_and_extract_submesh,
    setup_materials,
    setup_boundary_conditions_parent,
    setup_boundary_conditions_submesh,
    conducting,
    AIR,
    AIR_GAP,
)
from solve_equations_submesh import (
    build_forms_submesh,
    assemble_system_matrix_submesh,
    configure_solver_submesh,
)
from solver_utils_submesh import (
    make_config,
    measure_over,
    setup_sources,
    initialise_magnetisation,
    compute_B_field,
    solve_one_step_submesh,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
from load_mesh import omega_rs, omega_rpm, omega_c, omega_pm


def main():
    config = make_config()

    print("\n=== Loading mesh ===")
    (
        mesh_parent,
        mesh_conductor,
        cell_tags_parent,
        cell_tags_conductor,
        facet_tags_parent,
        entity_map,
    ) = load_mesh_and_extract_submesh(config.mesh_path)

    dx_parent = ufl.Measure("dx", domain=mesh_parent, subdomain_data=cell_tags_parent)
    dx_rs = measure_over(dx_parent, omega_rs())
    dx_rpm = measure_over(dx_parent, omega_rpm())
    dx_c = measure_over(dx_parent, omega_c())
    dx_pm = measure_over(dx_parent, omega_pm())
    dx_air = measure_over(dx_parent, AIR + AIR_GAP)
    dx_cond_parent = measure_over(dx_parent, conducting())

    print("\n=== Materials ===")
    sigma, nu = setup_materials(mesh_parent, cell_tags_parent, config)

    print("\n=== Function spaces ===")
    A_space = fem.functionspace(
        mesh_parent,
        basix.ufl.element("N1curl", mesh_parent.basix_cell(), config.degree_A)
    )
    B_space = fem.functionspace(
        mesh_parent,
        basix.ufl.element("Lagrange", mesh_parent.basix_cell(), config.degree_A, shape=(3,))
    )
    B_magnitude_space = fem.functionspace(
        mesh_parent,
        basix.ufl.element("Lagrange", mesh_parent.basix_cell(), config.degree_V)
    )
    A_prev = fem.Function(A_space, name="A_prev")
    A_prev.x.array[:] = 0.0
    A_prev.x.scatter_forward()

    print("\n=== Boundary conditions ===")
    bc_A = setup_boundary_conditions_parent(mesh_parent, facet_tags_parent, A_space)
    V_space = fem.functionspace(
        mesh_conductor,
        basix.ufl.element("Lagrange", mesh_conductor.basix_cell(), config.degree_V)
    )
    bc_V_list = setup_boundary_conditions_submesh(
        mesh_conductor, V_space, cell_tags_conductor, conducting()
    )
    block_bcs = [[bc_A], bc_V_list]

    print("\n=== Sources ===")
    J_z, M_vec = setup_sources(mesh_parent)
    initialise_magnetisation(mesh_parent, cell_tags_parent, M_vec, config)

    print("\n=== Forms ===")
    a_blocks, L_blocks, a_block_form = build_forms_submesh(
        mesh_parent, A_space, V_space,
        sigma, nu, J_z, M_vec, A_prev,
        dx_parent, dx_rs, dx_rpm, dx_c, dx_pm,
        config, entity_map, dx_cond_parent,
        dx_air=dx_air,
    )

    print("\n=== Assembly ===")
    mats, mat_nest = assemble_system_matrix_submesh(
        mesh_parent, a_blocks, block_bcs,
        A_space, V_space, a_block_form,
    )

    print("\n=== Solver ===")
    ksp = configure_solver_submesh(
        mesh_parent, mat_nest, mats, A_space, V_space, config,
        cell_tags_parent=cell_tags_parent,
        conductor_markers=conducting(),
    )

    writer = None
    A_lag = None
    if config.write_results:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
        if mesh_parent.comm.rank == 0:
            if config.results_path.exists():
                config.results_path.unlink()
            h5_path = config.results_path.with_suffix(".h5")
            if h5_path.exists():
                h5_path.unlink()
        mesh_parent.comm.barrier()
        writer = io.XDMFFile(MPI.COMM_WORLD, str(config.results_path), "w")
        writer.write_mesh(mesh_parent)
        A_lag = fem.Function(fem.functionspace(mesh_parent, ("Lagrange", 1, (3,))), name="A")

    print("\n=== Time loop ===")
    t = 0.0
    for step in range(config.num_steps):
        t += config.dt
        if mesh_parent.comm.rank == 0:
            print(f"\nStep {step+1}/{config.num_steps}: t={t*1e3:.3f} ms")

        A_sol, V_sol, residual_norm, rhs_norm, relative_residual = solve_one_step_submesh(
            mesh_parent, A_space, V_space,
            cell_tags_parent, config, ksp, mat_nest,
            a_blocks, L_blocks, block_bcs,
            J_z, M_vec, A_prev, t,
        )

        B_sol, B_mag, max_B = compute_B_field(
            mesh_parent, A_sol, B_space, B_magnitude_space
        )

        if mesh_parent.comm.rank == 0:
            print(f"  Residual ||b-Ax||: {residual_norm:.6e}   Relative: {relative_residual:.4e}   ||b||: {rhs_norm:.6e}   Max |B|: {max_B:.4e} T")

        if writer is not None:
            A_lag.interpolate(A_sol)
            writer.write_function(A_lag, t)
            writer.write_function(B_sol, t)
            writer.write_function(B_mag, t)

    if writer is not None:
        writer.close()

    if mesh_parent.comm.rank == 0:
        print("\n=== Done ===")


if __name__ == "__main__":
    main()
