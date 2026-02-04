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
from dof_mapping import create_dof_mapper
from interpolate_materials import interpolate_sigma_to_submesh


def main():
    config = make_config()

    print("\n=== Loading mesh ===")
    (
        mesh_parent,
        mesh_conductor,
        cell_tags_parent,
        cell_tags_conductor,
        facet_tags_parent,
        conductor_cells,
        entity_map,
    ) = load_mesh_and_extract_submesh(config.mesh_path)

    dx_parent = ufl.Measure("dx", domain=mesh_parent, subdomain_data=cell_tags_parent)
    dx_rs = measure_over(dx_parent, omega_rs())
    dx_rpm = measure_over(dx_parent, omega_rpm())
    dx_c = measure_over(dx_parent, omega_c())
    dx_pm = measure_over(dx_parent, omega_pm())
    dx_conductor = ufl.Measure("dx", domain=mesh_conductor)
    # Conductor measure on parent (for entity_maps coupling: a01, a10, a11 use this)
    dx_cond_parent = measure_over(dx_parent, conducting())

    print("\n=== Materials ===")
    sigma, nu, density = setup_materials(mesh_parent, cell_tags_parent, config)
    sigma_submesh = interpolate_sigma_to_submesh(
        mesh_parent, mesh_conductor, sigma, entity_map, cell_tags_parent
    )

    print("\n=== Function spaces ===")
    A_space = fem.functionspace(
        mesh_parent,
        basix.ufl.element("N1curl", mesh_parent.basix_cell(), config.degree_A)
    )
    V_space = fem.functionspace(
        mesh_conductor,
        basix.ufl.element("Lagrange", mesh_conductor.basix_cell(), config.degree_V)
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

    print("\n=== Boundary conditions ===")
    bc_A = setup_boundary_conditions_parent(mesh_parent, facet_tags_parent, A_space)
    bc_V = setup_boundary_conditions_submesh(
        mesh_conductor, V_space, cell_tags_conductor, conducting()
    )
    block_bcs = [[bc_A], [bc_V]]

    print("\n=== Sources ===")
    J_z, M_vec = setup_sources(mesh_parent)
    initialise_magnetisation(mesh_parent, cell_tags_parent, M_vec, config)

    print("\n=== DOF mapper ===")
    dof_mapper = create_dof_mapper(
        mesh_parent, mesh_conductor, A_space, V_space,
        entity_map, cell_tags_parent,
    )

    print("\n=== Forms ===")
    a_blocks, L_blocks, a00_spd_form, interpolation_data, a_block_form, L_block_form = build_forms_submesh(
        mesh_parent, mesh_conductor, A_space, V_space,
        sigma, nu, J_z, M_vec, A_prev,
        dx_parent, dx_rs, dx_rpm, dx_c, dx_pm,
        dx_conductor, config, entity_map, dx_cond_parent,
    )
    interpolation_data["sigma_submesh"] = sigma_submesh

    print("\n=== Assembly ===")
    mats, mat_nest, A00_standalone, A00_spd, interpolation_data = assemble_system_matrix_submesh(
        mesh_parent, a_blocks, block_bcs,
        a00_spd_form, interpolation_data, A_space, V_space,
        a_block_form,
    )

    print("\n=== Solver ===")
    ksp = configure_solver_submesh(
        mesh_parent, mat_nest, mats, A_space, V_space, A00_spd, config
    )

    writer = None
    A_lag = None
    if config.write_results:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
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
            mesh_parent, mesh_conductor, A_space, V_space,
            cell_tags_parent, config, ksp, mat_nest,
            a_blocks, L_blocks, block_bcs,
            J_z, M_vec, A_prev, t,
        )

        if mesh_parent.comm.rank == 0:
            ksp_its = ksp.getIterationNumber()
            reason = ksp.getConvergedReason()
            print(f"  Solver: its={ksp_its}, rel_res={relative_residual:.4e}, reason={reason}")
            if reason < 0:
                print(f"  Warning: solver diverged (reason={reason})")
            elif reason == 0:
                print(f"  Warning: solver did not converge within max iterations")

        B_sol, B_mag, max_B, min_B, norm_B, B_dg = compute_B_field(
            mesh_parent, A_sol, B_space, B_magnitude_space
        )

        if mesh_parent.comm.rank == 0:
            print(f"  Max |B|: {max_B:.4e} T")

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
