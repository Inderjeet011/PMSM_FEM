#!/usr/bin/env python3
"""3D A-V solver: A on parent mesh, V on conductor submesh."""

import sys
import time
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, io
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
from load_mesh import omega_rs, omega_rpm, omega_c, omega_pm

from load_mesh_submesh import (
    conducting,
    load_mesh_and_extract_submesh,
    setup_boundary_conditions_parent,
    setup_boundary_conditions_submesh,
    setup_materials,
)
from solve_equations_submesh import (
    assemble_system_matrix_submesh,
    build_forms_submesh,
    configure_solver_submesh,
)
from solver_utils_submesh import (
    compute_B_field,
    initialise_magnetisation,
    make_config,
    measure_over,
    setup_sources,
    solve_one_step_submesh,
)


def main():
    t_start = time.perf_counter()
    config = make_config()

    (mesh_parent, mesh_conductor, cell_tags_parent, cell_tags_conductor,
     facet_tags_parent, entity_map) = load_mesh_and_extract_submesh(config.mesh_path)
    rank0 = mesh_parent.comm.rank == 0
    if rank0:
        print("\n=== Setup ===")

    dx_parent = ufl.Measure("dx", domain=mesh_parent, subdomain_data=cell_tags_parent)
    dx_rs = measure_over(dx_parent, omega_rs())
    dx_rpm = measure_over(dx_parent, omega_rpm())
    dx_c = measure_over(dx_parent, omega_c())
    dx_pm = measure_over(dx_parent, omega_pm())
    dx_cond_parent = measure_over(dx_parent, conducting())

    sigma, nu = setup_materials(mesh_parent, cell_tags_parent, config)

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

    bc_A = setup_boundary_conditions_parent(mesh_parent, facet_tags_parent, A_space)
    V_space = fem.functionspace(
        mesh_conductor,
        basix.ufl.element("Lagrange", mesh_conductor.basix_cell(), config.degree_V)
    )
    bc_V_list, v_drive_funcs = setup_boundary_conditions_submesh(
        mesh_conductor, V_space, cell_tags_conductor, conducting(), config
    )
    block_bcs = [[bc_A], bc_V_list]

    J_z, M_vec = setup_sources(mesh_parent)
    initialise_magnetisation(mesh_parent, cell_tags_parent, M_vec, config)

    a_blocks, L_blocks, a_block_form = build_forms_submesh(
        mesh_parent, A_space, V_space,
        sigma, nu, J_z, M_vec, A_prev,
        dx_parent, dx_rs, dx_rpm, dx_c, dx_pm,
        config, entity_map, dx_cond_parent,
    )

    mats, mat_nest = assemble_system_matrix_submesh(
        mesh_parent, a_blocks, block_bcs,
        A_space, V_space, a_block_form,
    )

    ksp = configure_solver_submesh(
        mesh_parent, mat_nest, mats, A_space, V_space, config,
    )

    writer = None
    A_lag = None
    if config.write_results:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
        if rank0:
            for p in [config.results_path, config.results_path.with_suffix(".h5")]:
                if p.exists():
                    p.unlink()
        mesh_parent.comm.barrier()
        writer = io.XDMFFile(MPI.COMM_WORLD, str(config.results_path), "w")
        writer.write_mesh(mesh_parent)
        A_lag = fem.Function(fem.functionspace(mesh_parent, ("Lagrange", 1, (3,))), name="A")

    if rank0:
        print("\n=== Solving ===")
    t = 0.0
    for step in range(config.num_steps):
        t += config.dt
        if rank0:
            print(f"\nStep {step+1}/{config.num_steps}: t={t*1e3:.3f} ms")

        # Update 3-phase voltage BCs: V_pos = +V_amp·sin(ω_e·t + β), V_neg = -V_amp·sin(ω_e·t + β)
        for phase in v_drive_funcs:
            v_val = float(config.V_amp) * np.sin(config.omega_e * t + phase["beta"])
            phase["pos"].x.array[:] = v_val
            phase["pos"].x.scatter_forward()
            phase["neg"].x.array[:] = -v_val
            phase["neg"].x.scatter_forward()

        A_sol, V_sol, residual_norm, rhs_norm = solve_one_step_submesh(
            mesh_parent, A_space, V_space,
            cell_tags_parent, config, ksp, mat_nest,
            a_blocks, L_blocks, block_bcs,
            J_z, M_vec, A_prev, t,
        )

        B_sol, B_mag, B_dg, max_B = compute_B_field(
            mesh_parent, A_sol, B_space, B_magnitude_space
        )

        if rank0:
            print(f"  Residual: {residual_norm:.6e}   ||b||: {rhs_norm:.6e}   Max |B|: {max_B:.4e} T")

        if writer is not None:
            A_lag.interpolate(A_sol)
            writer.write_function(A_lag, t)
            writer.write_function(V_sol, t)
            writer.write_function(B_dg, t)
            writer.write_function(B_sol, t)
            writer.write_function(B_mag, t)

    if writer is not None:
        writer.close()

    if rank0:
        print(f"\n=== Done ===  (time: {time.perf_counter() - t_start:.2f} s)")


if __name__ == "__main__":
    main()
