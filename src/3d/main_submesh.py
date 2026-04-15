#!/usr/bin/env python3
"""
Proof-of-concept: 3D A-V solver with submesh approach.

A (vector potential) lives on the full parent mesh.
V (scalar potential) lives exclusively on the conductor submesh.

Run:
  cd src/3d
  python main_submesh.py
"""

import basix.ufl
from dolfinx import fem, io
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import ufl
import numpy as np
import time
from pathlib import Path
import shutil

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
    compute_J_field_conductor,
    solve_one_step_submesh,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from load_mesh_submesh import omega_rs, omega_rpm, omega_c, omega_pm


def main():
    t_start = time.perf_counter()
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
    B_motor_writer = None
    V_bp_writer = None
    J_bp_writer = None
    V_bp_func = None
    J_bp_func = None
    if config.write_results:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
        bp_dir = config.results_path.parent
        if mesh_parent.comm.rank == 0:
            if config.results_path.exists():
                config.results_path.unlink()
            h5_path = config.results_path.with_suffix(".h5")
            if h5_path.exists():
                h5_path.unlink()
            for p in [
                bp_dir / "V_field_submesh.bp",
                bp_dir / "J_field_submesh.bp",
                bp_dir / "B_field_motor_submesh.bp",
            ]:
                if p.exists():
                    shutil.rmtree(p)
        mesh_parent.comm.barrier()
        writer = io.XDMFFile(MPI.COMM_WORLD, str(config.results_path), "w")
        writer.write_mesh(mesh_parent)
        A_lag = fem.Function(fem.functionspace(mesh_parent, ("Lagrange", 1, (3,))), name="A")
        V_bp_func = fem.Function(V_space, name="V")
        V_bp_writer = VTXWriter(mesh_parent.comm, str(bp_dir / "V_field_submesh.bp"), V_bp_func)
        J_bp_func = fem.Function(
            fem.functionspace(mesh_conductor, ("DG", 0, (mesh_conductor.geometry.dim,))),
            name="J",
        )
        J_bp_writer = VTXWriter(mesh_parent.comm, str(bp_dir / "J_field_submesh.bp"), J_bp_func)

        motor_tags = [4, 5, 6] + list(range(7, 13)) + list(range(13, 23))
        tdim = mesh_parent.topology.dim
        motor_cells = []
        for tag in motor_tags:
            cells = cell_tags_parent.find(tag)
            if cells.size > 0:
                motor_cells.append(cells)
        if motor_cells:
            target_cells = np.unique(np.concatenate(motor_cells)).astype(np.int32)
            submesh, subdomain_motor_to_domain = create_submesh(
                mesh_parent, tdim, target_cells
            )[:2]
            smsh_cell_imap = submesh.topology.index_map(tdim)
            smsh_cells = np.arange(
                smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts, dtype=np.int32
            )
            parent_cells = subdomain_motor_to_domain.sub_topology_to_topology(
                smsh_cells, inverse=False
            )
            B_motor = fem.Function(
                fem.functionspace(
                    submesh,
                    ("Discontinuous Lagrange", int(config.degree_A), (submesh.geometry.dim,)),
                ),
                name="B_motor",
            )
            B_motor_writer = {
                "parent_cells": parent_cells,
                "smsh_cells": smsh_cells,
                "function": B_motor,
                "writer": VTXWriter(
                    mesh_parent.comm, str(bp_dir / "B_field_motor_submesh.bp"), B_motor
                ),
            }

    print("\n=== Time loop ===")
    t = 0.0
    for step in range(config.num_steps):
        t += config.dt
        if mesh_parent.comm.rank == 0:
            print(f"\nStep {step+1}/{config.num_steps}: t={t*1e3:.3f} ms")

        A_prev_for_J = fem.Function(A_space, name="A_prev_J")
        A_prev_for_J.x.array[:] = A_prev.x.array[:]
        A_prev_for_J.x.scatter_forward()

        A_sol, V_sol, residual_norm, rhs_norm, relative_residual = solve_one_step_submesh(
            mesh_parent, A_space, V_space,
            cell_tags_parent, config, ksp, mat_nest,
            a_blocks, L_blocks, block_bcs,
            J_z, M_vec, A_prev, t,
        )

        B_sol, B_mag, B_dg, max_B, avg_B = compute_B_field(
            mesh_parent, A_sol, B_space, B_magnitude_space
        )

        if mesh_parent.comm.rank == 0:
            print(
                f"  Residual ||b-Ax||: {residual_norm:.6e}   Relative: {relative_residual:.4e}   "
                f"||b||: {rhs_norm:.6e}   Max |B|: {max_B:.4e} T   Avg |B|: {avg_B:.4e} T"
            )

        if B_motor_writer is not None:
            B_vis = fem.Function(
                fem.functionspace(
                    mesh_parent,
                    ("Discontinuous Lagrange", int(config.degree_A), (mesh_parent.geometry.dim,)),
                )
            )
            B_vis.interpolate(
                fem.Expression(ufl.curl(A_sol), B_vis.function_space.element.interpolation_points)
            )
            bw = B_motor_writer
            bw["function"].interpolate(B_vis, cells0=bw["parent_cells"], cells1=bw["smsh_cells"])
            bw["writer"].write(t)

        if V_bp_writer is not None and V_bp_func is not None:
            V_bp_func.x.array[:] = V_sol.x.array[:]
            V_bp_func.x.scatter_forward()
            V_bp_writer.write(t)
        if J_bp_writer is not None and J_bp_func is not None:
            J_total = compute_J_field_conductor(
                mesh_parent,
                mesh_conductor,
                entity_map,
                A_sol,
                A_prev_for_J,
                V_sol,
                sigma,
                config.dt,
                component="total",
                degree=config.degree_A,
            )
            J_bp_func.x.array[:] = J_total.x.array[:]
            J_bp_func.x.scatter_forward()
            J_bp_writer.write(t)

        if writer is not None:
            A_lag.interpolate(A_sol)
            writer.write_function(A_lag, t)
            writer.write_function(V_sol, t)
            writer.write_function(B_dg, t)
            writer.write_function(B_sol, t)
            writer.write_function(B_mag, t)

    if writer is not None:
        writer.close()
    if B_motor_writer is not None:
        B_motor_writer["writer"].close()
    if V_bp_writer is not None:
        V_bp_writer.close()
    if J_bp_writer is not None:
        J_bp_writer.close()

    if mesh_parent.comm.rank == 0:
        print(f"\n=== Done ===  (time: {time.perf_counter() - t_start:.2f} s)")


if __name__ == "__main__":
    main()