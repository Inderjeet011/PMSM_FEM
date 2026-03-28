#!/usr/bin/env python3
"""3D A-V solver: A on parent mesh, V on conductor submesh."""

import sys
import time
import shutil
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_submesh
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))

from load_mesh_submesh import (
    conducting,
    load_mesh_and_extract_submesh,
    omega_c,
    omega_pm,
    omega_rpm,
    omega_rs,
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
    compute_J_field_conductor,
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
    bc_V_list = setup_boundary_conditions_submesh(
        mesh_conductor, V_space, cell_tags_conductor, conducting(), config
    )
    block_bcs = [[bc_A], bc_V_list]

    coil_volumes = {}
    for coil_marker in omega_c():
        dx_coil = measure_over(dx_parent, (coil_marker,))
        coil_volumes[int(coil_marker)] = float(fem.assemble_scalar(fem.form(1.0 * dx_coil)))
    config.coil_volumes = coil_volumes
    if rank0:
        print("  Coil volumes:")
        for coil_marker in sorted(coil_volumes):
            print(f"    tag {coil_marker}: {coil_volumes[coil_marker]:.6e} m³")

    J_z, M_vec = setup_sources(mesh_parent)
    initialise_magnetisation(mesh_parent, cell_tags_parent, M_vec, config)

    a_blocks, L_blocks, a_block_form = build_forms_submesh(
        mesh_parent, A_space, V_space,
        sigma, nu, J_z, M_vec, A_prev,
        dx_parent, dx_rs, dx_rpm, dx_c, dx_pm,
        config, entity_map, dx_cond_parent,
    )

    mats, mat_nest = assemble_system_matrix_submesh(
        mesh_parent, block_bcs, A_space, V_space, a_block_form
    )

    ksp = configure_solver_submesh(
        mesh_parent, mat_nest, mats, A_space, V_space, config,
    )

    writer = None
    A_lag = None
    B_motor_writer = None
    V_bp_writer = None
    J_bp_writer = None
    V_bp_func = None
    J_bp_func = None
    bp_dir = None
    if config.write_results:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
        bp_dir = config.results_path.parent
        if rank0:
            for p in [config.results_path, config.results_path.with_suffix(".h5")]:
                if p.exists():
                    p.unlink()
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
            submesh, subdomain_motor_to_domain = create_submesh(mesh_parent, tdim, target_cells)[:2]
            smsh_cell_imap = submesh.topology.index_map(tdim)
            smsh_cells = np.arange(
                smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts, dtype=np.int32
            )
            parent_cells = subdomain_motor_to_domain.sub_topology_to_topology(
                smsh_cells, inverse=False
            )

            Submesh_vec_vis = fem.functionspace(
                submesh,
                ("Discontinuous Lagrange", int(config.degree_A), (submesh.geometry.dim,)),
            )
            B_motor = fem.Function(Submesh_vec_vis, name="B_motor")
            B_motor_writer = {
                "parent_cells": parent_cells,
                "smsh_cells": smsh_cells,
                "function": B_motor,
                "writer": VTXWriter(
                    mesh_parent.comm, str(bp_dir / "B_field_motor_submesh.bp"), B_motor
                ),
            }

    if rank0:
        print("\n=== Solving ===")
    t = 0.0
    for step in range(config.num_steps):
        t += config.dt
        if rank0:
            print(f"\nStep {step+1}/{config.num_steps}: t={t*1e3:.3f} ms")

        A_prev_for_J = fem.Function(A_space, name="A_prev_J")
        A_prev_for_J.x.array[:] = A_prev.x.array[:]
        A_prev_for_J.x.scatter_forward()

        A_sol, V_sol, residual_norm, rhs_norm = solve_one_step_submesh(
            mesh_parent, A_space, V_space,
            cell_tags_parent, config, ksp, mat_nest,
            a_blocks, L_blocks, block_bcs,
            J_z, M_vec, A_prev, t,
        )

        B_sol, B_mag, max_B = compute_B_field(
            mesh_parent, A_sol, B_space, B_magnitude_space
        )

        if B_motor_writer is not None:
            vector_vis = fem.functionspace(
                mesh_parent,
                ("Discontinuous Lagrange", int(config.degree_A), (mesh_parent.geometry.dim,)),
            )
            B_vis = fem.Function(vector_vis, name="B_vis")
            B_vis.interpolate(fem.Expression(ufl.curl(A_sol), vector_vis.element.interpolation_points))

            bw = B_motor_writer
            bw["function"].interpolate(B_vis, cells0=bw["parent_cells"], cells1=bw["smsh_cells"])
            bw["writer"].write(t)

        if V_bp_writer is not None:
            V_bp_func.x.array[:] = V_sol.x.array[:]
            V_bp_func.x.scatter_forward()
            V_bp_writer.write(t)

        if J_bp_writer is not None:
            J_sol = compute_J_field_conductor(
                mesh_parent,
                mesh_conductor,
                entity_map,
                A_sol,
                A_prev_for_J,
                V_sol,
                sigma,
                config.dt,
            )
            J_bp_func.x.array[:] = J_sol.x.array[:]
            J_bp_func.x.scatter_forward()
            J_bp_writer.write(t)

        if rank0:
            print(f"  Residual: {residual_norm:.6e}   ||b||: {rhs_norm:.6e}   Max |B|: {max_B:.4e} T")

        if writer is not None:
            A_lag.interpolate(A_sol)
            writer.write_function(A_lag, t)
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

    if rank0:
        print(f"\n=== Done ===  (time: {time.perf_counter() - t_start:.2f} s)")


if __name__ == "__main__":
    main()