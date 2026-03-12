#!/usr/bin/env python3
"""
Minimal entrypoint for the 3D transient A–V solver.

Run:
  cd src/3d
  python main.py
"""

import basix.ufl
from dolfinx import fem, io
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import ufl

from load_mesh import (
    COILS,
    MAGNETS,
    conducting,
    load_mesh,
    omega_c,
    omega_pm,
    omega_rpm,
    omega_rs,
    setup_boundary_conditions,
    setup_materials,
)
from solve_equations import (
        assemble_system_matrix,
        build_forms,
        configure_solver,
)
from solver_utils import (
        compute_B_field,
        compute_B_in_airgap,
        compute_torque_maxwell_surface,
        initialise_magnetisation,
        make_config,
        make_ground_bc_V,
        measure_over,
        setup_sources,
        solve_one_step,
)


def main():
    config = make_config()

    # Mesh + region tags
    mesh, ct, ft = load_mesh(config.mesh_path)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    # Paper-consistent measures
    dx_rs = measure_over(dx, omega_rs())
    dx_rpm = measure_over(dx, omega_rpm())
    dx_c = measure_over(dx, omega_c())
    dx_pm = measure_over(dx, omega_pm())

    # Materials
    sigma, nu, _density = setup_materials(mesh, ct, config)

    # Function spaces
    A_space = fem.functionspace(mesh, basix.ufl.element("N1curl", mesh.basix_cell(), config.degree_A))
    V_space = fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_V))
    B_space = fem.functionspace(
        mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_A, shape=(3,))
    )
    B_magnitude_space = fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_V))
    A_prev = fem.Function(A_space, name="A_prev")

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    # Use existing helper only for A (outer air-box); construct V BCs
    # directly on the coils so we know exactly where the terminals are.
    bc_A, _, _ = setup_boundary_conditions(mesh, ft, A_space, V_space)

    from load_mesh import COILS
    import numpy as np

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    cell2vert = mesh.topology.connectivity(tdim, 0)
    coords = mesh.geometry.x

    def _lower_coil_cells(marker: int) -> np.ndarray:
        cells = ct.find(marker)
        if cells.size == 0:
            return np.array([], dtype=np.int32)
        verts = np.unique(np.concatenate([cell2vert.links(int(c)) for c in cells]))
        z = coords[verts, 2]
        z_min, z_max = float(z.min()), float(z.max())
        tol_z = 0.05 * max(z_max - z_min, 1e-6)
        bottom_verts = set(int(v) for v in verts[z <= z_min + tol_z])
        lower_cells = []
        for c in cells:
            if any(int(v) in bottom_verts for v in cell2vert.links(int(c))):
                lower_cells.append(int(c))
        return np.array(lower_cells, dtype=np.int32)

    coil7_lower_cells = _lower_coil_cells(COILS[0])
    coil8_lower_cells = _lower_coil_cells(COILS[1])

    # Map those cells to V dofs
    def _cells_to_dofs(cells: np.ndarray) -> np.ndarray:
        if cells.size == 0:
            return np.array([], dtype=np.int32)
        dof_lists = [V_space.dofmap.cell_dofs(int(c)) for c in cells]
        return np.unique(np.concatenate(dof_lists)) if dof_lists else np.array([], dtype=np.int32)

    dofs7 = _cells_to_dofs(coil7_lower_cells)
    dofs8 = _cells_to_dofs(coil8_lower_cells)

    # Dirichlet BCs: V = 10 V on coil 7 lower, V = 0 on coil 8 lower
    bc_V_list = []
    if dofs7.size > 0:
        v_plus = fem.Function(V_space)
        v_plus.x.array[:] = 10.0
        bc_V_list.append(fem.dirichletbc(v_plus, dofs7))
    if dofs8.size > 0:
        v_zero = fem.Function(V_space)
        v_zero.x.array[:] = 0.0
        bc_V_list.append(fem.dirichletbc(v_zero, dofs8))

    # Fall back to old behaviour if we somehow failed to detect lower coil cells
    if not bc_V_list:
        _, _, old_block_bcs = setup_boundary_conditions(mesh, ft, A_space, V_space)
        bc_V_list = old_block_bcs[1]

    # Block-structured BCs for the solver
    block_bcs = [[bc_A], bc_V_list]

    # Sources (coil currents + permanent magnets)
    J_z, M_vec = setup_sources(mesh)
    initialise_magnetisation(mesh, ct, M_vec, config)

    # Assemble + solver
    from load_mesh import EXTERIOR_FACET_TAG

    a_blocks, L_blocks, a00_spd_form = build_forms(
        mesh,
        A_space,
        V_space,
        sigma,
        nu,
        J_z,
        M_vec,
        A_prev,
        dx,
        dx_rs,
        dx_rpm,
        dx_c,
        dx_pm,
        None,
        config,
        exterior_facet_tag=EXTERIOR_FACET_TAG,
    )
    mat_blocks, mat_nest, _, A00_spd, _ = assemble_system_matrix(mesh, a_blocks, block_bcs, a00_spd_form)
    ksp = configure_solver(
        mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd, config,
        cell_tags=ct, conductor_markers=conducting(),
    )

    # Optional output
    writer = None
    A_lag = None
    B_motor_writer = None
    if config.write_results:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
        writer = io.XDMFFile(MPI.COMM_WORLD, str(config.results_path), "w")
        writer.write_mesh(mesh)
        A_lag = fem.Function(fem.functionspace(mesh, ("Lagrange", 1, (3,))), name="A")

        # Prepare a VTX writer for B field restricted to the motor submesh.
        # Motor := everything except air / airgap.
        motor_tags = [4, 5, 6, 7, 8] + list(range(13, 23))
        tdim = mesh.topology.dim
        motor_cells = []
        for tag in motor_tags:
            cells = ct.find(tag)
            if cells.size > 0:
                motor_cells.append(cells)
        if motor_cells:
            import numpy as np

            target_cells = np.unique(np.concatenate(motor_cells)).astype(np.int32)
            submesh, subdomain_motor_to_domain = create_submesh(mesh, tdim, target_cells)[:2]
            smsh_cell_imap = submesh.topology.index_map(tdim)
            smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts, dtype=np.int32)
            parent_cells = subdomain_motor_to_domain.sub_topology_to_topology(smsh_cells, inverse=False)

            deg = int(config.degree_A)
            Submesh_vec_vis = fem.functionspace(
                submesh,
                ("Discontinuous Lagrange", deg, (submesh.geometry.dim,)),
            )
            B_motor = fem.Function(Submesh_vec_vis, name="B_motor")
            B_motor_writer = {
                "submesh": submesh,
                "parent_cells": parent_cells,
                "smsh_cells": smsh_cells,
                "function": B_motor,
                "writer": VTXWriter(mesh.comm, "B_field_motor.bp", B_motor),
            }

    # Time loop
    t = 0.0
    for step in range(config.num_steps):
        t += config.dt

        A_sol, V_sol = solve_one_step(
            mesh,
            A_space,
            V_space,
            ct,
            config,
            ksp,
            mat_nest,
            a_blocks,
            L_blocks,
            block_bcs,
            J_z,
            M_vec,
            A_prev,
            t,
        )

        # Compute B = curl(A)
        # - B_dg: DG0 cell data (most "physics-first")
        # - B / B_Magnitude: projected fields for nicer visualization
        B_sol, B_magnitude_sol, max_B, _min_B, _norm_B, B_dg = compute_B_field(
            mesh, A_sol, B_space, B_magnitude_space
        )

        # If requested, also write B restricted to the motor submesh.
        if B_motor_writer is not None:
            # Build a DG representation of B on the parent mesh, then restrict.
            deg = int(config.degree_A)
            vector_vis = fem.functionspace(
                mesh, ("Discontinuous Lagrange", deg, (mesh.geometry.dim,))
            )
            curlA = ufl.curl(A_sol)
            B_expr = fem.Expression(curlA, vector_vis.element.interpolation_points)
            B_vis = fem.Function(vector_vis)
            B_vis.interpolate(B_expr)

            bw = B_motor_writer
            bw["function"].interpolate(
                B_vis, cells0=bw["parent_cells"], cells1=bw["smsh_cells"]
            )
            bw["writer"].write(t)

        # Compute B-field statistics in airgap (use B_sol for better accuracy)
        B_airgap_stats = compute_B_in_airgap(mesh, B_dg, ct, B_sol=B_sol)
        
        # Calculate electromagnetic torque (Maxwell stress tensor on MidAir surface)
        torque = compute_torque_maxwell_surface(mesh, A_sol, ft, config, midair_tag=2)

        if mesh.comm.rank == 0:
            try:
                its = ksp.getIterationNumber()
                reason = ksp.getConvergedReason()
            except Exception:
                its = None
                reason = None
            print(
                f"step {step+1}/{config.num_steps}: t={t*1e3:.3f} ms, "
                f"max|B|={max_B:.3e} T, its={its}, reason={reason}"
            )
            print(f"  Airgap B: avg={B_airgap_stats['B_avg']:.3e} T, "
                  f"max={B_airgap_stats['B_max']:.3e} T, "
                  f"radial={B_airgap_stats['B_radial_avg']:.3e} T, "
                  f"tangential={B_airgap_stats['B_tangential_avg']:.3e} T")
            print(f"  Torque: {torque:.4e} N·m ({torque*1e3:.4f} mN·m)")
            if B_airgap_stats.get('B_magnet_avg', 0) > 0:
                print(f"  Comparison: Magnet avg={B_airgap_stats['B_magnet_avg']:.3e} T, "
                      f"Rotor avg={B_airgap_stats['B_rotor_avg']:.3e} T")
                if B_airgap_stats.get('airgap_radii'):
                    r_min, r_max = B_airgap_stats['airgap_radii']
                    print(f"  Airgap region: r=[{r_min:.4f}, {r_max:.4f}] m "
                          f"(expected: [0.042, 0.044] m)")

        if writer is not None:
            try:
                A_lag.interpolate(A_sol)
            except Exception:  # pragma: no cover
                A_lag.x.array[:] = 0.0
            V_sol.name = "V"
            B_dg.name = "B_dg"
            B_sol.name = "B"
            B_magnitude_sol.name = "B_Magnitude"
            writer.write_function(A_lag, t)
            writer.write_function(V_sol, t)
            writer.write_function(B_dg, t)
            writer.write_function(B_sol, t)
            writer.write_function(B_magnitude_sol, t)

    if writer is not None:
        writer.close()
    if B_motor_writer is not None:
        B_motor_writer["writer"].close()


if __name__ == "__main__":
    main()
