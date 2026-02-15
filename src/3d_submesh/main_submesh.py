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
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
import numpy as np
from pathlib import Path

from entity_map_utils import get_entity_map

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
    rotate_magnetization,
    compute_B_field,
    solve_one_step_submesh,
    solve_magnet_only_initial_guess,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
from load_mesh import omega_rs, omega_rpm, omega_c, omega_pm, MAGNETS
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
    ) = load_mesh_and_extract_submesh(config.mesh_path, config=config)

    dx_parent = ufl.Measure("dx", domain=mesh_parent, subdomain_data=cell_tags_parent)
    dx_rs = measure_over(dx_parent, omega_rs())
    dx_rpm = measure_over(dx_parent, omega_rpm())
    dx_c = measure_over(dx_parent, omega_c())
    dx_pm = measure_over(dx_parent, omega_pm())
    dx_air = measure_over(dx_parent, AIR + AIR_GAP)
    dx_airgap = measure_over(dx_parent, AIR_GAP)
    dx_conductor = ufl.Measure("dx", domain=mesh_conductor)
    dx_cond_parent = measure_over(dx_parent, conducting(config))

    print("\n=== Materials ===")
    sigma, nu, density = setup_materials(mesh_parent, cell_tags_parent, config)

    # Weaken A-V coupling in PM to reduce B suppression (sigma_pm_coupling_scale < 1)
    sigma_coupling = None
    scale = float(getattr(config, "sigma_pm_coupling_scale", 1.0))
    if scale != 1.0:
        sigma_coupling = fem.Function(sigma.function_space)
        sigma_coupling.x.array[:] = sigma.x.array[:]
        for marker in MAGNETS:
            cells = cell_tags_parent.find(marker)
            if cells.size > 0:
                sigma_coupling.x.array[cells] *= scale
        sigma_coupling.x.scatter_forward()
        if mesh_parent.comm.rank == 0:
            print(f"  sigma_pm_coupling_scale={scale} (weakened in PM)")

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
    # Background magnet solution (updated each step when enabled)
    A_bg = fem.Function(A_space, name="A_bg")
    A_bg_prev = fem.Function(A_space, name="A_bg_prev")

    print("\n=== Boundary conditions ===")
    bc_A = setup_boundary_conditions_parent(mesh_parent, facet_tags_parent, A_space)
    sigma_submesh = interpolate_sigma_to_submesh(
        mesh_parent, mesh_conductor, sigma, entity_map, cell_tags_parent
    )
    V_space = fem.functionspace(
        mesh_conductor,
        basix.ufl.element("Lagrange", mesh_conductor.basix_cell(), config.degree_V)
    )
    bc_V_list, voltage_update_data = setup_boundary_conditions_submesh(
        mesh_conductor, V_space, cell_tags_conductor, conducting(config), config
    )
    block_bcs = [[bc_A], bc_V_list]

    print("\n=== Sources ===")
    J_z, M_vec = setup_sources(mesh_parent)
    initialise_magnetisation(mesh_parent, cell_tags_parent, M_vec, config)

    # For diagnostics: interpolate submesh V onto parent (conductor cells only)
    V_parent_space = fem.functionspace(
        mesh_parent,
        basix.ufl.element("Lagrange", mesh_parent.basix_cell(), config.degree_V),
    )
    V_parent = fem.Function(V_parent_space, name="V_parent")
    DG0 = fem.functionspace(mesh_parent, ("DG", 0))
    divJ_dg = fem.Function(DG0, name="divJ")

    print("\n=== DOF mapper ===")
    dof_mapper = create_dof_mapper(
        mesh_parent, mesh_conductor, A_space, V_space,
        entity_map, cell_tags_parent,
    )

    print("\n=== Forms ===")
    use_bg = bool(getattr(config, "use_magnet_background", False))
    a_blocks, L_blocks, a00_spd_form, interpolation_data, a_block_form, L_block_form = build_forms_submesh(
        mesh_parent, mesh_conductor, A_space, V_space,
        sigma, nu, J_z, M_vec, A_prev,
        dx_parent, dx_rs, dx_rpm, dx_c, dx_pm,
        dx_conductor, config, entity_map, dx_cond_parent,
        dx_air=dx_air,
        sigma_coupling=sigma_coupling,
        A_background=A_bg if use_bg else None,
        A_background_prev=A_bg_prev if use_bg else None,
        include_magnet_source_in_coupled=(not use_bg),
    )
    interpolation_data["sigma_submesh"] = sigma_submesh

    print("\n=== Assembly ===")
    mats, mat_nest, A00_standalone, A00_spd, interpolation_data = assemble_system_matrix_submesh(
        mesh_parent, a_blocks, block_bcs,
        a00_spd_form, interpolation_data, A_space, V_space,
        a_block_form,
    )

    print("\n=== Solver ===")
    # Postmortem: skip interior nodes for AMS when config.use_interior_nodes is False
    cell_tags_for_solver = cell_tags_parent if getattr(config, "use_interior_nodes", True) else None
    conductor_markers_for_solver = conducting(config) if getattr(config, "use_interior_nodes", True) else ()
    ksp = configure_solver_submesh(
        mesh_parent, mat_nest, mats, A_space, V_space, A00_spd, config,
        cell_tags_parent=cell_tags_for_solver,
        conductor_markers=conductor_markers_for_solver,
    )

    writer = None
    A_lag = None
    if config.write_results:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove existing files so HDF5 can create/lock (avoids "unable to lock file" on rerun/WSL)
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

    # Magnet-only as A_prev for first step: improves RHS, can strengthen B
    x0_A_mag = None
    if getattr(config, "use_magnet_initial_guess", False):
        if mesh_parent.comm.rank == 0:
            print("\n=== Magnet-only as A_prev for first step ===")
        rotate_magnetization(cell_tags_parent, M_vec, config, config.dt)
        x0_A_mag = solve_magnet_only_initial_guess(
            mesh_parent, A_space, nu, M_vec, cell_tags_parent,
            facet_tags_parent, dx_parent, dx_pm, config,
        )
        # If using background decomposition, store the magnet solution in A_bg/A_bg_prev.
        if use_bg:
            tmp = petsc.create_vector(A_space)
            x0_A_mag.copy(tmp)
            A_bg.x.array[:] = tmp.array[:A_bg.x.array.size]
            A_bg.x.scatter_forward()
            A_bg_prev.x.array[:] = A_bg.x.array[:]
            A_bg_prev.x.scatter_forward()
        # IMPORTANT:
        # - If using background decomposition, A_prev is the *correction* history field (start at 0),
        #   so we do NOT seed it with the magnet solution.
        # - If not using background, we keep the original behavior of seeding A_prev.
        if use_bg:
            A_prev.x.array[:] = 0.0
            A_prev.x.scatter_forward()
            x0_A_mag = None  # do not use magnet solution as Krylov x0 for correction solve
        else:
            # Copy magnet solution into A_prev for first step (scaled for convergence)
            scale = float(getattr(config, "magnet_A_prev_scale", 1.0))
            tmp = petsc.create_vector(A_space)
            x0_A_mag.copy(tmp)
            A_prev.x.array[:] = scale * tmp.array[:A_prev.x.array.size]
            A_prev.x.scatter_forward()
            # Optionally use as Krylov x0 (scaled) for better convergence
            if not getattr(config, "use_magnet_as_ksp_x0", False):
                x0_A_mag = None
        if mesh_parent.comm.rank == 0:
            print("  A_prev set from magnet-only solve")
        # Scale x0 for Krylov if using (same scale as A_prev for consistency)
        if (not use_bg) and getattr(config, "use_magnet_as_ksp_x0", False) and x0_A_mag is not None:
            x0_A_mag.scale(scale)  # scale in-place for Krylov x0

    print("\n=== Time loop ===")
    t = 0.0
    V_prev = None  # For Krylov initial guess on steps 2+
    for step in range(config.num_steps):
        t += config.dt
        if mesh_parent.comm.rank == 0:
            print(f"\nStep {step+1}/{config.num_steps}: t={t*1e3:.3f} ms")

        # Update background magnet solution each step (keeps magnet B from collapsing)
        if use_bg:
            # shift previous
            A_bg_prev.x.array[:] = A_bg.x.array[:]
            A_bg_prev.x.scatter_forward()
            # rotate M and solve magnet-only at current t
            rotate_magnetization(cell_tags_parent, M_vec, config, t)
            x_mag = solve_magnet_only_initial_guess(
                mesh_parent, A_space, nu, M_vec, cell_tags_parent,
                facet_tags_parent, dx_parent, dx_pm, config,
            )
            tmp = petsc.create_vector(A_space)
            x_mag.copy(tmp)
            A_bg.x.array[:] = tmp.array[:A_bg.x.array.size]
            A_bg.x.scatter_forward()

        # Krylov initial guess:
        # - Background decomposition: solve is for correction field -> start at 0 on step 1.
        # - Otherwise: can use magnet-only as x0 on the first step.
        if (not use_bg) and step == 0 and x0_A_mag is not None:
            x0_A_vec = x0_A_mag
            x0_V_vec = None
        elif V_prev is not None:
            x0_A_vec = A_prev.x.petsc_vec
            x0_V_vec = V_prev.x.petsc_vec
        else:
            x0_A_vec = None
            x0_V_vec = None

        A_sol, V_sol, residual_norm, rhs_norm, relative_residual = solve_one_step_submesh(
            mesh_parent, mesh_conductor, A_space, V_space,
            cell_tags_parent, config, ksp, mat_nest,
            a_blocks, L_blocks, block_bcs,
            J_z, M_vec, A_prev, t,
            voltage_update_data=voltage_update_data,
            x0_A_vec=x0_A_vec,
            x0_V_vec=x0_V_vec,
        )
        V_prev = V_sol

        if mesh_parent.comm.rank == 0:
            ksp_its = ksp.getIterationNumber()
            reason = ksp.getConvergedReason()
            petsc_rnorm = ksp.getResidualNorm()
            # Actual residual = ||b - Ax|| (error of Ax=b). Relative = actual/||b||.
            # Converged = PETSc stopped because relative residual <= rtol (reason=2); not converged = hit max_it (reason=-3) or diverged (reason<0).
            converged = reason > 0  # PETSc: positive reason = converged (e.g. 2=rtol), -3=max_it, <0=diverged
            rtol = float(getattr(config, "outer_rtol", 1e-4))
            print(f"  Solver: its={ksp_its}, reason={reason}  Converged(PETSc): {converged}")
            print(f"  True residual ||b-Ax||: {residual_norm:.6e}   RHS ||b||: {rhs_norm:.6e}   PETSc rnorm: {petsc_rnorm:.6e}")
            print(f"  True relative ||b-Ax||/||b||: {relative_residual:.4e}   (target rtol={rtol:g})")
            if (not converged) and (relative_residual <= rtol):
                print("  Note: true residual meets rtol, but PETSc did not declare convergence (its internal norm differs).")
            if reason == -3:
                print("  Status: hit max iterations (not converged by PETSc)")
            elif reason < 0:
                print(f"  Status: diverged (reason={reason})")
            elif not converged:
                print(f"  Status: not converged (reason={reason})")

        # Total A for B-field: background magnet field + correction (if enabled)
        A_for_B = A_sol
        if use_bg:
            A_total = fem.Function(A_space, name="A_total")
            A_total.x.array[:] = A_sol.x.array[:] + A_bg.x.array[:]
            A_total.x.scatter_forward()
            A_for_B = A_total
        B_sol, B_mag, max_B, min_B, norm_B, B_dg = compute_B_field(
            mesh_parent, A_for_B, B_space, B_magnitude_space
        )

        if mesh_parent.comm.rank == 0:
            print(f"  Max |B|: {max_B:.4e} T")

        # Rotation visibility diagnostic: airgap-averaged B direction (Bx, By) and angle.
        if bool(getattr(config, "diagnose_rotation", True)):
            try:
                vol_local = fem.assemble_scalar(fem.form(1.0 * dx_airgap))
                vol = float(mesh_parent.comm.allreduce(vol_local, op=MPI.SUM))
                bx_local = fem.assemble_scalar(fem.form(B_dg[0] * dx_airgap))
                by_local = fem.assemble_scalar(fem.form(B_dg[1] * dx_airgap))
                bx = float(mesh_parent.comm.allreduce(bx_local, op=MPI.SUM)) / max(vol, 1e-30)
                by = float(mesh_parent.comm.allreduce(by_local, op=MPI.SUM)) / max(vol, 1e-30)
                angle_deg = float(np.degrees(np.arctan2(by, bx)))
                if mesh_parent.comm.rank == 0:
                    print(f"  Airgap <B>: ({bx:.4e}, {by:.4e}) T, angle={angle_deg:.2f} deg")
            except Exception as e:  # pragma: no cover
                if mesh_parent.comm.rank == 0:
                    print(f"  Airgap <B> diagnostic: skipped ({type(e).__name__}: {e})")

        # --- Divergence-free diagnostic on conductor: div( J ) ~ 0
        # Approximate conductive current:
        #   J ≈ σ * ( (A_total^{n+1}-A_total^{n})/dt + ∇V )
        # with V interpolated from submesh to parent conductor cells.
        if bool(getattr(config, "diagnose_divJ", False)):
            try:
                # Interpolate V_sol (submesh) into V_parent (parent) on conductor cells
                tdim = mesh_parent.topology.dim
                n_sub = mesh_conductor.topology.index_map(tdim).size_local
                sub_cells = np.arange(n_sub, dtype=np.int32)
                parent_cells = get_entity_map(entity_map, inverse=False)[:n_sub].astype(np.int32)
                V_parent.x.array[:] = 0.0
                V_parent.interpolate(V_sol, cells0=parent_cells, cells1=sub_cells)
                V_parent.x.scatter_forward()

                inv_dt = 1.0 / float(config.dt)
                A_tot_now = A_for_B
                A_tot_prev = A_prev
                if use_bg:
                    A_tot_prev = fem.Function(A_space)
                    A_tot_prev.x.array[:] = A_prev.x.array[:] + A_bg_prev.x.array[:]
                    A_tot_prev.x.scatter_forward()

                J_expr = sigma * (inv_dt * (A_tot_now - A_tot_prev) + ufl.grad(V_parent))
                divJ_expr = ufl.div(J_expr)
                divJ_dg.interpolate(fem.Expression(divJ_expr, DG0.element.interpolation_points))
                divJ_dg.x.scatter_forward()

                # L2 norm over conductor region on parent
                l2_local = fem.assemble_scalar(fem.form((divJ_dg * divJ_dg) * dx_cond_parent))
                l2 = mesh_parent.comm.allreduce(l2_local, op=MPI.SUM)
                l2 = float(np.sqrt(l2))
                # max |divJ|
                imap = DG0.dofmap.index_map
                n_owned = imap.size_local
                local_max = float(np.max(np.abs(divJ_dg.x.array[:n_owned]))) if n_owned else 0.0
                max_abs = float(mesh_parent.comm.allreduce(local_max, op=MPI.MAX))
                if mesh_parent.comm.rank == 0:
                    print(f"  div(J) diagnostic on conductor: ||divJ||_L2={l2:.4e}, max|divJ|={max_abs:.4e}")
            except Exception as e:
                if mesh_parent.comm.rank == 0:
                    print(f"  div(J) diagnostic: skipped ({type(e).__name__}: {e})")

        if writer is not None:
            # Write total field (so animation shows magnet rotation), not just correction.
            A_lag.interpolate(A_for_B)
            writer.write_function(A_lag, t)
            writer.write_function(B_sol, t)
            writer.write_function(B_mag, t)

    if writer is not None:
        writer.close()

    if mesh_parent.comm.rank == 0:
        print("\n=== Done ===")


if __name__ == "__main__":
    main()
