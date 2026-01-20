#!/usr/bin/env python3
"""
Minimal entrypoint for the 3D transient A–V solver.

Run:
  cd src/3d
  python main.py
"""

import basix.ufl
from dolfinx import fem, io
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

    # Boundary conditions (block_bcs is what the solver needs)
    _, _, block_bcs = setup_boundary_conditions(mesh, ft, A_space, V_space)
    block_bcs[1].append(make_ground_bc_V(V_space, ct, conducting()))

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
    ksp = configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd, config)

    # Optional output
    writer = None
    A_lag = None
    if config.write_results:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
        writer = io.XDMFFile(MPI.COMM_WORLD, str(config.results_path), "w")
        writer.write_mesh(mesh)
        A_lag = fem.Function(fem.functionspace(mesh, ("Lagrange", 1, (3,))), name="A")

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


if __name__ == "__main__":
    main()
