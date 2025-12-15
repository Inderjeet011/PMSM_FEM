#!/usr/bin/env python3
"""Simple 3D A-V solver for PMSM."""

import csv
import basix.ufl
import numpy as np
from dolfinx import fem, io
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from compute_B import compute_B_field
from load_mesh import (
        DomainTags3D,
        SimulationConfig3D,
        load_mesh,
        maybe_retag_cells,
        setup_boundary_conditions,
        setup_materials,
)
from solve_equations import (
        assemble_system_matrix,
        build_forms,
        configure_solver,
        current_stats,
        initialise_magnetisation,
        rebuild_linear_forms,
        rotate_magnetization,
        setup_sources,
        update_currents,
)


def copy_block_to_function(block_vec, target):
        """Copy PETSc vector to Function."""
        with block_vec.localForm() as src:
            target.x.array[:] = src.array_r[:target.x.array.size]
        target.x.scatter_forward()


def measure_over(dx, markers):
        """Create measure over multiple markers."""
        measure = None
        for marker in markers:
            term = dx(marker)
            measure = term if measure is None else measure + term
        return measure


def assemble_rhs(mesh, L_blocks, a_blocks, block_bcs):
        """Assemble RHS vector."""
        block_vecs = []
        for i, L_form in enumerate(L_blocks):
            vec = petsc.create_vector(L_form)
            petsc.assemble_vector(vec, L_form)
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            if i < len(a_blocks):
                for j, a_form in enumerate(a_blocks[i]):
                    if a_form is not None and i != j and j < len(block_bcs) and block_bcs[j]:
                        petsc.apply_lifting(vec, [a_form], bcs=[block_bcs[j]])
            
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # Apply boundary conditions to both blocks (A and V)
            if i < len(block_bcs) and block_bcs[i]:
                petsc.set_bc(vec, block_bcs[i])
            
            block_vecs.append(vec)
        
        return PETSc.Vec().createNest(block_vecs, comm=mesh.comm)


def solve_one_step(mesh, A_space, V_space, ct, config, ksp, mat_nest, a_blocks, 
                    L_blocks, block_bcs, sigma, J_z, M_vec, A_prev, dx, dx_magnets, t):
        """Solve one time step."""
        if mesh.comm.rank == 0:
            print(f"Solving at t = {t*1e3:.3f} ms")
        
        update_currents(mesh, ct, J_z, config, t)
        rotate_magnetization(mesh, ct, M_vec, config, t)
        
        L_blocks = rebuild_linear_forms(mesh, A_space, V_space, sigma, J_z, M_vec, 
                                        A_prev, dx, dx_magnets, config)
        
        rhs = assemble_rhs(mesh, L_blocks, a_blocks, block_bcs)
        
        # Diagnostic: Check sources and RHS at each step
        if mesh.comm.rank == 0:
            import sys
            # Check source values
            max_J = current_stats(J_z)
            # M_vec is a Function, access via x.array
            M_array = M_vec.x.array.reshape((-1, 3))
            max_M = np.max(np.linalg.norm(M_array, axis=1))
            mean_M = np.mean(np.linalg.norm(M_array, axis=1))
            
            # Check RHS norms
            rhs_A = rhs.getNestSubVecs()[0]
            rhs_V = rhs.getNestSubVecs()[1]
            with rhs_A.localForm() as local:
                norm_A = np.linalg.norm(local.array_r)
                max_A = np.max(np.abs(local.array_r))
            with rhs_V.localForm() as local:
                norm_V = np.linalg.norm(local.array_r)
                max_V = np.max(np.abs(local.array_r))
            
            print(f"  [DIAG] Sources: max|J_z|={max_J:.3e} A/m², max|M|={max_M:.3e} A/m, mean|M|={mean_M:.3e} A/m", flush=True)
            print(f"  [DIAG] RHS: ||b_A||={norm_A:.6e}, max|b_A|={max_A:.6e}, ||b_V||={norm_V:.6e}, max|b_V|={max_V:.6e}", flush=True)
        
        sol_blocks = [
            petsc.create_vector(L_blocks[0]),
            petsc.create_vector(L_blocks[1]),
        ]
        sol = PETSc.Vec().createNest(sol_blocks, comm=mesh.comm)
        
        # Set initial guess to zero explicitly
        sol.set(0.0)
        
        # Check convergence reason
        if mesh.comm.rank == 0:
            print(f"  [DIAG] About to call ksp.solve()...")
        import time
        t_solve_start = time.time()
        ksp.solve(rhs, sol)
        t_solve_end = time.time()
        if mesh.comm.rank == 0:
            print(f"[TIME] outer_solve_wall_s = {t_solve_start-t_solve_start + (t_solve_end-t_solve_start):.3f}", flush=True)
        
        # Diagnostic: Check convergence and solution
        if mesh.comm.rank == 0:
            reason = ksp.getConvergedReason()
            iterations = ksp.getIterationNumber()
            residual = ksp.getResidualNorm()
            
            # Map convergence reason code to name
            reason_map = {
                -3: "DIVERGED_ITS (max iterations)",
                -9: "DIVERGED_NANORINF",
                -11: "DIVERGED_BREAKDOWN",
                1: "CONVERGED_RTOL",
                2: "CONVERGED_ATOL",
                3: "CONVERGED_ITS",
            }
            reason_str = reason_map.get(reason, f"UNKNOWN({reason})")
            print(f"[OUTER-FINAL] reason={reason_str}, its={iterations}, true_resid={residual:.6e}", flush=True)
        
        A_sol = fem.Function(A_space, name="A")
        V_sol = fem.Function(V_space, name="V")
        copy_block_to_function(sol.getNestSubVecs()[0], A_sol)
        copy_block_to_function(sol.getNestSubVecs()[1], V_sol)
        
        # Diagnostic: Check solution norms
        if mesh.comm.rank == 0:
            # Use .x.array directly (dolfinx Vector), no localForm
            A_array = A_sol.x.array.reshape((-1, 3))
            norm_A = np.linalg.norm(A_array)
            max_A = np.max(np.linalg.norm(A_array, axis=1))
            mean_A = np.mean(np.linalg.norm(A_array, axis=1))

            V_array = V_sol.x.array
            norm_V = np.linalg.norm(V_array)
            max_V = np.max(np.abs(V_array))
            mean_V = np.mean(np.abs(V_array))

            print(f"  [DIAG] Solution: ||A||={norm_A:.6e} Wb/m, max|A|={max_A:.6e} Wb/m, mean|A|={mean_A:.6e} Wb/m", flush=True)
            print(f"  [DIAG] Solution: ||V||={norm_V:.6e} V, max|V|={max_V:.6e} V, mean|V|={mean_V:.6e} V", flush=True)
        
        A_prev.x.array[:] = A_sol.x.array[:]
        iterations = ksp.getIterationNumber()
        residual = ksp.getResidualNorm()
        max_J = current_stats(J_z)
        
        if mesh.comm.rank == 0:
            print(f"  Done: {iterations} iterations, residual = {residual:.2e}")
            print("")  # Blank line for readability
        
        return A_sol, V_sol, iterations, max_J, residual, L_blocks


def run_solver(config=None):
        """Main solver function."""
        if config is None:
            config = SimulationConfig3D()
        
        print("\n=== SOLVER SETUP ===")
        
        # Load mesh
        print("Loading mesh...")
        mesh, ct, ft = load_mesh(config.mesh_path)
        ct = maybe_retag_cells(mesh, ct)
        
        # Setup measures
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
        if ft is not None:
            ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
        else:
            ds = ufl.ds(domain=mesh)
        dx_conductors = measure_over(dx, DomainTags3D.conducting())
        dx_magnets = measure_over(dx, DomainTags3D.MAGNETS)
        
        # Setup materials
        print("Setting up materials...")
        sigma, nu, density = setup_materials(mesh, ct, config)
        
        # Setup function spaces
        print("Setting up function spaces...")
        nedelec = basix.ufl.element("N1curl", mesh.basix_cell(), config.degree_A)
        lagrange = basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_V)
        A_space = fem.functionspace(mesh, nedelec)
        V_space = fem.functionspace(mesh, lagrange)
        lagrange_vec = basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_A, shape=(3,))
        B_space = fem.functionspace(mesh, lagrange_vec)
        B_magnitude_space = fem.functionspace(mesh, lagrange)
        A_prev = fem.Function(A_space, name="A_prev")
        
        if mesh.comm.rank == 0:
            ndofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs
            print(f"DOFs: A={ndofs}, V={V_space.dofmap.index_map.size_global}")
        
        # Setup boundary conditions
        print("Setting up boundary conditions...")
        bc_A, bc_V, block_bcs = setup_boundary_conditions(mesh, ft, A_space, V_space)

        # Ground V inside conductor network to remove constant null mode
        from solve_equations import make_ground_bc_V
        bc_ground = make_ground_bc_V(mesh, V_space, ct, DomainTags3D.conducting())
        block_bcs[1].append(bc_ground)
        # Grounding diagnostics
        if mesh.comm.rank == 0:
            print("[DIAG] Ground BC for V created (one DOF globally should be constrained).")
        try:
            print(f"[DIAG] rank={mesh.comm.rank} ground dofs={bc_ground.dof_indices().size}")
        except AttributeError:
            # Older dolfinx: no dof_indices; best-effort diagnostic
            print(f"[DIAG] rank={mesh.comm.rank} ground BC applied (no dof_indices API).")
        
        # Setup sources
        print("Setting up sources...")
        J_z, M_vec = setup_sources(mesh, ct)
        initialise_magnetisation(mesh, ct, M_vec, config)
        
        # Build forms and assemble matrix
        print("Building forms and assembling matrix...")
        from load_mesh import EXTERIOR_FACET_TAG
        a_blocks, L_blocks, a00_spd_form = build_forms(mesh, A_space, V_space, sigma, nu, J_z, M_vec,
                                                        A_prev, dx, dx_conductors, dx_magnets, ds, config,
                                                        exterior_facet_tag=EXTERIOR_FACET_TAG)
        mat_blocks, mat_nest, _, A00_spd = assemble_system_matrix(mesh, a_blocks, block_bcs, a00_spd_form)
        
        # Configure solver
        print("Configuring solver...")
        ksp = configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd, config.degree_A)
        
        print("Setup complete!\n")
        
        # Time loop
        print("=== TIME LOOP ===")
        num_steps = config.num_steps
        dt = config.dt
        current_time = 0.0
        
        if mesh.comm.rank == 0:
            print(f"Steps: {num_steps}, dt: {dt*1e3:.3f} ms\n")
        
        writer = None
        if config.write_results:
            config.results_path.parent.mkdir(parents=True, exist_ok=True)
            if mesh.comm.rank == 0:
                print(f"Writing results to: {config.results_path}")
            writer = io.XDMFFile(MPI.COMM_WORLD, str(config.results_path), "w")
            writer.write_mesh(mesh)
            if ct is not None:
                writer.write_meshtags(ct, mesh.geometry)
            if ft is not None:
                writer.write_meshtags(ft, mesh.geometry)
        
        config.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.diagnostics_path, "w", newline="") as diag_file:
            diag_writer = csv.writer(diag_file)
            diag_writer.writerow(["step", "time_ms", "norm_A", "iterations", "residual", 
                                "max_J", "max_B", "min_B", "norm_B"])
            
            for step in range(1, num_steps + 1):
                current_time += dt
                A_sol, V_sol, its, max_J, residual, L_blocks = solve_one_step(
                    mesh, A_space, V_space, ct, config, ksp, mat_nest, a_blocks, L_blocks,
                    block_bcs, sigma, J_z, M_vec, A_prev, dx, dx_magnets, current_time
                )
                
                if mesh.comm.rank == 0:
                    print("Computing B = curl(A)...")
                
                debug_B = (step == 1)  # Debug first step only
                # For visualization: compute B on the full domain.
                # You can threshold/clip the outer air box in ParaView without changing physics.
                B_sol, B_magnitude_sol, max_B, min_B, norm_B, B_dg = compute_B_field(
                    mesh, A_sol, B_space, B_magnitude_space, config, 
                    cell_tags=ct, debug=debug_B, restrict_to_airgap=False
                )
                
                if writer is not None:
                    A_lag_space = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
                    A_lag = fem.Function(A_lag_space, name="A")
                    try:
                        A_lag.interpolate(A_sol)
                    except:
                        A_lag.x.array[:] = 0.0
                    
                    V_sol.name = "V"
                    B_dg.name = "B_dg"
                    B_sol.name = "B"
                    B_magnitude_sol.name = "B_Magnitude"
                    
                    writer.write_function(A_lag, current_time)
                    writer.write_function(V_sol, current_time)
                    # DG0 B written as cell data
                    writer.write_function(B_dg, current_time)
                    # Smoothed / projected B and its magnitude for visualization
                    writer.write_function(B_sol, current_time)
                    writer.write_function(B_magnitude_sol, current_time)
                
                norm_A = np.linalg.norm(A_sol.x.array)
                diag_writer.writerow([step, current_time * 1e3, norm_A, its, residual,
                                    max_J, max_B, min_B, norm_B])
                
                if mesh.comm.rank == 0:
                    print(f"Step {step}/{num_steps}: t={current_time*1e3:.3f} ms, "
                        f"||A||={norm_A:.3e}, ||B||={norm_B:.3e}, max|B|={max_B:.3e} T\n")

                # For now, stop after first step to ensure clean output files for visualization.
                # This does not change physics; it only shortens the simulated time window.
                if step == 1:
                    break
        
        if writer is not None:
            writer.close()
            if mesh.comm.rank == 0:
                print(f"Results written to: {config.results_path}")
        
        print("=== DONE ===")


if __name__ == "__main__":
        run_solver()
