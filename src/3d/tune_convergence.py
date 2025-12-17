#!/usr/bin/env python3
"""Convergence tuning sweep for A-block preconditioner parameters."""

import sys
sys.path.insert(0, 'src/3d')

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
    EXTERIOR_FACET_TAG,
)
from solve_equations import (
    assemble_system_matrix,
    build_forms,
    configure_solver,
    current_stats,
    initialise_magnetisation,
    make_ground_bc_V,
    rebuild_linear_forms,
    rotate_magnetization,
    setup_sources,
    update_currents,
)


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
        
        if i < len(block_bcs) and block_bcs[i]:
            petsc.set_bc(vec, block_bcs[i])
        
        block_vecs.append(vec)
    
    return PETSc.Vec().createNest(block_vecs, comm=mesh.comm)


def solve_one_step_tuning(mesh, A_space, V_space, ct, config, ksp, mat_nest, a_blocks, 
                         L_blocks, block_bcs, sigma, J_z, M_vec, A_prev, dx, dx_magnets, t):
    """Solve one time step and return convergence info."""
    if mesh.comm.rank == 0:
        print(f"Solving at t = {t*1e3:.3f} ms")
    
    update_currents(mesh, ct, J_z, config, t)
    rotate_magnetization(mesh, ct, M_vec, config, t)
    
    L_blocks = rebuild_linear_forms(mesh, A_space, V_space, sigma, J_z, M_vec, 
                                    A_prev, dx, dx_magnets, config)
    
    rhs = assemble_rhs(mesh, L_blocks, a_blocks, block_bcs)
    
    sol_blocks = [
        petsc.create_vector(L_blocks[0]),
        petsc.create_vector(L_blocks[1]),
    ]
    sol = PETSc.Vec().createNest(sol_blocks, comm=mesh.comm)
    sol.set(0.0)
    
    # Collect residual history
    residual_history = []
    def _monitor(ksp_obj, its, rnorm):
        residual_history.append((its, float(rnorm)))
        if mesh.comm.rank == 0:
            print(f"  [OUTER] its={its:3d}, true_rnorm={rnorm:.6e}")
    
    ksp.setMonitor(_monitor)
    
    ksp.solve(rhs, sol)
    
    reason = ksp.getConvergedReason()
    iterations = ksp.getIterationNumber()
    residual = ksp.getResidualNorm()
    
    A_sol = fem.Function(A_space, name="A")
    V_sol = fem.Function(V_space, name="V")
    with sol.getNestSubVecs()[0].localForm() as src:
        A_sol.x.array[:] = src.array_r[:A_sol.x.array.size]
    with sol.getNestSubVecs()[1].localForm() as src:
        V_sol.x.array[:] = src.array_r[:V_sol.x.array.size]
    A_sol.x.scatter_forward()
    V_sol.x.scatter_forward()
    
    A_prev.x.array[:] = A_sol.x.array[:]
    
    return residual_history, reason, iterations, residual


def run_tuning_sweep(beta_pc_values=[0.3, 0.5, 0.7, 1.0], alpha_spd_factor=1.0):
    """Run convergence tuning sweep."""
    config = SimulationConfig3D()
    config.alpha_spd_factor = alpha_spd_factor
    config.num_steps = 1  # Only test first step
    config.write_results = False  # Don't write files during tuning
    
    print("\n" + "="*80)
    print(f"CONVERGENCE TUNING SWEEP")
    print(f"alpha_spd_factor = {alpha_spd_factor}")
    print(f"beta_pc values: {beta_pc_values}")
    print("="*80 + "\n")
    
    # Load mesh and setup (only once)
    print("Loading mesh and setting up...")
    mesh, ct, ft = load_mesh(config.mesh_path)
    ct = maybe_retag_cells(mesh, ct)
    
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    if ft is not None:
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    else:
        ds = ufl.ds(domain=mesh)
    dx_conductors = measure_over(dx, DomainTags3D.conducting())
    dx_magnets = measure_over(dx, DomainTags3D.MAGNETS)
    
    sigma, nu, density = setup_materials(mesh, ct, config)
    
    import basix.ufl
    nedelec = basix.ufl.element("N1curl", mesh.basix_cell(), config.degree_A)
    lagrange = basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_V)
    A_space = fem.functionspace(mesh, nedelec)
    V_space = fem.functionspace(mesh, lagrange)
    A_prev = fem.Function(A_space, name="A_prev")
    
    bc_A, bc_V, block_bcs = setup_boundary_conditions(mesh, ft, A_space, V_space)
    bc_ground = make_ground_bc_V(mesh, V_space, ct, DomainTags3D.conducting())
    block_bcs[1].append(bc_ground)
    
    J_z, M_vec = setup_sources(mesh, ct)
    initialise_magnetisation(mesh, ct, M_vec, config)
    
    # Build forms (only once, alpha_spd_factor is in config)
    a_blocks, L_blocks, a00_spd_form, a00_motional_form = build_forms(
        mesh, A_space, V_space, sigma, nu, J_z, M_vec,
        A_prev, dx, dx_conductors, dx_magnets, ds, config,
        exterior_facet_tag=EXTERIOR_FACET_TAG
    )
    
    results = []
    current_time = config.dt
    
    for beta_pc in beta_pc_values:
        config.beta_pc = beta_pc
        print(f"\n{'='*80}")
        print(f"Testing: beta_pc = {beta_pc}, alpha_spd_factor = {alpha_spd_factor}")
        print(f"{'='*80}\n")
        
        # Assemble with current beta_pc
        mat_blocks, mat_nest, _, A00_spd, A00_pc = assemble_system_matrix(
            mesh, a_blocks, block_bcs, a00_spd_form, a00_motional_form, beta_pc=beta_pc
        )
        
        # Configure solver
        ksp = configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd, A00_pc, config.degree_A)
        
        # Solve one step
        residual_history, reason, iterations, final_residual = solve_one_step_tuning(
            mesh, A_space, V_space, ct, config, ksp, mat_nest, a_blocks, L_blocks,
            block_bcs, sigma, J_z, M_vec, A_prev, dx, dx_magnets, current_time
        )
        
        # Map convergence reason
        reason_map = {
            -3: "DIVERGED_ITS",
            -9: "DIVERGED_NANORINF",
            -11: "DIVERGED_BREAKDOWN",
            1: "CONVERGED_RTOL",
            2: "CONVERGED_ATOL",
            3: "CONVERGED_ITS",
        }
        reason_str = reason_map.get(reason, f"UNKNOWN({reason})")
        
        results.append({
            'beta_pc': beta_pc,
            'alpha_spd_factor': alpha_spd_factor,
            'reason': reason_str,
            'iterations': iterations,
            'final_residual': final_residual,
            'residual_history': residual_history
        })
        
        if mesh.comm.rank == 0:
            print(f"\n[RESULT] beta_pc={beta_pc}, reason={reason_str}, its={iterations}, final_resid={final_residual:.6e}")
        
        # Clean up KSP for next iteration
        ksp.destroy()
        if A00_pc is not None:
            A00_pc.destroy()
        if A00_spd is not None:
            A00_spd.destroy()
        for row in mat_blocks:
            for mat in row:
                if mat is not None:
                    mat.destroy()
        mat_nest.destroy()
    
    # Print summary
    if mesh.comm.rank == 0:
        print("\n" + "="*80)
        print("TUNING SWEEP SUMMARY")
        print("="*80)
        for r in results:
            print(f"beta_pc={r['beta_pc']:.1f}: {r['reason']}, its={r['iterations']}, final_resid={r['final_residual']:.6e}")
        print("="*80)
        
        # Print full residual history for beta_pc=1.0
        print("\n" + "="*80)
        print("FULL RESIDUAL HISTORY FOR beta_pc=1.0, alpha_spd_factor=1.0")
        print("="*80)
        for r in results:
            if r['beta_pc'] == 1.0:
                print(f"\nIteration | Residual")
                print(f"----------|----------")
                for its, rnorm in r['residual_history']:
                    print(f"{its:9d} | {rnorm:.6e}")
                break
        print("="*80)


if __name__ == "__main__":
    # Run sweep with alpha_spd_factor=1.0 and beta_pc values [0.3, 0.5, 0.7, 1.0]
    run_tuning_sweep(beta_pc_values=[0.3, 0.5, 0.7, 1.0], alpha_spd_factor=1.0)

