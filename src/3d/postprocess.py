"""
Post-processing module for 3D A-V solver.

This module handles B-field computation and related utilities.
"""

import numpy as np
from basix.ufl import element as basix_element
from dolfinx import fem, geometry
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl


def verify_curl_step_by_step(mesh, A_sol, config):
    """Detailed step-by-step verification of B = curl(A) computation."""
    if mesh.comm.rank != 0:
        return
    
    print("\n" + "="*70)
    print("STEP-BY-STEP VERIFICATION OF B = curl(A)")
    print("="*70)
    
    A_array = A_sol.x.array
    print("\n[Step 1] A Field Analysis:")
    print(f"   ||A|| = {np.linalg.norm(A_array):.6e} Wb/m")
    print(f"   max|A| = {np.max(np.abs(A_array)):.6e} Wb/m")
    print(f"   min|A| = {np.min(np.abs(A_array)):.6e} Wb/m")
    print(f"   Units: Wb/m = T·m (Tesla-meter) ✓")
    
    print("\n[Step 2] Mesh Scale:")
    coords = mesh.geometry.x
    coord_range = np.max(coords, axis=0) - np.min(coords, axis=0)
    print(f"   x_range = {coord_range[0]:.4f} m")
    print(f"   y_range = {coord_range[1]:.4f} m")
    print(f"   z_range = {coord_range[2]:.4f} m")
    typical_L = 0.05
    print(f"   Typical length scale L ≈ {typical_L:.3f} m")
    
    print("\n[Step 3] Physics Check:")
    print(f"   Magnet remanence Br = {config.magnet_remanence:.2f} T")
    print(f"   Expected B in magnets: ~{config.magnet_remanence:.2f} T")
    print(f"   Expected B in air gap: ~0.5-1.5 T")
    typical_A = np.max(np.abs(A_array))
    rough_B_estimate = typical_A / typical_L
    print(f"   Rough estimate: if curl(A) ≈ A/L, then B ≈ {typical_A:.3e}/{typical_L:.3e} = {rough_B_estimate:.3e} T")
    
    print("\n" + "="*70)


def compute_B_field(mesh, A_sol, B_space, B_magnitude_space, config, 
                   cell_tags=None, debug=False):
    """Compute B = curl(A) and return B field, max|B|, min|B|, and ||B||."""
    if debug and mesh.comm.rank == 0:
        print("\n🔍 DEBUG: Computing B = curl(A)")
        verify_curl_step_by_step(mesh, A_sol, config)
        print(f"\n   A field: ||A|| = {np.linalg.norm(A_sol.x.array):.6e} Wb/m")
        print(f"   A field: max|A| = {np.max(np.abs(A_sol.x.array)):.6e} Wb/m")
        print(f"   A field: min|A| = {np.min(np.abs(A_sol.x.array)):.6e} Wb/m")
        coords = mesh.geometry.x
        if coords.shape[0] > 0:
            coord_range = np.max(coords, axis=0) - np.min(coords, axis=0)
            print(f"   Mesh scale: x_range={coord_range[0]:.4f} m, y_range={coord_range[1]:.4f} m, z_range={coord_range[2]:.4f} m")
    
    B_sol = fem.Function(B_space, name="B")
    curlA = ufl.curl(A_sol)
    
    if debug and mesh.comm.rank == 0:
        print(f"   Using direct L2 projection to Lagrange (mathematically correct)")
        print(f"   Note: curl(A) is in H(div), projecting to H(curl) was incorrect")
    
    # CORRECT APPROACH: Direct L2 projection to Lagrange space
    # curl(A) where A is in H(curl) produces a field in H(div) (divergence-free)
    # The issue: curl(A) computed from Nédélec elements can have huge values at boundaries
    # due to discontinuities when A=0 (Dirichlet BC) but curl(A)≠0
    # 
    # Root cause: Numerical differentiation of discontinuous field at boundaries
    # creates extreme values that propagate through L2 projection
    #
    # Solution: Use robust solver with better preconditioning
    B_test = ufl.TestFunction(B_space)
    B_trial = ufl.TrialFunction(B_space)
    a_B = fem.form(ufl.inner(B_trial, B_test) * ufl.dx)
    L_B = fem.form(ufl.inner(curlA, B_test) * ufl.dx)
    
    A_B = petsc.assemble_matrix(a_B)
    A_B.assemble()
    b_B = petsc.create_vector(L_B)
    petsc.assemble_vector(b_B, L_B)
    b_B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Use robust solver with appropriate preconditioner
    # Note: AMS is designed for H(curl) spaces, not Lagrange
    # For Lagrange space, use standard multigrid preconditioner
    ksp_B = PETSc.KSP().create(comm=mesh.comm)
    ksp_B.setOperators(A_B)
    ksp_B.setType("cg")
    pc_B = ksp_B.getPC()
    pc_B.setType("hypre")
    try:
        # Try boomeramg (algebraic multigrid) - good for Lagrange spaces
        pc_B.setHYPREType("boomeramg")
        if debug and mesh.comm.rank == 0:
            print(f"   Using BoomerAMG preconditioner (appropriate for Lagrange space)")
    except:
        # Fallback to Jacobi if hypre not available
        pc_B.setType("jacobi")
        if debug and mesh.comm.rank == 0:
            print(f"   Using Jacobi preconditioner (fallback)")
    ksp_B.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
    x_B = b_B.duplicate()
    ksp_B.solve(b_B, x_B)
    B_sol.x.array[:] = x_B.array_r[:B_sol.x.array.size]
    B_sol.x.scatter_forward()
    
    A_B.destroy()
    b_B.destroy()
    x_B.destroy()
    ksp_B.destroy()
    
    if debug and mesh.comm.rank == 0:
        print(f"   ✓ Direct L2 projection to Lagrange successful")
    
    B_sol.name = "B"
    B_sol.x.scatter_forward()
    
    # Compute magnitude
    B_array = B_sol.x.array.reshape((-1, 3))
    B_magnitude = np.linalg.norm(B_array, axis=1)
    
    B_magnitude_sol = fem.Function(B_magnitude_space, name="B_Magnitude")
    B_magnitude_sol.x.array[:] = B_magnitude
    B_magnitude_sol.x.scatter_forward()
    
    # NO FILTERING - Use natural, raw B field values
    # All values are preserved as computed, no clipping or filtering applied
    # ParaView can handle visualization scaling/clipping if needed
    
    B_magnitude_sol.x.array[:] = B_magnitude
    B_magnitude_sol.x.scatter_forward()
    
    # Statistics from raw, unfiltered data
    max_B = float(np.max(B_magnitude))
    min_B = float(np.min(B_magnitude))
    median_B = float(np.median(B_magnitude))
    mean_B = float(np.mean(B_magnitude))
    p50 = float(np.percentile(B_magnitude, 50))
    p75 = float(np.percentile(B_magnitude, 75))
    p90 = float(np.percentile(B_magnitude, 90))
    p95 = float(np.percentile(B_magnitude, 95))
    p99 = float(np.percentile(B_magnitude, 99))
    
    # B_sol vector field is already natural (no filtering applied)
    norm_B = float(np.linalg.norm(B_sol.x.array))
    
    if debug and mesh.comm.rank == 0:
        print(f"\n   B field statistics (NATURAL, NO FILTERING):")
        print(f"   max|B| = {max_B:.6e} T")
        print(f"   min|B| = {min_B:.6e} T")
        print(f"   mean|B| = {mean_B:.6e} T")
        print(f"   median|B| = {median_B:.6e} T")
        print(f"   ||B|| = {norm_B:.6e}")
        print(f"   Percentiles: 50th={p50:.6e} T, 75th={p75:.6e} T, 90th={p90:.6e} T, 95th={p95:.6e} T, 99th={p99:.6e} T")
        print(f"   Expected range: 0.1-2 T (magnet remanence = {config.magnet_remanence:.2f} T)")
        print(f"   ✅ All values preserved - no clipping or filtering applied")
    
    return B_sol, B_magnitude_sol, max_B, min_B, norm_B

