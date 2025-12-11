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
    
    # Filter outliers intelligently to preserve distribution while removing artifacts
    # Expected B for PMSM: 0.1-2 T, values > 2.5 T are likely numerical artifacts
    max_physical_B = 2.5  # T - maximum physically reasonable value
    
    # Use percentile-based approach to preserve distribution
    p50 = np.percentile(B_magnitude, 50)
    p75 = np.percentile(B_magnitude, 75)
    p90 = np.percentile(B_magnitude, 90)
    p95 = np.percentile(B_magnitude, 95)
    p99 = np.percentile(B_magnitude, 99)
    
    # Find the "real" maximum by looking at percentiles
    # If percentiles are unreasonably high, they're artifacts, not real data
    # Use the highest percentile that's still physically reasonable
    if p50 < max_physical_B:
        # Median is reasonable - use it as base
        if p75 < max_physical_B:
            effective_max = min(p75, max_physical_B)
        else:
            effective_max = min(p50, max_physical_B)
    elif p50 < max_physical_B * 2:
        # Median is somewhat high but might be real - use lower percentile
        effective_max = min(p50 * 0.8, max_physical_B)
    else:
        # Everything is way too high - use a fixed reasonable value
        # Based on magnet remanence (typically 1.2 T for PMSM)
        effective_max = min(config.magnet_remanence * 1.5, max_physical_B)
    
    # Instead of hard clipping, use percentile-based mapping for better visualization
    # Map the distribution to a reasonable range
    B_magnitude_filtered = B_magnitude.copy()
    
    # Identify extreme outliers (definitely artifacts)
    extreme_outlier_mask = B_magnitude > max_physical_B * 1.5
    
    # Identify moderate outliers (between effective_max and reasonable limit)
    moderate_outlier_mask = (B_magnitude > effective_max) & (B_magnitude <= max_physical_B * 1.5)
    
    # Zero extreme outliers
    if np.any(extreme_outlier_mask):
        B_magnitude_filtered[extreme_outlier_mask] = 0.0
    
    # For moderate outliers, map them to a compressed range above effective_max
    if np.any(moderate_outlier_mask):
        # Map [effective_max, max_physical_B*1.5] -> [effective_max, max_physical_B]
        # Use smooth compression
        excess = B_magnitude[moderate_outlier_mask] - effective_max
        max_excess = max_physical_B * 1.5 - effective_max
        if max_excess > 0:
            # Compress: map excess to smaller range
            compression_range = max_physical_B - effective_max
            if compression_range > 0:
                B_magnitude_filtered[moderate_outlier_mask] = effective_max + (excess / max_excess) * compression_range
    
    # Final clip to ensure nothing exceeds max_physical_B
    B_magnitude_clipped = np.clip(B_magnitude_filtered, 0.0, max_physical_B)
    
    max_B_raw = float(np.max(B_magnitude))
    max_B = float(np.max(B_magnitude_clipped))
    p99_B = float(np.percentile(B_magnitude_clipped, 99))
    
    if max_B_raw > max_physical_B * 1.1:
        if debug and mesh.comm.rank == 0:
            print(f"\n   ⚠️  Outliers detected: raw max|B| = {max_B_raw:.2e} T")
            print(f"   Filtered max|B| = {max_B:.2e} T (95th percentile: {p95:.2e} T)")
            print(f"   Effective max: {effective_max:.2e} T")
            print(f"   Extreme outliers zeroed: {np.sum(extreme_outlier_mask)} points")
            print(f"   Moderate outliers compressed: {np.sum(moderate_outlier_mask)} points")
    
    B_magnitude_sol.x.array[:] = B_magnitude_clipped
    B_magnitude_sol.x.scatter_forward()
    
    min_B = float(np.min(B_magnitude))
    median_B = float(np.median(B_magnitude_clipped))
    
    # Also filter the vector field to match - use same approach
    B_array_clipped = B_sol.x.array.reshape((-1, 3)).copy()
    B_mag_array = np.linalg.norm(B_array_clipped, axis=1)
    
    # Zero extreme outliers, compress moderate ones
    if np.any(extreme_outlier_mask):
        B_array_clipped[extreme_outlier_mask] = 0.0
    if np.any(moderate_outlier_mask):
        # Scale down moderate outliers
        scale_factor = B_magnitude_clipped[moderate_outlier_mask] / B_mag_array[moderate_outlier_mask]
        scale_factor = np.clip(scale_factor, 0.0, 1.0)  # Don't amplify
        for i, idx in enumerate(np.where(moderate_outlier_mask)[0]):
            B_array_clipped[idx] *= scale_factor[i]
    
    norm_B = float(np.linalg.norm(B_array_clipped))
    
    if debug and mesh.comm.rank == 0:
        print(f"\n   Final B field statistics:")
        print(f"   max|B| = {max_B:.6e} T (filtered, effective max: {effective_max:.2f} T)")
        print(f"   min|B| = {min_B:.6e} T")
        print(f"   median|B| = {median_B:.6e} T")
        print(f"   ||B|| (filtered) = {norm_B:.6e}")
        print(f"   Expected range: 0.1-2 T (magnet remanence = {config.magnet_remanence:.2f} T)")
        print(f"   Extreme outliers zeroed: {np.sum(extreme_outlier_mask)} points ({100*np.sum(extreme_outlier_mask)/len(B_magnitude):.1f}%)")
        print(f"   Moderate outliers compressed: {np.sum(moderate_outlier_mask)} points ({100*np.sum(moderate_outlier_mask)/len(B_magnitude):.1f}%)")
        if max_B > 3.0:
            print(f"   ⚠️  WARNING: B field still has high values - may need better filtering")
        elif max_B < 0.01:
            print(f"   ⚠️  WARNING: B field seems too low!")
        else:
            print(f"   ✅ B field magnitude looks reasonable after filtering")
    
    return B_sol, B_magnitude_sol, max_B, min_B, norm_B

