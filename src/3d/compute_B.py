"""Post-processing: compute B field from A."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
from load_mesh import DomainTags3D

# Air gap tags
AIR_GAP = (2, 3)

# Motor-region tags: everything except the outer air box
MOTOR_TAGS = (
    DomainTags3D.ROTOR
    + DomainTags3D.STATOR
    + DomainTags3D.COILS
    + DomainTags3D.MAGNETS
    + DomainTags3D.ALUMINIUM
    + DomainTags3D.AIR_GAP
)


def compute_B_field(
    mesh,
    A_sol,
    B_space,
    B_magnitude_space,
    config,
    cell_tags=None,
    debug=False,
    restrict_to_airgap=False,
    restrict_to_motor=False,
):
    """Compute B = curl(A)."""
    if debug and mesh.comm.rank == 0:
        print("\n" + "="*70)
        print("DEBUG: Computing B = curl(A) - Step by step analysis")
        print("="*70)
        
        # Check A field
        A_array = A_sol.x.array
        print(f"\n[Step 1] A field analysis:")
        print(f"   ||A|| = {np.linalg.norm(A_array):.6e} Wb/m")
        print(f"   max|A| = {np.max(np.abs(A_array)):.6e} Wb/m")
        print(f"   min|A| = {np.min(np.abs(A_array)):.6e} Wb/m")
        print(f"   mean|A| = {np.mean(np.abs(A_array)):.6e} Wb/m")
        
        # Check mesh scale
        coords = mesh.geometry.x
        coord_range = np.max(coords, axis=0) - np.min(coords, axis=0)
        print(f"\n[Step 2] Mesh scale:")
        print(f"   x_range = {coord_range[0]:.4f} m")
        print(f"   y_range = {coord_range[1]:.4f} m")
        print(f"   z_range = {coord_range[2]:.4f} m")
        typical_L = np.mean(coord_range[:2])  # Typical length scale
        print(f"   Typical length scale L ≈ {typical_L:.4f} m")
        
        # Rough estimate
        max_A = np.max(np.abs(A_array))
        rough_B_estimate = max_A / typical_L if typical_L > 0 else 0
        print(f"\n[Step 3] Rough physics check:")
        print(f"   If curl(A) ≈ A/L, then B ≈ {max_A:.3e}/{typical_L:.4f} = {rough_B_estimate:.3e} T")
        print(f"   Expected B range: 0.1-2 T (magnet remanence = {config.magnet_remanence:.2f} T)")
    
    # Use DG space for B computation - handles discontinuities better.
    # Natural choice: DG0 vector field (piecewise constant per cell).
    DG_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    B_dg = fem.Function(DG_vec, name="B_dg")
    curlA = ufl.curl(A_sol)
    
    if debug and mesh.comm.rank == 0:
        print(f"\n[Step 4] Using DG space for B computation (handles discontinuities)...")
        # Try to evaluate curl(A) at a few points to see its magnitude
        try:
            # Create a test function to evaluate curl(A)
            curlA_func = fem.Function(DG_vec)
            curlA_func.interpolate(curlA)
            curlA_array = curlA_func.x.array.reshape((-1, 3))
            curlA_mag = np.linalg.norm(curlA_array, axis=1)
            print(f"   curl(A) in DG space: max|curl(A)| = {np.max(curlA_mag):.6e}")
            print(f"   curl(A) in DG space: mean|curl(A)| = {np.mean(curlA_mag):.6e}")
            print(f"   curl(A) in DG space: ||curl(A)|| = {np.linalg.norm(curlA_array):.6e}")
        except Exception as e:
            if debug and mesh.comm.rank == 0:
                print(f"   Could not evaluate curl(A) directly: {e}")
    
    # Project curl(A) to DG space using L2 projection
    # ALWAYS compute on full mesh for correct physics, then mask if needed
    B_test_dg = ufl.TestFunction(DG_vec)
    B_trial_dg = ufl.TrialFunction(DG_vec)
    
    # Compute B on full mesh (correct physics)
    a_B_dg = fem.form(ufl.inner(B_trial_dg, B_test_dg) * ufl.dx)
    L_B_dg = fem.form(ufl.inner(curlA, B_test_dg) * ufl.dx)
    
    A_B_dg = petsc.assemble_matrix(a_B_dg)
    A_B_dg.assemble()
    b_B_dg = petsc.create_vector(L_B_dg)
    petsc.assemble_vector(b_B_dg, L_B_dg)
    b_B_dg.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    if debug and mesh.comm.rank == 0:
        with b_B_dg.localForm() as b_local:
            b_array = b_local.array_r
            print(f"   RHS vector: ||b_B_dg|| = {np.linalg.norm(b_array):.6e}")
            print(f"   max|b_B_dg| = {np.max(np.abs(b_array)):.6e}")
    
    # Solve - DG space is diagonal, so this should be fast and stable
    ksp_B_dg = PETSc.KSP().create(comm=mesh.comm)
    ksp_B_dg.setOperators(A_B_dg)
    ksp_B_dg.setType("preonly")  # Direct solve since matrix is diagonal
    pc_B_dg = ksp_B_dg.getPC()
    pc_B_dg.setType("jacobi")  # Diagonal matrix, Jacobi is exact
    x_B_dg = b_B_dg.duplicate()
    ksp_B_dg.solve(b_B_dg, x_B_dg)
    
    B_dg.x.array[:] = x_B_dg.array_r[:B_dg.x.array.size]
    B_dg.x.scatter_forward()
    
    # Mask B_dg to zero outside requested region (after computing on full mesh)
    if (restrict_to_airgap or restrict_to_motor) and cell_tags is not None:
        B_dg_array = B_dg.x.array.reshape((-1, 3))
        tdim = mesh.topology.dim
        num_cells = mesh.topology.index_map(tdim).size_local
        
        if restrict_to_airgap:
            # Get airgap cells
            target_cells = set()
            for tag in AIR_GAP:
                cells = cell_tags.find(tag)
                target_cells.update(cells.tolist())
            if mesh.comm.rank == 0:
                print(f"[MASK] Masking B_dg to airgap: {len(target_cells)} cells")
        elif restrict_to_motor:
            # Get motor cells
            target_cells = set()
            for tag in MOTOR_TAGS:
                cells = cell_tags.find(tag)
                target_cells.update(cells.tolist())
            if mesh.comm.rank == 0:
                print(f"[MASK] Masking B_dg to motor: {len(target_cells)} cells")
        
        # Zero B_dg for cells NOT in target region
        for cell_idx in range(num_cells):
            if cell_idx not in target_cells:
                B_dg_array[cell_idx, :] = 0.0
        
        B_dg.x.array[:] = B_dg_array.flatten()
        B_dg.x.scatter_forward()
    
    if debug and mesh.comm.rank == 0:
        iterations_B = ksp_B_dg.getIterationNumber()
        with x_B_dg.localForm() as x_local:
            x_array = x_local.array_r
            print(f"\n[Step 5] DG projection solve:")
            print(f"   Iterations: {iterations_B}")
            print(f"   ||B_dg|| = {np.linalg.norm(x_array):.6e}")
            print(f"   max|B_dg| = {np.max(np.abs(x_array)):.6e}")
            
            # If restricted to air gap, check values only in air gap cells
            if restrict_to_airgap and cell_tags is not None:
                B_dg_array = B_dg.x.array.reshape((-1, 3))
                B_dg_mag = np.linalg.norm(B_dg_array, axis=1)
                # Find air gap cells
                airgap_cells = []
                for tag in AIR_GAP:
                    cells = cell_tags.find(tag)
                    airgap_cells.extend(cells.tolist())
                if airgap_cells:
                    airgap_cells = np.array(airgap_cells)
                    airgap_B = B_dg_mag[airgap_cells]
                    print(f"\n   Air gap only (restricted region):")
                    print(f"   Air gap cells: {len(airgap_cells)}")
                    print(f"   max|B| in air gap = {np.max(airgap_B):.6e} T")
                    print(f"   mean|B| in air gap = {np.mean(airgap_B):.6e} T")
                    print(f"   median|B| in air gap = {np.median(airgap_B):.6e} T")
    
    # ------------------------------------------------------------------
    # Region-wise diagnostics in the natural DG0 space (cell-wise B)
    # ------------------------------------------------------------------
    if cell_tags is not None:
        # B_dg has one vector value per cell in DG0.
        B_dg_array = B_dg.x.array.reshape((-1, 3))
        B_dg_mag = np.linalg.norm(B_dg_array, axis=1)

        # Define material/region groups
        region_defs = {
            "AirGap": DomainTags3D.AIR_GAP,
            "PM": DomainTags3D.MAGNETS,
            "Iron": DomainTags3D.ROTOR + DomainTags3D.STATOR,
            "Conductors": DomainTags3D.conducting(),
        }

        if mesh.comm.rank == 0:
            print("\n[B-REGION] Cell-wise DG0 B statistics by region:")

        for region_name, markers in region_defs.items():
            cell_ids = []
            for m in markers:
                cells_m = cell_tags.find(m)
                if cells_m.size > 0:
                    cell_ids.append(cells_m)
            if not cell_ids:
                continue
            cells = np.concatenate(cell_ids).astype(np.int64)
            vals = B_dg_mag[cells]
            if vals.size == 0:
                continue

            max_B_r = float(np.max(vals))
            mean_B_r = float(np.mean(vals))
            med_B_r = float(np.median(vals))
            p90 = float(np.percentile(vals, 90))
            p99 = float(np.percentile(vals, 99))
            frac_gt_10 = 100.0 * float(np.mean(vals > 10.0))
            frac_gt_100 = 100.0 * float(np.mean(vals > 100.0))

            if mesh.comm.rank == 0:
                print(
                    f"  - {region_name:10s}: "
                    f"Ncells={vals.size:6d}, "
                    f"max|B|={max_B_r:8.3e} T, "
                    f"mean|B|={mean_B_r:8.3e} T, "
                    f"median|B|={med_B_r:8.3e} T, "
                    f"P90={p90:8.3e} T, P99={p99:8.3e} T, "
                    f">%10T={frac_gt_10:6.2f}%, >100T={frac_gt_100:6.2f}%"
                )

    # ------------------------------------------------------------------
    # Optional smoothing / projection for visualization
    # ------------------------------------------------------------------
    if debug and mesh.comm.rank == 0:
        print(f"\n[Step 6] Interpolating from DG to Lagrange for visualization...")
    
    # For visualization: interpolate from masked B_dg to nodal space
    # This ensures nodal fields match the cell data exactly (no interpolation artifacts)
    B_sol = fem.Function(B_space, name="B")
    
    if (restrict_to_airgap or restrict_to_motor) and cell_tags is not None:
        # Get target cells
        if restrict_to_airgap:
            target_cells = set()
            for tag in AIR_GAP:
                cells = cell_tags.find(tag)
                target_cells.update(cells.tolist())
        elif restrict_to_motor:
            target_cells = set()
            for tag in MOTOR_TAGS:
                cells = cell_tags.find(tag)
                target_cells.update(cells.tolist())
        
        # Zero B_sol first
        B_sol.x.array[:] = 0.0
        
        # Interpolate B_dg (cell-wise) to nodal space, but only in target cells
        # For each target cell, set nodal DOFs to the cell's B_dg value
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim, 0)
        c2v = mesh.topology.connectivity(tdim, 0)
        dofmap = B_space.dofmap
        imap = dofmap.index_map
        size_local = imap.size_local
        num_cells_local = mesh.topology.index_map(tdim).size_local
        
        B_dg_array = B_dg.x.array.reshape((-1, 3))
        B_sol_array = B_sol.x.array.reshape((-1, 3))
        
        kept_dofs = set()
        for cell_idx in target_cells:
            if cell_idx < num_cells_local:
                # Get B_dg value for this cell
                B_cell = B_dg_array[cell_idx, :]
                # Set all DOFs of this cell to B_cell
                cell_dofs = dofmap.cell_dofs(cell_idx)
                for dof in cell_dofs:
                    if dof < size_local:
                        B_sol_array[dof, :] = B_cell  # Use cell value directly
                        kept_dofs.add(int(dof))
        
        B_sol.x.array[:] = B_sol_array.flatten()
        B_sol.x.scatter_forward()
        
        if mesh.comm.rank == 0:
            region_name = "airgap" if restrict_to_airgap else "motor"
            zeroed = size_local - len(kept_dofs)
            print(f"[MASK] B_sol interpolated from masked B_dg ({region_name}): kept={len(kept_dofs)}, zeroed={zeroed}")
    else:
        # Full mesh: standard projection
        B_test = ufl.TestFunction(B_space)
        B_trial = ufl.TrialFunction(B_space)
        a_B = fem.form(ufl.inner(B_trial, B_test) * ufl.dx)
        L_B = fem.form(ufl.inner(B_dg, B_test) * ufl.dx)
        
        A_B = petsc.assemble_matrix(a_B)
        A_B.assemble()
        b_B = petsc.create_vector(L_B)
        petsc.assemble_vector(b_B, L_B)
        b_B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        ksp_B = PETSc.KSP().create(comm=mesh.comm)
        ksp_B.setOperators(A_B)
        ksp_B.setType("cg")
        pc_B = ksp_B.getPC()
        pc_B.setType("hypre")
        try:
            pc_B.setHYPREType("boomeramg")
        except:
            pc_B.setType("jacobi")
        
        ksp_B.setTolerances(rtol=1e-6, atol=1e-8, max_it=100)
        x_B = b_B.duplicate()
        ksp_B.solve(b_B, x_B)
        B_sol.x.array[:] = x_B.array_r[:B_sol.x.array.size]
        B_sol.x.scatter_forward()
        
        # Clean up
        A_B.destroy()
        b_B.destroy()
        x_B.destroy()
        ksp_B.destroy()
    
    # Clean up
    A_B_dg.destroy()
    b_B_dg.destroy()
    x_B_dg.destroy()
    ksp_B_dg.destroy()
    
    # Only clean up if we used projection (not interpolation)
    if not ((restrict_to_airgap or restrict_to_motor) and cell_tags is not None):
        A_B.destroy()
        b_B.destroy()
        x_B.destroy()
        ksp_B.destroy()
    
    B_sol.name = "B"
    
    # Compute magnitude
    B_array = B_sol.x.array.reshape((-1, 3))
    B_magnitude = np.linalg.norm(B_array, axis=1)
    
    if debug and mesh.comm.rank == 0:
        print(f"\n[Step 7] B field after projection to Lagrange:")
        print(f"   ||B|| = {np.linalg.norm(B_array):.6e} T·m")
        print(f"   max|B| = {np.max(B_magnitude):.6e} T")
        print(f"   min|B| = {np.min(B_magnitude):.6e} T")
        print(f"   mean|B| = {np.mean(B_magnitude):.6e} T")
        print(f"   median|B| = {np.median(B_magnitude):.6e} T")
        
        # Check percentiles to see distribution
        p50 = np.percentile(B_magnitude, 50)
        p75 = np.percentile(B_magnitude, 75)
        p90 = np.percentile(B_magnitude, 90)
        p95 = np.percentile(B_magnitude, 95)
        p99 = np.percentile(B_magnitude, 99)
        print(f"   Percentiles: 50th={p50:.3e} T, 75th={p75:.3e} T, 90th={p90:.3e} T, 95th={p95:.3e} T, 99th={p99:.3e} T")
        
        # Check for outliers
        outlier_mask = B_magnitude > 10.0  # Values > 10 T are likely artifacts
        num_outliers = np.sum(outlier_mask)
        if num_outliers > 0:
            print(f"\n[Step 8] OUTLIER DETECTION:")
            print(f"   ⚠️  Found {num_outliers} points with |B| > 10 T ({100*num_outliers/len(B_magnitude):.2f}%)")
            print(f"   Max outlier value: {np.max(B_magnitude[outlier_mask]):.6e} T")
            print(f"   These are likely numerical artifacts at boundaries/discontinuities")
        else:
            print(f"\n[Step 8] No extreme outliers detected (all values < 10 T)")
        
        print("="*70 + "\n")
    
    B_magnitude_sol = fem.Function(B_magnitude_space, name="B_Magnitude")
    B_magnitude_sol.x.array[:] = B_magnitude
    B_magnitude_sol.x.scatter_forward()
    
    # No clipping - use raw computed values
    max_B = float(np.max(B_magnitude))
    min_B = float(np.min(B_magnitude))
    norm_B = float(np.linalg.norm(B_array))
    
    if debug and mesh.comm.rank == 0:
        print(f"Final B field: max={max_B:.3e} T, min={min_B:.3e} T, ||B||={norm_B:.3e}")
    
    return B_sol, B_magnitude_sol, max_B, min_B, norm_B, B_dg
