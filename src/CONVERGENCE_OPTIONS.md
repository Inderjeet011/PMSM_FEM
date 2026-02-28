# Solver convergence options

Summary of what was tried and what can be done to improve convergence across the codebase.

---

## Progress: where we were → what we achieved (3d_submesh)

| Metric | Before (iterative) | After fixes |
|--------|--------------------|-------------|
| **Max \|B\|** | ~10⁻⁷–10⁻¹⁵ T (wrong) | ~2.8–3.4 T (correct) |
| **Residual \|\|b−Ax\|\|** | Large / no convergence | ~0.5–1.5 per step |
| **Relative residual** | ~1 or N/A | ~2×10⁻⁵ |
| **Outer iterations** | Many / no convergence | ~3 per step |
| **Solver** | Only LU gave correct B | Iterative (Schur + AMS + LU on V) works |

**Reference:** LU (monolithic direct solve) gave correct B ~2.7 T and confirmed the formulation; the iterative setup was wrong, not the physics.

---

## Changes made to fix convergence (3d_submesh)

### 1. Formulation fixes (essential)

| Change | Description |
|--------|-------------|
| **A–V coupling sign** | `a10 = -sigma * inner(grad(q), A)` (correct for S = dt·V) |
| **Coil mass term** | Added `(sigma/dt)*A·v` on `dx_cond_parent` so iterative solver converges |
| **RHS lifting** | `petsc.apply_lifting` for both A and V blocks with Dirichlet BCs |
| **A_prev init** | `A_prev.x.array[:] = 0.0` before first time step |
| **V grounding** | One DOF per conductor marker grounded to remove V nullspace |

### 2. Interior nodes (AMS preconditioner)

| Change | Description |
|--------|-------------|
| **setHYPREAMSSetInteriorNodes** | Marks nodes for AMS: 1.0 = exterior (non-conductor), 0.0 = interior (conductor) |
| **Implementation** | Build scalar function on V_ams; set 0 on DOFs belonging to conductor cells (coils, magnets, rotor) |
| **When used** | When `cell_tags_parent` and `conductor_markers` are provided |

### 3. Solver structure

| Component | Setting |
|-----------|---------|
| **Outer** | FGMRES, unpreconditioned norm, `outer_atol` for ||Ax−b|| |
| **Preconditioner** | Schur fieldsplit, LOWER fact, SELFP Schur pre |
| **A-block** | FGMRES + AMS on A00_full (no regularization) |
| **V-block** | preonly + LU (MUMPS) |
| **AMS** | Uses A00_full (not A00_spd), no gauge regularization |

### 4. Iteration / tolerance changes tried

| Parameter | Original | Tried values |
|-----------|----------|--------------|
| **outer_atol** | 0.5 | 0.1, 0.05 |
| **outer_max_it** | 100 | 200, 300 |
| **outer_restart** | 150 | 200 |
| **ksp_A_max_it** | 8 | 12, 20, 25 |
| **ksp_A_restart** | 35 | 40, 45, 50 |
| **ksp_A_rtol** | 2e-2 | 1e-2, 5e-3 |
| **schur_fact_type** | lower | full |

### 5. AMS preconditioner tuning tried

| Parameter | Original | Tried |
|-----------|----------|-------|
| **pc_hypre_ams_max_iter** | 1 | 2, 3 |
| **pc_hypre_ams_tol** | 0 | 1e-4, 1e-6 |
| **pc_hypre_ams_print_level** | 1 | 0 (to reduce output) |
| **pc_hypre_ams_projection_frequency** | 50 | (unchanged) |
| **pc_hypre_ams_cycle_type** | 13 | (unchanged) |

### 6. Conducting region / materials

| Change | Description |
|--------|-------------|
| **conducting()** | Added ALUMINIUM: `ROTOR + ALUMINIUM + MAGNETS + COILS` (matches full 3d) |
| **sigma_al_override** | Set to `None` when including ALUMINIUM (use model σ=3.72e7); `0.0` causes singular system → NaN |

### 7. Direct LU option

| Change | Description |
|--------|-------------|
| **use_direct** | Config option: `True` = monolithic LU (MUMPS) on full system → near-zero residual |

### 8. Removed (cleanup)

- Regularization: `epsilon_A`, `epsilon_A_spd`, `gauge_alpha`, `sigma_air_min`
- `A00_spd`, `use_schur` branch
- `interpolation_data`, `sigma_submesh`, `L_block_form`, `density`
- Voltage path (current source only)

---

## Current behaviour (3d_submesh after fixes)

- **Coupled A–V** (`src/3d_submesh/`): Schur (FULL) + AMS + LU on V; converges in a few outer iterations; \|\|b−Ax\|\| ~0.5–1.5, rel_res ~2×10⁻⁵, Max \|B\| ~2.8–3.4 T.
- **3d (full mesh)**: `src/3d/` — outer KSP can hit max iterations; rel_res ~1e-3–1e-4.
- **Direct LU**: `use_direct=True` in config gives near-zero residual (for verification).

**RHS div-free:** Unprojected RHS has ||G^T b_A|| ≠ 0 (e.g. ~1.3e3). Optional projection (`project_rhs_div_free=True`) makes ||G^T b_A|| = 0; with current assembly the whole A-block RHS is in range(G), so the projected RHS is zero and the solution is B = 0. Default is no projection (non-zero B, ~3.4 T).

---

## Why the coupled submesh solver often doesn’t converge

1. **Schur preconditioner**  
   The fieldsplit PC uses the Schur complement **S = A11 − A10·A00⁻¹·A01**. We approximate it by **A11** only (`schur_pre_type="a11"`). When the coupling blocks A01, A10 are strong, **S** is very different from A11, so the preconditioner is poor and the outer GMRES makes little progress (e.g. rel_res stays ~1).  
   **Try:** Set `schur_pre_type="full"` in config to use the exact Schur complement (expensive, for testing). If the solver then converges, the bottleneck is the A11 approximation.

2. **Formulation alignment with 3d**  
   The submesh coupling forms (a01, a10, a11, L1) are now aligned with the 3d solver: same scaling (σ, σ/dt), same a10 rotation term on `dx_rpm`, and L1 with (σ/dt)A_prev on `dx_rs`. Previously, different scaling (e.g. extra dt in a01/a11, missing inv_dt in a10/L1) could make the block system ill-conditioned or inconsistent.

3. **Other levers**  
   See below: more inner A-block iterations (`ksp_A_max_it`), **additive** fieldsplit (`field_split_type="additive"`), or a direct solve on the full system to confirm the linear system is solvable.

---

## Options to try (by area)

### 1. Inner A-block iterations (fieldsplit Schur)

The A-block in the Schur preconditioner is solved approximately with a small number of inner iterations.

- **3d**: `solve_equations.py` uses `ksp_A_max_it` default **2**; `solver_utils.py` sets 5. Low inner iterations can make the Schur complement approximation poor.
- **3d_submesh**: uses `ksp_A_max_it=5`.

**Try:** Increase `ksp_A_max_it` to **15–30** in config (both 3d and 3d_submesh).

### 2. Outer KSP norm type

Convergence is checked using a residual norm. If the preconditioned norm is used, the printed rel_res (true residual) can disagree with PETSc’s convergence decision.

**Try:** Set `ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)` for the **outer** KSP (coupled and decoupled) so convergence matches the printed ||b−Ax||/||b||.

### 3. Fieldsplit type: Schur vs additive; Schur preconditioner

- **Current (3d, 3d_submesh):** Schur complement (LOWER, A11) — good when the V-block is cheap and the Schur complement is well approximated.
- **arshads_code/mixed_domain_AV.py:** Uses **ADDITIVE** fieldsplit: both blocks preconditioned separately and added; outer KSP sees the full system with a block-diagonal-style PC.

**Try:**  
- `field_split_type = "additive"` when Schur fails.  
- For Schur runs, `schur_pre_type = "full"` (exact Schur) to test if A11 approximation is the bottleneck; config in `solver_utils_submesh.py`.

### 4. AMS preconditioner tuning

- **projection_frequency:** We use 25; `arshads_code` uses **50**; `maxwell_curl_curl` uses **5**. Worth trying 5 and 50.
- **pc_hypre_ams_cycle_type:** We use 13; `team30` uses 1. Try 1 or 7 (`maxwell_curl_curl`).
- **setHYPRESetBetaPoissonMatrix(None):** Used in `arshads_code/uncoupled.py` and commented in `mixed_domain_AV.py`. Can change the AMS hierarchy; try if available in your HYPRE/DOLFINx build.

### 5. Decoupled A-only: preconditioner matrix and AMS

- **With BCs on A00_spd:** We apply the same BCs to the SPD preconditioner matrix so it matches the system. That makes the PC matrix non-symmetric; AMS may expect a symmetric operator.
- **Try:** For decoupled AMS, **do not** apply BCs to `A00_spd` (keep it symmetric). Ensure the outer KSP applies BCs to the residual/initial guess so the iteration stays in the correct subspace.
- **Decoupled with AMS:** Residual was flat; ILU at least reduced it to ~0.18. So either fix AMS (e.g. interior nodes, edge vectors, projection_frequency) or keep ILU and relax outer_rtol to ~0.2 to get “convergence” for testing.

### 6. Relax outer tolerance temporarily

To get convergence (reason 2) while tuning:

- Set `outer_rtol=1e-3` or `1e-2` so the solver declares convergence earlier.
- Then tighten again once the preconditioner is improved.

### 7. Initial guess (time stepping)

Using the previous time step as the initial guess can reduce iterations.

**Try:** For step > 1, set the solution vector to (A_prev, V_prev) before calling `ksp.solve(b, x)`. Requires passing the previous solution into the solve step and copying into the PETSc vec.

### 8. GMRES restart

Larger restart can improve convergence at the cost of memory and work per iteration.

- **arshads_code:** GMRES restart **100**; we use **30**.
- **Try:** Increase `ksp_A_restart` and outer GMRES restart (if set) to 50–100.

### 9. Direct solve (for debugging)

To check that the linear system itself is solvable:

- Use `pc_type=lu` and `pc_factor_mat_solver_type=mumps` (or another direct solver) for the **outer** KSP or for both blocks. If the direct solve works, the issue is preconditioner/iterative setup.

### 10. 3d normal solver: inner A-block

`src/3d/solve_equations.py` line 176: default `ksp_A_max_it` is **2** (overridden by config to 5 in solver_utils). Consider raising the default to 10 so that configs without explicit `ksp_A_max_it` still get reasonable inner iterations.

---

## Files to edit

| Goal | File(s) |
|------|--------|
| Inner A-block iterations | `3d/solver_utils.py`, `3d_submesh/solver_utils_submesh.py` — increase `ksp_A_max_it` (e.g. 15). |
| Outer norm type | `3d_submesh/solve_equations_submesh.py` — `configure_solver_submesh`, `configure_solver_decoupled`: call `ksp.setNormType(UNPRECONDITIONED)`. |
| AMS projection_frequency | `3d_submesh/solve_equations_submesh.py`, `3d/solve_equations.py` — set 50 (or try 5). |
| ADDITIVE fieldsplit | `3d_submesh/solve_equations_submesh.py` — branch on `config.field_split_type` and set `pc.setFieldSplitType(ADDITIVE)` (no Schur options). |
| Schur pre FULL | `solver_utils_submesh.py` — set `schur_pre_type="full"`; in `solve_equations_submesh.py` `configure_solver_submesh` already respects it. |
| Decoupled SPD without BCs | `3d_submesh/solve_equations_submesh.py` — `assemble_system_decoupled`: when using AMS, optionally assemble `A00_spd` without BCs. |
| Default ksp_A_max_it in 3d | `3d/solve_equations.py` — change default from 2 to 10. |

---

## References in repo

- **arshads_code/mixed_domain_AV.py**: ADDITIVE fieldsplit, GMRES restart 100, rtol/atol 1e-14, AMS preonly, projection_frequency 50, setHYPRESetInterpolations for degree>1.
- **arshads_code/steady_state.py**, **mixed_domain_steady.py**: Same AMS/boomeramg pattern, SCHUR not used.
- **maxwell_curl_curl/interior_nodes.py**: AMS options dict, projection_frequency 5, ksp_monitor_true_residual.
- **team30/solve_3D_time.py**: ksp_initial_guess_nonzero, ksp_norm_type unpreconditioned, pc_hypre_ams_cycle_type 1.
