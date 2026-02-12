# Solver convergence options

Summary of what was tried and what can be done to improve convergence across the codebase.

## Current behaviour

- **3d (normal A–V)**: `src/3d/` — outer KSP hits max iterations (reason -3); rel_res ~ 1e-3–1e-4.
- **3d_submesh (coupled A–V)**: `src/3d_submesh/` with `use_coupling=True` — same: max iterations, rel_res ~ 9e-4.
- **3d_submesh (decoupled A-only)**: `use_coupling=False`, `decoupled_pc="ilu"` — residual stalls ~0.18; with AMS residual is flat (no decrease).

RHS is **discrete divergence-free** (||G^T b|| = 0), so the system is compatible.

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
