# Convergence postmortem (3d_submesh A–V solver)

## What the residual numbers mean

- **Actual residual** = \|\|b − Ax\|\| (2-norm of the error of the linear system Ax = b). This is computed after the solve as `res = b - A*x`; we report \|\|res\|\|.
- **RHS norm** = \|\|b\|\|. Scale of the right-hand side.
- **Relative residual** = \|\|b − Ax\|\| / \|\|b\|\|. How small the error is compared to the size of b. Often reported as “rel_res”.
- **PETSc rnorm** = value returned by `KSPGetResidualNorm()`. For our setup (unpreconditioned norm) it is the same as the actual residual \|\|b − Ax\|\| at the last iteration.
- **Convergence check:** PETSc declares convergence when  
  \|\|r\|\| ≤ max(rtol × \|\|r_0\|\|, abstol),  
  where r = b − Ax is the current residual and r_0 is the initial residual. With zero initial guess, r_0 = b, so this is equivalent to relative residual \|\|r\|\|/\|\|b\|\| ≤ rtol (when abstol is 0). So **convergence** means: actual residual has dropped so that (actual / \|\|b\|\|) ≤ rtol.

So: **actual residual** is the real error; **relative residual** is that error divided by \|\|b\|\|; the solver “converges” when that relative residual is ≤ rtol (e.g. 1e-4). The solver **converges** only when PETSc reports reason > 0.

**What was the actual residual in our runs?** With motion term on we got **actual residual ≈ 26058** and a much larger \|\|b\|\|, so relative residual was small (e.g. 7e-5). The solver **did not converge** in any run: we always hit max iterations (reason = -3). Convergence would mean the solver stops with reason > 0 because \|\|b−Ax\|\|/\|\|b\|\| ≤ rtol.

---

Systematically change **one option at a time** in `solver_utils_submesh.make_config()`, run the solver, and compare **rel_res** and iteration count to find what prevents convergence.

## Run results (one-at-a-time)

| Config change | rel_res | reason | Note |
|---------------|---------|--------|------|
| **Baseline** (voltage, ADDITIVE, interior nodes, no motion) | 9.77e+08 | -3 (max it) | Residual huge – formulation/scaling issue without motion term |
| source_type=**current** | 4.18e+09 | -3 | Worse |
| **use_schur=True** | (timeout) | - | SCHUR much slower, run aborted |
| **use_interior_nodes=False** | inf | -9 | Immediate failure (NaN/Inf); interior nodes required |
| **use_motion_term=True** | **2.07e-01** → **7.1e-05** (300 it) | -3 | Large improvement; residual stalls ~7e-5 |

**Conclusion:** Enabling **use_motion_term=True** is necessary for a well-scaled residual. With it, rel_res reaches ~7e-5 in 300 iterations (stall). Keeping motion term on; optionally increase `outer_max_it` or set `outer_rtol=1e-4` and accept slow convergence.

## Config toggles (in `solver_utils_submesh.py`)

| Option | Default | Effect |
|--------|--------|--------|
| `use_schur` | `False` | `True` = SCHUR fieldsplit + full mat_nest as P; `False` = ADDITIVE + block-diagonal P |
| `use_interior_nodes` | `True` | `True` = pass interior nodes to Hypre AMS; `False` = skip |
| `use_motion_term` | `False` | `True` = add −σ(u_rot × curl A)·v on rotor (dx_rpm) in a00; `False` = skip |
| `source_type` | `"voltage"` | `"voltage"` = V on coils; `"current"` = prescribed J in coils |

## Suggested order to try (one at a time)

1. **Baseline**  
   Leave all toggles at default. Run and note `rel_res` and `reason` (e.g. -3 = max iterations).

2. **Source type**  
   Set `source_type="current"`. Run. If rel_res improves or converges, voltage drive may be stressing the saddle-point structure.

3. **Fieldsplit: SCHUR**  
   Set `use_schur=True`. Run. Compare with ADDITIVE (default). Sometimes SCHUR with A11 as Schur pre converges better or worse.

4. **Interior nodes**  
   Set `use_interior_nodes=False`. Run. If rel_res improves, interior-node marking for AMS may be wrong or unhelpful for this mesh.

5. **Motion term**  
   Set `use_motion_term=True`. Run. If rel_res improves, the removed motion term may be needed for consistency; if it worsens, the term may hurt conditioning.

6. **Relax tolerances / more iterations**  
   In config set e.g. `outer_rtol=1e-3`, `outer_max_it=300`. If the solver then converges with a looser tol or more iterations, the preconditioner is slow but not broken.

7. **Sigma in air**  
   Try `sigma_air_min=0` (no extra σ in air) or `sigma_air_min=1e-6` and compare. Affects conditioning of the A-block.

## How to run

From repo root (or `src/3d_submesh`):

```bash
python -m src.3d_submesh.main_submesh
# or
cd src/3d_submesh && python main_submesh.py
```

Watch the log for lines like:

- `KSP reason` (e.g. -3 = max it)
- `rel_res` / relative residual
- Final residual norm from `getResidualNorm()`

Record each run (toggle changed, rel_res, reason, iterations) to see which change helps or hurts.
