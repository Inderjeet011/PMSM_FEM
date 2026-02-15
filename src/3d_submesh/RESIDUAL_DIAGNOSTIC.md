# True Residual Diagnostic – Root Cause Analysis

## Summary

The solver reports "converged" (reason=2) but the **true residual** stays ~1.0. This document explains why and what to do.

---

## Root Cause

### 1. PETSc uses preconditioned residual for convergence

- **PETSc rnorm** ≈ 230 (what PETSc monitors)
- **True residual** ||b−Ax|| ≈ 23,400
- **Ratio** PETSc/true ≈ **0.01** (1%)

So PETSc converges on the **preconditioned** residual, not the true residual. The block-diagonal preconditioner P = diag(A00, A11) makes the preconditioned residual small while the true residual stays large.

### 2. Direct LU diagnostic

- **Direct LU** (MUMPS) on the monolithic system: **reason = -11** (KSP_DIVERGED_NAN)
- Residual = NaN, 0 iterations
- Likely causes: singular or ill-conditioned matrix, or LU breakdown

So we cannot use a direct solve to confirm that the system is well-posed.

### 3. Why the preconditioned residual is small

With ADDITIVE fieldsplit and P = diag(A00, A11):

- The preconditioner ignores the coupling blocks A01, A10
- GMRES minimizes the preconditioned residual ||P⁻¹(b−Ax)||
- P⁻¹ is block-diagonal, so it can make that norm small even when ||b−Ax|| is large

---

## Options to Improve True Residual

| Option | Effect | Cost |
|--------|--------|------|
| **SCHUR + schur_pre_type="full"** | Uses exact Schur complement, better preconditioner | Very slow (~hours) |
| **SCHUR + schur_pre_type="selfp"** | Approximate Schur with diag(A00) | ~2.5 min |
| **Force true residual for convergence** | Set `-ksp_norm_type unpreconditioned` (already in config) | May not converge (hit max_it) |
| **Increase outer_max_it** | More iterations | Slower, may still stall |

---

## Config Levers

In `solver_utils_submesh.py`:

- `outer_norm_type="unpreconditioned"` – intended to use true residual; PETSc may still use preconditioned internally for fieldsplit
- `diagnostic_direct_solve=True` – runs direct LU after iterative (needs A_mono)
- `use_schur=True` + `schur_pre_type="full"` – best preconditioner, very expensive

---

## B-Field Note

Max |B| ≈ 2.4 T is physically reasonable. The solution may still be useful for B-field even when the true residual is large, because the error can be mostly in the gradient (null-space) component, which does not affect curl(A) = B.
