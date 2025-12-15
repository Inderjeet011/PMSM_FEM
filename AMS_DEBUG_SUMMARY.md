# AMS Convergence Issue -9 (DIVERGED_NANORINF) Root Cause Analysis

## Problem
- Convergence reason: -9 (DIVERGED_NANORINF)
- Iterations: 0
- Residual: inf
- Solver breaks down immediately, before starting iterations

## What We've Verified
1. ✅ Matrices have no NaN/Inf values
2. ✅ RHS is non-zero (norm=2.5e4)
3. ✅ Discrete gradient builds correctly (size 50328×7504, norm 3.17e2)
4. ✅ Edge constant vectors build correctly (all have reasonable norms ~4.5)
5. ✅ A00_spd is created and is SPD
6. ✅ Weak boundary conditions implemented correctly

## Critical Finding
**Segfault occurs when calling `pc_A.setUp()`** - This means AMS preconditioner setup is FAILING.

## Root Cause Hypothesis
The AMS preconditioner cannot be initialized with the current configuration. Possible reasons:

1. **Edge constant vectors format**: The vectors might not be in the format AMS expects
   - Currently using `f.x.petsc_vec` after interpolation
   - AMS might need a different vector structure

2. **Fieldsplit + AMS interaction**: Setting operators on sub-KSP after fieldsplit might break AMS
   - Fieldsplit automatically extracts operators from P
   - Overriding them might cause AMS to receive incompatible data

3. **Matrix structure**: AMS might need the matrix in a specific format
   - Currently using nested matrices
   - AMS might need direct matrix access

4. **Discrete gradient compatibility**: The discrete gradient might not match the matrix structure
   - Size: 50328×7504 (A_space × V_space)
   - This should be correct, but AMS might have specific requirements

## What Works
- GAMG preconditioner works fine (converges in 100 iterations)
- Matrix assembly is correct
- RHS assembly is correct
- All forms are built correctly

## Next Steps to Debug
1. Try AMS WITHOUT fieldsplit (set up KSPs manually)
2. Check if edge constant vectors need to be in a different format
3. Verify discrete gradient is compatible with AMS requirements
4. Check PETSc/Hypre AMS documentation for exact requirements
5. Consider using a simpler test case to isolate the issue

## Current Configuration
- Weak boundary conditions (penalty term alpha=1e6)
- SPD preconditioning matrix (A00_spd)
- Discrete gradient: V_space → A_space
- Edge constant vectors: interpolated constant fields (1,0,0), (0,1,0), (0,0,1)
- Fieldsplit with additive type
- AMS for A block, GAMG for V block

