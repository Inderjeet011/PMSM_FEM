# 2D vs 3D Source Comparison

## Source Parameters

| Parameter | 2D | 3D | Ratio (2D/3D) |
|----------|----|----|----------------|
| J_peak | 7.07e6 A/m² | 4.38e6 A/m² | 1.61x |
| B_rem | 1.4 T | 1.2 T | 1.17x |
| M_rem | 1.11e6 A/m | 9.55e5 A/m | 1.16x |

## RHS Application

### 2D
```python
# Current source
L += self.J_z * v * self.dx(coil)

# PM source
M_vec = ufl.as_vector((self.M_x, self.M_y))
L += -ufl.inner(M_vec, curl_v) * dxc(Omega_pm)
```

### 3D
```python
# Current source (z-component only - correct for 3D)
J_term = dt * mu0 * J_z * v[2] * dx

# Time-stepping term
lagging = sigma * mu0 * ufl.inner(A_prev, v) * dx

# PM source
pm_term = -ufl.inner(M_vec, curlv) * dx_magnets

L0 = J_term + lagging + pm_term
```

## Current Issue

- **B field is 6e-9 T** (expected 0.1-2 T) - **8 orders of magnitude too small**
- **Solver not converging**: residual = 2.67e+02, hit max iterations (100)
- **Sources are smaller** but not enough to explain 8 orders of magnitude difference

## Likely Causes

1. **Solver convergence**: Residual 2.67e+02 is very high - solution not converged
2. **Initial conditions**: A_prev = 0 at first step (no history)
3. **Source application**: Need to verify J_z and M_vec are correctly set in cells
4. **Mesh/material properties**: May need to check if materials are correctly assigned

## Recommendations

1. ✅ **Already done**: Increased max_it to 500, relaxed tolerances
2. **Check RHS norms**: Verify J_z and M_vec have reasonable values
3. **Check source application**: Verify sources are applied to correct cell regions
4. **Check convergence**: Monitor if residual decreases with more iterations
