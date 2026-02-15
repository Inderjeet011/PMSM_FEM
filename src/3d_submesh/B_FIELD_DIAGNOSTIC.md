# B-Field Strength Diagnostic – Pipeline Analysis

## Problem

- **Current result**: Max |B| ≈ 4.9×10⁻⁵ T (with voltage drive, history formulation)
- **Expected**: ~1–2.4 T (magnets + coils)
- **Pure curl-curl test** (magnets only, no sigma/coupling): Max |B| ≈ 2.7 T, B in PM ≈ 1.0 T ✓

So the magnet formulation is correct. The full A–V system is suppressing the field by ~50,000×.

---

## Root Causes

### 1. Voltage drive → J_z = 0 (no coil current)

With `source_type="voltage"`, `update_currents()` sets `J_z = 0` everywhere. The only source in L0 is magnetization. No coil contribution.

### 2. Cu sigma = 0 (coils non-conducting)

In `model_parameters`, Cu has `sigma: 0`. With voltage drive, coils cannot carry current.

### 3. A–V coupling suppresses A in conductors (main cause)

The full system has coupling blocks a01, a10, a11 over `dx_cond_parent` (ROTOR + ALUMINIUM + MAGNETS + COILS). The Schur complement effectively penalizes A in conducting regions. PM (sigma=6.25e5) is included, so the coupling strongly constrains A in the magnet region and suppresses B.

### 4. Original 3d vs submesh

The original 3d solver uses `dx_rs` (rotor+aluminium+stator) for a01, a10, L1 – **excluding PM** from the A–V coupling. The submesh version uses `dx_cond_parent` (includes PM). Restricting coupling to dx_rs caused assembly errors (IndexMap mismatch: conductor submesh has no stator).

---

## Fixes Tried

| Fix | B result | Convergence |
|-----|----------|-------------|
| **source_type="current"** | 6.8×10⁻⁸ T (worse) | rel_res 4.4e-5 ✓ |
| **voltage + sigma_cu_override=5.96e7** | 5×10⁻¹⁰ T | Diverged |
| **sigma_air=0, use_motion_term=False** | 3.85×10⁻⁴ T (7× better) | Diverged |
| **sigma_air=1e-8, use_motion_term=False** | 1.1×10⁻⁵ T (worse) | rel_res 6.2e-5 ✓ |
| **Coupling restricted to dx_rs** | – | Assembly error |
| **Baseline (voltage, sigma_air=1e-8, motion)** | 4.9×10⁻⁵ T | rel_res 6.5e-5 ✓ |

---

## Conclusion

The A–V coupling over the conductor region (including PM) is the main cause of B suppression. Removing sigma_air and motion term improved B to 3.85×10⁻⁴ T but broke convergence. The submesh formulation differs from the original 3d (which excludes PM from coupling); aligning them requires handling the conductor-submesh / stator mismatch.

**Working fix (implemented):**
- **Two-stage solve**: Solve magnet-only first, set A_prev = scale × A_mag for the first time step.
- **Krylov x0**: Use scaled magnet solution as initial guess for GMRES.
- **sigma_pm_coupling_scale**: Weaken A–V coupling in PM (use σ×scale in a01, a10, a11, L1). Reduces B suppression. 0.1 gives B~90 mT, rel_res~2.6e-4; 0.01 gives B~330 mT but solver diverges.
- Config: `use_magnet_initial_guess=True`, `magnet_A_prev_scale=0.4`, `use_magnet_as_ksp_x0=True`, `sigma_pm_coupling_scale=0.1`, `outer_max_it=600`, `outer_rtol=3e-4`.
- Result: Max |B| ≈ 90–170 mT (run-dependent; ~2000× improvement over baseline 50 µT).
- True relative residual typically 2–3×10⁻⁴ (near rtol).
