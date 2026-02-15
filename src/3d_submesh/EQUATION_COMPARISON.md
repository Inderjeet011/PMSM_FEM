# Equation Comparison: 3d_submesh vs Arshad vs Team30

## Summary of Formulations

### Standard A-V weak form (backward Euler)

**A-equation:** ν∇×∇×A + σ(A−A_n)/dt + σ∇V = J₀  
**V-equation:** ∇·(σ(A−A_n)/dt + σ∇V) = 0

**Weak form:**
- a00: (ν∇×∇×A, v) + (σ/dt·A, v)
- a01: (σ∇V, v)
- a10: (σ/dt·A, ∇q)  ← **positive** sign
- a11: (σ∇V, ∇q)
- L0: (J₀, v) + (σ/dt·A_n, v)
- L1: (σ/dt·A_n, ∇q)

---

## Arshad (mixed_domain_AV.py)

| Block | Arshad | Scaling |
|-------|--------|---------|
| a00 | dt·ν·curl·curl + σ·u·v | Whole domain; multiplies curl by dt |
| a01 | dt·σ·grad(S)·v | Conductor |
| a10 | σ·grad(q)·u | Conductor, **no dt** |
| a11 | dt·σ·grad(S)·grad(q) | Conductor |
| L0 | σ·u_n·v | Whole |
| L1 | grad(q)·σ·u_n | Conductor |

**Differences from standard:**
- Uses dt as a global scaling factor (different time discretization)
- a10 has **no** 1/dt factor (Arshad’s scaling differs)
- Single conductor (copper rod), no motion term, no magnets

---

## Team30 (solve_3D_time_AV.py)

| Block | Team30 | Notes |
|-------|--------|-------|
| a00 | dt·(1/μ_R)·curl·curl + σ·μ₀·A·v | Uses μ₀ in mass term |
| a01 | μ₀·σ·inner(v, grad(S)) | |
| a10 | **zero·div(A)·q** | Gauge term (Coulomb gauge) |
| a11 | μ₀·σ·inner(grad(S), grad(q)) | |
| L0 | dt·μ₀·J₀·v[2] + σ·μ₀·inner(A_prev, v) | |
| L1 | 0 | L1 = 0 |

**Differences:**
1. **a10 = div(A)·q** – Coulomb gauge; we do not have this
2. **L1 = 0** – no A_prev in V-equation RHS
3. **μ₀** appears in σ terms (σ·μ₀)
4. No motion term

---

## 3d_submesh (current)

| Block | 3d_submesh | Notes |
|-------|------------|-------|
| a00 | ν·curl·curl + (σ/dt)·A·v + motion | dx_rs, dx_air; motion on dx_rpm |
| a01 | σ_coupling·grad(S)·v | No dt; σ_coupling = 0 in PM, coils |
| a10 | **−(σ_coupling/dt)·A·∇q** + motion | **Minus sign** |
| a11 | σ·grad(S)·grad(q) | No dt |
| L0 | J_z·v + (σ/dt)·A_prev·v + M·curl(v) | Coils, conductors, magnets |
| L1 | (σ_coupling/dt)·A_prev·∇q | |

**Potential issues:**
1. **a10 sign** – Standard form has +(σ/dt)·A·∇q; we use −(σ/dt)·A·∇q. Worth checking.
2. **a11 scaling** – We use σ·grad·grad without dt; Arshad uses dt·σ·grad·grad. Depends on time scheme.
3. **dx_cond_parent** – Coupling only over conductor submesh (ROTOR+ALUMINIUM+MAGNETS+COILS); STATOR not in submesh, so no V there.

---

## 3d (normal, same mesh)

Same structure as 3d_submesh: a10 = −(σ/dt)·A·∇q + motion. So the sign is consistent with our main 3d solver.

---

## Possible Missing or Different Terms

| Item | Arshad | Team30 | 3d_submesh |
|------|--------|--------|------------|
| **Coulomb gauge (div A)** | No | Yes (a10 = div(A)·q) | No |
| **Motion term** | No | No | Yes |
| **Magnets (M·curl)** | No | No | Yes |
| **Coil current J_z** | No | Yes | Yes |
| **σ in a11** | dt·σ | μ₀·σ | σ (no dt) |
| **a10 scaling** | σ (no dt) | zero (gauge) | −σ/dt |

---

## Recommendations

1. **a10 sign** – Verify against the standard weak form; +(σ/dt)·A·∇q may be correct.
2. **Coulomb gauge** – Team30’s div(A)·q can improve uniqueness; consider adding as an option.
3. **L1 = 0 in Team30** – Their V-equation RHS is zero; ours uses (σ/dt)·A_prev·∇q. Ours matches the standard time-dependent formulation.
4. **μ₀ in Team30** – They use σ·μ₀; we use σ. Check physical units and consistency with your model.
