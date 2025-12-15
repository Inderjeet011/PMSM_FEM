## PMSM FEM Simulation (2D + 3D)

Permanent Magnet Synchronous Motor (PMSM) electromagnetic simulation using FEniCS/DOLFINx.

### Project Structure

```text
FEniCS/
├── src/
│   ├── 2d/                      # Legacy 2D solver and mesh tools
│   └── 3d/                      # 3D A–V eddy-current solver (current focus)
├── results/                     # Output files (auto-generated, not tracked)
└── README.md                    # This file
```

### 3D A–V eddy-current solver (current setup)

- **Spaces**:
  - `A` in Nédélec H(curl) (`N1curl`, degree 1)
  - `V` in Lagrange H¹ (degree 1)
- **Block system**: 2×2 PETSc `MatNest` with blocks `A00, A01, A10, A11`:
  - `A00`: `nu * curl A·curl v + (mu0 * sigma / dt) A·v` + motional term + weak BC penalty on the exterior.
  - `A01`: `mu0 * sigma v·∇V` restricted to conductor region.
  - `A10`: `(mu0 * sigma / dt) A·∇q` in conductors.
  - `A11`: `mu0 * sigma ∇V·∇q` in conductors + small stabilization on zero-σ regions.
- **Time stepping**:
  - Backward Euler in time; `1/dt` appears on σ-coupled “mass” and A–V constraint terms, not on curl–curl or sources.
- **Permanent magnets & currents**:
  - `M_vec` is magnetization in A/m, initialized as `Br / mu0` in magnet regions and rotated with rotor angle.
  - PM forcing uses `-∫ μ0 M·curl v dx_magnets` (no extra dt).
  - Coil source uses `∫ J_z v_z dx` (J in A/m², no μ0 or dt factor).
- **Boundary conditions**:
  - `A`: weakly enforced `A → 0` on the exterior via a penalty term on `A00`/`A00_spd` (no strong Dirichlet BCs).
  - `V`: strong Dirichlet BC on exterior + one MPI-safe grounded DOF inside the conductor network to remove the constant null mode.

### Solver and preconditioning (3D)

- **Global KSP**:
  - `FGMRES` on the full 2×2 nested matrix with:
    - `rtol=1e-6`, `atol=0.0`, `max_it=300`
    - Python monitor + `-ksp_monitor_true_residual` for outer residuals.
- **Preconditioner**:
  - PETSc `PCFIELDSPLIT` with **Schur complement**:
    - `PCCompositeType.SCHUR`, `SchurFactType.LOWER`, `SchurPreType.A11`.
  - **A-block sub-KSP** (`A00`):
    - `KSPType.PREONLY` with Hypre **AMS**:
      - Operator and PC matrix: SPD `A00_spd` (curl–curl + σμ0 mass + small ε mass shift + scaled weak BC penalty, no `1/dt` amplification).
      - Discrete gradient `G: H¹ → H(curl)` built from unconstrained P1 scalar space.
      - Coordinate or edge-constant vectors passed to AMS (via `setHYPRECoordinateVectors` or `setHYPRESetEdgeConstantVectors(G*x, G*y, G*z)`).
      - AMS projection enabled on the A-fieldsplit.
  - **V-block sub-KSP** (Schur system in `V`):
    - Operator: Schur complement `S` from the PC.
    - Preconditioner matrix: `A11`.
    - `KSPType.FGMRES` with small fixed work (`max_it=3`) and Hypre **BoomerAMG** on `A11` (HMIS/extrapolation/symmetric-SOR options).

### B-field postprocessing (3D)

- **Primary field**:
  - `B_dg` in DG0 vector space (piecewise constant per cell), computed via L² projection of `curl(A)` onto DG0.
  - Written to XDMF as **cell data**: this is the “physics first” B-field.
- **Region-wise diagnostics** (DG0):
  - For each of AirGap, PM, Iron, and Conductors:
    - `max|B|`, `mean|B|`, `median|B|`, 90th and 99th percentiles.
    - Percentage of cells with `|B| > 10 T` and `|B| > 100 T`.
  - Current configuration (after correct scaling and PM μ0 fix) yields:
    - AirGap: `O(0.1)` T, no >10 T cells.
    - PM and Iron: `O(0.1–1)` T, no >10 T cells.
- **Visualization fields**:
  - Optional projection from `B_dg` into a user-chosen visualization space (`B_space`) and magnitude space (`B_magnitude_space`), written as `B` and `B_Magnitude` for ParaView.

### 2D legacy solver

The original 2D PMSM scripts (`src/2d`) are still present but not actively maintained. They used a simpler magnetostatic formulation with:

- Current density ≈ 3.0 MA/m².
- PM remanence ≈ 1.4 T.
- μᵣ ≈ 1000 in rotor and stator.
- Typical air-gap peak B-field ≈ 0.4 T.

### Requirements

- dolfinx (FEniCSx)
- gmsh
- numpy
- mpi4py
- petsc4py

### Running the 3D solver

From `src/3d`:

```bash
cd src/3d
python run_solver.py
```

Results (mesh, tags, `A`, `V`, `B_dg`, `B`, `B_Magnitude`) are written to:

- `results/3d/av_solver.xdmf`

You can inspect them in ParaView by opening that XDMF file and using:

- `B_dg` (cell data) to see the raw DG0 curl-based B-field.
- `B` / `B_Magnitude` (function data) for smoothed visualization.
