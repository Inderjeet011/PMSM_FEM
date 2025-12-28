## PMSM FEM Simulation (2D + 3D)

Permanent Magnet Synchronous Motor (PMSM) electromagnetic simulation using **FEniCSx / DOLFINx**.

This repo’s “current focus” is a **3D transient A–V eddy-current solver** with:
- rotating permanent magnets (magnetization is rotated each timestep)
- 3‑phase coil current drive
- PETSc block preconditioning tuned for H(curl)

### Quickstart (3D solver)

#### 1) Create an environment (FEniCSx)

How you install `dolfinx` depends on your platform. Two common options:

- **Conda (recommended)**: install from conda-forge (package name is usually `fenics-dolfinx`)
- **Docker**: use an official/known-good DOLFINx image and run this repo inside it

At minimum you’ll need: `dolfinx`, `petsc4py`, `mpi4py`, `numpy`, and (for mesh generation) `gmsh`.

#### 2) Generate (or use) the 3D mesh

This repo already ships a mesh in `meshes/3d/`:
- `meshes/3d/pmesh3D_ipm.xdmf` (+ `pmesh3D_ipm.h5`)

To regenerate it with Gmsh:

```bash
python src/3d/mesh_3D.py --res 0.01 --depth 0.057
```

This writes `meshes/3d/pmesh3D_ipm.xdmf` (and `.h5`).

#### 3) Run the solver

```bash
cd src/3d
python main.py
```

MPI is supported and often faster/more robust for larger meshes:

```bash
cd src/3d
mpirun -np 4 python main.py
```

### Output files (ParaView)

By default, results are written to:
- `results/3d/av_solver.xdmf` (and `results/3d/av_solver.h5`)

Fields written include:
- **`A`**: vector potential (interpolated into a Lagrange vector space for visualization)
- **`V`**: scalar potential
- **`B_dg`**: **DG0 (cell-wise constant) \(B=\nabla\times A\)**; this is the “physics first” B-field
- **`B` / `B_Magnitude`**: projected/smoothed B fields for visualization
- **`B_vis` / `B_vis_mag`**: visualization helpers

Open `results/3d/av_solver.xdmf` in ParaView. For the most faithful B-field, use **`B_dg` (Cell Data)**.

### Motor-only output (default)

The solver is configured to keep outputs small and ParaView-friendly:
- it runs on the full mesh, then **replaces** `results/3d/av_solver.xdmf/.h5` with a **motor-only** dataset (outer airbox removed)

This behavior is controlled by `SimulationConfig3D.output_motor_only` in `src/3d/load_mesh.py`.

### Utilities

#### Extract motor-only from an existing results file

If you already have an XDMF/H5 pair and want to remove the airbox:

```bash
python src/3d/extract_motor_only.py --input results/3d/av_solver.xdmf --output results/3d/av_solver_motor_only.xdmf
```

### Project structure

```text
FEniCS/
├── meshes/
│   ├── 2d/                      # 2D meshes (legacy)
│   └── 3d/                      # 3D PMSM meshes (XDMF/H5, plus .msh)
├── results/                     # Output files (auto-generated)
└── src/
    ├── 2d/                      # Legacy 2D solver and mesh tools
    └── 3d/                      # 3D A–V eddy-current solver (current focus)
```

### 3D A–V eddy-current formulation (implementation notes)

- **Spaces**:
  - `A` in Nédélec H(curl) (`N1curl`, degree 1)
  - `V` in Lagrange H¹ (degree 1)
- **Block system**: 2×2 PETSc `MatNest` with blocks `A00, A01, A10, A11`:
  - `A00`: `nu * curl(A)·curl(v) + (mu0 * sigma / dt) A·v` + motional term + weak BC penalty on the exterior
  - `A01`: `mu0 * sigma v·∇V` restricted to conductor region
  - `A10`: `(mu0 * sigma / dt) A·∇q` in conductors
  - `A11`: `mu0 * sigma ∇V·∇q` in conductors + small stabilization on zero‑σ regions
- **Time stepping**: backward Euler; `1/dt` appears on σ-coupled terms (not on curl–curl or sources)
- **Permanent magnets & currents**:
  - magnetization `M` initialized as `Br / mu0` in magnet regions and rotated with rotor angle
  - PM forcing uses `-∫ μ0 M·curl(v) dx_magnets`
  - coil source uses `∫ J_z v_z dx` (`J` in A/m²)
- **Boundary conditions**:
  - `A`: weakly enforced `A → 0` on the exterior via a penalty term (no strong Dirichlet BCs)
  - `V`: strong Dirichlet BC on exterior + one MPI-safe grounded DOF to remove the constant null mode

### Solver and preconditioning (3D)

The solver uses PETSc block preconditioning (`PCFIELDSPLIT`) and Hypre methods (AMS/BoomerAMG) to handle H(curl) efficiently. See `src/3d/solve_equations.py` for the exact configuration and tuning knobs.

### 2D legacy solver

The original 2D PMSM scripts (`src/2d`) are still present but not actively maintained.
