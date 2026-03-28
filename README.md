# PMSM_FEM

Finite-element PMSM models built with `FEniCSx` / `DOLFINx`. The repository contains a legacy 2D model plus several active 3D transient A-V eddy-current workflows for different coil geometries.

## What Is In The Repo

- `src/2d`: legacy 2D PMSM mesh generator and solver
- `src/3d`: main 3D transient A-V solver with bulk 3-phase current-driven coils
- `src/3d_loop_mesh`: 3D loop-coil variant with local BP/VTX outputs
- `src/3d_rod_mesh`: 3D rod-coil variant with extended copper regions

All 3D variants use:

- `A` in an `H(curl)` Nedelec space
- `V` on a conductor submesh
- rotating permanent-magnet excitation
- PETSc `fieldsplit` block solves with Hypre preconditioning

## Requirements

You need a working FEniCSx environment with at least:

- `dolfinx`
- `petsc4py`
- `mpi4py`
- `numpy`
- `gmsh`

Typical options are a conda-forge environment or a DOLFINx Docker image.

## Common Workflow

### 1. Generate A Mesh

Each 3D workflow has its own mesh generator and writes mesh files into the same folder as the script.

Examples:

```bash
cd src/3d
python mesh_3D.py --res 0.005 --depth 0.057
```

```bash
cd src/3d_loop_mesh
python mesh_3D.py
```

```bash
cd src/3d_rod_mesh
python mesh_3D.py
```

This produces files such as:

- `pmesh3D_ipm.msh`
- `pmesh3D_ipm.xdmf`
- `pmesh3D_ipm.h5`

### 2. Run A Solver

Run the solver from the corresponding folder:

```bash
cd src/3d
python main_submesh.py
```

```bash
cd src/3d_loop_mesh
python main_submesh.py
```

```bash
cd src/3d_rod_mesh
python main_submesh.py
```

MPI execution is also supported:

```bash
mpirun -np 4 python main_submesh.py
```

## Outputs

Each 3D folder writes results locally beside the solver scripts.

Common outputs include:

- `av_solver_submesh.xdmf`
- `av_solver_submesh.h5`
- `V_field_submesh.bp`
- `J_field_submesh.bp`
- `B_field_motor_submesh.bp`

The exact BP outputs depend on the workflow variant. Open the `xdmf` file in ParaView for the standard dataset, or the `.bp` folders for VTX/ADIOS output.

## Notes On The Active 3D Model

- The main `src/3d` workflow currently uses bulk 3-phase current-driven coil regions.
- Opposite coil sides are mapped to the same phase with opposite current sign.
- The `A`-block preconditioner in the active `src/3d` solver is Hypre `boomeramg` with `HMIS` coarsening.
- The main `src/3d` mesh now extrudes coil regions slightly beyond the motor stack in both `+z` and `-z`.

## Repo Layout

```text
PMSM_FEM/
├── README.md
└── src/
    ├── 2d/
    ├── 3d/
    ├── 3d_loop_mesh/
    └── 3d_rod_mesh/
```

## Status

This repository is under active development. Several 3D workflows coexist while geometry, excitation, and output handling are being compared and refined.
