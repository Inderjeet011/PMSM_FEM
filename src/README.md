# Source Code - PMSM FEM Simulation

This directory contains the Python source code for the Permanent Magnet Synchronous Motor (PMSM) finite element simulation.

## File Structure

```
src/
├── pm_motor_mesh_generator.py   # Step 1: Generate motor geometry mesh
├── mesh_viewer.py                # Step 2: Visualize and analyze mesh
├── maxwell_solver_2d.py          # Step 3: Run FEM simulation
└── airgap_field_extractor.py    # Step 4: Extract airgap magnetic field
```

## Usage

All scripts should be run **from within the `src/` directory**. They will automatically:
- Read mesh files from the parent directory (`../motor.msh`)
- Save results to the `../results/` directory

### Step-by-Step Workflow

#### 1. Generate the Motor Mesh
```bash
cd src
python3 pm_motor_mesh_generator.py
```
**Output:** `../motor.msh` (mesh file in root directory)

#### 2. Visualize the Mesh (Optional)
```bash
python3 mesh_viewer.py
```
**Output:** 
- `../results/motor_mesh_visualization.png`
- `../results/motor_mesh_quality.png`

#### 3. Run the FEM Solver
```bash
python3 maxwell_solver_2d.py
```
**Output:**
- `../results/results_2d_mixed.h5`
- `../results/results_2d_mixed.xdmf`

#### 4. Extract Airgap Field (Optional)
```bash
python3 airgap_field_extractor.py
```
**Output:**
- `../results/airgap_B_only.h5`
- `../results/airgap_B_only.xdmf`

## Directory Structure

```
PMSM_FEM/
├── src/                  # Source code (run scripts from here)
│   ├── *.py             # Python scripts
│   └── README.md        # This file
├── results/             # Output files (auto-created)
│   ├── *.xdmf          # Simulation results
│   ├── *.h5            # Data files
│   └── *.png           # Visualizations
├── motor.msh            # Mesh file (generated)
├── README.md            # Project documentation
└── TROUBLESHOOTING_GUIDE.md
```

## Requirements

- FEniCSx (dolfinx)
- gmsh
- numpy
- matplotlib
- mpi4py

## Notes

- All scripts use **relative paths** to maintain clean separation between source and results
- The `results/` directory is automatically created if it doesn't exist
- Large data files (`.h5`, `.xdmf`, `.msh`) are excluded from git by default

