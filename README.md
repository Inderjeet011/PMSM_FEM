# PMSM FEM Simulation

Permanent Magnet Synchronous Motor (PMSM) electromagnetic simulation using FEniCS/DOLFINx.

## Project Structure

```
FEniCS/
├── src/
│   ├── pm_mesh_generator_2d.py      # Generate motor mesh
│   ├── mesh_viewer.py               # Visualize mesh
│   ├── solver_2d.py                 # Run FEM simulation
│   └── airgap_field_extractor.py   # Extract air gap B-field
├── results/                          # Output files (auto-generated)
├── motor.msh                        # Mesh file (generated)
└── README.md                        # This file
```

## Usage

All commands should be run from the `src/` directory:

```bash
cd src

# Step 1: Generate mesh
python3 pm_mesh_generator_2d.py

# Step 2: Run simulation
python3 solver_2d.py

# Step 3: Extract air gap field
python3 airgap_field_extractor.py
```

## Current Configuration

- **Current Density**: 3.0 MA/m²
- **PM Remanence**: 1.4 T
- **Permeability**: μᵣ = 1000 (rotor/stator)
- **Air Gap B-field**: ~0.42 T (peak)

## Requirements

- FEniCSx (dolfinx)
- gmsh
- numpy
- matplotlib
- mpi4py

## Results

Simulation outputs are saved in `results/`:
- `results_2d_mixed.h5/.xdmf` - Full simulation results
- `airgap_B_only.h5/.xdmf` - Air gap B-field only

View results in ParaView:
```bash
paraview results/airgap_B_only.xdmf
```
