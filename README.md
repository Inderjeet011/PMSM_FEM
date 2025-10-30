# PM Motor FEniCS Simulation Suite

Professional permanent magnet (PM) motor electromagnetic simulation using FEniCS/Dolfinx.

## 📋 Overview

This suite provides a complete workflow for PM motor electromagnetic analysis:
1. **Mesh generation** - Professional motor geometry with split air gap
2. **Maxwell solver** - 2D A-V mixed formulation with rotation
3. **Visualization** - Mesh quality and field analysis
4. **Post-processing** - Air gap field extraction

Based on TEAM 30 benchmark techniques with Abhinav's parameters.

---

## 🗂️ Project Structure

```
FEniCS/
├── pm_motor_mesh_generator.py    # Step 1: Generate motor mesh
├── mesh_viewer.py                # Step 2: Inspect mesh quality
├── maxwell_solver_2d.py          # Step 3: Run EM simulation
├── airgap_field_extractor.py    # Step 4: Extract air gap B-field
├── README.md                     # This file
├── TROUBLESHOOTING_GUIDE.md     # Debugging help
├── latest_fem/                   # Working implementations
└── team_30/                      # TEAM 30 reference examples
```

---

## 🚀 Quick Start

### 1. Generate Motor Mesh

```bash
python3 pm_motor_mesh_generator.py
```

**Output:** `motor.msh` (Gmsh format)

**Features:**
- 8-pole PM motor (4 pole pairs)
- 6-slot stator with 3-phase windings
- Split air gap for robust meshing
- Area-based domain classification
- Adaptive mesh refinement

**Configuration** (edit in `MotorGeometry` class):
```python
r_rotor = 0.030          # Rotor outer: 30 mm
r_pm_out = 0.038         # PM outer: 38 mm
r_airgap_out = 0.040     # Air gap: 2 mm
r_stator_out = 0.057     # Stator outer: 57 mm
pole_pairs = 4           # 8 magnets total
n_coils = 6              # 6 slots
mesh_resolution = 0.002  # 2 mm base resolution
```

---

### 2. Inspect Mesh Quality

```bash
python3 mesh_viewer.py
```

**Output:**
- `motor_mesh_visualization.png` - Domain colors, structure, node density
- `motor_mesh_quality.png` - Quality distribution histogram

**Checks:**
- Element count and quality metrics
- Domain classification verification
- Mesh refinement in critical regions

---

### 3. Run Electromagnetic Simulation

```bash
python3 maxwell_solver_2d.py
```

**Output:** `results_2d_mixed.xdmf` + `results_2d_mixed.h5`

**Features:**
- A-V mixed formulation (magnetic vector potential + electric potential)
- Rotating PM magnetization
- 3-phase sinusoidal currents
- Rotor eddy currents
- Time-stepping solver

**Configuration** (edit in `SimulationConfig` class):
```python
pole_pairs = 2
frequency = 50.0           # Hz
J_peak = 1413810.0         # A/m² (current density)
B_rem = 1.05               # T (remanent flux)
dt = 0.002                 # 2 ms timestep
T_end = 0.04               # 40 ms (2 periods)
```

**Runtime:** ~2-5 minutes (depends on mesh size)

---

### 4. Extract Air Gap Field

```bash
python3 airgap_field_extractor.py
```

**Output:** `airgap_B_only.xdmf` + `airgap_B_only.h5`

**Features:**
- Computes B = curl(Az)
- Masks to air gap region only
- Calculates |B| magnitude
- Time-series animation data

**Visualization in ParaView:**
```bash
paraview airgap_B_only.xdmf
```
1. Select `B_magnitude` in dropdown
2. Click **PLAY ▶️** to see rotation
3. Use **Glyph** filter on `B_airgap` for vectors

---

## 📊 Key Parameters

### Motor Geometry
| Parameter | Value | Description |
|-----------|-------|-------------|
| Pole pairs | 4 | 8 magnets (N-S alternating) |
| Slots | 6 | 3-phase windings |
| Air gap | 2 mm | Split at 39 mm |
| PM thickness | 8 mm | 30-38 mm radial |
| Rotor outer | 30 mm | |
| Stator outer | 57 mm | |

### Electrical
| Parameter | Value | Description |
|-----------|-------|-------------|
| Frequency | 50 Hz | Electrical frequency |
| Speed | 750 RPM | Mechanical (ω_m = ω_e / pole_pairs) |
| Current | 1.41 MA/m² | Peak current density |
| B_rem | 1.05 T | PM remanent flux |

### Materials
| Material | σ (S/m) | μ_r |
|----------|---------|-----|
| Rotor | 2e6 | 100 |
| Stator | 0 | 100 |
| PM | 0 | 1.05 |
| Air gap | 0 | 1.0 |
| Coils | 0 | 0.999991 |

---

## 🛠️ Code Organization

All files follow a clean, object-oriented structure:

### Class-based Design
```python
# Mesh Generator
class PMMotorMeshGenerator:
    def __init__(self, geometry=None)
    def generate_mesh(self, output_file)
    # ... component methods

# Maxwell Solver
class MaxwellSolver2D:
    def __init__(self, mesh_file, config)
    def run()
    # ... setup and solve methods

# Mesh Viewer
class MotorMeshViewer:
    def __init__(self, mesh_file)
    def run()
    # ... analysis and plotting

# Field Extractor
class AirGapFieldExtractor:
    def __init__(self, config)
    def run()
    # ... extraction and export
```

### Configuration Classes
Each module has a configuration class for easy parameter changes:
- `MotorGeometry` - Geometric parameters
- `DomainMarkers` - Physical tags
- `SimulationConfig` - Solver parameters

---

## 🔧 Advanced Usage

### Custom Geometry

Edit `pm_motor_mesh_generator.py`:
```python
class MotorGeometry:
    r_rotor = 0.035          # Larger rotor
    pole_pairs = 3           # 6 magnets
    n_coils = 9              # 9 slots
    mesh_resolution = 0.001  # Finer mesh
```

### Custom Operating Point

Edit `maxwell_solver_2d.py`:
```python
class SimulationConfig:
    frequency = 60.0         # 60 Hz operation
    J_peak = 2e6             # Higher current
    T_end = 0.1              # Longer simulation
```

### Mesh Refinement Zones

In `pm_motor_mesh_generator.py`, modify `setup_mesh_refinement()`:
```python
# Extra fine in air gap
gmsh.model.mesh.field.setNumber(5, "LcMin", res * 0.2)  # Finer

# Coarser in outer air
gmsh.model.mesh.field.setNumber(2, "LcMax", 20 * res)  # Coarser
```

---

## 📐 Mesh Quality Guidelines

**Target Quality Metrics:**
- Minimum SICN: > 0.3 (acceptable)
- Average SICN: > 0.7 (good)
- Elements < 0.3: < 5% (acceptable)

**Critical Refinement Zones:**
1. **Air gap**: 0.3× base resolution (most critical!)
2. **PM boundaries**: 0.25× base resolution
3. **Coil regions**: 1× base resolution
4. **Outer air**: 10× base resolution (coarse OK)

---

## 🧪 Testing

### Verify Mesh
```bash
python3 mesh_viewer.py
# Check: All domains present, quality > 0.3
```

### Quick Test Run
```python
# In maxwell_solver_2d.py
class SimulationConfig:
    T_end = 0.004  # Only 2 timesteps
    dt = 0.002
```

### Check Convergence
Monitor solver output:
```
Step   1/20  t= 2.00 ms  ||Az||=1.23e-04
```
- `||Az||` should be stable (not growing)
- Solver should converge in < 100 iterations

---

## 📚 Technical Details

### A-V Mixed Formulation

**Variables:**
- `Az`: Magnetic vector potential (z-component)
- `V`: Electric scalar potential

**Equations:**
```
∇ × (ν ∇ × A) + σ(∂A/∂t + u×B + ∇V) = J_source + curl(M)
∇ · (σ(∂A/∂t + u×B + ∇V)) = 0
```

**Weak form:** See `maxwell_solver_2d.py` → `create_variational_form()`

### PM Magnetization

Radial magnetization pattern:
```python
M = sign × M_rem × [cos(θ), sin(θ)]
```
- N-poles: sign = +1 (outward)
- S-poles: sign = -1 (inward)
- Rotates with rotor: θ → θ + ω_m × t

### Air Gap Splitting

Key technique for robust meshing:
- Split at r = 39 mm (midpoint)
- Inner: 38-39 mm (next to PM)
- Outer: 39-40 mm (next to stator)
- Prevents thin element distortion

---

## 🐛 Troubleshooting

See `TROUBLESHOOTING_GUIDE.md` for detailed debugging steps.

**Common Issues:**

1. **"No rotor cells found"**
   - Check mesh generation completed successfully
   - Verify domain tags in mesh viewer

2. **Solver divergence**
   - Reduce timestep: `dt = 0.001`
   - Increase V regularization: `epsV = 1e-10`
   - Check mesh quality

3. **Poor mesh quality**
   - Decrease `mesh_resolution = 0.001`
   - Check air gap split worked correctly

4. **Missing B-field in air gap**
   - Verify simulation completed
   - Check `results_2d_mixed.h5` exists
   - Ensure air gap cells found in mesh

---

## 📖 References

1. **TEAM 30 Benchmark** - Standard test case for electric machines
2. **FEniCS Project** - https://fenicsproject.org/
3. **Gmsh** - https://gmsh.info/

---

## ✅ Checklist for New Simulations

- [ ] Modify geometry in `MotorGeometry` class
- [ ] Run mesh generator
- [ ] Inspect mesh quality (> 0.3 average)
- [ ] Update simulation config in `SimulationConfig`
- [ ] Run solver (check convergence)
- [ ] Extract air gap field
- [ ] Visualize in ParaView

---

## 📝 Notes

- All files are standalone (no dependencies between scripts except data files)
- Results files (.h5, .xdmf) can be large (> 100 MB)
- Mesh can be reused for multiple simulations
- ParaView recommended for 3D visualization

---

## 🤝 Contributing

When adding features:
1. Follow the class-based structure
2. Add configuration classes for parameters
3. Include docstrings and comments
4. Update this README

---

**Last Updated:** October 28, 2025
**Version:** 2.0 (Cleaned and Reorganized)

