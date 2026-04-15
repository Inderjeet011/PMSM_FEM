# PMSM_FEM

Transient **A–V** eddy-current models for PMSMs using **FEniCSx / DOLFINx**: one **2D** driver and three **3D** drivers (coil geometry / excitation differ).

## Layout

| Path | What it is |
|------|------------|
| `src/2d/` | 2D A–V (`Az`, `V`): `solve.py` |
| `src/3d_volume_coils/` | 3D: **bulk `J_z`** (three-phase) in coil volumes |
| `src/3d_loop_coils/` | 3D: **six loops**, **terminal voltages** |
| `src/3d_rod_coils/` | 3D: **rod-style** coils, same voltage drive as loop |

**Outputs** (when `write_results` in `utils.make_config()`): `result.xdmf` / `result.h5`, plus VTX folders **`V.bp`**, **`J.bp`**, **`B.bp`** (motor-only B). New runs usually wipe prior outputs in that folder.

## Requirements

`dolfinx`, `petsc4py`, `mpi4py`, `numpy`, `gmsh` — see **`requirements.txt`**. Prefer a **conda-forge** DOLFINx env or the official **DOLFINx** image so PETSc/HDF5 match.

```bash
python -c "import dolfinx, gmsh; print('OK')"
```

## Run

Generate mesh and solver **from the case directory** (paths are relative to each `main.py`).

```bash
cd src/3d_loop_coils
python mesh.py --res 0.005 --depth 0.057   # python mesh.py --help
python main.py
```

```bash
cd src/2d && python solve.py
```

**MPI:** `mpirun -np 4 python main.py` (same cwd as `mesh.xdmf`).

## Config & files

- **Tuning:** `utils.py` → `make_config()` (`dt`, `num_steps`, `V_amp` / currents, `write_results`, KSP, paths).
- **Materials / tags:** `mesh.py` → `model_parameters`; **`load_mesh.setup_materials`**. Typical 3D: air, airgap, rotor (tags 4–5 may both map to rotor on old/new meshes), stator, coils ~7–12, PMs ~13–22 — confirm in each `mesh.py`.
- **Per 3D folder:** `mesh.py`, `load_mesh.py`, `entity_map.py`, `forms.py`, `utils.py`, `main.py`.
