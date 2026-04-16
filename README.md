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


### Method 1: Docker workflow
Setup

Use either of the following methods.

Pull Python image
```bash
docker pull python:3.10
```

Start container
```bash
docker run -it python:3.10 bash
```

Install Miniconda
```bash
cd ~
apt update && apt install -y wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
```

```bash
Create and activate environment
conda create -n pmsm -c conda-forge python=3.12
conda activate pmsm
```

Install FEM dependencies
```bash
conda install -c conda-forge fenics-dolfinx=0.10.0 fenics-basix=0.10.0 fenics-ufl=2025.2.1 petsc4py=3.24.3 mpi4py=4.1.1
conda install -c conda-forge python-gmsh
```

Verify installation
```bash
python -c "import dolfinx, gmsh; print('OK')"
```

### Method 2: Docker workflow

Pull image:

```bash
docker pull jeet0003/my-app:v1
```

Run container:

```bash
docker run -it --name pmsm_container jeet0003/my-app:v1 bash
```

Inside the container:

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/Inderjeet011/PMSM_FEM.git
cd PMSM_FEM
conda activate pmsm
python main.py
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
