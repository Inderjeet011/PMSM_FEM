# ⚡ PMSM_FEM  
Transient **A–V eddy-current simulations** for Permanent Magnet Synchronous Motors (PMSMs) using **FEniCSx / DOLFINx**

---

## 📌 Overview

This project implements high-fidelity **finite element models (FEM)** for simulating **eddy currents and electromagnetic fields** in PMSMs using the **A–V formulation**.

It supports both **2D and 3D transient simulations** with different coil modeling strategies.

---

## 🧠 Key Features

- Transient **A–V formulation** (`A`, `V`)
- 2D and multiple 3D configurations
- Coil modeling:
  - Volume current excitation
  - Loop-based voltage excitation
  - Rod-based coils
- Configurable solver (time-stepping, PETSc)
- MPI parallel support
- ParaView-ready outputs

---

## 📁 Project Structure


PMSM_FEM/
│
├── src/
│ ├── 2d/
│ ├── 3d_volume_coils/
│ ├── 3d_loop_coils/
│ ├── 3d_rod_coils/
│
└── requirements.txt



---

## 🚀 Running the Project

### ▶️ 3D Example

```bash
cd src/3d_loop_coils
python mesh.py --res 0.005 --depth 0.057
python main.py

Parallel Execution
mpirun -np 4 python main.py

📊 Outputs
result.xdmf, result.h5
V.bp, J.bp, B.bp


⚙️ Setup Methods
🐳 Method 1: Docker (Recommended)
Pull image
docker pull jeet0003/my-app:v2
Run container
docker run -it --name pmsm_container jeet0003/my-app:v2 bash
Setup workspace
mkdir -p /workspace
cd /workspace
Clone repo
git clone https://github.com/Inderjeet011/PMSM_FEM.git
cd PMSM_FEM
Activate environment
conda activate pmsm
🧪 Method 2: Miniconda (Tested Working)
Install Miniconda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

Check:

conda --version
Create environment
conda create -n pmsm -c conda-forge python=3.12
conda activate pmsm
Install DOLFINx stack
conda install -c conda-forge \
  fenics-dolfinx=0.10.0 \
  fenics-basix=0.10.0 \
  fenics-ufl=2025.2.1 \
  petsc4py=3.24.3 \
  mpi4py=4.1.1
Install Gmsh
conda install -c conda-forge python-gmsh
Verify
python -c "import dolfinx, gmsh; print('Setup OK')"
