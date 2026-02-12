#!/bin/bash
# Generate mesh and run submesh solver.
# Default: 2mm near motor, grading to coarser elements with distance.
# Usage: ./run_coarse.sh [res]   e.g. ./run_coarse.sh 0.005  (5mm), ./run_coarse.sh 0.003 (3mm)
RES=${1:-0.005}
LCRATIO=${2:-40}
cd "$(dirname "$0")/../3d"
echo "=== Generating mesh: res=${RES}m, lc-max-ratio=${LCRATIO} ==="
# Use --no-optimize for faster mesh generation (Netgen optimization can take 10+ min for fine meshes)
python mesh_3D.py --res "$RES" --lc-max-ratio "$LCRATIO" --no-optimize
cd ../3d_submesh
echo "=== Running solver ==="
python main_submesh.py
