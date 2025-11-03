#!/usr/bin/env python3
"""
Compute air-gap mean |B| per timestep for the three standardized cases
and write a CSV at results/airgap_means_timeseries.csv.
"""

import csv
import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import fem
import ufl

CASES = [
    ("coils_only", "results_2d_current_coils_only.h5"),
    ("pm_only", "results_2d_current_pm_only.h5"),
    ("both", "results_2d_current_both.h5"),
]


def load_mesh(mesh_path: str):
    mesh, ct, *_ = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, rank=0, gdim=2)
    return mesh, ct


def compute_means_for_file(mesh, ct, h5_path: str) -> List[Tuple[int, float]]:
    V_scalar = fem.functionspace(mesh, ("CG", 1))
    V_vector = fem.functionspace(mesh, ("DG", 0, (2,)))
    Az = fem.Function(V_scalar)
    B_vec = fem.Function(V_vector)
    expr = fem.Expression(ufl.as_vector((Az.dx(1), -Az.dx(0))), V_vector.element.interpolation_points)

    airgap_cells = np.concatenate([ct.find(5), ct.find(6)])

    means: List[Tuple[int, float]] = []
    with h5py.File(h5_path, "r") as h5f:
        f0 = h5f["Function"]["f_0"]
        timesteps = sorted([k for k in f0.keys() if k.startswith("0_")])
        for idx, ts in enumerate(timesteps, start=1):
            data = f0[ts][:]
            Az.x.array[:] = data.flatten()[: len(Az.x.array)]
            B_vec.interpolate(expr)
            Bx = B_vec.sub(0).collapse().x.array
            By = B_vec.sub(1).collapse().x.array
            Bmag = np.sqrt(Bx**2 + By**2)
            vals = Bmag[airgap_cells]
            mean_val = float(np.mean(vals))
            means.append((idx, mean_val))
    return means


def main() -> None:
    base = os.path.dirname(__file__)
    mesh_path = os.path.join(base, "../motor.msh")
    results_dir = os.path.join(base, "../results")
    out_csv = os.path.join(results_dir, "airgap_means_timeseries.csv")

    mesh, ct = load_mesh(mesh_path)

    rows: List[Dict[str, object]] = []
    for label, fname in CASES:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        means = compute_means_for_file(mesh, ct, path)
        # Assuming dt=1 ms as used; if different, we could parse time from XDMF, but here use index*1.0 ms
        for idx, mean_val in means:
            time_ms = float(idx)  # 1 ms increments
            rows.append({
                "case": label,
                "timestep": idx,
                "time_ms": time_ms,
                "airgap_mean_T": mean_val,
            })

    os.makedirs(results_dir, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "timestep", "time_ms", "airgap_mean_T"]) 
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
