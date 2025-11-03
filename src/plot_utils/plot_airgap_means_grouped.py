#!/usr/bin/env python3
"""
Grouped bar chart of air-gap mean |B| vs time for three cases using
results/airgap_means_timeseries.csv. Writes results/airgap_means_grouped.png
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_series(csv_path: str) -> Dict[str, Dict[float, float]]:
    data: Dict[str, Dict[float, float]] = defaultdict(dict)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["case"].strip()
            t = float(row["time_ms"]) if row.get("time_ms") else 0.0
            val = float(row["airgap_mean_T"]) if row.get("airgap_mean_T") else 0.0
            data[case][t] = val
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped bar chart of air-gap mean |B| vs time")
    base_dir = os.path.dirname(__file__)
    parser.add_argument(
        "--csv",
        default=os.path.join(base_dir, "../results/airgap_means_timeseries.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(base_dir, "../results/airgap_means_grouped.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    series = read_series(args.csv)
    order = ["coils_only", "pm_only", "both"]
    labels = {
        "coils_only": "Coils Only",
        "pm_only": "Magnets Only",
        "both": "Both",
    }

    # Collect all times
    all_times = sorted({t for case in series.values() for t in case.keys()})
    x = np.arange(len(all_times))
    width = 0.25

    plt.figure(figsize=(12, 6))

    colors = {"coils_only": "#1f77b4", "pm_only": "#ff7f0e", "both": "#2ca02c"}
    offsets = {"coils_only": -width, "pm_only": 0.0, "both": width}

    for key in order:
        vals = [series.get(key, {}).get(t, 0.0) for t in all_times]
        plt.bar(x + offsets[key], vals, width=width, label=labels[key], color=colors[key], edgecolor="black")

    plt.xticks(x, [f"{t:.0f}" for t in all_times])
    plt.xlabel("Time [ms]")
    plt.ylabel("Mean |B| [T]")
    plt.title("Air-gap Mean |B| vs Time (Grouped)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
