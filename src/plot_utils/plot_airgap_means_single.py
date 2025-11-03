#!/usr/bin/env python3
"""
Plot a simple 3-bar chart: one bar per case (Coils Only, Magnets Only, Both),
using the average of air-gap mean |B| over time from airgap_means_timeseries.csv.
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_and_average(csv_path: str) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["case"].strip()
            try:
                val = float(row["airgap_mean_T"])
            except Exception:
                continue
            sums[case] += val
            counts[case] += 1
    return {k: (sums[k] / counts[k]) for k in sums if counts[k] > 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-bar chart of avg air-gap mean |B| per case")
    base_dir = os.path.dirname(__file__)
    parser.add_argument(
        "--csv",
        default=os.path.join(base_dir, "../results/airgap_means_timeseries.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(base_dir, "../results/airgap_means_single.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    avg = read_and_average(args.csv)
    order = ["coils_only", "pm_only", "both"]
    labels = ["Coils Only", "Magnets Only", "Both"]
    values: List[float] = [avg.get(k, 0.0) for k in order]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color="#1f77b4", edgecolor="black")
    plt.ylabel("Average Mean |B| [T]")
    plt.title("Air-gap Mean |B| (averaged over time)")
    ymax = max(values) * 1.25 if values else 1.0
    plt.ylim(0, ymax)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02*ymax, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
