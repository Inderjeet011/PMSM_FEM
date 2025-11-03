#!/usr/bin/env python3
"""
Plot a bar chart of time-averaged RMS |B| in the air gap from airgap_rms_period.csv.
"""

import argparse
import csv
import math
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_rms(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def label_for(fname: str) -> str:
    base = os.path.basename(fname)
    mapping = {
        "results_2d_coils_only.h5": "Coils Only (J≠0, B_rem=0)",
        "results_2d_pm_only.h5": "Magnets Only (J=0, B_rem=1.4T)",
        "results_2d_both.h5": "Both (PM + Coils)",
    }
    return mapping.get(base, base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bar chart of air-gap RMS |B| (T)")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "../results/airgap_rms_period.csv"),
        help="Path to airgap_rms_period.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "../results/airgap_rms_bar.png"),
        help="Output image path",
    )
    parser.add_argument("--ylim", type=float, default=None, help="Optional y max (Tesla)")
    args = parser.parse_args()

    rows = read_rms(args.csv)
    # Enforce desired order: coils-only, magnets-only, both
    desired_order = [
        "results_2d_coils_only.h5",
        "results_2d_pm_only.h5",
        "results_2d_both.h5",
    ]
    order_index = {name: i for i, name in enumerate(desired_order)}
    rows.sort(key=lambda r: order_index.get(r.get("file", ""), 999))

    labels: List[str] = []
    vals: List[float] = []
    for r in rows:
        f = r.get("file", "")
        try:
            v = float(r.get("airgap_rms_T", "nan"))
        except ValueError:
            v = math.nan
        if not f or math.isnan(v):
            continue
        labels.append(label_for(f))
        vals.append(v)

    if not labels:
        raise SystemExit("No valid rows to plot.")

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, vals, color="#1f77b4", edgecolor="black")
    plt.title("Air-gap RMS |B| over 20 ms (50 Hz)")
    plt.ylabel("|B|_RMS [Tesla]")
    plt.xticks(rotation=15, ha="right")

    ymax = args.ylim if args.ylim is not None else max(vals) * 1.25
    plt.ylim(0, ymax)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02*ymax, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


