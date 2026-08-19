#!/usr/bin/env python3
"""Generate case_table.csv for analytic cube dataset v3.

Design choices agreed in chat:
- 100 train-only cases
- seed = 42
- 12 affine coefficients sampled from U(-0.02, 0.02)
- write coefficients rounded to 6 decimals in CSV
- H = 1, num_frames = 21, n_nodes_axis = 4
- time_profile = quarter_sine
- E = 70e9, nu = 0.33

Default output layout:
    <script_dir>/cases_v3/00_manifest/case_table.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HEADER = [
    "case_id",
    "ax0", "ax1", "ax2", "ax3",
    "ay0", "ay1", "ay2", "ay3",
    "az0", "az1", "az2", "az3",
    "H",
    "num_frames",
    "n_nodes_axis",
    "time_profile",
    "E",
    "nu",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir / "cases_v3" / "00_manifest" / "case_table.csv"

    parser = argparse.ArgumentParser(description="Generate analytic cube v3 case table.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output case_table.csv path.")
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coeff-min", type=float, default=-0.02)
    parser.add_argument("--coeff-max", type=float, default=0.02)
    parser.add_argument("--round-decimals", type=int, default=6)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=21)
    parser.add_argument("--n-nodes-axis", type=int, default=4)
    parser.add_argument("--time-profile", type=str, default="quarter_sine")
    parser.add_argument("--E", type=float, default=70e9)
    parser.add_argument("--nu", type=float, default=0.33)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError("--num-cases must be positive")
    if not args.coeff_min < args.coeff_max:
        raise ValueError("Require coeff_min < coeff_max")
    if args.n_nodes_axis < 2:
        raise ValueError("n_nodes_axis must be at least 2")
    if args.num_frames < 2:
        raise ValueError("num_frames must be at least 2")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    coeffs = rng.uniform(args.coeff_min, args.coeff_max, size=(args.num_cases, 12))
    coeffs = np.round(coeffs, args.round_decimals)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for i in range(args.num_cases):
            row = [f"case_{i:05d}"]
            row.extend(f"{v:.{args.round_decimals}f}" for v in coeffs[i])
            row.extend([
                f"{args.H:g}",
                str(args.num_frames),
                str(args.n_nodes_axis),
                args.time_profile,
                f"{args.E:.1f}",
                f"{args.nu:g}",
            ])
            writer.writerow(row)

    print(f"[OK] Wrote {args.num_cases} cases to: {args.output}")
    print(f"[INFO] seed={args.seed}, coeff_range=[{args.coeff_min}, {args.coeff_max}], decimals={args.round_decimals}")


if __name__ == "__main__":
    main()
