"""
Plot ground truth velocity vs time step for a given sample.
Usage:
    python plot_velocity.py --sample_idx 29
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from physicsnemo.datapipes.gnn.utils import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=29)
    parser.add_argument("--pt_dir",     default="04_preprocessed_pt")
    parser.add_argument("--stats",      default="04_preprocessed_pt/node_stats.json")
    parser.add_argument("--output",     default="velocity_vs_timestep.png")
    args = parser.parse_args()

    pt_path = os.path.join(args.pt_dir, f"sample_{args.sample_idx:05d}.pt")
    print(f"Loading {pt_path} ...")
    sample = torch.load(pt_path, map_location="cpu", weights_only=False)

    # load normalization stats
    node_stats = load_json(args.stats)
    vel_mean = node_stats["velocity_mean"]  # (3,)
    vel_std  = node_stats["velocity_std"]   # (3,)

    num_steps = len(sample)
    timesteps = np.arange(num_steps)

    # collect per-step velocity stats (denormalized)
    vel_mean_per_step = []
    vel_max_per_step  = []
    vel_mag_mean      = []
    vel_mag_max       = []

    for t, item in enumerate(sample):
        graph = item["graph"]
        # graph.y[:, 0:3] = normalized velocity
        vel_norm = graph.y[:, :3]                          # (N, 3)
        vel      = vel_norm * vel_std + vel_mean           # denormalize

        # only free nodes (col 0 of one-hot)
        node_type = graph.x[:, :3]
        free_mask = node_type[:, 0] > 0.5

        vel_free = vel[free_mask]                          # (N_free, 3)
        mag      = torch.linalg.norm(vel_free, dim=-1)    # (N_free,)

        vel_mag_mean.append(mag.mean().item())
        vel_mag_max.append(mag.max().item())

    vel_mag_mean = np.array(vel_mag_mean)
    vel_mag_max  = np.array(vel_mag_max)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timesteps, vel_mag_mean, label="Mean |velocity| (free nodes)", linewidth=2.5)
    ax.plot(timesteps, vel_mag_max,  label="Max  |velocity| (free nodes)", linewidth=2.5, linestyle="--")
    ax.set_xlabel("Time Step", fontsize=18)
    ax.set_ylabel("Velocity Magnitude (m/step)", fontsize=18)
    ax.set_title(f"Ground Truth Velocity vs Time Step  (sample {args.sample_idx})",
                 fontsize=20, fontweight="bold")
    ax.legend(fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
