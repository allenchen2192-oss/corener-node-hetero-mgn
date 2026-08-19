"""
analyze_stress_distribution_m0040.py
======================================
Box plot: each sample (220 total) → one box showing the distribution of
element-level vm_stress at the FINAL timestep (t=20) within each material.

Layout: 5 stacked subplots  (Displacement | Si | Solder | UF | PEEQ)
X-axis: sample index S0001–S0220  (blue = train S0001-S0200, red = test S0201-S0220)
Y-axis: disp [mm], vm_stress [MPa], peeq [-]

Answers: "Is the stress distribution similar across all 220 samples?"
  - If boxes cluster tightly at the same level → trivial to learn (constant predictor works)
  - If median / spread varies widely across samples → meaningful variation → harder task

Usage:
  python analyze_stress_distribution_m0040.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

NPZ_DIR = "./02_abaqus_npz_m0040"
OUT_PNG = "./stress_distribution_m0040.png"

SAMPLES = [f"S{i:04d}" for i in range(1, 221)]   # S0001 … S0220
N_TRAIN = 200   # S0001-S0200 = train,  S0201-S0220 = test

# ── Load final-step element stress per material ───────────────────────────────

disp_boxes   = []   # node displacement magnitude at final step (all nodes)
si_boxes     = []   # list of arrays, one per sample
solder_boxes = []
uf_boxes     = []
peeq_boxes   = []   # PEEQ at solder elements at final step

print("Loading final-step data from 220 NPZ files ...")
for sid in tqdm(SAMPLES):
    path = os.path.join(NPZ_DIR, f"{sid}.npz")
    if not os.path.exists(path):
        # Insert NaN placeholder so indexing stays aligned
        disp_boxes.append(np.array([np.nan]))
        si_boxes.append(np.array([np.nan]))
        solder_boxes.append(np.array([np.nan]))
        uf_boxes.append(np.array([np.nan]))
        peeq_boxes.append(np.array([np.nan]))
        continue

    d        = np.load(path)
    vm_t20   = d["elem_vm_stress"][20].astype(np.float32)   # (M,) final step
    elem_mat = d["elem_mat"]                                  # (M,)

    # Displacement: magnitude from t=0 to t=20 for all nodes
    world_pos = d["world_pos"]                               # (T+1, N, 3)
    disp      = np.linalg.norm(
        world_pos[20].astype(np.float32) - world_pos[0].astype(np.float32), axis=1
    )                                                         # (N,)
    disp_boxes.append(disp)

    si_boxes.append(    vm_t20[(elem_mat == 0) | (elem_mat == 1)])   # already MPa
    solder_boxes.append(vm_t20[ elem_mat == 2])
    uf_boxes.append(    vm_t20[ elem_mat == 3])

    # PEEQ: solder elements only
    peeq_t20 = d["elem_peeq"][20].astype(np.float32)        # (M,)
    peeq_boxes.append(peeq_t20[elem_mat == 2])

# ── Summary stats ─────────────────────────────────────────────────────────────

def stats(boxes, label, unit="MPa"):
    medians = np.array([np.median(b) for b in boxes if not np.isnan(b).all()])
    mins    = np.array([b.min()      for b in boxes if not np.isnan(b).all()])
    maxs    = np.array([b.max()      for b in boxes if not np.isnan(b).all()])
    print(f"\n{label}:")
    print(f"  Median across samples : {medians.min():.4g} – {medians.max():.4g} {unit}"
          f"  (range {medians.max()-medians.min():.4g} {unit})")
    print(f"  Element min / max     : {mins.min():.4g} / {maxs.max():.4g} {unit}")
    print(f"  CV of medians         : {medians.std()/medians.mean()*100:.1f}%")

stats(disp_boxes,   "Displacement", unit="mm")
stats(si_boxes,     "Si",           unit="MPa")
stats(solder_boxes, "Solder",       unit="MPa")
stats(uf_boxes,     "UF",           unit="MPa")
stats(peeq_boxes,   "PEEQ (solder)", unit="-")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(5, 1, figsize=(44, 22), sharex=True)
fig.suptitle(
    "M0040 — Distribution at Final Step (t=20)\n"
    "Each box = one sample, whiskers = 5th–95th percentile",
    fontsize=14, fontweight="bold"
)

COLORS = {
    "Disp":   ("#1a7a4a", "#6ecf9e"),   # green tones
    "Si":     ("#2980b9", "#85c1e9"),   # (train, test)
    "Solder": ("#c0392b", "#f1948a"),
    "UF":     ("#7d3c98", "#c39bd3"),
    "PEEQ":   ("#d35400", "#f0a868"),   # orange tones
}

PANEL_DATA = [
    (disp_boxes,   "Disp",   "Displacement (mm)"),
    (si_boxes,     "Si",     "VM Stress — Si (MPa)"),
    (solder_boxes, "Solder", "VM Stress — Solder (MPa)"),
    (uf_boxes,     "UF",     "VM Stress — UF (MPa)"),
    (peeq_boxes,   "PEEQ",   "PEEQ — Solder (-)"),
]

for ax, (boxes, mat, ylabel) in zip(axes, PANEL_DATA):
    c_train, c_test = COLORS[mat]
    positions = np.arange(1, len(SAMPLES) + 1)

    # Separate train / test for colouring
    train_pos  = [p for p, s in zip(positions, SAMPLES) if int(s[1:]) <= N_TRAIN]
    test_pos   = [p for p, s in zip(positions, SAMPLES) if int(s[1:]) >  N_TRAIN]
    train_data = [b for b, s in zip(boxes, SAMPLES) if int(s[1:]) <= N_TRAIN]
    test_data  = [b for b, s in zip(boxes, SAMPLES) if int(s[1:]) >  N_TRAIN]

    bp_train = ax.boxplot(
        train_data, positions=train_pos,
        widths=0.7, patch_artist=True,
        whis=[5, 95], showfliers=False,
        boxprops=dict(facecolor=c_train, alpha=0.7),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=c_train, linewidth=0.8),
        capprops=dict(color=c_train, linewidth=0.8),
    )
    bp_test = ax.boxplot(
        test_data, positions=test_pos,
        widths=0.7, patch_artist=True,
        whis=[5, 95], showfliers=False,
        boxprops=dict(facecolor=c_test, alpha=0.9),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=c_test, linewidth=0.8),
        capprops=dict(color=c_test, linewidth=0.8),
    )

    # Vertical separator train / test
    ax.axvline(x=N_TRAIN + 0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(N_TRAIN + 1.5, ax.get_ylim()[0], "test →",
            fontsize=8, color="gray", va="bottom")

    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", labelsize=6)

    # X-tick: every 20 samples
    tick_pos = list(range(1, len(SAMPLES) + 1, 20)) + [len(SAMPLES)]
    tick_lbl = [SAMPLES[p - 1] for p in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right")

    # Legend
    patch_train = mpatches.Patch(color=c_train, alpha=0.8, label=f"Train (S0001–S{N_TRAIN:04d})")
    patch_test  = mpatches.Patch(color=c_test,  alpha=0.9, label=f"Test  (S{N_TRAIN+1:04d}–S0220)")
    ax.legend(handles=[patch_train, patch_test], fontsize=9, loc="upper left")


axes[-1].set_xlabel("Sample", fontsize=11)
axes[-1].set_xlim(0, len(SAMPLES) + 1)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
