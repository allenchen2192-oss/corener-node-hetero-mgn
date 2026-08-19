"""
plot_loss_curve.py
==================
Read TensorBoard event files and plot loss curves (log scale).

Usage:
  python plot_loss_curve.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ────────────────────────────────────────────────────────────────────

LOG_DIR    = "./tensorboard_logs_prevvel"
OUTPUT     = "./loss_curve.png"
TAGS       = ["loss_vel", "loss_stress", "loss_total"]   # tags to plot
SMOOTH     = 50    # moving average window (set 1 to disable)

# ── Read events ───────────────────────────────────────────────────────────────

ea = EventAccumulator(LOG_DIR)
ea.Reload()

available = ea.Tags().get("scalars", [])
print(f"Available tags: {available}")

tags_to_plot = [t for t in TAGS if t in available]
if not tags_to_plot:
    tags_to_plot = available
    print(f"Requested tags not found; plotting all: {tags_to_plot}")

# ── Smooth helper ─────────────────────────────────────────────────────────────

def moving_avg(y, w):
    if w <= 1 or len(y) < w:
        return y
    kernel = [1.0 / w] * w
    pad = w // 2
    y_pad = [y[0]] * pad + list(y) + [y[-1]] * pad
    return [sum(y_pad[i:i+w]) / w for i in range(len(y))]

# ── Build series dict (tag -> (steps, values)) ────────────────────────────────

series = {}
for tag in tags_to_plot:
    events = ea.Scalars(tag)
    series[tag] = (
        [e.step  for e in events],
        [e.value for e in events],
    )

# Compute loss_total = loss_vel + 0.5 * loss_stress if not directly available
W_STRESS = 0.5
if "loss_total" not in series and "loss_vel" in series and "loss_stress" in series:
    steps_v, vals_v = series["loss_vel"]
    steps_s, vals_s = series["loss_stress"]
    # align on common steps
    sv = dict(zip(steps_v, vals_v))
    ss = dict(zip(steps_s, vals_s))
    common = sorted(set(steps_v) & set(steps_s))
    total_vals = [sv[t] + ss[t] for t in common]
    series["loss_total"] = (common, total_vals)
    print("loss_total computed as loss_vel + loss_stress")

plot_order = ["loss_vel", "loss_stress", "loss_total"]
plot_series = [(t, *series[t]) for t in plot_order if t in series]

# ── Plot ──────────────────────────────────────────────────────────────────────

colors = {"loss_vel": "tab:blue", "loss_stress": "tab:orange", "loss_total": "tab:green"}

fig, ax = plt.subplots(figsize=(10, 5))

for tag, steps, values in plot_series:
    smoothed = moving_avg(values, SMOOTH)
    c = colors.get(tag, "tab:gray")
    ax.plot(steps, values,   color=c, alpha=0.2, linewidth=0.8)
    ax.plot(steps, smoothed, color=c, linewidth=2, label=tag)

ax.set_yscale("log")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (log scale)", fontsize=12)
ax.set_title("Training Loss Curve", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
