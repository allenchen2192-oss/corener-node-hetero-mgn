"""
mgn_flowchart.py
HeteroMGN architecture flowchart.
Left: overall pipeline  |  Right: processor block detail
Output: mgn_flowchart.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# ── Palette ───────────────────────────────────────────────────────────────────
C_NODE   = "#E8824A"   # orange  corner node
C_ELEM   = "#3A82C4"   # blue    element node
C_EDGE   = "#8B30B0"   # purple  edge
C_N2N    = "#8B30B0"
C_N2E    = "#3A3A3A"
C_E2N    = "#C05000"
C_PROC   = "#FFF3E0"
C_PROC_BD= "#D35400"
C_DEC    = "#E8F5E9"
C_DEC_BD = "#2E7D32"
C_ROLL   = "#FCE4EC"
C_ROLL_BD= "#C62828"
C_PANEL  = "#F8F9FF"
C_ENC    = "#EEF3FB"
C_ENC_BD = "#3A82C4"

fig = plt.figure(figsize=(24, 15))
ax  = fig.add_axes([0.01, 0.01, 0.98, 0.96])
ax.set_xlim(0, 24)
ax.set_ylim(0, 15)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# ── Helpers ───────────────────────────────────────────────────────────────────

def box(cx, cy, w, h, text, sub=None,
        fc="#EEF3FB", ec="#3A82C4", lw=2.0,
        fs=11, fw="bold", tc="#1a1a1a",
        sfs=8.5, stc="#555"):
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.08", facecolor=fc, edgecolor=ec, lw=lw, zorder=4))
    dy = 0.18 if sub else 0
    ax.text(cx, cy+dy, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, zorder=5)
    if sub:
        ax.text(cx, cy-0.22, sub, ha="center", va="center",
                fontsize=sfs, color=stc, style="italic", zorder=5)

def arr(x1, y1, x2, y2, color="#555", lw=1.8, ls="-",
        rad=0.0, label=None, lfs=8.5, lside="right"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle=ls, mutation_scale=12,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3, shrinkB=3))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset = 0.18 if lside == "right" else -0.18
        ha = "left" if lside == "right" else "right"
        ax.text(mx + offset, my, label,
                ha=ha, va="center", fontsize=lfs, color=color)


# ════════════════════════════════════════════════════════════════════════════
# LEFT — Main Pipeline
# ════════════════════════════════════════════════════════════════════════════
LX, LW, LH = 4.2, 5.0, 1.05

ax.text(LX, 14.6, "Overall Pipeline", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#1a1a1a")

# [1] Input
box(LX, 13.5, LW, LH,
    "Input Graph",
    "Corner Nodes (11D)  |  Element Nodes (8D)  |  Edges (8D)",
    fc="#F5F5F5", ec="#888", fs=12)

arr(LX, 12.97, LX, 12.47, label="raw features")

# [2] Encoder — three sub-boxes
EY = 11.75
ax.add_patch(FancyBboxPatch((LX-LW/2, EY-0.75), LW, 1.45,
    boxstyle="round,pad=0.08", facecolor=C_ENC, edgecolor=C_ENC_BD, lw=2.0, zorder=3))
ax.text(LX, EY+0.48, "Encoder", ha="center", fontsize=12,
        fontweight="bold", color=C_ENC_BD, zorder=5)
for i, (lbl, sub, c) in enumerate([
    ("Node Enc", "11D→128D", C_NODE),
    ("Elem Enc", "8D→128D",  C_ELEM),
    ("Edge Enc", "8D→128D",  C_EDGE),
]):
    sx = LX - 1.65 + i * 1.65
    ax.add_patch(FancyBboxPatch((sx-0.72, EY-0.60), 1.44, 0.72,
        boxstyle="round,pad=0.05", facecolor=c, edgecolor="white",
        alpha=0.9, lw=1.5, zorder=5))
    ax.text(sx, EY-0.14, lbl, ha="center", fontsize=8.5,
            fontweight="bold", color="white", zorder=6)
    ax.text(sx, EY-0.44, sub, ha="center", fontsize=7.5,
            color="white", zorder=6)

arr(LX, 11.0, LX, 10.5, label="h_node, h_elem, h_edge (128D each)")

# [3] Processor × K
PY = 9.75
ax.add_patch(FancyBboxPatch((LX-LW/2, PY-0.75), LW, 1.45,
    boxstyle="round,pad=0.08", facecolor=C_PROC, edgecolor=C_PROC_BD,
    lw=2.5, zorder=3))
ax.text(LX, PY+0.44, "Processor Block  × K", ha="center",
        fontsize=12, fontweight="bold", color=C_PROC_BD, zorder=5)
ax.text(LX, PY+0.04, "NodeEdgeConv (n2n)  |  SAGEConv (n2e)  |  SAGEConv (e2n)",
        ha="center", fontsize=8.5, color="#555", zorder=5)
ax.text(LX, PY-0.38, "Residual MLP update for h_node and h_elem",
        ha="center", fontsize=8, color="#888", style="italic", zorder=5)

# Dashed arrow pointing right to detail
ax.annotate("", xy=(8.7, PY), xytext=(LX+LW/2, PY),
    arrowprops=dict(arrowstyle="-|>", color=C_PROC_BD, lw=2.2,
                    linestyle="dashed", mutation_scale=14))
ax.text(7.5, PY+0.2, "see detail →", ha="center", fontsize=9,
        color=C_PROC_BD, fontweight="bold")

arr(LX, 9.0, LX, 8.5, label="h_node', h_elem' (128D)")

# [4] Decoder
DY = 7.75
ax.add_patch(FancyBboxPatch((LX-LW/2, DY-0.75), LW, 1.45,
    boxstyle="round,pad=0.08", facecolor=C_DEC, edgecolor=C_DEC_BD, lw=2.0, zorder=3))
ax.text(LX, DY+0.48, "Decoder", ha="center", fontsize=12,
        fontweight="bold", color=C_DEC_BD, zorder=5)
for i, (lbl, sub, c) in enumerate([
    ("Node Dec", "→ vel (3D)",       C_NODE),
    ("Elem Dec", "→ σ_VM + ε_p  (2D)", C_ELEM),
]):
    sx = LX - 1.2 + i * 2.4
    ax.add_patch(FancyBboxPatch((sx-1.0, DY-0.60), 2.0, 0.72,
        boxstyle="round,pad=0.05", facecolor=c, edgecolor="white",
        alpha=0.9, lw=1.5, zorder=5))
    ax.text(sx, DY-0.13, lbl, ha="center", fontsize=9,
            fontweight="bold", color="white", zorder=6)
    ax.text(sx, DY-0.44, sub, ha="center", fontsize=8, color="white", zorder=6)

arr(LX, 7.0, LX, 6.5, label="vel,  σ_VM,  ε_p")

# [5] Rollout
RLY = 5.7
ax.add_patch(FancyBboxPatch((LX-LW/2, RLY-1.05), LW, 2.0,
    boxstyle="round,pad=0.08", facecolor=C_ROLL, edgecolor=C_ROLL_BD,
    lw=2.2, zorder=3))
ax.text(LX, RLY+0.60, "Rollout (Autoregressive)", ha="center",
        fontsize=12, fontweight="bold", color=C_ROLL_BD, zorder=5)
ax.text(LX, RLY+0.22, "pos  ←  pos + vel", ha="center",
        fontsize=9.5, color="#333", family="monospace", zorder=5)
ax.text(LX, RLY-0.08, "prev_stress  ←  pred_stress", ha="center",
        fontsize=9.5, color="#333", family="monospace", zorder=5)
ax.text(LX, RLY-0.38, "prev_peeq  ←  pred_peeq", ha="center",
        fontsize=9.5, color="#333", family="monospace", zorder=5)
ax.text(LX, RLY-0.70, "t = 1, 2, … , T   (teacher-forced seed at t=1)",
        ha="center", fontsize=8.5, color="#888", style="italic", zorder=5)

# Feedback arrow: Rollout left side → loop around left → Input Graph top
_fx_left  = LX - LW/2        # left edge of boxes (= 1.7)
_fy_roll  = RLY               # vertical mid of Rollout left edge
_fx_loop  = 0.45              # how far left to extend
_fy_top   = 14.22             # above Input Graph top edge
_iy_top   = 13.5 + LH/2      # top edge of Input Graph box

# 3-segment L-shaped path: left → up → right
ax.plot([_fx_left, _fx_loop, _fx_loop, LX],
        [_fy_roll, _fy_roll, _fy_top,  _fy_top],
        color=C_ROLL_BD, lw=1.8, zorder=3,
        solid_capstyle="round", solid_joinstyle="round")
# Arrowhead pointing down into Input Graph top
ax.annotate("", xy=(LX, _iy_top), xytext=(LX, _fy_top),
    arrowprops=dict(arrowstyle="-|>", color=C_ROLL_BD, lw=1.8,
                    mutation_scale=12, shrinkA=0, shrinkB=2))
ax.text(_fx_loop - 0.32, (_fy_roll + _fy_top) / 2, "next\nstep",
        ha="center", va="center", fontsize=8.5, color=C_ROLL_BD,
        fontweight="bold")


# ════════════════════════════════════════════════════════════════════════════
# RIGHT — Processor Block Detail
# ════════════════════════════════════════════════════════════════════════════
# Panel
ax.add_patch(FancyBboxPatch((9.1, 2.5), 14.2, 12.0,
    boxstyle="round,pad=0.12", facecolor=C_PANEL, edgecolor=C_PROC_BD,
    lw=2.2, linestyle="--", zorder=1))
ax.text(16.2, 14.6, "Processor Block — One Iteration",
        ha="center", fontsize=14, fontweight="bold", color=C_PROC_BD)

NX, ELX = 13.5, 19.0   # node-side x, elem-side x
MX = (NX + ELX) / 2    # mid x

# ── (A) Inputs ────────────────────────────────────────────────────────────────
IY = 13.5
for cx, lbl, c in [(NX, "h_node  (128D)", C_NODE),
                   (ELX, "h_elem  (128D)", C_ELEM)]:
    ax.add_patch(FancyBboxPatch((cx-1.45, IY-0.38), 2.9, 0.76,
        boxstyle="round,pad=0.06", facecolor=c, edgecolor="white",
        alpha=0.9, lw=1.8, zorder=4))
    ax.text(cx, IY, lbl, ha="center", fontsize=10.5,
            fontweight="bold", color="white", zorder=5)

ax.add_patch(FancyBboxPatch((MX-1.3, IY-0.38), 2.6, 0.76,
    boxstyle="round,pad=0.06", facecolor=C_EDGE, edgecolor="white",
    alpha=0.9, lw=1.8, zorder=4))
ax.text(MX, IY, "h_edge  (128D)", ha="center", fontsize=10.5,
        fontweight="bold", color="white", zorder=5)

# ── (B) n2n NodeEdgeConv ─────────────────────────────────────────────────────
N2NY = 11.5
ax.add_patch(FancyBboxPatch((NX-1.6, N2NY-0.45), 3.2, 0.9,
    boxstyle="round,pad=0.07", facecolor="#F3E6FF", edgecolor=C_N2N, lw=2.0, zorder=4))
ax.text(NX, N2NY+0.12, "NodeEdgeConv", ha="center",
        fontsize=10, fontweight="bold", color=C_N2N, zorder=5)
ax.text(NX, N2NY-0.18, "n2n message passing", ha="center",
        fontsize=8.5, color=C_N2N, zorder=5)

# arrows into n2n
arr(NX, IY-0.38, NX, N2NY+0.45, color=C_N2N, label="msg_nn", lside="left")
arr(MX, IY-0.38, NX+0.9, N2NY+0.45, color=C_EDGE, rad=0.15)
ax.text(MX-0.3, (IY-0.38+N2NY+0.45)/2+0.15, "edge attr", ha="center",
        fontsize=8, color=C_EDGE)

# ── (C) n2e + e2n SAGEConv ───────────────────────────────────────────────────
SAGEY = 9.8
# n2e
ax.add_patch(FancyBboxPatch((NX-1.6, SAGEY-0.45), 3.2, 0.9,
    boxstyle="round,pad=0.07", facecolor="#E8E8E8", edgecolor=C_N2E, lw=2.0, zorder=4))
ax.text(NX, SAGEY+0.12, "SAGEConv", ha="center",
        fontsize=10, fontweight="bold", color=C_N2E, zorder=5)
ax.text(NX, SAGEY-0.18, "node → elem", ha="center",
        fontsize=8.5, color=C_N2E, zorder=5)

# e2n
ax.add_patch(FancyBboxPatch((ELX-1.6, SAGEY-0.45), 3.2, 0.9,
    boxstyle="round,pad=0.07", facecolor="#FFF0E0", edgecolor=C_E2N, lw=2.0, zorder=4))
ax.text(ELX, SAGEY+0.12, "SAGEConv", ha="center",
        fontsize=10, fontweight="bold", color=C_E2N, zorder=5)
ax.text(ELX, SAGEY-0.18, "elem → node", ha="center",
        fontsize=8.5, color=C_E2N, zorder=5)

# h_node → n2e
arr(NX, N2NY-0.45, NX, SAGEY+0.45, color=C_N2E, ls="dashed",
    label="msg_n2e", lside="left")
# h_elem → e2n
arr(ELX, IY-0.38, ELX, SAGEY+0.45, color=C_E2N, ls=(0,(3,2)),
    label="msg_e2n", lside="right")
# cross: h_node feeds e2n, h_elem feeds n2e
arr(NX+1.6, SAGEY, ELX-1.6, SAGEY, color="#888", lw=1.2, rad=-0.25)
arr(ELX-1.6, SAGEY, NX+1.6, SAGEY, color="#888", lw=1.2, rad=-0.25)
ax.text(MX, SAGEY+0.38, "(cross-type\nmessages)", ha="center",
        fontsize=8, color="#888", style="italic")

# ── (D) Residual MLP ─────────────────────────────────────────────────────────
MLPY = 8.1
# Node MLP
ax.add_patch(FancyBboxPatch((NX-1.8, MLPY-0.52), 3.6, 1.0,
    boxstyle="round,pad=0.07", facecolor="#FFE8D6", edgecolor=C_NODE, lw=2.0, zorder=4))
ax.text(NX, MLPY+0.23, "Node MLP  (+ residual)", ha="center",
        fontsize=9.5, fontweight="bold", color="#8B3A00", zorder=5)
ax.text(NX, MLPY-0.16, "msg_nn + msg_e2n + h_node", ha="center",
        fontsize=8.5, color="#8B3A00", style="italic", zorder=5)

# Elem MLP
ax.add_patch(FancyBboxPatch((ELX-1.8, MLPY-0.52), 3.6, 1.0,
    boxstyle="round,pad=0.07", facecolor="#D6E8FF", edgecolor=C_ELEM, lw=2.0, zorder=4))
ax.text(ELX, MLPY+0.23, "Elem MLP  (+ residual)", ha="center",
        fontsize=9.5, fontweight="bold", color="#1a4a8a", zorder=5)
ax.text(ELX, MLPY-0.16, "msg_n2e + h_elem", ha="center",
        fontsize=8.5, color="#1a4a8a", style="italic", zorder=5)

# arrows
arr(NX, SAGEY-0.45, NX, MLPY+0.52, color=C_NODE)
arr(ELX, SAGEY-0.45, ELX, MLPY+0.52, color=C_ELEM)
# e2n feeds node mlp
arr(ELX-1.6, SAGEY-0.45, NX+1.1, MLPY+0.52, color=C_E2N, lw=1.2, rad=0.2)
# n2e feeds elem mlp
arr(NX+1.6, SAGEY-0.45, ELX-1.1, MLPY+0.52, color=C_N2E, lw=1.2, rad=-0.2)

# ── (E) Outputs ──────────────────────────────────────────────────────────────
OY = 6.4
for cx, lbl, c in [(NX,  "h_node'  (updated 128D)", C_NODE),
                   (ELX, "h_elem'  (updated 128D)", C_ELEM)]:
    ax.add_patch(FancyBboxPatch((cx-1.7, OY-0.40), 3.4, 0.80,
        boxstyle="round,pad=0.07", facecolor=c, edgecolor="white",
        alpha=0.9, lw=1.8, zorder=4))
    ax.text(cx, OY, lbl, ha="center", fontsize=10,
            fontweight="bold", color="white", zorder=5)
    arr(cx, MLPY-0.52, cx, OY+0.40, color=c, lw=1.8)

# Repeat label
ax.text(MX, 5.5, "↑  Repeat × K iterations  ↑",
        ha="center", fontsize=11, fontweight="bold", color=C_PROC_BD,
        bbox=dict(boxstyle="round,pad=0.35", fc=C_PROC, ec=C_PROC_BD, lw=1.5))

# Per-material norm note
ax.text(MX, 4.5,
        "Per-material normalization: separate μ, σ for Si / Solder / UF\n"
        "→ prevents gradient interference between materials during training",
        ha="center", fontsize=9, color="#2E7D32", style="italic",
        bbox=dict(boxstyle="round,pad=0.35", fc="#E8F5E9", ec="#4CAF50", lw=1.5))

# ── Legend ────────────────────────────────────────────────────────────────────
LL = 9.8; LLY = 3.25; LLS = 1.85
for i, (c, txt) in enumerate([
    (C_NODE, "Corner Node"),
    (C_ELEM, "Element Node"),
    (C_N2N,  "n2n conv"),
    (C_N2E,  "n2e (dashed)"),
    (C_E2N,  "e2n (dotted)"),
]):
    lx = LL + i * LLS
    ax.add_patch(plt.Circle((lx, LLY+0.22), 0.09, color=c, zorder=5))
    ax.text(lx, LLY-0.08, txt, ha="center", fontsize=8, color=c)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig("mgn_flowchart.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved → mgn_flowchart.png")
