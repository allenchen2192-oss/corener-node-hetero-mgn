"""
mgn_comparison.py
Side-by-side diagram: Corner-Node MGN vs Heterogeneous MGN.
Output: mgn_comparison.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_NODE  = "#E8824A"   # orange  — corner node
C_ELEM  = "#3A82C4"   # blue    — element node
C_N2N   = "#A020A0"   # magenta — node↔node edge
C_N2E   = "#3A3A3A"   # dark gray  — node→elem arrow
C_E2N   = "#C05000"   # dark orange — elem→node arrow
C_QUAD  = "#D6E8F7"   # light blue quad fill
C_PANEL = "#F7F9FC"   # panel bg

# ── Geometry ──────────────────────────────────────────────────────────────────
CORNERS = np.array([[0.25, 0.25], [1.75, 0.25], [1.75, 1.55], [0.25, 1.55]])
ELEM_POS = np.array([1.0, 0.90])
NODE_R = 0.13
ELEM_W, ELEM_H = 0.34, 0.24
QUAD_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]

# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_ax(ax, title):
    ax.set_facecolor(C_PANEL)
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylim(-0.20, 2.00)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10,
                 color="#1a1a1a", fontfamily="DejaVu Sans")


def boundary_circle(center, r, direction):
    d = direction - center
    return center + d / np.linalg.norm(d) * r


def boundary_rect(center, w, h, direction):
    d = direction - center
    if np.linalg.norm(d) < 1e-9:
        return center.copy()
    dx, dy = d
    tx = (w / 2) / abs(dx) if abs(dx) > 1e-9 else np.inf
    ty = (h / 2) / abs(dy) if abs(dy) > 1e-9 else np.inf
    t = min(tx, ty)
    return center + np.array([dx, dy]) * t


def draw_n2n(ax, p1, p2):
    """Solid magenta line between two corner nodes (no arrowhead = undirected)."""
    a = boundary_circle(p1, NODE_R, p2)
    b = boundary_circle(p2, NODE_R, p1)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C_N2N, lw=2.2, zorder=2, solid_capstyle="round")


def curved_arrow(ax, src, dst, color, lw, ls, rad, gap_src, gap_dst, style="-|>"):
    ax.annotate(
        "", xy=dst, xytext=src,
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            linestyle=ls,
            shrinkA=gap_src * 72,   # matplotlib uses points; 72pt ≈ 1 inch
            shrinkB=gap_dst * 72,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=10,
        ),
        zorder=3,
    )


def draw_corner_node(ax, pos, label):
    ax.add_patch(Circle(pos, NODE_R, facecolor=C_NODE, edgecolor="white", lw=1.8, zorder=5))
    ax.text(pos[0], pos[1], label, ha="center", va="center",
            fontsize=6.8, color="white", fontweight="bold",
            multialignment="center", linespacing=1.35, zorder=6)


def draw_element_node(ax, pos, label):
    ax.add_patch(FancyBboxPatch(
        (pos[0] - ELEM_W/2, pos[1] - ELEM_H/2), ELEM_W, ELEM_H,
        boxstyle="round,pad=0.03",
        facecolor=C_ELEM, edgecolor="white", lw=1.8, zorder=5))
    ax.text(pos[0], pos[1], label, ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold",
            multialignment="center", linespacing=1.35, zorder=6)


def draw_quad_bg(ax):
    ax.add_patch(Polygon(CORNERS, closed=True,
                         facecolor=C_QUAD, edgecolor="none", alpha=0.55, zorder=0))


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(17, 8))
fig.patch.set_facecolor("white")
fig.suptitle("Corner-node MeshGraphNet  vs.  Heterogeneous MeshGraphNet",
             fontsize=15, fontweight="bold", y=0.98, color="#1a1a1a")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT: Corner-Node MGN
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
setup_ax(ax, "Corner-Node MeshGraphNet")
draw_quad_bg(ax)

for i, j in QUAD_EDGES:
    draw_n2n(ax, CORNERS[i], CORNERS[j])

for pos in CORNERS:
    draw_corner_node(ax, pos, "pos, vel\nσ_VM, ε_p")

# Description below nodes
ax.text(1.0, -0.12,
        "All quantities stored & predicted at corner nodes\n"
        "(elem→node averaging required for ground truth σ_VM, ε_p)",
        ha="center", va="top", fontsize=8, color="#555", style="italic",
        multialignment="center", linespacing=1.4)

# MP arrow annotation (curved loop at top)
ax.annotate("", xy=(0.25 - 0.02, 1.0), xytext=(1.75 + 0.02, 1.0),
            arrowprops=dict(arrowstyle="<->", color=C_N2N, lw=2.0,
                            connectionstyle="arc3,rad=0.55"))
ax.text(1.0, 1.86, "Node ↔ Node  message passing", ha="center", va="center",
        fontsize=8.5, color=C_N2N, style="italic")

# Legend (bottom-right of left panel)
lx, ly, ldy = 1.18, 0.68, 0.16
ax.plot([lx, lx + 0.28], [ly, ly], color=C_N2N, lw=2.2)
ax.text(lx + 0.33, ly, "Node – Node edge", fontsize=7.5, va="center", color=C_N2N)
ax.add_patch(Circle([lx + 0.14, ly - ldy], 0.05,
                    facecolor=C_NODE, edgecolor="white", lw=1, zorder=6))
ax.text(lx + 0.33, ly - ldy, "Corner Node", fontsize=7.5, va="center", color=C_NODE)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT: Heterogeneous MGN
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
setup_ax(ax, "Heterogeneous MeshGraphNet")
draw_quad_bg(ax)

for i, j in QUAD_EDGES:
    draw_n2n(ax, CORNERS[i], CORNERS[j])

# Per-corner: alternating curvature so N2E and E2N don't overlap
RADS = [0.22, -0.22, 0.22, -0.22]

# Compute shrink amounts in figure fraction (approximation)
# Node circle radius in data coords = NODE_R; elem half-extents = ELEM_W/2, ELEM_H/2
# We pass shrinkA/B in inches (using 72pt/in conversion in annotate)
SHRINK_NODE = NODE_R * 0.42   # rough inches at the figure scale
SHRINK_ELEM = 0.08

for i, pos in enumerate(CORNERS):
    rad = RADS[i]
    # Node → Elem: dashed dark gray
    src_n2e = boundary_circle(pos, NODE_R + 0.01, ELEM_POS)
    dst_n2e = boundary_rect(ELEM_POS, ELEM_W + 0.02, ELEM_H + 0.02, pos)
    curved_arrow(ax, src_n2e, dst_n2e,
                 color=C_N2E, lw=1.3, ls="dashed", rad=rad,
                 gap_src=0, gap_dst=0)
    # Elem → Node: dotted dark orange
    src_e2n = boundary_rect(ELEM_POS, ELEM_W + 0.02, ELEM_H + 0.02, pos)
    dst_e2n = boundary_circle(pos, NODE_R + 0.01, ELEM_POS)
    curved_arrow(ax, src_e2n, dst_e2n,
                 color=C_E2N, lw=1.6, ls=(0, (3, 2)), rad=-rad,
                 gap_src=0, gap_dst=0)

for pos in CORNERS:
    draw_corner_node(ax, pos, "pos, vel")

draw_element_node(ax, ELEM_POS, "σ_VM, ε_p")

# Description below
ax.text(1.0, -0.12,
        "Stress & PEEQ natively at element nodes — no interpolation needed\n"
        "Enables per-material normalization (Si / Solder / UF separately)",
        ha="center", va="top", fontsize=8, color="#555", style="italic",
        multialignment="center", linespacing=1.4)

# MP annotation at top
ax.text(1.0, 1.86,
        "Node ↔ Node  |  Node → Elem  |  Elem → Node  message passing",
        ha="center", va="center", fontsize=8, color="#333", style="italic")

# Legend (bottom-right of right panel)
lx, ly, ldy = 1.13, 0.72, 0.155
# Node–Node
ax.plot([lx, lx + 0.28], [ly, ly], color=C_N2N, lw=2.2)
ax.text(lx + 0.33, ly, "Node – Node", fontsize=7.5, va="center", color=C_N2N)
# Node → Elem
ax.plot([lx, lx + 0.28], [ly - ldy, ly - ldy], color=C_N2E, lw=1.3, ls="--")
ax.annotate("", xy=(lx + 0.28, ly - ldy), xytext=(lx + 0.22, ly - ldy),
            arrowprops=dict(arrowstyle="-|>", color=C_N2E, lw=1.3, mutation_scale=8))
ax.text(lx + 0.33, ly - ldy, "Node → Elem", fontsize=7.5, va="center", color=C_N2E)
# Elem → Node
ax.plot([lx, lx + 0.28], [ly - 2*ldy, ly - 2*ldy], color=C_E2N, lw=1.6, ls=(0, (3, 2)))
ax.annotate("", xy=(lx + 0.28, ly - 2*ldy), xytext=(lx + 0.22, ly - 2*ldy),
            arrowprops=dict(arrowstyle="-|>", color=C_E2N, lw=1.6, mutation_scale=8))
ax.text(lx + 0.33, ly - 2*ldy, "Elem → Node", fontsize=7.5, va="center", color=C_E2N)
# Node icon
ax.add_patch(Circle([lx + 0.14, ly - 3*ldy], 0.05,
                    facecolor=C_NODE, edgecolor="white", lw=1, zorder=6))
ax.text(lx + 0.33, ly - 3*ldy, "Corner Node", fontsize=7.5, va="center", color=C_NODE)
# Elem icon
ax.add_patch(FancyBboxPatch((lx + 0.07, ly - 4*ldy - 0.04), 0.14, 0.08,
                             boxstyle="round,pad=0.01",
                             facecolor=C_ELEM, edgecolor="white", lw=1, zorder=6))
ax.text(lx + 0.33, ly - 4*ldy, "Element Node", fontsize=7.5, va="center", color=C_ELEM)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.0, 1, 0.96])
plt.savefig("mgn_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved -> mgn_comparison.png")