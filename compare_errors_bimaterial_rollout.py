"""
compare_errors_bimaterial_rollout.py
=====================================
Compare vel-only vs stress-finetuned model on autoregressive rollout error.

Output: compare_errors_bimaterial_rollout.png

Metrics (averaged over samples):
  Displacement Error (%) = mean_node(|pred_pos - gt_pos|) / mean_node(|gt_disp|) * 100
  Stress Error (%)       = mean_node(|GFDM(pred) - GFDM(gt)|) / mean_node(GFDM(gt)) * 100
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

def build_multigraph(elem_conn, elem_mat, mat_E):
    seen = set()
    rows, cols, E_vals = [], [], []
    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E[mat_idx])
        for (a, b) in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst, int(mat_idx))
                if key not in seen:
                    seen.add(key)
                    rows.append(src); cols.append(dst); E_vals.append(E_elem)
    return np.array([rows, cols], dtype=np.int64), np.array(E_vals, dtype=np.float32)

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR = "./04_preprocessed_pt_bimaterial_old"
NPZ_DIR  = "./02_abaqus_npz_bimaterial_old"

MODELS = [
    {
        "label":       "vel-only (w_stress=0)",
        "ckpt":        "./checkpoints_bimaterial_h128_multigraph",
        "color":       "green",
        "marker":      "s",
        "num_outputs": 3,
        "direct_stress": False,
    },
    {
        "label":       "stress-finetuned (w_stress=0.5)",
        "ckpt":        "./checkpoints_bimaterial_stress_finetune",
        "color":       "red",
        "marker":      "D",
        "num_outputs": 3,
        "direct_stress": False,
    },
    {
        "label":       "direct-stress (4-output)",
        "ckpt":        "./checkpoints_bimaterial_direct_stress",
        "color":       "blue",
        "marker":      "o",
        "num_outputs": 4,
        "direct_stress": True,
    },
]

TRAIN_INDICES = list(range(10))
TEST_INDICES  = list(range(1000, 1010))
OUT_PNG       = "./compare_errors_bimaterial_rollout.png"

NUM_INPUT  = 9
NUM_EDGE   = 9
NUM_OUTPUT = 3
PROC_SIZE  = 4
HIDDEN     = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

# ── Load stats ──────────────────────────────────────────────────────────────────

node_stats = load_json(os.path.join(DATA_DIR, "node_stats_bimaterial.json"))
v_mean = node_stats["velocity_mean"].to(device)
v_std  = node_stats["velocity_std"].to(device)
d_mean = node_stats["disp_mean"].to(device)
d_std  = node_stats["disp_std"].to(device)
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats_bimaterial.json"))
e_mean = edge_stats["edge_mean"].to(device)
e_std  = edge_stats["edge_std"].to(device)
empty_wef = torch.zeros(0, NUM_EDGE, device=device)

# ── Helpers ─────────────────────────────────────────────────────────────────────

def build_ef(mesh_pos, world_pos, edge_index, edge_E, E_ref):
    src, dst = edge_index
    d_mesh  = mesh_pos[dst]  - mesh_pos[src]
    d_world = world_pos[dst] - world_pos[src]
    ef = torch.cat([
        d_mesh,  d_mesh.norm(dim=1, keepdim=True),
        d_world, d_world.norm(dim=1, keepdim=True),
        (edge_E / E_ref).unsqueeze(1),
    ], dim=1)
    return (ef - e_mean) / e_std


def node_E_from_npz(npz, N):
    E1, E2 = float(npz["mat_E"][0]), float(npz["mat_E"][1])
    mat_count = np.zeros((N, 2), dtype=np.int32)
    for c0, m in zip(npz["elem_conn"], npz["elem_mat"]):
        for ni in c0:
            mat_count[ni, int(m)] += 1
    mat_id = np.argmax(mat_count, axis=1)
    return torch.from_numpy(
        np.where(mat_id == 0, E1, E2).astype(np.float32)
    ).to(device)


def gfdm_vm(wp, ref, mesh_ei, node_E, nu=0.33):
    row, col = mesh_ei
    r_ij  = ref[col].float() - ref[row].float()
    du_ij = (wp.float() - ref.float())[col] - (wp.float() - ref.float())[row]
    w     = 1.0 / r_ij.pow(2).sum(-1, keepdim=True).clamp(min=1e-12)
    r_r   = w.unsqueeze(-1) * r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du  = w.unsqueeze(-1) * r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
    re    = row.view(-1, 1, 1).expand_as(r_r)
    N     = wp.shape[0]
    XtX   = torch.zeros(N, 3, 3, device=wp.device)
    XtY   = torch.zeros(N, 3, 3, device=wp.device)
    XtX.scatter_add_(0, re, r_r)
    XtY.scatter_add_(0, re, r_du)
    XtX  += 1e-6 * torch.eye(3, device=wp.device).unsqueeze(0)
    dU    = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    E     = node_E.float()
    lam   = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu    = E / (2 * (1 + nu))
    tr    = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx    = lam * tr + 2 * mu * dU[:, 0, 0]
    sy    = lam * tr + 2 * mu * dU[:, 1, 1]
    sz    = lam * tr + 2 * mu * dU[:, 2, 2]
    txy   = mu * (dU[:, 0, 1] + dU[:, 1, 0])
    txz   = mu * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz   = mu * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                              + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)


s_mean_val = float(node_stats["stress_mean"])
s_std_val  = float(node_stats["stress_std"])


def load_model(ckpt, num_outputs=3):
    m = HybridMeshGraphNet(
        NUM_INPUT, NUM_EDGE, num_outputs,
        processor_size=PROC_SIZE,
        hidden_dim_processor=HIDDEN, hidden_dim_node_encoder=HIDDEN,
        hidden_dim_edge_encoder=HIDDEN, hidden_dim_node_decoder=HIDDEN,
        mlp_activation_fn="relu", do_concat_trick=False,
        num_processor_checkpoint_segments=0, recompute_activation=False,
    ).to(device)
    m.eval()
    load_checkpoint(ckpt, models=m, device=device)
    return m


def eval_samples(model, indices, all_files, direct_stress=False):
    disp_acc   = defaultdict(list)
    stress_acc = defaultdict(list)

    for idx in indices:
        if idx >= len(all_files):
            continue
        fname = all_files[idx]
        sid   = fname.replace(".pt", "")
        data  = torch.load(os.path.join(DATA_DIR, fname),
                           map_location="cpu", weights_only=False)
        npz   = np.load(os.path.join(NPZ_DIR, sid + ".npz"))
        T     = len(data)
        N     = data[0]["graph"].num_nodes
        E1    = float(npz["mat_E"][0])

        ei_np, eE_np = build_multigraph(npz["elem_conn"], npz["elem_mat"], npz["mat_E"])
        edge_index = torch.from_numpy(ei_np).to(device)
        edge_E     = torch.from_numpy(eE_np).to(device)
        mesh_ei    = torch.from_numpy(npz["mesh_edge_index"].astype(np.int64)).to(device)
        ref_pos    = data[0]["graph"].mesh_pos.to(device)
        node_E     = node_E_from_npz(npz, N)
        node_type  = data[0]["graph"].x[:, 6:9].to(device)

        # GT positions
        gt_wp = [data[0]["graph"].world_pos.clone()]
        for step in data:
            gt_wp.append(gt_wp[-1] + step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu)

        # Autoregressive rollout (seed at t=1 from GT)
        cur      = gt_wp[1].to(device)
        prev_vel = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
        pred_wp  = [gt_wp[0], gt_wp[1]]  # t=0 and t=1 are exact

        pred_stress_direct = [None, None]  # only used for direct_stress models

        with torch.inference_mode():
            for _ in range(T - 1):
                ef = build_ef(ref_pos, cur, edge_index, edge_E, E1)
                nx = torch.cat([
                    (cur - ref_pos - d_mean) / d_std,
                    (prev_vel - v_mean) / v_std,
                    node_type,
                ], dim=1)
                out = model(nx, ef, empty_wef,
                            Data(x=nx, edge_index=edge_index, num_nodes=N))
                vel      = out[:, :3] * v_std + v_mean
                prev_vel = vel
                cur      = cur + vel
                pred_wp.append(cur.clone())
                if direct_stress:
                    # denormalize stress from 4th output
                    s_phys = (out[:, 3] * s_std_val + s_mean_val).cpu().numpy()
                    pred_stress_direct.append(s_phys)

        ref_np = ref_pos.cpu().numpy()

        for t in range(T):
            gt_np   = gt_wp[t + 1].numpy()
            pred_np = pred_wp[t + 1].cpu().numpy()

            # Displacement error (%)
            gt_disp_mag = np.linalg.norm(gt_np - ref_np, axis=1).mean()
            disp_err    = np.linalg.norm(pred_np - gt_np, axis=1).mean()
            disp_acc[t + 1].append(disp_err / max(gt_disp_mag, 1e-10) * 100)

            # Stress error
            with torch.inference_mode():
                gt_vm = gfdm_vm(torch.from_numpy(gt_np).to(device),
                                ref_pos, mesh_ei, node_E).cpu().numpy()

            if direct_stress and pred_stress_direct[t + 1] is not None:
                pred_vm = pred_stress_direct[t + 1]
            else:
                with torch.inference_mode():
                    pred_vm = gfdm_vm(torch.from_numpy(pred_np).to(device),
                                      ref_pos, mesh_ei, node_E).cpu().numpy()

            stress_err = np.abs(pred_vm - gt_vm).mean()
            stress_acc[t + 1].append(stress_err / max(gt_vm.mean(), 1e-10) * 100)

        print(f"    [{idx:4d}] {sid}  "
              f"final_disp={disp_acc[T][-1]:.2f}%  "
              f"final_stress={stress_acc[T][-1]:.2f}%")

    T_vals = sorted(disp_acc.keys())
    return (
        T_vals,
        [np.mean(disp_acc[t])   for t in T_vals],
        [np.mean(stress_acc[t]) for t in T_vals],
    )


# ── Main ────────────────────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
print(f"Found {len(all_files)} .pt files")

results = {}
for m_cfg in MODELS:
    print(f"\n=== {m_cfg['label']} ===")
    model = load_model(m_cfg["ckpt"], m_cfg.get("num_outputs", 3))
    ds = m_cfg.get("direct_stress", False)

    print("  Training samples ...")
    t_vals, d_train, s_train = eval_samples(model, TRAIN_INDICES, all_files, ds)
    print("  Test samples ...")
    _,      d_test,  s_test  = eval_samples(model, TEST_INDICES,  all_files, ds)

    results[m_cfg["label"]] = {
        "color": m_cfg["color"], "marker": m_cfg["marker"],
        "t": t_vals,
        "disp_train": d_train, "disp_test": d_test,
        "stress_train": s_train, "stress_test": s_test,
    }
    del model
    torch.cuda.empty_cache()

# ── Plot ────────────────────────────────────────────────────────────────────────

n_train = len(TRAIN_INDICES)
n_test  = len(TEST_INDICES)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Two-way comparison  "
    f"(train n={n_train}, test n={n_test}, {len(MODELS)} models)",
    fontsize=13,
)

for label, res in results.items():
    c, mk, t = res["color"], res["marker"], res["t"]
    axes[0].plot(t, res["disp_train"],  color=c, marker=mk, linestyle="-",
                 label=f"{label} - Training")
    axes[0].plot(t, res["disp_test"],   color=c, marker=mk, linestyle="--",
                 markerfacecolor="none", label=f"{label} - Test")
    axes[1].plot(t, res["stress_train"], color=c, marker=mk, linestyle="-",
                 label=f"{label} - Training")
    axes[1].plot(t, res["stress_test"],  color=c, marker=mk, linestyle="--",
                 markerfacecolor="none", label=f"{label} - Test")

for ax, ylabel in zip(axes, ["Displacement Error (%)", "Stress Error (%)"]):
    ax.set_title(ylabel)
    ax.set_xlabel("Time step t")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")