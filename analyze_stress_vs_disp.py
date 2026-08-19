"""
analyze_stress_vs_disp.py
==========================
驗證假說：GFDM stress 誤差是否在小位移時更大？

分兩層分析：
  (A) GFDM error：從 GT 位移算 GFDM stress vs Abaqus stress
      → 隔離 GFDM 本身的誤差（與 model 無關）
  (B) Model+GFDM error：從 pred 位移算 GFDM stress vs Abaqus stress
      → 完整的 pred stress 誤差

X 軸：每個時間步的平均位移大小 (m)
Y 軸：stress 相對誤差 = mean(|pred_vm - gt_vm|) / mean(gt_vm)

Usage:
  python analyze_stress_vs_disp.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from export_bimaterial_paraview import build_rollout_ef, node_material_id
from preprocess_bimaterial import build_multigraph
from train_bimaterial import BimaterialTrainer   # reuse _vm_stress_bimaterial

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR  = "./04_preprocessed_pt_bimaterial"
NPZ_DIR   = "./02_abaqus_npz_bimaterial"
CKPT_PATH = "./checkpoints_bimaterial_h128_multigraph"
OUT_DIR   = "./analyze_stress_vs_disp"

NUM_INPUT_FEATURES  = 9
NUM_EDGE_FEATURES   = 9
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128
NU                  = 0.33

TRAIN_INDICES = list(range(10))
TEST_INDICES  = list(range(1000, 1010))
SAMPLE_INDICES = TRAIN_INDICES + TEST_INDICES

os.makedirs(OUT_DIR, exist_ok=True)

# ── GFDM stress with per-node E ────────────────────────────────────────────────

def vm_stress_gfdm(wp, ref, mesh_ei, node_E, nu=NU):
    N = wp.shape[0]
    row, col = mesh_ei
    r_ij  = ref[col].float() - ref[row].float()
    u     = wp.float() - ref.float()
    du_ij = u[col] - u[row]
    w_ij  = 1.0 / (r_ij.pow(2).sum(-1, keepdim=True).clamp(min=1e-12))
    r_r   = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du  = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
    row_e = row.view(-1, 1, 1).expand_as(r_r)
    XtX   = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtY   = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtX.scatter_add_(0, row_e, r_r)
    XtY.scatter_add_(0, row_e, r_du)
    XtX  = XtX + 1e-6 * torch.eye(3, device=wp.device).unsqueeze(0)
    dU   = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    E    = node_E.float()
    lam  = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu   = E / (2 * (1 + nu))
    tr   = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx   = lam * tr + 2 * mu * dU[:, 0, 0]
    sy   = lam * tr + 2 * mu * dU[:, 1, 1]
    sz   = lam * tr + 2 * mu * dU[:, 2, 2]
    txy  = mu * (dU[:, 0, 1] + dU[:, 1, 0])
    txz  = mu * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz  = mu * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                              + 6*(txy**2+txz**2+tyz**2)) + 1e-30)

# ── Load stats & model ─────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

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

empty_wef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)

model = HybridMeshGraphNet(
    NUM_INPUT_FEATURES, NUM_EDGE_FEATURES, NUM_OUTPUT_FEATURES,
    processor_size=PROCESSOR_SIZE,
    hidden_dim_processor=HIDDEN_DIM,
    hidden_dim_node_encoder=HIDDEN_DIM,
    hidden_dim_edge_encoder=HIDDEN_DIM,
    hidden_dim_node_decoder=HIDDEN_DIM,
    mlp_activation_fn="relu",
    do_concat_trick=False,
    num_processor_checkpoint_segments=0,
    recompute_activation=False,
).to(device)
model.eval()
load_checkpoint(CKPT_PATH, models=model, device=device)

# ── File list ──────────────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)

# ── Collect per-timestep stats ─────────────────────────────────────────────────

results = {"train": [], "test": []}  # each entry: (disp_mag, err_gfdm, err_model)

for idx in SAMPLE_INDICES:
    fname     = all_files[idx]
    sample_id = fname.replace(".pt", "")
    split     = "train" if idx < 1000 else "test"
    print(f"\n[{split}] {sample_id}")

    data = torch.load(os.path.join(DATA_DIR, fname),
                      map_location="cpu", weights_only=False)
    T = len(data)
    N = data[0]["graph"].num_nodes

    npz       = np.load(os.path.join(NPZ_DIR, sample_id + ".npz"))
    elem_conn = npz["elem_conn"]
    elem_mat  = npz["elem_mat"]
    mat_E     = npz["mat_E"]
    E1        = float(mat_E[0])
    stress_vm_abaqus = npz["stress_vm"]   # (F, N) Pa

    edge_index_np, edge_E_np = build_multigraph(elem_conn, elem_mat, mat_E)
    edge_index = torch.from_numpy(edge_index_np).to(device)
    edge_E     = torch.from_numpy(edge_E_np).to(device)

    mesh_ei = torch.from_numpy(npz["mesh_edge_index"].astype(np.int64)).to(device)
    mat_id  = node_material_id(elem_conn, elem_mat, N)
    node_E  = torch.where(
        torch.from_numpy(mat_id).to(device) == 0,
        torch.full((N,), E1, device=device),
        torch.full((N,), float(mat_E[1]), device=device),
    )

    ref_pos      = data[0]["graph"].mesh_pos.to(device)
    node_type_oh = data[0]["graph"].x[:, 6:9].to(device)

    # GT world-pos from labels
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    # Model rollout
    gt_vel_0 = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    cur      = gt_wp_list[1].to(device)
    prev_vel = gt_vel_0
    pred_wp_list = [gt_wp_list[0], gt_wp_list[1].to(device)]

    with torch.inference_mode():
        for _ in range(T - 1):
            ef = build_rollout_ef(ref_pos, cur, edge_index, edge_E, E1,
                                  e_mean, e_std)
            disp_norm     = (cur - ref_pos - d_mean) / d_std
            prev_vel_norm = (prev_vel - v_mean) / v_std
            node_x        = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)
            g   = Data(x=node_x, edge_index=edge_index, num_nodes=N)
            out = model(node_x, ef, empty_wef, g)
            vel      = out[:, :3] * v_std + v_mean
            prev_vel = vel
            cur      = cur + vel
            pred_wp_list.append(cur)

    # Per-timestep analysis
    with torch.inference_mode():
        for t_idx in range(T):
            gt_wp  = gt_wp_list[t_idx + 1]
            if not isinstance(gt_wp, torch.Tensor):
                gt_wp = torch.from_numpy(gt_wp).to(device)
            gt_wp  = gt_wp.to(device)
            pred_wp = pred_wp_list[t_idx + 1]
            if not isinstance(pred_wp, torch.Tensor):
                pred_wp = torch.from_numpy(pred_wp).to(device)
            pred_wp = pred_wp.to(device)

            gt_abaqus = torch.from_numpy(
                stress_vm_abaqus[t_idx + 1].astype(np.float32)
            ).to(device)

            disp_mag = (gt_wp - ref_pos).norm(dim=1).mean().item()

            # (A) GFDM from GT displacement
            vm_gfdm_gt = vm_stress_gfdm(gt_wp, ref_pos, mesh_ei, node_E)
            err_gfdm   = ((vm_gfdm_gt - gt_abaqus).abs() /
                          gt_abaqus.clamp(min=1e3)).mean().item()

            # (B) GFDM from pred displacement
            vm_gfdm_pred = vm_stress_gfdm(pred_wp, ref_pos, mesh_ei, node_E)
            err_model    = ((vm_gfdm_pred - gt_abaqus).abs() /
                            gt_abaqus.clamp(min=1e3)).mean().item()

            results[split].append((disp_mag, err_gfdm, err_model))

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Stress error vs Displacement magnitude", fontsize=13)

colors = {"train": "#2196F3", "test": "#F44336"}

for ax, (label, ylabel) in zip(axes, [
    ("(A) GFDM from GT displacement\n(GFDM approximation error only)",
     "Relative stress error"),
    ("(B) GFDM from Model displacement\n(GFDM + model prediction error)",
     "Relative stress error"),
]):
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Mean displacement magnitude (m)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    for split in ["train", "test"]:
        data_pts = results[split]
        if not data_pts:
            continue
        x = np.array([d[0] for d in data_pts])
        y_gfdm  = np.array([d[1] for d in data_pts])
        y_model = np.array([d[2] for d in data_pts])
        y = y_gfdm if ax is axes[0] else y_model
        ax.scatter(x, y, s=12, alpha=0.6, color=colors[split], label=split)

        # trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, p(xs), color=colors[split], linewidth=1.5, linestyle="--")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "stress_error_vs_disp.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n圖片儲存至 {out_path}")
