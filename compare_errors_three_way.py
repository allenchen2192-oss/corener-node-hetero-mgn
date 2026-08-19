"""
compare_errors_three_way.py
===========================
Three-way rollout error comparison (all: no node_type, normalized):

  1. w_stress=0.5  : with GFDM stress supervision
  2. w_stress=0    : velocity loss only (skip t=0 in training)
  3. w_stress=0 + t=0 : velocity loss only, training includes t=0

Color scheme : one color per model (blue / green / red)
Line style   : solid = Train, dashed = Test
Marker       : circle / square / triangle per model

Usage:
  python compare_errors_three_way.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128

DATA_DIR = "./04_preprocessed_pt_abaqus_prevvel"

CONFIGS = {
    "w_stress=0.5": {
        "ckpt": "./checkpoints_abaqus_prevvel_1000data_1000nodes_h128_gfdm_stress_dispnorm_no_nodetype",
        "normalized": True,
        "use_node_type": False,
        "num_input_features": 6,
    },
    "w_stress=0": {
        "ckpt": "./checkpoints_abaqus_prevvel_1000data_1000nodes_h128_dispnorm_no_nodetype_no_stress",
        "normalized": True,
        "use_node_type": False,
        "num_input_features": 6,
    },
    "w_stress=0 (with t=0)": {
        "ckpt": "./checkpoints_abaqus_prevvel_1000data_1000nodes_h128_dispnorm_no_nodetype_witht0",
        "normalized": True,
        "use_node_type": False,
        "num_input_features": 6,
    },
}

NUM_TRAIN_SAMPLES = 100
NUM_TEST_SAMPLES  = 100
MAX_NODES_TRAIN   = 1000

OUTPUT_PLOT = "./compare_errors_three_way.png"


# ── Material constants ────────────────────────────────────────────────────────

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_edge_features(pos, edge_index):
    row, col = edge_index
    d = pos[col] - pos[row]
    n = torch.norm(d, dim=1, keepdim=True)
    return torch.cat([d, n], dim=1)


def compute_vm_stress(wp, ref, mesh_ei):
    row, col = mesh_ei
    r_ij  = ref[col].float() - ref[row].float()
    u     = wp.float() - ref.float()
    du_ij = u[col] - u[row]
    r_r   = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    N = wp.shape[0]
    XtX = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtY = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtX.scatter_add_(0, row_exp, r_r)
    XtY.scatter_add_(0, row_exp, r_du)
    XtX += 1e-6 * torch.eye(3, device=wp.device).unsqueeze(0)
    dU  = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy  = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz  = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                              + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)


# ── Setup (shared stats) ──────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean = torch.tensor(node_stats["velocity_mean"], device=device)
v_std  = torch.tensor(node_stats["velocity_std"],  device=device)
d_mean = torch.tensor(node_stats["disp_mean"],     device=device)
d_std  = torch.tensor(node_stats["disp_std"],      device=device)
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean_ref = torch.tensor(edge_stats["edge_mean"], device=device)[:4]
e_std_ref  = torch.tensor(edge_stats["edge_std"],  device=device)[:4]
e_mean_wld = torch.tensor(edge_stats["edge_mean"], device=device)[4:]
e_std_wld  = torch.tensor(edge_stats["edge_std"],  device=device)[4:]

empty_wef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)

all_files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.startswith("sample_") and f.endswith(".pt"))

train_indices = []
for i in range(2000):
    if len(train_indices) >= NUM_TRAIN_SAMPLES:
        break
    if i >= len(all_files):
        break
    d0 = torch.load(os.path.join(DATA_DIR, all_files[i]),
                    map_location="cpu", weights_only=False)
    if d0[0]["graph"].num_nodes <= MAX_NODES_TRAIN:
        train_indices.append(i)

test_indices = list(range(2000, min(2000 + NUM_TEST_SAMPLES, len(all_files))))
print(f"Train: {len(train_indices)} | Test: {len(test_indices)}\n")


# ── Rollout ───────────────────────────────────────────────────────────────────

def run_sample(fname, model, normalized, use_node_type=False):
    data = torch.load(os.path.join(DATA_DIR, fname),
                      map_location="cpu", weights_only=False)
    T = len(data)
    N = data[0]["graph"].num_nodes

    ref_pos      = data[0]["graph"].mesh_pos.to(device)
    mesh_ei      = data[0]["graph"].edge_index.to(device)
    node_type_oh = data[0]["graph"].x[:, 6:8].to(device)
    ref_feat_n   = (compute_edge_features(ref_pos, mesh_ei) - e_mean_ref) / e_std_ref

    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    gt_disp_mag_final = torch.norm(gt_wp_list[-1] - data[0]["graph"].mesh_pos, dim=1).max().item()
    if gt_disp_mag_final < 1e-10:
        return None

    gt_vel_0  = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    cur       = gt_wp_list[1].to(device)
    prev_vel  = gt_vel_0
    pred_wp_list = [gt_wp_list[0], gt_wp_list[1]]

    with torch.inference_mode():
        for _ in range(T - 1):
            world_feat = (compute_edge_features(cur, mesh_ei) - e_mean_wld) / e_std_wld
            mesh_ef    = torch.cat([ref_feat_n, world_feat], dim=1)

            if normalized:
                disp_part = (cur - ref_pos - d_mean) / d_std
                vel_part  = (prev_vel - v_mean) / v_std
            else:
                disp_part = cur - ref_pos
                vel_part  = prev_vel

            if use_node_type:
                node_x = torch.cat([disp_part, vel_part, node_type_oh], dim=1)
            else:
                node_x = torch.cat([disp_part, vel_part], dim=1)

            g   = Data(x=node_x, edge_index=mesh_ei, num_nodes=N)
            out = model(node_x, mesh_ef, empty_wef, g)
            vel = out[:, :3] * v_std + v_mean
            prev_vel = vel
            cur = cur + vel
            pred_wp_list.append(cur.cpu())

    disp_rel_per_t   = [0.0]
    stress_rel_per_t = [0.0]

    with torch.inference_mode():
        for t in range(2, T + 1):
            err = torch.norm(pred_wp_list[t] - gt_wp_list[t], dim=1).mean().item()
            disp_rel_per_t.append(err / gt_disp_mag_final * 100.0)

            vm_pred = compute_vm_stress(pred_wp_list[t].to(device), ref_pos, mesh_ei).cpu()
            vm_gt   = compute_vm_stress(gt_wp_list[t].to(device),   ref_pos, mesh_ei).cpu()
            gt_mean = vm_gt.mean().item()
            stress_rel_per_t.append(
                (vm_pred - vm_gt).abs().mean().item() / gt_mean * 100.0
                if gt_mean > 1e-10 else float("nan")
            )

    return np.array(disp_rel_per_t), np.array(stress_rel_per_t)


def collect(fnames, model, normalized, label, use_node_type=False):
    disp_list, stress_list = [], []
    for i, fname in enumerate(fnames):
        result = run_sample(fname, model, normalized, use_node_type)
        if result is None:
            continue
        disp_list.append(result[0])
        stress_list.append(result[1])
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(fnames)} done")
    return np.array(disp_list), np.array(stress_list)


# ── Run all three configs ─────────────────────────────────────────────────────

results = {}

for cfg_name, cfg in CONFIGS.items():
    print(f"\n=== {cfg_name} ===")
    n_in = cfg["num_input_features"]
    model = HybridMeshGraphNet(
        n_in, NUM_EDGE_FEATURES, NUM_OUTPUT_FEATURES,
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
    load_checkpoint(cfg["ckpt"], models=model, device=device)

    norm          = cfg["normalized"]
    use_node_type = cfg.get("use_node_type", False)
    train_d, train_s = collect([all_files[i] for i in train_indices], model, norm,
                                f"{cfg_name}/train", use_node_type)
    test_d,  test_s  = collect([all_files[i] for i in test_indices],  model, norm,
                                f"{cfg_name}/test",  use_node_type)
    results[cfg_name] = {
        "train_d": train_d, "train_s": train_s,
        "test_d":  test_d,  "test_s":  test_s,
    }
    del model


# ── Plot ──────────────────────────────────────────────────────────────────────

cfg_keys = list(CONFIGS.keys())
T_steps  = results[cfg_keys[0]]["train_d"].shape[1]
ts       = np.arange(1, 1 + T_steps)

# One color per model, solid=train / dashed=test, unique marker per model
model_style = {
    cfg_keys[0]: {"color": "tab:blue",  "marker": "o"},   # w_stress=0.5
    cfg_keys[1]: {"color": "tab:green", "marker": "s"},   # w_stress=0
    cfg_keys[2]: {"color": "tab:red",   "marker": "^"},   # w_stress=0 + t=0
}
split_ls = {"train": "-", "test": "--"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"No Node Type, Normalized — three-way comparison  (mean over {NUM_TRAIN_SAMPLES} samples)",
    fontsize=16,
)

for ax, (ylabel, key) in zip(axes, [("Displacement Error (%)", "d"), ("Stress Error (%)", "s")]):
    for cfg_name in cfg_keys:
        ms = model_style[cfg_name]
        for split in ["train", "test"]:
            arr = results[cfg_name][f"{split}_{key}"]
            mu  = np.nanmean(arr, axis=0)
            ls  = split_ls[split]
            label = f"{cfg_name} – {'Training' if split == 'train' else 'Test'}"
            ax.plot(ts, mu,
                    color=ms["color"], linestyle=ls, linewidth=2,
                    marker=ms["marker"], markersize=4, markevery=2,
                    label=label)

    ax.set_xlabel("Time step t", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(ylabel, fontsize=15)
    ax.set_xlim(2, 19)
    ax.set_xticks(range(2, 20))
    ax.tick_params(labelsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_PLOT}")
