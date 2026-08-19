"""
compare_errors_balanced.py
==========================
比�???model（w_stress=0/0.5）�???balanced model ??per-sample error??
- ??models：train=0-99（Dir0）、test=1000-1099（Dir3�?- ??balanced model：train=25/dir?�test=25/dir�? directions ??25�?"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint

DATA_DIR = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/04_preprocessed_pt_beam_prevvel"
NPZ_DIR  = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/02_beam_npz"

# old train/test (Dir0 only train, Dir3 only test)
OLD_TRAIN = list(range(0, 100))
OLD_TEST  = list(range(1000, 1100))

# balanced train/test (25 per direction)
BAL_TRAIN = list(range(0,25))   + list(range(275,300)) + list(range(550,575)) + list(range(825,850))
BAL_TEST  = list(range(250,275))+ list(range(525,550)) + list(range(800,825)) + list(range(1075,1100))

# (label, ckpt_path, color, use_per_sample_norm, train_indices, test_indices)
MODELS = [
    ("w_stress=0",
     "./checkpoints_3Dbeam_no_rod_balanced_w0",
     "tab:green",  False, BAL_TRAIN, BAL_TEST),
    ("w_stress=0.5",
     "./checkpoints_3Dbeam_no_rod_balanced_nonorm",
     "tab:blue",   False, BAL_TRAIN, BAL_TEST),
    ("w_stress=0, per-sample norm",
     "./checkpoints_3Dbeam_no_rod_balanced_w0_norm",
     "tab:orange", True,  BAL_TRAIN, BAL_TEST),
    ("w_stress=0.5, per-sample norm",
     "./checkpoints_3Dbeam_no_rod_balanced",
     "tab:red",    True,  BAL_TRAIN, BAL_TEST),
]

HIDE_COLORS = []   # set to e.g. ["tab:orange"] to skip those models
MODELS = [m for m in MODELS if m[2] not in HIDE_COLORS]

CKPT_EPOCH          = None
NUM_INPUT_FEATURES  = 8
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128
_suffix    = "_no_" + "_".join(c.replace("tab:","") for c in HIDE_COLORS) if HIDE_COLORS else ""
OUTPUT_PNG = f"compare_errors_balanced{_suffix}.png"

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


def compute_vm_stress_np(world_pos, ref_pos, edge_index):
    N        = ref_pos.shape[0]
    row, col = edge_index
    r_ij     = ref_pos[col] - ref_pos[row]
    du_ij    = (world_pos - ref_pos)[col] - (world_pos - ref_pos)[row]
    r_r      = r_ij[:, :, None] * r_ij[:, None, :]
    r_du     = r_ij[:, :, None] * du_ij[:, None, :]
    XtX = np.zeros((N, 3, 3), dtype=np.float64)
    XtY = np.zeros((N, 3, 3), dtype=np.float64)
    np.add.at(XtX, row, r_r)
    np.add.at(XtY, row, r_du)
    XtX += 1e-6 * np.eye(3)[None]
    dU   = np.linalg.solve(XtX, XtY).transpose(0, 2, 1)
    tr   = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx   = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy   = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz   = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy  = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz  = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz  = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return np.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                           + 6*(txy**2+txz**2+tyz**2)) + 1e-30).astype(np.float32)


def compute_edge_features_torch(pos, edge_index):
    row, col = edge_index
    d = pos[col] - pos[row]
    return torch.cat([d, torch.norm(d, dim=1, keepdim=True)], dim=1)


# ?�?� Load stats ?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean     = node_stats["velocity_mean"].to(device)
v_std      = node_stats["velocity_std"].to(device)
d_mean     = node_stats["disp_mean"].to(device)
d_std      = node_stats["disp_std"].to(device)
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean_ref = edge_stats["edge_mean"].to(device)[:4]
e_std_ref  = edge_stats["edge_std"].to(device)[:4]
e_mean_wld = edge_stats["edge_mean"].to(device)[4:]
e_std_wld  = edge_stats["edge_std"].to(device)[4:]
empty_wef  = torch.zeros(0, NUM_EDGE_FEATURES, device=device)


def build_model(ckpt_path):
    m = HybridMeshGraphNet(
        NUM_INPUT_FEATURES, NUM_EDGE_FEATURES, NUM_OUTPUT_FEATURES,
        processor_size=PROCESSOR_SIZE,
        hidden_dim_processor=HIDDEN_DIM,
        hidden_dim_node_encoder=HIDDEN_DIM,
        hidden_dim_edge_encoder=HIDDEN_DIM,
        hidden_dim_node_decoder=HIDDEN_DIM,
        mlp_activation_fn="relu",
        do_concat_trick=False,
        num_processor_checkpoint_segments=0,
    ).to(device)
    m.eval()
    load_checkpoint(ckpt_path, models=m, device=device, epoch=CKPT_EPOCH)
    return m


@torch.no_grad()
def rollout_final_error(model, steps, ref_pos_np, mesh_ei_np,
                        use_per_sample_norm=False):
    T            = len(steps)
    ref_pos      = steps[0]["graph"].mesh_pos.to(device)
    mesh_ei      = steps[0]["graph"].edge_index.to(device)
    node_type_oh = steps[0]["graph"].x[:, 6:8].to(device)
    ref_ef_norm  = (compute_edge_features_torch(ref_pos, mesh_ei) - e_mean_ref) / e_std_ref

    gt_wp = [steps[0]["graph"].world_pos.clone()]
    for s in steps:
        gt_wp.append(gt_wp[-1] + s["graph"].y[:, :3] * v_std_cpu + v_mean_cpu)

    wp_cur   = gt_wp[1].to(device)
    prev_vel = (gt_wp[1] - gt_wp[0]).to(device)

    for _ in range(T - 1):
        if use_per_sample_norm:
            node_scale    = prev_vel.norm(dim=1, keepdim=True).mean().clamp(min=1e-6)
            disp_norm     = (wp_cur - ref_pos) / node_scale
            prev_vel_norm = prev_vel / node_scale
        else:
            disp_norm     = ((wp_cur - ref_pos) - d_mean) / d_std
            prev_vel_norm = (prev_vel - v_mean) / v_std

        node_x  = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)
        wf      = (compute_edge_features_torch(wp_cur, mesh_ei) - e_mean_wld) / e_std_wld
        mesh_ef = torch.cat([ref_ef_norm, wf], dim=1)
        g_in    = Data(x=node_x, edge_index=mesh_ei, num_nodes=ref_pos.shape[0])
        pred    = model(node_x, mesh_ef, empty_wef, g_in)

        if use_per_sample_norm:
            vel_phys = pred.float() * node_scale
        else:
            vel_phys = pred.float() * v_std + v_mean

        wp_cur   = wp_cur + vel_phys
        prev_vel = vel_phys

    pred_np = wp_cur.cpu().numpy()
    gt_np   = gt_wp[-1].numpy()
    ref_np  = ref_pos.cpu().numpy()

    num_p   = np.linalg.norm(pred_np - gt_np, axis=1).sum()
    den_p   = np.linalg.norm(gt_np - ref_np, axis=1).sum() + 1e-30
    pos_pct = num_p / den_p * 100.0

    ps = compute_vm_stress_np(pred_np, ref_np, mesh_ei_np)
    gs = compute_vm_stress_np(gt_np,   ref_np, mesh_ei_np)
    stress_pct = np.abs(ps - gs).sum() / (np.abs(gs).sum() + 1e-30) * 100.0

    return pos_pct, stress_pct


# ?�?� Pre-load samples and max_disp ?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�

all_files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.startswith("sample_") and f.endswith(".pt"))

_cache = {}

def load_indices(indices, tag):
    if tag in _cache:
        return _cache[tag]
    samples, max_disps = [], []
    print(f"Pre-loading [{tag}] ({len(indices)} samples) ...")
    for idx in indices:
        fname = all_files[idx]
        steps = torch.load(os.path.join(DATA_DIR, fname),
                           map_location="cpu", weights_only=False)
        samples.append(steps)
        npz   = np.load(os.path.join(NPZ_DIR, fname.replace(".pt", ".npz")))
        disp  = np.linalg.norm(npz["world_pos"][-1] - npz["mesh_pos"], axis=1).max()
        max_disps.append(float(disp))
    _cache[tag] = (samples, np.array(max_disps))
    return _cache[tag]


# ?�?� Run rollout for all models ?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�

results = {}

for label, ckpt_path, color, use_psn, tr_idx, te_idx in MODELS:
    if not os.path.exists(ckpt_path):
        print(f"[skip] {ckpt_path} not found")
        continue

    print(f"\n=== {label} ===")
    model = build_model(ckpt_path)
    results[label] = {"color": color}

    for split, indices in [("train", tr_idx), ("test", te_idx)]:
        tag = f"{split}_{indices[0]}_{indices[-1]}"
        samples, disps = load_indices(indices, tag)
        ref_np = samples[0][0]["graph"].mesh_pos.numpy()
        ei_np  = samples[0][0]["graph"].edge_index.numpy()

        pos_errs, stress_errs = [], []
        for i, steps in enumerate(samples):
            pe, se = rollout_final_error(model, steps, ref_np, ei_np,
                                         use_per_sample_norm=use_psn)
            pos_errs.append(pe)
            stress_errs.append(se)
            if (i + 1) % 20 == 0:
                print(f"  [{split}] {i+1}/{len(samples)} done")

        results[label][split] = {
            "pos":    np.array(pos_errs),
            "stress": np.array(stress_errs),
            "disps":  disps,
        }
        print(f"  [{split}] mean pos={np.mean(pos_errs):.2f}%  "
              f"stress={np.mean(stress_errs):.2f}%")

    del model
    torch.cuda.empty_cache()


# ?�?� Plot ?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "Per-sample Final Error vs Max Displacement  (t=19)\n",
    fontsize=20
)

MARKER_SIZE = 40
metrics = [("Displacement Error (%)", "pos"), ("Von Mises Stress Error (%)", "stress")]
splits  = [("Train", "train"), ("Test", "test")]

for row, (ylabel, metric_key) in enumerate(metrics):
    for col, (split_label, split_key) in enumerate(splits):
        ax = axes[row][col]
        for label, res in results.items():
            if split_key not in res:
                continue
            d    = res[split_key]
            errs = d[metric_key]
            ax.scatter(d["disps"] * 1000, errs,
                       color=res["color"], label=label,
                       alpha=0.65, s=MARKER_SIZE, edgecolors="none")
        ax.set_xlabel("Max Displacement at t=19 (mm)", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(f"{split_label} — {ylabel}", fontsize=17)
        ax.legend(fontsize=14)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)
        if metric_key == "stress":
            ax.set_ylim(0, 200)
        else:
            ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\n[saved] {OUTPUT_PNG}")
