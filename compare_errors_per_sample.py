"""
compare_errors_per_sample.py
=============================
Per-sample error scatter plot for beam bending models.
X-axis: max displacement of each sample (from NPZ ground truth)
Y-axis: final rollout displacement / stress error at t=19

Shows whether large vs small deformation explains error differences.

Output: compare_errors_per_sample.png
"""

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


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/04_preprocessed_pt_beam_prevvel"
NPZ_DIR  = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/02_beam_npz"

MODELS = [
    # (label, ckpt_path, color, use_per_sample_norm)
    ("w_stress=0.5",                      "./checkpoints_3Dbeam_no_rod_stress05",                    "tab:blue",  False),
    ("w_stress=0.5, rel loss",            "./checkpoints_3Dbeam_no_rod_stress05_relloss",            "tab:red",   False),
    ("w_stress=0.5, per-sample, clamp1e6","./checkpoints_3Dbeam_no_rod_stress05_persample_clamp1e6","tab:green", True),
]

TRAIN_INDICES = list(range(0,25)) + list(range(275,300)) + list(range(550,575)) + list(range(825,850))   # 25 per direction
TEST_INDICES  = list(range(250,275)) + list(range(525,550)) + list(range(800,825)) + list(range(1075,1100))  # 25 per direction
CKPT_EPOCH    = None

NUM_INPUT_FEATURES  = 8
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128

OUTPUT_PNG = "compare_errors_per_sample.png"


# ── Physics ───────────────────────────────────────────────────────────────────

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


# ── Load stats ────────────────────────────────────────────────────────────────

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
    """Returns (final_pos_pct, final_stress_pct) at last timestep."""
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

    for k in range(T - 1):
        if use_per_sample_norm:
            # 與 train_prevvel.py forward() 完全一致
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

    # Displacement error
    num_p = np.linalg.norm(pred_np - gt_np, axis=1).sum()
    den_p = np.linalg.norm(gt_np - ref_pos_np, axis=1).sum() + 1e-30
    pos_pct = num_p / den_p * 100.0

    # Stress error
    ps = compute_vm_stress_np(pred_np, ref_pos_np, mesh_ei_np)
    gs = compute_vm_stress_np(gt_np,   ref_pos_np, mesh_ei_np)
    stress_pct = np.abs(ps - gs).sum() / (np.abs(gs).sum() + 1e-30) * 100.0

    return pos_pct, stress_pct


# ── Load test PT files & get max_disp from NPZ ────────────────────────────────

all_files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.startswith("sample_") and f.endswith(".pt"))

def load_split_with_disp(indices, label):
    samples, max_disps = [], []
    print(f"Pre-loading {label} samples ({len(indices)}) ...")
    for idx in indices:
        fname     = all_files[idx]
        sample_id = fname.replace(".pt", "")
        steps     = torch.load(os.path.join(DATA_DIR, fname),
                               map_location="cpu", weights_only=False)
        samples.append(steps)
        npz_path = os.path.join(NPZ_DIR, sample_id + ".npz")
        d        = np.load(npz_path)
        disp_mag = np.linalg.norm(d["world_pos"][-1] - d["mesh_pos"], axis=1).max()
        max_disps.append(float(disp_mag))
    return samples, np.array(max_disps)

train_samples, train_disps = load_split_with_disp(TRAIN_INDICES, "train")
test_samples,  test_disps  = load_split_with_disp(TEST_INDICES,  "test")

ref_pos_np = test_samples[0][0]["graph"].mesh_pos.numpy()
mesh_ei_np = test_samples[0][0]["graph"].edge_index.numpy()

print(f"train max_disp: {train_disps.min():.4f} – {train_disps.max():.4f} m")
print(f"test  max_disp: {test_disps.min():.4f}  – {test_disps.max():.4f} m")


# ── Run per-sample rollout for all models ─────────────────────────────────────

# results[label] = {"train": (pos_errs, stress_errs), "test": ..., "color": ...}
results = {}

for label, ckpt_path, color, use_psn in MODELS:
    if not os.path.exists(ckpt_path):
        print(f"[skip] {ckpt_path}")
        continue
    print(f"\n=== {label} ===")
    model = build_model(ckpt_path)
    results[label] = {"color": color}

    for split, samples in [("train", train_samples), ("test", test_samples)]:
        pos_errs, stress_errs = [], []
        for i, steps in enumerate(samples):
            pos_pct, stress_pct = rollout_final_error(
                model, steps, ref_pos_np, mesh_ei_np,
                use_per_sample_norm=use_psn)
            pos_errs.append(pos_pct)
            stress_errs.append(stress_pct)
            if (i + 1) % 20 == 0:
                print(f"  [{split}] {i+1}/{len(samples)} done")
        results[label][split] = (np.array(pos_errs), np.array(stress_errs))
        print(f"  [{split}] mean pos={np.mean(pos_errs):.2f}%  stress={np.mean(stress_errs):.2f}%")

    del model
    torch.cuda.empty_cache()


# ── Plot: 2 rows × 2 cols  (row=metric, col=train/test) ──────────────────────

FONT_TITLE   = 18
FONT_LABEL   = 15
FONT_TICK    = 13
FONT_LEGEND  = 12
FONT_SUPTITLE= 20
MARKER_SIZE  = 40

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Per-sample Final Error vs Max Displacement  (t=19)\n"
             "Left=Train (100)  Right=Test (100)", fontsize=FONT_SUPTITLE)

metrics = [("Displacement Error (%)", 0), ("Von Mises Stress Error (%)", 1)]
splits  = [("Train", "train", train_disps), ("Test", "test", test_disps)]

for row, (ylabel, err_idx) in enumerate(metrics):
    for col, (split_label, split_key, disps) in enumerate(splits):
        ax = axes[row][col]
        for label, res in results.items():
            pos_e, stress_e = res[split_key]
            errs  = pos_e if err_idx == 0 else stress_e
            color = res["color"]
            ax.scatter(disps * 1000, errs, color=color, label=label,
                       alpha=0.65, s=MARKER_SIZE, edgecolors="none")

        ax.set_xlabel("Max Displacement at t=19 (mm)", fontsize=FONT_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.set_title(f"{split_label} — {ylabel}", fontsize=FONT_TITLE)
        ax.legend(fontsize=FONT_LEGEND)
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\n[saved] {OUTPUT_PNG}")
