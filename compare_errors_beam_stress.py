"""
compare_errors_beam_stress.py
==============================
比較 3D beam bending 兩個模型的 rollout 誤差：
  - Model A: checkpoints_3Dbeam_no_rod        (w_stress=0)
  - Model B: checkpoints_3Dbeam_no_rod_stress05 (w_stress=0.5)

誤差定義（與 compare_errors.py 一致）：
  pos_err(t)    = Σ_l ||p̂_l(t) - p_l(t)||₂ / Σ_l ||p_l(t) - p_ref_l||₂ × 100%
  stress_err(t) = Σ_l |σ̂_l(t) - σ_l(t)|   / Σ_l |σ_l(t)|               × 100%
  （stress 用 GFDM 從位移場計算）

Rollout seed: t=1（第一個載荷步），預測 t=2..T
Output: compare_errors_beam_stress.png
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

MODELS = [
    ("w_stress=0",              "./checkpoints_3Dbeam_no_rod",                   "tab:green"),
    ("w_stress=0.5",            "./checkpoints_3Dbeam_no_rod_stress05",          "tab:blue"),
    ("w_stress=0.5, low noise", "./checkpoints_3Dbeam_no_rod_stress05_lownoise", "tab:red"),
]

NUM_TRAIN = 100   # samples 00000–00099
NUM_TEST  = 100   # samples 01000–01099
TRAIN_INDICES = list(range(0,    NUM_TRAIN))
TEST_INDICES  = list(range(1000, 1000 + NUM_TEST))
CKPT_EPOCH    = None   # None = latest

NUM_INPUT_FEATURES  = 8
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128

OUTPUT_PNG = "compare_errors_beam_stress.png"


# ── Physics ───────────────────────────────────────────────────────────────────

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


def compute_vm_stress_np(world_pos, ref_pos, edge_index):
    """Returns (N,) Von Mises stress in Pa via GFDM (numpy)."""
    N       = ref_pos.shape[0]
    row, col = edge_index
    r_ij    = ref_pos[col] - ref_pos[row]        # (E, 3)
    du_ij   = (world_pos - ref_pos)[col] - (world_pos - ref_pos)[row]  # (E, 3)
    r_r     = r_ij[:, :, None] * r_ij[:, None, :]   # (E, 3, 3)
    r_du    = r_ij[:, :, None] * du_ij[:, None, :]
    XtX = np.zeros((N, 3, 3), dtype=np.float64)
    XtY = np.zeros((N, 3, 3), dtype=np.float64)
    np.add.at(XtX, row, r_r)
    np.add.at(XtY, row, r_du)
    XtX += 1e-6 * np.eye(3)[None]
    dU   = np.linalg.solve(XtX, XtY).transpose(0, 2, 1)   # (N, 3, 3)
    tr   = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx   = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy   = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz   = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy  = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz  = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz  = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                           + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30).astype(np.float32)


def compute_edge_features_torch(pos, edge_index):
    row, col = edge_index
    d = pos[col] - pos[row]
    return torch.cat([d, torch.norm(d, dim=1, keepdim=True)], dim=1)


# ── Load stats ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats  = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean      = node_stats["velocity_mean"].to(device)
v_std       = node_stats["velocity_std"].to(device)
d_mean      = node_stats["disp_mean"].to(device)
d_std       = node_stats["disp_std"].to(device)
v_mean_cpu  = v_mean.cpu()
v_std_cpu   = v_std.cpu()

edge_stats  = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean_ref  = edge_stats["edge_mean"].to(device)[:4]
e_std_ref   = edge_stats["edge_std"].to(device)[:4]
e_mean_wld  = edge_stats["edge_mean"].to(device)[4:]
e_std_wld   = edge_stats["edge_std"].to(device)[4:]

empty_wef   = torch.zeros(0, NUM_EDGE_FEATURES, device=device)


# ── Model builder ─────────────────────────────────────────────────────────────

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
        recompute_activation=False,
    ).to(device)
    m.eval()
    load_checkpoint(ckpt_path, models=m, device=device, epoch=CKPT_EPOCH)
    return m


# ── Rollout for one sample ────────────────────────────────────────────────────

@torch.no_grad()
def rollout_sample(model, steps):
    """
    steps: list of dicts loaded from sample_*.pt
    Returns: (pred_wp_list, gt_wp_list) each length T-1
      pred_wp_list[k] = predicted world_pos at timestep k+2  (numpy)
      gt_wp_list[k]   = GT world_pos at timestep k+2         (numpy)
    """
    T = len(steps)   # number of transitions = F-1

    # Reconstruct GT world positions from stored velocities
    ref_pos      = steps[0]["graph"].mesh_pos.to(device)
    mesh_ei      = steps[0]["graph"].edge_index.to(device)
    node_type_oh = steps[0]["graph"].x[:, 6:8].to(device)   # (N, 2), constant

    gt_wp = [steps[0]["graph"].world_pos.clone()]   # t=0
    for s in steps:
        vel_raw = s["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp.append(gt_wp[-1] + vel_raw)
    # gt_wp[0..T]: world_pos at t=0 to t=T

    # Precompute normalized ref edge features
    ref_ef_norm = (compute_edge_features_torch(ref_pos, mesh_ei) - e_mean_ref) / e_std_ref

    # Seed at t=1
    wp_cur   = gt_wp[1].to(device)
    prev_vel = (gt_wp[1] - gt_wp[0]).to(device)

    pred_list, gt_list = [], []
    for k in range(T - 1):   # predict t=2..T
        disp_norm     = ((wp_cur - ref_pos) - d_mean) / d_std
        prev_vel_norm = (prev_vel - v_mean) / v_std
        node_x        = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)

        wf      = (compute_edge_features_torch(wp_cur, mesh_ei) - e_mean_wld) / e_std_wld
        mesh_ef = torch.cat([ref_ef_norm, wf], dim=1)

        g_in = Data(x=node_x, edge_index=mesh_ei, num_nodes=ref_pos.shape[0],
                    mesh_pos=ref_pos, world_pos=wp_cur)
        pred     = model(node_x, mesh_ef, empty_wef, g_in)
        vel_phys = pred.float() * v_std + v_mean
        wp_next  = wp_cur + vel_phys

        pred_list.append(wp_next.cpu().numpy())
        gt_list.append(gt_wp[k + 2].numpy())
        wp_cur   = wp_next
        prev_vel = vel_phys

    return pred_list, gt_list


def compute_errors(pred_list, gt_list, ref_np, mesh_ei_np):
    """Returns pos_pct and stress_pct arrays of shape (rollout_steps,)."""
    pos_pct, stress_pct = [], []
    for pred_wp, gt_wp in zip(pred_list, gt_list):
        # Displacement error
        num_p = np.linalg.norm(pred_wp - gt_wp, axis=1).sum()
        den_p = np.linalg.norm(gt_wp   - ref_np, axis=1).sum() + 1e-30
        pos_pct.append(num_p / den_p * 100.0)

        # Stress error
        ps = compute_vm_stress_np(pred_wp, ref_np, mesh_ei_np)
        gs = compute_vm_stress_np(gt_wp,   ref_np, mesh_ei_np)
        num_s = np.abs(ps - gs).sum()
        den_s = np.abs(gs).sum() + 1e-30
        stress_pct.append(num_s / den_s * 100.0)

    return np.array(pos_pct), np.array(stress_pct)


# ── Pre-load samples ──────────────────────────────────────────────────────────

all_files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.startswith("sample_") and f.endswith(".pt"))

def load_split(indices, label):
    files = []
    for idx in indices:
        if idx < len(all_files):
            files.append(os.path.join(DATA_DIR, all_files[idx]))
        else:
            print(f"[warn] index {idx} out of range")
    print(f"Pre-loading {label} ({len(files)} samples) ...")
    return [torch.load(fp, map_location="cpu", weights_only=False) for fp in files]

train_samples = load_split(TRAIN_INDICES, "train")
test_samples  = load_split(TEST_INDICES,  "test")

ref_pos_np = train_samples[0][0]["graph"].mesh_pos.numpy()
mesh_ei_np = train_samples[0][0]["graph"].edge_index.numpy()


# ── Run rollout for all models ────────────────────────────────────────────────

# results[label] = {"train": (pos_arr, stress_arr), "test": (pos_arr, stress_arr)}
results = {}

for label, ckpt_path, color in MODELS:
    if not os.path.exists(ckpt_path):
        print(f"[skip] checkpoint not found: {ckpt_path}")
        continue

    print(f"\n=== {label} ===")
    model = build_model(ckpt_path)
    results[label] = {"color": color}

    for split_name, split_samples, split_indices in [
        ("train", train_samples, TRAIN_INDICES),
        ("test",  test_samples,  TEST_INDICES),
    ]:
        pos_list, stress_list = [], []
        for i, steps in enumerate(split_samples):
            pred_list, gt_list = rollout_sample(model, steps)
            pos_pct, stress_pct = compute_errors(pred_list, gt_list, ref_pos_np, mesh_ei_np)
            pos_list.append(pos_pct)
            stress_list.append(stress_pct)
        pos_arr    = np.stack(pos_list,    axis=0)
        stress_arr = np.stack(stress_list, axis=0)
        results[label][split_name] = (pos_arr, stress_arr)
        print(f"  [{split_name}] final pos={pos_arr[:,-1].mean():.2f}%±{pos_arr[:,-1].std():.2f}%  "
              f"stress={stress_arr[:,-1].mean():.2f}%±{stress_arr[:,-1].std():.2f}%")

    del model
    torch.cuda.empty_cache()


# ── Plot ──────────────────────────────────────────────────────────────────────

if not results:
    print("No results to plot.")
else:
    first = next(iter(results.values()))
    rollout_steps = first["train"][0].shape[1]
    t_axis = np.arange(2, rollout_steps + 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"3D Beam Rollout Error Comparison\n"
        f"(mean over {NUM_TRAIN} train / {NUM_TEST} test samples)",
        fontsize=13,
    )

    for ax, (ylabel, idx) in zip(axes, [("Displacement Error (%)", 0),
                                         ("Von Mises Stress Error (%)", 1)]):
        for label, res in results.items():
            color = res["color"]
            for split, ls, marker, mfc in [
                ("train", "-",  "o", color),
                ("test",  "--", "o", "white"),
            ]:
                arr = res[split][idx]
                mu  = arr.mean(0)
                ax.plot(t_axis, mu,
                        color=color, linestyle=ls, linewidth=2,
                        marker=marker, markersize=5, markevery=2,
                        markerfacecolor=mfc, markeredgecolor=color,
                        label=f"{label} – {'Training' if split=='train' else 'Test'}")
                ax.fill_between(t_axis, mu - arr.std(0), mu + arr.std(0),
                                color=color, alpha=0.1)

        ax.set_xlabel("Time step t", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"\n[saved] {OUTPUT_PNG}")
