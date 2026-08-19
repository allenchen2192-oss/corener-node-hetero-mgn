"""
validate_h256.py
================
Teacher-forced validation on S0201-S0220 for h256 checkpoints.
Checks every 5 epochs from epoch 500 onward.
Metric: vel + stress loss (normalized space, same as training).
"""

import os
import torch
from torch.amp import autocast
from torch_geometric.data import Batch

from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_DIR   = "./checkpoints_m0035_h256"
SAMPLE_DIR = "./04_preprocessed_pt_m0035"
VAL_IDS    = [f"S{i:04d}" for i in range(201, 221)]   # S0201–S0220

HIDDEN_DIM = 256
NUM_INPUT  = 17
NUM_EDGE   = 9
NUM_OUTPUT = 10
PROC_SIZE  = 15

START_EPOCH = 500
STEP        = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data loading ──────────────────────────────────────────────────────────────

def load_val_data(sample_dir, sample_ids):
    data = []
    print(f"Loading {len(sample_ids)} validation samples ...")
    for sid in sample_ids:
        pt_path = os.path.join(sample_dir, f"{sid}.pt")
        if not os.path.exists(pt_path):
            print(f"  MISSING: {pt_path}"); continue
        items = torch.load(pt_path, map_location="cpu", weights_only=False)
        data.extend(item for item in items if int(item["graph"].step_index.item()) > 0)
    print(f"  {len(data)} timesteps loaded.")
    return data


def collate_fn(batch):
    graphs    = [item["graph"]               for item in batch]
    mesh_efs  = [item["mesh_edge_features"]  for item in batch]
    world_efs = [item["world_edge_features"] for item in batch]
    return (
        Batch.from_data_list(graphs),
        torch.cat(mesh_efs,  dim=0),
        torch.cat(world_efs, dim=0),
    )


# ── Evaluate one checkpoint ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    criterion = torch.nn.MSELoss()
    vel_sum, stress_sum, peeq_sum, n = 0.0, 0.0, 0.0, 0

    for batched_graph, mesh_ef, world_ef in dataloader:
        batched_graph = batched_graph.to(DEVICE)
        mesh_ef       = mesh_ef.to(DEVICE)
        world_ef      = world_ef.to(DEVICE)

        with autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
            pred = model(batched_graph.x.float(), mesh_ef, world_ef, batched_graph)
        pred = pred.float()
        y    = batched_graph.y.to(DEVICE).float()

        vel_sum    += criterion(pred[:, :3],  y[:, :3]).item()
        stress_sum += criterion(pred[:, 3:9], y[:, 3:9]).item()

        sol = (batched_graph.node_mat == 2).to(DEVICE)
        if sol.any():
            peeq_sum += criterion(pred[sol, 9:10], y[sol, 9:10]).item()
        n += 1

    model.train()
    return vel_sum / n, stress_sum / n, peeq_sum / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    val_data   = load_val_data(SAMPLE_DIR, VAL_IDS)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=32, shuffle=False, collate_fn=collate_fn,
    )

    # Collect available epochs
    epochs = [
        ep for ep in range(START_EPOCH, 2001, STEP)
        if os.path.exists(os.path.join(CKPT_DIR, f"checkpoint.0.{ep}.pt"))
    ]
    if not epochs:
        print(f"No checkpoints found in {CKPT_DIR} from epoch {START_EPOCH}.")
        return
    print(f"Checking {len(epochs)} checkpoints: epoch {epochs[0]} → {epochs[-1]}\n")

    model = HybridMeshGraphNet(
        NUM_INPUT, NUM_EDGE, NUM_OUTPUT,
        processor_size=PROC_SIZE,
        hidden_dim_processor=HIDDEN_DIM,
        hidden_dim_node_encoder=HIDDEN_DIM,
        hidden_dim_edge_encoder=HIDDEN_DIM,
        hidden_dim_node_decoder=HIDDEN_DIM,
    ).to(DEVICE)

    CSV_PATH = os.path.join(CKPT_DIR, "validate_h256_results.csv")
    print(f"{'Epoch':>6}  {'vel':>10}  {'stress':>10}  {'peeq':>10}  {'vel+stress':>12}")
    print("-" * 58)

    results = []
    with open(CSV_PATH, "w") as fcsv:
        fcsv.write("epoch,vel,stress,peeq,vel+stress\n")
        for ep in epochs:
            load_checkpoint(CKPT_DIR, models=model, device=DEVICE, epoch=ep)

            vel, stress, peeq = evaluate(model, val_loader)
            combined = vel + stress
            results.append((ep, vel, stress, peeq, combined))
            print(f"{ep:>6}  {vel:>10.3e}  {stress:>10.3e}  {peeq:>10.3e}  {combined:>12.3e}", flush=True)
            fcsv.write(f"{ep},{vel:.6e},{stress:.6e},{peeq:.6e},{combined:.6e}\n")
            fcsv.flush()

    print("-" * 58)
    best_combined = min(results, key=lambda r: r[4])
    best_stress   = min(results, key=lambda r: r[2])
    best_vel      = min(results, key=lambda r: r[1])
    print(f"\nBest vel+stress : epoch {best_combined[0]}  vel={best_combined[1]:.3e}  stress={best_combined[2]:.3e}")
    print(f"Best stress     : epoch {best_stress[0]}  vel={best_stress[1]:.3e}  stress={best_stress[2]:.3e}")
    print(f"Best vel        : epoch {best_vel[0]}  vel={best_vel[1]:.3e}  stress={best_vel[2]:.3e}")
    print(f"\nResults saved → {CSV_PATH}")


if __name__ == "__main__":
    main()