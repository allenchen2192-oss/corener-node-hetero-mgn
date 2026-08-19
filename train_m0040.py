"""
train_m0040.py
Bipartite Heterogeneous GNN for M0040 IC package (Si / Solder / UF).

Graph structure:
  node    -> node    : C3D8 single graph mesh edges  (8D edge features)
  node    -> element : B_{0,2}                        (node belongs to element)
  element -> node    : B_{0,2}^T                      (element contains node)

Corner node features (11D):
  disp(3) + prev_vel(3) + node_type_oh(5)

Element node features (8D):
  mat_oh(3) + E_norm(1) + CTE_norm(1) + nu_norm(1) + vm_stress_t(1) + peeq_t(1)

Targets:
  corner node  : velocity     (3D)
  element node : [vm_stress, peeq]  (2D)
    peeq loss applied only on solder elements (masked MSE)
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import SAGEConv, MessagePassing
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint


# ── Dataset ───────────────────────────────────────────────────────────────────

class M0040Dataset(torch.utils.data.Dataset):
    """
    Loads preprocessed HeteroData .pt files from preprocess_m0040.py (Option B).
    Each S####.pt is a list of 19 HeteroData objects (t=1..19); they are
    flattened into a single list at construction time.
    """

    def __init__(self, preprocess_dir, sample_ids):
        self.hdata = []
        print(f"Loading {len(sample_ids)} samples from {preprocess_dir} ...")
        for sid in tqdm(sample_ids):
            pt_path = os.path.join(preprocess_dir, f"{sid}.pt")
            if not os.path.exists(pt_path):
                print(f"  MISSING: {sid}.pt")
                continue
            steps = torch.load(pt_path, map_location="cpu", weights_only=False)
            self.hdata.extend(steps)
        print(f"  {len(self.hdata)} total timesteps loaded.")

    def __len__(self):
        return len(self.hdata)

    def __getitem__(self, idx):
        return self.hdata[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class NodeEdgeConv(MessagePassing):
    """
    MLP-based message passing with encoded edge features (node->node).
    message(v->u) = MLP(cat(x_v, edge_emb_{v->u}))
    """

    def __init__(self, node_dim, edge_dim, out_dim, hidden):
        super().__init__(aggr="sum")
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class HeteroMGN(nn.Module):
    """
    Bipartite Heterogeneous MeshGraphNet (Hasse Graph Trick).

    Three message types per processor layer:
      node -> node    : NodeEdgeConv with 8D mesh edge features
      node -> element : SAGEConv  (B_{0,2}, no edge features)
      element -> node : SAGEConv  (B_{0,2}^T, no edge features)

    Edge embeddings are computed once before the processor loop and reused.
    """

    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim

        # Encoders
        self.node_enc = MLP(cfg.num_node_features, h, h)
        self.elem_enc = MLP(cfg.num_elem_features, h, h)
        self.edge_enc = MLP(cfg.num_edge_features, h, h)

        # Processor layers
        self.node_node_conv = nn.ModuleList([
            NodeEdgeConv(h, h, h, h) for _ in range(cfg.processor_size)
        ])
        self.node_elem_conv = nn.ModuleList([
            SAGEConv((h, h), h, aggr="mean") for _ in range(cfg.processor_size)
        ])
        self.elem_node_conv = nn.ModuleList([
            SAGEConv((h, h), h, aggr="mean") for _ in range(cfg.processor_size)
        ])

        # Residual update MLPs
        self.node_mlp = nn.ModuleList([MLP(h, h, h) for _ in range(cfg.processor_size)])
        self.elem_mlp = nn.ModuleList([MLP(h, h, h) for _ in range(cfg.processor_size)])

        # Decoders
        self.node_dec = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cfg.num_node_outputs)
        )
        self.elem_dec = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cfg.num_elem_outputs)
        )

    def forward(self, data):
        x_n = self.node_enc(data["node"].x)
        x_e = self.elem_enc(data["element"].x)

        ei_nn  = data["node",    "mesh", "node"   ].edge_index
        ea_nn  = data["node",    "mesh", "node"   ].edge_attr   # (E_mesh, 8)
        ei_n2e = data["node",    "in",   "element"].edge_index
        ei_e2n = data["element", "has",  "node"   ].edge_index

        ea_emb = self.edge_enc(ea_nn)   # encode once, reuse each layer

        for i in range(len(self.node_node_conv)):
            msg_nn  = self.node_node_conv[i](x_n, ei_nn, ea_emb)
            msg_n2e = self.node_elem_conv[i]((x_n, x_e), ei_n2e)
            msg_e2n = self.elem_node_conv[i]((x_e, x_n), ei_e2n)

            x_n = self.node_mlp[i](msg_nn + msg_e2n + x_n)
            x_e = self.elem_mlp[i](msg_n2e + x_e)

        vel_pred  = self.node_dec(x_n)   # (N, 3)
        elem_pred = self.elem_dec(x_e)   # (M, 2)  [vm_stress, peeq]
        return vel_pred, elem_pred


# ── Training ──────────────────────────────────────────────────────────────────

@hydra.main(config_path="conf", config_name="config_m0040", version_base=None)
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist   = DistributedManager()
    device = dist.device

    if dist.rank == 0:
        writer = SummaryWriter(to_absolute_path(cfg.tensorboard_log_dir))
        print(f"Device: {device}  |  world_size: {dist.world_size}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if hasattr(cfg, "train_exclude_ids") and cfg.train_exclude_ids:
        exclude = set(cfg.train_exclude_ids)
        train_ids = [f"S{i:04d}" for i in range(1, 221) if i not in exclude]
    else:
        train_ids = [f"S{i:04d}" for i in range(cfg.train_start, cfg.train_end + 1)]

    dataset = M0040Dataset(
        preprocess_dir=to_absolute_path(cfg.preprocess_dir),
        sample_ids=train_ids,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, drop_last=True,
        num_replicas=dist.world_size, rank=dist.rank,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size         =cfg.batch_size,
        sampler            =sampler,
        num_workers        =cfg.num_dataloader_workers,
        collate_fn         =Batch.from_data_list,
        pin_memory         =True,
        persistent_workers =(cfg.num_dataloader_workers > 0),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model     = HeteroMGN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate ** epoch
    )
    scaler = GradScaler()

    os.makedirs(to_absolute_path(cfg.ckpt_path), exist_ok=True)
    epoch_init = load_checkpoint(
        to_absolute_path(cfg.ckpt_path),
        models=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, device=device,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(epoch_init, cfg.epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_vel = total_vm = total_peeq = total_n = 0.0
        t_epoch_start = time.time()

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Displacement noise in normalized space
            if cfg.input_noise_std > 0:
                node_x = batch["node"].x.clone()
                node_x[:, :3] += torch.randn_like(node_x[:, :3]) * cfg.input_noise_std
                batch["node"].x = node_x

            with autocast("cuda", enabled=cfg.amp):
                vel_pred, elem_pred = model(batch)

                vel_gt  = batch["node"].y                 # (N_batch, 3)
                elem_gt = batch["element"].y              # (M_batch, 2)

                loss_vel = F.mse_loss(vel_pred, vel_gt)
                loss_vm  = F.mse_loss(elem_pred[:, 0:1], elem_gt[:, 0:1])

                # PEEQ loss: solder elements only
                solder_mask = batch["element"].solder_mask.bool()
                loss_peeq = F.mse_loss(
                    elem_pred[solder_mask, 1],
                    elem_gt[solder_mask, 1],
                )

                loss = cfg.w_vel * loss_vel + cfg.w_vm * loss_vm + cfg.w_peeq * loss_peeq

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            total_vel  += loss_vel.item()
            total_vm   += loss_vm.item()
            total_peeq += loss_peeq.item()
            total_n    += 1

        scheduler.step()

        if dist.rank == 0:
            avg_vel  = total_vel  / max(total_n, 1)
            avg_vm   = total_vm   / max(total_n, 1)
            avg_peeq = total_peeq / max(total_n, 1)
            epoch_sec = time.time() - t_epoch_start
            print(f"Epoch {epoch:4d} | vel={avg_vel:.3e}  vm={avg_vm:.3e}  peeq={avg_peeq:.3e}  [{epoch_sec:.1f}s]")
            writer.add_scalar("loss/vel",  avg_vel,  epoch)
            writer.add_scalar("loss/vm",   avg_vm,   epoch)
            writer.add_scalar("loss/peeq", avg_peeq, epoch)

            if epoch % cfg.ckpt_interval == 0:
                save_checkpoint(
                    to_absolute_path(cfg.ckpt_path),
                    models=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, epoch=epoch,
                )

    if dist.rank == 0:
        writer.close()


if __name__ == "__main__":
    main()
