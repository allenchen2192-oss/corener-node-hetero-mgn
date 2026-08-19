"""
train_m0040_cornernode.py
Corner-node-only MeshGraphNet for M0040 IC package.

No element nodes. Material info encoded in multigraph edges (key=src,dst,mat_id).
Node-level vm_stress and peeq predicted directly at corner nodes.

Node features (13D):
  disp(3) + prev_vel(3) + node_type_oh(5) + prev_vm_stress_node(1) + prev_peeq_node(1)

Edge features (14D):
  geom(8) + mat_oh(3) + E_norm(1) + CTE_norm(1) + nu_norm(1)

Targets (5D per node):
  vel(3) + vm_stress_node(1) + peeq_node(1)
  peeq loss masked to solder_node_mask only
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint


# ── Dataset ───────────────────────────────────────────────────────────────────

class CornerNodeDataset(torch.utils.data.Dataset):
    def __init__(self, preprocess_dir, sample_ids):
        self.data = []
        print(f"Loading {len(sample_ids)} samples from {preprocess_dir} ...")
        for sid in tqdm(sample_ids):
            pt_path = os.path.join(preprocess_dir, f"{sid}.pt")
            if not os.path.exists(pt_path):
                print(f"  MISSING: {sid}.pt")
                continue
            steps = torch.load(pt_path, map_location="cpu", weights_only=False)
            self.data.extend(steps)
        print(f"  {len(self.data)} total timesteps loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class NodeEdgeConv(MessagePassing):
    """message(v->u) = MLP(cat(x_v, edge_emb_{v->u}))"""

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


class CornerNodeMGN(nn.Module):
    """
    Standard MeshGraphNet on corner-node-only multigraph.

    Per processor layer:
      msg  = NodeEdgeConv(x_n, edge_index, edge_emb)
      x_n  = node_mlp(msg + x_n)            # residual update

    Edge embeddings are encoded once before the processor loop and reused.
    """

    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim

        self.node_enc = MLP(cfg.num_node_features, h, h)
        self.edge_enc = MLP(cfg.num_edge_features, h, h)

        self.conv     = nn.ModuleList([NodeEdgeConv(h, h, h, h) for _ in range(cfg.processor_size)])
        self.node_mlp = nn.ModuleList([MLP(h, h, h)             for _ in range(cfg.processor_size)])

        self.decoder = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cfg.num_outputs)
        )

    def forward(self, data):
        x        = self.node_enc(data.x)
        edge_emb = self.edge_enc(data.edge_attr)

        for conv, mlp in zip(self.conv, self.node_mlp):
            msg = conv(x, data.edge_index, edge_emb)
            x   = mlp(msg + x)

        return self.decoder(x)   # (N, 5): vel(3) + vm_stress(1) + peeq(1)


# ── Training ──────────────────────────────────────────────────────────────────

@hydra.main(config_path="conf", config_name="config_m0040_cornernode", version_base=None)
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist   = DistributedManager()
    device = dist.device

    if dist.rank == 0:
        writer = SummaryWriter(to_absolute_path(cfg.tensorboard_log_dir))
        print(f"Device: {device}  |  world_size: {dist.world_size}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ids = [f"S{i:04d}" for i in range(cfg.train_start, cfg.train_end + 1)]

    dataset = CornerNodeDataset(
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
    model     = CornerNodeMGN(cfg).to(device)
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

            if cfg.input_noise_std > 0:
                x = batch.x.clone()
                x[:, :3] += torch.randn_like(x[:, :3]) * cfg.input_noise_std
                batch.x = x

            with autocast("cuda", enabled=cfg.amp):
                pred = model(batch)              # (N_batch, 5)
                gt   = batch.y                   # (N_batch, 5)

                loss_vel  = F.mse_loss(pred[:, 0:3], gt[:, 0:3])
                loss_vm   = F.mse_loss(pred[:, 3],   gt[:, 3])

                solder_mask = batch.solder_node_mask.bool()
                loss_peeq = F.mse_loss(
                    pred[solder_mask, 4],
                    gt[solder_mask,   4],
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
