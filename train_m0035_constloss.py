"""
Train MeshGraphNet on M0035 IC package with constitutive consistency loss.

Extends train_m0035.py by adding a physics regularisation term:
  loss_const = MSE( sigma_const(u_pred), sigma_pred )  for Si + UF nodes

sigma_const is computed via B-matrix (FEM constitutive formula):
  sigma_const_node = vol-weighted average of C:(B_e @ u_pred - alpha*dT*I)
  over all elastic neighboring elements.

Both sigma_const and sigma_pred are compared in normalized stress space,
so w_const is on the same scale as w_stress.

Requires preprocess_m0035_constloss.py to have been run first to generate
the _topo.pt files in cfg.topo_dir.
"""

import json
import os
import time

import hydra
from hydra.utils import to_absolute_path
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    graphs    = [seq[0]["graph"]               for seq in batch]
    mesh_efs  = [seq[0]["mesh_edge_features"]  for seq in batch]
    world_efs = [seq[0]["world_edge_features"] for seq in batch]
    topos     = [seq[0]["topo"]                for seq in batch]

    batched_graph = Batch.from_data_list(graphs)

    # Build batched element tensors with per-sample node offset
    elem_conns, elem_mats, elem_Bs, elem_vols = [], [], [], []
    elem_Es, elem_nus, elem_CTEs, elem_batch_ids = [], [], [], []

    for i, topo in enumerate(topos):
        offset = int(batched_graph.ptr[i])
        n_el   = topo["elem_conn"].shape[0]
        elem_conns.append(topo["elem_conn"] + offset)
        elem_mats.append(topo["elem_mat"])
        elem_Bs.append(topo["elem_B"])
        elem_vols.append(topo["elem_vol"])
        elem_Es.append(topo["elem_E"])
        elem_nus.append(topo["elem_nu"])
        elem_CTEs.append(topo["elem_CTE"])
        elem_batch_ids.append(torch.full((n_el,), i, dtype=torch.long))

    topo_batch = {
        "elem_conn":  torch.cat(elem_conns,     dim=0),   # (B*E, 8)
        "elem_mat":   torch.cat(elem_mats,      dim=0),   # (B*E,)
        "elem_B":     torch.cat(elem_Bs,        dim=0),   # (B*E, 3, 8)
        "elem_vol":   torch.cat(elem_vols,      dim=0),   # (B*E,)
        "elem_E":     torch.cat(elem_Es,        dim=0),   # (B*E,)
        "elem_nu":    torch.cat(elem_nus,       dim=0),   # (B*E,)
        "elem_CTE":   torch.cat(elem_CTEs,      dim=0),   # (B*E,)
        "elem_batch": torch.cat(elem_batch_ids, dim=0),   # (B*E,) sample index
    }

    return (
        batched_graph,
        torch.cat(mesh_efs,  dim=0),
        torch.cat(world_efs, dim=0),
        topo_batch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class M0035Dataset(torch.utils.data.Dataset):
    def __init__(self, sample_dir, topo_dir, sample_ids):
        self.data = []
        print(f"Loading {len(sample_ids)} sample files into memory ...")
        for sid in sample_ids:
            pt_path   = os.path.join(sample_dir, f"{sid}.pt")
            topo_path = os.path.join(topo_dir,   f"{sid}_topo.pt")
            if not os.path.exists(pt_path):
                print(f"  MISSING: {pt_path}")
                continue
            if not os.path.exists(topo_path):
                print(f"  MISSING: {topo_path}")
                continue
            sample_data = torch.load(pt_path,   map_location="cpu", weights_only=False)
            topo        = torch.load(topo_path, map_location="cpu", weights_only=False)
            for item in sample_data:
                if int(item["graph"].step_index.item()) > 0:
                    item["topo"] = topo   # shared reference across timesteps
                    self.data.append(item)
        print(f"  {len(self.data)} timesteps loaded.")

    def __getitem__(self, idx):
        return [self.data[idx]]

    def __len__(self):
        return len(self.data)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class M0035Trainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp  = cfg.amp

        mlp_act    = "silu" if cfg.recompute_activation else "relu"
        sample_dir = to_absolute_path(cfg.preprocess_output_dir)
        topo_dir   = to_absolute_path(cfg.topo_dir)

        train_ids = [f"S{i:04d}" for i in range(1, cfg.train_n + 1)]
        dataset   = M0035Dataset(sample_dir, topo_dir, train_ids)

        sampler = DistributedSampler(
            dataset, shuffle=True, drop_last=True,
            num_replicas=self.dist.world_size, rank=self.dist.rank,
        )
        n_workers = cfg.num_dataloader_workers
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=sampler,
            pin_memory=False, num_workers=n_workers,
            collate_fn=collate_fn,
            persistent_workers=(n_workers > 0),
        )
        self.sampler = sampler

        val_n = getattr(cfg, "val_n", 0)
        if val_n > 0:
            val_ids = [f"S{i:04d}" for i in range(cfg.train_n + 1, cfg.train_n + val_n + 1)]
            val_dataset = M0035Dataset(sample_dir, topo_dir, val_ids)
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=cfg.batch_size, shuffle=False,
                pin_memory=False, num_workers=n_workers,
                collate_fn=collate_fn,
            )
        else:
            self.val_dataloader = None
        self._val_every = getattr(cfg, "val_every", 1)

        self.model = HybridMeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            processor_size=cfg.processor_size,
            hidden_dim_processor=cfg.hidden_dim,
            hidden_dim_node_encoder=cfg.hidden_dim,
            hidden_dim_edge_encoder=cfg.hidden_dim,
            hidden_dim_node_decoder=cfg.hidden_dim,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        ).to(self.dist.device)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()
        self.criterion = torch.nn.MSELoss()
        self.w_vel    = cfg.w_vel
        self.w_stress = cfg.w_stress
        self.w_peeq   = cfg.w_peeq
        self.w_const  = 0.0   # always starts at 0; activated by stress threshold
        self._train_step      = 0
        self._stress_every    = getattr(cfg, "stress_every", 1)
        self._stage1_threshold = getattr(cfg, "w_stress_stage1_threshold", None)
        self._stage2_threshold = getattr(cfg, "w_stress_stage2_threshold", None)
        self._warmup_stage    = 0
        self._cfg_w_peeq      = cfg.w_peeq

        self._load_stats(cfg)

        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam
                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except (ImportError, AttributeError):
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, foreach=True
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

        init_ckpt = getattr(cfg, "init_ckpt_path", None)
        if self.epoch_init == 0 and init_ckpt:
            load_checkpoint(
                to_absolute_path(init_ckpt),
                models=self.model,
                device=self.dist.device,
            )
            if self.dist.rank == 0:
                print(f"[warm-start] loaded model weights from {init_ckpt}")

        if self._stage1_threshold is not None:
            if self.epoch_init == 0:
                self.w_stress = 0.0
                self.w_peeq   = 0.0
                # w_const already 0; activated separately by stress threshold
            else:
                self._warmup_stage = 2
                # _const_stage stays 0; will recheck stress threshold

        if self.dist.rank == 0:
            self.writer = SummaryWriter(
                log_dir=to_absolute_path(cfg.tensorboard_log_dir)
            )

        self.input_noise_std = getattr(cfg, "input_noise_std", 0.0)
        self._cfg_w_const             = getattr(cfg, "w_const", 0.0)
        self._const_stress_threshold  = getattr(cfg, "w_const_stress_threshold", None)
        self._const_stage             = 0

    # ── stats ────────────────────────────────────────────────────────────────

    def _load_stats(self, cfg):
        sample_dir = to_absolute_path(cfg.preprocess_output_dir)
        with open(os.path.join(sample_dir, "node_stats_m0035.json")) as f:
            ns = json.load(f)
        dev = self.dist.device
        self.d_mean = torch.tensor(ns["disp_mean"], dtype=torch.float32).to(dev)
        self.d_std  = torch.tensor(ns["disp_std"],  dtype=torch.float32).to(dev)
        self.v_mean = torch.tensor(ns["vel_mean"],  dtype=torch.float32).to(dev)
        self.v_std  = torch.tensor(ns["vel_std"],   dtype=torch.float32).to(dev)
        self.stress_mean_Si     = torch.tensor(ns["stress_mean_Si"],     dtype=torch.float32).to(dev)
        self.stress_std_Si      = torch.tensor(ns["stress_std_Si"],      dtype=torch.float32).to(dev)
        self.stress_mean_Solder = torch.tensor(ns["stress_mean_Solder"], dtype=torch.float32).to(dev)
        self.stress_std_Solder  = torch.tensor(ns["stress_std_Solder"],  dtype=torch.float32).to(dev)
        self.stress_mean_UF     = torch.tensor(ns["stress_mean_UF"],     dtype=torch.float32).to(dev)
        self.stress_std_UF      = torch.tensor(ns["stress_std_UF"],      dtype=torch.float32).to(dev)

    # ── constitutive loss ────────────────────────────────────────────────────

    def _constitutive_loss(self, pred, graph, topo_batch):
        """
        Physics regularisation: σ_const(u_pred) should match σ_pred for elastic nodes.

        Uses precomputed B-matrices (FEM, exact for NLGEOM=NO linear elastic).
        Comparison is in normalized stress space (same scale as w_stress loss).
        Material matching: Si elements contribute only to Si corner nodes,
        UF elements only to UF corner nodes (matching verify_constitutive_v2.py).
        """
        dev = pred.device

        # Denormalize predicted velocity → physical displacement at next frame
        vel_phys = pred[:, :3] * self.v_std + self.v_mean   # (N, 3)
        u_next   = graph.world_pos.to(dev) + vel_phys - graph.mesh_pos.to(dev)   # (N, 3)

        ec    = topo_batch["elem_conn"].to(dev)    # (E_tot, 8)
        em    = topo_batch["elem_mat"].to(dev)     # (E_tot,)
        B     = topo_batch["elem_B"].to(dev)       # (E_tot, 3, 8)
        vol   = topo_batch["elem_vol"].to(dev)     # (E_tot,)
        E_e   = topo_batch["elem_E"].to(dev)       # (E_tot,)
        nu_e  = topo_batch["elem_nu"].to(dev)      # (E_tot,)
        cte_e = topo_batch["elem_CTE"].to(dev)     # (E_tot,)
        ebatch= topo_batch["elem_batch"].to(dev)   # (E_tot,)

        # Elastic elements only (skip Solder mat=2)
        el = em != 2
        if not el.any():
            return torch.zeros((), device=dev)

        ec_el    = ec[el]       # (E_el, 8)
        B_el     = B[el]        # (E_el, 3, 8)
        vol_el   = vol[el]      # (E_el,)
        E_el     = E_e[el]      # (E_el,)
        nu_el    = nu_e[el]     # (E_el,)
        cte_el   = cte_e[el]    # (E_el,)
        em_el    = em[el]       # (E_el,)
        eb_el    = ebatch[el]   # (E_el,)

        # delta_T at next frame per element  (graph.delta_T_next is (B_batch,) after PyG batch)
        dT_el = graph.delta_T_next.to(dev)[eb_el]   # (E_el,)

        # Gather node displacements  (E_el, 8, 3)
        u_elem = u_next[ec_el]   # (E_el, 8, 3)

        # Strain gradient: (E_el, 3, 3)  [alpha, beta] = dN_alpha @ u_beta
        grad = torch.bmm(B_el, u_elem)   # (E_el, 3, 8) @ (E_el, 8, 3)

        # Mechanical strains (thermal correction on diagonal)
        eps_th = cte_el * dT_el          # (E_el,)
        e11 = grad[:, 0, 0] - eps_th
        e22 = grad[:, 1, 1] - eps_th
        e33 = grad[:, 2, 2] - eps_th
        g12 = grad[:, 1, 0] + grad[:, 0, 1]   # engineering shear
        g13 = grad[:, 2, 0] + grad[:, 0, 2]
        g23 = grad[:, 2, 1] + grad[:, 1, 2]

        # Lame constants
        lam = E_el * nu_el / ((1 + nu_el) * (1 - 2 * nu_el))
        mu  = E_el / (2 * (1 + nu_el))
        tr  = e11 + e22 + e33

        sig_el = torch.stack([
            lam * tr + 2 * mu * e11,
            lam * tr + 2 * mu * e22,
            lam * tr + 2 * mu * e33,
            mu * g12,
            mu * g13,
            mu * g23,
        ], dim=1)   # (E_el, 6) in MPa

        # Material-matched scatter to nodes:
        # only contribute to corner nodes whose node_mat == element mat
        node_mat_b  = graph.node_mat.to(dev)           # (N,)
        node_mat_el = node_mat_b[ec_el]                 # (E_el, 8)
        em_exp      = em_el.unsqueeze(1).expand_as(node_mat_el)
        matched     = (node_mat_el == em_exp)           # (E_el, 8)

        # Flatten to (E_el*8,)
        ec_flat      = ec_el.reshape(-1)               # (E_el*8,)
        vol_rep      = vol_el.unsqueeze(1).expand(-1, 8).reshape(-1)        # (E_el*8,)
        matched_flat = matched.reshape(-1)              # (E_el*8,) bool
        vw_rep       = (vol_el[:, None] * sig_el).unsqueeze(1)\
                       .expand(-1, 8, -1).reshape(-1, 6)                    # (E_el*8, 6)

        # Zero out mismatched (different-material) corner nodes
        vol_rep  = vol_rep  * matched_flat.float()
        vw_rep   = vw_rep   * matched_flat.float().unsqueeze(-1)

        N = u_next.shape[0]
        sig_num  = torch.zeros(N, 6, device=dev)
        vol_sum  = torch.zeros(N,    device=dev)

        sig_num.scatter_add_(0, ec_flat.unsqueeze(-1).expand(-1, 6), vw_rep)
        vol_sum.scatter_add_(0, ec_flat, vol_rep)

        has_weight = vol_sum > 0
        if not has_weight.any():
            return torch.zeros((), device=dev)

        # Volume-weighted average constitutive stress (MPa)
        sig_const = torch.zeros(N, 6, device=dev)
        sig_const[has_weight] = sig_num[has_weight] / vol_sum[has_weight, None]

        # Normalize constitutive stress using per-material stats
        sig_const_n = torch.zeros(N, 6, device=dev)
        si_mask = (node_mat_b == 0) | (node_mat_b == 1)
        uf_mask = node_mat_b == 3
        if si_mask.any():
            sig_const_n[si_mask] = (sig_const[si_mask] - self.stress_mean_Si) / self.stress_std_Si
        if uf_mask.any():
            sig_const_n[uf_mask] = (sig_const[uf_mask] - self.stress_mean_UF) / self.stress_std_UF

        # Compare with predicted stress (already normalized)
        cmp_mask = has_weight & (si_mask | uf_mask)
        return self.criterion(sig_const_n[cmp_mask], pred[cmp_mask, 3:9])

    # ── validation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        sums = {"vel": 0., "stress": 0., "peeq": 0., "const": 0.}
        n = 0
        for batch in self.val_dataloader:
            batched_graph, batched_mesh_ef, batched_world_ef, topo_batch = batch
            batched_graph    = batched_graph.to(self.dist.device)
            batched_mesh_ef  = batched_mesh_ef.to(self.dist.device)
            batched_world_ef = batched_world_ef.to(self.dist.device)
            _, lv, ls, lp, lc = self.forward(
                batched_graph, batched_mesh_ef, batched_world_ef, topo_batch)
            sums["vel"]    += lv.item()
            sums["stress"] += ls.item()
            sums["peeq"]   += lp.item()
            sums["const"]  += lc.item()
            n += 1
        self.model.train()
        return {k: v / max(n, 1) for k, v in sums.items()}

    # ── forward / train_step ────────────────────────────────────────────────

    def forward(self, graph, mesh_ef, world_ef, topo_batch):
        node_x = graph.x.to(self.dist.device).float()

        if self.model.training and self.input_noise_std > 0:
            node_x = node_x.clone()
            node_x[:, :3] = (node_x[:, :3]
                             + torch.randn_like(node_x[:, :3]) * self.input_noise_std)

        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(node_x, mesh_ef, world_ef, graph)

        pred = pred.float()
        y    = graph.y.to(self.dist.device).float()

        dev = self.dist.device
        loss_vel    = self.criterion(pred[:, :3], y[:, :3])
        loss_stress = torch.zeros((), device=dev)
        loss_peeq   = torch.zeros((), device=dev)
        loss_const  = torch.zeros((), device=dev)

        if self.w_stress > 0 and self._train_step % self._stress_every == 0:
            loss_stress = self.criterion(pred[:, 3:9], y[:, 3:9])

            if self.w_peeq > 0:
                solder_mask = (graph.node_mat == 2).to(dev)
                if solder_mask.any():
                    loss_peeq = self.criterion(
                        pred[solder_mask, 9:10], y[solder_mask, 9:10]
                    )

        if self.w_const > 0:
            loss_const = self._constitutive_loss(pred, graph, topo_batch)

        loss = (self.w_vel    * loss_vel
              + self.w_stress * loss_stress
              + self.w_peeq   * loss_peeq
              + self.w_const  * loss_const)
        return loss, loss_vel, loss_stress, loss_peeq, loss_const

    def train_step(self, graph, mesh_ef, world_ef, topo_batch, epoch):
        mesh_ef  = mesh_ef.to(self.dist.device)
        world_ef = world_ef.to(self.dist.device)
        self.optimizer.zero_grad()
        self._train_step += 1

        loss, loss_vel, loss_stress, loss_peeq, loss_const = self.forward(
            graph, mesh_ef, world_ef, topo_batch)

        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
        self.scheduler.step()
        return loss, loss_vel, loss_stress, loss_peeq, loss_const


# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config_m0035_constloss")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = M0035Trainer(cfg, rank_zero_logger)
    rank_zero_logger.info(
        "Training started... [M0035 Constitutive Loss: w_const={:.3f}]".format(
            trainer._cfg_w_const)
    )

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.sampler.set_epoch(epoch)
        start = time.time()
        step_count = epoch * len(trainer.dataloader)

        progress_bar = tqdm(
            trainer.dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        stress_active_this_epoch = (trainer.w_stress > 0)   # must be True for the FULL epoch
        vel_loss_sum, stress_loss_sum, n_steps = 0.0, 0.0, 0

        for batch in progress_bar:
            batched_graph, batched_mesh_ef, batched_world_ef, topo_batch = batch
            batched_graph = batched_graph.to(dist.device)

            loss, loss_vel, loss_stress, loss_peeq, loss_const = trainer.train_step(
                batched_graph, batched_mesh_ef, batched_world_ef, topo_batch, epoch
            )
            del batched_graph, batched_mesh_ef, batched_world_ef, topo_batch

            vel_loss_sum    += loss_vel.item()
            stress_loss_sum += loss_stress.item()
            n_steps         += 1

            if dist.rank == 0 and step_count % 10 == 0:
                trainer.writer.add_scalar("loss/step",        loss.item(),        step_count)
                trainer.writer.add_scalar("loss_vel/step",    loss_vel.item(),    step_count)
                trainer.writer.add_scalar("loss_stress/step", loss_stress.item(), step_count)
                trainer.writer.add_scalar("loss_peeq/step",   loss_peeq.item(),   step_count)
                trainer.writer.add_scalar("loss_const/step",  loss_const.item(),  step_count)

            step_count += 1
            progress_bar.set_postfix(
                loss=f"{loss.item():.3e}",
                vel=f"{loss_vel.item():.3e}",
                s=f"{loss_stress.item():.3e}",
                peeq=f"{loss_peeq.item():.3e}",
                c=f"{loss_const.item():.3e}",
                wc=f"{trainer.w_const:.2f}",
            )

        epoch_avg_vel    = vel_loss_sum    / max(n_steps, 1)
        epoch_avg_stress = stress_loss_sum / max(n_steps, 1)

        # Two-stage vel-based warmup for stress + peeq
        if trainer._stage1_threshold is not None and trainer._warmup_stage == 0:
            if epoch_avg_vel < trainer._stage1_threshold:
                trainer.w_stress = 0.5 * cfg.w_stress
                trainer.w_peeq   = 0.5 * trainer._cfg_w_peeq
                trainer._warmup_stage = 1
                rank_zero_logger.info(
                    f"[warmup] epoch {epoch+1}: vel_loss={epoch_avg_vel:.3e} "
                    f"→ w_stress={trainer.w_stress:.2f}, w_peeq={trainer.w_peeq:.2f}"
                )
        if trainer._stage2_threshold is not None and trainer._warmup_stage == 1:
            if epoch_avg_vel < trainer._stage2_threshold:
                trainer.w_stress = cfg.w_stress
                trainer.w_peeq   = trainer._cfg_w_peeq
                trainer._warmup_stage = 2
                rank_zero_logger.info(
                    f"[warmup] epoch {epoch+1}: vel_loss={epoch_avg_vel:.3e} "
                    f"→ w_stress={cfg.w_stress}, w_peeq={trainer._cfg_w_peeq}"
                )

        # Independent constitutive-loss activation: triggers when stress has converged.
        # Only check when stress was active for the FULL epoch (not the transition epoch
        # where w_stress was 0 all along, giving a spuriously low epoch_avg_stress).
        if (trainer._const_stress_threshold is not None
                and trainer._const_stage == 0
                and stress_active_this_epoch):
            if epoch_avg_stress < trainer._const_stress_threshold:
                trainer.w_const = trainer._cfg_w_const
                trainer._const_stage = 1
                rank_zero_logger.info(
                    f"[const] epoch {epoch+1}: stress_loss={epoch_avg_stress:.3e} "
                    f"→ w_const={trainer._cfg_w_const}"
                )

        torch.cuda.empty_cache()
        rank_zero_logger.info(
            f"epoch: {epoch+1}, loss: {loss:10.3e}, vel: {loss_vel:10.3e}, "
            f"stress: {loss_stress:10.3e}, peeq: {loss_peeq:10.3e}, "
            f"const: {loss_const:10.3e}, "
            f"w_s={trainer.w_stress:.2f}, w_c={trainer.w_const:.2f}, "
            f"time: {(time.time() - start):.1f}s"
        )
        if dist.rank == 0:
            trainer.writer.add_scalar("loss",        loss.detach().cpu().item(),        epoch)
            trainer.writer.add_scalar("loss_vel",    loss_vel.detach().cpu().item(),    epoch)
            trainer.writer.add_scalar("loss_stress", loss_stress.detach().cpu().item(), epoch)
            trainer.writer.add_scalar("loss_peeq",   loss_peeq.detach().cpu().item(),   epoch)
            trainer.writer.add_scalar("loss_const",  loss_const.detach().cpu().item(),  epoch)
            trainer.writer.add_scalar("learning_rate",
                                      trainer.optimizer.param_groups[0]["lr"], epoch)

        if (trainer.val_dataloader is not None
                and (epoch + 1) % trainer._val_every == 0):
            val = trainer.validate()
            rank_zero_logger.info(
                f"  [val] vel: {val['vel']:.3e}  stress: {val['stress']:.3e}  "
                f"peeq: {val['peeq']:.3e}  const: {val['const']:.3e}"
            )
            if dist.rank == 0:
                trainer.writer.add_scalar("val/loss_vel",    val["vel"],    epoch)
                trainer.writer.add_scalar("val/loss_stress", val["stress"], epoch)
                trainer.writer.add_scalar("val/loss_peeq",   val["peeq"],   epoch)
                trainer.writer.add_scalar("val/loss_const",  val["const"],  epoch)

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger.info(f"Saved checkpoint (epoch {epoch + 1})")
        torch.cuda.empty_cache()

    rank_zero_logger.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()