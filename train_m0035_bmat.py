"""
Train MeshGraphNet on M0035: B-matrix stress for elastic materials.

For Si (mat=0,1) and UF (mat=3):
  - Model predicts vel only (pred[:, 3:9] ignored at training and inference)
  - Stress computed from vel_pred via B-matrix (linear elastic, exact)
  - Loss: MSE(sigma_bmat(vel_pred), sigma_GT)  [normalized per-material]
  - Gradient flows: loss -> sigma_bmat -> u_pred -> vel_pred -> model params

For Solder (mat=2, elastoplastic):
  - Model predicts vel + stress + PEEQ directly as before
  - Loss: MSE(pred_stress, sigma_GT) + MSE(pred_peeq, peeq_GT)

This eliminates gradient conflict: Si/UF stress is uniquely determined by vel,
so supervised stress loss and displacement consistency are the same constraint.
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
# Collate (needs topo for B-matrix)
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    graphs    = [seq[0]["graph"]               for seq in batch]
    mesh_efs  = [seq[0]["mesh_edge_features"]  for seq in batch]
    world_efs = [seq[0]["world_edge_features"] for seq in batch]
    topos     = [seq[0]["topo"]                for seq in batch]

    batched_graph = Batch.from_data_list(graphs)

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
        "elem_conn":  torch.cat(elem_conns,     dim=0),
        "elem_mat":   torch.cat(elem_mats,      dim=0),
        "elem_B":     torch.cat(elem_Bs,        dim=0),
        "elem_vol":   torch.cat(elem_vols,      dim=0),
        "elem_E":     torch.cat(elem_Es,        dim=0),
        "elem_nu":    torch.cat(elem_nus,       dim=0),
        "elem_CTE":   torch.cat(elem_CTEs,      dim=0),
        "elem_batch": torch.cat(elem_batch_ids, dim=0),
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
                    item["topo"] = topo
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
        self.w_vel       = cfg.w_vel
        self.w_stress_el = 0.0   # B-matrix elastic path; activated by warmup
        self.w_stress_so = 0.0   # Solder direct path;   activated by warmup
        self.w_peeq      = 0.0
        self._train_step      = 0
        self._stress_every    = getattr(cfg, "stress_every", 1)
        self._stage1_threshold = getattr(cfg, "w_stress_stage1_threshold", None)
        self._stage2_threshold = getattr(cfg, "w_stress_stage2_threshold", None)
        self._warmup_stage    = 0
        self._cfg_w_stress_el = getattr(cfg, "w_stress_el", 0.001)
        self._cfg_w_stress_so = getattr(cfg, "w_stress_so", 1.0)
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
                self.w_stress_el = 0.0
                self.w_stress_so = 0.0
                self.w_peeq      = 0.0
            else:
                self._warmup_stage = 2
                self.w_stress_el = self._cfg_w_stress_el
                self.w_stress_so = self._cfg_w_stress_so
                self.w_peeq      = self._cfg_w_peeq

        if self.dist.rank == 0:
            self.writer = SummaryWriter(
                log_dir=to_absolute_path(cfg.tensorboard_log_dir)
            )

        self.input_noise_std = getattr(cfg, "input_noise_std", 0.0)

    # ── stats ────────────────────────────────────────────────────────────────

    def _load_stats(self, cfg):
        sample_dir = to_absolute_path(cfg.preprocess_output_dir)
        with open(os.path.join(sample_dir, "node_stats_m0035.json")) as f:
            ns = json.load(f)
        dev = self.dist.device
        self.v_mean = torch.tensor(ns["vel_mean"],  dtype=torch.float32).to(dev)
        self.v_std  = torch.tensor(ns["vel_std"],   dtype=torch.float32).to(dev)
        self.stress_mean_Si     = torch.tensor(ns["stress_mean_Si"],     dtype=torch.float32).to(dev)
        self.stress_std_Si      = torch.tensor(ns["stress_std_Si"],      dtype=torch.float32).to(dev)
        self.stress_mean_Solder = torch.tensor(ns["stress_mean_Solder"], dtype=torch.float32).to(dev)
        self.stress_std_Solder  = torch.tensor(ns["stress_std_Solder"],  dtype=torch.float32).to(dev)
        self.stress_mean_UF     = torch.tensor(ns["stress_mean_UF"],     dtype=torch.float32).to(dev)
        self.stress_std_UF      = torch.tensor(ns["stress_std_UF"],      dtype=torch.float32).to(dev)

    # ── B-matrix stress for elastic nodes ────────────────────────────────────

    def _bmat_stress(self, pred, graph, topo_batch):
        """
        Compute B-matrix stress for Si/UF nodes using vel_pred.

        Returns:
          sig_bmat_n : (N, 6) normalized constitutive stress (per-material),
                       zero for Solder / nodes with no elastic element contribution
          elastic_mask : (N,) bool, True for nodes with valid B-matrix stress
        """
        dev = pred.device

        vel_phys = pred[:, :3] * self.v_std + self.v_mean   # denorm vel → (N, 3)
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
            N = u_next.shape[0]
            return (torch.zeros(N, 6, device=dev),
                    torch.zeros(N, dtype=torch.bool, device=dev))

        ec_el  = ec[el];   B_el  = B[el];   vol_el = vol[el]
        E_el   = E_e[el];  nu_el = nu_e[el]; cte_el = cte_e[el]
        em_el  = em[el];   eb_el = ebatch[el]

        dT_el  = graph.delta_T_next.to(dev)[eb_el]   # (E_el,)

        # Strain: B_el (E_el,3,8) @ u_elem (E_el,8,3) → (E_el,3,3)
        u_elem = u_next[ec_el]                     # (E_el, 8, 3)
        grad   = torch.bmm(B_el, u_elem)           # (E_el, 3, 3)

        eps_th = cte_el * dT_el                    # (E_el,)
        e11 = grad[:, 0, 0] - eps_th;  e22 = grad[:, 1, 1] - eps_th
        e33 = grad[:, 2, 2] - eps_th
        g12 = grad[:, 1, 0] + grad[:, 0, 1]
        g13 = grad[:, 2, 0] + grad[:, 0, 2]
        g23 = grad[:, 2, 1] + grad[:, 1, 2]

        lam = E_el * nu_el / ((1 + nu_el) * (1 - 2 * nu_el))
        mu  = E_el / (2 * (1 + nu_el))
        tr  = e11 + e22 + e33

        sig_el = torch.stack([
            lam * tr + 2 * mu * e11,
            lam * tr + 2 * mu * e22,
            lam * tr + 2 * mu * e33,
            mu * g12, mu * g13, mu * g23,
        ], dim=1)   # (E_el, 6) MPa

        # Material-matched scatter to nodes (Si element → Si nodes, UF → UF nodes)
        node_mat_b  = graph.node_mat.to(dev)             # (N,)
        node_mat_el = node_mat_b[ec_el]                  # (E_el, 8)
        em_exp      = em_el.unsqueeze(1).expand_as(node_mat_el)
        matched     = (node_mat_el == em_exp).float()    # (E_el, 8)

        ec_flat  = ec_el.reshape(-1)                     # (E_el*8,)
        vol_rep  = (vol_el[:, None] * matched).reshape(-1)           # (E_el*8,)
        vw_rep   = (vol_el[:, None, None] * sig_el[:, None, :]
                    * matched[:, :, None]).reshape(-1, 6)            # (E_el*8, 6)

        N = u_next.shape[0]
        sig_num = torch.zeros(N, 6, device=dev)
        vol_sum = torch.zeros(N,    device=dev)
        sig_num.scatter_add_(0, ec_flat.unsqueeze(-1).expand(-1, 6), vw_rep)
        vol_sum.scatter_add_(0, ec_flat, vol_rep)

        has_weight = vol_sum > 0
        sig_bmat   = torch.zeros(N, 6, device=dev)
        sig_bmat[has_weight] = sig_num[has_weight] / vol_sum[has_weight, None]

        # Normalize per material
        sig_bmat_n = torch.zeros(N, 6, device=dev)
        si_mask = (node_mat_b == 0) | (node_mat_b == 1)
        uf_mask = node_mat_b == 3
        if si_mask.any():
            sig_bmat_n[si_mask] = (sig_bmat[si_mask] - self.stress_mean_Si) / self.stress_std_Si
        if uf_mask.any():
            sig_bmat_n[uf_mask] = (sig_bmat[uf_mask] - self.stress_mean_UF) / self.stress_std_UF

        # Si excluded: E_Si=130 GPa causes ~26x gradient amplification
        # and rollout Si stress is always wrong regardless of training sel.
        # UF only: E_UF ~5-15 GPa, amplification ~2-5x, rollout R²~0.73.
        elastic_mask = has_weight & uf_mask
        sig_bmat_n = torch.nan_to_num(sig_bmat_n, nan=0.0, posinf=0.0, neginf=0.0)
        return sig_bmat_n, elastic_mask

    # ── forward / train_step ─────────────────────────────────────────────────

    def forward(self, graph, mesh_ef, world_ef, topo_batch):
        node_x = graph.x.to(self.dist.device).float()

        if self.model.training and self.input_noise_std > 0:
            node_x = node_x.clone()
            node_x[:, :3] = (node_x[:, :3]
                             + torch.randn_like(node_x[:, :3]) * self.input_noise_std)

        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(node_x, mesh_ef, world_ef, graph)

        pred = pred.float()
        y    = graph.y.to(self.dist.device).float()   # (N, 10)
        dev  = self.dist.device

        # Velocity loss: all nodes
        loss_vel = self.criterion(pred[:, :3], y[:, :3])

        loss_stress_el = torch.zeros((), device=dev)
        loss_stress_so = torch.zeros((), device=dev)
        loss_peeq      = torch.zeros((), device=dev)

        if (self.w_stress_el > 0 or self.w_stress_so > 0) and self._train_step % self._stress_every == 0:
            # Si/UF: B-matrix stress vs GT
            sig_bmat_n, elastic_mask = self._bmat_stress(pred, graph, topo_batch)
            if elastic_mask.any():
                loss_stress_el = self.criterion(
                    sig_bmat_n[elastic_mask], y[elastic_mask, 3:9]
                )

            # Solder: direct prediction vs GT
            solder_mask = (graph.node_mat == 2).to(dev)
            if solder_mask.any():
                loss_stress_so = self.criterion(
                    pred[solder_mask, 3:9], y[solder_mask, 3:9]
                )
                if self.w_peeq > 0:
                    loss_peeq = self.criterion(
                        pred[solder_mask, 9:10], y[solder_mask, 9:10]
                    )

        loss = (self.w_vel       * loss_vel
              + self.w_stress_el * loss_stress_el
              + self.w_stress_so * loss_stress_so
              + self.w_peeq      * loss_peeq)
        return loss, loss_vel, loss_stress_el, loss_stress_so, loss_peeq

    def train_step(self, graph, mesh_ef, world_ef, topo_batch, epoch):
        mesh_ef  = mesh_ef.to(self.dist.device)
        world_ef = world_ef.to(self.dist.device)
        self.optimizer.zero_grad()
        self._train_step += 1

        loss, loss_vel, loss_sel, loss_sso, loss_peeq = self.forward(
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
        return loss, loss_vel, loss_sel, loss_sso, loss_peeq


# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config_m0035_bmat")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = M0035Trainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Training started... [M0035 B-matrix stress for Si/UF]")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.sampler.set_epoch(epoch)
        start = time.time()
        step_count = epoch * len(trainer.dataloader)

        progress_bar = tqdm(
            trainer.dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        vel_loss_sum, n_steps = 0.0, 0

        for batch in progress_bar:
            batched_graph, batched_mesh_ef, batched_world_ef, topo_batch = batch
            batched_graph = batched_graph.to(dist.device)

            loss, loss_vel, loss_sel, loss_sso, loss_peeq = trainer.train_step(
                batched_graph, batched_mesh_ef, batched_world_ef, topo_batch, epoch
            )
            del batched_graph, batched_mesh_ef, batched_world_ef, topo_batch

            vel_loss_sum += loss_vel.item()
            n_steps      += 1

            if dist.rank == 0 and step_count % 10 == 0:
                trainer.writer.add_scalar("loss/step",           loss.item(),     step_count)
                trainer.writer.add_scalar("loss_vel/step",       loss_vel.item(), step_count)
                trainer.writer.add_scalar("loss_stress_el/step", loss_sel.item(), step_count)
                trainer.writer.add_scalar("loss_stress_so/step", loss_sso.item(), step_count)
                trainer.writer.add_scalar("loss_peeq/step",      loss_peeq.item(),step_count)

            step_count += 1
            progress_bar.set_postfix(
                loss=f"{loss.item():.3e}",
                vel=f"{loss_vel.item():.3e}",
                sel=f"{loss_sel.item():.3e}",
                sso=f"{loss_sso.item():.3e}",
                wsel=f"{trainer.w_stress_el:.4f}",
                wsso=f"{trainer.w_stress_so:.2f}",
            )

        epoch_avg_vel = vel_loss_sum / max(n_steps, 1)

        # Two-stage vel-based warmup for stress + peeq
        if trainer._stage1_threshold is not None and trainer._warmup_stage == 0:
            if epoch_avg_vel < trainer._stage1_threshold:
                trainer.w_stress_el = 0.5 * trainer._cfg_w_stress_el
                trainer.w_stress_so = 0.5 * trainer._cfg_w_stress_so
                trainer.w_peeq      = 0.5 * trainer._cfg_w_peeq
                trainer._warmup_stage = 1
                rank_zero_logger.info(
                    f"[warmup] epoch {epoch+1}: vel_loss={epoch_avg_vel:.3e} "
                    f"→ w_stress_el={trainer.w_stress_el:.4f}, "
                    f"w_stress_so={trainer.w_stress_so:.2f}, w_peeq={trainer.w_peeq:.2f}"
                )
        if trainer._stage2_threshold is not None and trainer._warmup_stage == 1:
            if epoch_avg_vel < trainer._stage2_threshold:
                trainer.w_stress_el = trainer._cfg_w_stress_el
                trainer.w_stress_so = trainer._cfg_w_stress_so
                trainer.w_peeq      = trainer._cfg_w_peeq
                trainer._warmup_stage = 2
                rank_zero_logger.info(
                    f"[warmup] epoch {epoch+1}: vel_loss={epoch_avg_vel:.3e} "
                    f"→ w_stress_el={trainer._cfg_w_stress_el:.4f}, "
                    f"w_stress_so={trainer._cfg_w_stress_so:.2f}, w_peeq={trainer._cfg_w_peeq:.2f}"
                )

        torch.cuda.empty_cache()
        rank_zero_logger.info(
            f"epoch: {epoch+1}, loss: {loss:10.3e}, vel: {loss_vel:10.3e}, "
            f"sel(Bmat): {loss_sel:10.3e}, sso(direct): {loss_sso:10.3e}, "
            f"peeq: {loss_peeq:10.3e}, "
            f"w_sel={trainer.w_stress_el:.4f}, w_sso={trainer.w_stress_so:.2f}, "
            f"time: {(time.time() - start):.1f}s"
        )
        if dist.rank == 0:
            trainer.writer.add_scalar("loss",            loss.detach().cpu().item(),     epoch)
            trainer.writer.add_scalar("loss_vel",        loss_vel.detach().cpu().item(), epoch)
            trainer.writer.add_scalar("loss_stress_el",  loss_sel.detach().cpu().item(), epoch)
            trainer.writer.add_scalar("loss_stress_so",  loss_sso.detach().cpu().item(), epoch)
            trainer.writer.add_scalar("loss_peeq",       loss_peeq.detach().cpu().item(),epoch)
            trainer.writer.add_scalar("learning_rate",
                                      trainer.optimizer.param_groups[0]["lr"], epoch)

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