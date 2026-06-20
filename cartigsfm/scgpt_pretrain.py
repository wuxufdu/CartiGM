"""P16: scGPT-style masked-expression pretraining on acc.h5ad."""
from __future__ import annotations
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .assets import prefer_device


@dataclass
class ScGPTConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_ff: int = 512
    dropout: float = 0.1
    mask_prob: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    n_cells: int = 50000
    n_steps: int = 400
    log_every: int = 20
    n_hvg: int = 2000
    hvg_min_mean: float = 0.01
    hvg_max_mean: float = 4.0
    seed: int = 0


def select_hvgs(X_csr, gene_names, cfg):
    n = X_csr.shape[0]
    means = np.asarray(X_csr.mean(axis=0)).ravel()
    sq = X_csr.multiply(X_csr).mean(axis=0)
    sq = np.asarray(sq).ravel()
    var = np.maximum(sq - means ** 2, 1e-8)
    disp = var / np.maximum(means, 1e-8)
    keep = (means >= cfg.hvg_min_mean) & (means <= cfg.hvg_max_mean)
    cand = np.where(keep)[0]
    if cand.size <= cfg.n_hvg:
        order = np.argsort(-disp)[: cfg.n_hvg]
    else:
        order = cand[np.argsort(-disp[cand])[: cfg.n_hvg]]
    return [gene_names[i] for i in order]


def load_acc_subsample(h5ad_path, cfg):
    import scanpy as sc
    import scipy.sparse as sp
    adata = sc.read_h5ad(h5ad_path, backed="r")
    print("[scgpt_pretrain] opened", h5ad_path, adata.shape, flush=True)
    gene_names = [str(g) for g in adata.var_names]
    rng = np.random.default_rng(cfg.seed)
    if adata.n_obs > cfg.n_cells:
        keep = np.sort(rng.choice(adata.n_obs, size=cfg.n_cells, replace=False))
        adata_sub = adata[keep].to_memory()
    else:
        adata_sub = adata.to_memory()
    print("[scgpt_pretrain] subsampled to", adata_sub.shape, flush=True)
    hvg = select_hvgs(adata_sub.X, gene_names, cfg)
    print("[scgpt_pretrain] selected", len(hvg), "HVGs", flush=True)
    var_list = list(adata_sub.var_names)
    hvg_idx = [var_list.index(g) for g in hvg]
    X_sub = adata_sub.X[:, hvg_idx]
    if sp.issparse(X_sub) and not sp.isspmatrix_csc(X_sub):
        X_sub = X_sub.tocsc()
    X_dense = np.asarray(X_sub.toarray(), dtype=np.float32) if sp.issparse(X_sub) else np.asarray(X_sub, dtype=np.float32)
    print("[scgpt_pretrain] dense shape", X_dense.shape, "size_mb", round(X_dense.nbytes / 1e6, 1), flush=True)
    return X_dense, hvg


def build_model(cfg, n_genes, device):
    import torch
    import torch.nn as nn

    class GeneEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gene_embed = nn.Embedding(n_genes, cfg.d_model)
            self.value_proj = nn.Linear(1, cfg.d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_genes, cfg.d_model))
            self.mask_embed = nn.Parameter(torch.zeros(cfg.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.dim_ff,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
            self.head = nn.Linear(cfg.d_model, 1)

        def forward(self, x, mask):
            gene_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            g_emb = self.gene_embed(gene_ids)
            v_emb = self.value_proj(x.unsqueeze(-1))
            pos = self.pos_embed[:, : x.shape[1], :]
            h = g_emb + v_emb + pos
            m = self.mask_embed.view(1, 1, -1).expand_as(h)
            h = torch.where(mask.unsqueeze(-1) > 0, m, h)
            h = self.encoder(h)
            return self.head(h).squeeze(-1)

    return GeneEncoder().to(device)


def cosine_warmup_lr(step, cfg):
    warmup = max(1, cfg.n_steps // 10)
    if step < warmup:
        return cfg.lr * step / warmup
    progress = (step - warmup) / max(1, cfg.n_steps - warmup)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def pretrain(h5ad_path, outdir, cfg=None):
    import torch
    import torch.nn.functional as F

    cfg = cfg or ScGPTConfig()
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    device = prefer_device(None)
    print("[scgpt_pretrain] device =", device, flush=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X_dense, hvg = load_acc_subsample(h5ad_path, cfg)
    n_genes = X_dense.shape[1]
    X_t = torch.from_numpy(X_dense).to(device).float()
    X_t = torch.clamp(X_t, min=0.0, max=20.0)
    n_cells = X_t.shape[0]
    model = build_model(cfg, n_genes, device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[scgpt_pretrain] params =", round(n_params / 1e6, 2), "M", flush=True)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    losses = []
    t0 = time.time()
    for step in range(1, cfg.n_steps + 1):
        idx = torch.randint(0, n_cells, (cfg.batch_size,), device=device)
        x = X_t[idx]
        mask = (torch.rand_like(x) < cfg.mask_prob).float()
        target = x * mask
        pred = model(x, mask)
        loss = F.mse_loss(pred * mask, target, reduction="sum") / mask.sum().clamp(min=1.0)
        lr = cosine_warmup_lr(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach().cpu()))
        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            avg = sum(losses[-cfg.log_every:]) / max(1, len(losses[-cfg.log_every:]))
            print("[scgpt_pretrain] step", step, "/", cfg.n_steps, "loss=", round(avg, 4), "lr=", f"{lr:.2e}", "elapsed=", round(elapsed, 1), "s", flush=True)
        if step % max(1, cfg.n_steps // 4) == 0 and step < cfg.n_steps:
            interim = {
                "config": asdict(cfg),
                "n_genes": int(n_genes),
                "hvg": hvg,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "interim_step": int(step),
                "interim_loss": float(np.mean(losses[-20:])),
                "device": str(device),
            }
            torch.save(interim, outdir_p / "scgpt_small.interim.pt")

    final_loss = float(np.mean(losses[-20:]))
    ckpt = {
        "config": asdict(cfg),
        "n_genes": int(n_genes),
        "hvg": hvg,
        "state_dict": model.state_dict(),
        "final_train_loss": final_loss,
        "device": str(device),
    }
    torch.save(ckpt, outdir_p / "scgpt_small.pt")
    (outdir_p / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    pd.DataFrame({"gene": hvg}).to_csv(outdir_p / "gene_order.tsv", sep="\t", index=False)
    pd.DataFrame({"step": list(range(1, len(losses) + 1)), "loss": losses}).to_csv(outdir_p / "train_loss.tsv", sep="\t", index=False)
    summary = {
        "n_genes": int(n_genes),
        "n_cells_used": int(n_cells),
        "n_params": int(n_params),
        "final_train_loss": final_loss,
        "device": str(device),
        "n_steps": int(cfg.n_steps),
        "batch_size": int(cfg.batch_size),
        "elapsed_seconds": round(time.time() - t0, 1),
        "hvg_first20": hvg[:20],
    }
    (outdir_p / "pretrain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[scgpt_pretrain] done. final_loss=", round(final_loss, 4), "elapsed=", round(time.time() - t0, 1), "s", flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="scGPT-style MLM pretraining on acc.h5ad")
    p.add_argument("--h5ad", default=r"F:\cartifm\acc.h5ad")
    p.add_argument("--outdir", default=r"F:\cartifm\outputs\scgpt_pretrain")
    p.add_argument("--n-cells", type=int, default=50000)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    cfg = ScGPTConfig(
        n_cells=args.n_cells,
        n_hvg=args.n_hvg,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
    )
    summary = pretrain(args.h5ad, args.outdir, cfg)
    print(json.dumps(summary, indent=2))
