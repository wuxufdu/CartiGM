"""Leave-batch-out evaluation of the bundled v1 cs_classifier on EBR.

Trains three classifiers, each holding out one of {ear, nose, rib}, then
evaluates on that held-out batch only. This stresses cross-batch
generalization, which the within-cluster cell-level holdout can mask.

Outputs:
  outputs/training_local/lbo_v1_metrics.tsv
  outputs/training_local/lbo_v1_per_celltype.tsv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("F:/cartifm/outputs/training_subset")
OUT = Path("F:/cartifm/outputs/training_local")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 0
EPOCHS = 60
BATCH_SIZE = 512
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.4
HIDDEN1 = 384
HIDDEN2 = 192
VAL_FRAC = 0.2
EBR_TRAIN_FRAC = 1.0
EBR_LABEL_WEIGHT = 3.0


class MLP(nn.Module):
    def __init__(self, n_in: int, n_classes: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(n_in)
        self.input_drop = nn.Dropout(0.1)
        self.net = nn.Sequential(
            nn.Linear(n_in, HIDDEN1), nn.LayerNorm(HIDDEN1), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2), nn.LayerNorm(HIDDEN2), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_drop(x)
        return self.net(x)


def stratified_split(samples: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(samples.tolist())))
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_set = set(uniq[:n_val].tolist())
    train_idx = np.where(~np.isin(samples, list(val_set)))[0]
    val_idx = np.where(np.isin(samples, list(val_set)))[0]
    return train_idx, val_idx


def train_one(
    X_acc: np.ndarray,
    y_acc: np.ndarray,
    samples: np.ndarray,
    X_ebr_train: np.ndarray,
    y_ebr_train: np.ndarray,
    n_classes: int,
    device: torch.device,
):
    train_idx, _ = stratified_split(samples, VAL_FRAC, SEED)
    X_train_np = np.concatenate([X_acc[train_idx], X_ebr_train], axis=0).astype(np.float32)
    y_train_np = np.concatenate([y_acc[train_idx], y_ebr_train], axis=0).astype(np.int64)
    sw = np.concatenate([
        np.ones(int(train_idx.size), dtype=np.float32),
        np.full(int(len(y_ebr_train)), EBR_LABEL_WEIGHT, dtype=np.float32),
    ], axis=0)
    Xt = torch.from_numpy(X_train_np).to(device)
    yt = torch.from_numpy(y_train_np).to(device)
    swt = torch.from_numpy(sw).to(device)

    model = MLP(X_acc.shape[1], n_classes).to(device)
    counts = np.bincount(y_train_np, minlength=n_classes)
    inv = 1.0 / np.maximum(counts, 1)
    cw = torch.from_numpy((inv / inv.sum() * n_classes).astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xt.shape[0]
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xt[idx]; yb = yt[idx]; sb = swt[idx]
            opt.zero_grad()
            logits = model(xb)
            ce = F.cross_entropy(logits, yb, weight=cw, reduction="none")
            loss = (ce * sb).sum() / sb.sum().clamp(min=1.0)
            loss.backward(); opt.step()
        sched.step()
    return model


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    acc = np.load(ROOT / "acc_train.npz", allow_pickle=True)
    X_acc = acc["X"].astype(np.float32)
    samples = acc["samples"].astype(str)
    labels = acc["labels"].astype(str)
    classes = sorted(set(labels.tolist()))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_acc = np.array([cls2idx[c] for c in labels], dtype=np.int64)

    ebr = np.load(ROOT / "ebr_eval.npz", allow_pickle=True)
    X_ebr = ebr["X"].astype(np.float32)
    ebr_celltype = ebr["celltype"].astype(str)
    ebr_batch = ebr["batch"].astype(str)
    y_ebr = np.array(
        [cls2idx[c] if c in cls2idx else -1 for c in ebr_celltype], dtype=np.int64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    rows_overall = []
    rows_celltype = []
    for held in ["ear", "nose", "rib"]:
        train_mask = (ebr_batch != held) & (y_ebr >= 0)
        test_mask = (ebr_batch == held) & (y_ebr >= 0)
        n_test = int(test_mask.sum())
        if n_test == 0:
            continue
        print(f"== holdout batch={held}: train_ebr={int(train_mask.sum())} test_ebr={n_test}")
        t0 = time.time()
        model = train_one(
            X_acc, y_acc, samples,
            X_ebr[train_mask], y_ebr[train_mask],
            len(classes), device,
        )
        Xt_test = torch.from_numpy(X_ebr[test_mask]).to(device)
        with torch.no_grad():
            pred_idx = model(Xt_test).argmax(dim=1).cpu().numpy()
        pred = np.array([classes[i] for i in pred_idx])
        truth = ebr_celltype[test_mask]
        correct = (pred == truth).astype(np.int64)
        rows_overall.append({
            "held_out_batch": held,
            "n_test_cells": n_test,
            "accuracy_cell": round(float(correct.mean()), 4),
            "n_correct_cell": int(correct.sum()),
            "elapsed_s": round(time.time() - t0, 1),
        })
        ct_df = pd.DataFrame({"celltype": truth, "correct": correct})
        agg = ct_df.groupby("celltype").agg(n=("correct", "size"), correct=("correct", "sum"))
        agg["recall"] = (agg["correct"] / agg["n"]).round(4)
        agg = agg.reset_index()
        agg["held_out_batch"] = held
        rows_celltype.append(agg)
        print(f"   acc_cell={correct.mean():.4f} elapsed={time.time()-t0:.1f}s")

    pd.DataFrame(rows_overall).to_csv(OUT / "lbo_v1_metrics.tsv", sep="\t", index=False)
    if rows_celltype:
        pd.concat(rows_celltype, ignore_index=True).to_csv(
            OUT / "lbo_v1_per_celltype.tsv", sep="\t", index=False
        )
    print("wrote", OUT / "lbo_v1_metrics.tsv")


if __name__ == "__main__":
    main()
