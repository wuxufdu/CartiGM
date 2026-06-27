"""v2 classifier training with augmentation aimed at cross-batch robustness.

Adds Gaussian input noise + MixUp + slightly larger weight decay to fight
the EBR cross-batch (ear/nose/rib) holdout drop seen on v1. Trains:

  1. A LBO model per held-out batch (ear / nose / rib), reports cell-level
     accuracy on the held-out batch; this is the authoritative
     cross-batch generalization measurement.
  2. A within-cluster cell-level holdout model trained on all batches, for
     direct comparison with the v1 76.6% number. The within-cluster model is
     the candidate replacement for cs_classifier_v1.
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
OUT = Path("F:/cartifm/outputs/training_local/v2")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 0
EPOCHS = 100
BATCH_SIZE = 512
LR = 5e-4
WEIGHT_DECAY = 3e-3
DROPOUT = 0.5
INPUT_DROPOUT = 0.15
HIDDEN1 = 512
HIDDEN2 = 256
VAL_FRAC = 0.2
EBR_TRAIN_FRAC = 0.7
EBR_LABEL_WEIGHT = 3.0
NOISE_STD = 0.15
MIXUP_ALPHA = 0.2


class MLP(nn.Module):
    def __init__(self, n_in: int, n_classes: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(n_in)
        self.input_drop = nn.Dropout(INPUT_DROPOUT)
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


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0); continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s))


def balanced_acc(y_true, y_pred, n_classes):
    recalls = []
    for c in range(n_classes):
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            continue
        recalls.append(float((y_pred[idx] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def train_loop(
    Xt: torch.Tensor,
    yt: torch.Tensor,
    swt: torch.Tensor,
    n_classes: int,
    cw: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
) -> nn.Module:
    model = MLP(Xt.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xt.shape[0]
    g = torch.Generator(device=device)
    g.manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device, generator=g)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xt[idx]; yb = yt[idx]; sb = swt[idx]
            if NOISE_STD > 0:
                xb = xb + NOISE_STD * torch.randn_like(xb)
            if MIXUP_ALPHA > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                lam = max(lam, 1.0 - lam)
                jdx = torch.randperm(xb.size(0), device=device, generator=g)
                xb_mix = lam * xb + (1 - lam) * xb[jdx]
                yb2 = yb[jdx]; sb2 = sb[jdx]
                opt.zero_grad()
                logits = model(xb_mix)
                ce_a = F.cross_entropy(logits, yb, weight=cw, reduction="none")
                ce_b = F.cross_entropy(logits, yb2, weight=cw, reduction="none")
                loss = (
                    lam * (ce_a * sb).sum() / sb.sum().clamp(min=1.0)
                    + (1 - lam) * (ce_b * sb2).sum() / sb2.sum().clamp(min=1.0)
                )
            else:
                opt.zero_grad()
                logits = model(xb)
                ce = F.cross_entropy(logits, yb, weight=cw, reduction="none")
                loss = (ce * sb).sum() / sb.sum().clamp(min=1.0)
            loss.backward(); opt.step()
        sched.step()
    return model


def build_train_tensors(
    X_acc: np.ndarray, y_acc: np.ndarray, train_idx: np.ndarray,
    X_ebr_part: np.ndarray, y_ebr_part: np.ndarray,
    n_classes: int, device: torch.device,
):
    X_train_np = np.concatenate([X_acc[train_idx], X_ebr_part], axis=0).astype(np.float32)
    y_train_np = np.concatenate([y_acc[train_idx], y_ebr_part], axis=0).astype(np.int64)
    sw = np.concatenate([
        np.ones(int(train_idx.size), dtype=np.float32),
        np.full(int(len(y_ebr_part)), EBR_LABEL_WEIGHT, dtype=np.float32),
    ], axis=0)
    counts = np.bincount(y_train_np, minlength=n_classes)
    inv = 1.0 / np.maximum(counts, 1)
    cw = torch.from_numpy((inv / inv.sum() * n_classes).astype(np.float32)).to(device)
    return (
        torch.from_numpy(X_train_np).to(device),
        torch.from_numpy(y_train_np).to(device),
        torch.from_numpy(sw).to(device),
        cw,
        y_train_np,
    )


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("loading acc_train.npz")
    acc = np.load(ROOT / "acc_train.npz", allow_pickle=True)
    X_acc = acc["X"].astype(np.float32)
    samples = acc["samples"].astype(str)
    labels = acc["labels"].astype(str)
    genes = acc["genes"].astype(str)
    classes = sorted(set(labels.tolist()))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_acc = np.array([cls2idx[c] for c in labels], dtype=np.int64)

    print("loading ebr_eval.npz")
    ebr = np.load(ROOT / "ebr_eval.npz", allow_pickle=True)
    X_ebr = ebr["X"].astype(np.float32)
    ebr_celltype = ebr["celltype"].astype(str)
    ebr_batch = ebr["batch"].astype(str)
    ebr_cluster = ebr["cluster"].astype(str)
    y_ebr = np.array(
        [cls2idx[c] if c in cls2idx else -1 for c in ebr_celltype], dtype=np.int64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    print("classes:", classes)
    n_classes = len(classes)

    train_idx, val_idx = stratified_split(samples, VAL_FRAC, SEED)
    print(f"acc train={train_idx.size} val={val_idx.size}")

    # ---- (1) Leave-batch-out ----
    print("\n== leave-batch-out evaluation ==")
    rows_lbo = []
    rows_ct = []
    for held in ["ear", "nose", "rib"]:
        train_mask = (ebr_batch != held) & (y_ebr >= 0)
        test_mask = (ebr_batch == held) & (y_ebr >= 0)
        n_test = int(test_mask.sum())
        if n_test == 0:
            continue
        t0 = time.time()
        Xt, yt, swt, cw, _ = build_train_tensors(
            X_acc, y_acc, train_idx,
            X_ebr[train_mask], y_ebr[train_mask],
            n_classes, device,
        )
        model = train_loop(Xt, yt, swt, n_classes, cw, device, epochs=EPOCHS)
        Xt_test = torch.from_numpy(X_ebr[test_mask]).to(device)
        model.eval()
        with torch.no_grad():
            pred_idx = model(Xt_test).argmax(dim=1).cpu().numpy()
        pred = np.array([classes[i] for i in pred_idx])
        truth = ebr_celltype[test_mask]
        correct = (pred == truth).astype(np.int64)
        rows_lbo.append({
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
        rows_ct.append(agg)
        print(f"  hold-out {held}: acc_cell={correct.mean():.4f} elapsed={time.time()-t0:.1f}s")
    pd.DataFrame(rows_lbo).to_csv(OUT / "lbo_metrics.tsv", sep="\t", index=False)
    if rows_ct:
        pd.concat(rows_ct, ignore_index=True).to_csv(
            OUT / "lbo_per_celltype.tsv", sep="\t", index=False
        )

    lbo_mean = float(np.mean([r["accuracy_cell"] for r in rows_lbo])) if rows_lbo else 0.0
    print(f"LBO mean cell accuracy: {lbo_mean:.4f}")


    # ---- (2) Within-cluster cell-level holdout (matches v1 protocol) ----
    print("\n== within-cluster cell-level holdout ==")
    rng_e = np.random.default_rng(SEED + 7)
    pair_id = np.array([f"{b}|{c}" for b, c in zip(ebr_batch, ebr_cluster)])
    ebr_train_mask = np.zeros(len(y_ebr), dtype=bool)
    ebr_test_mask = np.zeros(len(y_ebr), dtype=bool)
    for unit in np.unique(pair_id):
        idx = np.where(pair_id == unit)[0]
        rng_e.shuffle(idx)
        n_tr = int(round(len(idx) * EBR_TRAIN_FRAC))
        ebr_train_mask[idx[:n_tr]] = True
        ebr_test_mask[idx[n_tr:]] = True
    ebr_train_mask &= (y_ebr >= 0)
    print(f"ebr-train cells {int(ebr_train_mask.sum())}; ebr-test cells {int(ebr_test_mask.sum())}")

    Xt, yt, swt, cw, y_train_np = build_train_tensors(
        X_acc, y_acc, train_idx,
        X_ebr[ebr_train_mask], y_ebr[ebr_train_mask],
        n_classes, device,
    )
    t0 = time.time()
    model = train_loop(Xt, yt, swt, n_classes, cw, device, epochs=EPOCHS)
    Xt_val = torch.from_numpy(X_acc[val_idx]).to(device)
    model.eval()
    with torch.no_grad():
        pred_val = model(Xt_val).argmax(dim=1).cpu().numpy()
    acc_val = float((pred_val == y_acc[val_idx]).mean())
    f1_val = macro_f1(y_acc[val_idx], pred_val, n_classes)
    bacc_val = balanced_acc(y_acc[val_idx], pred_val, n_classes)
    print(f"  acc-val: acc={acc_val:.4f} f1={f1_val:.4f} bacc={bacc_val:.4f}")

    Xt_ebr = torch.from_numpy(X_ebr[ebr_test_mask]).to(device)
    with torch.no_grad():
        pred_ebr_idx = model(Xt_ebr).argmax(dim=1).cpu().numpy()
    pred_ebr = np.array([classes[i] for i in pred_ebr_idx])
    truth_ebr = ebr_celltype[ebr_test_mask]
    correct = (pred_ebr == truth_ebr).astype(np.int64)
    n_ebr = int(ebr_test_mask.sum())
    metrics = {
        "n_cells": n_ebr,
        "ebr_present_classes": int(len(set(truth_ebr.tolist()))),
        "accuracy_cell": round(float(correct.mean()), 4),
        "n_correct_cell": int(correct.sum()),
    }
    df_cluster = pd.DataFrame({
        "batch": ebr_batch[ebr_test_mask],
        "cluster": ebr_cluster[ebr_test_mask],
        "pred": pred_ebr,
        "celltype": truth_ebr,
        "correct": correct,
    })
    rows_clus = []
    for (b, cl), grp in df_cluster.groupby(["batch", "cluster"]):
        rows_clus.append({
            "batch": b, "cluster": cl, "n_cells": int(len(grp)),
            "majority_pred": grp["pred"].value_counts().idxmax(),
            "majority_celltype": grp["celltype"].value_counts().idxmax(),
            "match": int(grp["pred"].value_counts().idxmax() == grp["celltype"].value_counts().idxmax()),
            "cell_recall": round(float(grp["correct"].mean()), 4),
        })
    metrics["cluster_top1_match"] = round(float(np.mean([r["match"] for r in rows_clus])), 4)
    metrics["cluster_n"] = len(rows_clus)
    pd.DataFrame([metrics]).to_csv(OUT / "within_cluster_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(rows_clus).to_csv(OUT / "within_cluster_per_cluster.tsv", sep="\t", index=False)
    by_celltype = pd.DataFrame({"celltype": truth_ebr, "correct": correct})
    per_ct = by_celltype.groupby("celltype").agg(n=("correct", "size"), correct=("correct", "sum"))
    per_ct["recall"] = (per_ct["correct"] / per_ct["n"]).round(4)
    per_ct.reset_index().to_csv(OUT / "within_cluster_per_celltype.tsv", sep="\t", index=False)
    print(f"  within-cluster: cell={metrics['accuracy_cell']:.4f} cluster={metrics['cluster_top1_match']:.4f}")

    ckpt = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "genes": genes.tolist(),
        "config": {
            "n_in": int(X_acc.shape[1]),
            "n_classes": n_classes,
            "hidden1": HIDDEN1,
            "hidden2": HIDDEN2,
            "dropout": DROPOUT,
            "input_dropout": INPUT_DROPOUT,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "noise_std": NOISE_STD,
            "mixup_alpha": MIXUP_ALPHA,
            "weight_decay": WEIGHT_DECAY,
        },
    }
    torch.save(ckpt, OUT / "classifier_v2.pt")
    summary = {
        "lbo_per_batch": rows_lbo,
        "lbo_mean_accuracy_cell": round(lbo_mean, 4),
        "within_cluster": metrics,
        "acc_val": {
            "accuracy": round(acc_val, 4),
            "macro_f1": round(f1_val, 4),
            "balanced_acc": round(bacc_val, 4),
        },
        "elapsed_total_s": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("v2 summary saved ->", OUT / "summary.json")


if __name__ == "__main__":
    main()
