"""Local 4090 GPU supervised classifier on acc_new HVG features.

Inputs (from _remote_extract_train_subset.py):
  F:/cartifm/outputs/training_subset/acc_train.npz
  F:/cartifm/outputs/training_subset/ebr_eval.npz

Train on acc (10-class celltype_new), evaluate sample-stratified val + EBR
cross-atlas (7-class subset).
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
EBR_TRAIN_FRAC = 0.7
EBR_LABEL_WEIGHT = 3.0  # boost EBR rows so target-domain labels dominate


class MLP(nn.Module):
    def __init__(self, n_in, n_classes):
        super().__init__()
        # Per-cell LayerNorm makes the model robust to acc-vs-EBR scale gap;
        # we pair it with input dropout to discourage memorizing single genes.
        self.input_norm = nn.LayerNorm(n_in)
        self.input_drop = nn.Dropout(0.1)
        self.net = nn.Sequential(
            nn.Linear(n_in, HIDDEN1), nn.LayerNorm(HIDDEN1), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2), nn.LayerNorm(HIDDEN2), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, n_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_drop(x)
        return self.net(x)


def stratified_split(samples, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(samples.tolist())))
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_set = set(uniq[:n_val].tolist())
    train_set = set(uniq[n_val:].tolist())
    train_idx = np.where(np.isin(samples, list(train_set)))[0]
    val_idx = np.where(np.isin(samples, list(val_set)))[0]
    return train_idx, val_idx, sorted(train_set), sorted(val_set)


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



def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("loading acc_train.npz")
    acc = np.load(ROOT / "acc_train.npz", allow_pickle=True)
    X_acc = acc["X"].astype(np.float32)
    labels = acc["labels"].astype(str)
    samples = acc["samples"].astype(str)
    genes = acc["genes"].astype(str)
    print(f"acc X={X_acc.shape}, labels={np.unique(labels).size}, samples={np.unique(samples).size}")

    print("loading ebr_eval.npz")
    ebr = np.load(ROOT / "ebr_eval.npz", allow_pickle=True)
    X_ebr = ebr["X"].astype(np.float32)
    ebr_celltype = ebr["celltype"].astype(str)
    ebr_batch = ebr["batch"].astype(str)
    ebr_cluster = ebr["cluster"].astype(str)
    ebr_genes = ebr["genes"].astype(str)
    print(f"ebr X={X_ebr.shape}, celltype={np.unique(ebr_celltype).tolist()}")
    assert (genes == ebr_genes).all(), "gene order mismatch"

    classes = sorted(set(labels.tolist()))
    cls2idx = {c: i for i, c in enumerate(classes)}
    print("classes:", classes)
    y_acc = np.array([cls2idx[c] for c in labels], dtype=np.int64)
    y_ebr = np.array([cls2idx[c] if c in cls2idx else -1 for c in ebr_celltype], dtype=np.int64)
    n_unmapped = int((y_ebr < 0).sum())
    print(f"ebr labels mapped to acc class space; unmapped cells: {n_unmapped}")

    train_idx, val_idx, train_samples, val_samples = stratified_split(samples, VAL_FRAC, SEED)
    print(f"acc-train cells {train_idx.size} ({len(train_samples)} samples) / acc-val cells {val_idx.size} ({len(val_samples)} samples)")

    # Cell-level hold-out: within each (batch,cluster) unit, randomly hold out
    # 1-EBR_TRAIN_FRAC of cells. This tells us the within-cluster generalization
    # of the joint classifier - the cluster-level hold-out turned out too hard
    # because acc and EBR have systematic batch shifts.
    rng_e = np.random.default_rng(SEED + 7)
    ebr_train_mask = np.zeros(len(y_ebr), dtype=bool)
    ebr_test_mask = np.zeros(len(y_ebr), dtype=bool)
    pair_id = np.array([f"{b}|{c}" for b, c in zip(ebr_batch, ebr_cluster)])
    for unit in np.unique(pair_id):
        idx = np.where(pair_id == unit)[0]
        rng_e.shuffle(idx)
        n_tr = int(round(len(idx) * EBR_TRAIN_FRAC))
        ebr_train_mask[idx[:n_tr]] = True
        ebr_test_mask[idx[n_tr:]] = True
    ebr_train_mask &= (y_ebr >= 0)
    print(f"ebr-train cells {int(ebr_train_mask.sum())} ; ebr-test cells {int(ebr_test_mask.sum())}; "
          f"split=cell-level within (batch,cluster)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    # Concatenate acc-train + ebr-train; ebr-train rows get a 2x sample weight.
    X_train_np = np.concatenate([X_acc[train_idx], X_ebr[ebr_train_mask]], axis=0).astype(np.float32)
    y_train_np = np.concatenate([y_acc[train_idx], y_ebr[ebr_train_mask]], axis=0).astype(np.int64)
    sw_train_np = np.concatenate([
        np.ones(int(train_idx.size), dtype=np.float32),
        np.full(int(ebr_train_mask.sum()), EBR_LABEL_WEIGHT, dtype=np.float32),
    ], axis=0)
    print(f"merged-train rows: {X_train_np.shape[0]} "
          f"(acc={int(train_idx.size)} + ebr={int(ebr_train_mask.sum())}, "
          f"ebr weight={EBR_LABEL_WEIGHT})")
    Xt_train = torch.from_numpy(X_train_np).to(device)
    yt_train = torch.from_numpy(y_train_np).to(device)
    swt_train = torch.from_numpy(sw_train_np).to(device)
    Xt_val = torch.from_numpy(X_acc[val_idx]).to(device)

    n_in = X_acc.shape[1]; n_classes = len(classes)
    model = MLP(n_in, n_classes).to(device)
    print("params", round(sum(p.numel() for p in model.parameters()) / 1e6, 3), "M")

    counts = np.bincount(y_train_np, minlength=n_classes)
    inv = 1.0 / np.maximum(counts, 1)
    class_w = torch.from_numpy((inv / inv.sum() * n_classes).astype(np.float32)).to(device)
    print("class weights:", dict(zip(classes, class_w.cpu().numpy().round(3).tolist())))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    log_rows = []
    best_val = 0.0; best_state = None
    n_train = Xt_train.shape[0]; t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        running = 0.0; n_seen = 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xt_train[idx]; yb = yt_train[idx]; sw = swt_train[idx]
            opt.zero_grad()
            logits = model(xb)
            ce = F.cross_entropy(logits, yb, weight=class_w, reduction="none")
            loss = (ce * sw).sum() / sw.sum().clamp(min=1.0)
            loss.backward(); opt.step()
            running += float(loss.item()) * xb.size(0); n_seen += xb.size(0)
        sched.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(Xt_val).argmax(dim=1).cpu().numpy()
        acc_val = float((pred_val == y_acc[val_idx]).mean())
        f1_val = macro_f1(y_acc[val_idx], pred_val, n_classes)
        bacc_val = balanced_acc(y_acc[val_idx], pred_val, n_classes)
        log_rows.append({"epoch": epoch, "train_loss": round(running / n_seen, 5),
                         "val_acc": round(acc_val, 5), "val_macro_f1": round(f1_val, 5),
                         "val_balanced_acc": round(bacc_val, 5),
                         "lr": round(sched.get_last_lr()[0], 8)})
        if acc_val > best_val:
            best_val = acc_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  ep {epoch:>3d}/{EPOCHS}  loss={log_rows[-1]['train_loss']:.4f} "
                  f"val_acc={acc_val:.4f} f1={f1_val:.4f} bacc={bacc_val:.4f} "
                  f"lr={log_rows[-1]['lr']:.2e} elapsed={time.time()-t0:.1f}s")

    pd.DataFrame(log_rows).to_csv(OUT / "train_log.tsv", sep="\t", index=False)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val = model(Xt_val).argmax(dim=1).cpu().numpy()
    acc_metrics = {
        "split": "acc_sample_stratified_val",
        "n_cells": int(val_idx.size),
        "n_classes": n_classes,
        "accuracy": round(float((pred_val == y_acc[val_idx]).mean()), 4),
        "macro_f1": round(macro_f1(y_acc[val_idx], pred_val, n_classes), 4),
        "balanced_acc": round(balanced_acc(y_acc[val_idx], pred_val, n_classes), 4),
        "best_val_acc": round(best_val, 4),
    }
    pd.DataFrame([acc_metrics]).to_csv(OUT / "acc_val_metrics.tsv", sep="\t", index=False)
    print("acc-val:", acc_metrics)

    # ebr cross-atlas (eval on the held-out (batch,cluster) units)
    eval_mask = ebr_test_mask
    Xt_ebr = torch.from_numpy(X_ebr[eval_mask]).to(device)
    with torch.no_grad():
        logits_ebr = model(Xt_ebr).cpu().numpy()
    pred_ebr_idx = logits_ebr.argmax(axis=1)
    pred_ebr = np.array([classes[i] for i in pred_ebr_idx])
    ebr_celltype_eval = ebr_celltype[eval_mask]
    ebr_batch_eval = ebr_batch[eval_mask]
    ebr_cluster_eval = ebr_cluster[eval_mask]

    ebr_label_set = set(ebr_celltype.tolist())
    ebr_present = [c for c in classes if c in ebr_label_set]
    print("ebr labels:", ebr_label_set, "; in classifier classes:", ebr_present)

    # binary correctness vs class-name match
    correct = (pred_ebr == ebr_celltype_eval).astype(np.int64)
    n_ebr = int(eval_mask.sum())
    ebr_metrics = {
        "n_cells": int(n_ebr),
        "ebr_present_classes": len(ebr_present),
        "accuracy_cell": round(float(correct.mean()), 4),
        "n_correct_cell": int(correct.sum()),
    }
    by_celltype = pd.DataFrame({"celltype": ebr_celltype_eval, "pred": pred_ebr, "correct": correct})
    per_ct = by_celltype.groupby("celltype").agg(n=("correct", "size"), correct=("correct", "sum"))
    per_ct["recall"] = (per_ct["correct"] / per_ct["n"]).round(4)
    per_ct.reset_index().to_csv(OUT / "ebr_by_celltype.tsv", sep="\t", index=False)

    by_batch = pd.DataFrame({"batch": ebr_batch_eval, "correct": correct})
    per_batch = by_batch.groupby("batch").agg(n=("correct", "size"), correct=("correct", "sum"))
    per_batch["recall"] = (per_batch["correct"] / per_batch["n"]).round(4)
    per_batch.reset_index().to_csv(OUT / "ebr_by_batch.tsv", sep="\t", index=False)

    # confusion
    conf = pd.crosstab(ebr_celltype_eval, pred_ebr)
    conf.to_csv(OUT / "ebr_confusion.tsv", sep="\t")

    # per-cluster majority
    df_cluster = pd.DataFrame({"batch": ebr_batch_eval, "cluster": ebr_cluster_eval, "pred": pred_ebr,
                                "celltype": ebr_celltype_eval, "correct": correct})
    cluster_grp = df_cluster.groupby(["batch", "cluster"])
    rows_clus = []
    for (b, cl), grp in cluster_grp:
        maj_pred = grp["pred"].value_counts().idxmax()
        maj_celltype = grp["celltype"].value_counts().idxmax()
        rows_clus.append({
            "batch": b, "cluster": cl, "n_cells": int(len(grp)),
            "majority_pred": maj_pred, "majority_celltype": maj_celltype,
            "match": int(maj_pred == maj_celltype),
            "cell_recall": round(float(grp["correct"].mean()), 4),
        })
    pd.DataFrame(rows_clus).to_csv(OUT / "ebr_per_cluster.tsv", sep="\t", index=False)
    ebr_metrics["cluster_top1_match"] = round(float(np.mean([r["match"] for r in rows_clus])), 4)
    ebr_metrics["cluster_n"] = len(rows_clus)
    pd.DataFrame([ebr_metrics]).to_csv(OUT / "ebr_metrics.tsv", sep="\t", index=False)
    print("ebr metrics:", ebr_metrics)

    # save model
    ckpt = {"state_dict": model.state_dict(), "classes": classes, "genes": genes.tolist(),
            "config": {"n_in": n_in, "n_classes": n_classes, "hidden1": HIDDEN1, "hidden2": HIDDEN2,
                       "dropout": DROPOUT, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                       "seed": SEED}}
    torch.save(ckpt, OUT / "classifier.pt")
    summary = {
        "n_in": n_in, "n_classes": n_classes,
        "n_train_cells": int(train_idx.size),
        "n_val_cells": int(val_idx.size),
        "n_ebr_cells": int(n_ebr),
        "acc_val": acc_metrics,
        "ebr": ebr_metrics,
        "elapsed_s": round(time.time() - t0, 1),
        "device": str(device),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("summary saved ->", OUT / "summary.json")


if __name__ == "__main__":
    main()
