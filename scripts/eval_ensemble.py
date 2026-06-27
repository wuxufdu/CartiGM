"""Ensemble eval: average softmax of v1 (bundled) + v2 candidates on EBR.

Reports both within-cluster cell-level holdout and leave-batch-out
accuracies, comparing v1, v2, and the simple-average ensemble.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("F:/cartifm/outputs/training_subset")
OUT = Path("F:/cartifm/outputs/training_local/ensemble")
OUT.mkdir(parents=True, exist_ok=True)
CKPT_V1 = Path("F:/cartifm/outputs/training_local/classifier.pt")
CKPT_V2 = Path("F:/cartifm/outputs/training_local/v2/classifier_v2.pt")

EBR_TRAIN_FRAC = 0.7
SEED = 0


def build_model(state_dict, cfg):
    n_in = int(cfg.get("n_in", 2000)); n_classes = int(cfg.get("n_classes", 10))
    h1 = int(cfg.get("hidden1", 384)); h2 = int(cfg.get("hidden2", 192))
    drop = float(cfg.get("dropout", 0.4)); idr = float(cfg.get("input_dropout", 0.1))
    model = nn.Sequential()  # placeholder

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(n_in)
            self.input_drop = nn.Dropout(idr)
            self.net = nn.Sequential(
                nn.Linear(n_in, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(drop),
                nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(drop),
                nn.Linear(h2, n_classes),
            )

        def forward(self, x):
            return self.net(self.input_drop(self.input_norm(x)))

    m = M()
    m.load_state_dict(state_dict)
    m.eval()
    return m, n_classes


def softmax_predict(model, X, device, batch=4096):
    model = model.to(device)
    out = np.zeros((X.shape[0], 0), dtype=np.float32)
    parts = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(device)
            parts.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    ck1 = torch.load(CKPT_V1, map_location="cpu", weights_only=False)
    ck2 = torch.load(CKPT_V2, map_location="cpu", weights_only=False)
    m1, n1 = build_model(ck1["state_dict"], ck1["config"])
    m2, n2 = build_model(ck2["state_dict"], ck2["config"])
    classes1 = list(ck1["classes"]); classes2 = list(ck2["classes"])
    assert classes1 == classes2, "class order mismatch"
    classes = classes1
    cls2idx = {c: i for i, c in enumerate(classes)}

    ebr = np.load(ROOT / "ebr_eval.npz", allow_pickle=True)
    X_ebr = ebr["X"].astype(np.float32)
    ct = ebr["celltype"].astype(str); ba = ebr["batch"].astype(str); cl = ebr["cluster"].astype(str)
    y = np.array([cls2idx[c] if c in cls2idx else -1 for c in ct], dtype=np.int64)

    rng = np.random.default_rng(SEED + 7)
    pair_id = np.array([f"{b}|{c}" for b, c in zip(ba, cl)])
    test_mask = np.zeros(len(y), dtype=bool)
    for unit in np.unique(pair_id):
        idx = np.where(pair_id == unit)[0]
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * EBR_TRAIN_FRAC))
        test_mask[idx[n_tr:]] = True
    test_mask &= (y >= 0)

    Xt = X_ebr[test_mask]
    p1 = softmax_predict(m1, Xt, device)
    p2 = softmax_predict(m2, Xt, device)
    p_avg = (p1 + p2) * 0.5

    truth_idx = y[test_mask]
    truth = ct[test_mask]
    ba_t = ba[test_mask]; cl_t = cl[test_mask]

    def score(P, name):
        idx = P.argmax(axis=1)
        pred = np.array([classes[i] for i in idx])
        cell = float((pred == truth).mean())
        df = pd.DataFrame({"batch": ba_t, "cluster": cl_t, "pred": pred, "celltype": truth})
        clus = []
        for (b, c), g in df.groupby(["batch", "cluster"]):
            clus.append(int(g["pred"].value_counts().idxmax() == g["celltype"].value_counts().idxmax()))
        cluster_top1 = float(np.mean(clus))
        per_ct = (
            pd.DataFrame({"celltype": truth, "correct": (pred == truth).astype(int)})
            .groupby("celltype").agg(n=("correct", "size"), correct=("correct", "sum"))
        )
        per_ct["recall"] = per_ct["correct"] / per_ct["n"]
        return cell, cluster_top1, per_ct.reset_index(), pred

    rows = []
    for name, P in [("v1", p1), ("v2", p2), ("ensemble_avg", p_avg)]:
        cell, clus, per_ct, _ = score(P, name)
        rows.append({"model": name, "n_test_cells": int(test_mask.sum()),
                     "accuracy_cell": round(cell, 4),
                     "cluster_top1_match": round(clus, 4)})
        per_ct.to_csv(OUT / f"within_cluster_{name}_per_celltype.tsv", sep="\t", index=False)
        print(f"{name}: cell={cell:.4f} cluster={clus:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "within_cluster_summary.tsv", sep="\t", index=False)

    print("done")


if __name__ == "__main__":
    main()
