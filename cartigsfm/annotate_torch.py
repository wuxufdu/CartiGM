"""GPU-runnable trainer that drops in for celltypist.train."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .assets import prefer_device


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _label_encode(y: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
    classes = sorted(set(str(v) for v in y))
    idx = {c: i for i, c in enumerate(classes)}
    encoded = np.fromiter((idx[str(v)] for v in y), dtype=np.int64, count=-1)
    return encoded, classes


def _feature_select(
    x: np.ndarray, top_n: int = 300, min_expr: float = 0.1
) -> np.ndarray:
    """Pick the top-N most-expressed genes (celltypist-style rank-by-mean)."""
    if x.shape[1] <= top_n:
        return np.arange(x.shape[1])
    keep = np.where(x.mean(axis=0) > float(min_expr))[0]
    if keep.size <= top_n:
        keep = np.arange(x.shape[1])
    sub = x[:, keep]
    order = np.argsort(-sub.mean(axis=0))[: int(top_n)]
    return keep[order]


def train_logreg_torch(
    ref,
    *,
    labels: str,
    feature_selection: bool = True,
    max_iter: int = 200,
    lr: float = 0.1,
    batch_size: int = 256,
    l2: float = 1e-4,
    top_n_features: int = 300,
    device: Optional[str] = None,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a multiclass logistic regression on ref.X using torch on the chosen device."""
    import torch
    import torch.nn as nn

    chosen = prefer_device(device)
    rng = np.random.default_rng(int(seed))

    if labels not in ref.obs.columns:
        raise ValueError("ref.obs has no column " + repr(labels))
    y_text = ref.obs[labels].astype(str).values
    x_dense = _to_dense(ref.X)
    gene_names = [str(g) for g in ref.var_names]

    if feature_selection:
        keep_idx = _feature_select(x_dense, top_n=int(top_n_features))
        x_dense = x_dense[:, keep_idx]
        gene_names = [gene_names[i] for i in keep_idx]

    y, classes = _label_encode(y_text)
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("need at least 2 classes; got " + repr(classes))
    n, d = x_dense.shape

    torch.manual_seed(int(seed))
    model = nn.Linear(int(d), int(n_classes)).to(chosen)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(l2))
    loss_fn = nn.CrossEntropyLoss()

    x_t = torch.from_numpy(x_dense).to(chosen).float()
    y_t = torch.from_numpy(y).to(chosen)

    n_iter = int(max(1, max_iter))
    for epoch in range(n_iter):
        perm = torch.from_numpy(rng.permutation(n)).to(chosen)
        loss_acc = 0.0
        n_batches = 0
        for start in range(0, n, int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            logits = model(x_t[idx])
            loss = loss_fn(logits, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += float(loss.detach().cpu())
            n_batches += 1
        if verbose and epoch % 10 == 0:
            with torch.no_grad():
                full_loss = float(loss_fn(model(x_t), y_t).cpu())
            print("[train_logreg_torch] epoch", epoch, "batch_loss=", round(loss_acc / max(1, n_batches), 4), "full_loss=", round(full_loss, 4))

    with torch.no_grad():
        coef_t = model.weight.detach().cpu().numpy()
        intercept_t = model.bias.detach().cpu().numpy()

    return {
        "coef_": coef_t.astype(np.float32),
        "intercept_": intercept_t.astype(np.float32),
        "classes_": list(classes),
        "features": gene_names,
        "device": str(chosen),
        "torch_version": getattr(torch, "__version__", ""),
        "n_iter": n_iter,
        "n_samples": int(n),
        "n_features": int(d),
        "n_classes": int(n_classes),
    }


def torch_model_to_dataframe(model: Dict[str, Any]) -> pd.DataFrame:
    """Render a torch-trained model as a tidy (class, feature, weight) frame."""
    coef = model["coef_"]
    classes = model["classes_"]
    features = model["features"]
    rows = []
    for i, cls in enumerate(classes):
        for j, feat in enumerate(features):
            rows.append({
                "class": str(cls),
                "feature": str(feat),
                "weight": float(coef[i, j]),
            })
    return pd.DataFrame(rows)
