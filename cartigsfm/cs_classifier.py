"""Production cell-subtype classifier (P-F).

An EBR-supervised + acc_new-supervised hybrid MLP that maps log1p-normalized
expression on a fixed 2000-gene HVG basis to a 10-class softmax over the
cell_subtype panel of cartilage_dictionary_v1.

The trained checkpoint and HVG gene basis are bundled in
``cartigsfm/resources/cs_classifier_v1/`` (``classifier.pt`` +
``hvg_genes.tsv``). The checkpoint stores the canonical class order, gene
order, and config so the loader can validate input dimensionality at
inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CSClassifierConfig:
    n_in: int = 2000
    n_classes: int = 10
    hidden1: int = 384
    hidden2: int = 192
    dropout: float = 0.4
    input_dropout: float = 0.1


class CSClassifier(nn.Module):
    """Match the architecture of scripts/train_cartigm_classifier.py."""

    def __init__(self, cfg: CSClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.n_in)
        self.input_drop = nn.Dropout(cfg.input_dropout)
        self.net = nn.Sequential(
            nn.Linear(cfg.n_in, cfg.hidden1), nn.LayerNorm(cfg.hidden1), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden1, cfg.hidden2), nn.LayerNorm(cfg.hidden2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden2, cfg.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_drop(x)
        return self.net(x)


def load_classifier(ckpt_path: str | Path, device: str | torch.device | None = None):
    """Load a CSClassifier checkpoint produced by train_cartigm_classifier.py.

    Returns ``(model, classes, genes, config)``.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_d = ckpt.get("config", {})
    cfg = CSClassifierConfig(
        n_in=int(cfg_d.get("n_in", 2000)),
        n_classes=int(cfg_d.get("n_classes", 10)),
        hidden1=int(cfg_d.get("hidden1", 384)),
        hidden2=int(cfg_d.get("hidden2", 192)),
        dropout=float(cfg_d.get("dropout", 0.4)),
    )
    model = CSClassifier(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if device is not None:
        model = model.to(device)
    classes: List[str] = list(ckpt.get("classes", []))
    genes: List[str] = list(ckpt.get("genes", []))
    return model, classes, genes, cfg


def bundled_classifier_path() -> Path:
    """Return the filesystem path of the bundled v1 classifier checkpoint."""
    p = resources.files("cartigsfm").joinpath(
        "resources", "cs_classifier_v1", "classifier.pt"
    )
    return Path(str(p))


def load_bundled_classifier(device: str | torch.device | None = None):
    """Load the bundled cs_classifier_v1 checkpoint.

    Returns ``(model, classes, genes, config)``.
    """
    return load_classifier(bundled_classifier_path(), device=device)


def align_to_genes(
    X: np.ndarray,
    src_genes: Sequence[str],
    dst_genes: Sequence[str],
) -> tuple[np.ndarray, int]:
    """Reorder/pad a (n_cells, n_src) matrix to a (n_cells, n_dst) matrix.

    Genes present in ``src_genes`` are copied to the matching column in
    ``dst_genes``; missing destination genes are filled with zeros. Returns
    the aligned dense float32 array plus the number of source-side hits.
    """
    src_idx = {g: i for i, g in enumerate(src_genes)}
    n = X.shape[0]
    out = np.zeros((n, len(dst_genes)), dtype=np.float32)
    hit = 0
    is_sparse = hasattr(X, "tocsc")
    if is_sparse:
        X = X.tocsc()
    for j, g in enumerate(dst_genes):
        i = src_idx.get(g)
        if i is None:
            continue
        if is_sparse:
            col = np.asarray(X[:, i].toarray()).ravel()
        else:
            col = X[:, i]
        out[:, j] = np.asarray(col, dtype=np.float32)
        hit += 1
    return out, hit


def predict_from_array(
    X: np.ndarray,
    model: CSClassifier,
    classes: Sequence[str],
    device: str | torch.device | None = None,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict class index + probability matrix from an ``(n_cells, n_genes)`` array.

    Caller is responsible for aligning columns to the checkpoint's `genes`
    order and using log1p_norm (or otherwise scale-matched) values.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n = X.shape[0]
    out_idx = np.zeros(n, dtype=np.int64)
    out_probs = np.zeros((n, len(classes)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            out_idx[i:i + batch_size] = logits.argmax(dim=1).cpu().numpy()
            out_probs[i:i + batch_size] = probs.cpu().numpy()
    return out_idx, out_probs
