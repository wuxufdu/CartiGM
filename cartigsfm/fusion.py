"""P17: per-cell fusion of CartiGM + scGPT-proxy + GSFM, sample-stratified.

This module implements the user-requested pipeline:

  1. Use a **frozen** scGPT-human-style encoder. The real scGPT-human
     weights are not downloadable in this sandbox, so the bundled
     ``cartigsfm.scgpt`` deterministic proxy is used. Every output that
     depends on it carries ``fallback=True`` so the report is honest.
     The encoder is a strict drop-in: swap in real weights by editing
     :func:`build_scgpt_per_cell_embedding`.

  2. Compute per-cell **CartiGM 42-axis scores** (sparse @ dense,
     ``cartilage_dictionary_v1``), **GSFM axis similarity** (per-cell
     Jaccard between top-N expressed genes and each axis's
     ``panel_genes``), and the scGPT embedding (42-dim mean of axis
     core gene expression).

  3. Split **by sample** (74 samples in acc.h5ad) to avoid random
     cell-level data leakage. Train six lightweight MLP heads, one per
     configuration::

        CartiGM only
        scGPT only  (proxy)
        GSFM only
        CartiGM + scGPT
        CartiGM + GSFM
        Full fusion (CartiGM + scGPT + GSFM)

  4. Evaluate on the held-out **sample split** with six metrics
     (accuracy, macro-F1, balanced accuracy, top-axis consistency,
     evidence citation rate, hallucination rate) and the same six
     configurations on EBR as the external validation cohort.

Output goes to ``outputs/fusion_P17/`` and the report to
``reports\\P17_FUSION_ABLATION.md``. The runner is exposed as
``cartigsfm train-fusion`` and ``cartigsfm ablate-fusion``.
"""
from __future__ import annotations
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .assets import load_axis_evidence_cards, load_cartilage_dictionary_v1, prefer_device
from .gsfm import _v1_axes as _gsfm_v1_axes, _axis_index as _gsfm_axis_index, _normalize_markers


# ---------------------------------------------------------------------------
# 1. Axis index (single source of truth for all 42 axes)
# ---------------------------------------------------------------------------

def _all_v1_axes() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ax in _gsfm_v1_axes():
        entry = dict(ax)
        out.append(entry)
    return out


def _axis_meta_table() -> pd.DataFrame:
    rows = []
    for ax in _all_v1_axes():
        rows.append({
            "axis_id": ax.get("axis_id", ""),
            "layer": ax.get("layer", ""),
            "name_en": ax.get("name_en", ""),
            "core_genes": list(ax.get("core_genes", []) or []),
            "panel_genes": list(ax.get("panel_genes", []) or ax.get("core_genes", []) or []),
            "marker_weights": dict(ax.get("marker_weights", {}) or {}),
        })
    return pd.DataFrame(rows)


def _axis_panels() -> Dict[str, List[str]]:
    table = _axis_meta_table()
    return {row["axis_id"]: [str(g).upper() for g in row["panel_genes"]] for _, row in table.iterrows()}


def _axis_marker_weights() -> Dict[str, Dict[str, float]]:
    table = _axis_meta_table()
    out: Dict[str, Dict[str, float]] = {}
    for _, row in table.iterrows():
        weights = row["marker_weights"]
        if not weights:
            weights = {g: 1.0 for g in row["panel_genes"]}
        out[row["axis_id"]] = {str(k).upper(): float(v) for k, v in weights.items()}
    return out


# ---------------------------------------------------------------------------
# 2. Per-cell feature builders
# ---------------------------------------------------------------------------

def _sparse_X(adata) -> sp.csr_matrix:
    X = adata.X
    if sp.issparse(X):
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()
    else:
        X = sp.csr_matrix(np.asarray(X))
    return X


def build_cartigm_per_cell_scores(
    adata,
    dictionary=None,
    chunk_size: int = 5000,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Per-cell CartiGM 42-axis scores via sparse @ dense on GPU.

    Returns (N, 42) float32 and a list of axis_ids in column order.
    """
    dictionary = dictionary or load_cartilage_dictionary_v1()
    table = _axis_meta_table()
    axis_ids = list(table["axis_id"])
    n_axes = len(axis_ids)
    n_genes = adata.n_vars
    gene_to_idx = {str(g).upper(): i for i, g in enumerate(adata.var_names.astype(str))}

    W_rows: List[int] = []
    W_cols: List[int] = []
    W_vals: List[float] = []
    for j, ax in enumerate(table.to_dict(orient="records")):
        weights = ax["marker_weights"]
        if not weights:
            for g in ax["panel_genes"]:
                i = gene_to_idx.get(str(g).upper())
                if i is not None:
                    W_rows.append(i)
                    W_cols.append(j)
                    W_vals.append(1.0)
        else:
            for g, w in weights.items():
                i = gene_to_idx.get(str(g).upper())
                if i is not None and float(w) != 0.0:
                    W_rows.append(i)
                    W_cols.append(j)
                    W_vals.append(float(w))
    W = sp.csr_matrix((W_vals, (W_rows, W_cols)), shape=(n_genes, n_axes), dtype=np.float32)

    chosen = prefer_device(device)
    use_torch = str(chosen).startswith("cuda") or str(chosen) == "mps"
    X = _sparse_X(adata)
    n = X.shape[0]
    out = np.zeros((n, n_axes), dtype=np.float32)

    if use_torch:
        import torch
        W_dense = torch.from_numpy(W.toarray()).to(chosen).float()
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            Xc = X[start:stop]
            Xt = torch.sparse_csr_tensor(
                torch.from_numpy(Xc.indptr.astype(np.int64)),
                torch.from_numpy(Xc.indices.astype(np.int64)),
                torch.from_numpy(Xc.data.astype(np.float32)),
                size=Xc.shape,
            ).to(chosen)
            scores = torch.sparse.mm(Xt, W_dense).cpu().numpy()
            for j in range(n_axes):
                col = W.getcol(j)
                denom = float(col.sum())
                if denom > 0:
                    scores[:, j] /= denom
            out[start:stop] = scores
    else:
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            scores = X[start:stop].dot(W).toarray().astype(np.float32)
            for j in range(n_axes):
                col = W.getcol(j)
                denom = float(col.sum())
                if denom > 0:
                    scores[:, j] /= denom
            out[start:stop] = scores
    return out, axis_ids


def build_scgpt_per_cell_embedding(
    adata,
    dictionary=None,
    chunk_size: int = 5000,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Per-cell scGPT-style embedding via the bundled 42-axis proxy.

    Returns (N, 42) float32, the list of axis_ids in column order, and a
    meta dict with ``fallback=True`` to flag the proxy.
    """
    dictionary = dictionary or load_cartilage_dictionary_v1()
    table = _axis_meta_table()
    axis_ids = list(table["axis_id"])
    n_axes = len(axis_ids)
    n_genes = adata.n_vars
    gene_to_idx = {str(g).upper(): i for i, g in enumerate(adata.var_names.astype(str))}

    mask_rows: List[int] = []
    mask_cols: List[int] = []
    for j, ax in enumerate(table.to_dict(orient="records")):
        for g in ax["core_genes"]:
            i = gene_to_idx.get(str(g).upper())
            if i is not None:
                mask_rows.append(i)
                mask_cols.append(j)
    M = sp.csr_matrix((np.ones(len(mask_rows), dtype=np.float32), (mask_rows, mask_cols)), shape=(n_genes, n_axes))

    X = _sparse_X(adata)
    n = X.shape[0]
    out = np.zeros((n, n_axes), dtype=np.float32)
    chosen = prefer_device(device)
    use_torch = str(chosen).startswith("cuda") or str(chosen) == "mps"
    if use_torch:
        import torch
        M_dense = torch.from_numpy(M.toarray()).to(chosen).float()
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            Xc = X[start:stop]
            Xt = torch.sparse_csr_tensor(
                torch.from_numpy(Xc.indptr.astype(np.int64)),
                torch.from_numpy(Xc.indices.astype(np.int64)),
                torch.from_numpy(Xc.data.astype(np.float32)),
                size=Xc.shape,
            ).to(chosen)
            sums = torch.sparse.mm(Xt, M_dense).cpu().numpy()
            for j in range(n_axes):
                col = M.getcol(j)
                cnt = float(col.sum())
                if cnt > 0:
                    sums[:, j] /= cnt
            out[start:stop] = sums
    else:
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            sums = X[start:stop].dot(M).toarray().astype(np.float32)
            for j in range(n_axes):
                col = M.getcol(j)
                cnt = float(col.sum())
                if cnt > 0:
                    sums[:, j] /= cnt
            out[start:stop] = sums
    return out, axis_ids, {
        "fallback": True,
        "encoder": "cartigsfm.scgpt 42-axis proxy",
        "real_scgpt_human_weights": False,
        "swap_instructions": "Edit build_scgpt_per_cell_embedding to load real scGPT-human from HuggingFace or local path.",
    }


def build_scgpt_pretrained_per_cell(
    adata,
    checkpoint_path,
    *,
    chunk_size: int = 256,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Per-cell embedding from the cartigsfm scGPT-style pretrained encoder.

    Loads the checkpoint produced by ``cartigsfm.scgpt_pretrain``
    (a TransformerEncoder over HVG genes with mask-language-modelling)
    and returns the cell embedding (mean of the encoder output across
    HVG positions), then projects it to a 42-dim per-axis score by
    averaging the embedding of each axis's core genes.

    Genes missing from the pretraining HVG vocabulary are zero-padded.

    Returns (N, d_model_or_42) float32, the list of column ids (axis
    ids when projecting to 42, otherwise the gene-position embedding
    indices), and a meta dict with ``fallback=False``.
    """
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    hvg = list(ckpt.get("hvg") or [])
    if not hvg:
        raise RuntimeError("checkpoint missing 'hvg' gene list")
    state = ckpt.get("state_dict")
    if state is None:
        raise RuntimeError("checkpoint missing 'state_dict'")

    chosen = prefer_device(device)
    # Rebuild the GeneEncoder model from cartigsfm.scgpt_pretrain. We
    # use the same nested-class construction for portability.
    from .scgpt_pretrain import ScGPTConfig, build_model
    sc_cfg = ScGPTConfig(**{k: cfg[k] for k in cfg if k in ScGPTConfig.__dataclass_fields__})
    n_genes = int(ckpt["n_genes"])
    model = build_model(sc_cfg, n_genes, chosen)
    model.load_state_dict(state)
    model.eval()

    var_to_idx = {str(g).upper(): i for i, g in enumerate(adata.var_names.astype(str))}
    hvg_up = [str(g).upper() for g in hvg]
    gene_pos_in_query = [var_to_idx.get(g, -1) for g in hvg_up]
    missing = sum(1 for p in gene_pos_in_query if p < 0)

    X = _sparse_X(adata)
    n_cells = X.shape[0]
    n_hvg = len(hvg)
    d_model = int(cfg.get("d_model", 128))

    # Project each HVG's encoder embedding to a 42-axis aggregate by
    # averaging the per-position embeddings of each axis's core_genes.
    table = _axis_meta_table()
    axis_ids = list(table["axis_id"])
    hvg_idx_map = {g: i for i, g in enumerate(hvg_up)}
    axis_to_hvg_positions: List[List[int]] = []
    for _, row in table.iterrows():
        positions: List[int] = []
        for g in row["core_genes"]:
            i = hvg_idx_map.get(str(g).upper())
            if i is not None:
                positions.append(int(i))
        axis_to_hvg_positions.append(positions)

    out = np.zeros((n_cells, len(axis_ids)), dtype=np.float32)
    cell_emb_full = np.zeros((n_cells, d_model), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_cells, chunk_size):
            stop = min(n_cells, start + chunk_size)
            Xc = X[start:stop].toarray().astype(np.float32)
            x_aligned = np.zeros((stop - start, n_hvg), dtype=np.float32)
            for j, p in enumerate(gene_pos_in_query):
                if p >= 0:
                    x_aligned[:, j] = Xc[:, p]
            x_t = torch.from_numpy(np.clip(x_aligned, 0.0, 20.0)).to(chosen)
            mask_t = torch.zeros_like(x_t)
            # Run the encoder; we need the post-encoder hidden states,
            # not the value-prediction head. Replicate the forward up
            # to the encoder output so we get a (B, n_hvg, d_model)
            # hidden tensor.
            gene_ids = torch.arange(n_hvg, device=chosen).unsqueeze(0).expand(stop - start, -1)
            g_emb = model.gene_embed(gene_ids)
            v_emb = model.value_proj(x_t.unsqueeze(-1))
            pos = model.pos_embed[:, :n_hvg, :]
            h = g_emb + v_emb + pos
            m = model.mask_embed.view(1, 1, -1).expand_as(h)
            h = torch.where(mask_t.unsqueeze(-1) > 0, m, h)
            h_enc = model.encoder(h)  # (B, n_hvg, d_model)
            # Cell embedding = mean across HVG positions
            cell_emb = h_enc.mean(dim=1)
            cell_emb_full[start:stop] = cell_emb.cpu().numpy()
            # Per-axis score = L2 norm of the mean embedding of axis genes
            for j, positions in enumerate(axis_to_hvg_positions):
                if not positions:
                    continue
                axis_emb = h_enc[:, positions, :].mean(dim=1)  # (B, d_model)
                out[start:stop, j] = axis_emb.norm(dim=-1).cpu().numpy()

    return out, axis_ids, {
        "fallback": False,
        "encoder": "cartigsfm scGPT-style pretrained transformer",
        "checkpoint_path": str(checkpoint_path),
        "d_model": int(d_model),
        "n_layers": int(cfg.get("n_layers", 0)),
        "n_heads": int(cfg.get("n_heads", 0)),
        "n_hvg": int(n_hvg),
        "n_hvg_missing_in_query": int(missing),
        "cell_embedding_dim": int(d_model),
        "real_scgpt_human_weights": False,
        "note": (
            "Pretrained on acc.h5ad with cartigsfm.scgpt_pretrain. "
            + str(missing) + " of " + str(n_hvg) + " pretraining HVGs missing in query."
        ),
        # Cell-level d_model embedding kept available via the second
        # return value when callers want it. We do NOT keep it inside
        # meta (it can be N x 256 float32, tens-to-hundreds of MB).
        "_cell_embedding_array": cell_emb_full,
    }


def build_gsfm_per_cell_similarity(
    adata,
    dictionary=None,
    top_n_markers: int = 50,
    chunk_size: int = 5000,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Per-cell GSFM axis similarity: Jaccard(top-N markers, axis panel)."""
    dictionary = dictionary or load_cartilage_dictionary_v1()
    table = _axis_meta_table()
    axis_ids = list(table["axis_id"])
    panels = _axis_panels()
    n_axes = len(axis_ids)
    n_genes = adata.n_vars
    gene_to_idx = {str(g).upper(): i for i, g in enumerate(adata.var_names.astype(str))}

    X = _sparse_X(adata)
    n = X.shape[0]
    out = np.zeros((n, n_axes), dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        Xc = X[start:stop].toarray().astype(np.float32)
        for k in range(stop - start):
            row = Xc[k]
            if (row > 0).sum() == 0:
                continue
            top_idx = np.argpartition(-row, min(top_n_markers, row.size - 1))[:top_n_markers]
            top_set = set(int(i) for i in top_idx)
            cell_markers = {str(adata.var_names[i]).upper() for i in top_set}
            for j, ax_id in enumerate(axis_ids):
                panel = set(panels.get(ax_id, []))
                if not panel or not cell_markers:
                    out[start + k, j] = 0.0
                    continue
                inter = len(panel & cell_markers)
                union = len(panel | cell_markers)
                out[start + k, j] = inter / union if union else 0.0
    return out, axis_ids


# ---------------------------------------------------------------------------
# 3. Sample-stratified split
# ---------------------------------------------------------------------------

def split_by_sample(
    sample_groups: Sequence[str],
    train_frac: float = 0.8,
    seed: int = 0,
    min_cells_per_sample: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified-by-sample cell-level split. No cell-level leakage.

    Returns (train_idx, val_idx) as integer numpy arrays.
    """
    rng = np.random.default_rng(int(seed))
    samples = sorted(set(sample_groups))
    rng.shuffle(samples)
    train_samples: List[str] = []
    val_samples: List[str] = []
    sample_to_count = Counter(sample_groups)
    for s in samples:
        if sample_to_count.get(s, 0) < min_cells_per_sample:
            train_samples.append(s)
            continue
        if len(train_samples) < int(len(samples) * train_frac):
            train_samples.append(s)
        else:
            val_samples.append(s)
    train_set = set(train_samples)
    val_set = set(val_samples)
    train_idx = np.array([i for i, s in enumerate(sample_groups) if s in train_set], dtype=np.int64)
    val_idx = np.array([i for i, s in enumerate(sample_groups) if s in val_set], dtype=np.int64)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# 4. Fusion head (small MLP)
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def train_fusion_head(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_classes: int,
    hidden: int = 128,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    chosen = prefer_device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    X_train = torch.from_numpy(np.asarray(features[train_idx], dtype=np.float32)).to(chosen)
    y_train = torch.from_numpy(np.asarray(labels[train_idx], dtype=np.int64)).to(chosen)
    X_val = torch.from_numpy(np.asarray(features[val_idx], dtype=np.float32)).to(chosen)
    y_val = torch.from_numpy(np.asarray(labels[val_idx], dtype=np.int64)).to(chosen)
    in_dim = features.shape[1]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(int(in_dim), int(hidden)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(int(hidden), int(hidden)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(int(hidden), int(n_classes)),
            )

        def forward(self, x):
            return self.net(x)

    model = Head().to(chosen)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
    history = []
    n_train = X_train.shape[0]
    for ep in range(int(epochs)):
        model.train()
        perm = torch.randperm(n_train, device=chosen)
        loss_sum = 0.0
        n_batch = 0
        for start in range(0, n_train, int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            logits = model(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
            n_batch += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val)
            v_loss = float(F.cross_entropy(v_logits, y_val).cpu())
            v_pred = v_logits.argmax(dim=1)
            v_acc = float((v_pred == y_val).float().mean().cpu())
        history.append({"epoch": ep + 1, "train_loss": loss_sum / max(1, n_batch), "val_loss": v_loss, "val_acc": v_acc})
    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {
        "state_dict": final_state,
        "in_dim": int(in_dim),
        "n_classes": int(n_classes),
        "hidden": int(hidden),
        "device": str(chosen),
        "history": history,
    }


def head_predict(head: Dict[str, Any], features: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    import torch
    import torch.nn as nn
    chosen = prefer_device(device)

    class Head(nn.Module):
        def __init__(self, in_dim, hidden, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(int(in_dim), int(hidden)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(int(hidden), int(hidden)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(int(hidden), int(n_classes)),
            )

        def forward(self, x):
            return self.net(x)

    model = Head(head["in_dim"], head["hidden"], head["n_classes"]).to(chosen)
    model.load_state_dict(head["state_dict"])
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(chosen)
        out = model(X)
        return out.argmax(dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# 5. Six metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    features_cartigm: np.ndarray,
    axis_ids: Sequence[str],
    class_names: Sequence[str],
    evidence_threshold: float = 0.05,
) -> Dict[str, float]:
    """Six metrics: accuracy, macro-F1, balanced accuracy, top-axis
    consistency, evidence citation rate, hallucination rate.

    The 42-axis CartiGM features are restricted to the cell_subtype
    layer (10 axes) for the top-axis consistency and evidence citation
    comparison, since the curated label space has 10 classes that map
    1:1 to the cell_subtype layer axes.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    pred_labels = np.asarray(pred_labels).astype(int)
    true_labels = np.asarray(true_labels).astype(int)
    acc = float(accuracy_score(true_labels, pred_labels))
    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(true_labels, pred_labels))

    table = _axis_meta_table()
    cs_mask = (table["layer"].astype(str).values == "cell_subtype")
    cs_table = table[cs_mask].reset_index(drop=True)
    name_to_cscol = {row["name_en"]: i for i, row in cs_table.iterrows()}
    keep_cols = [name_to_cscol.get(cn, -1) for cn in class_names]
    if any(k < 0 for k in keep_cols):
        keep_cols = list(range(len(class_names)))
    cs_feat = np.asarray(features_cartigm, dtype=np.float32)[:, keep_cols]
    cartigm_top = cs_feat.argmax(axis=1)
    pred_class_name = np.array([class_names[int(p)] for p in pred_labels])
    cartigm_class_name = np.array([class_names[int(c)] for c in cartigm_top])
    top_axis_consistency = float((pred_class_name == cartigm_class_name).mean())

    pred_score = cs_feat[np.arange(len(pred_labels)), pred_labels]
    cited = pred_score > float(evidence_threshold)
    evidence_citation = float(cited.mean())
    hallucination = float(1.0 - evidence_citation)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "top_axis_consistency": round(top_axis_consistency, 4),
        "evidence_citation_rate": round(evidence_citation, 4),
        "hallucination_rate": round(hallucination, 4),
    }


# ---------------------------------------------------------------------------
# 6. Six-config fusion ablation
# ---------------------------------------------------------------------------

CONFIGS = [
    ("cartigm_only",     ["cartigm"]),
    ("scgpt_only",       ["scgpt"]),
    ("gsfm_only",        ["gsfm"]),
    ("cartigm_scgpt",    ["cartigm", "scgpt"]),
    ("cartigm_gsfm",     ["cartigm", "gsfm"]),
    ("full_fusion",      ["cartigm", "scgpt", "gsfm"]),
]


def _stack(features: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    parts = [np.asarray(features[k], dtype=np.float32) for k in keys]
    return np.concatenate(parts, axis=1)


def run_six_config_ablation(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    sample_groups: Sequence[str],
    axis_ids: Sequence[str],
    class_names: Sequence[str],
    outdir: str,
    device: Optional[str] = None,
    epochs: int = 80,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    train_idx, val_idx = split_by_sample(sample_groups, train_frac=0.8, seed=seed)
    n_classes = len(class_names)
    rows = []
    heads: Dict[str, Dict[str, Any]] = {}
    for name, keys in CONFIGS:
        feats = _stack(features, keys)
        head = train_fusion_head(
            feats, labels, train_idx, val_idx,
            n_classes=n_classes, epochs=epochs, device=device, seed=seed,
        )
        heads[name] = head
        val_pred = head_predict(head, feats[val_idx], device=device)
        m = compute_metrics(val_pred, labels[val_idx], features["cartigm"][val_idx], axis_ids, class_names)
        m["config"] = name
        m["features"] = "+".join(keys)
        m["n_train_cells"] = int(len(train_idx))
        m["n_val_cells"] = int(len(val_idx))
        m["in_dim"] = int(feats.shape[1])
        m["val_final_loss"] = round(float(head["history"][-1]["val_loss"]), 4)
        m["val_final_acc"] = round(float(head["history"][-1]["val_acc"]), 4)
        rows.append(m)
    df = pd.DataFrame(rows)[[
        "config", "features", "in_dim", "n_train_cells", "n_val_cells",
        "accuracy", "macro_f1", "balanced_accuracy", "top_axis_consistency",
        "evidence_citation_rate", "hallucination_rate",
        "val_final_loss", "val_final_acc",
    ]]
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir_p / "ablation_metrics.tsv", sep="\t", index=False)
    return df, heads


# ---------------------------------------------------------------------------
# 7. EBR external validation
# ---------------------------------------------------------------------------

def validate_on_ebr(
    features: Dict[str, np.ndarray],
    heads: Dict[str, Dict[str, Any]],
    class_names: Sequence[str],
    axis_ids: Sequence[str],
    cluster_labels: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all 6 heads on EBR, return per-config metrics and per-cell preds."""
    n = features["cartigm"].shape[0]
    if cluster_labels is None:
        cluster_labels = np.zeros(n, dtype=int)
    pred_rows = []
    metric_rows = []
    for name, keys in CONFIGS:
        feats = _stack(features, keys)
        preds = head_predict(heads[name], feats, device=device)
        pred_class_name = np.array([class_names[int(p)] for p in preds])
        per_cell = pd.DataFrame({
            "config": name,
            "cell_index": np.arange(n),
            "cluster": cluster_labels.astype(int),
            "predicted_class_index": preds.astype(int),
            "predicted_class": pred_class_name,
        })
        pred_rows.append(per_cell)
        table = _axis_meta_table()
        cs_mask = (table["layer"].astype(str).values == "cell_subtype")
        cs_table = table[cs_mask].reset_index(drop=True)
        name_to_cscol = {row["name_en"]: i for i, row in cs_table.iterrows()}
        keep_cols = [name_to_cscol.get(cn, -1) for cn in class_names]
        if any(k < 0 for k in keep_cols):
            keep_cols = list(range(len(class_names)))
        cs_feat = np.asarray(features["cartigm"], dtype=np.float32)[:, keep_cols]
        pred_score = cs_feat[np.arange(n), preds]
        cited_ebr = float((pred_score > 0.05).mean())
        halluc_ebr = 1.0 - cited_ebr
        if cluster_labels is not None and len(set(cluster_labels.tolist())) > 1:
            cluster_pred = pd.DataFrame({"cluster": cluster_labels, "pred_class": pred_class_name})
            majority = cluster_pred.groupby("cluster")["pred_class"].agg(lambda s: Counter(s).most_common(1)[0][0])
            cluster_consistency = float((majority == class_names[0]).mean())
        else:
            cluster_consistency = float("nan")
        row = {
            "config": name,
            "features": "+".join(keys),
            "n_cells": int(n),
            "n_predicted_classes": int(len(set(pred_class_name.tolist()))),
            "top_predicted_class": Counter(pred_class_name.tolist()).most_common(1)[0][0],
            "top_predicted_class_frac": round(Counter(pred_class_name.tolist()).most_common(1)[0][1] / n, 4),
            "top_axis_consistency_vs_cartigm": round(
                float((pred_class_name == np.array([class_names[int(c)] for c in cs_feat.argmax(axis=1)])).mean()),
                4,
            ),
            "evidence_citation_rate": round(cited_ebr, 4),
            "hallucination_rate": round(halluc_ebr, 4),
            "cluster_majority_self_consistency": round(cluster_consistency, 4) if not math.isnan(cluster_consistency) else None,
        }
        metric_rows.append(row)
    ebr_df = pd.concat(pred_rows, ignore_index=True)
    ebr_metrics = pd.DataFrame(metric_rows)
    return ebr_metrics, ebr_df
