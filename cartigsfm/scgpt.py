"""P13: scGPT-style frozen single-cell / cluster expression encoder.

The full scGPT-human foundation model (~400 MB) is not bundled in this
sandbox. This module implements the same surface API in a deterministic
proxy that:

  * Accepts an h5ad path (requires ``anndata``) or a raw genes x samples
    ``pandas.DataFrame``. This is the "expression" branch, distinct from
    the marker-list-only GSFM branch.
  * For each cluster (or sample column) it computes the mean expression of
    every axis's ``core_genes`` in the cluster, producing a 42-dimensional
    feature vector per cluster. This is the "frozen" cluster embedding.
  * Returns a per-cluster, per-axis score table that the CartiAgent LLM
    agent can consume as a tool call.

The module is **frozen** (no training, no fine-tuning). The embedding is
explicitly a deterministic function of the bundled dictionary and the
input expression, so outputs are reproducible.

Public API
----------
scgpt_encode_dataframe(expr_df, gene_col=None, sample_cols=None) -> dict
scgpt_encode_h5ad(h5ad_path, cluster_col="cluster", ...) -> dict
scgpt_encode_cluster(h5ad_path, cluster_col="cluster", ...) -> dict
tool_scgpt_encode(h5ad_path=None, expr_df=None, ...) -> dict
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .assets import load_cartilage_dictionary_v1
from .dictionary import load_alias_map
from .interpret import axis_safety_class
from .projection import _axis_anti_weights, _axis_gene_weights


def _v1_axes() -> List[Dict[str, Any]]:
    dictionary = load_cartilage_dictionary_v1()
    out: List[Dict[str, Any]] = []
    for layer, layer_obj in (dictionary.get("layers") or {}).items():
        for axis in layer_obj.get("axes", []):
            entry = dict(axis)
            entry["layer"] = layer
            out.append(entry)
    return out


def _axis_index() -> Dict[str, Dict[str, Any]]:
    return {a.get("axis_id"): a for a in _v1_axes() if a.get("axis_id")}


def _normalize_expression_index(expr_df: pd.DataFrame) -> pd.DataFrame:
    df = expr_df.copy()
    df.index = [str(g).strip().upper() for g in df.index]
    context_overrides = {
        "TAZ": "WWTR1", "YAP": "YAP1", "PDGFR": "PDGFRA",
        "TGFB": "TGFB1",
    }
    alias_map = load_alias_map()
    df.index = [context_overrides.get(g, alias_map.get(g, g)) for g in df.index]
    df = df.groupby(level=0).mean(numeric_only=True)
    return df.apply(pd.to_numeric, errors="coerce")


def _cluster_axis_embedding(expr_df: pd.DataFrame) -> Dict[str, Any]:
    expr = _normalize_expression_index(expr_df)
    if expr.shape[0] == 0 or expr.shape[1] == 0:
        empty_long = pd.DataFrame(columns=[
            "axis_id", "layer", "cluster", "score",
            "marker_n", "anti_n", "safety_classification",
        ])
        return {
            "axis_scores": empty_long,
            "cluster_embedding": pd.DataFrame(),
            "n_axes_scored": 0,
            "n_clusters": 0,
        }
    rows: List[Dict[str, Any]] = []
    axes = _axis_index()
    for ax_id, axis in axes.items():
        marker_weights = _axis_gene_weights(axis)
        anti_weights = _axis_anti_weights(axis)
        in_m = [g for g in marker_weights if g in expr.index]
        in_a = [g for g in anti_weights if g in expr.index]
        if not in_m:
            continue
        marker_score = expr.loc[in_m].mean(axis=0)
        if in_a:
            anti_score = expr.loc[in_a].mean(axis=0)
        else:
            anti_score = pd.Series(0.0, index=expr.columns)
        score = marker_score - 0.5 * anti_score
        score_values = np.nan_to_num(score.to_numpy(dtype=float), nan=0.0)
        if not np.any(np.abs(score_values) > 0):
            continue
        for cluster, value in score.items():
            rows.append({
                "axis_id": ax_id,
                "layer": axis.get("layer"),
                "cluster": str(cluster),
                "score": round(float(value), 6),
                "marker_n": len(in_m),
                "anti_n": len(in_a),
                "safety_classification": axis_safety_class(axis),
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return {
            "axis_scores": long_df,
            "cluster_embedding": pd.DataFrame(),
            "n_axes_scored": 0,
            "n_clusters": int(expr.shape[1]),
        }
    wide = long_df.pivot_table(
        index="cluster", columns="axis_id", values="score", aggfunc="mean",
    ).fillna(0.0)
    top_row = long_df.sort_values("score", ascending=False).groupby("cluster").head(1)
    top_map = dict(zip(top_row["cluster"], top_row["axis_id"]))
    top_score_map = dict(zip(top_row["cluster"], top_row["score"]))
    wide["top_axis_id"] = pd.Series(top_map)
    wide["top_score"] = pd.Series(top_score_map)
    return {
        "axis_scores": long_df,
        "cluster_embedding": wide,
        "n_axes_scored": int(long_df["axis_id"].nunique()),
        "n_clusters": int(expr.shape[1]),
    }


def scgpt_encode_dataframe(expr_df: pd.DataFrame, *,
                            gene_col: Optional[str] = None,
                            sample_cols: Optional[Iterable[str]] = None,
                            cluster_names: Optional[Iterable[str]] = None
                            ) -> Dict[str, Any]:
    df = expr_df.copy()
    if gene_col and gene_col in df.columns:
        df = df.set_index(gene_col)
    if sample_cols is not None:
        cols = [c for c in sample_cols if c in df.columns]
        if not cols:
            raise ValueError(
                f"none of the requested sample_cols {list(sample_cols)!r} "
                f"are present in the expression frame"
            )
        df = df[cols]
    if cluster_names is not None:
        names = list(cluster_names)
        if len(names) != df.shape[1]:
            raise ValueError(
                f"cluster_names has {len(names)} entries but expression has "
                f"{df.shape[1]} sample columns"
            )
        df = df.copy()
        df.columns = names
    res = _cluster_axis_embedding(df)
    n_genes = int(_normalize_expression_index(df).shape[0])
    return {
        "branch": "scgpt",
        "input_kind": "dataframe",
        "n_genes": n_genes,
        "n_clusters": res["n_clusters"],
        "n_axes_scored": res["n_axes_scored"],
        "axis_scores": res["axis_scores"],
        "cluster_embedding": res["cluster_embedding"],
        "note": (
            "scGPT branch: per-cluster mean core-gene expression embedding. "
            "Frozen feature extractor; no joint training."
        ),
    }


def scgpt_encode_h5ad(h5ad_path: Union[str, Path], *,
                      cluster_col: str = "cluster",
                      celltype_col: Optional[str] = None,
                      celltype_regex: Optional[str] = "chondro|cartilage",
                      layer: Optional[str] = None,
                      min_cells: int = 10,
                      ) -> Dict[str, Any]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "scgpt_encode_h5ad requires anndata. Install with: pip install anndata"
        ) from exc
    adata = ad.read_h5ad(str(h5ad_path))
    if cluster_col not in adata.obs:
        raise KeyError(f"cluster column {cluster_col!r} not found in adata.obs")
    if celltype_col and celltype_regex:
        if celltype_col not in adata.obs:
            raise KeyError(f"celltype column {celltype_col!r} not found in adata.obs")
        mask = adata.obs[celltype_col].astype(str).str.contains(
            celltype_regex, case=False, na=False, regex=True,
        )
        adata = adata[mask].copy()
    if adata.n_obs == 0:
        raise ValueError("no cells left after celltype filtering")
    cluster_labels = adata.obs[cluster_col].astype(str)
    counts = cluster_labels.value_counts()
    keep_clusters = counts[counts >= min_cells].index
    if len(keep_clusters) == 0:
        raise ValueError(f"no clusters with at least {min_cells} cells")
    adata = adata[cluster_labels.isin(keep_clusters)].copy()
    cluster_labels = adata.obs[cluster_col].astype(str)
    codes, labels = pd.factorize(cluster_labels, sort=True)
    X = adata.layers[layer] if layer else adata.X
    try:
        from scipy import sparse
        if sparse.issparse(X):
            design = sparse.csr_matrix(
                (np.ones(len(codes)), (codes, np.arange(len(codes)))),
                shape=(len(labels), len(codes)),
            )
            sums = design @ X
            means = sums.multiply(1 / np.bincount(codes)[:, None]).toarray()
        else:
            means = np.vstack([
                np.asarray(X[codes == i]).mean(axis=0) for i in range(len(labels))
            ])
    except ImportError:
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        means = np.vstack([X_arr[codes == i].mean(axis=0) for i in range(len(labels))])
    pb = pd.DataFrame(
        means.T,
        index=[str(g).upper() for g in adata.var_names],
        columns=[f"cluster::{l}" for l in labels],
    )
    pb = pb.groupby(level=0).mean(numeric_only=True)
    pb = pb.reset_index().rename(columns={"index": "gene"})
    res = scgpt_encode_dataframe(pb, gene_col="gene")
    res["input_kind"] = "h5ad"
    res["h5ad_path"] = str(h5ad_path)
    res["cluster_col"] = cluster_col
    res["min_cells"] = int(min_cells)
    res["n_cells_kept"] = int(adata.n_obs)
    res["cluster_ids"] = [str(l) for l in labels]
    return res


def scgpt_encode_cluster(h5ad_path: Union[str, Path], *,
                          cluster_col: str = "cluster", **kwargs) -> Dict[str, Any]:
    return scgpt_encode_h5ad(h5ad_path, cluster_col=cluster_col, **kwargs)


def tool_scgpt_encode(h5ad_path: Optional[Union[str, Path]] = None,
                      expr_df: Optional[pd.DataFrame] = None,
                      *, cluster_col: str = "cluster",
                      gene_col: Optional[str] = None,
                      sample_cols: Optional[Iterable[str]] = None,
                      top_axes_per_cluster: int = 3,
                      **kwargs) -> Dict[str, Any]:
    if h5ad_path:
        res = scgpt_encode_h5ad(h5ad_path, cluster_col=cluster_col, **kwargs)
    elif expr_df is not None:
        res = scgpt_encode_dataframe(
            expr_df, gene_col=gene_col, sample_cols=sample_cols,
        )
    else:
        return {
            "branch": "scgpt",
            "result": None,
            "note": "provide either h5ad_path or expr_df",
        }
    emb = res.get("cluster_embedding")
    summary: List[Dict[str, Any]] = []
    if isinstance(emb, pd.DataFrame) and not emb.empty and "top_axis_id" in emb.columns:
        for cluster_id, row in emb.iterrows():
            entry = {
                "cluster": str(cluster_id),
                "top_axis_id": str(row.get("top_axis_id", "")),
                "top_score": float(row.get("top_score", 0.0)) if row.get("top_score") is not None else None,
            }
            drop_cols = [c for c in ("top_axis_id", "top_score") if c in emb.columns]
            scores = emb.drop(columns=drop_cols, errors="ignore")
            top_axes = scores.loc[cluster_id].sort_values(ascending=False).head(int(top_axes_per_cluster))
            entry["top_axes"] = [
                {"axis_id": str(ax_id), "score": float(v)}
                for ax_id, v in top_axes.items()
            ]
            summary.append(entry)
    res["per_cluster_summary"] = summary
    return res
