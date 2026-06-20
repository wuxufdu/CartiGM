"""P4 independent single-cell validation helpers."""
from __future__ import annotations

import re
from typing import Any, Dict, List
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .assets import load_cartilage_dictionary_v1
from .dictionary import load_alias_map
from .projection import project_dictionary_v1_bulk


DEFAULT_KEY_MARKERS = [
    "SOX9", "ACAN", "COL2A1", "COL9A1", "COL11A1",
    "ELN", "FBLN5", "LOX", "CNMD", "TNMD",
    "MGP", "TIMP3", "TNFRSF11B", "ANKH", "ENPP1", "FRZB",
    "COL10A1", "RUNX2", "IBSP", "SPP1",
    "MMP13", "ADAMTS5", "IL1B", "TNF", "VEGFA",
]


# ---------------------------------------------------------------------------
# Obs-column auto-detection (sample / tissue / cluster / celltype)
# ---------------------------------------------------------------------------

_SAMPLE_HINTS = ("sample", "sample_id", "orig.ident", "donor", "patient", "batch_id")
_TISSUE_HINTS = ("tissue", "batch", "cartilage", "region", "organ", "anatomical_site", "anatomy")
_CLUSTER_HINTS = ("cluster", "leiden", "louvain", "seurat_clusters", "id_cluster")
_CELLTYPE_HINTS = ("celltype", "cell_type", "annotation", "cell_class", "celltype_pred", "celltypist")

_TISSUE_KEYWORDS = {
    "ear", "rib", "nose", "knee", "hip", "ankle", "wrist",
    "articular", "hyaline", "elastic", "fibro", "growth_plate",
    "trachea", "bronchus", "meniscus", "ivd", "synovium",
    "control", "oa", "normal", "healthy", "patient",
}
_CELLTYPE_KEYWORDS = {
    "esc", "hc", "fc", "fcp", "prolif", "regc", "reg",
    "chondrocyte", "fibrochondrocyte", "mesenchymal", "stromal",
    "homeostatic", "remodeling", "inflammatory", "maturation",
    "fibro", "matrix", "plasticity", "interface", "prg4",
}


def _cardinality(series) -> int:
    try:
        return int(series.astype(str).nunique())
    except Exception:
        return 0


def _has_any_token(values, keywords) -> int:
    """Count how many unique values in ``values`` contain any of ``keywords``."""
    seen = set()
    for v in values:
        s = str(v).lower()
        for kw in keywords:
            if kw in s:
                seen.add(v)
                break
    return len(seen)


def _score_sample(col_name: str, series) -> float:
    s = col_name.lower()
    n_unique = _cardinality(series)
    n_total = max(int(series.shape[0]), 1)
    if n_unique < 2:
        return 0.0
    name_score = 0.0
    for hint in _SAMPLE_HINTS:
        if hint in s:
            name_score = max(name_score, 0.9)
    if "batch" in s and "tissue" not in s:
        name_score = max(name_score, 0.55)
    if "barcode" in s or "id" == s:
        name_score = max(name_score, 0.3)
    card_ratio = n_unique / n_total
    if card_ratio > 0.7:
        card_score = 0.7
    elif card_ratio > 0.2:
        card_score = 0.5
    elif n_unique >= 3:
        card_score = 0.3
    else:
        card_score = 0.1
    return round(name_score * 0.6 + card_score * 0.4, 3)


def _score_tissue(col_name: str, series) -> float:
    s = col_name.lower()
    n_unique = _cardinality(series)
    name_score = 0.0
    for hint in _TISSUE_HINTS:
        if hint in s:
            name_score = max(name_score, 0.9)
    if "trait" in s or "group" in s or "condition" in s:
        name_score = max(name_score, 0.5)
    if "cartilage" in s:
        name_score = max(name_score, 0.7)
    if not (2 <= n_unique <= 50):
        card_score = 0.0
    else:
        card_score = 0.8
    uniq = series.astype(str).unique().tolist()[:50]
    kw_hits = _has_any_token(uniq, _TISSUE_KEYWORDS)
    kw_score = min(kw_hits / max(n_unique, 1), 1.0) * 0.4
    return round(name_score * 0.5 + card_score * 0.3 + kw_score, 3)


def _score_cluster(col_name: str, series) -> float:
    s = col_name.lower()
    n_unique = _cardinality(series)
    name_score = 0.0
    for hint in _CLUSTER_HINTS:
        if hint in s:
            name_score = max(name_score, 0.95)
    if "annot" in s and "harmony" in s:
        name_score = max(name_score, 0.8)
    if 3 <= n_unique <= 200:
        card_score = 0.85
    elif n_unique > 200:
        card_score = 0.4
    else:
        card_score = 0.0
    return round(name_score * 0.6 + card_score * 0.4, 3)


def _score_celltype(col_name: str, series) -> float:
    s = col_name.lower()
    n_unique = _cardinality(series)
    name_score = 0.0
    for hint in _CELLTYPE_HINTS:
        if hint in s:
            name_score = max(name_score, 0.9)
    if "chondrocyte" in s or "general" in s:
        name_score = max(name_score, 0.7)
    if not (2 <= n_unique <= 200):
        card_score = 0.0
    else:
        card_score = 0.6
    uniq = series.astype(str).unique().tolist()[:200]
    kw_hits = _has_any_token(uniq, _CELLTYPE_KEYWORDS)
    kw_score = min(kw_hits / max(n_unique, 1), 1.0) * 0.4
    return round(name_score * 0.5 + card_score * 0.2 + kw_score, 3)


def auto_detect_obs_columns(adata) -> Dict[str, Any]:
    """Inspect ``adata.obs`` and propose the best sample / tissue / cluster / celltype columns.

    Returns a dict with one entry per role; each entry lists ``best`` (None if
    no candidate clears the 0.30 floor), ``confidence`` (0..1), and
    ``alternatives`` (other columns with their scores, sorted descending).
    """
    obs = adata.obs
    summary: Dict[str, Any] = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "obs_columns": [str(c) for c in obs.columns],
        "var_index_name": str(adata.var.index.name) if adata.var.index.name else None,
        "var_names_are_gene_symbols": bool(
            adata.var_names.astype(str).str.match(r"^[A-Z][A-Z0-9._-]{0,15}$").mean() > 0.5
        ) if adata.n_vars else False,
    }
    scorers = {
        "sample_col": _score_sample,
        "tissue_col": _score_tissue,
        "cluster_col": _score_cluster,
        "celltype_col": _score_celltype,
    }
    for role, scorer in scorers.items():
        candidates: List[Dict[str, Any]] = []
        for col in obs.columns:
            score = scorer(str(col), obs[col])
            if score > 0.0:
                n_unique = _cardinality(obs[col])
                candidates.append({
                    "column": str(col),
                    "score": score,
                    "n_unique": n_unique,
                })
        candidates.sort(key=lambda r: r["score"], reverse=True)
        best = candidates[0] if candidates and candidates[0]["score"] >= 0.30 else None
        summary[role] = {
            "best": best["column"] if best else None,
            "confidence": best["score"] if best else 0.0,
            "alternatives": candidates[:5],
        }
    return summary


def _get_matrix(adata, layer: str | None = None):
    if layer:
        if layer not in adata.layers:
            raise ValueError(f"layer {layer!r} not found in h5ad layers")
        return adata.layers[layer]
    return adata.X


def _read_h5ad(path: str | Path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "p4-project --h5ad requires anndata. Install with: pip install anndata"
        ) from exc
    return ad.read_h5ad(path)


def _filter_cells(adata, celltype_col: str | None, celltype_regex: str | None):
    if not celltype_col or not celltype_regex:
        return adata
    if celltype_col not in adata.obs:
        raise KeyError(f"celltype column {celltype_col!r} not found in adata.obs")
    mask = adata.obs[celltype_col].astype(str).str.contains(celltype_regex, case=False, na=False, regex=True)
    return adata[mask].copy()


def pseudobulk_from_h5ad(
    h5ad: str | Path,
    sample_col: str,
    tissue_col: str,
    cluster_col: str,
    celltype_col: str | None = None,
    celltype_regex: str | None = "chondro|cartilage",
    layer: str | None = None,
    min_cells: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample x cluster pseudobulk from h5ad.

    Returns ``(pseudobulk, metadata)`` where pseudobulk is genes x groups with
    a leading ``gene`` column, and metadata is indexed by group id.
    """
    adata = _read_h5ad(h5ad)
    required = [sample_col, tissue_col, cluster_col]
    missing = [col for col in required if col not in adata.obs]
    if missing:
        raise KeyError(f"missing required adata.obs columns: {missing}")
    adata = _filter_cells(adata, celltype_col, celltype_regex)
    if adata.n_obs == 0:
        raise ValueError("no cells left after celltype filtering")
    obs = adata.obs[[sample_col, tissue_col, cluster_col]].astype(str).copy()
    obs.columns = ["sample", "tissue", "cluster"]
    group_id = obs[["sample", "tissue", "cluster"]].agg("|".join, axis=1)
    counts = group_id.value_counts()
    keep_groups = counts[counts >= min_cells].index
    keep = group_id.isin(keep_groups).to_numpy()
    adata = adata[keep].copy()
    obs = obs.loc[adata.obs_names].copy()
    group_id = obs[["sample", "tissue", "cluster"]].agg("|".join, axis=1)
    if len(group_id) == 0:
        raise ValueError(f"no sample-cluster groups with at least {min_cells} cells")
    codes, labels = pd.factorize(group_id, sort=True)
    n_groups = len(labels)
    X = _get_matrix(adata, layer)
    try:
        from scipy import sparse
        if sparse.issparse(X):
            design = sparse.csr_matrix(
                (np.ones(len(codes)), (codes, np.arange(len(codes)))),
                shape=(n_groups, len(codes)),
            )
            sums = design @ X
            means = sums.multiply(1 / np.bincount(codes)[:, None]).toarray()
        else:
            means = np.vstack([np.asarray(X[codes == i]).mean(axis=0) for i in range(n_groups)])
    except ImportError:
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        means = np.vstack([X_arr[codes == i].mean(axis=0) for i in range(n_groups)])
    pb = pd.DataFrame(means.T, index=[str(g).upper() for g in adata.var_names], columns=labels)
    pb = pb.groupby(level=0).mean(numeric_only=True)
    pb.insert(0, "gene", pb.index)
    meta = pd.DataFrame([label.split("|", 2) for label in labels], index=labels, columns=["sample", "tissue", "cluster"])
    meta["n_cells"] = [int((codes == i).sum()) for i in range(n_groups)]
    return pb.reset_index(drop=True), meta


# ---------------------------------------------------------------------------
# Streaming pseudobulk for large h5ads (e.g. integrated atlas 400k+ cells)
# ---------------------------------------------------------------------------

def _autostream_threshold_bytes() -> int:
    """Files larger than this on disk switch to streaming automatically."""
    return 2 * 1024 ** 3


def _resolve_chunk_size(n_genes: int, *, target_dense_bytes: int = 2 * 1024 ** 3,
                         floor: int = 1000, ceiling: int = 20000) -> int:
    """Pick a chunk size so a dense chunk stays under target_dense_bytes."""
    per_cell = max(int(n_genes) * 4, 1)
    auto = max(floor, min(ceiling, target_dense_bytes // per_cell))
    return int(auto)


def pseudobulk_streaming(
    h5ad: str | Path,
    sample_col: str,
    tissue_col: str,
    cluster_col: str,
    celltype_col: str | None = None,
    celltype_regex: str | None = "chondro|cartilage",
    min_cells: int = 10,
    chunk_size: int | None = None,
    layer: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Streaming pseudobulk for large h5ads.

    Iterates ``adata`` in ``backed='r'`` mode, chunk by chunk, and accumulates
    per-(sample, tissue, cluster) sums into a sparse ``(n_groups, n_genes)``
    matrix without ever materialising the full expression matrix. Designed
    for atlas-scale inputs (300k-500k cells x 40k genes sparse) where
    loading everything would OOM a 64 GB machine.

    Returns the same ``(pseudobulk, metadata)`` shape as
    :func:`pseudobulk_from_h5ad`.
    """
    try:
        import anndata as ad
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "pseudobulk_streaming requires anndata + scipy."
        ) from exc

    adata = ad.read_h5ad(str(h5ad), backed="r")
    n_cells = int(adata.n_obs)
    n_genes = int(adata.n_vars)
    for col in (sample_col, tissue_col, cluster_col):
        if col not in adata.obs:
            raise KeyError(f"missing adata.obs column: {col!r}")

    if celltype_col and celltype_regex:
        if celltype_col not in adata.obs:
            raise KeyError(f"celltype column {celltype_col!r} not found in adata.obs")
        mask = adata.obs[celltype_col].astype(str).str.contains(
            celltype_regex, case=False, na=False, regex=True,
        ).to_numpy()
    else:
        mask = np.ones(n_cells, dtype=bool)

    if not mask.any():
        raise ValueError("no cells left after celltype filtering")

    group_id_full = (
        adata.obs[sample_col].astype(str)
        + "|"
        + adata.obs[tissue_col].astype(str)
        + "|"
        + adata.obs[cluster_col].astype(str)
    ).to_numpy()

    unique_groups = np.unique(group_id_full[mask])
    n_groups = int(len(unique_groups))
    group_to_code = {g: i for i, g in enumerate(unique_groups.tolist())}
    group_codes_full = np.fromiter(
        (group_to_code[g] for g in group_id_full.tolist()),
        count=n_cells, dtype=np.int32,
    )

    sums = sparse.csr_matrix((n_groups, n_genes), dtype=np.float32)
    counts = np.zeros(n_groups, dtype=np.int64)

    cs = int(chunk_size) if chunk_size else _resolve_chunk_size(n_genes)
    X_src = adata.layers[layer] if layer else adata.X

    for start in range(0, n_cells, cs):
        end = min(start + cs, n_cells)
        chunk_mask = mask[start:end]
        if not chunk_mask.any():
            continue
        chunk_codes = group_codes_full[start:end][chunk_mask]
        try:
            chunk_X = X_src[start:end][chunk_mask]
        except Exception:
            chunk_X = X_src[start:end]
            chunk_X = chunk_X[chunk_mask]
        if not sparse.issparse(chunk_X):
            chunk_X = sparse.csr_matrix(chunk_X)
        design = sparse.csr_matrix(
            (np.ones(len(chunk_codes), dtype=np.float32),
             (chunk_codes, np.arange(len(chunk_codes)))),
            shape=(n_groups, len(chunk_codes)),
        )
        sums = sums + design @ chunk_X
        np.add.at(counts, chunk_codes, 1)

    keep_mask = counts >= int(min_cells)
    if not keep_mask.any():
        raise ValueError(
            f"no sample-cluster groups with at least {min_cells} cells "
            f"(largest group has {int(counts.max()) if counts.size else 0} cells)"
        )
    sums_kept = sums[keep_mask]
    counts_kept = counts[keep_mask]
    groups_kept = unique_groups[keep_mask]
    means = sums_kept.multiply(1.0 / counts_kept[:, None]).toarray()

    var_index_upper = pd.Index([str(g).upper() for g in adata.var_names.astype(str)])
    pb = pd.DataFrame(means.T, index=var_index_upper, columns=groups_kept)
    pb = pb.groupby(level=0).mean(numeric_only=True)
    pb.insert(0, "gene", pb.index)
    meta = pd.DataFrame(
        [g.split("|", 2) for g in groups_kept.tolist()],
        index=groups_kept, columns=["sample", "tissue", "cluster"],
    )
    meta["n_cells"] = counts_kept.astype(int)
    return pb.reset_index(drop=True), meta


def _write_tissue_comparisons(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    sample_ids = meta.index.astype(str)
    sample_meta = meta.copy().reset_index(drop=True)
    sample_meta["sample"] = sample_ids
    merged = scores.merge(sample_meta[["sample", "tissue"]], on="sample", how="left")
    rows = []
    tissues = sorted(t for t in merged["tissue"].dropna().unique())
    for layer, layer_df in merged.groupby("layer"):
        for axis_id, axis_df in layer_df.groupby("axis_id"):
            means = axis_df.groupby("tissue")["score"].mean().to_dict()
            row = {"layer": layer, "axis_id": axis_id}
            row.update({f"mean_{tissue}": means.get(tissue, np.nan) for tissue in tissues})
            if means:
                top_tissue = max(means, key=means.get)
                row["top_tissue"] = top_tissue
                row["top_score"] = means[top_tissue]
            rows.append(row)
    return pd.DataFrame(rows)


def _write_marker_validation(pseudobulk: pd.DataFrame, meta: pd.DataFrame, markers: Iterable[str]) -> pd.DataFrame:
    expr = pseudobulk.set_index("gene")
    rows = []
    for gene in markers:
        gene_u = gene.upper()
        if gene_u not in expr.index:
            rows.append({"gene": gene_u, "present": False})
            continue
        values = expr.loc[gene_u]
        for group, value in values.items():
            rows.append({
                "gene": gene_u,
                "present": True,
                "sample": group,
                "tissue": meta.loc[group, "tissue"] if group in meta.index else "",
                "cluster": meta.loc[group, "cluster"] if group in meta.index else "",
                "expression": float(value),
            })
    return pd.DataFrame(rows)


def run_p4_project(
    outdir: str | Path,
    h5ad: str | Path | None = None,
    pseudobulk_tsv: str | Path | None = None,
    meta_tsv: str | Path | None = None,
    sample_col: str = "sample",
    tissue_col: str = "tissue",
    cluster_col: str = "cluster",
    celltype_col: str | None = None,
    celltype_regex: str | None = "chondro|cartilage",
    layer: str | None = None,
    min_cells: int = 10,
    gene_col: str = "gene",
    anti_lambda: float = 0.5,
    streaming: bool | None = None,
    chunk_size: int | None = None,
) -> dict[str, Path]:
    """Run a compact P4 independent-validation projection workflow."""
    out = Path(outdir)
    tsv_dir = out / "tsv"
    docs_dir = out / "docs"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    if h5ad:
        use_streaming = streaming
        if use_streaming is None:
            try:
                size = Path(h5ad).stat().st_size
                use_streaming = size > _autostream_threshold_bytes()
            except OSError:
                use_streaming = False
        if use_streaming:
            pseudobulk, meta = pseudobulk_streaming(
                h5ad,
                sample_col=sample_col,
                tissue_col=tissue_col,
                cluster_col=cluster_col,
                celltype_col=celltype_col,
                celltype_regex=celltype_regex,
                layer=layer,
                min_cells=min_cells,
                chunk_size=chunk_size,
            )
        else:
            pseudobulk, meta = pseudobulk_from_h5ad(
                h5ad,
                sample_col=sample_col,
                tissue_col=tissue_col,
                cluster_col=cluster_col,
                celltype_col=celltype_col,
                celltype_regex=celltype_regex,
                layer=layer,
                min_cells=min_cells,
            )
    elif pseudobulk_tsv and meta_tsv:
        pseudobulk = pd.read_csv(pseudobulk_tsv, sep=None, engine="python")
        meta = pd.read_csv(meta_tsv, sep=None, engine="python", index_col=0)
        if "tissue" not in meta.columns:
            raise KeyError("metadata TSV must include a tissue column")
    else:
        raise ValueError("provide either --h5ad or both --pseudobulk and --meta")
    pseudobulk_path = tsv_dir / "p4_self_sample_cluster_pseudobulk.tsv"
    meta_path = tsv_dir / "p4_self_sample_cluster_meta.tsv"
    pseudobulk.to_csv(pseudobulk_path, sep="\t", index=False)
    meta.to_csv(meta_path, sep="\t")
    dictionary = load_cartilage_dictionary_v1()
    scores = project_dictionary_v1_bulk(
        pseudobulk,
        dictionary,
        gene_col=gene_col,
        anti_lambda=anti_lambda,
        alias_map=load_alias_map(),
    )
    scores_path = tsv_dir / "p4_sample_cluster_three_layer_scores.tsv"
    scores.to_csv(scores_path, sep="\t", index=False)
    top = scores.sort_values(["sample", "layer", "score"], ascending=[True, True, False]).groupby(["sample", "layer"]).head(1)
    top_path = tsv_dir / "p4_sample_cluster_top_assignments.tsv"
    top.to_csv(top_path, sep="\t", index=False)
    comparisons = _write_tissue_comparisons(scores, meta)
    comparisons_path = tsv_dir / "p4_tissue_axis_summary.tsv"
    comparisons.to_csv(comparisons_path, sep="\t", index=False)
    marker_validation = _write_marker_validation(pseudobulk, meta, DEFAULT_KEY_MARKERS)
    marker_path = tsv_dir / "p4_marker_validation_table.tsv"
    marker_validation.to_csv(marker_path, sep="\t", index=False)
    report_path = docs_dir / "P4_INDEPENDENT_VALIDATION_REPORT.md"
    tissues = ", ".join(sorted(meta["tissue"].astype(str).unique()))
    report = [
        "# P4 Independent Validation Report",
        "",
        "## Scope",
        "This report was generated by `cartigsfm p4-project` using the bundled unchanged `cartilage_dictionary_v1`.",
        "",
        "## Input Summary",
        f"- sample-cluster groups: {len(meta)}",
        f"- tissues: {tissues}",
        f"- genes in pseudobulk: {pseudobulk.shape[0]}",
        "",
        "## Outputs",
        "- `p4_sample_cluster_three_layer_scores.tsv`: all 42-axis scores.",
        "- `p4_sample_cluster_top_assignments.tsv`: top axis per layer and sample-cluster.",
        "- `p4_tissue_axis_summary.tsv`: tissue-level mean scores.",
        "- `p4_marker_validation_table.tsv`: key marker expression table.",
        "",
        "## Interpretation Boundary",
        "Use these outputs as independent in-house projection evidence. Do not claim clinical validation, causality, or therapeutic targeting from these scores alone.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "pseudobulk": pseudobulk_path,
        "meta": meta_path,
        "scores": scores_path,
        "top_assignments": top_path,
        "tissue_summary": comparisons_path,
        "marker_validation": marker_path,
        "report": report_path,
    }
