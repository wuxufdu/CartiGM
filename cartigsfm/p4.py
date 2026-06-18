"""P4 independent single-cell validation helpers."""
from __future__ import annotations

import re
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
) -> dict[str, Path]:
    """Run a compact P4 independent-validation projection workflow."""
    out = Path(outdir)
    tsv_dir = out / "tsv"
    docs_dir = out / "docs"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    if h5ad:
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
