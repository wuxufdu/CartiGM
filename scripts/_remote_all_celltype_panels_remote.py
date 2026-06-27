"""Remote script: one wilcoxon DE pass on EBR.h5ad celltype labels and dump
per-celltype core / panel / anti gene lists as JSON.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scanpy as sc
import anndata as ad

H5 = os.environ.get("EBR_H5", "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad")
OUT = os.environ.get("PANEL_OUT", "/tmp/all_celltype_panels.json")

TOP_K_MARKER = 30
TOP_K_PANEL = 50
TOP_K_ANTI = 30
MIN_PCT_IN = 0.20
MAX_PCT_BG = 0.95
MIN_LOG2FC = 0.0
ADJ_P = 1e-5
ANTI_PCT_REST_MIN = 0.30
ANTI_PCT_IN_MAX = 0.30


def _names_col(rec, group):
    return np.asarray([row[group] for row in rec["names"]])


def _rec_col(rec, name, group):
    obj = rec[name]
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        return np.asarray([row[group] for row in obj])
    return np.asarray(obj[group].values)


def _pts_aligned(rec, key, group, names):
    df = rec[key]
    return df[group].reindex(names).to_numpy(dtype=float)


def main() -> int:
    a = ad.read_h5ad(H5)
    print("loaded shape", a.shape, file=sys.stderr)
    ct = a.obs["celltype"].astype(str)
    print("celltype counts:\n", ct.value_counts(), file=sys.stderr)
    if "log1p_norm" in a.layers:
        a.X = a.layers["log1p_norm"]
        print("using layers['log1p_norm']", file=sys.stderr)
    a.obs["celltype"] = ct

    sc.tl.rank_genes_groups(
        a, groupby="celltype", method="wilcoxon",
        n_genes=a.shape[1], pts=True, use_raw=False,
    )
    rec = a.uns["rank_genes_groups"]
    groups = list(rec["names"].dtype.names)
    print("groups:", groups, file=sys.stderr)

    payload = {"groups": groups, "method": "wilcoxon",
               "layer": "log1p_norm", "by_group": {}}
    counts = ct.value_counts().to_dict()
    for g in groups:
        names = _names_col(rec, g)
        scores = _rec_col(rec, "scores", g)
        log2fc = _rec_col(rec, "logfoldchanges", g)
        pvals_adj = _rec_col(rec, "pvals_adj", g)
        pts = _pts_aligned(rec, "pts", g, names)
        pts_rest = _pts_aligned(rec, "pts_rest", g, names)

        ok = ((pts >= MIN_PCT_IN)
              & (pts_rest <= MAX_PCT_BG)
              & (log2fc > MIN_LOG2FC)
              & (pvals_adj < ADJ_P))
        keep = np.where(ok)[0]
        core_idx = keep[:TOP_K_MARKER]
        panel_idx = keep[:TOP_K_PANEL]

        def pack(idx):
            return [{
                "gene": str(names[i]),
                "score": float(round(scores[i], 3)),
                "log2fc": float(round(log2fc[i], 3)),
                "pct_in": float(round(pts[i], 4)),
                "pct_rest": float(round(pts_rest[i], 4)),
                "pvals_adj": float(pvals_adj[i]),
            } for i in idx]

        order = np.argsort(log2fc)
        anti = []
        for j in order:
            if log2fc[j] >= 0:
                break
            if pts_rest[j] < ANTI_PCT_REST_MIN:
                continue
            if pts[j] > ANTI_PCT_IN_MAX:
                continue
            if pvals_adj[j] >= ADJ_P:
                continue
            anti.append({
                "gene": str(names[j]),
                "log2fc": float(round(log2fc[j], 3)),
                "pct_in": float(round(pts[j], 4)),
                "pct_rest": float(round(pts_rest[j], 4)),
                "pvals_adj": float(pvals_adj[j]),
            })
            if len(anti) >= TOP_K_ANTI:
                break

        payload["by_group"][g] = {
            "n_target": int(counts.get(g, 0)),
            "n_other": int(a.n_obs - counts.get(g, 0)),
            "core_genes": pack(core_idx),
            "panel_genes_full": pack(panel_idx),
            "anti_genes": anti,
        }
        print(g, "core", len(core_idx), "anti", len(anti), file=sys.stderr)

    with open(OUT, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", OUT, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
