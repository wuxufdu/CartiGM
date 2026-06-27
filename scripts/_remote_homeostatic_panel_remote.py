"""This file is shipped to the remote host and executed there with the
squidp311 conda env to rebuild the Homeostatic_Chondrocytes panel using the
celltype labels in EBR.h5ad.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scanpy as sc
import anndata as ad

H5 = os.environ.get("EBR_H5", "/home/wuxu/jupyter/MJ/newh5ad/EBL/EBR.h5ad")
OUT = os.environ.get("PANEL_OUT", "/tmp/homeostatic_panel.json")
TARGET = "Homeostatic_Chondrocytes"
TOP_K_MARKER = 30
TOP_K_PANEL = 50
TOP_K_ANTI = 30
MIN_PCT_IN = 0.20
MAX_PCT_BG = 0.95


def _col(rec, name, group):
    """Pull one column from rank_genes_groups output.

    For 'names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj' the entry is
    a numpy recarray with one field per group. For 'pts' / 'pts_rest' newer
    scanpy returns a DataFrame indexed by gene with one column per group.
    """
    obj = rec[name]
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        return np.asarray([row[group] for row in obj])
    # DataFrame-like (pts / pts_rest)
    return np.asarray(obj[group].values)


def main() -> int:
    a = ad.read_h5ad(H5)
    print("loaded shape", a.shape, file=sys.stderr)
    ct = a.obs["celltype"].astype(str)
    print("celltype counts:", file=sys.stderr)
    print(ct.value_counts(), file=sys.stderr)

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
    assert TARGET in groups, f"{TARGET} missing"

    names = _col(rec, "names", TARGET)
    scores = _col(rec, "scores", TARGET)
    log2fc = _col(rec, "logfoldchanges", TARGET)
    pvals_adj = _col(rec, "pvals_adj", TARGET)
    # 'pts' / 'pts_rest' are DataFrames keyed by gene name; reindex to match
    # the per-group ordered 'names' array.
    pts_df = rec["pts"]
    pts_rest_df = rec["pts_rest"]
    pts = pts_df[TARGET].reindex(names).to_numpy(dtype=float)
    pts_rest = pts_rest_df[TARGET].reindex(names).to_numpy(dtype=float)

    ok = (pts >= MIN_PCT_IN) & (pts_rest <= MAX_PCT_BG) & (log2fc > 0) & (pvals_adj < 1e-5)
    keep_idx = np.where(ok)[0]
    core_idx = keep_idx[:TOP_K_MARKER]
    panel_idx = keep_idx[:TOP_K_PANEL]

    def pack(idx):
        out = []
        for i in idx:
            out.append({
                "gene": str(names[i]),
                "score": float(round(scores[i], 3)),
                "log2fc": float(round(log2fc[i], 3)),
                "pct_in": float(round(pts[i], 4)),
                "pct_rest": float(round(pts_rest[i], 4)),
                "pvals_adj": float(pvals_adj[i]),
            })
        return out

    core_genes = pack(core_idx)
    panel_genes_full = pack(panel_idx)

    # anti_genes: most depleted in Homeostatic
    order = np.argsort(log2fc)
    anti_keep = []
    for j in order:
        if log2fc[j] >= 0:
            break
        if pts_rest[j] < 0.30:
            continue
        if pts[j] > 0.30:
            continue
        if pvals_adj[j] >= 1e-5:
            continue
        anti_keep.append({
            "gene": str(names[j]),
            "log2fc": float(round(log2fc[j], 3)),
            "pct_in": float(round(pts[j], 4)),
            "pct_rest": float(round(pts_rest[j], 4)),
            "pvals_adj": float(pvals_adj[j]),
        })
        if len(anti_keep) >= TOP_K_ANTI:
            break

    payload = {
        "axis_id": "cell_subtype::Homeostatic_Chondrocytes",
        "n_target": int((ct == TARGET).sum()),
        "n_other": int((ct != TARGET).sum()),
        "groupby": "celltype",
        "method": "wilcoxon",
        "layer": "log1p_norm",
        "core_genes": core_genes,
        "panel_genes_full": panel_genes_full,
        "anti_genes": anti_keep,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", OUT, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
