"""Remote DE on acc_new.h5ad using celltype_new (10 cs subtypes)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scanpy as sc
import anndata as ad

H5 = os.environ.get("ACC_H5", "/home/wuxu/jupyter/MJ/newh5ad/acc_new.h5ad")
OUT = os.environ.get("PANEL_OUT", "/tmp/all_celltype_panels_acc.json")
LABEL_COL = os.environ.get("LABEL_COL", "celltype_new")
MAX_PER_GROUP = int(os.environ.get("MAX_PER_GROUP", "5000"))
SEED = int(os.environ.get("SEED", "0"))

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


def _balanced_subsample(adata, key, max_per_group, seed):
    rng = np.random.default_rng(seed)
    labels = adata.obs[key].astype(str).values
    keep_idx = []
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        keep_idx.append(idx)
    keep_idx = np.sort(np.concatenate(keep_idx))
    return adata[keep_idx].copy()


def _ensure_log1p(adata):
    if "log1p_norm" in adata.layers:
        adata.X = adata.layers["log1p_norm"]
        print("layer log1p_norm -> X", file=sys.stderr)
        return
    X = adata.X
    sample = X[:1000].toarray() if hasattr(X, "toarray") else np.asarray(X[:1000])
    mx = float(sample.max()) if sample.size else 0.0
    looks_int = bool(np.all(sample == np.round(sample)))
    print(f"X probe max={mx:.3f} looks_int={looks_int}", file=sys.stderr)
    if mx > 50 or looks_int:
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"]
            print("layer counts -> X", file=sys.stderr)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print("normalize_total + log1p applied", file=sys.stderr)
    else:
        print("assuming X already log1p", file=sys.stderr)


def main() -> int:
    print(f"reading {H5}", file=sys.stderr)
    a = ad.read_h5ad(H5)
    print("loaded shape", a.shape, file=sys.stderr)
    if LABEL_COL not in a.obs.columns:
        print(f"missing label col {LABEL_COL}", file=sys.stderr)
        return 2
    a.obs[LABEL_COL] = a.obs[LABEL_COL].astype(str)
    print("label counts:\n", a.obs[LABEL_COL].value_counts(), file=sys.stderr)

    a = _balanced_subsample(a, LABEL_COL, MAX_PER_GROUP, SEED)
    print("balanced shape", a.shape, file=sys.stderr)

    _ensure_log1p(a)

    a.obs[LABEL_COL] = a.obs[LABEL_COL].astype("category")
    sc.tl.rank_genes_groups(
        a, groupby=LABEL_COL, method="wilcoxon",
        n_genes=a.shape[1], pts=True, use_raw=False,
    )
    rec = a.uns["rank_genes_groups"]
    groups = list(rec["names"].dtype.names)
    print("groups:", groups, file=sys.stderr)

    payload = {
        "source": H5,
        "label_col": LABEL_COL,
        "max_per_group": MAX_PER_GROUP,
        "seed": SEED,
        "groups": groups,
        "method": "wilcoxon",
        "by_group": {},
    }
    counts = a.obs[LABEL_COL].value_counts().to_dict()
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
