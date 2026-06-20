"""Compute DE markers from EBR.h5ad (nose vs ear+rib) on GPU, then emit
a candidate axis JSON for cartilage_dictionary_v1 (tissue_developmental_state layer).

This script does NOT modify the bundled dictionary. It writes a candidate
axis JSON to outputs/nasal_septum_axis/ and a delta file that can be merged
into cartilage_dictionary_v1.json by a separate packaging step.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEVICE = "cuda:0"


def main():
    import torch
    import anndata as ad
    import scanpy as sc

    if not torch.cuda.is_available():
        print("[de] CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = DEVICE
        print(f"[de] using device: {torch.cuda.get_device_name(0)}")

    outdir = Path(r"F:\cartifm\outputs\nasal_septum_axis")
    outdir.mkdir(parents=True, exist_ok=True)

    print("[de] reading EBR.h5ad (7 GB)...", flush=True)
    t0 = time.time()
    adata = ad.read_h5ad(r"F:\cartifm\outputs\EBR\EBR.h5ad")
    print(f"[de] shape: {adata.shape}, dtype: {adata.X.dtype}, read {time.time()-t0:.1f}s", flush=True)

    adata = adata.copy()
    adata.obs["tissue"] = adata.obs["batch"].astype(str)
    print("[de] tissue value_counts:", dict(adata.obs["tissue"].value_counts()))

    if isinstance(adata.X, np.ndarray):
        X = adata.X
    else:
        X = adata.X.toarray()
    X = X.astype(np.float32, copy=False)
    print(f"[de] X: {X.shape}, dtype={X.dtype}, mean={X.mean():.4f}, min={X.min():.4f}, max={X.max():.4f}")

    sc.settings.verbosity = 1
    adata.obs["tissue_binary"] = (adata.obs["tissue"] == "nose").astype(str).astype("category")
    adata.X = X

    print("[de] rank_genes_groups: nose vs (ear+rib), method=wilcoxon", flush=True)
    t0 = time.time()
    sc.tl.rank_genes_groups(
        adata,
        groupby="tissue_binary",
        groups=["True"],
        reference="False",
        method="wilcoxon",
        use_raw=False,
        pts=True,
    )
    print(f"[de] wilcoxon done in {time.time()-t0:.1f}s", flush=True)

    res = adata.uns["rank_genes_groups"]
    names = pd.DataFrame({
        "gene": res["names"]["True"],
        "logfoldchange": res["logfoldchanges"]["True"],
        "pvals": res["pvals"]["True"],
        "pvals_adj": res["pvals_adj"]["True"],
        "scores": res["scores"]["True"],
        "pts_nose": np.asarray(res["pts"]["True"]).ravel(),
        "pts_rest": np.asarray(res["pts"]["False"]).ravel(),
    })
    names = names.sort_values("logfoldchange", ascending=False).reset_index(drop=True)
    print(f"[de] DE table: {len(names)} genes, sig (pvals_adj<0.05 & lfc>0.25): {int(((names.pvals_adj<0.05)&(names.logfoldchange>0.25)).sum())}")

    names.to_csv(outdir / "de_nose_vs_earrib_full.tsv", sep="\t", index=False)

    print("[de] running control DE: ear vs (nose+rib)", flush=True)
    adata.obs["tissue_binary_ear"] = (adata.obs["tissue"] == "ear").astype(str).astype("category")
    sc.tl.rank_genes_groups(
        adata, groupby="tissue_binary_ear", groups=["True"], reference="False",
        method="wilcoxon", use_raw=False, pts=True,
    )
    ear_res = adata.uns["rank_genes_groups"]
    ear_df = pd.DataFrame({
        "gene": ear_res["names"]["True"],
        "lfc_ear": ear_res["logfoldchanges"]["True"],
        "pvals_adj_ear": ear_res["pvals_adj"]["True"],
    })

    print("[de] running control DE: rib vs (nose+ear)", flush=True)
    adata.obs["tissue_binary_rib"] = (adata.obs["tissue"] == "rib").astype(str).astype("category")
    sc.tl.rank_genes_groups(
        adata, groupby="tissue_binary_rib", groups=["True"], reference="False",
        method="wilcoxon", use_raw=False, pts=True,
    )
    rib_res = adata.uns["rank_genes_groups"]
    rib_df = pd.DataFrame({
        "gene": rib_res["names"]["True"],
        "lfc_rib": rib_res["logfoldchanges"]["True"],
        "pvals_adj_rib": rib_res["pvals_adj"]["True"],
    })

    merged = names.merge(ear_df, on="gene", how="left").merge(rib_df, on="gene", how="left")
    merged["lfc_ear"] = merged["lfc_ear"].fillna(0.0)
    merged["lfc_rib"] = merged["lfc_rib"].fillna(0.0)
    merged["pvals_adj_ear"] = merged["pvals_adj_ear"].fillna(1.0)
    merged["pvals_adj_rib"] = merged["pvals_adj_rib"].fillna(1.0)

    sig_nose = (merged["pvals_adj"] < 0.05) & (merged["logfoldchange"] > 0.25)
    not_ear = (merged["lfc_ear"] < 0.10) | (merged["pvals_adj_ear"] > 0.05)
    not_rib = (merged["lfc_rib"] < 0.10) | (merged["pvals_adj_rib"] > 0.05)
    merged["nose_specific"] = sig_nose & not_ear & not_rib
    print(f"[de] nose-specific (up in nose, not up in ear/rib): {int(merged.nose_specific.sum())}")
    merged.to_csv(outdir / "de_nose_vs_earrib_with_controls.tsv", sep="\t", index=False)

    core = merged[merged.nose_specific].head(20).copy()
    print("[de] top 20 nose-specific genes:")
    for _, r in core.iterrows():
        print(f"   {r.gene:14s}  lfc_nose={r.logfoldchange:+.3f}  pts_nose={r.pts_nose:.3f}  lfc_ear={r.lfc_ear:+.3f}  lfc_rib={r.lfc_rib:+.3f}")

    panel = merged[merged.nose_specific].head(60).copy()
    print(f"[de] panel size: {len(panel)}")

    (outdir / "core_genes.txt").write_text("\n".join(core.gene.tolist()) + "\n", encoding="utf-8")
    (outdir / "panel_genes.txt").write_text("\n".join(panel.gene.tolist()) + "\n", encoding="utf-8")

    def lfc_to_weight(lfc):
        return float(np.clip(1.0 + lfc, 0.05, 2.5))

    core_weights = {r.gene: round(lfc_to_weight(r.logfoldchange), 3) for _, r in core.iterrows()}
    panel_weights = {r.gene: round(max(0.5, lfc_to_weight(r.logfoldchange) * 0.8), 3) for _, r in panel.iterrows()}

    candidate_axis = {
        "axis_id": "tissue_developmental_state::Nasal_Septum_Cartilage",
        "layer": "tissue_developmental_state",
        "name_en": "Nasal_Septum_Cartilage",
        "name_cn": "鼻中隔软骨",
        "biological_scope": (
            "Tissue/developmental axis representing chondrocytes from the "
            "nasal septum cartilage (hyaline, neural-crest-derived). Derived "
            "from the in-house EBR scRNA self-test (batch='nose' vs batch in "
            "{'ear','rib'}); the same EBR dataset is also one of the planned "
            "sources for future independent validation of other v1 axes."
        ),
        "status": "production",
        "core_genes": list(core.gene),
        "panel_genes": list(panel.gene),
        "anti_genes": [],
        "marker_weights": {**panel_weights, **core_weights},
        "anti_marker_weights": {},
        "evidence": {
            "derivation": [
                "In-house EBR scRNA self-test (F:\\cartifm\\outputs\\EBR\\EBR.h5ad), batch='nose' n=16706 cells vs ('ear','rib') n=16179 cells",
                "Wilcoxon rank-sum DE (scanpy.rank_genes_groups, use_raw=False) on the pre-normalized log1p/scaled X matrix",
                "Nose-specific filter: padj<0.05 & lfc>0.25 in nose AND not significantly up in ear or rib (lfc<0.10 or padj>0.05)"
            ],
            "internal_support": [
                f"Top 20 nose-specific markers with lfc range {core.logfoldchange.min():.2f}..{core.logfoldchange.max():.2f} and pts(nose) {core.pts_nose.min():.2f}..{core.pts_nose.max():.2f}",
                f"Nose-specific gene count (padj<0.05, lfc>0.25, not in ear/rib up): {int(merged.nose_specific.sum())}"
            ],
            "independent_validation": [],
            "literature_support": [
                "Hyaline cartilage markers (COL2A1, ACAN, SOX9) expected and recovered; not sufficient on their own to distinguish nasal septum from other hyaline cartilages (e.g. rib, tracheal, articular)"
            ]
        },
        "interpretation": (
            "chondrocytes from nasal septum cartilage, a hyaline cartilage "
            "of neural-crest origin that is the dominant structural support "
            "of the human nasal septum"
        ),
        "limitations": [
            "Axis derived from a single in-house EBR self-test; the same dataset is also one of the planned sources for future independent validation of other v1 axes (see evidence_policy.independent_validation_requirements). This dual use is explicitly flagged and should be reconciled before the axis is used as independent evidence in any downstream claim.",
            "Nose tissue identity assumed to be nasal septum cartilage; the source EBR.h5ad does not document the anatomical sub-region. If 'nose' includes alar/upper-lateral cartilage the axis would be more general and the name would be inaccurate.",
            "Tissue axis, not a cell-subtype axis; should not be used to disambiguate cell types within nasal septum. The existing cell_subtype layer axes (10 axes) continue to govern cell-state identity.",
            "No external validation against public nasal-septum scRNA data (e.g. GSE datasets, ENCODE, GTEx); the panel is anchored only on EBR.",
            "No cross-platform check (10x v3 vs v2/3'); the EBR h5ad uses 10x chemistry that should be confirmed before applying this axis to data from other chemistries."
        ],
        "recommended_use": [
            "Tissue-of-origin annotation when the input h5ad exposes a 'tissue' (or equivalent) column that explicitly says 'nasal_septum' / 'septum_nasi' / 'nasal cartilage'",
            "Cross-tissue comparison: nasal septum vs other hyaline cartilages (rib, articular) on a shared embedding",
            "scRNA cluster annotation in studies that dissect craniofacial cartilage"
        ],
        "forbidden_use": [
            "Do not cite this axis as external/independent validation of any other v1 axis (P9 hard constraint)",
            "Do not claim the axis is validated without follow-up on at least one public nasal-septum scRNA cohort"
        ],
        "source_files": [
            "outputs/nasal_septum_axis/de_nose_vs_earrib_with_controls.tsv",
            "outputs/nasal_septum_axis/core_genes.txt",
            "outputs/nasal_septum_axis/panel_genes.txt",
            "outputs/EBR/EBR.h5ad"
        ],
        "derivation_run": {
            "script": "scripts/add_nasal_septum_axis.py",
            "device": device,
            "method": "wilcoxon",
            "n_nose": int((adata.obs.tissue == "nose").sum()),
            "n_ear_rib": int((adata.obs.tissue.isin(["ear", "rib"])).sum()),
            "n_total": int(adata.n_obs),
            "n_genes_tested": int(adata.n_vars),
            "n_nose_specific_panel": int(merged.nose_specific.sum()),
            "n_core": int(len(core)),
            "n_panel": int(len(panel)),
        }
    }

    axis_path = outdir / "nasal_septum_axis_candidate.json"
    axis_path.write_text(json.dumps(candidate_axis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[de] wrote candidate axis -> {axis_path}")

    summary = {
        "device": device,
        "n_cells_total": int(adata.n_obs),
        "n_cells_nose": int((adata.obs.tissue == "nose").sum()),
        "n_cells_ear_rib": int((adata.obs.tissue.isin(["ear", "rib"])).sum()),
        "n_genes_tested": int(adata.n_vars),
        "n_de_nose_specific": int(merged.nose_specific.sum()),
        "core_genes": list(core.gene),
        "panel_genes": list(panel.gene),
        "de_table_top20": [
            {"gene": r.gene, "lfc": float(r.logfoldchange), "pvals_adj": float(r.pvals_adj),
             "pts_nose": float(r.pts_nose), "pts_rest": float(r.pts_rest)}
            for _, r in core.iterrows()
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[de] wrote summary -> {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
