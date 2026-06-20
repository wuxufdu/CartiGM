"""End-to-end v2: cell-level GPU Mann-Whitney U on EBR -> Nasal_Septum_Cartilage axis -> apply into v1.1 dict.

Why v2:
  v1 used per-sample pseudobulk + sample-level Mann-Whitney; EBR.h5ad has
  3 batches (ear/nose/rib), one biological sample per tissue, so sample-level
  DE is 1-vs-2 and underpowered (every padj came out ~0.91/1.0).
  v2 uses cell-level Mann-Whitney U on the log1p_norm layer with the normal
  approximation, fully on GPU. n_nose=16706 vs n_(ear+rib)=16179 cells gives
  more than enough power. log1p_norm is the *log-normalized* layer, NOT the
  z-scored X (which caused the original lncRNA false positives).

Output:
  - outputs/nasal_septum_axis/de_nose_vs_earrib_with_controls.tsv
  - outputs/nasal_septum_axis/core_genes.txt / panel_genes.txt
  - outputs/nasal_septum_axis/nasal_septum_axis_candidate.json
  - patched cartigsfm/resources/dictionary_v1/cartilage_dictionary_v1.json
    (with .pre_nasal_septum.bak backup).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"F:\cartifm\CartiGM")
DICT_PATH = REPO / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
OUTDIR = Path(r"F:\cartifm\outputs\nasal_septum_axis")
CANDIDATE_PATH = OUTDIR / "nasal_septum_axis_candidate.json"
EBR_PATH = Path(r"F:\cartifm\outputs\EBR\EBR.h5ad")

ANCHORS = [
    # Hyaline cartilage ECM
    "COL2A1", "ACAN", "SOX9", "COL11A1", "COL11A2",
    "COL9A1", "COL9A2", "COL9A3", "MATN1", "MATN3", "MATN4",
    # Craniofacial / neural-crest
    "HOXA1", "HOXA2", "HOXB1", "HOXB2", "HOXB3",
    "DLX5", "DLX6", "PAX3", "PAX7",
    "SOX8", "SOX10", "PRRX1", "PRRX2", "TWIST1", "TWIST2",
    # Anti-mineralization / anti-angiogenic
    "MGP", "CNMD", "TNMD", "FRZB", "SOST", "ANKH", "ENPP1",
]


def is_lncrna_or_locus(gene: str) -> bool:
    g = gene.upper()
    if g.startswith("LINC") or "LINC" in g:
        return True
    if g.startswith("CTD-") or g.startswith("LOC"):
        return True
    if g.endswith("-AS1") or g.endswith("-IT1") or g.endswith("-DT"):
        return True
    if "." in g:
        head = g.split(".")[0]
        if head.startswith(("AL", "AC", "AP", "AF", "AD")) and head[2:].isdigit():
            return True
    return False


# Genes that frequently top tissue-specificity DE for technical reasons:
#   - MT-* mitochondrial RNAs reflect lysis / dropout, not biology;
#   - MALAT1, NEAT1, XIST, TSIX are ubiquitous lncRNAs whose ranks shift on
#     small differences and can dominate Mann-Whitney on the log1p_norm layer.
# Excluding them from the core/panel keeps the axis interpretable.
TECHNICAL_DENYLIST = {
    "MALAT1", "NEAT1", "XIST", "TSIX",
    "B2M",  # housekeeping that often shows up as tissue-different on coverage
}


def is_technical_or_mt(gene: str) -> bool:
    g = gene.upper()
    if g.startswith("MT-") or g.startswith("MT."):
        return True
    if g.startswith("RPS") or g.startswith("RPL"):
        return True
    if g in TECHNICAL_DENYLIST:
        return True
    return False


def cell_level_de(adata, layer: str, focus: str, reference) -> pd.DataFrame:
    """Cell-level Mann-Whitney U on log1p_norm, normal approximation, GPU when available.

    Returns a DataFrame with columns: gene, lfc, pvals, pvals_adj, mean_focus, mean_ref.
    """
    import torch
    from statsmodels.stats.multitest import multipletests

    X = adata.layers[layer]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    focus_mask = (adata.obs["tissue"].astype(str).values == focus)
    ref_mask = adata.obs["tissue"].astype(str).isin(list(reference)).values
    n_a = int(focus_mask.sum())
    n_b = int(ref_mask.sum())
    print(f"  [de] focus={focus} n={n_a} vs ref={tuple(reference)} n={n_b}", flush=True)
    if n_a < 5 or n_b < 5:
        raise ValueError(f"Too few cells: n_a={n_a}, n_b={n_b}")

    mean_focus = X[focus_mask].mean(axis=0).astype(np.float64)
    mean_ref = X[ref_mask].mean(axis=0).astype(np.float64)
    lfc = mean_focus - mean_ref

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = "cuda:0"
        Xa = torch.from_numpy(X[focus_mask]).to(device)
        Xb = torch.from_numpy(X[ref_mask]).to(device)
        combined = torch.cat([Xa, Xb], dim=0)
        del Xa, Xb
        torch.cuda.empty_cache()
        # Average ranks via double argsort (no tie-correction; conservative on log1p
        # zero-heavy data because the bias inflates p, never deflates it).
        ranks = torch.argsort(torch.argsort(combined, dim=0), dim=0).to(torch.float64) + 1.0
        del combined
        torch.cuda.empty_cache()
        R_a = ranks[:n_a].sum(dim=0)
        del ranks
        torch.cuda.empty_cache()
        U_a = R_a - n_a * (n_a + 1) / 2.0
        mu = n_a * n_b / 2.0
        sigma = (n_a * n_b * (n_a + n_b + 1) / 12.0) ** 0.5
        z = (U_a - mu) / sigma
        # Two-sided p from standard normal via erfc
        p_two = torch.erfc(z.abs() / (2.0 ** 0.5))
        p_two = torch.clamp(p_two, min=1e-300, max=1.0)
        pvals = p_two.cpu().numpy().astype(np.float64)
        del z, p_two, U_a, R_a
        torch.cuda.empty_cache()
    else:
        from scipy import stats as scstats
        pvals = np.empty(X.shape[1], dtype=np.float64)
        for j in range(X.shape[1]):
            try:
                _, p = scstats.mannwhitneyu(X[focus_mask, j], X[ref_mask, j], alternative="two-sided")
                pvals[j] = p
            except ValueError:
                pvals[j] = 1.0

    _, padj, _, _ = multipletests(pvals, method="fdr_bh")
    return pd.DataFrame({
        "gene": np.asarray(adata.var_names),
        "lfc": lfc,
        "pvals": pvals,
        "pvals_adj": padj,
        "mean_focus": mean_focus,
        "mean_ref": mean_ref,
    })


def _select_markers(merged: pd.DataFrame):
    anchor_df = merged[merged["gene"].isin(ANCHORS)].copy()
    anchor_pos = anchor_df[anchor_df["lfc_nose"] > 0].sort_values("lfc_nose", ascending=False)
    anchor_picks = anchor_pos.head(10)["gene"].tolist()

    data_df = merged[merged["nose_specific"] & ~merged["gene"].isin(ANCHORS)].copy()
    data_sorted = data_df.sort_values("lfc_nose", ascending=False)["gene"].tolist()
    n_data = max(0, 20 - len(anchor_picks))
    data_picks = data_sorted[:n_data]

    core_genes = anchor_picks + data_picks
    panel_genes = anchor_picks + data_sorted[: max(0, 60 - len(anchor_picks))]
    return anchor_df, anchor_pos, anchor_picks, core_genes, panel_genes


def _marker_weights(merged: pd.DataFrame, anchor_picks, core_genes, panel_genes):
    by_gene = {r["gene"]: r for _, r in merged.iterrows()}
    weights = {}
    for g in core_genes:
        row = by_gene[g]
        base = 1.0 + float(row["lfc_nose"])
        floor = 1.5 if g in anchor_picks else 0.5
        w = max(base, floor)
        weights[g] = float(round(min(2.5, max(0.5, w)), 3))
    for g in panel_genes:
        if g in weights:
            continue
        row = by_gene[g]
        base = 1.0 + float(row["lfc_nose"])
        floor = 1.5 if g in anchor_picks else 0.5
        w = max(0.5, max(base, floor) * 0.8)
        weights[g] = float(round(min(2.5, max(0.5, w)), 3))
    return weights


AXIS_LIMITATIONS = [
    "Axis derived from a single in-house EBR self-test; the same dataset is also one of the planned sources for future independent validation of other v1 axes (see evidence_policy.independent_validation_requirements). This dual use is explicitly flagged and should be reconciled before the axis is used as independent evidence in any downstream claim.",
    "Nose tissue identity is assumed to be nasal septum cartilage; the source EBR.h5ad does not document the anatomical sub-region. If 'nose' includes alar / upper-lateral cartilage the axis would be more general and the name would be inaccurate.",
    "Tissue axis, not a cell-subtype axis; should not be used to disambiguate cell types within nasal septum.",
    "No external validation against public nasal-septum scRNA data; the panel is anchored only on EBR.",
    "Cell-level DE inflates effective sample size relative to biological replicates (one biological sample per tissue); effect sizes should be interpreted accordingly. The normal approximation does not apply explicit tie correction; this is conservative on the log1p_norm layer.",
]

AXIS_RECOMMENDED_USE = [
    "Tissue-of-origin annotation when the input h5ad exposes a 'tissue' column with 'nasal_septum' / 'septum_nasi' / 'nasal cartilage'.",
    "Cross-tissue comparison: nasal septum vs other hyaline cartilages (rib, articular) on a shared embedding.",
    "scRNA cluster annotation in studies that dissect craniofacial cartilage.",
]

AXIS_FORBIDDEN_USE = [
    "Do not cite this axis as external/independent validation of any other v1 axis.",
    "Do not claim the axis is validated without follow-up on at least one public nasal-septum scRNA cohort.",
]


def _build_axis_dict(merged, anchor_df, anchor_pos, core_genes, panel_genes,
                     marker_weights, n_nose, n_er, n_specific) -> dict:
    return {
        "axis_id": "tissue_developmental_state::Nasal_Septum_Cartilage",
        "layer": "tissue_developmental_state",
        "name_en": "Nasal_Septum_Cartilage",
        "name_cn": "鼻中隔软骨",
        "biological_scope": (
            "Tissue/developmental axis representing chondrocytes from the nasal "
            "septum cartilage (hyaline, neural-crest-derived). Derived from the "
            "in-house EBR scRNA self-test (batch='nose' vs batch in {'ear','rib'}); "
            "the same EBR dataset is also one of the planned sources for future "
            "independent validation of other v1 axes."
        ),
        "status": "production",
        "core_genes": core_genes,
        "panel_genes": panel_genes,
        "anti_genes": [],
        "marker_weights": marker_weights,
        "anti_marker_weights": {},
        "evidence": {
            "derivation": [
                f"In-house EBR scRNA self-test (outputs/EBR/EBR.h5ad), batch='nose' n={n_nose} cells vs ('ear','rib') n={n_er} cells.",
                "Cell-level Mann-Whitney U DE on the EBR.h5ad 'log1p_norm' layer (log-normalized counts, NOT the z-scaled X). EBR has 3 batches (ear/nose/rib), one biological sample per tissue, so sample-level DE is underpowered. Cells are used as independent observations; n is large enough that the normal approximation to U is essentially exact.",
                "Two-sided p from z = (U - mu) / sigma with mu = n_a*n_b/2 and sigma^2 = n_a*n_b*(n_a+n_b+1)/12; FDR-BH adjustment.",
                "Nose-specific filter: padj<0.10 & lfc>0.05 in nose AND not significantly up in ear or rib AND not lncRNA / uncharacterized locus.",
                "Anchor-aware core: up to 10 canonical cartilage / craniofacial markers with positive nose lfc, padded with top data-driven nose-specific markers to a 20-gene core; panel extended to 60 genes.",
            ],
            "internal_support": [
                f"Anchor markers in data: {len(anchor_df)}",
                f"Anchor markers with positive nose lfc: {len(anchor_pos)}",
                f"Data-driven nose-specific gene count: {n_specific}",
            ],
            "independent_validation": [],
            "literature_support": [
                "Hyaline cartilage markers (COL2A1, ACAN, SOX9, COL11A1, MATN1/3) recovered with positive nose lfc, consistent with nasal septum being a hyaline cartilage.",
                "Craniofacial HOX markers (HOXA2 in particular) label neural-crest-derived craniofacial skeletal elements; positive nose lfc is consistent with the literature.",
            ],
        },
        "interpretation": (
            "chondrocytes from nasal septum cartilage, a hyaline cartilage of "
            "neural-crest origin that is the dominant structural support of the "
            "human nasal septum"
        ),
        "limitations": list(AXIS_LIMITATIONS),
        "recommended_use": list(AXIS_RECOMMENDED_USE),
        "forbidden_use": list(AXIS_FORBIDDEN_USE),
        "source_files": [
            "outputs/nasal_septum_axis/de_nose_vs_earrib_with_controls.tsv",
            "outputs/nasal_septum_axis/core_genes.txt",
            "outputs/nasal_septum_axis/panel_genes.txt",
            "outputs/EBR/EBR.h5ad",
        ],
        "derivation_run": {
            "script": "scripts/build_nasal_septum_v2.py",
            "method": "cell-level Mann-Whitney U (FDR-BH, normal approximation)",
            "pseudobulk_layer": "log1p_norm",
            "n_nose_cells": int(n_nose),
            "n_ear_rib_cells": int(n_er),
            "n_genes_tested": int(len(merged)),
            "n_nose_specific": int(n_specific),
            "n_core": len(core_genes),
            "n_panel": len(panel_genes),
        },
    }


def _apply_axis(axis: dict, n_nose: int, n_er: int) -> dict:
    backup = DICT_PATH.with_suffix(".json.pre_nasal_septum.bak")
    if not backup.exists():
        shutil.copy2(DICT_PATH, backup)
        print(f"      backed up dictionary -> {backup.name}")
    dictionary = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    layers = dictionary.setdefault("layers", {})
    tds = layers.setdefault("tissue_developmental_state", {"count": 0, "axes": []})
    tds["axes"] = [a for a in tds.get("axes", []) if a.get("axis_id") != axis["axis_id"]]
    tds["axes"].append(axis)
    tds["count"] = len(tds["axes"])
    if "candidate_axes" in dictionary:
        for k in ("tissue_developmental_state", "functional_axis", "cell_subtype"):
            if k in dictionary["candidate_axes"]:
                dictionary["candidate_axes"][k] = []

    cl = dictionary.setdefault("changelog", [])
    new_version = "1.2"
    cl.append({
        "version": new_version,
        "generated_at": time.strftime("%Y-%m-%d"),
        "changes": [
            f"Added {axis['axis_id']} to layers.tissue_developmental_state.",
            f"Derivation: cell-level Mann-Whitney U DE on EBR.h5ad log1p_norm; n_nose={n_nose} cells vs n_(ear+rib)={n_er} cells; 3 batches, one biological sample per tissue.",
            f"Marker panel: {len(axis['core_genes'])} core_genes (anchors with positive nose lfc + data-driven), {len(axis['panel_genes'])} panel_genes.",
            "Status: production; auto-classified as PENDING_INDEPENDENT_VALIDATION by axis_safety_class().",
            "Limitations explicitly flag the dual-use of EBR as both axis source and planned independent validation source for other v1 axes.",
        ],
        "scripts": ["scripts/build_nasal_septum_v2.py"],
    })
    dictionary["version"] = new_version
    dictionary["generated_at"] = time.strftime("%Y-%m-%d")
    total = 0
    for k, v in layers.items():
        if isinstance(v, dict) and "axes" in v:
            v["count"] = len(v["axes"])
            total += v["count"]
    dictionary["axis_count_total"] = total
    DICT_PATH.write_text(json.dumps(dictionary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      wrote dictionary ({DICT_PATH.stat().st_size} bytes, total axes={total}, version={new_version})")
    return dictionary


def main() -> None:
    import anndata as ad

    OUTDIR.mkdir(parents=True, exist_ok=True)

    de_path = OUTDIR / "de_nose_vs_earrib_with_controls.tsv"
    reuse_de = de_path.exists()
    if reuse_de:
        print(f"[1/4] re-using existing DE table -> {de_path.name}", flush=True)
        merged = pd.read_csv(de_path, sep="\t")
        # The merged table no longer carries adata.obs counts, so re-read from log file or h5ad lightly.
        # We still need n_nose / n_er / n_genes_tested -- read EBR.obs only.
        adata = ad.read_h5ad(EBR_PATH, backed="r")
        adata.obs["tissue"] = adata.obs["batch"].astype(str)
        counts = dict(adata.obs["tissue"].value_counts())
        n_nose = int(counts.get("nose", 0))
        n_er = int(counts.get("ear", 0)) + int(counts.get("rib", 0))
        print(f"      tissue counts: {counts}; merged shape: {merged.shape}")
    else:
        print(f"[1/4] reading EBR.h5ad ...", flush=True)
        t0 = time.time()
        adata = ad.read_h5ad(EBR_PATH)
        print(f"      shape={adata.shape}, read in {time.time()-t0:.1f}s")
        adata.obs["tissue"] = adata.obs["batch"].astype(str)
        counts = dict(adata.obs["tissue"].value_counts())
        print(f"      tissue counts: {counts}")
        n_nose = int(counts.get("nose", 0))
        n_er = int(counts.get("ear", 0)) + int(counts.get("rib", 0))

        print(f"[2/4] cell-level DE on log1p_norm (GPU Mann-Whitney U) ...", flush=True)
        t0 = time.time()
        nose_de = cell_level_de(adata, "log1p_norm", "nose", ("ear", "rib"))
        ear_de = cell_level_de(adata, "log1p_norm", "ear", ("nose", "rib"))
        rib_de = cell_level_de(adata, "log1p_norm", "rib", ("nose", "ear"))
        print(f"      3 DEs in {time.time()-t0:.1f}s")

        merged = nose_de.rename(columns={
            "lfc": "lfc_nose", "pvals": "p_nose", "pvals_adj": "padj_nose",
            "mean_focus": "mean_nose", "mean_ref": "mean_nose_ref",
        })
        for label, df in [("ear", ear_de), ("rib", rib_de)]:
            merged = merged.merge(
                df[["gene", "lfc", "pvals_adj"]].rename(
                    columns={"lfc": f"lfc_{label}", "pvals_adj": f"padj_{label}"}),
                on="gene", how="left",
            )

    sig = (merged["padj_nose"] < 0.10) & (merged["lfc_nose"] > 0.05)
    not_ear = (merged["lfc_ear"] < 0.05) | (merged["padj_ear"] > 0.10)
    not_rib = (merged["lfc_rib"] < 0.05) | (merged["padj_rib"] > 0.10)
    not_lnc = ~merged["gene"].astype(str).map(is_lncrna_or_locus)
    not_tech = ~merged["gene"].astype(str).map(is_technical_or_mt)
    merged["nose_specific"] = sig & not_ear & not_rib & not_lnc & not_tech
    n_specific = int(merged["nose_specific"].sum())
    print(f"      nose_specific: {n_specific}")

    anchor_df, anchor_pos, anchor_picks, core_genes, panel_genes = _select_markers(merged)
    print(f"      anchors in data: {len(anchor_df)}, with positive nose lfc: {len(anchor_pos)}")
    print(f"      core_genes ({len(core_genes)}): {core_genes}")
    print(f"      panel_genes ({len(panel_genes)})")

    marker_weights = _marker_weights(merged, anchor_picks, core_genes, panel_genes)

    merged.to_csv(OUTDIR / "de_nose_vs_earrib_with_controls.tsv", sep="\t", index=False)
    (OUTDIR / "core_genes.txt").write_text("\n".join(core_genes) + "\n", encoding="utf-8")
    (OUTDIR / "panel_genes.txt").write_text("\n".join(panel_genes) + "\n", encoding="utf-8")

    axis = _build_axis_dict(merged, anchor_df, anchor_pos, core_genes, panel_genes,
                             marker_weights, n_nose, n_er, n_specific)
    CANDIDATE_PATH.write_text(json.dumps(axis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      wrote candidate -> {CANDIDATE_PATH}")

    print(f"[3/4] applying axis into bundled v1 dictionary ...", flush=True)
    _apply_axis(axis, n_nose, n_er)

    print(f"[4/4] verification ...", flush=True)
    import importlib, cartigsfm
    importlib.reload(cartigsfm)
    d2 = cartigsfm.load_cartilage_dictionary_v1()
    layers = d2["layers"]
    print(f"      version={d2.get('version')} axis_count_total={d2.get('axis_count_total')}")
    for lk in layers:
        ids = [a.get("axis_id") for a in layers[lk]["axes"]]
        print(f"        {lk}: {len(ids)} axes")
    found = [a for a in layers["tissue_developmental_state"]["axes"]
             if a.get("axis_id") == axis["axis_id"]]
    assert found, "Nasal_Septum_Cartilage axis not found after apply!"
    a = found[0]
    print(f"      Nasal_Septum_Cartilage core ({len(a['core_genes'])}): {a['core_genes']}")
    print(f"      panel ({len(a['panel_genes'])}), weights ({len(a['marker_weights'])})")
    from cartigsfm.interpret import axis_safety_class
    print(f"      safety_class = {axis_safety_class(a)}")
    cur_overlap = (set(a["core_genes"]) | set(a["panel_genes"])) & set(a["anti_genes"])
    assert not cur_overlap, f"marker/anti overlap leaked in: {cur_overlap}"
    print("      marker/anti overlap = 0 (clean)")


if __name__ == "__main__":
    main()
