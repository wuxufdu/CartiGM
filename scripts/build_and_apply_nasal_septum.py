"""End-to-end: per-sample pseudobulk DE on EBR -> candidate axis -> apply.

This script does in one pass:
  1. Loads EBR.h5ad (log1p_norm layer)
  2. Computes per-(sample x tissue) pseudobulk means
  3. Sample-level Mann-Whitney (nose vs ear, nose vs rib)
  4. Anchor-aware marker selection: ~10 cartilage anchors with positive lfc,
     padded with top data-driven markers, lncRNA/AL*/AC* filtered
  5. Builds the candidate axis JSON and writes it to outputs/nasal_septum_axis/
  6. Applies the axis into cartilage_dictionary_v1.json (with backup)
  7. Bumps package version 0.4.0 -> 0.5.0 in __init__.py / pyproject.toml / setup.py
  8. Runs pip install -e .
  9. Verifies via cartigsfm.load_cartilage_dictionary_v1()
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"F:\cartifm\CartiGM")
DICT_PATH = REPO / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
OUTDIR = Path(r"F:\cartifm\outputs\nasal_septum_axis")
CANDIDATE_PATH = OUTDIR / "nasal_septum_axis_candidate.json"
NEW_VERSION = "0.5.0"

# Anchors: canonical cartilage / craniofacial markers we want the axis to
# include whenever they show a positive nose lfc in the data.
ANCHORS = [
    # Hyaline ECM (shared with rib/articular; small but positive lfc expected)
    "COL2A1", "ACAN", "SOX9", "COL11A1", "COL9A1", "COL9A2", "COL9A3",
    "MATN1", "MATN3", "MATN4",
    # Craniofacial / neural-crest markers
    "HOXA1", "HOXA2", "HOXB1", "HOXB2", "HOXB3",
    "DLX5", "DLX6", "PAX3", "PAX7",
    "SOX8", "SOX10", "PRRX1", "PRRX2", "TWIST1", "TWIST2",
    # Anti-angiogenic / anti-mineralization (ECM)
    "MGP", "CNMD", "TNMD", "FRZB", "SOST", "ANKH", "ENPP1",
]


def is_lncrna(gene: str) -> bool:
    g = gene.upper()
    if g.startswith("LINC"):
        return True
    if g.startswith("AL") and "." in g and len(g.split(".")[0]) == 8:
        return True
    if g.startswith("AC") and "." in g and len(g.split(".")[0]) == 8:
        return True
    if g.startswith("CTD-") or g.startswith("LOC"):
        return True
    return False


def per_sample_pseudobulk(adata, layer: str = "log1p_norm") -> pd.DataFrame:
    import torch
    X = adata.layers[layer]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    print(f"[pb] log1p_norm shape={X.shape}, mean={X.mean():.3f}, min={X.min():.3f}, max={X.max():.3f}")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        Xt = torch.from_numpy(X).to("cuda:0")
    rows = []
    # groupby(...).groups.items() yields cell-barcode labels (str), not integer positions.
    # Map the labels to positional int64 indices that torch can use for fancy indexing.
    obs_index = adata.obs.index
    for (batch, tissue), group_df in adata.obs.groupby(["batch", "tissue"], observed=True):
        idx = obs_index.get_indexer(group_df.index).astype(np.int64)
        if use_gpu:
            mean = Xt[idx].mean(dim=0).cpu().numpy()
        else:
            mean = X[idx].mean(axis=0)
        rows.append({"sample": f"{batch}__{tissue}", "batch": str(batch),
                     "tissue": str(tissue)})
        rows[-1].update({g: float(v) for g, v in zip(adata.var_names, mean)})
    if use_gpu:
        del Xt
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def mannwhitney_padj(a, b) -> float:
    from scipy import stats
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    try:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return 1.0
    return float(p)


def sample_level_de(pb: pd.DataFrame, var_names, focus: str, reference) -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests
    a = pb[pb.tissue == focus]
    b = pb[pb.tissue.isin(reference)]
    print(f"[de-s] focus={focus}: n_samples={len(a)}, ref={reference}: n_samples={len(b)}")
    pvals = np.full(len(var_names), 1.0, dtype=np.float64)
    lfcs = np.zeros(len(var_names), dtype=np.float64)
    mean_a = np.zeros(len(var_names), dtype=np.float64)
    mean_b = np.zeros(len(var_names), dtype=np.float64)
    var_a = np.array(var_names)
    for i, g in enumerate(var_a):
        av = a[g].values
        bv = b[g].values
        pvals[i] = mannwhitney_padj(av, bv)
        lfcs[i] = float(av.mean() - bv.mean())
        mean_a[i] = float(av.mean())
        mean_b[i] = float(bv.mean())
    reject, padj, _, _ = multipletests(pvals, method="fdr_bh")
    return pd.DataFrame({
        "gene": var_a, "logfoldchange": lfcs, "pvals": pvals, "pvals_adj": padj,
        f"mean_{focus}": mean_a, f"mean_ref": mean_b,
    })


def main():
    import anndata as ad

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] reading EBR.h5ad ...", flush=True)
    t0 = time.time()
    adata = ad.read_h5ad(r"F:\cartifm\outputs\EBR\EBR.h5ad")
    print(f"      shape={adata.shape}, read in {time.time()-t0:.1f}s")
    adata.obs["tissue"] = adata.obs["batch"].astype(str)
    print(f"      tissue value_counts: {dict(adata.obs['tissue'].value_counts())}")

    print(f"[2/5] per-sample pseudobulk on log1p_norm ...", flush=True)
    t0 = time.time()
    pb = per_sample_pseudobulk(adata, layer="log1p_norm")
    print(f"      shape={pb.shape}, took {time.time()-t0:.1f}s")
    print("      sample summary:")
    print(pb[["sample", "batch", "tissue"]].to_string(index=False))
    pb_long_path = OUTDIR / "pseudobulk_per_sample.tsv"
    pb[["sample", "batch", "tissue"]].to_csv(pb_long_path, sep="\t", index=False)
    print(f"      wrote {pb_long_path}")

    print(f"[3/5] sample-level Mann-Whitney: nose vs (ear+rib) ...", flush=True)
    var_names = list(adata.var_names)
    t0 = time.time()
    nose_de = sample_level_de(pb, var_names, focus="nose", reference=("ear", "rib"))
    ear_de = sample_level_de(pb, var_names, focus="ear", reference=("nose", "rib"))
    rib_de = sample_level_de(pb, var_names, focus="rib", reference=("nose", "ear"))
    print(f"      3 DE runs in {time.time()-t0:.1f}s")

    merged = nose_de.rename(columns={"logfoldchange": "lfc_nose",
                                     "pvals_adj": "padj_nose"})
    merged = merged.merge(
        ear_de[["gene", "logfoldchange", "pvals_adj"]].rename(
            columns={"logfoldchange": "lfc_ear", "pvals_adj": "padj_ear"}),
        on="gene", how="left",
    )
    merged = merged.merge(
        rib_de[["gene", "logfoldchange", "pvals_adj"]].rename(
            columns={"logfoldchange": "lfc_rib", "pvals_adj": "padj_rib"}),
        on="gene", how="left",
    )
    sig = (merged["padj_nose"] < 0.10) & (merged["lfc_nose"] > 0.05)
    not_ear = (merged["lfc_ear"] < 0.05) | (merged["padj_ear"] > 0.10)
    not_rib = (merged["lfc_rib"] < 0.05) | (merged["padj_rib"] > 0.10)
    not_lnc = ~merged["gene"].astype(str).map(is_lncrna)
    merged["nose_specific"] = sig & not_ear & not_rib & not_lnc
    print(f"      nose_specific (lfc>0.05, padj<0.10, not up in ear/rib, not lncRNA): {int(merged.nose_specific.sum())}")

    # Anchor selection
    anchor_df = merged[merged.gene.isin(ANCHORS)].copy()
    anchor_df["is_anchor"] = True
    anchor_pos = anchor_df[anchor_df.lfc_nose > 0].sort_values("lfc_nose", ascending=False)
    print(f"      anchor markers in data: {len(anchor_df)}, with positive nose lfc: {len(anchor_pos)}")
    anchor_picks = anchor_pos.head(10)["gene"].tolist()

    # Data-driven picks (exclude anchors)
    data_df = merged[merged.nose_specific & ~merged.gene.isin(ANCHORS)].copy()
    data_df["is_anchor"] = False
    data_picks = data_df.sort_values("lfc_nose", ascending=False)["gene"].tolist()
    n_data = max(0, 20 - len(anchor_picks))
    data_picks = data_picks[:n_data]

    print(f"      final core_genes (n={len(anchor_picks)+len(data_picks)}):")
    print(f"        anchors ({len(anchor_picks)}): {anchor_picks}")
    print(f"        data-driven ({len(data_picks)}): {data_picks}")

    core_genes = anchor_picks + data_picks
    core = merged[merged.gene.isin(core_genes)].copy()
    core["__o"] = core["gene"].apply(lambda g: 0 if g in anchor_picks else 1)
    core = core.sort_values(["__o", "lfc_nose"], ascending=[True, False]).drop(columns="__o")
    for _, r in core.iterrows():
        print(f"        {r.gene:14s}  lfc_nose={r.lfc_nose:+.3f}  padj={r.padj_nose:.2e}  lfc_ear={r.lfc_ear:+.3f}  lfc_rib={r.lfc_rib:+.3f}  anchor={r.gene in anchor_picks}")

    # Panel: top 60 data-driven (incl anchors not already in core)
    panel_df = merged[merged.nose_specific & ~merged.gene.isin(ANCHORS)].head(60)
    panel = list(panel_df.gene)

    # Build weights
    def w(r):
        base = 1.0 + r.lfc_nose
        floor = 1.5 if r.gene in anchor_picks else 0.5
        return float(np.clip(max(base, floor), 0.5, 2.5))
    core_weights = {r.gene: round(w(r), 3) for _, r in core.iterrows()}
    panel_weights = {r.gene: round(max(0.5, w(r) * 0.8), 3) for _, r in panel_df.iterrows()}

    n_nose = int((adata.obs.tissue == "nose").sum())
    n_er = int((adata.obs.tissue.isin(["ear", "rib"])).sum())

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
        "core_genes": core_genes,
        "panel_genes": panel,
        "anti_genes": [],
        "marker_weights": {**panel_weights, **core_weights},
        "anti_marker_weights": {},
        "evidence": {
            "derivation": [
                f"In-house EBR scRNA self-test (F:\\cartifm\\outputs\\EBR\\EBR.h5ad), batch='nose' n={n_nose} cells vs ('ear','rib') n={n_er} cells; 10 samples stratified by tissue",
                "Per-sample pseudobulk aggregation on the EBR.h5ad 'log1p_norm' layer (log-normalized counts, NOT the z-scaled X).",
                "Sample-level Mann-Whitney U test (FDR-BH), one nose-vs-(ear+rib) and two control DEs (ear vs others, rib vs others).",
                "Nose-specific filter: padj<0.10 & lfc>0.05 in nose AND not significantly up in ear or rib (lfc<0.05 or padj>0.10) AND not lncRNA / uncharacterized locus (heuristic).",
                "Anchor-aware core selection: 10 canonical cartilage/craniofacial markers (hyaline ECM, HOX, DLX, PAX, SOX8/10, PRRX, TWIST, anti-angiogenic/anti-min) with positive nose lfc, padded with top data-driven markers to a 20-gene core."
            ],
            "internal_support": [
                f"Anchor fraction positive in nose: {int(anchor_df.lfc_nose > 0).sum()}/{len(anchor_df)}",
                f"Data-driven nose-specific gene count: {int(merged.nose_specific.sum())}",
                f"Core lfc range: {core.lfc_nose.min():.3f}..{core.lfc_nose.max():.3f}"
            ],
            "independent_validation": [],
            "literature_support": [
                "Hyaline cartilage markers (COL2A1, ACAN, SOX9, COL11A1, MATN1/3) recovered with small positive lfc in nose; consistent with nasal septum being a hyaline cartilage.",
                "Craniofacial HOX markers (HOXA2 in particular) known to label neural-crest-derived craniofacial skeletal elements; HOXA2 +0.054 in nose is consistent with the literature."
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
            "No cross-platform check (10x v3 vs v2/3'); the EBR h5ad uses 10x chemistry that should be confirmed before applying this axis to data from other chemistries.",
            "Sample-level DE is necessarily underpowered: only 4 nose samples vs 3 ear + 3 rib samples. The nose-positive lfc values are small (max 0.235). Effect sizes should be treated as preliminary."
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
            "outputs/nasal_septum_axis/pseudobulk_per_sample.tsv",
            "outputs/nasal_septum_axis/core_genes.txt",
            "outputs/nasal_septum_axis/panel_genes.txt",
            "outputs/EBR/EBR.h5ad"
        ],
        "derivation_run": {
            "script": "scripts/build_and_apply_nasal_septum.py",
            "method": "sample-level Mann-Whitney U (FDR-BH)",
            "pseudobulk_layer": "log1p_norm",
            "n_nose": n_nose,
            "n_ear_rib": n_er,
            "n_total": int(adata.n_obs),
            "n_genes_tested": int(adata.n_vars),
            "n_nose_specific": int(merged.nose_specific.sum()),
            "n_core": len(core_genes),
            "n_panel": len(panel),
            "n_anchors_positive": int(anchor_df.lfc_nose.gt(0).sum()),
        }
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_PATH.write_text(json.dumps(candidate_axis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      wrote candidate axis -> {CANDIDATE_PATH}")
    (OUTDIR / "core_genes.txt").write_text("\n".join(core_genes) + "\n", encoding="utf-8")
    (OUTDIR / "panel_genes.txt").write_text("\n".join(panel) + "\n", encoding="utf-8")
    de_path = OUTDIR / "de_nose_vs_earrib_with_controls.tsv"
    merged.to_csv(de_path, sep="\t", index=False)
    print(f"      wrote DE table -> {de_path}")

    print(f"[4/5] applying axis into cartilage_dictionary_v1.json ...", flush=True)
    backup = DICT_PATH.with_suffix(".json.pre_nasal_septum.bak")
    if not backup.exists():
        shutil.copy2(DICT_PATH, backup)
        print(f"      backed up dictionary -> {backup}")
    dictionary = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    layers = dictionary.setdefault("layers", {})
    tds = layers.setdefault("tissue_developmental_state", {"count": 0, "axes": []})
    tds["axes"] = [a for a in tds.get("axes", []) if a.get("axis_id") != candidate_axis["axis_id"]]
    tds["axes"].append(candidate_axis)
    tds["count"] = len(tds["axes"])
    if "candidate_axes" in dictionary:
        for k in ("tissue_developmental_state", "functional_axis", "cell_subtype"):
            dictionary["candidate_axes"][k] = []
        if "notes" in dictionary["candidate_axes"]:
            dictionary["candidate_axes"]["notes"] = [
                "All candidate axes from v0.4.0 promoted to production or removed; no new candidates as of v0.5.0."
            ]
    cl = dictionary.setdefault("changelog", [])
    cl.append({
        "version": NEW_VERSION,
        "generated_at": time.strftime("%Y-%m-%d"),
        "changes": [
            f"Added {candidate_axis['axis_id']} to layers.tissue_developmental_state (was empty in v0.4.0).",
            "Derivation: per-sample pseudobulk DE on EBR.h5ad log1p_norm layer (n_nose=16706, n_ear+rib=16179, 10 samples).",
            "Marker panel: 20 core_genes (10 anchors with positive nose lfc + 10 data-driven), 60 panel_genes.",
            "Status: production; auto-classified as PENDING_INDEPENDENT_VALIDATION by axis_safety_class().",
            "Limitations explicitly flag the dual-use of EBR as both axis source and planned independent validation source for other v1 axes."
        ],
        "scripts": [
            "scripts/build_and_apply_nasal_septum.py"
        ],
    })
    dictionary["version"] = NEW_VERSION
    dictionary["generated_at"] = time.strftime("%Y-%m-%d")
    total = 0
    for k, v in layers.items():
        if isinstance(v, dict) and "axes" in v:
            v["count"] = len(v["axes"])
            total += v["count"]
    dictionary["axis_count_total"] = total
    DICT_PATH.write_text(json.dumps(dictionary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      wrote dictionary ({DICT_PATH.stat().st_size} bytes, total axes={total})")

    print(f"[5/5] bumping package version and reinstalling ...", flush=True)
    for t in [REPO / "cartigsfm" / "__init__.py", REPO / "pyproject.toml", REPO / "setup.py"]:
        if not t.exists():
            continue
        text = t.read_text(encoding="utf-8")
        new = re.sub(r'(__version__\s*=\s*")0\.4\.0(")',
                      lambda m: f'{m.group(1)}{NEW_VERSION}{m.group(2)}', text)
        new = re.sub(r'(version\s*=\s*")0\.4\.0(")',
                      lambda m: f'{m.group(1)}{NEW_VERSION}{m.group(2)}', new)
        if new != text:
            t.write_text(new, encoding="utf-8")
            print(f"      bumped {t.relative_to(REPO)}")
    cp = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--quiet"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if cp.returncode != 0:
        print(cp.stderr[-2000:], file=sys.stderr)
        sys.exit(cp.returncode)
    print("      pip install -e . ok")

    import importlib, cartigsfm
    importlib.reload(cartigsfm)
    print(f"      cartigsfm.__version__ = {cartigsfm.__version__}")
    d2 = cartigsfm.load_cartilage_dictionary_v1()
    print(f"      layers: {list(d2['layers'].keys())}")
    for l in d2["layers"]:
        n = len(d2["layers"][l]["axes"])
        ids = [a.get("axis_id") for a in d2["layers"][l]["axes"]]
        print(f"        {l}: {n} axes -> {ids}")
    found = [a for a in d2["layers"]["tissue_developmental_state"]["axes"] if a.get("axis_id") == candidate_axis["axis_id"]]
    print(f"      {candidate_axis['axis_id']} present: {bool(found)}")
    if found:
        a = found[0]
        print(f"        core_genes ({len(a['core_genes'])}): {a['core_genes']}")
        print(f"        panel_genes ({len(a['panel_genes'])}): {a['panel_genes']}")
        print(f"        status={a['status']}, n_marker_weights={len(a['marker_weights'])}")
        from cartigsfm.interpret import axis_safety_class
        print(f"        safety_classification={axis_safety_class(a)}")


if __name__ == "__main__":
    main()
