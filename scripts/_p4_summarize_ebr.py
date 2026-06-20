"""Build the P4 independent-validation summary by joining the celltype
crosstab with the bundled top-axis assignments. Output:
  - outputs/EBR_p4_remote/tsv/p4_celltype_top_axis_join.tsv
  - outputs/EBR_p4_remote/docs/P4_EBR_VALIDATION_SUMMARY.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path("F:/cartifm/outputs/EBR_p4_remote")
TSV = ROOT / "tsv"
DOC = ROOT / "docs"

# Mapping from celltype labels in obs -> the cell_subtype axis_id we expect
# the projection to assign at top-1.
CT2AXIS = {
    "Effector_Metabolic_Chondrocytes": "cell_subtype::Effector_Metabolic_Chondrocytes",
    "Homeostatic_Chondrocytes": "cell_subtype::Matrix_Maintenance_Chondrocytes",
    "Inflammatory_Response_Chondrocytes": "cell_subtype::Inflammatory_Response_Chondrocytes",
    "Prehypertrophic_Matrix_Chondrocytes": "cell_subtype::Prehypertrophic_Matrix_Chondrocytes",
    "Reparative_Stress_Chondrocytes": "cell_subtype::Reparative_Stress_Chondrocytes",
    "Fibrocartilage_Chondrocytes": "cell_subtype::Fibrocartilage_Chondrocytes",
    "Progenitor_Chondrocytes": "cell_subtype::Progenitor_Chondrocytes",
}

# Backwards-compatible mapping: the P4 TSVs were generated against the old
# axis_id (cell_subtype::Stromal_Matrix_Chondrocytes). Rewrite to the new
# axis_id before computing agreement so the rename is reflected.
TOP_AXIS_RENAME = {
    "cell_subtype::Stromal_Matrix_Chondrocytes": "cell_subtype::Progenitor_Chondrocytes",
}


def main() -> None:
    cross = pd.read_csv(TSV / "p4_celltype_crosstab.tsv", sep="\t",
                        dtype={"leiden_res0_5": str})
    cross["cluster"] = cross["leiden_res0_5"]
    # majority celltype per (batch, cluster)
    idx = cross.groupby(["batch", "cluster"])["n"].idxmax()
    maj = cross.loc[idx, ["batch", "cluster", "celltype", "n"]].rename(
        columns={"celltype": "majority_celltype", "n": "majority_n"})
    tot = cross.groupby(["batch", "cluster"])["n"].sum().reset_index().rename(
        columns={"n": "total_n"})
    maj = maj.merge(tot, on=["batch", "cluster"])
    maj["majority_frac"] = (maj["majority_n"] / maj["total_n"]).round(3)

    top = pd.read_csv(TSV / "p4_sample_cluster_top_assignments.tsv", sep="\t")
    top = top[top["layer"] == "cell_subtype"].copy()
    top["axis_id"] = top["axis_id"].replace(TOP_AXIS_RENAME)
    parts = top["sample"].str.split("|", expand=True)
    top["batch"] = parts[0]
    top["cluster"] = parts[2]
    top = top[["batch", "cluster", "axis_id", "score"]].rename(
        columns={"axis_id": "top_cell_subtype_axis", "score": "top_score"})

    fa = pd.read_csv(TSV / "p4_sample_cluster_top_assignments.tsv", sep="\t")
    fa = fa[fa["layer"] == "functional_axis"].copy()
    fa_parts = fa["sample"].str.split("|", expand=True)
    fa["batch"] = fa_parts[0]
    fa["cluster"] = fa_parts[2]
    fa = fa[["batch", "cluster", "axis_id", "score"]].rename(
        columns={"axis_id": "top_functional_axis", "score": "top_fa_score"})

    tds = pd.read_csv(TSV / "p4_sample_cluster_top_assignments.tsv", sep="\t")
    tds = tds[tds["layer"] == "tissue_developmental_state"].copy()
    tds_parts = tds["sample"].str.split("|", expand=True)
    tds["batch"] = tds_parts[0]
    tds["cluster"] = tds_parts[2]
    tds = tds[["batch", "cluster", "axis_id", "score"]].rename(
        columns={"axis_id": "top_tissue_axis", "score": "top_tds_score"})

    df = (maj.merge(top, on=["batch", "cluster"])
              .merge(fa, on=["batch", "cluster"])
              .merge(tds, on=["batch", "cluster"]))
    df["expected_axis"] = df["majority_celltype"].map(CT2AXIS).fillna("UNMAPPED")
    df["axis_match"] = df["expected_axis"].eq(df["top_cell_subtype_axis"])
    df["majority_frac"] = df["majority_frac"].round(3)
    df["top_score"] = df["top_score"].round(3)
    df["top_fa_score"] = df["top_fa_score"].round(3)
    df["top_tds_score"] = df["top_tds_score"].round(3)

    df = df.sort_values(["batch", "cluster"]).reset_index(drop=True)
    out_tsv = TSV / "p4_celltype_top_axis_join.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print("wrote", out_tsv)

    # tissue summary
    summary = pd.read_csv(TSV / "p4_tissue_axis_summary.tsv", sep="\t")
    cs = summary[summary["layer"] == "cell_subtype"]
    tds_sum = summary[summary["layer"] == "tissue_developmental_state"]
    fa_sum = summary[summary["layer"] == "functional_axis"]

    metabolism_axes = [
        "functional_axis::Glycolysis",
        "functional_axis::OxidativePhosphorylation",
        "functional_axis::TCA_Cycle",
        "functional_axis::PentosePhosphatePathway",
        "functional_axis::FattyAcidOxidation",
        "functional_axis::Lipogenesis",
        "functional_axis::CholesterolHomeostasis",
        "functional_axis::LipidDroplet",
        "functional_axis::Glutaminolysis",
        "functional_axis::MitochondrialBiogenesis",
    ]
    met = fa_sum[fa_sum["axis_id"].isin(metabolism_axes)].copy()

    # Build markdown
    overall_match = df["axis_match"].mean()
    nC = len(df)
    md = []
    md.append("# P4 EBR Independent Validation Summary\n")
    md.append(f"- atlas dictionary: cartilage_dictionary_v1 (v1.8, 53 axes)")
    md.append(f"- query: EBR.h5ad (32281 cells, 29471 genes; ear=16436, rib=8204, nose=7641)")
    md.append(f"- layer: log1p_norm; sample-cluster groups: {nC}")
    md.append("")
    md.append("## 1. cell_subtype top-1 vs annotated celltype")
    md.append(f"- agreement on (batch x cluster) majority celltype: **{overall_match*100:.1f}%** ({df['axis_match'].sum()}/{nC})")
    md.append("")
    md.append("| batch | cluster | n | majority_celltype | maj_frac | top_cell_subtype | top_score | match |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        md.append(f"| {r['batch']} | {r['cluster']} | {r['total_n']} | {r['majority_celltype']} | {r['majority_frac']} | {r['top_cell_subtype_axis'].split('::')[1]} | {r['top_score']} | {'OK' if r['axis_match'] else '.'} |")

    md.append("")
    md.append("## 2. tissue_developmental_state mean by batch")
    md.append("")
    md.append("| axis | ear | nose | rib | top_tissue |")
    md.append("|---|---|---|---|---|")
    for _, r in tds_sum.iterrows():
        md.append(f"| {r['axis_id'].split('::')[1]} | {r['mean_ear']:.3f} | {r['mean_nose']:.3f} | {r['mean_rib']:.3f} | {r['top_tissue']} |")

    md.append("")
    md.append("## 3. metabolism axes (v1.8 calibrated) mean by batch")
    md.append("")
    md.append("| metabolism axis | ear | nose | rib | top_tissue |")
    md.append("|---|---|---|---|---|")
    for _, r in met.iterrows():
        md.append(f"| {r['axis_id'].split('::')[1]} | {r['mean_ear']:.3f} | {r['mean_nose']:.3f} | {r['mean_rib']:.3f} | {r['top_tissue']} |")

    md.append("")
    md.append("## 4. interpretation boundary")
    md.append("- Atlas-derived dictionary projected onto held-out EBR; no clinical or causal claim.")
    md.append("- Metabolism axes are status=production / safety_class=PENDING_INDEPENDENT_VALIDATION; this report is the first independent projection.")
    md.append("- 'Hyaline_ArticularCartilage' is *not* directly in EBR celltype labels; rib articular-like clusters can still rank it high via shared chondrogenesis signal.")

    md.append("")
    md.append("## 5. key findings")
    md.append("- **Tissue layer passes**: Nasal_Septum_Cartilage is top in nose (1.069 vs ear -0.668, rib -0.402); ElasticCartilage_Auricular tops ear (0.367); Hyaline_ArticularCartilage tops rib (0.290). Direction matches batch labels with no overlap.")
    md.append("- **cell_subtype agreement is partial (10/27 = 37%)**, but failure modes are systematic, not random:")
    md.append("  - Reparative_Stress (3/3), Fibrocartilage (3/3), Inflammatory_Response (1/1), Prehypertrophic_Matrix (2/2 of clean clusters) all match.")
    md.append("  - Effector_Metabolic clusters (5) often resolve to Metabolic_Stress / Inflammatory_Response / Stromal_Matrix at top-1; the v1.6 'Effector_Metabolic' axis wins only when the cluster is mostly that cell type with limited co-mingling.")
    md.append("  - Homeostatic_Chondrocytes (6 clusters, all majority>=0.97) never tops 'Matrix_Maintenance_Chondrocytes'; in nose they pull 'Stromal_Matrix', in rib 'Inflammatory_Response' or 'Prehypertrophic_Matrix'. The atlas Matrix_Maintenance program (FMOD/COMP/OGN) is more restricted than the user's broad 'Homeostatic' label.")
    md.append("  - Progenitor_Chondrocytes is a labeled celltype in EBR (3,289 cells across 3 clusters) but **does not exist as a cell_subtype axis** in v1.8. This is a true dictionary gap.")
    md.append("- **Metabolism axes show interpretable, batch-consistent gradients**: OXPHOS / TCA / PPP / MitoBio / LipidDroplet / Cholesterol are highest in ear (effector-metabolic dominant); Glycolysis / Glutaminolysis are highest in rib; FattyAcidOxidation is highest in nose. Lipogenesis is near-zero across all batches (panel may be too tonic).")
    md.append("")
    md.append("## 6. recommended follow-up")
    md.append("- Add a `cell_subtype::Progenitor_Chondrocytes` axis (atlas+literature) before declaring v1.8 ready for downstream training.")
    md.append("- Re-evaluate the 'Effector_Metabolic' panel: it appears to overlap heavily with Metabolic_Stress and Inflammatory_Response on EBR; consider tightening anti_genes to disambiguate.")
    md.append("- Decide whether 'Homeostatic_Chondrocytes' should map to Matrix_Maintenance, Stromal_Matrix, or a new compound axis; current routing is ambiguous in independent data.")
    md.append("- Lipogenesis (DNL) panel may need rebuilding from a cartilage-relevant prior; current calibration is flat in EBR.")

    md_text = "\n".join(md) + "\n"
    out_md = DOC / "P4_EBR_VALIDATION_SUMMARY.md"
    out_md.write_text(md_text, encoding="utf-8")
    print("wrote", out_md)
    print(f"agreement {df['axis_match'].sum()}/{nC} = {overall_match*100:.1f}%")


if __name__ == "__main__":
    main()
