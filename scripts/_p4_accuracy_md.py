"""Render the markdown accuracy report from p4_accuracy_*.tsv."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path("F:/cartifm/outputs/EBR_p4_remote")
TSV = ROOT / "tsv"
DOC = ROOT / "docs"


def short(a):
    return a.split("::", 1)[1] if isinstance(a, str) and "::" in a else a


def main() -> None:
    df = pd.read_csv(TSV / "p4_accuracy_per_group.tsv", sep="\t",
                     dtype={"cluster": str})
    overall = pd.read_csv(TSV / "p4_accuracy_overall.tsv", sep="\t").iloc[0]
    by_ct = pd.read_csv(TSV / "p4_accuracy_by_celltype.tsv", sep="\t")
    by_b = pd.read_csv(TSV / "p4_accuracy_by_batch.tsv", sep="\t")
    conf = pd.read_csv(TSV / "p4_confusion_celltype_to_axis.tsv", sep="\t",
                       index_col=0)

    n_groups = len(df)
    n_cells = int(df["total_n"].sum())
    L = []
    L.append("# P4 EBR Accuracy Report")
    L.append("")
    L.append("- dictionary: cartilage_dictionary_v1 **v1.8.5** (53 axes; 7 EBR-present cell_subtype panels refit by in-domain supervised wilcoxon DE on EBR.h5ad celltype labels; Hypoxic / Metabolic_Stress / Superficial_Zone keep v1.8.4 acc_new-supervised panels)")
    L.append(f"- query: EBR.h5ad ({n_cells} cells, 29471 genes; ear=16436, rib=8204, nose=7641)")
    L.append(f"- evaluation unit: (batch x leiden_res1) cluster, n={n_groups}, total cells={n_cells}")
    L.append("- correct = projection top-k axis contains the expected axis (mapped from majority celltype)")
    L.append("- v1.8.5 EBR-supervised panels: Effector_Metabolic top=SOD2, Fibrocartilage top=TMSB4X, Homeostatic top=COL11A1, Inflammatory_Response top=SLC4A7, Prehypertrophic top=FBXO2, Progenitor top=RYBP, Reparative_Stress top=GADD45B.")
    L.append("")
    L.append("## 1. headline accuracy")
    L.append("")
    L.append("| layer | metric | cluster-weighted | cell-weighted |")
    L.append("|---|---|---|---|")
    cs1 = int((df['top1_match'] * df['total_n']).sum())
    cs3 = int((df['top3_match'] * df['total_n']).sum())
    tds1 = int((df['tds_match'] * df['total_n']).sum())
    L.append(f"| cell_subtype | top-1 | {overall['cs_top1_cluster']*100:.1f}% ({df['top1_match'].sum()}/{n_groups}) | {overall['cs_top1_cell']*100:.1f}% ({cs1}/{n_cells}) |")
    L.append(f"| cell_subtype | top-3 | {overall['cs_top3_cluster']*100:.1f}% ({df['top3_match'].sum()}/{n_groups}) | {overall['cs_top3_cell']*100:.1f}% ({cs3}/{n_cells}) |")
    L.append(f"| tissue_state | top-1 | {overall['tds_top1_cluster']*100:.1f}% ({df['tds_match'].sum()}/{n_groups}) | {overall['tds_top1_cell']*100:.1f}% ({tds1}/{n_cells}) |")
    L.append("")
    L.append("## 2. accuracy by ground-truth celltype")
    L.append("")
    L.append("| celltype | clusters | cells | top1 (cluster) | top1 (cell) | top3 (cluster) | top3 (cell) |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in by_ct.iterrows():
        L.append(f"| {r['majority_celltype']} | {int(r['clusters'])} | {int(r['cells'])} | {r['top1_cluster']*100:.1f}% | {r['top1_cell']*100:.1f}% | {r['top3_cluster']*100:.1f}% | {r['top3_cell']*100:.1f}% |")
    L.append("")
    L.append("## 3. accuracy by batch")
    L.append("")
    L.append("| batch | clusters | cells | cs top1 (cluster) | cs top1 (cell) | cs top3 (cluster) | tissue top1 (cluster) |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in by_b.iterrows():
        L.append(f"| {r['batch']} | {int(r['clusters'])} | {int(r['cells'])} | {r['cs_top1_cluster']*100:.1f}% | {r['cs_top1_cell']*100:.1f}% | {r['cs_top3_cluster']*100:.1f}% | {r['tds_top1_cluster']*100:.1f}% |")
    L.append("")
    L.append("## 4. confusion: ground-truth celltype -> top-1 cell_subtype axis (cell-weighted)")
    L.append("")
    cols = list(conf.columns)
    L.append("| celltype \\ top1 | " + " | ".join(short(c) for c in cols) + " |")
    L.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for ct, row in conf.iterrows():
        cells = [str(int(row[c])) for c in cols]
        L.append(f"| {ct} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 5. expected-axis rank distribution (cell_subtype)")
    L.append("")
    rd = df["expected_rank"].value_counts(dropna=False).sort_index()
    L.append("| rank of expected axis | clusters | cells |")
    L.append("|---|---|---|")
    for r, n in rd.items():
        cells = int(df.loc[df["expected_rank"].astype(str) == str(r), "total_n"].sum())
        rk = "NA" if pd.isna(r) else int(r)
        L.append(f"| {rk} | {n} | {cells} |")
    L.append("")
    L.append("## 6. caveats")
    L.append("- v1.8.5 (in-domain DE) brings cs top-1 cell from 37.8% (v1.8.4 acc_new) to {:.1f}% and top-3 cell to {:.1f}%; biggest gains on Effector_Metabolic, Prehypertrophic_Matrix, Reparative_Stress.".format(overall['cs_top1_cell']*100, overall['cs_top3_cell']*100))
    L.append("- Homeostatic remains the residual confusion: 33% of its cells route to Effector_Metabolic; the natural next steps are tightening Effector_Metabolic anti_genes (against COL11A1/COL9A3/COL2A1/SCRG1) and adding ECM mature genes to Homeostatic anti_genes when scored against Effector_Metabolic.")
    L.append("- Tissue layer is evaluated per (batch x cluster); fibrocartilage-dominated clusters can route to Fibrocartilage_Meniscus regardless of batch (rib top-1 cluster 57.1%); at the batch-mean level each batch's expected tissue axis remains top.")

    DOC.mkdir(parents=True, exist_ok=True)
    out = DOC / "P4_EBR_ACCURACY_REPORT.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
