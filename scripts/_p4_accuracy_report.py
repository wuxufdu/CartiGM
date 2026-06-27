"""P4 accuracy report on EBR.h5ad for cartilage_dictionary v1.8.1."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("F:/cartifm/outputs/EBR_p4_remote")
TSV = ROOT / "tsv"
DOC = ROOT / "docs"

CT2AXIS = {
    "Effector_Metabolic_Chondrocytes": "cell_subtype::Effector_Metabolic_Chondrocytes",
    "Homeostatic_Chondrocytes": "cell_subtype::Homeostatic_Chondrocytes",
    "Inflammatory_Response_Chondrocytes": "cell_subtype::Inflammatory_Response_Chondrocytes",
    "Prehypertrophic_Matrix_Chondrocytes": "cell_subtype::Prehypertrophic_Matrix_Chondrocytes",
    "Reparative_Stress_Chondrocytes": "cell_subtype::Reparative_Stress_Chondrocytes",
    "Fibrocartilage_Chondrocytes": "cell_subtype::Fibrocartilage_Chondrocytes",
    "Progenitor_Chondrocytes": "cell_subtype::Progenitor_Chondrocytes",
    "Hypoxic_Chondrocytes": "cell_subtype::Hypoxic_Chondrocytes",
    "Metabolic_Stress_Chondrocytes": "cell_subtype::Metabolic_Stress_Chondrocytes",
    "Superficial_Chondrocytes": "cell_subtype::Superficial_Zone_Chondrocytes",
    "Superficial_Zone_Chondrocytes": "cell_subtype::Superficial_Zone_Chondrocytes",
}
SCORE_RENAME = {
    "cell_subtype::Stromal_Matrix_Chondrocytes": "cell_subtype::Progenitor_Chondrocytes",
    "cell_subtype::Matrix_Maintenance_Chondrocytes": "cell_subtype::Homeostatic_Chondrocytes",
}
TISSUE2AXIS = {
    "ear": "tissue_developmental_state::ElasticCartilage_Auricular",
    "nose": "tissue_developmental_state::Nasal_Septum_Cartilage",
    "rib": "tissue_developmental_state::Hyaline_ArticularCartilage",
}


def _build():
    scores = pd.read_csv(TSV / "p4_sample_cluster_three_layer_scores.tsv", sep="\t")
    scores["axis_id"] = scores["axis_id"].replace(SCORE_RENAME)
    p = scores["sample"].str.split("|", expand=True)
    scores["batch"] = p[0]
    scores["cluster"] = p[2]

    cross = pd.read_csv(TSV / "p4_celltype_crosstab.tsv", sep="\t",
                        dtype={"leiden_res1": str, "leiden_res0_5": str})
    if "leiden_res1" in cross.columns:
        cross["cluster"] = cross["leiden_res1"]
    else:
        cross["cluster"] = cross["leiden_res0_5"]
    idx = cross.groupby(["batch", "cluster"])["n"].idxmax()
    maj = cross.loc[idx, ["batch", "cluster", "celltype", "n"]].rename(
        columns={"celltype": "majority_celltype", "n": "majority_n"})
    tot = cross.groupby(["batch", "cluster"])["n"].sum().reset_index().rename(
        columns={"n": "total_n"})
    maj = maj.merge(tot, on=["batch", "cluster"])
    maj["majority_frac"] = (maj["majority_n"] / maj["total_n"]).round(3)
    maj["expected_axis"] = maj["majority_celltype"].map(CT2AXIS)

    cs = scores[scores["layer"] == "cell_subtype"].copy()
    cs = cs.sort_values(["batch", "cluster", "score"],
                        ascending=[True, True, False])
    cs["rank"] = cs.groupby(["batch", "cluster"]).cumcount() + 1
    rk = cs.set_index(["batch", "cluster", "axis_id"])["rank"].to_dict()
    top1 = cs[cs["rank"] == 1][["batch", "cluster", "axis_id", "score"]].rename(
        columns={"axis_id": "top1_axis", "score": "top1_score"})
    top3 = (cs[cs["rank"] <= 3].groupby(["batch", "cluster"])["axis_id"]
              .apply(list).reset_index().rename(columns={"axis_id": "top3_axes"}))

    df = maj.merge(top1, on=["batch", "cluster"]).merge(top3, on=["batch", "cluster"])
    df["top1_match"] = df["expected_axis"] == df["top1_axis"]
    df["top3_match"] = df.apply(lambda r: r["expected_axis"] in r["top3_axes"], axis=1)
    df["expected_rank"] = df.apply(
        lambda r: rk.get((r["batch"], r["cluster"], r["expected_axis"])), axis=1)

    tds = scores[scores["layer"] == "tissue_developmental_state"].copy()
    tds = tds.sort_values(["batch", "cluster", "score"],
                          ascending=[True, True, False])
    tds_top = (tds.groupby(["batch", "cluster"]).head(1)
                  [["batch", "cluster", "axis_id"]]
                  .rename(columns={"axis_id": "tds_top1_axis"}))
    df = df.merge(tds_top, on=["batch", "cluster"])
    df["tds_expected"] = df["batch"].map(TISSUE2AXIS)
    df["tds_match"] = df["tds_expected"] == df["tds_top1_axis"]
    return df


def _aggregates(df: pd.DataFrame):
    n_groups = len(df)
    n_cells = int(df["total_n"].sum())
    overall = {
        "groups": n_groups,
        "cells": n_cells,
        "cs_top1_cluster": df["top1_match"].mean(),
        "cs_top1_cell": (df["top1_match"] * df["total_n"]).sum() / n_cells,
        "cs_top3_cluster": df["top3_match"].mean(),
        "cs_top3_cell": (df["top3_match"] * df["total_n"]).sum() / n_cells,
        "tds_top1_cluster": df["tds_match"].mean(),
        "tds_top1_cell": (df["tds_match"] * df["total_n"]).sum() / n_cells,
    }
    by_ct = df.groupby("majority_celltype").agg(
        clusters=("top1_match", "size"),
        cells=("total_n", "sum"),
        top1_cluster=("top1_match", "mean"),
        top3_cluster=("top3_match", "mean"),
    ).reset_index()
    cw = (df.assign(w1=df["top1_match"] * df["total_n"],
                    w3=df["top3_match"] * df["total_n"])
            .groupby("majority_celltype")
            .agg(w1=("w1", "sum"), w3=("w3", "sum"), cells=("total_n", "sum")))
    cw["top1_cell"] = cw["w1"] / cw["cells"]
    cw["top3_cell"] = cw["w3"] / cw["cells"]
    by_ct = by_ct.merge(cw[["top1_cell", "top3_cell"]],
                        left_on="majority_celltype", right_index=True)
    by_ct = by_ct.sort_values("cells", ascending=False)
    by_b = df.groupby("batch").agg(
        clusters=("top1_match", "size"),
        cells=("total_n", "sum"),
        cs_top1_cluster=("top1_match", "mean"),
        cs_top3_cluster=("top3_match", "mean"),
        tds_top1_cluster=("tds_match", "mean"),
    ).reset_index()
    cw_b = (df.assign(w1=df["top1_match"] * df["total_n"],
                      w3=df["top3_match"] * df["total_n"],
                      wt=df["tds_match"] * df["total_n"])
              .groupby("batch")
              .agg(w1=("w1", "sum"), w3=("w3", "sum"), wt=("wt", "sum"),
                   cells=("total_n", "sum")))
    cw_b["cs_top1_cell"] = cw_b["w1"] / cw_b["cells"]
    cw_b["cs_top3_cell"] = cw_b["w3"] / cw_b["cells"]
    cw_b["tds_top1_cell"] = cw_b["wt"] / cw_b["cells"]
    by_b = by_b.merge(cw_b[["cs_top1_cell", "cs_top3_cell", "tds_top1_cell"]],
                      left_on="batch", right_index=True)
    conf = (df.groupby(["majority_celltype", "top1_axis"])["total_n"].sum()
              .unstack(fill_value=0))
    return overall, by_ct, by_b, conf


def main() -> None:
    df = _build()
    df.to_csv(TSV / "p4_accuracy_per_group.tsv", sep="\t", index=False)
    overall, by_ct, by_b, conf = _aggregates(df)
    pd.DataFrame([overall]).to_csv(TSV / "p4_accuracy_overall.tsv", sep="\t", index=False)
    by_ct.to_csv(TSV / "p4_accuracy_by_celltype.tsv", sep="\t", index=False)
    by_b.to_csv(TSV / "p4_accuracy_by_batch.tsv", sep="\t", index=False)
    conf.to_csv(TSV / "p4_confusion_celltype_to_axis.tsv", sep="\t")
    print("ok; cs top1 cluster:", overall["cs_top1_cluster"])


if __name__ == "__main__":
    main()
