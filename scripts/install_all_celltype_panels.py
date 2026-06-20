"""Install all 10 cell_subtype panels from the acc_new wilcoxon DE pass.

Bumps dictionary v1.8.3 -> v1.8.4. Source:
outputs/EBR_p4_remote/calibration/all_celltype_panels_acc.json
(based on /home/wuxu/jupyter/MJ/newh5ad/acc_new.h5ad celltype_new column,
balanced subsample 5000/group, log1p_norm X).
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
KB = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_cartigsfm_knowledge_base.json"
PANEL = Path("F:/cartifm/outputs/EBR_p4_remote/calibration/all_celltype_panels_acc.json")

TOP_TARGET_WEIGHT = 1.5
SOURCE_LABEL = "acc_new.h5ad celltype_new (10 cs subtypes, balanced 5000/group)"

# Map DE group label -> dictionary axis_id suffix.
GROUP_TO_AXIS = {
    "Effector_Metabolic_Chondrocytes":     "Effector_Metabolic_Chondrocytes",
    "Fibrocartilage_Chondrocytes":         "Fibrocartilage_Chondrocytes",
    "Homeostatic_Chondrocytes":            "Homeostatic_Chondrocytes",
    "Hypoxic_Chondrocytes":                "Hypoxic_Chondrocytes",
    "Inflammatory_Response_Chondrocytes":  "Inflammatory_Response_Chondrocytes",
    "Metabolic_Stress_Chondrocytes":       "Metabolic_Stress_Chondrocytes",
    "Prehypertrophic_Matrix_Chondrocytes": "Prehypertrophic_Matrix_Chondrocytes",
    "Progenitor_Chondrocytes":             "Progenitor_Chondrocytes",
    "Reparative_Stress_Chondrocytes":      "Reparative_Stress_Chondrocytes",
    "Superficial_Chondrocytes":            "Superficial_Zone_Chondrocytes",
}


def _normalize(score_pairs, top_target):
    if not score_pairs:
        return {}
    mx = max(w for _, w in score_pairs) or 1.0
    return {g: round(float(w) * top_target / mx, 3) for g, w in score_pairs}


def _build_axis_payload(g_entry):
    core = g_entry["core_genes"]
    panel_full = g_entry["panel_genes_full"]
    anti = g_entry["anti_genes"]

    marker_pairs = [(g["gene"], max(g["log2fc"], 0.0) * g["pct_in"]) for g in core]
    marker_weights = _normalize(marker_pairs, TOP_TARGET_WEIGHT)

    anti_pairs = [(g["gene"], abs(min(g["log2fc"], 0.0)) * g["pct_rest"]) for g in anti]
    anti_weights = _normalize(anti_pairs, TOP_TARGET_WEIGHT)

    return {
        "core_genes": [g["gene"] for g in core],
        "panel_genes": [g["gene"] for g in panel_full],
        "anti_genes": [g["gene"] for g in anti],
        "marker_weights": marker_weights,
        "anti_marker_weights": anti_weights,
        "n_target": int(g_entry.get("n_target", 0)),
        "n_other": int(g_entry.get("n_other", 0)),
        "top_marker": core[0]["gene"] if core else "",
        "top_anti": anti[0]["gene"] if anti else "",
    }


def main() -> None:
    payload = json.loads(PANEL.read_text(encoding="utf-8"))
    by_group = payload["by_group"]

    bak = DICT.with_name(DICT.name + ".pre_v1_8_4.bak")
    if not bak.exists():
        shutil.copy2(DICT, bak)
        print("backup ->", bak)

    d = json.loads(DICT.read_text(encoding="utf-8"))
    layer = d["layers"]["cell_subtype"]

    installed = {}
    skipped_groups = []
    missing_axes = []

    axes_by_id = {ax.get("axis_id"): ax for ax in layer["axes"]}
    for g, axis_suffix in GROUP_TO_AXIS.items():
        if g not in by_group:
            skipped_groups.append(g)
            continue
        axis_id = f"cell_subtype::{axis_suffix}"
        ax = axes_by_id.get(axis_id)
        if ax is None:
            missing_axes.append(axis_id)
            continue
        built = _build_axis_payload(by_group[g])
        ax["core_genes"] = built["core_genes"]
        ax["panel_genes"] = built["panel_genes"]
        ax["anti_genes"] = built["anti_genes"]
        ax["marker_weights"] = built["marker_weights"]
        ax["anti_marker_weights"] = built["anti_marker_weights"]
        ax["status"] = "production"
        ax["interpretation"] = (
            f"{axis_suffix.replace('_',' ')} state. Marker panel rebuilt "
            f"{date.today().isoformat()} by supervised wilcoxon DE on "
            f"acc_new.h5ad celltype_new (10-way 1-vs-rest, balanced 5000/group, "
            f"log1p_norm X; n_target={built['n_target']}, "
            f"n_other={built['n_other']}). Top marker {built['top_marker']}, "
            f"top anti-marker {built['top_anti']}."
        )
        ax.setdefault("evidence", {})
        ax["evidence"]["derivation"] = [
            f"Atlas-supervised DE: source={SOURCE_LABEL}; group={g}; "
            f"n_target={built['n_target']}, n_other={built['n_other']}.",
            "wilcoxon, pts>=0.20, pvals_adj<1e-5, log2fc>0 for markers; "
            "log2fc<0, pct_rest>=0.30, pct_in<=0.30 for anti-markers.",
        ]
        ax["evidence"]["internal_support"] = [
            f"Top marker {built['top_marker']} (atlas-internal log2fc).",
            "30 core markers; ranked specificity x coverage weighting.",
        ]
        lims = ax.setdefault("limitations", [])
        lims = [l for l in lims if "Panel pending rebuild" not in l
                                  and "calibrated on EBR.h5ad alone" not in l]
        extra = (
            "Panel calibrated on acc_new.h5ad atlas labels; pending "
            "independent validation on EBR.h5ad held-out fold "
            "(axis_safety_class=PENDING_INDEPENDENT_VALIDATION)."
        )
        if extra not in lims:
            lims.append(extra)
        ax["limitations"] = lims
        ax["source_files"] = [
            "outputs/EBR_p4_remote/calibration/all_celltype_panels_acc.json",
            "scripts/_remote_all_celltype_panels_acc_remote.py",
            "scripts/install_all_celltype_panels.py",
        ]
        installed[axis_id] = built["top_marker"]

    d.setdefault("changelog", []).append({
        "version": "v1.8.4",
        "date": date.today().isoformat(),
        "summary": (
            "Rebuilt all 10 cell_subtype panels from a single supervised "
            "wilcoxon DE pass on acc_new.h5ad celltype_new (balanced 5000/group, "
            "log1p_norm). 30 core / 50 panel / 30 anti per axis. Replaces the "
            "v1.6 literature-prior + v1.8.3 EBR Homeostatic-only rebuild."
        ),
    })
    d["version"] = "v1.8.4"
    DICT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dictionary updated:", DICT)

    # KB sync
    kb_bak = KB.with_name(KB.name + ".pre_v1_8_4.bak")
    if not kb_bak.exists():
        shutil.copy2(KB, kb_bak)
    kb = json.loads(KB.read_text(encoding="utf-8"))
    kb_matched = 0
    for entry in kb.get("dictionary_knowledge", []):
        aid = entry.get("axis_id")
        if aid in installed:
            ax = axes_by_id[aid]
            entry["core_genes"] = ax["core_genes"]
            entry["panel_genes"] = ax["panel_genes"]
            entry["anti_genes"] = ax["anti_genes"]
            entry["interpretation"] = ax["interpretation"]
            entry["evidence_level"] = "ATLAS_INTERNAL"
            kb_matched += 1
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("KB axis entries updated:", kb_matched)

    print()
    print("== installed", len(installed), "axes ==")
    for aid, top in installed.items():
        print(f"  {aid:<55s} top={top}")
    if skipped_groups:
        print("[skip groups]", skipped_groups)
    if missing_axes:
        print("[missing axes]", missing_axes)


if __name__ == "__main__":
    main()
