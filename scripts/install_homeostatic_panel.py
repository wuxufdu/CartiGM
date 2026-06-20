"""Install the EBR-derived Homeostatic_Chondrocytes panel into the dictionary
and the RAG knowledge base. Bumps version v1.8.2 -> v1.8.3.

- core_genes: top 30 wilcoxon markers (Homeostatic vs others, EBR celltype)
- panel_genes: top 50
- anti_genes: top 30 most-depleted-in-Homeostatic genes
- marker_weights / anti_marker_weights normalized so top weight = 1.5 (the
  bundled v1.0 convention).
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
KB = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_cartigsfm_knowledge_base.json"
PANEL = Path("F:/cartifm/outputs/EBR_p4_remote/calibration/homeostatic_panel.json")

AXIS_ID = "cell_subtype::Homeostatic_Chondrocytes"
TOP_TARGET_WEIGHT = 1.5


def _normalize(score_pairs, top_target):
    """Given list[(gene, raw_weight)], rescale so max -> top_target."""
    if not score_pairs:
        return {}
    mx = max(w for _, w in score_pairs) or 1.0
    return {g: round(float(w) * top_target / mx, 3) for g, w in score_pairs}


def main() -> None:
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    core = panel["core_genes"]
    panel_full = panel["panel_genes_full"]
    anti = panel["anti_genes"]

    # raw weight = log2fc * pct_in (specificity x coverage); same flavor as
    # the metabolism calibration, supervised version.
    marker_pairs = [(g["gene"], max(g["log2fc"], 0.0) * g["pct_in"]) for g in core]
    marker_weights = _normalize(marker_pairs, TOP_TARGET_WEIGHT)

    anti_pairs = [(g["gene"], abs(min(g["log2fc"], 0.0)) * g["pct_rest"]) for g in anti]
    anti_weights = _normalize(anti_pairs, TOP_TARGET_WEIGHT)

    core_genes_list = [g["gene"] for g in core]
    panel_genes_list = [g["gene"] for g in panel_full]
    anti_genes_list = [g["gene"] for g in anti]

    # ----- backup -----
    bak = DICT.with_name(DICT.name + ".pre_homeostatic_panel.bak")
    if not bak.exists():
        shutil.copy2(DICT, bak)
        print("backup ->", bak)

    d = json.loads(DICT.read_text(encoding="utf-8"))
    layer = d["layers"]["cell_subtype"]
    matched = 0
    for ax in layer["axes"]:
        if ax.get("axis_id") == AXIS_ID:
            ax["core_genes"] = core_genes_list
            ax["panel_genes"] = panel_genes_list
            ax["anti_genes"] = anti_genes_list
            ax["marker_weights"] = marker_weights
            ax["anti_marker_weights"] = anti_weights
            ax["status"] = "production"
            ax["interpretation"] = (
                "Homeostatic chondrocyte state. Marker panel rebuilt 2026-06-20 "
                "by supervised wilcoxon on EBR.h5ad celltype labels "
                "(Homeostatic_Chondrocytes vs the other 6 chondrocyte states, "
                "9493 vs 22788 cells, log1p_norm layer). Top markers reflect "
                "mature cartilage ECM (COL2A1, COL9A1/2/3, COL11A1/2, FMOD, "
                "PRELP, SMOC2, SCRG1, S100B, IGFBP7); anti-markers reflect "
                "inflammation / stress programs (CXCL8, LOX, CHI3L1, AKR1C1, "
                "IRAK2)."
            )
            ax.setdefault("evidence", {})
            ax["evidence"]["derivation"] = [
                "Atlas-derived axis renamed from Matrix_Maintenance "
                "(cartilage_dictionary v1.8.2 -> v1.8.3).",
                "Panel rebuilt by supervised wilcoxon on EBR.h5ad "
                "(Homeostatic_Chondrocytes vs others; 9493 vs 22788 cells; "
                "log1p_norm layer; pts>=0.20, pvals_adj<1e-5, log2fc>0 for "
                "markers; pvals_adj<1e-5, pct_rest>=0.30, pct_in<=0.30, "
                "log2fc<0 for anti-markers).",
            ]
            ax["evidence"]["internal_support"] = [
                f"Top marker COL11A1 (log2fc=1.46, pct_in=0.91, pct_rest=0.68).",
                "20 of 30 core markers are cartilage ECM / matrix genes.",
            ]
            lims = ax.setdefault("limitations", [])
            lims = [l for l in lims if "Panel pending rebuild" not in l]
            extra = (
                "Panel calibrated on EBR.h5ad alone; pending independent "
                "validation on a held-out atlas (axis_safety_class="
                "PENDING_INDEPENDENT_VALIDATION)."
            )
            if extra not in lims:
                lims.append(extra)
            ax["limitations"] = lims
            ax["source_files"] = [
                "outputs/EBR_p4_remote/calibration/homeostatic_panel.json",
                "scripts/_remote_homeostatic_panel_remote.py",
                "scripts/install_homeostatic_panel.py",
            ]
            matched += 1
    assert matched == 1, f"expected 1 axis match, got {matched}"

    d.setdefault("changelog", []).append({
        "version": "v1.8.3",
        "date": date.today().isoformat(),
        "summary": (
            "Rebuilt cell_subtype::Homeostatic_Chondrocytes panel from a "
            "supervised EBR wilcoxon DE analysis (n=9493 target vs 22788 "
            "background). 30 core markers, 50 panel markers, 30 anti-markers. "
            "Aimed at fixing the v1.8.2 P4 0% top-1 on the 8606-cell "
            "Homeostatic majority clusters."
        ),
    })
    d["version"] = "v1.8.3"
    DICT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dictionary updated:", DICT)

    # ----- KB -----
    kb_bak = KB.with_name(KB.name + ".pre_homeostatic_panel.bak")
    if not kb_bak.exists():
        shutil.copy2(KB, kb_bak)
    kb = json.loads(KB.read_text(encoding="utf-8"))
    kb_matched = 0
    for entry in kb.get("dictionary_knowledge", []):
        if entry.get("axis_id") == AXIS_ID:
            entry["core_genes"] = core_genes_list
            entry["panel_genes"] = panel_genes_list
            entry["anti_genes"] = anti_genes_list
            entry["interpretation"] = (
                "Homeostatic chondrocyte state; panel rebuilt 2026-06-20 by "
                "supervised wilcoxon DE on EBR.h5ad celltype labels."
            )
            entry["evidence_level"] = "ATLAS_INTERNAL"
            kb_matched += 1
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("KB axis entries updated:", kb_matched)

    print()
    print("Top 10 markers (weight):")
    for g, w in list(marker_weights.items())[:10]:
        print(f"  {g:>10s}  {w}")
    print("Top 10 anti-markers (weight):")
    for g, w in list(anti_weights.items())[:10]:
        print(f"  {g:>10s}  {w}")


if __name__ == "__main__":
    main()
