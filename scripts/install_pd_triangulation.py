"""P-D: triangulate anti_genes between mutually-confused cell_subtype axes.

v1.8.5 confusion (P4 on EBR.h5ad) shows:
  - 33% Homeostatic -> Effector_Metabolic
  - 12% Effector_Metabolic -> Inflammatory_Response
  - mutual leakage among Inflammatory / Reparative / Effector cores.

We tighten anti_genes per axis with curated cross-axis core injections, then
renormalize each axis's anti weights so the new top stays at 1.5. Bumps
v1.8.5 -> v1.8.6.
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
KB = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_cartigsfm_knowledge_base.json"

TOP_TARGET_WEIGHT = 1.5

HOMEOSTATIC_ECM = [
    ("COL2A1", 1.5), ("COL11A1", 1.4), ("COL9A1", 1.3), ("COL9A2", 1.3),
    ("COL9A3", 1.3), ("COL11A2", 1.2), ("FMOD", 1.2), ("PRELP", 1.1),
    ("SCRG1", 1.1), ("SMOC2", 1.1), ("CILP", 1.0), ("CILP2", 1.0),
    ("ITM2A", 0.9), ("S100B", 0.9), ("DCN", 0.9), ("COMP", 0.9),
]
INFLAM_CORE = [
    ("MMP1", 1.4), ("MMP3", 1.4), ("BMP2", 1.2), ("SLC4A7", 1.1),
    ("TNFRSF11B", 1.1), ("SERPINE2", 1.0), ("CXCL8", 1.0), ("SOD2", 1.0),
    ("CD55", 0.9),
]
REPARATIVE_IEG = [
    ("FOS", 1.4), ("JUN", 1.3), ("FOSB", 1.3), ("ATF3", 1.2),
    ("JUNB", 1.2), ("EGR1", 1.1), ("DNAJB1", 1.0), ("HSPA1A", 1.0),
    ("GADD45B", 1.0), ("IER2", 0.9), ("IER5", 0.9), ("PPP1R15A", 0.9),
]
EFFECTOR_METABOLIC_CORE = [
    ("SOD2", 1.4), ("HMGA1", 1.3), ("TXN", 1.2), ("TALDO1", 1.1),
    ("PDCD5", 1.0), ("PSMA7", 1.0), ("PSMD8", 0.9), ("ATP6V0E1", 0.9),
    ("SEC61G", 0.9), ("RAN", 0.9), ("TXNRD1", 1.1), ("FOSL1", 1.0),
    ("HMOX1", 0.9), ("G0S2", 0.9),
]
METABOLIC_STRESS_CORE = [
    ("MIF", 1.3), ("NDRG1", 1.2), ("RPS16", 1.0), ("ENO1", 1.0),
    ("RPL22", 0.9), ("NPM1", 0.9), ("RPL36AL", 0.9), ("ZFAS1", 0.9),
    ("RPS21", 0.8), ("RACK1", 0.8),
]
SUPERFICIAL_CORE = [
    ("PRG4", 1.5), ("CRTAC1", 1.2), ("HTRA1", 1.0), ("FN1", 1.0),
    ("ABI3BP", 0.9), ("CRLF1", 0.9), ("TIMP3", 0.9), ("TNXB", 0.8),
]

CROSS_ANTI = {
    "cell_subtype::Effector_Metabolic_Chondrocytes": (
        HOMEOSTATIC_ECM + REPARATIVE_IEG + INFLAM_CORE[:5] + SUPERFICIAL_CORE[:4]
    ),
    "cell_subtype::Homeostatic_Chondrocytes": (
        EFFECTOR_METABOLIC_CORE + INFLAM_CORE[:6] + METABOLIC_STRESS_CORE[:6] + SUPERFICIAL_CORE[:4]
    ),
    "cell_subtype::Inflammatory_Response_Chondrocytes": (
        EFFECTOR_METABOLIC_CORE[:8] + REPARATIVE_IEG[:6] + HOMEOSTATIC_ECM[:8]
    ),
    "cell_subtype::Metabolic_Stress_Chondrocytes": (
        EFFECTOR_METABOLIC_CORE + HOMEOSTATIC_ECM[:10] + INFLAM_CORE[:5]
    ),
    "cell_subtype::Reparative_Stress_Chondrocytes": (
        EFFECTOR_METABOLIC_CORE[:8] + INFLAM_CORE[:6] + HOMEOSTATIC_ECM[:8]
    ),
    "cell_subtype::Hypoxic_Chondrocytes": (
        EFFECTOR_METABOLIC_CORE[:8] + HOMEOSTATIC_ECM[:8] + INFLAM_CORE[:5]
    ),
    "cell_subtype::Prehypertrophic_Matrix_Chondrocytes": (
        EFFECTOR_METABOLIC_CORE[:6] + INFLAM_CORE[:5] + REPARATIVE_IEG[:5]
    ),
    "cell_subtype::Fibrocartilage_Chondrocytes": (
        HOMEOSTATIC_ECM[:6] + EFFECTOR_METABOLIC_CORE[:5]
    ),
    "cell_subtype::Progenitor_Chondrocytes": (
        EFFECTOR_METABOLIC_CORE[:6] + HOMEOSTATIC_ECM[:8] + INFLAM_CORE[:5]
    ),
    "cell_subtype::Superficial_Zone_Chondrocytes": (
        EFFECTOR_METABOLIC_CORE[:6] + HOMEOSTATIC_ECM[:6] + REPARATIVE_IEG[:4]
    ),
}


def _renormalize(weights, top_target):
    if not weights:
        return weights
    mx = max(weights.values()) or 1.0
    return {g: round(float(w) * top_target / mx, 3) for g, w in weights.items()}


def main() -> None:
    bak = DICT.with_name(DICT.name + ".pre_v1_8_6.bak")
    if not bak.exists():
        shutil.copy2(DICT, bak)
        print("backup ->", bak)

    d = json.loads(DICT.read_text(encoding="utf-8"))
    layer = d["layers"]["cell_subtype"]
    axes_by_id = {ax.get("axis_id"): ax for ax in layer["axes"]}

    summary = {}
    for axis_id, extra in CROSS_ANTI.items():
        ax = axes_by_id.get(axis_id)
        if ax is None:
            print("[skip] axis missing:", axis_id)
            continue
        anti = list(ax.get("anti_genes", []))
        weights = dict(ax.get("anti_marker_weights", {}))
        own_protected = set(ax.get("core_genes", [])[:30])
        own_protected.update(ax.get("panel_genes", [])[:20])

        added = 0
        for gene, raw_w in extra:
            if gene in own_protected:
                continue
            if gene not in anti:
                anti.append(gene)
                added += 1
            cur = weights.get(gene, 0.0)
            weights[gene] = max(cur, float(raw_w))

        weights = _renormalize(weights, TOP_TARGET_WEIGHT)
        ax["anti_genes"] = anti
        ax["anti_marker_weights"] = weights
        ax.setdefault("evidence", {})
        deriv = list(ax["evidence"].get("derivation", []))
        deriv.append(
            f"P-D triangulation ({date.today().isoformat()}): added {added} "
            f"cross-axis anti markers; total anti list = {len(anti)}; weights "
            "renormalized so max=1.5 (axis-internal cross-axis suppression)."
        )
        ax["evidence"]["derivation"] = deriv
        summary[axis_id] = {"new_anti_added": added, "total_anti": len(anti)}

    d.setdefault("changelog", []).append({
        "version": "v1.8.6",
        "date": date.today().isoformat(),
        "summary": (
            "P-D triangulation: cross-injected anti markers among 10 cs axes "
            "to suppress mutual leakage between Effector_Metabolic / "
            "Homeostatic / Inflammatory / Reparative_Stress / Hypoxic / "
            "Metabolic_Stress / Prehypertrophic / Superficial / Progenitor / "
            "Fibrocartilage."
        ),
    })
    d["version"] = "v1.8.6"
    DICT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dictionary updated:", DICT)

    kb_bak = KB.with_name(KB.name + ".pre_v1_8_6.bak")
    if not kb_bak.exists():
        shutil.copy2(KB, kb_bak)
    kb = json.loads(KB.read_text(encoding="utf-8"))
    for entry in kb.get("dictionary_knowledge", []):
        aid = entry.get("axis_id")
        if aid in summary:
            ax = axes_by_id[aid]
            entry["anti_genes"] = ax["anti_genes"]
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("== triangulation summary ==")
    for aid, info in summary.items():
        print(f"  {aid:<55s} +{info['new_anti_added']:>3d} anti  total={info['total_anti']}")


if __name__ == "__main__":
    main()
