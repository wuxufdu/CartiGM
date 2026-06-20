"""Rename cell_subtype::Stromal_Matrix_Chondrocytes -> Progenitor_Chondrocytes
across the cartilage dictionary, the RAG knowledge base and ablation.py.

The rename is name-only: the SCARA3/COLEC12/VCAM1/LTBP2/CDH13 panel is kept as
placeholder until a Progenitor-specific panel (e.g. PRG4/CD146/PDGFRA/LEPR
plus atlas-derived progenitor markers) is rebuilt and re-calibrated. A
limitation entry is added to make this explicit, and the old axis_id is
registered as alias for backwards compatibility.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
KB = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_cartigsfm_knowledge_base.json"
EVID = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_axis_evidence_cards.json"
ABL = ROOT / "cartigsfm" / "ablation.py"

OLD_AXIS = "cell_subtype::Stromal_Matrix_Chondrocytes"
NEW_AXIS = "cell_subtype::Progenitor_Chondrocytes"
OLD_NAME_EN = "Stromal_Matrix_Chondrocytes"
NEW_NAME_EN = "Progenitor_Chondrocytes"
NEW_NAME_CN = "祖细胞样软骨细胞"
NEW_INTERPRETATION = (
    "Progenitor-like chondrocyte state. Renamed from atlas axis "
    "'Stromal_Matrix_Chondrocytes' on 2026-06-20 to better reflect biology of "
    "chondrogenic progenitor populations observed in EBR cartilage. The current "
    "marker panel still reflects the original SCARA3 / COLEC12 / VCAM1 / LTBP2 / "
    "CDH13 stromal program; an independent progenitor marker panel "
    "(e.g. PRG4, PDGFRA, LEPR, MCAM/CD146) will be rebuilt and re-calibrated "
    "before downstream training."
)
EXTRA_LIMITATION = (
    "Panel pending rebuild: marker / anti-marker panels still reflect the "
    "previous Stromal_Matrix axis and have not been re-derived against a "
    "progenitor-specific gene prior."
)
EXTRA_ALIAS = [
    OLD_AXIS,
    OLD_NAME_EN,
    "Fibro_Matrix",
    "Stromal matrix chondrocytes",
]


def _patch_axis(ax: dict) -> None:
    ax["axis_id"] = NEW_AXIS
    ax["name_en"] = NEW_NAME_EN
    ax["name_cn"] = NEW_NAME_CN
    ax["interpretation"] = NEW_INTERPRETATION
    aliases = list(ax.get("aliases", []) or [])
    for a in EXTRA_ALIAS:
        if a not in aliases:
            aliases.append(a)
    ax["aliases"] = aliases
    lims = list(ax.get("limitations", []) or [])
    if EXTRA_LIMITATION not in lims:
        lims.append(EXTRA_LIMITATION)
    ax["limitations"] = lims


def patch_dictionary() -> None:
    d = json.loads(DICT.read_text(encoding="utf-8"))
    layer = d["layers"]["cell_subtype"]
    matched = 0
    for ax in layer["axes"]:
        if ax.get("axis_id") == OLD_AXIS:
            _patch_axis(ax)
            matched += 1
    assert matched == 1, f"expected 1 cell_subtype axis match, got {matched}"

    aliases_block = d.setdefault("axis_id_aliases", {})
    aliases_block[OLD_AXIS] = NEW_AXIS

    changelog = d.setdefault("changelog", [])
    changelog.append({
        "version": "v1.8.1",
        "date": "2026-06-20",
        "summary": (
            "Renamed cell_subtype::Stromal_Matrix_Chondrocytes -> "
            "cell_subtype::Progenitor_Chondrocytes. axis_id alias kept for "
            "backwards compatibility; marker / anti-marker panels unchanged "
            "pending progenitor-specific recalibration."
        ),
    })
    d["version"] = "v1.8.1"

    DICT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dictionary updated: {DICT}")


def patch_kb() -> None:
    d = json.loads(KB.read_text(encoding="utf-8"))
    matched = 0
    for entry in d.get("dictionary_knowledge", []):
        if entry.get("axis_id") == OLD_AXIS:
            entry["axis_id"] = NEW_AXIS
            entry["name_en"] = NEW_NAME_EN
            entry["name_cn"] = NEW_NAME_CN
            interp = entry.get("interpretation") or ""
            entry["interpretation"] = NEW_INTERPRETATION
            entry.setdefault("aliases", [])
            for a in EXTRA_ALIAS:
                if a not in entry["aliases"]:
                    entry["aliases"].append(a)
            lims = entry.setdefault("limitations", [])
            if EXTRA_LIMITATION not in lims:
                lims.append(EXTRA_LIMITATION)
            matched += 1
    print(f"KB axis entries updated: {matched}")
    KB.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def patch_evidence_cards() -> None:
    if not EVID.exists():
        print("no evidence cards file; skip")
        return
    txt = EVID.read_text(encoding="utf-8")
    if OLD_AXIS not in txt and OLD_NAME_EN not in txt:
        print("no evidence-card hits; skip")
        return
    d = json.loads(txt)
    matched = 0
    def walk(o):
        nonlocal matched
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if isinstance(v, str):
                    if v == OLD_AXIS:
                        o[k] = NEW_AXIS; matched += 1
                    elif v == OLD_NAME_EN:
                        o[k] = NEW_NAME_EN; matched += 1
                else:
                    walk(v)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                if isinstance(v, str):
                    if v == OLD_AXIS:
                        o[i] = NEW_AXIS; matched += 1
                    elif v == OLD_NAME_EN:
                        o[i] = NEW_NAME_EN; matched += 1
                else:
                    walk(v)
    walk(d)
    print(f"evidence-card string substitutions: {matched}")
    EVID.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def patch_ablation() -> None:
    txt = ABL.read_text(encoding="utf-8")
    new = txt.replace(OLD_AXIS, NEW_AXIS)
    if new != txt:
        ABL.write_text(new, encoding="utf-8")
        print(f"ablation.py updated")
    else:
        print("ablation.py: no occurrences")


def main() -> None:
    patch_dictionary()
    patch_kb()
    patch_evidence_cards()
    patch_ablation()


if __name__ == "__main__":
    main()
