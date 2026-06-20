"""Rename cell_subtype::Matrix_Maintenance_Chondrocytes -> Homeostatic_Chondrocytes
across the cartilage dictionary, the RAG knowledge base, and downstream code.

Pure rename: the FMOD / COMP / OGN / CILP2 / SMOC2 panel is kept as-is. The
old axis_id is registered as alias for backwards compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
KB = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_cartigsfm_knowledge_base.json"
EVID = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_axis_evidence_cards.json"
ABL = ROOT / "cartigsfm" / "ablation.py"
SMOKE = ROOT / "smoke_interpret.py"
TEST = ROOT / "tests" / "test_cartigsfm_interpret.py"

OLD_AXIS = "cell_subtype::Matrix_Maintenance_Chondrocytes"
NEW_AXIS = "cell_subtype::Homeostatic_Chondrocytes"
OLD_NAME_EN = "Matrix_Maintenance_Chondrocytes"
NEW_NAME_EN = "Homeostatic_Chondrocytes"
NEW_NAME_CN = "稳态软骨细胞"
NEW_INTERPRETATION = (
    "Homeostatic chondrocyte state. Renamed from atlas axis "
    "'Matrix_Maintenance_Chondrocytes' on 2026-06-20 to align the dictionary "
    "with the EBR celltype label 'Homeostatic_Chondrocytes'. Marker panel is "
    "unchanged: FMOD, COMP, OGN, CILP2, SMOC2, CILP, PRELP, FIBIN, COL11A1, "
    "PLA2G2A. The program reflects matrix-maintenance / steady-state cartilage "
    "biology rather than disease-specific or progenitor states."
)
EXTRA_ALIAS = [
    OLD_AXIS,
    OLD_NAME_EN,
    "Homeostatic_Matrix",
    "Homeostatic chondrocytes",
    "HomC",
    "HomCs",
    "matrix maintenance chondrocytes",
]


def patch_dictionary() -> None:
    d = json.loads(DICT.read_text(encoding="utf-8"))
    layer = d["layers"]["cell_subtype"]
    matched = 0
    for ax in layer["axes"]:
        if ax.get("axis_id") == OLD_AXIS:
            ax["axis_id"] = NEW_AXIS
            ax["name_en"] = NEW_NAME_EN
            ax["name_cn"] = NEW_NAME_CN
            ax["interpretation"] = NEW_INTERPRETATION
            aliases = list(ax.get("aliases", []) or [])
            for a in EXTRA_ALIAS:
                if a not in aliases:
                    aliases.append(a)
            ax["aliases"] = aliases
            matched += 1
    assert matched == 1, f"expected 1 match, got {matched}"

    aliases_block = d.setdefault("axis_id_aliases", {})
    aliases_block[OLD_AXIS] = NEW_AXIS

    changelog = d.setdefault("changelog", [])
    changelog.append({
        "version": "v1.8.2",
        "date": "2026-06-20",
        "summary": (
            "Renamed cell_subtype::Matrix_Maintenance_Chondrocytes -> "
            "cell_subtype::Homeostatic_Chondrocytes to match EBR celltype "
            "labels. Marker / anti-marker panels unchanged; alias kept for "
            "backwards compatibility."
        ),
    })
    d["version"] = "v1.8.2"
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
            entry["interpretation"] = NEW_INTERPRETATION
            aliases = entry.setdefault("aliases", [])
            for a in EXTRA_ALIAS:
                if a not in aliases:
                    aliases.append(a)
            matched += 1
    KB.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"KB axis entries updated: {matched}")


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
    EVID.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"evidence-card substitutions: {matched}")


def patch_text(path: Path) -> None:
    if not path.exists():
        return
    txt = path.read_text(encoding="utf-8")
    new = txt.replace(OLD_AXIS, NEW_AXIS)
    if new != txt:
        path.write_text(new, encoding="utf-8")
        print(f"updated: {path}")


def main() -> None:
    patch_dictionary()
    patch_kb()
    patch_evidence_cards()
    for p in (ABL, SMOKE, TEST):
        patch_text(p)


if __name__ == "__main__":
    main()
