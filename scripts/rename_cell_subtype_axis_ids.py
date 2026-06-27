"""Rename cell_subtype axis_id values to align with the literature-aligned
name_en already shipped in v1.5.

- Bundled cartilage_dictionary_v1.json:
    * axis_id rewritten on each cell_subtype axis.
    * Old short id added to that axis's aliases list (front).
    * Top-level axis_id_aliases map upserted (old fq -> new fq).
    * version 1.5 -> 1.6, generated_at = today, changelog appended.
- cartigsfm/resources/rag_v1/p6_cartigsfm_knowledge_base.json: axis_id strings rewritten.
- cartigsfm/ablation.py: hard-coded axis_id strings rewritten.
- tests/test_cartigsfm_interpret.py + smoke_interpret.py: same.

Backups:
- bundled dictionary -> .pre_v1_6.bak (only if not already present).

The script is rerun-safe.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

REPO = Path(r"F:\cartifm\CartiGM")
DICT_PATH = REPO / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
RAG_KB_PATH = REPO / "cartigsfm" / "resources" / "rag_v1" / "p6_cartigsfm_knowledge_base.json"
ABLATION_PATH = REPO / "cartigsfm" / "ablation.py"
TEST_INTERPRET_PATH = REPO / "tests" / "test_cartigsfm_interpret.py"
SMOKE_PATH = REPO / "smoke_interpret.py"

RENAME = {
    "EC_Lipo_Plasticity": "Effector_Metabolic_Chondrocytes",
    "Fibro_Matrix": "Stromal_Matrix_Chondrocytes",
    "Homeostatic_Matrix": "Matrix_Maintenance_Chondrocytes",
    "Hypoxia_Adaptive": "Hypoxic_Chondrocytes",
    "Hypoxia_Metabolic_Stress": "Metabolic_Stress_Chondrocytes",
    "Inflammatory_Remodeling": "Inflammatory_Response_Chondrocytes",
    "Maturation_Matrix": "Prehypertrophic_Matrix_Chondrocytes",
    "Mesenchymal_Remodeling": "Fibrocartilage_Chondrocytes",
    "PRG4_Interface": "Superficial_Zone_Chondrocytes",
    "Stress_IEG": "Reparative_Stress_Chondrocytes",
}
OLD_TO_NEW_FQ = {f"cell_subtype::{o}": f"cell_subtype::{n}" for o, n in RENAME.items()}
NEW_VERSION = "1.6"


def patch_dictionary() -> tuple[int, int]:
    backup = DICT_PATH.with_suffix(".json.pre_v1_6.bak")
    if not backup.exists():
        shutil.copy2(DICT_PATH, backup)
        print(f"  backed up dictionary -> {backup.name}")
    d = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    cs_axes = d["layers"]["cell_subtype"]["axes"]
    renamed = 0
    aliases_added = 0
    for a in cs_axes:
        old = a.get("axis_id", "")
        new = OLD_TO_NEW_FQ.get(old)
        if not new:
            continue
        old_short = old.split("::", 1)[1]
        new_short = new.split("::", 1)[1]
        aliases = a.get("aliases") or []
        if old_short not in aliases:
            a["aliases"] = [old_short] + [x for x in aliases if x != old_short]
            aliases_added += 1
        if a.get("name_en") != new_short:
            print(f"  WARN: name_en={a.get('name_en')!r} != {new_short!r}; setting name_en")
            a["name_en"] = new_short
        a["axis_id"] = new
        renamed += 1

    aid_aliases = d.get("axis_id_aliases") or {}
    for o, n in OLD_TO_NEW_FQ.items():
        aid_aliases[o] = n
    d["axis_id_aliases"] = aid_aliases

    if renamed > 0 or aliases_added > 0:
        d["version"] = NEW_VERSION
        d["generated_at"] = time.strftime("%Y-%m-%d")
        cl = d.get("changelog") or []
        cl.append({
            "version": NEW_VERSION,
            "generated_at": d["generated_at"],
            "changes": [
                "Renamed cell_subtype axis_id values to match the literature-aligned name_en introduced in v1.5.",
                "Old short axis_ids are preserved in each axis's aliases list and a top-level axis_id_aliases map (old fq -> new fq) was added so legacy callers keep resolving.",
                "Mapping: " + ", ".join(f"{o}->{n}" for o, n in RENAME.items()),
            ],
            "scripts": ["scripts/rename_cell_subtype_axis_ids.py"],
        })
        d["changelog"] = cl
        DICT_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote dictionary v{NEW_VERSION} ({DICT_PATH.stat().st_size} bytes)")
    else:
        print("  dictionary already at new ids; no write")
    return renamed, aliases_added


def patch_rag_kb() -> int:
    if not RAG_KB_PATH.exists():
        return 0
    text = RAG_KB_PATH.read_text(encoding="utf-8")
    n = 0
    for old, new in OLD_TO_NEW_FQ.items():
        pattern = re.escape(f'"{old}"')
        new_q = f'"{new}"'
        new_text, k = re.subn(pattern, new_q, text)
        if k:
            text = new_text
            n += k
    if n:
        RAG_KB_PATH.write_text(text, encoding="utf-8")
        print(f"  rewrote {n} axis_id occurrences in {RAG_KB_PATH.name}")
    else:
        print(f"  no rewrites needed in {RAG_KB_PATH.name}")
    return n


def patch_text_file(path: Path) -> int:
    if not path.exists():
        print(f"  skip (not present): {path.name}")
        return 0
    text = path.read_text(encoding="utf-8")
    n = 0
    for old, new in OLD_TO_NEW_FQ.items():
        for q in ('"', "'"):
            old_lit = f"{q}{old}{q}"
            new_lit = f"{q}{new}{q}"
            if old_lit in text:
                text = text.replace(old_lit, new_lit)
                n += 1
    if n:
        path.write_text(text, encoding="utf-8")
        print(f"  rewrote {n} quoted axis_id occurrences in {path.name}")
    else:
        print(f"  no quoted axis_id rewrites in {path.name}")
    return n


def main() -> None:
    print("== dictionary ==")
    renamed, aliased = patch_dictionary()
    print(f"  renamed axes: {renamed}, aliases added: {aliased}")

    print("== RAG knowledge base ==")
    patch_rag_kb()

    print("== ablation.py ==")
    patch_text_file(ABLATION_PATH)
    print("== tests/test_cartigsfm_interpret.py ==")
    patch_text_file(TEST_INTERPRET_PATH)
    print("== smoke_interpret.py ==")
    patch_text_file(SMOKE_PATH)

    # Final verification
    print("== verify ==")
    import importlib, sys
    mods = [m for m in list(sys.modules) if m == "cartigsfm" or m.startswith("cartigsfm.")]
    for m in mods:
        del sys.modules[m]
    import cartigsfm
    d = cartigsfm.load_cartilage_dictionary_v1()
    cs_ids = [a["axis_id"] for a in d["layers"]["cell_subtype"]["axes"]]
    print(f"  dict version={d.get('version')} cell_subtype axis_ids:")
    for x in cs_ids:
        print(f"    - {x}")
    leftover_old = [a for a in cs_ids if any(a.endswith("::" + o) for o in RENAME.keys())]
    assert not leftover_old, f"old axis_ids leaked through: {leftover_old}"
    print(f"  axis_id_aliases keys ({len(d.get('axis_id_aliases') or {})}): "
          f"{list((d.get('axis_id_aliases') or {}).keys())}")
    print("  OK")


if __name__ == "__main__":
    main()
