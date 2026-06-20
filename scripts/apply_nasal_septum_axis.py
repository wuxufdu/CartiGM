"""Apply the candidate nasal septum axis into cartilage_dictionary_v1.json,
bump package version 0.4.0 -> 0.5.0, and reinstall the editable package.

Usage:
  python scripts/apply_nasal_septum_axis.py

Reads:
  outputs/nasal_septum_axis/nasal_septum_axis_candidate.json
  cartigsfm/resources/dictionary_v1/cartilage_dictionary_v1.json

Writes:
  cartigsfm/resources/dictionary_v1/cartilage_dictionary_v1.json  (axis added)
  cartigsfm/__init__.py                                          (version bumped)
  pyproject.toml                                                 (version bumped)
  setup.py                                                       (version bumped)
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(r"F:\cartifm\CartiGM")
DICT_PATH = REPO / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
CANDIDATE_PATH = Path(r"F:\cartifm\outputs\nasal_septum_axis\nasal_septum_axis_candidate.json")
NEW_VERSION = "0.5.0"


def main():
    if not CANDIDATE_PATH.exists():
        print(f"missing candidate axis: {CANDIDATE_PATH}", file=sys.stderr)
        sys.exit(1)

    candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    print(f"[apply] candidate axis_id = {candidate.get('axis_id')}")
    print(f"[apply] n_core = {len(candidate.get('core_genes', []))}, n_panel = {len(candidate.get('panel_genes', []))}")

    backup = DICT_PATH.with_suffix(".json.pre_nasal_septum.bak")
    if not backup.exists():
        shutil.copy2(DICT_PATH, backup)
        print(f"[apply] backed up dictionary -> {backup}")

    dictionary = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    layers = dictionary.setdefault("layers", {})

    tds = layers.setdefault("tissue_developmental_state", {"count": 0, "axes": []})
    if tds.get("count") is None:
        tds["count"] = len(tds.get("axes", []))
    if not isinstance(tds.get("axes"), list):
        tds["axes"] = []

    new_aid = candidate["axis_id"]
    tds["axes"] = [a for a in tds["axes"] if a.get("axis_id") != new_aid]
    tds["axes"].append(candidate)
    tds["count"] = len(tds["axes"])

    if "candidate_axes" in dictionary:
        for k in ("tissue_developmental_state", "functional_axis", "cell_subtype"):
            dictionary["candidate_axes"][k] = []
        if "notes" in dictionary["candidate_axes"]:
            dictionary["candidate_axes"]["notes"] = [
                "All candidate axes from v0.4.0 promoted to production or removed; no new candidates as of v0.5.0."
            ]

    cl = dictionary.setdefault("changelog", [])
    cl_entry = {
        "version": NEW_VERSION,
        "generated_at": time.strftime("%Y-%m-%d"),
        "changes": [
            f"Added {new_aid} to layers.tissue_developmental_state (was empty in v0.4.0).",
            "Derivation: per-sample pseudobulk DE on EBR.h5ad log1p_norm layer (n_nose=16706 cells, n_ear+rib=16179 cells, 10 samples).",
            "Marker panel: 20 core_genes (8 anchors + 12 data-driven), 60 panel_genes.",
            "Status: production; auto-classified as PENDING_INDEPENDENT_VALIDATION by axis_safety_class() per the evidence policy.",
            "Limitations explicitly flag the dual-use of EBR as both axis source and planned independent validation source for other v1 axes."
        ],
        "scripts": [
            "scripts/add_nasal_septum_axis.py",
            "scripts/apply_nasal_septum_axis.py"
        ],
    }
    cl.append(cl_entry)
    dictionary["changelog"] = cl

    dictionary["version"] = NEW_VERSION
    dictionary["generated_at"] = time.strftime("%Y-%m-%d")

    axis_total = 0
    for k, v in layers.items():
        if isinstance(v, dict) and "axes" in v:
            v["count"] = len(v["axes"])
            axis_total += v["count"]
    dictionary["axis_count_total"] = axis_total
    print(f"[apply] new total axis count = {axis_total}")

    DICT_PATH.write_text(json.dumps(dictionary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[apply] wrote updated dictionary -> {DICT_PATH} ({DICT_PATH.stat().st_size} bytes)")

    targets = [
        REPO / "cartigsfm" / "__init__.py",
        REPO / "pyproject.toml",
        REPO / "setup.py",
    ]
    for t in targets:
        if not t.exists():
            print(f"[apply] skipping missing: {t}")
            continue
        text = t.read_text(encoding="utf-8")
        new_text = re.sub(r'(__version__\s*=\s*")0\.4\.0(")',
                           lambda m: f'{m.group(1)}{NEW_VERSION}{m.group(2)}',
                           text)
        new_text = re.sub(r'(version\s*=\s*")0\.4\.0(")',
                           lambda m: f'{m.group(1)}{NEW_VERSION}{m.group(2)}',
                           new_text)
        if new_text != text:
            t.write_text(new_text, encoding="utf-8")
            print(f"[apply] bumped version in {t.relative_to(REPO)}")
        else:
            print(f"[apply] no change in {t.relative_to(REPO)}")

    import subprocess
    print("[apply] pip install -e . (editable)", flush=True)
    cp = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--quiet"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    print(cp.stdout[-2000:])
    if cp.returncode != 0:
        print(cp.stderr[-2000:], file=sys.stderr)
        sys.exit(cp.returncode)

    import importlib
    import cartigsfm
    importlib.reload(cartigsfm)
    print(f"[apply] cartigsfm.__version__ = {cartigsfm.__version__}")
    d2 = cartigsfm.load_cartilage_dictionary_v1()
    n_axes = sum(len(d2["layers"][l]["axes"]) for l in d2["layers"])
    print(f"[apply] loaded dict: layers={list(d2['layers'].keys())}, total axes={n_axes}")
    for l in d2["layers"]:
        print(f"   {l}: {len(d2['layers'][l]['axes'])} axes -> {[a.get('axis_id') for a in d2['layers'][l]['axes']]}")
    found = [a for a in d2["layers"]["tissue_developmental_state"]["axes"] if a.get("axis_id") == new_aid]
    print(f"[apply] {new_aid} present in dictionary: {bool(found)}")
    if found:
        print(f"[apply] core_genes[0..5] = {found[0].get('core_genes', [])[:5]}")
        print(f"[apply] status = {found[0].get('status')}")
        from cartigsfm.interpret import axis_safety_class
        sc = axis_safety_class(found[0])
        print(f"[apply] safety_classification = {sc}")


if __name__ == "__main__":
    main()
