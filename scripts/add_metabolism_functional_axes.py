"""Bump the cartilage dictionary from v1.6 to v1.7 by appending 10 metabolism
functional axes (Glycolysis, OXPHOS, TCA, PPP, FAO, Lipogenesis, Cholesterol,
LipidDroplet, Glutaminolysis, MitochondrialBiogenesis) plus matching entries in
the bundled P6 RAG knowledge base and axis evidence cards.

Status of every new axis is ``literature_prior`` so that
:func:`cartigsfm.interpret.axis_safety_class` returns ``EXPLORATORY`` until a
future atlas-calibration pass populates ``freq_in`` / ``freq_bg`` /
``log2_spec`` from real data. ``core_genes`` follow the
``{gene, freq_in, freq_bg, log2_spec, weight}`` schema with the three
frequency fields set to ``None`` so the slots are explicit and downstream
tools can detect the calibration gap.

Run with::

    cd F:\\cartifm\\CartiGM
    & .venv\\Scripts\\python.exe scripts\\add_metabolism_functional_axes.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _metabolism_axis_panels as MP  # noqa: E402


DICT_PATH = ROOT / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
RAG_PATH = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_cartigsfm_knowledge_base.json"
CARDS_PATH = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_axis_evidence_cards.json"

NEW_VERSION = "1.7"
OLD_VERSION = "1.6"


def _build_core_genes(core_pairs):
    """Convert ``[(gene, weight), ...]`` to the dict-with-frequency layout used
    by every existing literature_prior axis."""
    out = []
    for gene, weight in core_pairs:
        out.append({
            "gene": str(gene),
            "freq_in": None,
            "freq_bg": None,
            "log2_spec": None,
            "weight": float(weight),
        })
    return out


def _build_marker_weights(core_pairs, panel):
    """Match the v1.0 convention: every panel gene gets weight 1.0; core genes
    inherit their explicit weight (>= 1.0)."""
    weights = {g: 1.0 for g in panel}
    for gene, weight in core_pairs:
        weights[str(gene)] = max(float(weight), weights.get(str(gene), 1.0))
    return weights


def _build_axis(spec):
    axis_id_short = spec["axis_id"]
    axis_id = f"functional_axis::{axis_id_short}"
    core = _build_core_genes(spec["core"])
    panel = list(spec["panel"])
    weights = _build_marker_weights(spec["core"], panel)
    return {
        "axis_id": axis_id,
        "layer": "functional_axis",
        "name_en": spec["name_en"],
        "name_cn": spec["name_cn"],
        "biological_scope": spec["biological_scope"],
        "status": "literature_prior",
        "core_genes": core,
        "panel_genes": panel,
        "anti_genes": list(spec.get("anti", [])),
        "marker_weights": weights,
        "evidence": {
            "derivation": [
                "Curated literature panel (v1.7 metabolism extension): KEGG / Reactome / MSigDB Hallmark anchors plus cartilage-specific reviews.",
            ],
            "internal_support": [
                "Awaiting atlas-calibration pass: freq_in / freq_bg / log2_spec are null until P3 frequency table is regenerated for v1.7.",
            ],
            "independent_validation": [],
            "literature_support": list(spec.get("literature", [])),
        },
        "interpretation": spec["interpretation"],
        "limitations": [
            "Literature-prior only; not yet calibrated against the v1 atlas. axis_safety_class() returns EXPLORATORY.",
            "Core gene weights are heuristic priors (1.5 -> 0.3) and will be re-estimated from atlas frequencies in the next P3 pass.",
        ],
        "recommended_use": [
            "Functional enrichment",
            "Module scoring",
            "Pathway projection",
        ],
        "source_files": [
            "scripts/_metabolism_axis_panels.py",
            "scripts/add_metabolism_functional_axes.py",
        ],
    }


def _build_kb_card(axis):
    return {
        "axis_id": axis["axis_id"],
        "layer": axis["layer"],
        "name_en": axis["name_en"],
        "name_cn": axis["name_cn"],
        "biological_scope": axis["biological_scope"],
        "core_genes": axis["core_genes"],
        "panel_genes": axis["panel_genes"],
        "anti_genes": axis["anti_genes"],
        "interpretation": axis["interpretation"],
        "limitations": axis["limitations"],
        "recommended_use": axis["recommended_use"],
        "evidence_level": "LITERATURE_PRIOR",
        "robustness_score_correlation": None,
        "robustness_top_label_agreement": None,
    }


def _build_evidence_card(axis):
    name = axis["name_en"]
    return {
        "axis_id": axis["axis_id"],
        "layer": axis["layer"],
        "name_en": name,
        "name_cn": axis["name_cn"],
        "biological_meaning": axis["biological_scope"],
        "core_supporting_genes": axis["core_genes"],
        "anti_genes": axis["anti_genes"],
        "expected_biological_contexts": axis["recommended_use"],
        "atlas_observations": [],
        "p5_robustness_status": {
            "mean_score_correlation": None,
            "mean_top_label_agreement": None,
            "robustness_note": "Literature-prior axis (v1.7 metabolism extension); no P5 atlas robustness available yet.",
        },
        "confidence_level": "EXPLORATORY",
        "known_limitations": axis["limitations"],
        "recommended_interpretation_wording": [
            f"This sample shows enrichment for {name} gene module",
            f"This cluster exhibits {name} transcriptional signature",
            f"Atlas-internal analysis is pending for the {name} axis (literature prior only).",
        ],
        "forbidden_overclaim_wording": [
            "DO NOT claim: 'proves {axis_name} causes disease X'",
            "DO NOT claim: 'experimentally validated'",
            "DO NOT claim: 'independent validation confirms'",
            "DO NOT claim: 'therapeutic target' without external validation",
            "DO NOT make causal claims from atlas associations alone",
        ],
        "key_gene_evidence": [
            {"gene": g["gene"], "weight": g["weight"], "source": "literature_prior"}
            for g in axis["core_genes"][:8]
        ],
    }


def _backup(path, suffix):
    bak = path.with_name(path.name + suffix)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup -> {bak}")


def main():
    if not DICT_PATH.exists():
        print(f"missing: {DICT_PATH}", file=sys.stderr)
        return 2
    print(f"Reading dictionary {DICT_PATH}")
    d = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    if d.get("version") != OLD_VERSION:
        print(f"WARNING: dictionary version is {d.get('version')!r}, expected {OLD_VERSION!r}; proceeding anyway")

    fa_axes = d["layers"]["functional_axis"]["axes"]
    existing_ids = {a["axis_id"] for a in fa_axes}
    specs = MP.axis_specs()

    new_axes = []
    for spec in specs:
        axis = _build_axis(spec)
        if axis["axis_id"] in existing_ids:
            print(f"  SKIP (already present): {axis['axis_id']}")
            continue
        new_axes.append(axis)

    if not new_axes:
        print("Nothing to add. Exiting.")
        return 0

    _backup(DICT_PATH, ".pre_v1_7.bak")

    fa_axes.extend(new_axes)
    d["layers"]["functional_axis"]["axes"] = fa_axes
    d["layers"]["functional_axis"]["count"] = len(fa_axes)

    total = sum(d["layers"][k]["count"] for k in d["layers"])
    d["axis_count_total"] = total
    d["version"] = NEW_VERSION
    d["generated_at"] = date.today().isoformat()

    d.setdefault("changelog", []).append({
        "version": NEW_VERSION,
        "generated_at": date.today().isoformat(),
        "changes": [
            f"Added {len(new_axes)} metabolism functional axes: "
            + ", ".join(a["name_en"] for a in new_axes) + ".",
            "All new axes use status=literature_prior (axis_safety_class() -> EXPLORATORY).",
            "core_genes use the {gene, freq_in, freq_bg, log2_spec, weight} schema "
            "with freq_in/freq_bg/log2_spec set to null pending the next atlas-calibration pass.",
            f"Layer counts after extension: cell_subtype={d['layers']['cell_subtype']['count']}, "
            f"tissue_developmental_state={d['layers']['tissue_developmental_state']['count']}, "
            f"functional_axis={d['layers']['functional_axis']['count']} ({total} total).",
        ],
        "scripts": [
            "scripts/_metabolism_axis_panels.py",
            "scripts/add_metabolism_functional_axes.py",
        ],
    })

    DICT_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dictionary v{NEW_VERSION} -> {DICT_PATH}")
    print(f"  functional_axis count: {d['layers']['functional_axis']['count']}")
    print(f"  axis_count_total: {total}")

    # ---- RAG knowledge base ---------------------------------------------------
    if RAG_PATH.exists():
        kb = json.loads(RAG_PATH.read_text(encoding="utf-8"))
        dk = kb.get("dictionary_knowledge")
        if isinstance(dk, list):
            existing_kb_ids = {it.get("axis_id") for it in dk if isinstance(it, dict)}
            for axis in new_axes:
                if axis["axis_id"] in existing_kb_ids:
                    continue
                dk.append(_build_kb_card(axis))
            kb["dictionary_knowledge"] = dk
            kb["generated_at"] = date.today().isoformat()
            _backup(RAG_PATH, ".pre_v1_7.bak")
            RAG_PATH.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote RAG KB ({len(dk)} dictionary_knowledge entries) -> {RAG_PATH}")
        else:
            print("RAG KB has no list-shaped 'dictionary_knowledge'; skipping KB update")
    else:
        print(f"RAG KB not found at {RAG_PATH}; skipping KB update")

    # ---- Axis evidence cards --------------------------------------------------
    if CARDS_PATH.exists():
        cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
        if isinstance(cards, list):
            existing_card_ids = {it.get("axis_id") for it in cards if isinstance(it, dict)}
            for axis in new_axes:
                if axis["axis_id"] in existing_card_ids:
                    continue
                cards.append(_build_evidence_card(axis))
            _backup(CARDS_PATH, ".pre_v1_7.bak")
            CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote axis evidence cards ({len(cards)} entries) -> {CARDS_PATH}")
        else:
            print("axis evidence cards file is not a list; skipping cards update")
    else:
        print(f"axis evidence cards not found at {CARDS_PATH}; skipping cards update")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
