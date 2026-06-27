"""P12: GSFM-style frozen gene-set / axis embedding branch.

The full GSFM (Gene-Set Foundation Model) PPMI + SVD weights (~100 MB) are
not bundled in this sandbox. This module implements the same surface API in
a deterministic proxy that:

  * Builds a 42-axis feature vector from the bundled ``cartilage_dictionary_v1``
    and P6 ``p6_axis_evidence_cards`` metadata. **No bulk expression is used**
    here; GSFM is the "gene-set / axis" branch, distinct from the scGPT branch.
  * Computes gene-set -> axis similarity scores using a weighted Jaccard
    coefficient over each axis's ``marker_weights`` (or ``panel_genes``).
  * Returns a structured output that the CartiAgent LLM agent can consume as a
    tool call.

The module is **frozen** (no training, no fine-tuning). The "embedding" is
explicitly a dict of named features so downstream tools can also serialize it
to JSON / inspect it.

Public API
----------
gsfm_axis_table()                                  -> pandas.DataFrame
gsfm_axis_embedding(axis_id)                       -> dict
gsfm_axis_similarity(marker_list, axis_id)         -> float
gsfm_marker_axes(marker_list, top_n=5)             -> list[dict]
tool_gsfm_score(marker_list, axis_id=None, top_n=5) -> dict
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .assets import load_axis_evidence_cards, load_cartilage_dictionary_v1
from .dictionary import load_alias_map
from .interpret import axis_safety_class
from .projection import _axis_anti_weights, _axis_gene_weights
from .scoring import resolve_aliases


# ---------------------------------------------------------------------------
# Internal axis index
# ---------------------------------------------------------------------------

def _v1_axes() -> List[Dict[str, Any]]:
    dictionary = load_cartilage_dictionary_v1()
    out: List[Dict[str, Any]] = []
    for layer, layer_obj in (dictionary.get("layers") or {}).items():
        for axis in layer_obj.get("axes", []):
            entry = dict(axis)
            entry["layer"] = layer
            out.append(entry)
    return out


def _axis_index() -> Dict[str, Dict[str, Any]]:
    return {a.get("axis_id"): a for a in _v1_axes() if a.get("axis_id")}


def _evidence_index() -> Dict[str, Dict[str, Any]]:
    cards = load_axis_evidence_cards() or []
    out: Dict[str, Dict[str, Any]] = {}
    for c in cards:
        if isinstance(c, dict) and c.get("axis_id"):
            out[str(c["axis_id"])] = c
    return out


def _normalize_markers(markers) -> List[str]:
    if markers is None:
        return []
    if isinstance(markers, str):
        markers = [t.strip() for t in re.split(r"[\s,;]+", markers) if t.strip()]
    resolved = resolve_aliases(
        [str(m).strip() for m in markers if str(m).strip()],
        load_alias_map(),
    )
    return sorted({g.upper() for g in resolved})


# ---------------------------------------------------------------------------
# Axis feature table and per-axis embedding
# ---------------------------------------------------------------------------

def gsfm_axis_table() -> pd.DataFrame:
    cards = _evidence_index()
    rows: List[Dict[str, Any]] = []
    for axis in _v1_axes():
        ax_id = axis.get("axis_id")
        card = cards.get(ax_id, {})
        marker_weights = _axis_gene_weights(axis)
        anti_weights = _axis_anti_weights(axis)
        atlas_obs = card.get("atlas_observations") or []
        bio_ctx = card.get("expected_biological_contexts") or []
        key_gene_ev = card.get("key_gene_evidence") or []
        robustness = card.get("p5_robustness_status") or {}
        rows.append({
            "axis_id": ax_id,
            "layer": axis.get("layer"),
            "name_en": axis.get("name_en"),
            "name_cn": axis.get("name_cn"),
            "n_core": len(marker_weights),
            "n_anti": len(anti_weights),
            "n_evidence_internal": len(atlas_obs),
            "n_evidence_literature": len(bio_ctx) + len(key_gene_ev),
            "n_evidence_independent": 0,
            "has_literature_support": bool(bio_ctx or key_gene_ev),
            "has_independent_validation": False,
            "confidence_level": card.get("confidence_level") or "",
            "robustness_label_agreement": robustness.get("mean_top_label_agreement"),
            "status": axis.get("status"),
            "safety_classification": axis_safety_class(axis),
        })
    return pd.DataFrame(rows)


def gsfm_axis_embedding(axis_id: str) -> Dict[str, Any]:
    axes = _axis_index()
    card = _evidence_index().get(axis_id, {})
    if axis_id not in axes:
        return {
            "axis_id": axis_id,
            "found": False,
            "safety_classification": "UNREVIEWED",
            "core_genes": [],
            "anti_genes": [],
            "core_weight_sum": 0.0,
        }
    axis = axes[axis_id]
    marker_weights = _axis_gene_weights(axis)
    anti_weights = _axis_anti_weights(axis)
    atlas_obs = card.get("atlas_observations") or []
    bio_ctx = card.get("expected_biological_contexts") or []
    key_gene_ev = card.get("key_gene_evidence") or []
    confidence_level = card.get("confidence_level") or ""
    robustness = card.get("p5_robustness_status") or {}
    return {
        "axis_id": axis_id,
        "found": True,
        "layer": axis.get("layer"),
        "name_en": axis.get("name_en"),
        "name_cn": axis.get("name_cn"),
        "status": axis.get("status"),
        "safety_classification": axis_safety_class(axis),
        "core_genes": sorted(marker_weights.keys()),
        "anti_genes": sorted(anti_weights.keys()),
        "core_weight_sum": round(sum(marker_weights.values()), 3),
        "n_evidence_internal": len(atlas_obs),
        "n_evidence_literature": len(bio_ctx) + len(key_gene_ev),
        "n_evidence_independent": 0,
        "n_atlas_observations": len(atlas_obs),
        "n_biological_contexts": len(bio_ctx),
        "n_key_gene_evidence": len(key_gene_ev),
        "confidence_level": confidence_level,
        "robustness_label_agreement": robustness.get("mean_top_label_agreement"),
        "has_literature_support": bool(bio_ctx or key_gene_ev),
        "has_independent_validation": False,
        "atlas_observations_preview": [str(x)[:120] for x in atlas_obs[:3]],
        "biological_contexts_preview": [str(x)[:120] for x in bio_ctx[:3]],
    }
    return {
        "axis_id": axis_id,
        "found": True,
        "layer": axis.get("layer"),
        "name_en": axis.get("name_en"),
        "name_cn": axis.get("name_cn"),
        "status": axis.get("status"),
        "safety_classification": axis_safety_class(axis),
        "core_genes": sorted(marker_weights.keys()),
        "anti_genes": sorted(anti_weights.keys()),
        "core_weight_sum": round(sum(marker_weights.values()), 3),
        "n_evidence_internal": len(ev.get("internal_support") or []),
        "n_evidence_independent": len(ev.get("independent_validation") or []),
        "n_evidence_literature": len(ev.get("literature_support") or []),
        "has_literature_support": bool(ev.get("literature_support")),
        "has_independent_validation": bool(ev.get("independent_validation")),
    }


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------

def gsfm_axis_similarity(marker_list, axis_id: str) -> float:
    query = set(_normalize_markers(marker_list))
    if not query:
        return 0.0
    axes = _axis_index()
    if axis_id not in axes:
        return 0.0
    marker_weights = _axis_gene_weights(axes[axis_id])
    if not marker_weights:
        return 0.0
    shared = query & set(marker_weights.keys())
    if not shared:
        return 0.0
    sum_shared = sum(marker_weights[g] for g in shared)
    sum_axis = sum(marker_weights.values())
    sum_query = float(len(query))
    denom = sum_query + sum_axis - sum_shared
    if denom <= 0:
        return 0.0
    return round(sum_shared / denom, 6)


def gsfm_marker_axes(marker_list, *, top_n: int = 5,
                     min_overlap: int = 1) -> List[Dict[str, Any]]:
    query = set(_normalize_markers(marker_list))
    axes = _axis_index()
    rows: List[Dict[str, Any]] = []
    for ax_id, axis in axes.items():
        marker_weights = _axis_gene_weights(axis)
        if not marker_weights:
            continue
        shared = sorted(query & set(marker_weights.keys()))
        if len(shared) < min_overlap:
            continue
        score = gsfm_axis_similarity(query, ax_id)
        rows.append({
            "axis_id": ax_id,
            "layer": axis.get("layer"),
            "name_en": axis.get("name_en"),
            "name_cn": axis.get("name_cn"),
            "score": score,
            "shared_genes": shared,
            "shared_n": len(shared),
            "core_n": len(marker_weights),
            "safety_classification": axis_safety_class(axis),
        })
    rows.sort(key=lambda r: (r["score"], r["shared_n"]), reverse=True)
    return rows[: max(int(top_n or 0), 0)]


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------

def tool_gsfm_score(marker_list, axis_id: Optional[str] = None,
                    *, top_n: int = 5) -> Dict[str, Any]:
    if isinstance(marker_list, str):
        marker_list = [t.strip() for t in re.split(r"[\s,;]+", marker_list) if t.strip()]
    query = sorted({g.upper() for g in (marker_list or [])})
    if not query:
        return {
            "branch": "gsfm",
            "input_genes": [],
            "result": None,
            "note": "empty marker list",
        }
    if axis_id:
        score = gsfm_axis_similarity(query, axis_id)
        return {
            "branch": "gsfm",
            "input_genes": query,
            "axis_id": axis_id,
            "result": {"similarity": score, "embedding": gsfm_axis_embedding(axis_id)},
        }
    top = gsfm_marker_axes(query, top_n=top_n)
    return {
        "branch": "gsfm",
        "input_genes": query,
        "result": {
            "top_axes": top,
            "n_axes_scored": len(_axis_index()),
            "note": (
                "GSFM branch: gene-set / axis similarity view. "
                "Not a substitute for expression-based (scGPT) projection."
            ),
        },
    }
