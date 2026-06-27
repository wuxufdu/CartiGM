"""Evidence-constrained interpretation for CartiGSFM scores.

This module turns a gene list, a P4 outdir, or a P4-style long-form score
CSV into a structured interpretation that is grounded in:

  - the bundled three-layer ``cartilage_dictionary_v1`` (axis evidence,
    supporting genes, limitations, status),
  - the bundled P6 CartiGSFM-RAG claim safety classifier and prompt
    templates (claim registry, recommended vs forbidden wording),
  - the P9 hard constraints from the LoRA prototype model card.

The module deliberately does *not* call any LLM. It produces a
deterministic, machine-checkable interpretation so that downstream
agents and pipelines can rely on the safety classification.

Public API
----------
interpret_gene_list(genes, ...) -> dict
interpret_p4_dir(outdir, ...) -> dict
interpret_p4_csv(path, ...) -> dict
classify_claim(text) -> dict
apply_safety_filter(interpretation, ...) -> dict
render_markdown(interpretation) -> str
render_json(interpretation) -> str
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .assets import (
    load_cartilage_dictionary_v1,
    load_claim_safety_classifier,
)
from .dictionary import load_alias_map
from .scoring import resolve_aliases
from . import projection
from .projection import project_dictionary_v1_bulk


# ---------------------------------------------------------------------------
# Safety metadata (P9 hard constraints + claim-safety rules)
# ---------------------------------------------------------------------------

SAFETY_LABELS = (
    "MAIN_TEXT_READY",
    "SUPPLEMENTARY_ONLY",
    "EXPLORATORY",
    "PENDING_INDEPENDENT_VALIDATION",
    "NOT_SUPPORTED",
    "UNREVIEWED",
)

# Hard constraints from P9 model card / training_config.hard_constraints.
HARD_CONSTRAINTS = [
    "Do not claim CartiGSFM is a trained large language model.",
    "Do not claim external/independent validation; P4 is pending.",
    "Do not claim Inflammation_NFkB / IL1 / TNF significantly increased in OA (FDR not met).",
    "Do not invent gene names, p-values, sample sizes, or numerical results.",
    "Do not claim therapeutic targets or causal mechanism from CartiGSFM scores.",
]

# Regex-based overclaim guard. Each tuple is (pattern, rationale).
# Matched against the user free-text claim in classify_claim().
FORBIDDEN_PHRASES = [
    (r"\bLLM\b|large language model|trained (?:cartilage )?(?:model|LLM)",
     "Do not call CartiGSFM a trained LLM."),
    (r"externally validated|independent validation confirms|experimentally validated|experimentally confirmed",
     "Do not claim external/experimental validation; P4 is pending."),
    (r"clinically useful|clinical validation|ready for translation|therapeutic target|drug target|causal proof|\bproves?\b|\bdemonstrates?\b",
     "Do not claim therapeutic/clinical/causal conclusions from scores."),
    (r"inflammation.*significantly increased.*OA|OA.*inflammation.*significantly",
     "Inflammation axes are not FDR significant in OA."),
]

# Axis-level status -> safety mapping (covers cartilage_dictionary_v1).
_DEFAULT_STATUS_TO_SAFETY = {
    "production": "PENDING_INDEPENDENT_VALIDATION",
    "reference": "SUPPLEMENTARY_ONLY",
    "literature_prior": "EXPLORATORY",
}


# ---------------------------------------------------------------------------
# Axis-level helpers
# ---------------------------------------------------------------------------

def _iter_v1_axes(dictionary):
    layers = (dictionary or {}).get("layers") or {}
    for layer, layer_obj in layers.items():
        for axis in (layer_obj or {}).get("axes", []):
            yield layer, axis


def _v1_axis_index(dictionary):
    return {axis.get("axis_id"): axis for _, axis in _iter_v1_axes(dictionary)}


def axis_safety_class(axis):
    """Map a v1 axis dict to a safety classification label.

    Priority:
      1. explicit ``safety_classification`` field (e.g. RAG-merged axes);
      2. ``status`` field mapped through ``_DEFAULT_STATUS_TO_SAFETY``;
      3. fallback to PENDING_INDEPENDENT_VALIDATION.
    """
    if axis.get("safety_classification"):
        return str(axis["safety_classification"])
    status = str(axis.get("status") or "").strip().lower()
    return _DEFAULT_STATUS_TO_SAFETY.get(status, "PENDING_INDEPENDENT_VALIDATION")


def _axis_supporting_genes(axis):
    """Return the curated ``core_genes`` (or panel_genes fallback) as a flat list."""
    out = []
    for entry in axis.get("core_genes") or []:
        if isinstance(entry, dict):
            gene = entry.get("gene")
        else:
            gene = entry
        if gene:
            out.append(str(gene))
    if not out:
        for entry in axis.get("panel_genes") or []:
            if isinstance(entry, dict):
                gene = entry.get("gene")
            else:
                gene = entry
            if gene:
                out.append(str(gene))
    return out


def _axis_evidence_fields(axis):
    ev = axis.get("evidence") or {}
    return {
        "derivation": list(ev.get("derivation") or []),
        "internal_support": list(ev.get("internal_support") or []),
        "independent_validation": list(ev.get("independent_validation") or []),
        "literature_support": list(ev.get("literature_support") or []),
    }


_EXPERIMENT_TEMPLATE = {
    "cell_subtype": (
        "Flow cytometry or single-cell RNA-seq cluster re-annotation with a "
        "marker panel of: {genes}."
    ),
    "tissue_developmental_state": (
        "Histology + IHC / RNAscope for tissue-defining markers: {genes}."
    ),
    "functional_axis": (
        "Pathway-level qPCR or bulk RNA-seq module scoring with axis genes: "
        "{genes}."
    ),
}


def _confidence_from_overlap(marker_n, total_core):
    if not marker_n or marker_n <= 0:
        return {"label": "none", "value": 0.0,
                "basis": "no input gene matches the axis core_genes"}
    total = max(int(total_core or 0), 1)
    value = min(marker_n / total, 1.0)
    if value >= 0.30:
        label = "high"
    elif value >= 0.10:
        label = "medium"
    else:
        label = "low"
    return {
        "label": label,
        "value": round(value, 3),
        "basis": f"{marker_n}/{total} axis core_genes are present in the input",
    }


def _confidence_from_p4(score, n_samples):
    s = abs(float(score or 0.0))
    n = int(n_samples or 0)
    if s >= 0.5 and n >= 2:
        label, value = "high", min(1.0, 0.7 + 0.1 * n)
    elif s >= 0.2 and n >= 1:
        label, value = "medium", 0.5
    else:
        label, value = "low", 0.2
    return {
        "label": label,
        "value": round(value, 3),
        "basis": f"score={s:.3f} supported by n_samples={n}",
    }


def suggested_validation_experiment(axis, *, top_n_genes=8):
    """Return a single-sentence wet-lab / dry-lab follow-up for one axis."""
    layer = axis.get("layer") or "functional_axis"
    template = _EXPERIMENT_TEMPLATE.get(layer, _EXPERIMENT_TEMPLATE["functional_axis"])
    genes = _axis_supporting_genes(axis)[:top_n_genes]
    if not genes:
        return (f"{template.split(':')[0]}: (no core_genes curated for this axis).")
    gene_list = ", ".join(genes)
    return template.format(genes=gene_list)


def build_axis_interpretation(axis, *, score=None, sample=None, top_n_genes=10,
                               marker_n=None, total_core=None, n_samples=None):
    """Return a structured, evidence-bound interpretation for one axis.

    ``marker_n`` and ``total_core`` (when given) drive the ``confidence`` field
    via overlap; otherwise the caller must pre-compute ``confidence`` and
    ``suggested_validation_experiment`` for p4-style flows.
    """
    core_total = int(total_core) if total_core is not None else len(_axis_supporting_genes(axis))
    if marker_n is not None:
        confidence = _confidence_from_overlap(marker_n, core_total)
    else:
        confidence = _confidence_from_p4(score, n_samples)
    return {
        "axis_id": axis.get("axis_id"),
        "layer": axis.get("layer"),
        "name_en": axis.get("name_en"),
        "name_cn": axis.get("name_cn"),
        "score": score,
        "sample": sample,
        "status": axis.get("status"),
        "safety_classification": axis_safety_class(axis),
        "supporting_genes": _axis_supporting_genes(axis)[:top_n_genes],
        "evidence": _axis_evidence_fields(axis),
        "limitations": list(axis.get("limitations") or []),
        "recommended_wording": list(axis.get("recommended_use") or []),
        "forbidden_wording": [],
        "confidence": confidence,
        "suggested_validation_experiment": suggested_validation_experiment(
            axis, top_n_genes=top_n_genes
        ),
    }


# ---------------------------------------------------------------------------
# Score-table summarization (shared by all three input modes)
# ---------------------------------------------------------------------------

def _summarize_score_table(scores, *, mode, input_meta, top_per_layer=3, overall_top=5):
    """Build an interpretation dict from a long-form score table.

    ``scores`` must have at least ``axis_id`` and ``score`` columns; a
    ``layer`` column is optional but recommended.
    """
    dictionary = load_cartilage_dictionary_v1()
    axes_by_id = _v1_axis_index(dictionary)

    unknown = sorted({str(a) for a in scores["axis_id"].astype(str).unique()
                      if str(a) not in axes_by_id})
    if unknown:
        scores = scores[~scores["axis_id"].astype(str).isin(unknown)].copy()

    if scores is None or len(scores) == 0:
        return {
            "mode": mode,
            "input": input_meta,
            "axis_count_scored": 0,
            "axis_count_kept": 0,
            "top_axes_per_layer": [],
            "overall_top_axes": [],
            "safety_summary": {},
            "warnings": (
                ["Empty score table; nothing to interpret."]
                + ([f"{len(unknown)} axis_id(s) are not in cartilage_dictionary_v1 and "
                    f"are reported with empty evidence/limitations: "
                    + ", ".join(unknown[:5])
                    + ("..." if len(unknown) > 5 else "")]
                   if unknown else [])
            ),
        }

    axes = []
    for axis_id, group in scores.groupby("axis_id"):
        if str(axis_id) in unknown:
            continue
        axis = axes_by_id.get(axis_id, {})
        top_row = group.sort_values("score", ascending=False).iloc[0]
        marker_n = int(top_row["marker_n"]) if "marker_n" in group.columns else None
        n_samples = int(group.shape[0]) if "sample" in group.columns else None
        axes.append(build_axis_interpretation(
            axis,
            score=float(top_row["score"]),
            sample=str(top_row.get("sample", "")) if "sample" in group.columns else None,
            marker_n=marker_n,
            n_samples=n_samples,
        ))

    by_layer = {}
    for a in axes:
        layer = a.get("layer") or ""
        by_layer.setdefault(layer, []).append(a)

    kept = []
    for layer, items in by_layer.items():
        items = sorted(items, key=lambda x: x.get("score") or 0, reverse=True)
        kept.extend(items[:top_per_layer])

    overall = sorted(axes, key=lambda x: x.get("score") or 0, reverse=True)[:overall_top]

    safety_summary = {}
    for a in kept:
        s = a["safety_classification"]
        safety_summary[s] = safety_summary.get(s, 0) + 1

    warnings = []
    if unknown:
        warnings.append(
            f"{len(unknown)} axis_id(s) are not in cartilage_dictionary_v1 and "
            f"are reported with empty evidence/limitations: "
            + ", ".join(unknown[:5])
            + ("..." if len(unknown) > 5 else "")
        )

    return {
        "mode": mode,
        "input": input_meta,
        "axis_count_scored": len(axes),
        "axis_count_kept": len(kept),
        "top_axes_per_layer": kept,
        "overall_top_axes": overall,
        "safety_summary": safety_summary,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Input modes
# ---------------------------------------------------------------------------

def interpret_gene_list(genes, *, top_per_layer=3, overall_top=5):
    """Score a gene list against v1 dictionary and return evidence-bound interpretation.

    Uses a direct per-axis overlap score against the curated
    ``core_genes`` (or ``panel_genes`` fallback) of each axis. The
    "expression" path through ``project_dictionary_v1_bulk`` is
    skipped intentionally because z-scoring over a single sample
    collapses every score to zero. The set-overlap score is the right
    scale for gene-list queries; downstream callers wanting
    abundance-weighted scores should run ``p4-project`` first and use
    ``interpret_p4_dir``.
    """
    genes_list = [str(g).strip() for g in genes if str(g).strip()]
    resolved = resolve_aliases(genes_list, load_alias_map())
    dictionary = load_cartilage_dictionary_v1()
    axes_by_id = _v1_axis_index(dictionary)
    if not resolved:
        return {
            "mode": "genes",
            "input": {"input_genes": genes_list, "resolved_genes": []},
            "axis_count_scored": 0,
            "axis_count_kept": 0,
            "top_axes_per_layer": [],
            "overall_top_axes": [],
            "safety_summary": {},
            "warnings": ["Empty gene list after normalization."],
        }
    query_set = {g.upper() for g in resolved}
    rows = []
    for axis_id, axis in axes_by_id.items():
        marker_weights = projection._axis_gene_weights(axis)
        anti_weights = projection._axis_anti_weights(axis)
        if not marker_weights:
            continue
        hits = query_set & set(marker_weights.keys())
        if not hits:
            continue
        overlap = len(hits)
        overlap_score = sum(marker_weights[g] for g in hits)
        denom = sum(marker_weights.values()) or 1.0
        score = round(overlap_score / denom, 6)
        rows.append({
            "axis_id": axis_id,
            "layer": axis.get("layer"),
            "name_en": axis.get("name_en"),
            "sample": "query",
            "score": score,
            "marker_n": overlap,
            "anti_n": len(query_set & set(anti_weights.keys())),
        })
    if not rows:
        return {
            "mode": "genes",
            "input": {"input_genes": genes_list, "resolved_genes": resolved,
                      "n_input": len(genes_list), "n_resolved": len(resolved)},
            "axis_count_scored": 0,
            "axis_count_kept": 0,
            "top_axes_per_layer": [],
            "overall_top_axes": [],
            "safety_summary": {},
            "warnings": ["No v1 axis contains any of the input genes."],
        }
    df = pd.DataFrame(rows)
    df["total_core"] = df["axis_id"].map(
        lambda aid: len(projection._axis_gene_weights(axes_by_id.get(aid, {})).keys())
    )
    interp = _summarize_score_table(
        df,
        mode="genes",
        input_meta={"input_genes": genes_list, "resolved_genes": resolved,
                    "n_input": len(genes_list), "n_resolved": len(resolved),
                    "scoring": "weighted set overlap on core_genes"},
        top_per_layer=top_per_layer,
        overall_top=overall_top,
    )
    return interp



def interpret_p4_dir(outdir, *, top_per_layer=3, overall_top=5):
    """Interpret a P4 outdir produced by ``cartigsfm p4-project``.

    Reads ``tsv/p4_sample_cluster_three_layer_scores.tsv``. The
    interpretation is grouped per axis_id and reports the highest scoring
    sample-cluster for each axis.
    """
    outdir = Path(outdir)
    tsv_dir = outdir / "tsv"
    scores_path = tsv_dir / "p4_sample_cluster_three_layer_scores.tsv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. Run cartigsfm p4-project first."
        )
    scores = pd.read_csv(scores_path, sep="\t")
    return _summarize_score_table(
        scores,
        mode="p4_dir",
        input_meta={"p4_outdir": str(outdir), "n_rows": int(len(scores))},
        top_per_layer=top_per_layer,
        overall_top=overall_top,
    )


def interpret_p4_csv(path, *, axis_col="axis_id", score_col="score",
                     layer_col="layer", sample_col="sample",
                     top_per_layer=3, overall_top=5):
    """Interpret a long-form score CSV (axis_id, score, [layer], [sample])."""
    df = pd.read_csv(path, sep=None, engine="python")
    missing = [c for c in (axis_col, score_col) if c not in df.columns]
    if missing:
        raise KeyError(f"score CSV is missing required column(s): {missing}")
    return _summarize_score_table(
        df,
        mode="p4_csv",
        input_meta={"csv": str(path), "n_rows": int(len(df))},
        top_per_layer=top_per_layer,
        overall_top=overall_top,
    )


# ---------------------------------------------------------------------------
# Claim safety
# ---------------------------------------------------------------------------

def classify_claim(text):
    """Classify a user-asserted claim against bundled claim safety rules.

    Looks for an exact match in ``p6_claim_safety_classifier`` first, then
    falls back to the regex-based overclaim guard.
    """
    text_norm = text.strip().casefold()
    if not text_norm:
        return {
            "claim": text,
            "matched": False,
            "safety_classification": "UNREVIEWED",
            "can_claim": False,
            "rationale": "Empty claim.",
        }
    for entry in load_claim_safety_classifier():
        if str(entry.get("claim", "")).strip().casefold() == text_norm:
            out = dict(entry)
            out["matched"] = True
            return out
    for pat, msg in FORBIDDEN_PHRASES:
        if re.search(pat, text, flags=re.IGNORECASE):
            return {
                "claim": text,
                "matched": False,
                "safety_classification": "NOT_SUPPORTED",
                "can_claim": False,
                "rationale": f"Overclaim guard: {msg}",
                "matched_pattern": pat,
            }
    return {
        "claim": text,
        "matched": False,
        "safety_classification": "UNREVIEWED",
        "can_claim": True,
        "rationale": "No exact safety entry matched and no overclaim pattern hit. "
                     "Treat as unverified until reviewed against evidence.",
    }


def apply_safety_filter(interpretation, *, additional_claims=None):
    """Inject hard constraints, run claim safety checks, and return a new dict.

    This function does not mutate ``interpretation``. The returned dict has
    additional keys: ``hard_constraints``, ``cannot_claim``, ``claim_audits``.
    """
    interp = dict(interpretation)
    warnings = list(interp.get("warnings") or [])
    for c in HARD_CONSTRAINTS:
        warnings.append(f"HARD_CONSTRAINT: {c}")
    interp["warnings"] = warnings
    interp["hard_constraints"] = list(HARD_CONSTRAINTS)

    cannot_claim = list(interp.get("cannot_claim") or [])
    cannot_claim.extend(HARD_CONSTRAINTS)
    audits = list(interp.get("claim_audits") or [])
    if additional_claims:
        for c in additional_claims:
            entry = classify_claim(c)
            audits.append(entry)
            if not entry.get("can_claim", True):
                cls = entry.get("safety_classification")
                rat = entry.get("rationale")
                cannot_claim.append(
                    f"claim {c!r} -> {cls}: {rat}"
                )
    interp["cannot_claim"] = cannot_claim
    interp["claim_audits"] = audits
    return interp


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_markdown(interpretation):
    """Render an interpretation as a Markdown report."""
    md_lines = []
    mode = interpretation.get("mode", "?")
    md_lines.append(f"# CartiGSFM Evidence-Constrained Interpretation ({mode})")
    md_lines.append("")
    inp = interpretation.get("input") or {}
    md_lines.append("## Input")
    if inp:
        for k, v in inp.items():
            md_lines.append(f"- {k}: {v}")
    else:
        md_lines.append("- (none)")
    md_lines.append("")

    ss = interpretation.get("safety_summary") or {}
    if ss:
        md_lines.append("## Safety Summary (top axes per layer)")
        for k, v in sorted(ss.items()):
            md_lines.append(f"- {k}: {v}")
        md_lines.append("")

    hc = interpretation.get("hard_constraints") or []
    if hc:
        md_lines.append("## Hard Constraints (from P9 model card)")
        for c in hc:
            md_lines.append(f"- {c}")
        md_lines.append("")

    kept = interpretation.get("axis_count_kept", 0)
    md_lines.append(f"## Top Axes per Layer (kept={kept})")
    for a in interpretation.get("top_axes_per_layer") or []:
        ax_id = a.get("axis_id")
        ax_layer = a.get("layer")
        md_lines.append(f"### {ax_id} (layer: {ax_layer})")
        ax_score = a.get("score")
        ax_sample = a.get("sample")
        ax_status = a.get("status")
        ax_safety = a.get("safety_classification")
        conf = a.get("confidence") or {}
        md_lines.append(
            f"- score: {ax_score} sample: {ax_sample} status: {ax_status} "
            f"safety: {ax_safety} confidence: {conf.get('label', 'n/a')} "
            f"({conf.get('value', 0):.3f}, {conf.get('basis', '')})"
        )
        sg = a.get("supporting_genes") or []
        if sg:
                    sg_joined = ", ".join(sg[:10])
        md_lines.append(f"- supporting genes: {sg_joined}")
        ev = a.get("evidence") or {}
        indep = ev.get("independent_validation") or []
        if indep:
            md_lines.append(f"- independent validation: {len(indep)} item(s)")
        else:
            md_lines.append("- independent validation: NONE (atlas-internal only)")
        rw = a.get("recommended_wording") or []
        if rw:
            md_lines.append("- recommended wording:")
            for w in rw[:3]:
                md_lines.append(f"  - {w}")
        lim = a.get("limitations") or []
        if lim:
            md_lines.append("- limitations:")
            for w in lim[:3]:
                md_lines.append(f"  - {w}")
        sve = a.get("suggested_validation_experiment")
        if sve:
            md_lines.append(f"- suggested validation experiment: {sve}")
        md_lines.append("")

    ot = interpretation.get("overall_top_axes") or []
    if ot:
        md_lines.append("## Overall Top Axes")
        for a in ot:
            ot_id = a.get("axis_id")
            ot_score = a.get("score")
            ot_safety = a.get("safety_classification")
            md_lines.append(f"- {ot_id} score={ot_score} safety={ot_safety}")
        md_lines.append("")

    cc = interpretation.get("cannot_claim") or []
    if cc:
        md_lines.append("## Cannot Claim")
        for c in cc:
            md_lines.append(f"- {c}")
        md_lines.append("")

    audits = interpretation.get("claim_audits") or []
    if audits:
        md_lines.append("## Claim Audits")
        for a in audits:
            audit_claim = a.get("claim")
            audit_safety = a.get("safety_classification")
            audit_can = a.get("can_claim")
            md_lines.append(f"- {audit_claim} -> {audit_safety} (can_claim={audit_can})")
        md_lines.append("")

    wn = interpretation.get("warnings") or []
    if wn:
        md_lines.append("## Warnings")
        for w in wn:
            md_lines.append(f"- {w}")
        md_lines.append("")
    return "\n".join(md_lines) + "\n"


def render_json(interpretation):
    """Render an interpretation as a JSON string."""
    return json.dumps(interpretation, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def build_interpretation(*, mode, input_value=None, genes=None,
                         top_per_layer=3, overall_top=5):
    """Dispatch helper used by the CLI and tests."""
    if mode == "genes":
        if genes is None and input_value:
            genes = [g.strip() for g in str(input_value).replace(",", "\n").splitlines() if g.strip()]
        if genes is None:
            raise ValueError("mode=genes requires --genes or positional input")
        return interpret_gene_list(genes, top_per_layer=top_per_layer, overall_top=overall_top)
    if mode == "p4-dir":
        if not input_value:
            raise ValueError("mode=p4-dir requires an --input path")
        return interpret_p4_dir(input_value, top_per_layer=top_per_layer, overall_top=overall_top)
    if mode == "p4-csv":
        if not input_value:
            raise ValueError("mode=p4-csv requires an --input path")
        return interpret_p4_csv(input_value, top_per_layer=top_per_layer, overall_top=overall_top)
    raise ValueError(f"unknown mode {mode!r}; expected genes|p4-dir|p4-csv")
