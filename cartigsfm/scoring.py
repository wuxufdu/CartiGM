"""Score a gene query against CartiGSFM dictionaries.

Mirrors the scripts/16 combined-score logic but exposed as a clean API:
    overlap_score + (anti-penalty) per subtype.
"""
from __future__ import annotations
from typing import Dict, Iterable, List

import pandas as pd


CONTEXT_OVERRIDES = {
    "TAZ": "WWTR1",
    "YAP": "YAP1",
    "PDGFR": "PDGFRA",
    "TGFB": "TGFB1",
    "TNFRSF11B": "TNFRSF11B",
}


def resolve_aliases(genes: Iterable[str], alias_map: Dict[str, str] | None = None) -> List[str]:
    """Resolve common aliases to current symbols and de-duplicate in order."""
    alias_map = alias_map or {}
    out = []
    seen = set()
    for raw in genes:
        g = str(raw).strip().upper()
        if not g:
            continue
        mapped = CONTEXT_OVERRIDES.get(g, alias_map.get(g, g))
        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out


def score_query(
    genes: Iterable[str],
    dictionary: Dict,
    anti_penalty: float = 1.0,
    min_overlap: int = 1,
) -> pd.DataFrame:
    """Score a gene query against every subtype in dictionary.

    Returns a DataFrame sorted by combined descending with columns:
        subtype, overlap, overlap_score, anti_overlap, anti_score, combined
    """
    q_set = set(str(g) for g in genes if g)
    if not q_set:
        return pd.DataFrame(
            columns=["subtype", "overlap", "overlap_score",
                     "anti_overlap", "anti_score", "combined"]
        )
    rows = []
    for cat, e in dictionary.items():
        panel = e.get("panel_weights") or {}
        anti = e.get("anti_panel_weights") or {}
        if not panel:
            continue
        # weighted overlap
        overlap_genes = q_set & set(panel.keys())
        overlap = len(overlap_genes)
        if overlap < min_overlap:
            continue
        overlap_score = sum(float(panel[g]) for g in overlap_genes) / max(
            1.0, sum(float(w) for w in panel.values())
        )
        anti_overlap_genes = q_set & set(anti.keys())
        anti_overlap = len(anti_overlap_genes)
        anti_score = (
            sum(float(anti[g]) for g in anti_overlap_genes) / max(
                1.0, sum(float(w) for w in anti.values())
            )
            if anti else 0.0
        )
        combined = overlap_score - anti_penalty * anti_score
        rows.append({
            "subtype": cat,
            "overlap": overlap,
            "overlap_score": round(overlap_score, 4),
            "anti_overlap": anti_overlap,
            "anti_score": round(anti_score, 4),
            "combined": round(combined, 4),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("combined", ascending=False).reset_index(drop=True)


def score_function_query(
    genes: Iterable[str],
    specificity: Dict,
    dictionary: Dict | None = None,
    min_overlap: int = 1,
) -> pd.DataFrame:
    """Score a gene query against function specificity markers.

    Function dictionaries store consensus genes separately from specificity
    markers, so this uses marker overlap as the primary axis and reports
    consensus overlap as supporting evidence.
    """
    q_set = set(str(g) for g in genes if g)
    columns = [
        "function", "overlap", "overlap_score", "marker_weight",
        "consensus_overlap", "consensus_score", "combined",
    ]
    if not q_set:
        return pd.DataFrame(columns=columns)
    rows = []
    dictionary = dictionary or {}
    for cat, info in specificity.items():
        markers = info.get("markers") or []
        marker_weights = {
            str(m.get("gene")): float(m.get("weight", 1.0))
            for m in markers
            if isinstance(m, dict) and m.get("gene")
        }
        if not marker_weights:
            continue
        marker_hits = q_set & set(marker_weights)
        overlap = len(marker_hits)
        if overlap < min_overlap:
            continue
        overlap_score = overlap / max(1, len(q_set))
        marker_weight = sum(marker_weights[g] for g in marker_hits) / max(
            1.0, sum(marker_weights.values())
        )
        consensus = set((dictionary.get(cat) or {}).get("consensus_genes") or [])
        consensus_overlap = len(q_set & consensus) if consensus else 0
        consensus_score = consensus_overlap / max(1, len(q_set)) if consensus else 0.0
        combined = overlap_score + marker_weight + 0.25 * consensus_score
        rows.append({
            "function": cat,
            "overlap": overlap,
            "overlap_score": round(overlap_score, 4),
            "marker_weight": round(marker_weight, 4),
            "consensus_overlap": consensus_overlap,
            "consensus_score": round(consensus_score, 4),
            "combined": round(combined, 4),
        })
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    return df.sort_values("combined", ascending=False).reset_index(drop=True)
