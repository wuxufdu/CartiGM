"""Project expression matrices onto CartiGSFM subtype and function panels."""
from __future__ import annotations
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def _prepare_expression(
    expr_df: pd.DataFrame,
    sample_cols: Optional[Iterable[str]] = None,
    gene_col: Optional[str] = None,
    alias_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Return numeric genes x samples matrix with optional symbol normalization."""
    df = expr_df.copy()
    if gene_col and gene_col in df.columns:
        df = df.set_index(gene_col)
    if sample_cols:
        df = df[list(sample_cols)]
    df.index = [str(g).strip().upper() for g in df.index]
    if alias_map:
        context_overrides = {
            "TAZ": "WWTR1",
            "YAP": "YAP1",
            "PDGFR": "PDGFRA",
            "TGFB": "TGFB1",
            "TNFRSF11B": "TNFRSF11B",
        }
        df.index = [context_overrides.get(g, alias_map.get(g, g)) for g in df.index]
        df = df.groupby(level=0).mean(numeric_only=True)
    return df.apply(pd.to_numeric, errors="coerce")


def project_bulk(
    expr_df: pd.DataFrame,
    dictionary: Dict,
    sample_cols: Optional[Iterable[str]] = None,
    gene_col: Optional[str] = None,
    anti_lambda: float = 0.5,
    alias_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Project a bulk expression matrix onto cgrm subtypes.

    expr_df rows = genes, columns = samples (plus optional gene_col).
    Each (sample, subtype) score = mean panel z-score - anti_lambda * mean anti z-score.
    """
    df = _prepare_expression(expr_df, sample_cols=sample_cols, gene_col=gene_col, alias_map=alias_map)
    # row-z each gene across samples to compare against panels in z-space
    z = df.apply(_zscore, axis=1)
    rows = []
    for cat, e in dictionary.items():
        panel = list((e.get("panel_weights") or {}).keys())
        anti = list((e.get("anti_panel_weights") or {}).keys())
        in_p = [g for g in panel if g in z.index]
        in_a = [g for g in anti if g in z.index]
        if not in_p:
            continue
        p_score = z.loc[in_p].mean(axis=0)
        a_score = z.loc[in_a].mean(axis=0) if in_a else 0.0
        score = p_score - anti_lambda * a_score
        for sample, val in score.items():
            rows.append({
                "subtype": cat,
                "sample": sample,
                "score": round(float(val), 4),
                "panel_n": len(in_p),
                "anti_n": len(in_a),
            })
    return pd.DataFrame(rows)


def project_function_bulk(
    expr_df: pd.DataFrame,
    specificity: Dict,
    dictionary: Optional[Dict] = None,
    sample_cols: Optional[Iterable[str]] = None,
    gene_col: Optional[str] = None,
    alias_map: Optional[Dict[str, str]] = None,
    consensus_weight: float = 0.25,
) -> pd.DataFrame:
    """Project an expression matrix onto function marker axes.

    Scores are computed in gene-wise z-score space. The primary score is the
    marker-weighted mean z-score; when consensus genes are available, a smaller
    consensus mean term is added as supporting evidence.
    """
    df = _prepare_expression(expr_df, sample_cols=sample_cols, gene_col=gene_col, alias_map=alias_map)
    z = df.apply(_zscore, axis=1)
    dictionary = dictionary or {}
    rows = []
    for cat, info in specificity.items():
        markers = info.get("markers") or []
        marker_weights = {
            str(m.get("gene")): float(m.get("weight", 1.0))
            for m in markers
            if isinstance(m, dict) and m.get("gene")
        }
        in_m = [g for g in marker_weights if g in z.index]
        if not in_m:
            continue
        weights = pd.Series({g: marker_weights[g] for g in in_m}, dtype=float)
        marker_score = z.loc[in_m].mul(weights, axis=0).sum(axis=0) / weights.sum()
        consensus = [g for g in ((dictionary.get(cat) or {}).get("consensus_genes") or []) if g in z.index]
        consensus_score = z.loc[consensus].mean(axis=0) if consensus else 0.0
        score = marker_score + consensus_weight * consensus_score
        for sample, val in score.items():
            rows.append({
                "function": cat,
                "sample": sample,
                "score": round(float(val), 4),
                "marker_n": len(in_m),
                "consensus_n": len(consensus),
            })
    return pd.DataFrame(rows)


def _axis_gene_weights(axis: Dict) -> Dict[str, float]:
    weights = axis.get("marker_weights") or {}
    if weights:
        return {str(g).upper(): float(w) for g, w in weights.items()}
    panel = axis.get("panel_genes") or []
    out = {}
    for item in panel:
        if isinstance(item, dict):
            gene = item.get("gene")
            weight = item.get("weight", 1.0)
        else:
            gene = item
            weight = 1.0
        if gene:
            out[str(gene).upper()] = float(weight)
    return out


def _axis_anti_weights(axis: Dict) -> Dict[str, float]:
    weights = axis.get("anti_marker_weights") or {}
    if weights:
        return {str(g).upper(): float(w) for g, w in weights.items()}
    anti = axis.get("anti_genes") or []
    out = {}
    for item in anti:
        if isinstance(item, dict):
            gene = item.get("gene")
            weight = item.get("weight", 1.0)
        else:
            gene = item
            weight = 1.0
        if gene:
            out[str(gene).upper()] = float(weight)
    return out


def project_dictionary_v1_bulk(
    expr_df: pd.DataFrame,
    cartilage_dictionary_v1: Dict,
    sample_cols: Optional[Iterable[str]] = None,
    gene_col: Optional[str] = None,
    anti_lambda: float = 0.5,
    alias_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Project expression onto all three layers of ``cartilage_dictionary_v1``.

    The input is a genes x samples matrix, optionally with a gene column.
    Each axis score is a weighted mean gene-wise z-score minus an optional
    anti-marker penalty. The output is long-form with columns:
    ``layer, axis_id, name_en, sample, score, marker_n, anti_n``.
    """
    columns = ["layer", "axis_id", "name_en", "name_cn", "sample", "score", "marker_n", "anti_n"]
    df = _prepare_expression(expr_df, sample_cols=sample_cols, gene_col=gene_col, alias_map=alias_map)
    z = df.apply(_zscore, axis=1)
    rows = []
    for layer, layer_obj in (cartilage_dictionary_v1.get("layers") or {}).items():
        for axis in layer_obj.get("axes", []):
            marker_weights = _axis_gene_weights(axis)
            anti_weights = _axis_anti_weights(axis)
            in_m = [g for g in marker_weights if g in z.index]
            if not in_m:
                continue
            weights = pd.Series({g: marker_weights[g] for g in in_m}, dtype=float)
            marker_score = z.loc[in_m].mul(weights, axis=0).sum(axis=0) / weights.sum()
            in_a = [g for g in anti_weights if g in z.index]
            if in_a:
                anti_series = pd.Series({g: anti_weights[g] for g in in_a}, dtype=float)
                anti_score = z.loc[in_a].mul(anti_series, axis=0).sum(axis=0) / anti_series.sum()
            else:
                anti_score = 0.0
            score = marker_score - anti_lambda * anti_score
            for sample, value in score.items():
                rows.append({
                    "layer": layer,
                    "axis_id": axis.get("axis_id", ""),
                    "name_en": axis.get("name_en", ""),
                    "name_cn": axis.get("name_cn", ""),
                    "sample": sample,
                    "score": round(float(value), 4),
                    "marker_n": len(in_m),
                    "anti_n": len(in_a),
                })
    return pd.DataFrame(rows, columns=columns)
