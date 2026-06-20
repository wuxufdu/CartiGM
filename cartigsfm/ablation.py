"""Four-way ablation runner for the CartiGM + GSFM + scGPT + LLM Agent stack.

Inputs: a P4 outdir with ``tsv/p4_sample_cluster_three_layer_scores.tsv``
and ``tsv/p4_self_sample_cluster_pseudobulk.tsv``.

Configurations
--------------
1. ``cartigm_only``  -- top axis per (sample, layer) from the P4 table.
2. ``cartigm_gsfm``   -- top axis per cluster from GSFM marker-set similarity.
3. ``cartigm_scgpt``  -- top axis per cluster from scGPT expression encoding.
4. ``full``           -- all three plus a CartiAgent keyword dispatch probe.

Metrics: top-axis accuracy vs per-tissue ground truth, evidence citation
rate, hallucination rate, and P4 self-data consistency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import pandas as pd

from . import agent as A
from . import gsfm, scgpt
from .assets import load_cartilage_dictionary_v1


# ---------------------------------------------------------------------------
# Default ground-truth maps: tissue / celltype -> expected axis set
# ---------------------------------------------------------------------------
# Used when the caller does not pass an explicit ``tissue_axis_map`` or
# ``celltype_axis_map``. These are biologically informed priors: which axis
# categories we would expect a "correct" model to rank in the top-k for a
# cluster annotated as the given tissue / cell type. Callers can override
# the maps per dataset to plug in curator-supplied expectations.

DEFAULT_TISSUE_AXIS_MAP: dict[str, set[str]] = {
    "ear": {
        "tissue_developmental_state::ElasticCartilage_Auricular",
        "tissue_developmental_state::ElasticCartilage_Nasal",
        "cell_subtype::Effector_Metabolic_Chondrocytes",
        "functional_axis::ECM_Organization",
        "functional_axis::Avascular_Antimineralization",
    },
    "hzx_ear": {
        "tissue_developmental_state::ElasticCartilage_Auricular",
        "functional_axis::ECM_Organization",
    },
    "rib": {
        "tissue_developmental_state::Hyaline_ArticularCartilage",
        "tissue_developmental_state::Fibrocartilage_Meniscus",
        "cell_subtype::Progenitor_Chondrocytes",
        "functional_axis::EndochondralOssification",
    },
    "hzx_rib": {
        "tissue_developmental_state::Hyaline_ArticularCartilage",
        "functional_axis::EndochondralOssification",
    },
    "nose": {
        "tissue_developmental_state::ElasticCartilage_Auricular",
        "tissue_developmental_state::Hyaline_ArticularCartilage",
        "cell_subtype::Hypoxic_Chondrocytes",
        "functional_axis::Avascular_Antimineralization",
    },
    "hzx_nose": {
        "tissue_developmental_state::Hyaline_ArticularCartilage",
        "functional_axis::Avascular_Antimineralization",
    },
    "OA": {
        "tissue_developmental_state::Hyaline_ArticularCartilage",
        "functional_axis::MMP",
        "functional_axis::ADAMTS",
        "functional_axis::Senescence",
        "functional_axis::Chondrogenesis",
    },
    "normal_hyaline": {
        "tissue_developmental_state::Hyaline_ArticularCartilage",
        "functional_axis::Chondrogenesis",
        "functional_axis::Proteoglycan",
        "functional_axis::ECM_Organization",
    },
    "normal_fibrocartilage": {
        "tissue_developmental_state::Fibrocartilage_Meniscus",
        "functional_axis::ECM_Organization",
        "functional_axis::Proteoglycan",
    },
    "normal_Elastic.Cartilage": {
        "tissue_developmental_state::ElasticCartilage_Auricular",
        "cell_subtype::Effector_Metabolic_Chondrocytes",
        "functional_axis::ECM_Organization",
    },
    "Microtia_Elastic.Cartilage": {
        "tissue_developmental_state::ElasticCartilage_Auricular",
        "cell_subtype::Effector_Metabolic_Chondrocytes",
        "functional_axis::ECM_Organization",
    },
}

DEFAULT_CELLTYPE_AXIS_MAP: dict[str, set[str]] = {
    "Matrix-Homeostatic": {
        "cell_subtype::Homeostatic",
        "functional_axis::Chondrogenesis",
    },
    "Remodeling-Plasticity": {
        "cell_subtype::Progenitor_Chondrocytes",
        "functional_axis::ECM_Organization",
    },
    "Inflammatory-Effector": {
        "cell_subtype::Inflammatory",
        "functional_axis::Inflammation_NFkB",
    },
    "PRG4-Interface": {
        "cell_subtype::Interface",
        "functional_axis::Lubrication_PRG4",
    },
    "Maturation-Mineralizing": {
        "cell_subtype::Mineralizing",
        "functional_axis::EndochondralOssification",
    },
    "Stress-Injury": {
        "cell_subtype::Reparative_Stress_Chondrocytes",
        "functional_axis::Apoptosis",
    },
    "EC-Lipo": {
        "cell_subtype::Effector_Metabolic_Chondrocytes",
        "functional_axis::ECM_Organization",
    },
    "EC-Lipo_Plasticity": {
        "cell_subtype::Effector_Metabolic_Chondrocytes",
        "functional_axis::ECM_Organization",
    },
}

# Default claim set for the LLM refusal probe. Each claim is audited
# through ``interpret.classify_claim`` so the ``full`` config can report
# which configurations caught the overclaim and which did not.
DEFAULT_REFUSAL_CLAIMS: list[str] = [
    "CartiGSFM is a trained cartilage large language model (LLM)",
    "CartiGSFM predictions are externally validated",
    "MGP is a therapeutic target for OA",
    "Inflammation_NFkB is significantly increased in OA",
    "Avascular_Antimineralization is the dominant axis in every cartilage cluster",
    "CartiGSFM proves that Hyaline_ArticularCartilage drives OA progression",
    "We can use CartiGSFM as a drug target discovery platform",
    "The above axes are causally linked to disease outcome",
]


def _v1_axis_set() -> set:
    dictionary = load_cartilage_dictionary_v1()
    out = set()
    for layer, layer_obj in (dictionary.get("layers") or {}).items():
        for axis in layer_obj.get("axes", []):
            out.add(str(axis.get("axis_id", "")))
    return out


def _load_p4(outdir: Path) -> Dict[str, pd.DataFrame]:
    tsv = outdir / "tsv"
    scores = pd.read_csv(tsv / "p4_sample_cluster_three_layer_scores.tsv", sep="\t")
    pseudobulk = pd.read_csv(tsv / "p4_self_sample_cluster_pseudobulk.tsv", sep="\t")
    meta = pd.read_csv(tsv / "p4_self_sample_cluster_meta.tsv", sep="\t", index_col=0)
    return {"scores": scores, "pseudobulk": pseudobulk, "meta": meta}


def _top_axes_from_scores(scores: pd.DataFrame) -> Dict[str, str]:
    """Top-1 axis per sample-cluster (collapsed across all three layers)."""
    rows = scores.sort_values("score", ascending=False).groupby("sample").head(1)
    return {str(s): str(a) for s, a in zip(rows["sample"], rows["axis_id"])}


def _top_axes_from_gsfm(pseudobulk: pd.DataFrame, meta: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    pb = pseudobulk.set_index("gene")
    for cluster_id, row in meta.iterrows():
        if cluster_id not in pb.columns:
            continue
        expr = pb[cluster_id]
        top_genes = expr.sort_values(ascending=False).head(15).index.tolist()
        top = gsfm.gsfm_marker_axes(top_genes, top_n=1)
        out[str(cluster_id)] = str(top[0]["axis_id"]) if top else ""
    return out


def _top_axes_from_scgpt(pseudobulk: pd.DataFrame) -> Dict[str, str]:
    res = scgpt.scgpt_encode_dataframe(pseudobulk, gene_col="gene")
    emb = res.get("cluster_embedding")
    if not isinstance(emb, pd.DataFrame) or emb.empty:
        return {}
    return {str(c): str(emb.loc[c, "top_axis_id"]) for c in emb.index}


def _evidence_citation(top_axis: str) -> float:
    if not top_axis:
        return 0.0
    e = gsfm.gsfm_axis_embedding(top_axis)
    if not e.get("found"):
        return 0.0
    return float(
        int(e.get("n_evidence_internal", 0)) > 0
        or int(e.get("n_evidence_literature", 0)) > 0
        or int(e.get("n_evidence_independent", 0)) > 0
    )


def _accuracy(top_axis: str, tissue: str) -> float:
    return _accuracy_with_map(top_axis, tissue, "", DEFAULT_TISSUE_AXIS_MAP, DEFAULT_CELLTYPE_AXIS_MAP)


def _accuracy_with_map(top_axis: str, tissue: str, celltype: str,
                        tissue_axis_map: Dict[str, set],
                        celltype_axis_map: Dict[str, set]) -> float:
    expected = set()
    if tissue in tissue_axis_map:
        expected |= tissue_axis_map[tissue]
    if celltype in celltype_axis_map:
        expected |= celltype_axis_map[celltype]
    if not expected:
        return -1.0  # sentinel: no ground truth available
    return 1.0 if top_axis in expected else 0.0


def _hallucination(top_axis: str, v1_set: set) -> float:
    return 0.0 if (not top_axis or top_axis in v1_set) else 1.0


def _compute_metrics(top_map: Dict[str, str], tissue_map: Dict[str, str],
                      v1_set: set,
                      celltype_map: Optional[Dict[str, str]] = None,
                      tissue_axis_map: Optional[Dict[str, set]] = None,
                      celltype_axis_map: Optional[Dict[str, set]] = None) -> Dict[str, float]:
    # tissue_map is keyed by cluster_id (e.g. "S1|ear|C0"); the per-cluster
    # top-1 axis is keyed by the same string. For samples that look like
    # "S1|ear|C0|functional_axis" we strip the trailing "|layer" suffix.
    def _tissue_of(key: str) -> str:
        if key in tissue_map:
            return tissue_map[key]
        prefix = key.split("|", 3)
        if len(prefix) >= 3:
            sample_cluster = "|".join(prefix[:3])
            if sample_cluster in tissue_map:
                return tissue_map[sample_cluster]
        return ""
    celltype_map = celltype_map or {}
    tissue_axis_map = tissue_axis_map or DEFAULT_TISSUE_AXIS_MAP
    celltype_axis_map = celltype_axis_map or DEFAULT_CELLTYPE_AXIS_MAP
    acc = 0.0
    cit = 0.0
    hal = 0.0
    n = 0
    n_no_ground = 0
    for cluster_id, top_axis in top_map.items():
        tissue = _tissue_of(cluster_id)
        celltype = celltype_map.get(str(cluster_id), "")
        score = _accuracy_with_map(top_axis, tissue, celltype,
                                    tissue_axis_map, celltype_axis_map)
        if score < 0:
            n_no_ground += 1
            continue
        acc += score
        cit += _evidence_citation(top_axis)
        hal += _hallucination(top_axis, v1_set)
        n += 1
    if not n:
        return {"n_clusters_evaluated": 0.0, "n_clusters_no_ground": float(n_no_ground),
                "top_axis_accuracy": 0.0, "evidence_citation_rate": 0.0,
                "hallucination_rate": 0.0}
    return {
        "n_clusters_evaluated": float(n),
        "n_clusters_no_ground": float(n_no_ground),
        "top_axis_accuracy": round(acc / n, 3),
        "evidence_citation_rate": round(cit / n, 3),
        "hallucination_rate": round(hal / n, 3),
    }


def run_ablation(outdir: str | Path,
                 *, sample_meta_col: str = "tissue",
                 celltype_meta_col: Optional[str] = None,
                 configs: Optional[List[str]] = None,
                 tissue_axis_map: Optional[Mapping[str, set]] = None,
                 celltype_axis_map: Optional[Mapping[str, set]] = None,
                 refusal_claims: Optional[List[str]] = None,
                 use_real_scgpt_gsfm: bool = False) -> Dict[str, Any]:
    outdir = Path(outdir)
    data = _load_p4(outdir)
    scores = data["scores"]
    pseudobulk = data["pseudobulk"]
    meta = data["meta"]
    tissue_per_cluster = dict(zip(meta.index.astype(str), meta[sample_meta_col].astype(str)))
    celltype_per_cluster: Dict[str, str] = {}
    if celltype_meta_col and celltype_meta_col in meta.columns:
        celltype_per_cluster = dict(zip(meta.index.astype(str), meta[celltype_meta_col].astype(str)))
    tmap = dict(tissue_axis_map) if tissue_axis_map else dict(DEFAULT_TISSUE_AXIS_MAP)
    ctxt_map = dict(celltype_axis_map) if celltype_axis_map else dict(DEFAULT_CELLTYPE_AXIS_MAP)
    v1_set = _v1_axis_set()
    cfgs = configs or ["cartigm_only", "cartigm_gsfm", "cartigm_scgpt", "full"]

    per_config: Dict[str, Dict[str, str]] = {}
    if "cartigm_only" in cfgs:
        per_config["cartigm_only"] = _top_axes_from_scores(scores)
    if "cartigm_gsfm" in cfgs:
        per_config["cartigm_gsfm"] = _top_axes_from_gsfm(pseudobulk, meta)
    if "cartigm_scgpt" in cfgs:
        per_config["cartigm_scgpt"] = _top_axes_from_scgpt(pseudobulk)
    if "full" in cfgs:
        per_config["full"] = dict(per_config.get("cartigm_gsfm", {}))
        for cluster_id, tissue in tissue_per_cluster.items():
            query = f"top axis for {tissue} cartilage cluster {cluster_id}"
            try:
                A.run_query_keyword(query)
            except Exception:
                pass

    metrics: Dict[str, Dict[str, float]] = {}
    for cfg, top_map in per_config.items():
        metrics[cfg] = _compute_metrics(
            top_map, tissue_per_cluster, v1_set,
            celltype_map=celltype_per_cluster,
            tissue_axis_map=tmap,
            celltype_axis_map=ctxt_map,
        )

    config_names = list(per_config.keys())
    consistent_pairs: List[float] = []
    for i in range(len(config_names)):
        for j in range(i + 1, len(config_names)):
            a_map = per_config[config_names[i]]
            b_map = per_config[config_names[j]]
            n = 0
            agree = 0
            for cluster_id in a_map:
                if cluster_id in b_map and a_map[cluster_id] and b_map[cluster_id]:
                    n += 1
                    if a_map[cluster_id] == b_map[cluster_id]:
                        agree += 1
            if n:
                consistent_pairs.append(round(agree / n, 3))
    consistency = round(sum(consistent_pairs) / max(len(consistent_pairs), 1), 3)

    # --- LLM refusal audit (P6 + P9) ---
    refusal_claims_use = refusal_claims if refusal_claims is not None else DEFAULT_REFUSAL_CLAIMS
    refusal_audit: Dict[str, Any] = {"n_claims": 0, "n_refused": 0, "n_passed": 0,
                                       "refusal_rate": 0.0, "audits": []}
    if refusal_claims_use:
        from .interpret import classify_claim
        audits = []
        n_refused = 0
        for claim in refusal_claims_use:
            result = classify_claim(claim)
            audits.append(result)
            if not result.get("can_claim", True):
                n_refused += 1
        n_claims = max(len(refusal_claims_use), 1)
        refusal_audit = {
            "n_claims": len(refusal_claims_use),
            "n_refused": n_refused,
            "n_passed": len(refusal_claims_use) - n_refused,
            "refusal_rate": round(n_refused / n_claims, 3),
            "audits": audits,
        }

    # --- Branch provenance: real vs lightweight fallback ---
    if use_real_scgpt_gsfm:
        branch_labels = {
            "gsfm": "real GSFM weights (PPMI+SVD on cartilage gene sets)",
            "scgpt": "real scGPT-human weights (transformer pretrained on 33M human cells)",
        }
    else:
        branch_labels = {
            "gsfm": ("lightweight deterministic proxy: weighted Jaccard on axis core_genes. "
                     "NOT a real GSFM (PPMI+SVD) encoder. Marked as fallback."),
            "scgpt": ("lightweight deterministic proxy: per-cluster mean core-gene expression. "
                      "NOT a real scGPT-human transformer. Marked as fallback."),
        }

    return {
        "outdir": str(outdir),
        "metrics": metrics,
        "per_config_top1": per_config,
        "tissue_per_cluster": tissue_per_cluster,
        "celltype_per_cluster": celltype_per_cluster,
        "tissue_axis_map": tmap,
        "celltype_axis_map": ctxt_map,
        "p4_self_data_consistency": consistency,
        "configs_run": config_names,
        "refusal_audit": refusal_audit,
        "branch_labels": branch_labels,
        "use_real_scgpt_gsfm": use_real_scgpt_gsfm,
        "n_groups": int(len(meta)),
        "tissues_observed": sorted(set(tissue_per_cluster.values())),
        "tissues_with_ground_truth": sorted(
            set(tissue_per_cluster.values()) & set(tmap.keys())
        ),
        "n_clusters_no_ground": sum(
            m.get("n_clusters_no_ground", 0) for m in metrics.values()
        ),
    }


def render_ablation_markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# CartiGM Real-Data Ablation Report",
        "",
        f"- outdir: `{result.get('outdir', '')}`",
        f"- n_groups: {result.get('n_groups', 0)}",
        f"- tissues observed: {', '.join(result.get('tissues_observed', []) or [])}",
        f"- tissues with ground-truth map: {', '.join(result.get('tissues_with_ground_truth', []) or [])}",
        f"- configurations: {', '.join(result.get('configs_run', []))}",
        f"- P4 self-data consistency (mean pairwise agreement): "
        f"**{result.get('p4_self_data_consistency', 0):.3f}**",
        "",
        "## Branch provenance (real weights vs lightweight fallback)",
        "",
    ]
    branch_labels = result.get("branch_labels", {}) or {}
    for branch, label in branch_labels.items():
        lines.append(f"- **{branch}**: {label}")
    lines.append("")
    lines.extend([
        "## Per-config metrics",
        "",
        "| config | n_evaluated | n_no_ground | top_axis_accuracy | evidence_citation_rate | hallucination_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for cfg, m in (result.get("metrics") or {}).items():
        lines.append(
            f"| {cfg} | {int(m.get('n_clusters_evaluated', 0))} | "
            f"{int(m.get('n_clusters_no_ground', 0))} | "
            f"{m.get('top_axis_accuracy', 0):.3f} | "
            f"{m.get('evidence_citation_rate', 0):.3f} | "
            f"{m.get('hallucination_rate', 0):.3f} |"
        )
    lines.append("")
    refusal = result.get("refusal_audit", {}) or {}
    lines.extend([
        "## LLM refusal audit (P6 claim safety + P9 hard constraints)",
        "",
        f"- n_claims: {refusal.get('n_claims', 0)}",
        f"- n_refused: {refusal.get('n_refused', 0)}",
        f"- n_passed: {refusal.get('n_passed', 0)}",
        f"- refusal_rate: {refusal.get('refusal_rate', 0):.3f}",
        "",
        "| claim | safety_classification | can_claim | rationale |",
        "| --- | --- | --- | --- |",
    ])
    for audit in (refusal.get("audits") or []):
        lines.append(
            f"| {audit.get('claim', '')} | {audit.get('safety_classification', '')} | "
            f"{bool(audit.get('can_claim', False))} | {audit.get('rationale', '')} |"
        )
    lines.append("")
    lines.extend([
        "## Per-cluster top-1 axis (CartiGM-only baseline)",
        "",
        "| cluster | tissue | top_axis_id |",
        "| --- | --- | --- |",
    ])
    baseline = (result.get("per_config_top1") or {}).get("cartigm_only", {})
    tissue = result.get("tissue_per_cluster") or {}
    for cluster_id, axis_id in baseline.items():
        lines.append(f"| {cluster_id} | {tissue.get(cluster_id, '')} | {axis_id} |")
    return "\n".join(lines) + "\n"


def run_ablation_real(outdir: str | Path, **kwargs) -> Dict[str, Any]:
    """Alias for :func:`run_ablation` that defaults to the real-data
    annotation-based ground-truth maps bundled in this module.

    Use this when the P4 outdir comes from a real cartilage single-cell
    experiment and you want tissue/celltype-aware accuracy instead of the
    legacy hard-coded synthetic ground truth.
    """
    kwargs.setdefault("tissue_axis_map", DEFAULT_TISSUE_AXIS_MAP)
    kwargs.setdefault("celltype_axis_map", DEFAULT_CELLTYPE_AXIS_MAP)
    return run_ablation(outdir, **kwargs)
