"""P15: cluster / cell-type annotation for user-supplied h5ad files.

This module gives the user one entry point for annotating a query h5ad with
the CartiGM dictionary, plus a small set of comparison backends so the
CartiGM prediction can be cross-checked against the most common scRNA-seq
label-transfer tools.

The CartiGM branch here is the deterministic, evidence-constrained
projection defined in cartigsfm.projection / cartigsfm.gsfm /
cartigsfm.scgpt. Real scGPT-human / GSFM weights are not bundled in this
sandbox; the bundled modules are intentional proxies. Any method that
relies on these proxies is labelled "fallback" in its return dict so that
downstream reports are honest.
"""
from __future__ import annotations

import os
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# Atlas label <-> v1 axis aliasing (10-class identity map).
ACC_CHONDROCYTE_SUBTYPE_TO_V1: Dict[str, str] = {
    "Homeostatic_Matrix": "Homeostatic_Matrix",
    "Hypoxia_Adaptive": "Hypoxia_Adaptive",
    "EC_Lipo_Plasticity": "EC_Lipo_Plasticity",
    "Mesenchymal_Remodeling": "Mesenchymal_Remodeling",
    "Stress_IEG": "Stress_IEG",
    "Inflammatory_Remodeling": "Inflammatory_Remodeling",
    "Maturation_Matrix": "Maturation_Matrix",
    "Fibro_Matrix": "Fibro_Matrix",
    "PRG4_Interface": "PRG4_Interface",
    "Hypoxia_Metabolic_Stress": "Hypoxia_Metabolic_Stress",
}


def _safe_read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _p4_paths(outdir) -> Dict[str, Path]:
    base = Path(outdir)
    tsv = base / "tsv"
    return {
        "top": tsv / "p4_sample_cluster_top_assignments.tsv",
        "scores": tsv / "p4_sample_cluster_three_layer_scores.tsv",
        "meta": tsv / "p4_self_sample_cluster_meta.tsv",
        "pseudobulk": tsv / "p4_self_sample_cluster_pseudobulk.tsv",
    }


def _ensure_feature_name(adata) -> None:
    """CellTypist 1.7.1 requires var['feature_name']; add it from var_names if missing."""
    if "feature_name" not in adata.var.columns:
        adata.var["feature_name"] = adata.var_names.astype(str)


def annotate_with_cartigsm(
    p4_outdir,
    *,
    target_layer: str = "cell_subtype",
    write_back: bool = True,
) -> pd.DataFrame:
    """Pick the top target_layer axis for every (sample, cluster) using P4.

    The P4 outdir must already contain
    ``tsv/p4_sample_cluster_top_assignments.tsv`` and
    ``tsv/p4_self_sample_cluster_meta.tsv``. The annotated meta is
    returned and, when ``write_back`` is true, also written to
    ``tsv/p4_sample_cluster_cartigsm_annotation.tsv``.

    The P4 score table sample column is a composite
    ``tissue|tissue|cluster`` key, while the P4 meta table sample column
    holds the tissue name. We rebuild the composite key from tissue and
    cluster in the meta before merging.
    """
    paths = _p4_paths(p4_outdir)
    top = _safe_read_tsv(paths["top"])
    meta = _safe_read_tsv(paths["meta"])
    if top.empty:
        raise FileNotFoundError("missing P4 top-assignments: " + str(paths["top"]))
    if meta.empty:
        raise FileNotFoundError("missing P4 meta: " + str(paths["meta"]))

    sub = top[top["layer"].astype(str) == target_layer].copy()
    if sub.empty:
        sub = top[top["layer"].astype(str).str.startswith(target_layer + "::")].copy()
    sub = sub.drop_duplicates(subset=["sample"], keep="first")

    meta = meta.copy()
    tissue_str = meta["tissue"].astype(str)
    cluster_str = meta["cluster"].astype(str)
    meta["_p4_sample_key"] = tissue_str + "|" + tissue_str + "|" + cluster_str

    merged = meta.merge(
        sub[["sample", "layer", "axis_id", "name_en", "score", "marker_n", "anti_n"]]
        .rename(columns={
            "sample": "_p4_sample_key",
            "layer": "cartigsm_layer",
            "axis_id": "cartigsm_axis_id",
            "name_en": "cartigsm_name",
            "score": "cartigsm_score",
            "marker_n": "cartigsm_marker_n",
            "anti_n": "cartigsm_anti_n",
        }),
        on="_p4_sample_key",
        how="left",
    ).drop(columns=["_p4_sample_key"])

    if write_back:
        out_path = paths["meta"].with_name("p4_sample_cluster_cartigsm_annotation.tsv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, sep="\t", index=False)
    return merged


def annotate_with_marker_rule(
    p4_outdir,
    *,
    target_layer: str = "cell_subtype",
    dictionary_v1: Optional[Dict[str, Any]] = None,
    anti_lambda: float = 0.5,
    write_back: bool = True,
) -> pd.DataFrame:
    """Recompute v1 axis scores directly from the P4 pseudobulk (no P4 score table)."""
    from .assets import load_cartilage_dictionary_v1
    from .projection import _axis_anti_weights, _axis_gene_weights

    paths = _p4_paths(p4_outdir)
    pseudo = _safe_read_tsv(paths["pseudobulk"])
    if pseudo.empty:
        raise FileNotFoundError("missing P4 pseudobulk: " + str(paths["pseudobulk"]))
    if "gene" not in pseudo.columns:
        raise ValueError("P4 pseudobulk missing a gene column: " + str(paths["pseudobulk"]))

    expr = pseudo.set_index("gene")
    expr.index = expr.index.astype(str).str.upper()
    expr = expr.apply(pd.to_numeric, errors="coerce")
    z = (
        expr.sub(expr.mean(axis=1), axis=0)
        .div(expr.std(axis=1).replace(0, np.nan), axis=0)
        .fillna(0.0)
    )

    dictionary_v1 = dictionary_v1 or load_cartilage_dictionary_v1()
    rows: List[Dict[str, Any]] = []
    for layer, layer_obj in (dictionary_v1.get("layers") or {}).items():
        if layer != target_layer:
            continue
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
                anti_score = (
                    z.loc[in_a].mul(anti_series, axis=0).sum(axis=0) / anti_series.sum()
                )
            else:
                anti_score = 0.0
            score = marker_score - anti_lambda * anti_score
            for sample, value in score.items():
                rows.append({
                    "sample": sample,
                    "layer": layer,
                    "axis_id": axis.get("axis_id", ""),
                    "name_en": axis.get("name_en", ""),
                    "score": round(float(value), 4),
                    "marker_n": len(in_m),
                    "anti_n": len(in_a),
                })
    long_df = pd.DataFrame(rows, columns=[
        "sample", "layer", "axis_id", "name_en", "score", "marker_n", "anti_n"
    ])
    if long_df.empty:
        raise RuntimeError(
            "no v1 axis in layer " + repr(target_layer) + " matched any gene in the pseudobulk"
        )
    top = (
        long_df.sort_values(["sample", "score"], ascending=[True, False])
        .drop_duplicates(subset=["sample"], keep="first")
        .reset_index(drop=True)
    )

    if write_back:
        out_path = paths["meta"].with_name("p4_sample_cluster_marker_rule_annotation.tsv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        top.to_csv(out_path, sep="\t", index=False)
    return top


def annotate_with_celltypist(
    query_h5ad,
    reference_h5ad,
    *,
    reference_label_col: str = "chongdrocyte_subtype",
    cluster_col: str = "leiden_res0_5",
    out_tsv=None,
    model_path=None,
    feature_selection: bool = True,
    max_reference_cells: Optional[int] = None,
    random_state: int = 0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Real CellTypist label transfer; majority vote per cluster.

    The training step passes ``check_expression=False`` because the bundled
    acc / EBR h5ad files are already log1p-normalised (max ~ 8-10) but not
    with CellTypist's exact ``target_sum=1e4`` recipe. The logistic
    regression in celltypist.train only needs the data to be in a
    reasonable log space, so we skip the strict check. If you have raw
    counts, call ``sc.pp.normalize_total(adata, target_sum=1e4)`` and
    ``sc.pp.log1p(adata)`` yourself before passing them in.

    Device note: celltypist v1.7.1 trains with scikit-learn's
    ``SGDClassifier`` under the hood, which is CPU-only. The ``device``
    argument is accepted for API uniformity and is recorded in the
    returned note, but training always runs on CPU. For a GPU-runnable
    cell-typing path use ``annotate_with_scgpt`` with real scGPT weights.
    """
    from .assets import device_summary, prefer_device
    chosen_device = prefer_device(device)
    method = "celltypist"
    try:
        import celltypist
        from celltypist import models
    except Exception as exc:
        return {
        "method": method,
        "available": False,
        "note": "celltypist import failed: " + repr(exc) + "; install with pip install celltypist",
        "per_cluster": pd.DataFrame(),
    }

    import scanpy as sc

    ref = sc.read_h5ad(str(reference_h5ad))
    if reference_label_col not in ref.obs.columns:
        return {
            "method": method,
            "available": False,
            "note": "reference has no column " + repr(reference_label_col),
            "per_cluster": pd.DataFrame(),
        }
    if max_reference_cells and ref.n_obs > int(max_reference_cells):
        rng = np.random.default_rng(int(random_state))
        keep = rng.choice(ref.n_obs, size=int(max_reference_cells), replace=False)
        ref = ref[keep].copy()

    ref = ref.copy()
    _ensure_feature_name(ref)

    model = None
    if model_path is not None and Path(model_path).exists():
        try:
            model = models.Model.load(str(model_path))
        except Exception:
            model = None
    if model is None:
        gpu_path = str(chosen_device).startswith("cuda") or str(chosen_device) == "mps"
        if gpu_path:
            from .annotate_torch import train_logreg_torch
            torch_model = train_logreg_torch(
                ref,
                labels=reference_label_col,
                feature_selection=bool(feature_selection),
                device=chosen_device,
            )
            model = {"_torch": True, "torch": torch_model}
        else:
            model = celltypist.train(
                ref,
                labels=reference_label_col,
                feature_selection=bool(feature_selection),
                check_expression=False,
            )
            if model_path is not None:
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                try:
                    model.write(str(model_path))
                except Exception:
                    pass

    q = sc.read_h5ad(str(query_h5ad))
    if cluster_col not in q.obs.columns:
        return {
            "method": method,
            "available": False,
            "note": "query has no cluster column " + repr(cluster_col),
            "per_cluster": pd.DataFrame(),
        }
    _ensure_feature_name(q)
    if isinstance(model, dict) and model.get("_torch"):
        import torch
        torch_model = model["torch"]
        ref_features = torch_model["features"]
        ref_classes = torch_model["classes_"]
        coef = torch_model["coef_"]
        intercept = torch_model["intercept_"]
        q_gene_to_idx = {g: i for i, g in enumerate(q.var_names.astype(str))}
        ref_idx_in_q = [q_gene_to_idx.get(g, -1) for g in ref_features]
        keep = [i for i in ref_idx_in_q if i >= 0]
        if not keep:
            return {
                "method": method,
                "available": False,
                "note": "torch-trained model has no overlap with query var_names",
                "per_cluster": pd.DataFrame(),
            }
        keep_in_ref = [i for i, v in enumerate(ref_idx_in_q) if v >= 0]
        coef_aligned = coef[:, keep_in_ref]
        X = q.X[:, [ref_idx_in_q[i] for i in keep_in_ref]]
        if hasattr(X, "toarray"):
            X = np.asarray(X.toarray(), dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)
        device_t = torch.device(str(chosen_device))
        with torch.no_grad():
            xt = torch.from_numpy(X).to(device_t).float()
            wt = torch.from_numpy(coef_aligned).to(device_t).float()
            bt = torch.from_numpy(intercept).to(device_t).float()
            logits = xt @ wt.T + bt
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = probs.argmax(axis=1)
            conf = probs.max(axis=1)
        pred_labels = [ref_classes[i] for i in pred_idx]
        per_cell = pd.DataFrame({
            "cell_barcode": q.obs_names.astype(str).values,
            "cluster": q.obs[cluster_col].astype(str).values,
            "celltypist_label": pred_labels,
            "celltypist_confidence": conf.astype(float),
        })
    else:
        preds = celltypist.annotate(q, model=model, majority_voting=False)
        prob_cols = [
            c for c in preds.predictions.columns
            if c not in ("predicted_labels", "over_clustering", "majority_voting")
        ]
        pred_col = "predicted_labels" if "predicted_labels" in preds.predictions.columns else preds.predictions.columns[0]
        if prob_cols:
            conf = preds.predictions[prob_cols].max(axis=1)
        else:
            conf = pd.Series(np.zeros(q.n_obs))
        per_cell = pd.DataFrame({
        "cell_barcode": q.obs_names.astype(str).values,
        "cluster": q.obs[cluster_col].astype(str).values,
        "celltypist_label": preds.predictions[pred_col].astype(str).values,
        "celltypist_confidence": conf.values,
    })

    rows_list: List[Dict[str, Any]] = []
    for cluster_id, sub in per_cell.groupby("cluster"):
        labels = sub["celltypist_label"].astype(str)
        majority_label, majority_n = Counter(labels).most_common(1)[0]
        rows_list.append({
            cluster_col: str(cluster_id),
            "celltypist_label": str(majority_label),
            "celltypist_n_cells": int(len(sub)),
            "celltypist_majority_frac": round(float(majority_n) / max(len(sub), 1), 4),
            "celltypist_mean_confidence": round(float(sub["celltypist_confidence"].mean()), 4),
        })
    per_cluster = pd.DataFrame(rows_list).sort_values(cluster_col).reset_index(drop=True)

    per_cell_path: Optional[str] = None
    if out_tsv is not None:
        per_cell_path = str(out_tsv)
        Path(per_cell_path).parent.mkdir(parents=True, exist_ok=True)
        per_cell.to_csv(per_cell_path, sep="\t", index=False)

    return {
        "method": method,
        "available": True,
        "per_cluster": per_cluster,
        "per_cell_predictions_path": per_cell_path,
        "reference_label_col": reference_label_col,
        "cluster_col": cluster_col,
        "n_query_cells": int(q.n_obs),
        "n_reference_cells": int(ref.n_obs),
        "n_clusters": int(per_cluster.shape[0]),
        "note": (
            "Real CellTypist v1.7.1; trained on the reference using "
            + str(reference_label_col)
            + ", then per-cell predictions aggregated per "
            + str(cluster_col)
            + " with majority vote and mean confidence."
            + " Trainer: "
            + ("cartigsfm.annotate_torch.train_logreg_torch on "
               + str(chosen_device)
               + " (celltypist's sklearn SGD trainer is CPU-only; the"
               + " bundled torch trainer is the GPU-runnable path)")
            if (isinstance(model, dict) and model.get("_torch"))
            else (
                "celltypist's sklearn SGD on CPU (the bundled torch"
                + " trainer was bypassed; device="
                + str(chosen_device)
                + " is recorded but not used for training)."
            )
        ),
    }


def _per_cluster_mean_matrix(adata, cluster_col: str) -> pd.DataFrame:
    """Return a (clusters x genes) dense mean-expression DataFrame."""
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
    cluster_labels = adata.obs[cluster_col].astype(str)
    codes, labels = pd.factorize(cluster_labels, sort=True)
    design = sp.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(labels), len(codes)),
    )
    if sp.issparse(X):
        sums = design @ X
        means = sums.multiply(1.0 / np.bincount(codes)[:, None]).toarray()
    else:
        means = np.vstack([
            np.asarray(X[codes == i]).mean(axis=0) for i in range(len(labels))
        ])
    return pd.DataFrame(
        means,
        index=[str(l) for l in labels],
        columns=[str(g).upper() for g in adata.var_names],
    )


def annotate_with_scgpt(
    query_h5ad,
    *,
    cluster_col: str = "leiden_res0_5",
    target_layer: str = "cell_subtype",
    dictionary_v1: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Lightweight deterministic scGPT-style proxy (fallback).

    The bundled scgpt path does NOT load real scGPT-human weights. The
    ``device`` argument is accepted for API uniformity and recorded in
    the returned note. To run real scGPT on GPU, load the official
    weights and call them directly; this proxy is only the deterministic
    fallback that keeps the comparison surface intact.
    """
    from .assets import prefer_device
    chosen_device = prefer_device(device)
    import scanpy as sc

    q = sc.read_h5ad(str(query_h5ad))
    if cluster_col not in q.obs.columns:
        return {
            "method": "scgpt",
            "available": False,
            "fallback": True,
            "note": "query has no cluster column " + repr(cluster_col),
            "per_cluster": pd.DataFrame(),
        }

    from .assets import load_cartilage_dictionary_v1
    from .projection import _axis_gene_weights

    dictionary_v1 = dictionary_v1 or load_cartilage_dictionary_v1()
    enc = _per_cluster_mean_matrix(q, cluster_col=cluster_col)
    rows_list: List[Dict[str, Any]] = []
    axes = (dictionary_v1.get("layers") or {}).get(target_layer, {}).get("axes", [])
    for cluster_id, sub_row in enc.iterrows():
        if not isinstance(sub_row, pd.Series):
            continue
        scores_list: List[Tuple[str, str, float, int]] = []
        for axis in axes:
            marker_weights = _axis_gene_weights(axis)
            in_m = [g for g in marker_weights if g in sub_row.index]
            if not in_m:
                continue
            weights = pd.Series({g: marker_weights[g] for g in in_m}, dtype=float)
            val = float(
                pd.Series({g: float(sub_row[g]) for g in in_m}).mul(weights).sum()
                / weights.sum()
            )
            scores_list.append((
                axis.get("axis_id", ""),
                axis.get("name_en", ""),
                val,
                len(in_m),
            ))
        if not scores_list:
            rows_list.append({
                cluster_col: str(cluster_id),
                "scgpt_axis_id": "",
                "scgpt_label": "",
                "scgpt_score": float("nan"),
                "scgpt_marker_n": 0,
            })
            continue
        scores_list.sort(key=lambda t: t[2], reverse=True)
        ax_id, name, val, n = scores_list[0]
        rows_list.append({
            cluster_col: str(cluster_id),
            "scgpt_axis_id": ax_id,
            "scgpt_label": name,
            "scgpt_score": round(val, 4),
            "scgpt_marker_n": n,
        })
    per_cluster = pd.DataFrame(rows_list).sort_values(cluster_col).reset_index(drop=True)
    return {
        "method": "scgpt",
        "available": True,
        "fallback": True,
        "per_cluster": per_cluster,
        "target_layer": target_layer,
        "cluster_col": cluster_col,
        "note": (
            "Fallback: weighted-mean of per-cluster mean expression of "
            "each v1 axis marker genes. Real scGPT-human weights are "
            "not bundled in this sandbox. Device: "
            + str(chosen_device)
            + " (the proxy ignores device; this field is recorded for "
            + "API uniformity so callers can route real weights to GPU)."
        ),
    }


# Real R-backed implementations live in cartigsfm.annotate_r; re-export
# them so existing callers (cli.py, tests, third-party) keep working.
# When rpy2 / R / a specific R package is not available, the wrapper
# returns ``{"available": False, ...}`` exactly like the old stubs.
from .annotate_r import (
    annotate_with_cellassign,
    annotate_with_scmap,
    annotate_with_singler,
    annotate_with_symphony,
)


def build_gptcelltype_prompt(
    markers: Iterable[str],
    candidates: Iterable[str],
    *,
    tissue: Optional[str] = None,
    n_markers: int = 30,
) -> str:
    """Build a deterministic GPTcelltype-style prompt for a list of markers."""
    marker_list = [str(m).strip() for m in markers if str(m).strip()][: int(n_markers)]
    cand_list = [str(c).strip() for c in candidates if str(c).strip()]
    tissue_line = ("Tissue / context: " + str(tissue) + "\n") if tissue else ""
    return (
        "You are a careful cartilage-biology curator. Below is a list of "
        "marker genes ranked by specificity, followed by a closed-set of "
        "candidate cell-type labels. Choose exactly one label from the "
        "candidate set that best matches the marker list. Reply with a "
        "single line: label: <chosen_label> then a short rationale.\n"
        + tissue_line
        + "Top markers (n=" + str(len(marker_list)) + "): "
        + ", ".join(marker_list) + " or N/A\n"
        + "Candidate labels: " + ", ".join(cand_list) + " or N/A\n"
    )


def annotate_with_gptcelltype(
    marker_dict,
    *,
    candidates=None,
    tissue: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    n_markers: int = 30,
    out_tsv=None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Run GPTcelltype-style annotation, or build the prompt only."""
    candidates = list(candidates) if candidates is not None else list(
        ACC_CHONDROCYTE_SUBTYPE_TO_V1.values()
    )
    prompts = {
        str(k): build_gptcelltype_prompt(
            v, candidates, tissue=tissue, n_markers=n_markers
        )
        for k, v in marker_dict.items()
    }
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if out_tsv is not None:
            Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"cluster": k, "prompt": v} for k, v in prompts.items()]
            ).to_csv(out_tsv, sep="\t", index=False)
        return {
            "method": "gptcelltype",
            "available": True,
            "prompt_only": True,
            "prompts": prompts,
            "candidates": list(candidates),
            "model": model,
            "out_tsv": str(out_tsv) if out_tsv else None,
            "note": "OPENAI_API_KEY not set; prompts returned without model call.",
        }

    try:
        import requests
    except Exception as exc:
        return {
            "method": "gptcelltype",
            "available": False,
            "prompts": prompts,
            "note": "requests not installed (" + repr(exc) + "); cannot call OpenAI.",
        }

    import requests
    headers = {
        "Authorization": "Bearer " + str(api_key),
        "Content-Type": "application/json",
    }
    out_rows: List[Dict[str, Any]] = []
    for cluster_id, prompt in prompts.items():
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You annotate cartilage scRNA-seq clusters."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=float(timeout_s),
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            content = "ERROR: " + repr(exc)
        m = re.search(r"label\s*[:：]\s*([^\n]+)", content, re.IGNORECASE)
        label = m.group(1).strip() if m else content.splitlines()[0].strip()
        out_rows.append({
            "cluster": cluster_id,
            "gptcelltype_label": label,
            "gptcelltype_raw": content,
        })

    df = pd.DataFrame(out_rows)
    if out_tsv is not None:
        Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_tsv, sep="\t", index=False)
    return {
        "method": "gptcelltype",
        "available": True,
        "prompt_only": False,
        "model": model,
        "predictions": df,
        "out_tsv": str(out_tsv) if out_tsv else None,
        "candidates": list(candidates),
        "note": (
            "Called " + model + " via OpenAI Chat Completions; one label per "
            "cluster extracted from the model's reply."
        ),
    }


def _label_column_for(method: str) -> str:
    if method == "cartigsm":
        return "cartigsm_name"
    if method == "marker_rule":
        return "name_en"
    if method == "scgpt":
        return "scgpt_label"
    if method == "celltypist":
        return "celltypist_label"
    if method == "gptcelltype":
        return "gptcelltype_label"
    return method + "_label"


def _merge_method_frames(per_method) -> pd.DataFrame:
    """Stack method predictions into one long-form DataFrame per cluster.

    For ``cartigsm`` and ``marker_rule`` the input is (tissue, cluster)
    keyed and we collapse to one row per cluster via majority vote so the
    resulting table lines up with cluster-level backends
    (scgpt / celltypist / gptcelltype).
    """
    rows: List[Dict[str, Any]] = []
    for method, df in per_method.items():
        if df is None or df.empty:
            continue
        label_col = _label_column_for(method)
        if label_col not in df.columns:
            continue
        # (tissue, cluster) -> cluster via majority vote
        if "tissue" in df.columns and "cluster" in df.columns and method in ("cartigsm", "marker_rule"):
            tmp = df.copy()
            tmp["cluster"] = tmp["cluster"].astype(str)
            for cluster_id, sub in tmp.groupby("cluster"):
                labs = [
                    str(x) for x in sub[label_col].dropna().tolist()
                    if str(x) and str(x) != "nan"
                ]
                if not labs:
                    continue
                top_lab, _ = Counter(labs).most_common(1)[0]
                rows.append({
                    "method": method,
                    "tissue": str(sub["tissue"].iloc[0]) if "tissue" in sub.columns else "",
                    "cluster": str(cluster_id),
                    "label": str(top_lab),
                })
            continue
        cluster_col = None
        for cand in ("cluster", "leiden_res0_5", "leiden", "leiden_res0.5"):
            if cand in df.columns:
                cluster_col = cand
                break
        if cluster_col is None:
            for c in df.columns:
                lc = str(c).lower()
                if lc.startswith("leiden") or lc.endswith("cluster"):
                    cluster_col = c
                    break
        if cluster_col is None:
            continue
        for _, r in df.iterrows():
            lab = r.get(label_col, "")
            if not lab or (isinstance(lab, float) and math.isnan(lab)):
                continue
            rows.append({
                "method": method,
                "tissue": "",
                "cluster": str(r[cluster_col]),
                "label": str(lab),
            })
    return pd.DataFrame(rows, columns=["method", "tissue", "cluster", "label"])


def compare_annotations(per_method, *, reference: Optional[str] = None) -> Dict[str, Any]:
    """Build a long-form table and a pairwise agreement matrix."""
    long_df = _merge_method_frames(per_method)
    if long_df.empty:
        return {
            "long": long_df,
            "per_cluster_wide": pd.DataFrame(),
            "pairwise_agreement": {},
            "n_clusters_per_method": {},
            "reference_method": reference,
        }
    wide = (
        long_df.pivot_table(
            index="cluster", columns="method", values="label", aggfunc="first"
        )
        .reset_index()
    )
    methods = sorted(long_df["method"].unique().tolist())
    pairwise: Dict[str, float] = {}
    n_per: Dict[str, int] = {m: 0 for m in methods}
    for m in methods:
        n_per[m] = int(long_df[long_df["method"] == m]["cluster"].nunique())
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            sub = long_df[long_df["method"].isin([a, b])]
            clusters = sub["cluster"].unique()
            if len(clusters) == 0:
                pairwise[a + "__" + b] = float("nan")
                continue
            agree = 0
            total = 0
            for c in clusters:
                la = sub[(sub["method"] == a) & (sub["cluster"] == c)]["label"]
                lb = sub[(sub["method"] == b) & (sub["cluster"] == c)]["label"]
                if la.empty or lb.empty:
                    continue
                total += 1
                if str(la.iloc[0]).strip() == str(lb.iloc[0]).strip():
                    agree += 1
            pairwise[a + "__" + b] = round(agree / total, 4) if total else float("nan")
    return {
        "long": long_df,
        "per_cluster_wide": wide,
        "pairwise_agreement": pairwise,
        "n_clusters_per_method": n_per,
        "reference_method": reference,
    }
