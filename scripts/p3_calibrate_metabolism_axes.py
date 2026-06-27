"""P3 atlas calibration for the 10 metabolism axes added in v1.7.

For every axis we

1. read the panel + prior weights from the dictionary,
2. score each EBR cell on the ``log1p_norm`` layer using the literature-prior
   weights,
3. select the top ``--top-frac`` (default 5%) of cells as the in-context set,
4. compute ``freq_in`` (fraction of in-context cells with non-zero expression
   per gene), ``freq_bg`` (same fraction on all cells), and
   ``log2_spec = log2((freq_in + eps) / (freq_bg + eps))``,
5. derive a calibrated ``weight = max(0, log2_spec) * freq_in`` rescaled so the
   top gene matches the bundled v1.0 convention (1.5).

The script then bumps the dictionary v1.7 -> v1.8 with the calibrated
``core_genes`` and promotes the calibrated axes to ``status="production"`` so
:func:`cartigsfm.interpret.axis_safety_class` returns
``PENDING_INDEPENDENT_VALIDATION`` (matches every other production axis in v1.7).

RAG KB cards and axis evidence cards are refreshed in lockstep.

GPU is used for the cell-by-panel scoring (torch sparse on cuda when
available); the per-gene non-zero counts run on CPU because they are a
single sparse-column reduction and scipy is already vectorized.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _metabolism_axis_panels as MP  # noqa: E402


DICT_PATH = ROOT / "cartigsfm" / "resources" / "dictionary_v1" / "cartilage_dictionary_v1.json"
RAG_PATH = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_cartigsfm_knowledge_base.json"
CARDS_PATH = ROOT / "cartigsfm" / "resources" / "rag_v1" / "p6_axis_evidence_cards.json"
REPORT_PATH = ROOT / "cartigsfm" / "resources" / "dictionary_v1" / "p3_calibration_metabolism_v1_7.json"

OLD_VERSION = "1.7"
NEW_VERSION = "1.8"
EPS = 0.01
TOP_TARGET_WEIGHT = 1.5

METAB_AXIS_IDS = [
    "functional_axis::Glycolysis",
    "functional_axis::OxidativePhosphorylation",
    "functional_axis::TCA_Cycle",
    "functional_axis::PentosePhosphatePathway",
    "functional_axis::FattyAcidOxidation",
    "functional_axis::Lipogenesis",
    "functional_axis::CholesterolHomeostasis",
    "functional_axis::LipidDroplet",
    "functional_axis::Glutaminolysis",
    "functional_axis::MitochondrialBiogenesis",
]


def _load_log1p(adata):
    if "log1p_norm" in adata.layers:
        X = adata.layers["log1p_norm"]
    else:
        X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    return X.tocsr().astype(np.float32)


def _gene_index(var_names):
    return {str(g).upper(): i for i, g in enumerate(var_names)}


def _score_cells_gpu(X_csr, gene_idx, weight_vec, device):
    """Return (n_cells,) float32 score = X_csr[:, idx] @ weight_vec, on device."""
    import torch
    n = X_csr.shape[0]
    Xc = X_csr[:, gene_idx]
    Xt = torch.sparse_csr_tensor(
        torch.from_numpy(Xc.indptr.astype(np.int64)),
        torch.from_numpy(Xc.indices.astype(np.int64)),
        torch.from_numpy(Xc.data.astype(np.float32)),
        size=Xc.shape,
    ).to(device)
    w = torch.from_numpy(weight_vec.astype(np.float32)).to(device)
    s = (Xt @ w).cpu().numpy()
    del Xt, w
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return s.astype(np.float32)


def _nonzero_freq(X_csr, gene_idx, mask=None):
    """Per-gene fraction of non-zero entries, optionally restricted to mask."""
    if mask is None:
        sub = X_csr[:, gene_idx]
    else:
        sub = X_csr[mask][:, gene_idx]
    n = sub.shape[0]
    if n == 0:
        return np.zeros(len(gene_idx), dtype=np.float64)
    nz = (sub != 0).sum(axis=0)
    return np.asarray(nz, dtype=np.float64).ravel() / float(n)


def _calibrate_axis(axis_dict, X_csr, gene_to_idx, top_frac, device):
    panel = [str(g).upper() for g in axis_dict.get("panel_genes") or []]
    panel_idx = [(g, gene_to_idx[g]) for g in panel if g in gene_to_idx]
    if not panel_idx:
        return None
    panel_genes = [g for g, _ in panel_idx]
    panel_cols = np.array([i for _, i in panel_idx], dtype=np.int64)

    # prior weights from the dictionary (panel weight 1.0, core gene > 1.0)
    mw = axis_dict.get("marker_weights") or {}
    prior_weights = np.array(
        [float(mw.get(g, 1.0)) for g in panel_genes], dtype=np.float32
    )

    # GPU scoring on the prior-weighted panel
    scores = _score_cells_gpu(X_csr, panel_cols, prior_weights, device)
    n_cells = scores.shape[0]
    k = max(1, int(round(top_frac * n_cells)))
    # top-k cell mask (descending by score)
    top_idx = np.argpartition(-scores, k - 1)[:k]
    mask = np.zeros(n_cells, dtype=bool)
    mask[top_idx] = True

    freq_in = _nonzero_freq(X_csr, panel_cols, mask=mask)
    freq_bg = _nonzero_freq(X_csr, panel_cols, mask=None)
    log2_spec = np.log2((freq_in + EPS) / (freq_bg + EPS))
    raw_weight = np.clip(log2_spec, 0.0, None) * freq_in
    if raw_weight.max() > 0:
        scale = TOP_TARGET_WEIGHT / raw_weight.max()
    else:
        scale = 0.0
    weight = raw_weight * scale

    # rank by weight desc, keep top 30 as core_genes
    order = np.argsort(-weight, kind="stable")
    top = order[:30]
    new_core = []
    for j in top:
        new_core.append({
            "gene": panel_genes[j],
            "freq_in": float(round(freq_in[j], 4)),
            "freq_bg": float(round(freq_bg[j], 4)),
            "log2_spec": float(round(log2_spec[j], 3)),
            "weight": float(round(weight[j], 3)),
        })

    # refresh marker_weights: panel base 1.0, core gene calibrated weight
    new_marker_weights = {g: 1.0 for g in panel_genes}
    for entry in new_core:
        new_marker_weights[entry["gene"]] = max(
            new_marker_weights.get(entry["gene"], 1.0), float(entry["weight"])
        )

    summary = {
        "n_cells_total": int(n_cells),
        "n_cells_in_context": int(k),
        "top_frac": float(top_frac),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_max": float(scores.max()),
        "core_top_gene": new_core[0]["gene"] if new_core else None,
        "core_top_weight": new_core[0]["weight"] if new_core else None,
    }
    return new_core, new_marker_weights, summary


def _backup(path, suffix):
    bak = path.with_name(path.name + suffix)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup -> {bak}")


def _refresh_kb_card(card, axis):
    card["core_genes"] = axis["core_genes"]
    card["panel_genes"] = axis["panel_genes"]
    card["anti_genes"] = axis["anti_genes"]
    card["interpretation"] = axis["interpretation"]
    card["limitations"] = axis["limitations"]
    card["recommended_use"] = axis["recommended_use"]
    card["evidence_level"] = "ATLAS_INTERNAL"
    return card


def _refresh_evidence_card(card, axis):
    card["core_supporting_genes"] = axis["core_genes"]
    card["anti_genes"] = axis["anti_genes"]
    card["expected_biological_contexts"] = axis["recommended_use"]
    card["known_limitations"] = axis["limitations"]
    card["confidence_level"] = "MEDIUM"
    card["p5_robustness_status"] = {
        "mean_score_correlation": None,
        "mean_top_label_agreement": None,
        "robustness_note": "Atlas-internal calibration on EBR.h5ad (P3 v1.7 -> v1.8 metabolism pass).",
    }
    card["key_gene_evidence"] = [
        {"gene": g["gene"], "weight": g["weight"], "source": "atlas_calibrated"}
        for g in axis["core_genes"][:8]
    ]
    return card


def _device(arg):
    if arg == "cpu":
        import torch
        return torch.device("cpu")
    import torch
    if torch.cuda.is_available() and arg in {"auto", "cuda"}:
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", default="F:/cartifm/outputs/EBR/EBR.h5ad")
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dry-run", action="store_true",
                   help="compute calibration but do not write the dictionary")
    args = p.parse_args()

    print(f"Reading dictionary {DICT_PATH}")
    d = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    if d.get("version") != OLD_VERSION:
        print(f"WARNING: dictionary version {d.get('version')!r}, expected {OLD_VERSION!r}")

    fa_axes = d["layers"]["functional_axis"]["axes"]
    by_id = {a["axis_id"]: a for a in fa_axes}
    targets = [by_id[aid] for aid in METAB_AXIS_IDS if aid in by_id]
    if len(targets) != len(METAB_AXIS_IDS):
        missing = set(METAB_AXIS_IDS) - set(by_id)
        raise SystemExit(f"missing axes: {missing}")

    import anndata as ad
    print(f"Loading {args.h5ad} (this may take a minute)")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  shape={adata.shape}")
    X_csr = _load_log1p(adata)
    gene_to_idx = _gene_index(adata.var_names)

    device = _device(args.device)
    print(f"  scoring device: {device}")

    summaries = {}
    calibrations = {}
    for axis in targets:
        print(f"--- {axis['axis_id']} ---")
        out = _calibrate_axis(axis, X_csr, gene_to_idx, args.top_frac, device)
        if out is None:
            print("  no panel overlap; skipping")
            continue
        new_core, new_mw, summary = out
        calibrations[axis["axis_id"]] = (new_core, new_mw)
        summaries[axis["axis_id"]] = summary
        print(f"  top: {summary['core_top_gene']} (w={summary['core_top_weight']})  "
              f"score p95={summary['score_p95']:.2f}  max={summary['score_max']:.2f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "generated_at": date.today().isoformat(),
        "atlas": str(args.h5ad),
        "top_frac": args.top_frac,
        "n_cells": int(X_csr.shape[0]),
        "summaries": summaries,
        "calibrations": {k: {
            "core_genes": v[0],
        } for k, v in calibrations.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report -> {REPORT_PATH}")

    if args.dry_run:
        print("--dry-run: not modifying dictionary or RAG resources")
        return 0

    _backup(DICT_PATH, ".pre_v1_8.bak")
    today = date.today().isoformat()
    new_axes_summary = []
    for axis in targets:
        if axis["axis_id"] not in calibrations:
            continue
        new_core, new_mw = calibrations[axis["axis_id"]]
        axis["core_genes"] = new_core
        axis["marker_weights"] = new_mw
        axis["status"] = "production"
        axis["evidence"]["derivation"] = [
            f"Atlas calibration on EBR.h5ad ({X_csr.shape[0]} cells, "
            f"top-{int(args.top_frac*100)}% in-context selection on prior-weighted panel score, "
            "log1p_norm layer)."
        ]
        axis["evidence"]["internal_support"] = [
            f"Top in-context core gene: {new_core[0]['gene']} "
            f"(freq_in={new_core[0]['freq_in']}, freq_bg={new_core[0]['freq_bg']}, "
            f"log2_spec={new_core[0]['log2_spec']}, weight={new_core[0]['weight']})."
        ]
        axis["evidence"]["independent_validation"] = []
        axis["limitations"] = [
            "Atlas-internal calibration on EBR.h5ad only; pending independent validation on a held-out atlas (axis_safety_class=PENDING_INDEPENDENT_VALIDATION).",
            "EBR is enriched for ear/rib/nose chondrocytes; absolute frequencies are biased toward this lineage mix.",
        ]
        axis.setdefault("source_files", []).extend([
            "scripts/p3_calibrate_metabolism_axes.py",
            f"cartigsfm/resources/dictionary_v1/{REPORT_PATH.name}",
        ])
        axis["source_files"] = sorted(set(axis["source_files"]))
        new_axes_summary.append((axis["name_en"], new_core[0]["gene"], new_core[0]["weight"]))

    d["version"] = NEW_VERSION
    d["generated_at"] = today
    d.setdefault("changelog", []).append({
        "version": NEW_VERSION,
        "generated_at": today,
        "changes": [
            "P3 atlas calibration of the 10 metabolism functional axes added in v1.7.",
            f"Top-{int(args.top_frac*100)}% in-context selection on EBR.h5ad ({X_csr.shape[0]} cells) using prior-weighted panel scoring.",
            "core_genes refreshed with atlas-derived freq_in / freq_bg / log2_spec / weight; status promoted to production.",
            "Per-axis top core gene: " + ", ".join(f"{n}:{g}({w})" for n, g, w in new_axes_summary),
        ],
        "scripts": ["scripts/p3_calibrate_metabolism_axes.py"],
    })
    DICT_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dictionary v{NEW_VERSION} -> {DICT_PATH}")

    if RAG_PATH.exists():
        kb = json.loads(RAG_PATH.read_text(encoding="utf-8"))
        dk = kb.get("dictionary_knowledge")
        if isinstance(dk, list):
            by_kb = {it.get("axis_id"): it for it in dk if isinstance(it, dict)}
            for axis in targets:
                if axis["axis_id"] in by_kb:
                    _refresh_kb_card(by_kb[axis["axis_id"]], axis)
            kb["generated_at"] = today
            _backup(RAG_PATH, ".pre_v1_8.bak")
            RAG_PATH.write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Refreshed RAG KB -> {RAG_PATH}")

    if CARDS_PATH.exists():
        cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
        if isinstance(cards, list):
            by_card = {it.get("axis_id"): it for it in cards if isinstance(it, dict)}
            for axis in targets:
                if axis["axis_id"] in by_card:
                    _refresh_evidence_card(by_card[axis["axis_id"]], axis)
            _backup(CARDS_PATH, ".pre_v1_8.bak")
            CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Refreshed evidence cards -> {CARDS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
