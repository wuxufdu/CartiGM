"""End-to-end runner for the P17 fusion ablation.

Usage:
    F:\\cartifm\\CartiGM\\.venv\\Scripts\\python.exe -m cartigsfm.run_fusion

Reads acc.h5ad (subsample N cells stratified by sample), reads EBR.h5ad,
extracts per-cell CartiGM / scGPT-proxy / GSFM features, trains 6 fusion
heads with sample-stratified val, runs external validation on EBR, and
writes:

  F:\\cartifm\\outputs\\fusion_P17\\
      acc_subsample.h5ad            (subsample, label + sample intact)
      acc_features.npz              (cartigm / scgpt / gsfm + labels)
      ebr_features.npz              (same for EBR)
      ablation_metrics.tsv          (6 configs x 6 metrics on acc val)
      ebr_metrics.tsv               (6 configs x 5 metrics on EBR)
      ebr_per_cluster_predictions.tsv
      fusion_summary.json

  F:\\cartifm\\CartiGM\\reports\\P17_FUSION_ABLATION.md
"""
from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import fusion as F
from .assets import prefer_device


OUTDIR = Path(r"F:\cartifm\outputs\fusion_P17")
REPORT = Path(r"F:\cartifm\CartiGM\reports\P17_FUSION_ABLATION.md")
ACC_H5AD = r"F:\cartifm\acc.h5ad"
EBR_H5AD = r"F:\cartifm\outputs\EBR\EBR.h5ad"


def stratified_subsample_indices(samples: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    unique = sorted(set(samples.tolist()))
    per = max(1, int(n_target // max(1, len(unique))))
    chosen = []
    for s in unique:
        idx = np.where(samples == s)[0]
        if idx.size == 0:
            continue
        k = min(per, idx.size)
        if k >= idx.size:
            chosen.append(idx)
        else:
            sel = rng.choice(idx, size=k, replace=False)
            chosen.append(sel)
    out = np.sort(np.concatenate(chosen)).astype(np.int64)
    if out.size > n_target:
        out = np.sort(rng.choice(out, size=n_target, replace=False)).astype(np.int64)
    return out


def subsample_acc(n_cells: int, label_col: str, sample_col: str, seed: int):
    print(f"[run_fusion] opening acc.h5ad with backed='r' ...", flush=True)
    a = ad.read_h5ad(ACC_H5AD, backed="r")
    samples = a.obs[sample_col].astype(str).values
    labels_full = a.obs[label_col].astype(str).values
    print(f"[run_fusion] acc shape {a.shape}, n_samples {len(set(samples.tolist()))}, n_labels {len(set(labels_full.tolist()))}", flush=True)
    print(f"[run_fusion] label counts: {Counter(labels_full.tolist()).most_common(15)}", flush=True)
    keep = stratified_subsample_indices(samples, n_cells, seed)
    print(f"[run_fusion] subsampled to {keep.size} cells", flush=True)
    a_sub = a[keep].to_memory()
    a_sub.obs[label_col] = labels_full[keep]
    a_sub.obs[sample_col] = samples[keep]
    return a_sub, keep, samples[keep], labels_full[keep]


def load_ebr(cluster_col: str):
    print(f"[run_fusion] opening {EBR_H5AD} ...", flush=True)
    a = ad.read_h5ad(EBR_H5AD)
    print(f"[run_fusion] EBR shape {a.shape}", flush=True)
    if cluster_col not in a.obs.columns:
        cluster_col = [c for c in a.obs.columns if "leiden" in c.lower()][0]
    cluster = a.obs[cluster_col].astype(str).values
    return a, cluster


def write_report(
    acc_df: pd.DataFrame,
    ebr_df: pd.DataFrame,
    ebr_per_cluster: pd.DataFrame,
    train_samples: list,
    val_samples: list,
    n_acc: int,
    n_ebr: int,
    class_names: list,
    n_classes: int,
    elapsed: float,
) -> str:
    lines = []
    lines.append("# P17 - CartiGM + scGPT-proxy + GSFM Fusion Ablation")
    lines.append("")
    lines.append(f"Reference atlas: acc.h5ad ({n_acc} cells, sample-stratified 80/20 split)")
    lines.append(f"External validation: EBR.h5ad ({n_ebr} cells)")
    lines.append(f"Train samples: {len(train_samples)}; Val samples: {len(val_samples)}")
    lines.append(f"Classes: {n_classes} ({', '.join(class_names)})")
    lines.append(f"Elapsed: {elapsed:.1f}s")
    lines.append("")
    lines.append("## 1. On the acc sample-stratified held-out split")
    lines.append("")
    lines.append("| config | features | in_dim | accuracy | macro_f1 | balanced_acc | top_axis_consistency | evidence_citation | hallucination |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in acc_df.iterrows():
        lines.append(
            "| {config} | {features} | {in_dim} | {accuracy} | {macro_f1} | {balanced_accuracy} | {top_axis_consistency} | {evidence_citation_rate} | {hallucination_rate} |".format(
                config=row["config"],
                features=row["features"],
                in_dim=row["in_dim"],
                accuracy=row["accuracy"],
                macro_f1=row["macro_f1"],
                balanced_accuracy=row["balanced_accuracy"],
                top_axis_consistency=row["top_axis_consistency"],
                evidence_citation_rate=row["evidence_citation_rate"],
                hallucination_rate=row["hallucination_rate"],
            )
        )
    lines.append("")
    lines.append("## 2. External validation on EBR (no curator celltype labels)")
    lines.append("")
    lines.append("EBR has no curator cell-type labels, so accuracy / macro-F1 / balanced")
    lines.append("accuracy are not directly computable. We report top-axis consistency")
    lines.append("(does the predicted top axis agree with the per-cell CartiGM top axis?),")
    lines.append("evidence citation (does the predicted axis have non-zero CartiGM")
    lines.append("support?), hallucination rate, and the per-config dominant prediction.")
    lines.append("")
    lines.append("| config | features | n_cells | n_pred_classes | top_pred_class | top_pred_frac | top_axis_consistency | evidence_citation | hallucination |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in ebr_df.iterrows():
        lines.append(
            "| {config} | {features} | {n_cells} | {n_predicted_classes} | {top_predicted_class} | {top_predicted_class_frac} | {top_axis_consistency_vs_cartigm} | {evidence_citation_rate} | {hallucination_rate} |".format(
                config=row["config"],
                features=row["features"],
                n_cells=row["n_cells"],
                n_predicted_classes=row["n_predicted_classes"],
                top_predicted_class=row["top_predicted_class"],
                top_predicted_class_frac=row["top_predicted_class_frac"],
                top_axis_consistency_vs_cartigm=row["top_axis_consistency_vs_cartigm"],
                evidence_citation_rate=row["evidence_citation_rate"],
                hallucination_rate=row["hallucination_rate"],
            )
        )
    lines.append("")
    lines.append("## 3. Per-cluster majority prediction on EBR")
    lines.append("")
    lines.append("| cluster | n_cells | cartigm_only | scgpt_only | gsfm_only | cartigm_scgpt | cartigm_gsfm | full_fusion |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    pivot = ebr_per_cluster.pivot_table(
        index="cluster", columns="config", values="majority_pred", aggfunc="first"
    )
    if not pivot.empty:
        for cluster_id, row in pivot.iterrows():
            cells_in_cluster = int(ebr_per_cluster.loc[ebr_per_cluster["cluster"] == cluster_id, "n_cells"].iloc[0]) if "n_cells" in ebr_per_cluster.columns else 0
            lines.append(
                "| {cid} | {n} | {a} | {b} | {c} | {d} | {e} | {f} |".format(
                    cid=cluster_id,
                    n=cells_in_cluster,
                    a=str(row.get("cartigm_only", "")),
                    b=str(row.get("scgpt_only", "")),
                    c=str(row.get("gsfm_only", "")),
                    d=str(row.get("cartigm_scgpt", "")),
                    e=str(row.get("cartigm_gsfm", "")),
                    f=str(row.get("full_fusion", "")),
                )
            )
    lines.append("")
    lines.append("## 4. Caveats")
    lines.append("")
    lines.append("- The scGPT branch is the bundled 42-axis proxy (real scGPT-human")
    lines.append("  weights are not downloadable in this sandbox). Every result that")
    lines.append("  depends on it carries `fallback=True`. Swap in real weights by")
    lines.append("  editing `cartigsfm.fusion.build_scgpt_per_cell_embedding`.")
    lines.append("- GSFM here is the per-cell Jaccard between top-50 expressed genes")
    lines.append("  and each axis's `panel_genes` (no real GSFM weights loaded).")
    lines.append("- acc was subsampled to " + str(n_acc) + " cells stratified by sample")
    lines.append("  to keep the run in memory; the split is then sample-stratified")
    lines.append("  80/20 so no cell-level leakage is possible.")
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-acc-cells", type=int, default=30000)
    p.add_argument("--n-ebr-cells", type=int, default=0, help="0 = use all EBR cells")
    p.add_argument("--label-col", default="chongdrocyte_subtype")
    p.add_argument("--sample-col", default="sample")
    p.add_argument("--cluster-col", default="leiden_res0_5")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--outdir", default=str(OUTDIR))
    p.add_argument("--report", default=str(REPORT))
    p.add_argument("--scgpt-checkpoint", default=None,
                   help="path to a cartigsfm.scgpt_pretrain checkpoint (.pt). "
                        "If set, the scGPT branch uses the real pretrained "
                        "transformer (frozen) instead of the 42-axis proxy. "
                        "Falls back to the proxy if the file is missing.")
    p.add_argument("--scgpt-feature", default="auto",
                   choices=["auto", "axis42", "cell_dmodel"],
                   help="When using --scgpt-checkpoint: 'axis42' uses the "
                        "L2-norm of axis-gene embeddings (42-dim); "
                        "'cell_dmodel' uses the full d_model cell embedding; "
                        "'auto' picks cell_dmodel.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = prefer_device(args.device)
    print(f"[run_fusion] device = {device}", flush=True)

    a_sub, keep, sample_groups, labels_text = subsample_acc(
        args.n_acc_cells, args.label_col, args.sample_col, args.seed,
    )
    a_sub.write(outdir / "acc_subsample.h5ad")
    print(f"[run_fusion] wrote {outdir / 'acc_subsample.h5ad'}", flush=True)
    class_names = sorted(set(labels_text.tolist()))
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    labels_int = np.array([label_to_idx[s] for s in labels_text], dtype=np.int64)

    print(f"[run_fusion] extracting CartiGM per-cell on acc ...", flush=True)
    cartigm_acc, axis_ids = F.build_cartigm_per_cell_scores(a_sub, device=device)
    use_pretrained = bool(args.scgpt_checkpoint) and Path(args.scgpt_checkpoint).is_file()
    if use_pretrained:
        print(f"[run_fusion] extracting scGPT-pretrained per-cell on acc ...", flush=True)
        scgpt_acc_axis42, axis_ids2, scgpt_meta = F.build_scgpt_pretrained_per_cell(
            a_sub, args.scgpt_checkpoint, device=device,
        )
        scgpt_cell_emb_acc = scgpt_meta.pop("_cell_embedding_array", None)
        feature_pick = args.scgpt_feature if args.scgpt_feature != "auto" else "cell_dmodel"
        if feature_pick == "cell_dmodel" and scgpt_cell_emb_acc is not None:
            scgpt_acc = scgpt_cell_emb_acc
        else:
            scgpt_acc = scgpt_acc_axis42
        assert axis_ids == axis_ids2
    else:
        print(f"[run_fusion] extracting scGPT-proxy per-cell on acc ...", flush=True)
        scgpt_acc, axis_ids2, scgpt_meta = F.build_scgpt_per_cell_embedding(a_sub, device=device)
        assert axis_ids == axis_ids2
        scgpt_cell_emb_acc = None
        feature_pick = "axis42"
    print(f"[run_fusion] extracting GSFM per-cell on acc ...", flush=True)
    gsfm_acc, axis_ids3 = F.build_gsfm_per_cell_similarity(a_sub, top_n_markers=50, device=device)
    assert axis_ids == axis_ids3
    print(f"[run_fusion] done. acc features: cartigm {cartigm_acc.shape}, scgpt {scgpt_acc.shape}, gsfm {gsfm_acc.shape}", flush=True)

    np.savez_compressed(
        outdir / "acc_features.npz",
        cartigm=cartigm_acc,
        scgpt=scgpt_acc,
        gsfm=gsfm_acc,
        labels=labels_int,
        samples=sample_groups,
        axis_ids=np.array(axis_ids, dtype=object),
    )

    train_idx, val_idx = F.split_by_sample(sample_groups, train_frac=0.8, seed=args.seed)
    print(f"[run_fusion] sample-stratified split: train={len(train_idx)} val={len(val_idx)}", flush=True)
    train_samples = sorted(set(sample_groups[train_idx].tolist()))
    val_samples = sorted(set(sample_groups[val_idx].tolist()))

    features = {"cartigm": cartigm_acc, "scgpt": scgpt_acc, "gsfm": gsfm_acc}
    acc_df, heads = F.run_six_config_ablation(
        features, labels_int, sample_groups, axis_ids, class_names,
        outdir=str(outdir), device=args.device, epochs=args.epochs, seed=args.seed,
    )
    print(f"[run_fusion] acc ablation done", flush=True)
    print(acc_df.to_string(index=False), flush=True)

    a_ebr, cluster_ebr = load_ebr(args.cluster_col)
    if args.n_ebr_cells and args.n_ebr_cells < a_ebr.n_obs:
        rng = np.random.default_rng(int(args.seed))
        eb_keep = np.sort(rng.choice(a_ebr.n_obs, size=args.n_ebr_cells, replace=False))
        a_ebr = a_ebr[eb_keep].copy()
        cluster_ebr = cluster_ebr[eb_keep]
    print(f"[run_fusion] extracting features on EBR ({a_ebr.shape}) ...", flush=True)
    cartigm_ebr, _ = F.build_cartigm_per_cell_scores(a_ebr, device=device)
    if use_pretrained:
        scgpt_ebr_axis42, _, ebr_scgpt_meta = F.build_scgpt_pretrained_per_cell(
            a_ebr, args.scgpt_checkpoint, device=device,
        )
        scgpt_cell_emb_ebr = ebr_scgpt_meta.pop("_cell_embedding_array", None)
        if feature_pick == "cell_dmodel" and scgpt_cell_emb_ebr is not None:
            scgpt_ebr = scgpt_cell_emb_ebr
        else:
            scgpt_ebr = scgpt_ebr_axis42
    else:
        scgpt_ebr, _, _ = F.build_scgpt_per_cell_embedding(a_ebr, device=device)
    gsfm_ebr, _ = F.build_gsfm_per_cell_similarity(a_ebr, top_n_markers=50, device=device)
    np.savez_compressed(
        outdir / "ebr_features.npz",
        cartigm=cartigm_ebr, scgpt=scgpt_ebr, gsfm=gsfm_ebr,
        cluster=np.array(cluster_ebr, dtype=object),
    )
    ebr_features = {"cartigm": cartigm_ebr, "scgpt": scgpt_ebr, "gsfm": gsfm_ebr}
    cluster_int = np.array([hash(c) % (2 ** 31) for c in cluster_ebr], dtype=np.int64)
    ebr_metrics_df, ebr_per_cell = F.validate_on_ebr(
        ebr_features, heads, class_names, axis_ids,
        cluster_labels=cluster_int, device=args.device,
    )
    ebr_metrics_df.to_csv(outdir / "ebr_metrics.tsv", sep="\t", index=False)
    print(f"[run_fusion] EBR validation done", flush=True)
    print(ebr_metrics_df.to_string(index=False), flush=True)

    majority_rows = []
    for config_name in [c[0] for c in F.CONFIGS]:
        sub = ebr_per_cell[ebr_per_cell["config"] == config_name]
        for cluster_id, grp in sub.groupby("cluster"):
            labs = grp["predicted_class"].astype(str).tolist()
            top, top_n = Counter(labs).most_common(1)[0]
            majority_rows.append({
                "config": config_name,
                "cluster": str(cluster_ebr[np.where(cluster_int == cluster_id)[0][0]]),
                "majority_pred": top,
                "majority_n": int(top_n),
                "n_cells": int(len(grp)),
            })
    ebr_per_cluster = pd.DataFrame(majority_rows)
    ebr_per_cluster.to_csv(outdir / "ebr_per_cluster_predictions.tsv", sep="\t", index=False)

    summary = {
        "device": str(device),
        "scgpt_fallback": scgpt_meta,
        "n_acc_cells": int(a_sub.n_obs),
        "n_ebr_cells": int(a_ebr.n_obs),
        "n_classes": int(len(class_names)),
        "class_names": class_names,
        "axis_ids": axis_ids,
        "n_train_samples": int(len(train_samples)),
        "n_val_samples": int(len(val_samples)),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (outdir / "fusion_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report_md = write_report(
        acc_df, ebr_metrics_df, ebr_per_cluster,
        train_samples, val_samples,
        a_sub.n_obs, a_ebr.n_obs, class_names, len(class_names), time.time() - t0,
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report_md, encoding="utf-8")
    print(f"[run_fusion] wrote {args.report}", flush=True)
    print(f"[run_fusion] total elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
