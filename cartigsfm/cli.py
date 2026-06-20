"""Command-line interface for CartiGSFM.

Usage:
    python -m cartigsfm score --query my_genes.txt --version v0.3.1
    python -m cartigsfm project --matrix bulk.tsv --version v0.3.1 --out scores.tsv
    python -m cartigsfm versions
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .dictionary import (
    load_alias_map,
    load_dictionary,
    load_function_dictionary,
    load_function_specificity,
    list_function_versions,
    list_versions,
)
from .scoring import resolve_aliases, score_function_query, score_query
from .projection import project_bulk, project_function_bulk
from .p4 import run_p4_project
from . import interpret as _interpret
from .assets import (
    find_claim_safety,
    get_p9_adapter_path,
    list_cartilage_dictionary_versions,
    list_p9_versions,
    list_rag_versions,
    load_p9_hallucination_audit,
    load_p9_model_comparison,
    load_p9_training_config,
    load_cartilage_dictionary_v1,
    load_rag_knowledge_base,
    p9_is_adapter_available,
)


def cmd_score(args):
    text = Path(args.query).read_text(encoding="utf-8")
    genes = [g.strip() for g in text.splitlines() if g.strip()]
    if not args.no_alias:
        genes = resolve_aliases(genes, load_alias_map())
    frames = []
    if args.kind in ("subtype", "both"):
        d = load_dictionary(args.version)
        df_sub = score_query(genes, d, anti_penalty=args.anti_penalty)
        if not df_sub.empty:
            df_sub.insert(0, "kind", "subtype")
            df_sub = df_sub.rename(columns={"subtype": "category"})
        frames.append(df_sub)
    if args.kind in ("function", "both"):
        fn_spec = load_function_specificity(args.function_version)
        fn_dict = load_function_dictionary(args.function_version)
        df_fn = score_function_query(genes, fn_spec, fn_dict)
        if not df_fn.empty:
            df_fn.insert(0, "kind", "function")
            df_fn = df_fn.rename(columns={"function": "category"})
        frames.append(df_fn)
    frames = [f for f in frames if f is not None and not f.empty]
    df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not df.empty and "combined" in df.columns:
        df = df.sort_values("combined", ascending=False).reset_index(drop=True)
    if args.top is not None:
        df = df.head(args.top)
    if args.out:
        df.to_csv(args.out, sep="\t", index=False)
        print(f"wrote {args.out} ({len(df)} rows)")
    else:
        print(df.to_string(index=False))


def cmd_project(args):
    if args.matrix.endswith(".csv"):
        sep = ","
    else:
        sep = "\t"
    df = pd.read_csv(args.matrix, sep=sep)
    samples = args.samples.split(",") if args.samples else None
    alias_map = {} if args.no_alias else load_alias_map()
    frames = []
    if args.kind in ("subtype", "both"):
        d = load_dictionary(args.version)
        out_sub = project_bulk(
            df,
            d,
            sample_cols=samples,
            gene_col=args.gene_col,
            anti_lambda=args.anti_lambda,
            alias_map=alias_map,
        )
        if not out_sub.empty:
            out_sub.insert(0, "kind", "subtype")
            out_sub = out_sub.rename(columns={"subtype": "category"})
        frames.append(out_sub)
    if args.kind in ("function", "both"):
        fn_spec = load_function_specificity(args.function_version)
        fn_dict = load_function_dictionary(args.function_version)
        out_fn = project_function_bulk(
            df,
            fn_spec,
            fn_dict,
            sample_cols=samples,
            gene_col=args.gene_col,
            alias_map=alias_map,
            consensus_weight=args.consensus_weight,
        )
        if not out_fn.empty:
            out_fn.insert(0, "kind", "function")
            out_fn = out_fn.rename(columns={"function": "category"})
        frames.append(out_fn)
    frames = [f for f in frames if f is not None and not f.empty]
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    out.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out} ({len(out)} rows)")


def cmd_versions(args):
    print("subtype:")
    for v in list_versions():
        print(f"  {v}")
    print("function:")
    for v in list_function_versions():
        print(f"  {v}")
    print("cartilage-dictionary:")
    for v in list_cartilage_dictionary_versions():
        print(f"  {v}")
    print("rag:")
    for v in list_rag_versions():
        print(f"  {v}")
    print("p9:")
    for v in list_p9_versions():
        print(f"  {v}")


def _iter_dictionary_v1_axes(dictionary):
    for layer, layer_obj in dictionary.get("layers", {}).items():
        for axis in layer_obj.get("axes", []):
            yield layer, axis


def cmd_dictionary_v1(args):
    dictionary = load_cartilage_dictionary_v1()
    rows = []
    for layer, axis in _iter_dictionary_v1_axes(dictionary):
        rows.append({
            "layer": layer,
            "axis_id": axis.get("axis_id", ""),
            "name_en": axis.get("name_en", ""),
            "name_cn": axis.get("name_cn", ""),
            "core_genes_n": len(axis.get("core_genes", [])),
            "panel_genes_n": len(axis.get("panel_genes", [])),
            "status": axis.get("status", ""),
        })
    df = pd.DataFrame(rows)
    if args.layer:
        df = df[df["layer"] == args.layer]
    if args.out:
        df.to_csv(args.out, sep="\t", index=False)
        print(f"wrote {args.out} ({len(df)} rows)")
    else:
        print(
            json.dumps(
                {
                    "version": dictionary.get("version"),
                    "description": dictionary.get("description"),
                    "axis_count": len(rows),
                    "layers": {
                        layer: obj.get("count", len(obj.get("axes", [])))
                        for layer, obj in dictionary.get("layers", {}).items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.show_axes:
            print(df.to_string(index=False))


def cmd_rag_info(args):
    kb = load_rag_knowledge_base(args.version)
    summary = {
        "version": kb.get("version"),
        "description": kb.get("description"),
        "dictionary_axes": len(kb.get("dictionary_knowledge", []))
        if isinstance(kb.get("dictionary_knowledge"), list)
        else None,
        "claim_registry_classes": list(kb.get("claim_registry", {}).keys())
        if isinstance(kb.get("claim_registry"), dict)
        else None,
        "knowledge_base_complete": kb.get("knowledge_base_complete"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_claim_check(args):
    entry = find_claim_safety(args.claim, args.version)
    if entry is None:
        print(json.dumps({
            "claim": args.claim,
            "matched": False,
            "safety_classification": "UNKNOWN",
            "can_claim": False,
            "rationale": "No exact bundled claim-safety match. Treat as unsupported until reviewed.",
        }, ensure_ascii=False, indent=2))
        return
    print(json.dumps(entry, ensure_ascii=False, indent=2))


def cmd_p9_info(args):
    config = load_p9_training_config(args.version)
    comparison = load_p9_model_comparison(args.version)
    adapter_path = get_p9_adapter_path(args.adapter_dir)
    lora_only = next((row for row in comparison if row.get("system") == "lora_only"), {})
    base = next((row for row in comparison if row.get("system") == "base"), {})
    summary = {
        "version": args.version,
        "actually_trained": config.get("actually_trained"),
        "base_model": config.get("base_model"),
        "device": config.get("device"),
        "train_used": config.get("data", {}).get("n_train_used"),
        "train_available": config.get("data", {}).get("n_total_train"),
        "adapter_path": str(adapter_path),
        "adapter_available": p9_is_adapter_available(adapter_path),
        "base_refusal_when_required": base.get("refusal_when_required"),
        "lora_refusal_when_required": lora_only.get("refusal_when_required"),
        "base_evidence_citation_rate": base.get("evidence_citation_rate"),
        "lora_evidence_citation_rate": lora_only.get("evidence_citation_rate"),
        "known_limitations": config.get("limitations", []),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_p9_eval(args):
    rows = load_p9_model_comparison(args.version)
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, sep="\t", index=False)
        print(f"wrote {args.out} ({len(df)} rows)")
    else:
        print(df.to_string(index=False))
    if args.show_hallucinations:
        audit = pd.DataFrame(load_p9_hallucination_audit(args.version))
        print("\nP9 hallucination audit:")
        print(audit.to_string(index=False))


def cmd_p9_adapter_path(args):
    path = get_p9_adapter_path(args.adapter_dir)
    print(path)
    if args.check:
        print(f"available={p9_is_adapter_available(path)}")


def cmd_interpret(args):
    out_path = Path(args.out) if args.out else None
    if args.mode == "genes":
        if args.gene_file:
            text = Path(args.gene_file).read_text(encoding="utf-8")
            sep = chr(10)  # newline
            genes = [g.strip() for g in text.replace(",", sep).splitlines() if g.strip()]
        else:
            sep = chr(10)  # newline
            raw = args.genes or ""
            genes = [g.strip() for g in raw.replace(",", sep).splitlines() if g.strip()]
        interpretation = _interpret.interpret_gene_list(
            genes, top_per_layer=args.top_per_layer, overall_top=args.overall_top
        )
    elif args.mode == "p4-dir":
        interpretation = _interpret.interpret_p4_dir(
            args.input, top_per_layer=args.top_per_layer, overall_top=args.overall_top
        )
    elif args.mode == "p4-csv":
        interpretation = _interpret.interpret_p4_csv(
            args.input, top_per_layer=args.top_per_layer, overall_top=args.overall_top
        )
    else:
        raise ValueError("unknown --mode " + repr(args.mode))
    if args.claim:
        interpretation = _interpret.apply_safety_filter(interpretation, additional_claims=args.claim)
    fmt = args.format
    if out_path is None:
        if fmt == "json":
            print(_interpret.render_json(interpretation))
        else:
            print(_interpret.render_markdown(interpretation))
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            out_path.write_text(_interpret.render_json(interpretation), encoding="utf-8")
        else:
            out_path.write_text(_interpret.render_markdown(interpretation), encoding="utf-8")
        print("wrote " + str(out_path))

def cmd_agent(args):
    from . import agent as _agent
    if args.mode == "llm":
        model_path = args.model
        if model_path is None:
            print("--model is required for --mode llm", flush=True)
            return 1
        out = _agent.run_query_llm(
            args.query, model_path,
            max_iterations=args.max_iterations,
            max_new_tokens=args.max_new_tokens,
        )
        if args.format == "json":
            import json as _json
            print(_json.dumps(out, ensure_ascii=False, indent=2, default=str))
        else:
            print(out.get("answer", ""))
            for step in out.get("trace", []):
                if "tool" in step:
                    print(f"[tool {step['tool']} args={step['args']}]")
        return 0
    out = _agent.run_query_keyword(args.query)
    import json as _json
    if args.format == "json":
        print(_json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        print("tool:", out.get("tool"))
        if out.get("tool") == "cartigm_score":
            from . import interpret as _interpret
            print(_interpret.render_markdown(out["result"]))
        else:
            print(_json.dumps(out.get("result"), ensure_ascii=False, indent=2, default=str)[:4000])
    return 0

def cmd_p4_project(args):
    outputs = run_p4_project(
        outdir=args.outdir,
        h5ad=args.h5ad,
        pseudobulk_tsv=args.pseudobulk,
        meta_tsv=args.meta,
        sample_col=args.sample_col,
        tissue_col=args.tissue_col,
        cluster_col=args.cluster_col,
        celltype_col=args.celltype_col,
        celltype_regex=None if args.no_celltype_filter else args.celltype_regex,
        layer=args.layer,
        min_cells=args.min_cells,
        gene_col=args.gene_col,
        anti_lambda=args.anti_lambda,
        streaming=args.streaming,
        chunk_size=args.chunk_size,
    )
    print(f"wrote P4 delivery to {args.outdir}")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


def cmd_inspect_h5ad(args):
    from .p4 import auto_detect_obs_columns
    try:
        import anndata as ad
    except ImportError as exc:
        raise SystemExit("inspect-h5ad requires anndata. Install with: pip install anndata") from exc
    adata = ad.read_h5ad(args.h5ad, backed="r")
    summary = auto_detect_obs_columns(adata)
    if args.format == "json":
        import json as _json
        print(_json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return
    print(f"# h5ad inspect: {args.h5ad}")
    print(f"n_cells={summary['n_cells']}  n_genes={summary['n_genes']}")
    print(f"var_names_are_gene_symbols={summary['var_names_are_gene_symbols']}")
    print(f"obs_columns ({len(summary['obs_columns'])}): {summary['obs_columns']}")
    for role in ("sample_col", "tissue_col", "cluster_col", "celltype_col"):
        info = summary[role]
        alts = ", ".join(
            f"{a['column']}({a['score']:.2f})" for a in info["alternatives"][:3]
        ) or "n/a"
        print(f"  {role}: best={info['best']!r}  confidence={info['confidence']:.2f}  alts=[{alts}]")


def cmd_cs_predict(args):
    """P-F: per-cell cell_subtype prediction using the bundled v1 classifier."""
    import numpy as np
    import pandas as pd

    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "cs-predict requires anndata; install cartigsfm[scrna] or `pip install anndata`."
        ) from exc
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit("cs-predict requires torch; install pytorch first.") from exc

    from .cs_classifier import (
        align_to_genes,
        bundled_classifier_path,
        load_classifier,
        predict_from_array,
    )

    ckpt = Path(args.ckpt) if args.ckpt else bundled_classifier_path()
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sys.stderr.write(f"[cs-predict] loading classifier {ckpt} on {device}\n")
    model, classes, genes, cfg = load_classifier(ckpt, device=device)

    sys.stderr.write(f"[cs-predict] reading h5ad {args.h5ad}\n")
    adata = ad.read_h5ad(args.h5ad)
    if args.layer:
        if args.layer not in adata.layers:
            raise SystemExit(
                f"layer '{args.layer}' not in adata.layers (have: {list(adata.layers)})"
            )
        X = adata.layers[args.layer]
    else:
        X = adata.X

    src_genes = list(adata.var_names)
    sys.stderr.write(
        f"[cs-predict] aligning {X.shape[0]} cells x {X.shape[1]} genes -> "
        f"{cfg.n_in} HVG basis\n"
    )
    X_aligned, hit = align_to_genes(X, src_genes, genes)
    sys.stderr.write(
        f"[cs-predict] HVG basis hits: {hit}/{cfg.n_in} "
        f"({100.0 * hit / cfg.n_in:.1f}%); zero-padded the rest\n"
    )

    sys.stderr.write("[cs-predict] running forward pass\n")
    idx, probs = predict_from_array(
        X_aligned, model, classes, device=device, batch_size=int(args.batch_size)
    )
    pred = np.array(classes)[idx]

    obs_index = list(adata.obs_names)
    df = pd.DataFrame({"cell": obs_index, "predicted_celltype": pred})
    for j, c in enumerate(classes):
        df[f"prob::{c}"] = probs[:, j]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    sys.stderr.write(f"[cs-predict] wrote {out} ({len(df)} rows)\n")

    counts = pd.Series(pred).value_counts()
    print(f"# cs-predict | n_cells={len(df)} | hvg_hits={hit}/{cfg.n_in}")
    for c, n in counts.items():
        print(f"{c}\t{n}\t{n / len(df):.3f}")


def cmd_ablation(args):
    from .ablation import run_ablation, render_ablation_markdown
    result = run_ablation(args.outdir, sample_meta_col=args.meta_col)
    if args.format == "json":
        import json as _json
        # Convert pandas objects to JSON-safe form
        safe = {k: v for k, v in result.items() if k != "per_config_top1"}
        safe["per_config_top1"] = result["per_config_top1"]
        print(_json.dumps(safe, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_ablation_markdown(result))


def cmd_ablate_real(args):
    from .ablation import run_ablation_real, render_ablation_markdown
    kwargs = {
        "sample_meta_col": args.meta_col,
        "celltype_meta_col": args.celltype_col,
        "use_real_scgpt_gsfm": bool(getattr(args, "use_real_scgpt_gsfm", False)),
    }
    if args.no_refusal_audit:
        kwargs["refusal_claims"] = []
    result = run_ablation_real(args.outdir, **kwargs)
    if getattr(args, "json_out", None):
        import json as _json
        from pathlib import Path as _P
        json_path = _P(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(_json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"wrote {json_path}")
    if getattr(args, "out", None):
        from pathlib import Path as _P
        out_path = _P(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_ablation_markdown(result), encoding="utf-8")
        print(f"wrote {out_path}")
        return
    if args.format == "json":
        import json as _json
        print(_json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_ablation_markdown(result))


def cmd_annotate(args):
    import json as _json
    from pathlib import Path as _P
    from .annotate import (
        annotate_with_cartigsm,
        annotate_with_cellassign,
        annotate_with_celltypist,
        annotate_with_gptcelltype,
        annotate_with_marker_rule,
        annotate_with_scmap,
        annotate_with_scgpt,
        annotate_with_singler,
        annotate_with_symphony,
        compare_annotations,
    )
    from .assets import load_cartilage_dictionary_v1

    out_md = _P(args.out) if args.out else None
    out_cluster_tsv = _P(args.per_cluster_tsv) if args.per_cluster_tsv else None
    gptc_tsv = _P(args.gptcelltype_prompts_tsv) if args.gptcelltype_prompts_tsv else None

    method = args.method
    per_method: dict = {}
    notes: dict = {}

    if method in ("cartigsm", "all"):
        if not args.p4_outdir:
            print("--p4-outdir is required for cartigsm", flush=True)
            return 1
        df = annotate_with_cartigsm(args.p4_outdir, write_back=True)
        per_method["cartigsm"] = df
        notes["cartigsm"] = "CartiGSFM P4 + cartilage_dictionary_v1"

    if method in ("marker_rule", "all"):
        if not args.p4_outdir:
            print("--p4-outdir is required for marker_rule", flush=True)
            return 1
        df = annotate_with_marker_rule(args.p4_outdir, write_back=True)
        per_method["marker_rule"] = df
        notes["marker_rule"] = "marker-only re-run of cartilage_dictionary_v1 on the P4 pseudobulk"

    if method in ("scgpt", "all"):
        if not args.query_h5ad:
            print("--query-h5ad is required for scgpt", flush=True)
            return 1
        r = annotate_with_scgpt(
            args.query_h5ad,
            cluster_col=args.cluster_col,
            device=args.device,
        )
        per_method["scgpt"] = r.get("per_cluster")
        notes["scgpt"] = r.get("note", "")
        if r.get("per_cluster") is not None and not r["per_cluster"].empty and out_cluster_tsv is not None:
            tsv = _P(str(out_cluster_tsv).replace(".tsv", "_scgpt.tsv"))
            tsv.parent.mkdir(parents=True, exist_ok=True)
            r["per_cluster"].to_csv(tsv, sep="\t", index=False)
            print(f"wrote {tsv}")

    if method in ("celltypist", "all"):
        if not (args.query_h5ad and args.reference_h5ad):
            print("--query-h5ad and --reference-h5ad are required for celltypist", flush=True)
            return 1
        r = annotate_with_celltypist(
            args.query_h5ad,
            args.reference_h5ad,
            reference_label_col=args.reference_label_col,
            cluster_col=args.cluster_col,
            out_tsv=str(out_cluster_tsv).replace(".tsv", "_celltypist_per_cell.tsv") if out_cluster_tsv else None,
            max_reference_cells=args.max_reference_cells,
            device=args.device,
        )
        per_method["celltypist"] = r.get("per_cluster", None)
        notes["celltypist"] = r.get("note", "")
        if r.get("available") and r.get("per_cluster") is not None and not r["per_cluster"].empty and out_cluster_tsv is not None:
            tsv = _P(str(out_cluster_tsv).replace(".tsv", "_celltypist.tsv"))
            tsv.parent.mkdir(parents=True, exist_ok=True)
            r["per_cluster"].to_csv(tsv, sep="\t", index=False)
            print(f"wrote {tsv}")

    if method in ("gptcelltype", "all"):
        if not args.p4_outdir:
            print("--p4-outdir is required for gptcelltype marker generation", flush=True)
            return 1
        pseudo = _P(args.p4_outdir) / "tsv" / "p4_self_sample_cluster_pseudobulk.tsv"
        meta = _P(args.p4_outdir) / "tsv" / "p4_self_sample_cluster_meta.tsv"
        import pandas as _pd
        if pseudo.exists() and meta.exists():
            expr = _pd.read_csv(pseudo, sep="\t").set_index("gene")
            expr.index = expr.index.astype(str).str.upper()
            expr = expr.apply(_pd.to_numeric, errors="coerce").fillna(0.0)
            meta_df = _pd.read_csv(meta, sep="\t")
            samples = meta_df["tissue"].astype(str) + "|" + meta_df["tissue"].astype(str) + "|" + meta_df["cluster"].astype(str)
            sample_to_cluster = dict(zip(samples, meta_df["cluster"].astype(str)))
            cols = [c for c in expr.columns if c in sample_to_cluster]
            marker_dict = {
                sample_to_cluster[c]: expr[c].sort_values(ascending=False).head(20).index.tolist()
                for c in cols
            }
            gptc_tsv_str = str(gptc_tsv) if gptc_tsv else None
            r = annotate_with_gptcelltype(
                marker_dict,
                candidates=list(load_cartilage_dictionary_v1().get("layers", {}).get("cell_subtype", {}).get("axes", [])) and [ax.get("name_en", "") for ax in load_cartilage_dictionary_v1().get("layers", {}).get("cell_subtype", {}).get("axes", [])] or None,
                tissue="mixed (ear/nose/rib)",
                out_tsv=gptc_tsv_str,
            )
            notes["gptcelltype"] = r.get("note", "")

    # R-backed annotation backends. Each needs the query h5ad; SingleR /
    # scmap / Symphony also need a reference h5ad (acc.h5ad). CellAssign
    # uses the bundled cartilage_dictionary_v1 markers.
    r_call_args = dict(
        cluster_col=args.cluster_col,
    )
    for r_pkg, fn in (
        ("singler", annotate_with_singler),
        ("scmap", annotate_with_scmap),
        ("symphony", annotate_with_symphony),
        ("cellassign", annotate_with_cellassign),
    ):
        if method not in (r_pkg, "all"):
            continue
        if r_pkg in ("singler", "scmap", "symphony"):
            if not (args.query_h5ad and args.reference_h5ad):
                notes[r_pkg] = "skipped: --query-h5ad and --reference-h5ad are required for " + r_pkg
                continue
            kwargs = dict(
                query_h5ad=args.query_h5ad,
                reference_h5ad=args.reference_h5ad,
                reference_label_col=args.reference_label_col,
                cluster_col=args.cluster_col,
                max_reference_cells_per_label=getattr(args, "r_max_ref_per_label", 1500),
                random_state=getattr(args, "seed", 0),
            )
            if r_pkg == "symphony":
                kwargs["k"] = getattr(args, "symphony_k", 100)
                kwargs["d"] = getattr(args, "symphony_d", 20)
            r = fn(**kwargs)
        else:
            if not args.query_h5ad:
                notes[r_pkg] = "skipped: --query-h5ad is required for " + r_pkg
                continue
            r = fn(
                query_h5ad=args.query_h5ad,
                cluster_col=args.cluster_col,
                n_max_cells=getattr(args, "cellassign_max_cells", 8000),
                random_state=getattr(args, "seed", 0),
            )
        notes[r_pkg] = r.get("note", "")
        if r.get("available") and r.get("per_cluster") is not None and not r["per_cluster"].empty:
            per_method[r_pkg] = r["per_cluster"]
            if out_cluster_tsv is not None:
                tsv = _P(str(out_cluster_tsv).replace(".tsv", "_" + r_pkg + ".tsv"))
                tsv.parent.mkdir(parents=True, exist_ok=True)
                r["per_cluster"].to_csv(tsv, sep="\t", index=False)
                print("wrote " + str(tsv))

    available_methods = [m for m, df in per_method.items() if df is not None and not df.empty]
    if not available_methods:
        print("no backend produced per-cluster predictions", flush=True)
        return 0
    if method == "all":
        comparison = compare_annotations(per_method, reference="cartigsm")
        lines = ["# cartigsfm annotate (--method all) - cross-method summary", ""]
        lines.append("n_clusters_per_method: " + _json.dumps(comparison["n_clusters_per_method"], ensure_ascii=False))
        lines.append("")
        lines.append("pairwise_agreement: " + _json.dumps(comparison["pairwise_agreement"], ensure_ascii=False))
        lines.append("")
        wide = comparison.get("per_cluster_wide")
        if wide is not None and not wide.empty:
            lines.append("## per-cluster wide table")
            lines.append("")
            cols = [c for c in wide.columns]
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, row in wide.iterrows():
                lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
            lines.append("")
        lines.append("per_method notes:")
        for k, v in notes.items():
            lines.append("- " + k + ": " + str(v))
        if out_md is not None:
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text("\n".join(lines), encoding="utf-8")
            print("wrote " + str(out_md))
        else:
            print("\n".join(lines))
        return 0

    first_method = available_methods[0]
    df = per_method[first_method]
    if out_cluster_tsv is not None:
        out_cluster_tsv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_cluster_tsv, sep="\t", index=False)
        print("wrote " + str(out_cluster_tsv))
    else:
        print(df.to_string(index=False))
    return 0


def cmd_scgpt_pretrain(args):
    from .scgpt_pretrain import ScGPTConfig, pretrain
    cfg = ScGPTConfig(
        n_cells=args.n_cells,
        n_hvg=args.n_hvg,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        seed=args.seed,
    )
    summary = pretrain(args.h5ad, args.outdir, cfg)
    import json as _json
    print(_json.dumps(summary, indent=2))
    return 0


def cmd_train_fusion(args):
    """Wrap cartigsfm.run_fusion.main via argv injection."""
    import sys as _sys
    from . import run_fusion as _rf
    argv = ["cartigsfm-train-fusion"]
    for k, v in vars(args).items():
        if k in ("cmd", "func"):
            continue
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    saved = _sys.argv
    try:
        _sys.argv = argv
        _rf.main()
    finally:
        _sys.argv = saved
    return 0


def main():
    p = argparse.ArgumentParser(prog="cartigsfm",
                                description="CartiGSFM utilities")
    sub = p.add_subparsers(dest="cmd")

    sc = sub.add_parser("score", help="Score a gene list against the dictionary")
    sc.add_argument("--query", required=True)
    sc.add_argument("--kind", choices=["subtype", "function", "both"], default="subtype")
    sc.add_argument("--version", default="v0.3.1")
    sc.add_argument("--function-version", default="v0.6.5")
    sc.add_argument("--anti-penalty", type=float, default=1.0)
    sc.add_argument("--top", type=int, default=None)
    sc.add_argument("--out", default=None)
    sc.add_argument("--no-alias", action="store_true", help="skip HGNC alias resolution")
    sc.set_defaults(func=cmd_score)

    pr = sub.add_parser("project", help="Project a bulk expression matrix")
    pr.add_argument("--matrix", required=True)
    pr.add_argument("--kind", choices=["subtype", "function", "both"], default="subtype")
    pr.add_argument("--version", default="v0.3.1")
    pr.add_argument("--function-version", default="v0.6.5")
    pr.add_argument("--gene-col", default=None,
                    help="column with gene symbols (defaults to row index)")
    pr.add_argument("--samples", default=None,
                    help="comma-separated sample columns")
    pr.add_argument("--anti-lambda", type=float, default=0.5)
    pr.add_argument("--consensus-weight", type=float, default=0.25)
    pr.add_argument("--out", required=True)
    pr.add_argument("--no-alias", action="store_true", help="skip HGNC alias resolution")
    pr.set_defaults(func=cmd_project)

    sub.add_parser("versions", help="list installed cgrm versions").set_defaults(func=cmd_versions)

    dv1 = sub.add_parser("dictionary-v1", help="summarize bundled three-layer cartilage dictionary v1")
    dv1.add_argument("--layer", choices=["cell_subtype", "tissue_developmental_state", "functional_axis"], default=None)
    dv1.add_argument("--show-axes", action="store_true", help="print axis table after summary")
    dv1.add_argument("--out", default=None, help="write axis table to TSV")
    dv1.set_defaults(func=cmd_dictionary_v1)

    rag = sub.add_parser("rag-info", help="summarize bundled CartiGSFM-RAG resources")
    rag.add_argument("--version", default="v1")
    rag.set_defaults(func=cmd_rag_info)

    cc = sub.add_parser("claim-check", help="check an exact claim against bundled P6 safety rules")
    cc.add_argument("--claim", required=True)
    cc.add_argument("--version", default="v1")
    cc.set_defaults(func=cmd_claim_check)

    p9 = sub.add_parser("p9-info", help="summarize bundled P9 LoRA prototype metadata")
    p9.add_argument("--version", default="v1")
    p9.add_argument("--adapter-dir", default=None, help="optional local adapter directory")
    p9.set_defaults(func=cmd_p9_info)

    p9eval = sub.add_parser("p9-eval", help="print bundled P9 model comparison metrics")
    p9eval.add_argument("--version", default="v1")
    p9eval.add_argument("--out", default=None)
    p9eval.add_argument("--show-hallucinations", action="store_true")
    p9eval.set_defaults(func=cmd_p9_eval)

    p9path = sub.add_parser("p9-adapter-path", help="print local P9 adapter path")
    p9path.add_argument("--adapter-dir", default=None)

    ag = sub.add_parser("agent", help="CartiAgent: LLM-driven tool use over CartiGM (cartigm_score, p4_project, rag_evidence_lookup)")
    ag.add_argument("--query", required=True, help="natural language query; auto-routed by keyword or driven by --mode llm")
    ag.add_argument("--mode", choices=["keyword", "llm"], default="keyword",
                    help="keyword = rule-based dispatch; llm = Qwen2.5-7B native function calling")
    ag.add_argument("--model", default=None, help="local Qwen2.5-7B-Instruct path (required for --mode llm)")
    ag.add_argument("--max-iterations", type=int, default=4)
    ag.add_argument("--max-new-tokens", type=int, default=1024)
    ag.add_argument("--format", choices=["markdown", "json"], default="markdown")
    ag.set_defaults(func=cmd_agent)
    p9path.add_argument("--check", action="store_true")
    p9path.set_defaults(func=cmd_p9_adapter_path)

    interp = sub.add_parser("interpret", help="evidence-constrained interpretation of gene list / P4 score table")
    interp.add_argument("--mode", choices=["genes", "p4-dir", "p4-csv"], required=True,
                        help="genes: score a list of gene symbols; p4-dir: read cartigsfm p4-project outdir; p4-csv: read a long-form score CSV")
    interp.add_argument("--genes", default=None,
                        help="comma- or newline-separated gene list (use with --mode genes)")
    interp.add_argument("--gene-file", default=None,
                        help="path to a file with one gene per line (use with --mode genes)")
    interp.add_argument("--input", default=None,
                        help="path to a P4 outdir (--mode p4-dir) or to a long-form score TSV/CSV (--mode p4-csv)")
    interp.add_argument("--top-per-layer", type=int, default=3,
                        help="number of axes to keep per layer in the top-per-layer section")
    interp.add_argument("--overall-top", type=int, default=5,
                        help="number of axes in the overall top section")
    interp.add_argument("--claim", action="append", default=None,
                        help="audit an additional free-text claim against claim safety; pass multiple times for more")
    interp.add_argument("--format", choices=["markdown", "json"], default="markdown")
    interp.add_argument("--out", default=None, help="write report to this path (default: stdout)")
    interp.set_defaults(func=cmd_interpret)

    p4 = sub.add_parser("p4-project", help="project independent h5ad/pseudobulk data onto cartilage_dictionary_v1")
    src = p4.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad", default=None, help="input h5ad file; requires anndata")
    src.add_argument("--pseudobulk", default=None, help="genes x sample-cluster TSV with gene column")
    p4.add_argument("--meta", default=None, help="metadata TSV for --pseudobulk, indexed by sample-cluster id")
    p4.add_argument("--outdir", required=True)
    p4.add_argument("--sample-col", default="sample")
    p4.add_argument("--tissue-col", default="tissue")
    p4.add_argument("--cluster-col", default="cluster")
    p4.add_argument("--celltype-col", default=None)
    p4.add_argument("--celltype-regex", default="chondro|cartilage")
    p4.add_argument("--no-celltype-filter", action="store_true")
    p4.add_argument("--layer", default=None, help="optional h5ad layer to use instead of X")
    p4.add_argument("--min-cells", type=int, default=10)
    p4.add_argument("--gene-col", default="gene")
    p4.add_argument("--anti-lambda", type=float, default=0.5)
    p4.add_argument("--streaming", action="store_true", default=None,
                    help="force streaming pseudobulk (chunked backed-mode iteration); default: auto when h5ad > 2GB on disk")
    p4.add_argument("--no-streaming", dest="streaming", action="store_false", default=None,
                    help="force in-memory pseudobulk even for large h5ads")
    p4.add_argument("--chunk-size", type=int, default=None,
                    help="override the auto-resolved streaming chunk size (cells per chunk)")
    p4.set_defaults(func=cmd_p4_project)

    abl = sub.add_parser("ablation", help="four-way ablation: cartigm_only / cartigm_gsfm / cartigm_scgpt / full on a P4 outdir")
    abl.add_argument("--outdir", required=True, help="P4 outdir containing tsv/p4_sample_cluster_three_layer_scores.tsv and tsv/p4_self_sample_cluster_pseudobulk.tsv")
    abl.add_argument("--meta-col", default="tissue", help="metadata column whose values map cluster to ground-truth tissue")
    abl.add_argument("--format", choices=["markdown", "json"], default="markdown")
    abl.set_defaults(func=cmd_ablation)

    abr = sub.add_parser(
        "ablate-real",
        help="real-data ablation: annotation-based ground truth + P6/P9 LLM refusal audit. Use on a P4 outdir derived from a real cartilage single-cell experiment.",
    )
    abr.add_argument("--outdir", required=True, help="P4 outdir with tsv/p4_sample_cluster_three_layer_scores.tsv and tsv/p4_self_sample_cluster_meta.tsv")
    abr.add_argument("--meta-col", default="tissue", help="metadata column whose values map cluster to tissue (default 'tissue')")
    abr.add_argument("--celltype-col", default=None, help="optional metadata column whose values map cluster to celltype (e.g. 'chondrocyte-type')")
    abr.add_argument("--use-real-scgpt-gsfm", action="store_true",
                     help="if set, report that real scGPT-human / GSFM weights are in use; default marks both as lightweight fallback")
    abr.add_argument("--no-refusal-audit", action="store_true",
                     help="disable the P6/P9 LLM refusal probe")
    abr.add_argument("--format", choices=["markdown", "json"], default="markdown")
    abr.add_argument("--out", default=None, help="write the rendered report to this file (default: stdout)")
    abr.add_argument("--json-out", default=None, help="write the full JSON result to this file (default: skip)")
    abr.set_defaults(func=cmd_ablate_real)

    ins = sub.add_parser("inspect-h5ad", help="inspect a real h5ad and auto-detect sample / tissue / cluster / celltype columns")
    ins.add_argument("--h5ad", required=True)
    ins.add_argument("--format", choices=["markdown", "json"], default="markdown")
    ins.set_defaults(func=cmd_inspect_h5ad)

    ann = sub.add_parser("annotate", help="P15: annotate a query h5ad using one or more backends; emit per-cluster TSV + comparison Markdown")
    ann.add_argument("--method", choices=["cartigsm", "marker_rule", "scgpt", "celltypist", "gptcelltype",
                                          "singler", "scmap", "symphony", "cellassign", "all"], default="all",
                     help="which backend to run; 'all' runs every working backend and writes the cross-method comparison")
    ann.add_argument("--p4-outdir", default=None, help="P4 outdir (required for cartigsm / marker_rule)")
    ann.add_argument("--query-h5ad", default=None, help="query h5ad (required for scgpt / celltypist)")
    ann.add_argument("--reference-h5ad", default=None, help="reference h5ad (required for celltypist)")
    ann.add_argument("--reference-label-col", default="chongdrocyte_subtype", help="label column in the reference h5ad (default matches acc.h5ad)")
    ann.add_argument("--cluster-col", default="leiden_res0_5", help="cluster column in the query h5ad")
    ann.add_argument("--out", default=None, help="output Markdown report path")
    ann.add_argument("--per-cluster-tsv", default=None, help="output per-cluster TSV path (single-method run)")
    ann.add_argument("--gptcelltype-prompts-tsv", default=None, help="GPTcelltype prompts TSV (prompt-only mode if OPENAI_API_KEY is unset)")
    ann.add_argument("--max-reference-cells", type=int, default=50000, help="subsample cap for the CellTypist reference (avoids OOM on full-atlas)")
    ann.add_argument("--device", default=None, help="torch device for backends that support it; auto = cuda:0 if available else cpu. CellTypist records the choice but its sklearn SGD trainer is CPU-only.")
    ann.add_argument("--r-max-ref-per-label", type=int, default=1500,
                     help="cap on reference cells per label for SingleR / scmap / Symphony (keeps R session under 5 GB)")
    ann.add_argument("--symphony-k", type=int, default=100, help="Symphony number of soft clusters K")
    ann.add_argument("--symphony-d", type=int, default=20, help="Symphony number of harmonised PCs d")
    ann.add_argument("--cellassign-max-cells", type=int, default=8000,
                     help="subsample cap for the CellAssign query (TF training is slow)")
    ann.add_argument("--seed", type=int, default=0, help="random seed for reference subsampling")
    ann.set_defaults(func=cmd_annotate)

    sgp = sub.add_parser("scgpt-pretrain",
                         help="P16: scGPT-style MLM pretraining of a small transformer encoder on acc.h5ad (GPU)")
    sgp.add_argument("--h5ad", default=r"F:\cartifm\acc.h5ad")
    sgp.add_argument("--outdir", default=r"F:\cartifm\outputs\scgpt_pretrain")
    sgp.add_argument("--n-cells", type=int, default=80000)
    sgp.add_argument("--n-hvg", type=int, default=2000)
    sgp.add_argument("--n-steps", type=int, default=2000)
    sgp.add_argument("--batch-size", type=int, default=256)
    sgp.add_argument("--d-model", type=int, default=256)
    sgp.add_argument("--n-layers", type=int, default=6)
    sgp.add_argument("--n-heads", type=int, default=8)
    sgp.add_argument("--lr", type=float, default=8e-4)
    sgp.add_argument("--seed", type=int, default=0)
    sgp.set_defaults(func=cmd_scgpt_pretrain)

    tf = sub.add_parser("train-fusion",
                        help="P17: per-cell fusion ablation (CartiGM + scGPT-pretrained + GSFM, sample-stratified)")
    tf.add_argument("--n-acc-cells", type=int, default=30000)
    tf.add_argument("--n-ebr-cells", type=int, default=0)
    tf.add_argument("--label-col", default="chongdrocyte_subtype")
    tf.add_argument("--sample-col", default="sample")
    tf.add_argument("--cluster-col", default="leiden_res0_5")
    tf.add_argument("--epochs", type=int, default=80)
    tf.add_argument("--seed", type=int, default=0)
    tf.add_argument("--device", default=None)
    tf.add_argument("--outdir", default=r"F:\cartifm\outputs\fusion_P17")
    tf.add_argument("--report", default=r"F:\cartifm\CartiGM\reports\P17_FUSION_ABLATION.md")
    tf.add_argument("--scgpt-checkpoint", default=None,
                    help="path to a cartigsfm scGPT-pretrain checkpoint (.pt). If set, use real pretrained encoder.")
    tf.add_argument("--scgpt-feature", default="auto", choices=["auto", "axis42", "cell_dmodel"])
    tf.set_defaults(func=cmd_train_fusion)

    csp = sub.add_parser(
        "cs-predict",
        help="P-F: predict cell_subtype labels for an h5ad using the bundled v1 classifier",
    )
    csp.add_argument("--h5ad", required=True, help="input h5ad")
    csp.add_argument("--out", required=True, help="output TSV (per-cell predictions + class probabilities)")
    csp.add_argument(
        "--layer",
        default=None,
        help="adata layer to use (default: .X). For EBR-style data with negative .X use 'log1p_norm'.",
    )
    csp.add_argument(
        "--ckpt",
        default=None,
        help="override checkpoint path (default: bundled cs_classifier_v1/classifier.pt)",
    )
    csp.add_argument(
        "--device",
        default=None,
        help="torch device; default = cuda if available else cpu",
    )
    csp.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="inference batch size (cells per forward pass)",
    )
    csp.set_defaults(func=cmd_cs_predict)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
