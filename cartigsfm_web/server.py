from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path
from typing import Any

import cartigsfm
from cartigsfm.interpret import apply_safety_filter, interpret_gene_list, render_markdown
from cartigsfm.projection import _axis_anti_weights, _axis_gene_weights
from cartigsfm.scoring import resolve_aliases


def _require_fastapi():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "cartigsfm-web requires FastAPI. Install with: pip install 'cartigsfm[web]' "
            "or pip install fastapi uvicorn"
        ) from exc
    return FastAPI, HTTPException, FileResponse, StaticFiles, BaseModel, Field


FastAPI, HTTPException, FileResponse, StaticFiles, BaseModel, Field = _require_fastapi()


class GeneScoreRequest(BaseModel):
    genes: str | list[str] = Field(..., description="Gene symbols as comma/newline text or a JSON list")
    top: int = Field(5, ge=1, le=50)
    anti_penalty: float = Field(1.0, ge=0.0, le=5.0)
    no_alias: bool = False


class InterpretRequest(BaseModel):
    genes: str | list[str] = Field(..., description="Gene symbols as comma/newline text or a JSON list")
    top_per_layer: int = Field(3, ge=1, le=20)
    overall_top: int = Field(8, ge=1, le=50)
    claims: str | list[str] | None = Field(
        None,
        description="Optional manuscript claims. Newline-separated text or a JSON list.",
    )


class ClaimCheckRequest(BaseModel):
    claim: str = Field(..., min_length=1)


class P4CommandRequest(BaseModel):
    h5ad_path: str = "your_data.h5ad"
    outdir: str = "cartigm_p4_out"
    sample_col: str = "sample"
    tissue_col: str = "tissue"
    cluster_col: str = "cluster"
    celltype_col: str = "celltype"
    layer: str | None = None
    no_celltype_filter: bool = False
    min_cells: int | None = Field(None, ge=1)
    streaming: str = Field("auto", description="auto, force, or off")
    chunk_size: int | None = Field(None, ge=1)


class InspectCommandRequest(BaseModel):
    h5ad_path: str = "your_data.h5ad"
    output_format: str = Field("markdown", description="markdown or json")


class CSPredictCommandRequest(BaseModel):
    h5ad_path: str = "your_data.h5ad"
    out: str = "cartigm_cs_predictions.tsv"
    mode: str = Field("ensemble", description="v1, v2, or ensemble")
    layer: str | None = None
    device: str | None = None
    batch_size: int = Field(4096, ge=1)


def _parse_genes(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        tokens = raw
    else:
        tokens = re.split(r"[\s,;]+", raw)
    genes = [str(token).strip().upper() for token in tokens if str(token).strip()]
    if not genes:
        raise ValueError("Provide at least one gene symbol.")
    return genes


def _parse_claims(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        claims = [str(item).strip() for item in raw if str(item).strip()]
    else:
        claims = [line.strip() for line in str(raw).splitlines() if line.strip()]
    return claims


def _iter_axes(dictionary: dict[str, Any]):
    for layer, layer_obj in (dictionary.get("layers") or {}).items():
        for axis in layer_obj.get("axes", []):
            yield layer, axis


def _score_axis(genes: list[str], layer: str, axis: dict[str, Any], anti_penalty: float) -> dict[str, Any] | None:
    q = set(genes)
    marker_weights = _axis_gene_weights(axis)
    anti_weights = _axis_anti_weights(axis)
    marker_hits = sorted(q & set(marker_weights))
    anti_hits = sorted(q & set(anti_weights))
    if not marker_hits and not anti_hits:
        return None
    marker_den = max(1.0, sum(float(value) for value in marker_weights.values()))
    anti_den = max(1.0, sum(float(value) for value in anti_weights.values()))
    marker_score = sum(float(marker_weights[gene]) for gene in marker_hits) / marker_den
    anti_score = sum(float(anti_weights[gene]) for gene in anti_hits) / anti_den if anti_weights else 0.0
    combined = marker_score - anti_penalty * anti_score
    return {
        "layer": layer,
        "axis_id": axis.get("axis_id", ""),
        "name_en": axis.get("name_en", ""),
        "name_cn": axis.get("name_cn", ""),
        "combined": round(combined, 5),
        "marker_score": round(marker_score, 5),
        "anti_score": round(anti_score, 5),
        "marker_hits": marker_hits,
        "anti_hits": anti_hits,
        "n_marker_hits": len(marker_hits),
        "n_anti_hits": len(anti_hits),
        "evidence_level": axis.get("evidence_level", "ATLAS_INTERNAL"),
        "interpretation": axis.get("interpretation", ""),
        "limitations": axis.get("limitations", []),
    }


def _score_gene_list(raw_genes: str | list[str], top: int = 5, anti_penalty: float = 1.0, no_alias: bool = False) -> dict[str, Any]:
    genes = _parse_genes(raw_genes)
    if not no_alias:
        genes = resolve_aliases(genes, cartigsfm.load_alias_map())
    dictionary = cartigsfm.load_cartilage_dictionary_v1()
    rows = []
    for layer, axis in _iter_axes(dictionary):
        scored = _score_axis(genes, layer, axis, anti_penalty=anti_penalty)
        if scored is not None:
            rows.append(scored)
    rows = sorted(rows, key=lambda item: item["combined"], reverse=True)
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(row["layer"], []).append(row)
    for layer in list(by_layer):
        by_layer[layer] = by_layer[layer][:top]
    overall = rows[:top]
    return {
        "input_genes": genes,
        "n_input_genes": len(genes),
        "top": top,
        "overall": overall,
        "by_layer": by_layer,
        "safety_note": (
            "Gene-list scoring is a dictionary-based screening output. Treat high scores as hypotheses "
            "that require expression-level and experimental validation."
        ),
    }


def _interpret_gene_list(raw_genes: str | list[str], *, top_per_layer: int, overall_top: int,
                         claims: str | list[str] | None) -> dict[str, Any]:
    genes = _parse_genes(raw_genes)
    extra_claims = _parse_claims(claims)
    interpretation = interpret_gene_list(
        genes,
        top_per_layer=top_per_layer,
        overall_top=overall_top,
    )
    interpretation = apply_safety_filter(interpretation, additional_claims=extra_claims)
    markdown = render_markdown(interpretation)
    return {
        "interpretation": interpretation,
        "markdown": markdown,
        "mode": interpretation.get("mode"),
        "safety_summary": interpretation.get("safety_summary", {}),
        "cannot_claim": interpretation.get("cannot_claim", []),
        "top_axes_per_layer": interpretation.get("top_axes_per_layer", []),
        "overall_top_axes": interpretation.get("overall_top_axes", []),
        "warnings": interpretation.get("warnings", []),
    }


def _dictionary_summary() -> dict[str, Any]:
    dictionary = cartigsfm.load_cartilage_dictionary_v1()
    layers = {}
    for layer, layer_obj in (dictionary.get("layers") or {}).items():
        axes = []
        for axis in layer_obj.get("axes", []):
            axes.append(
                {
                    "axis_id": axis.get("axis_id", ""),
                    "name_en": axis.get("name_en", ""),
                    "name_cn": axis.get("name_cn", ""),
                    "core_genes": axis.get("core_genes", [])[:15],
                    "anti_genes": axis.get("anti_genes", [])[:10],
                    "evidence_level": axis.get("evidence_level", ""),
                    "interpretation": axis.get("interpretation", ""),
                    "aliases": axis.get("aliases", []),
                }
            )
        layers[layer] = {"count": layer_obj.get("count", len(axes)), "axes": axes}
    return {
        "version": dictionary.get("version", ""),
        "generated_at": dictionary.get("generated_at", ""),
        "description": dictionary.get("description", ""),
        "layers": layers,
    }


def _claim_check(claim: str) -> dict[str, Any]:
    exact = cartigsfm.find_claim_safety(claim)
    if exact is not None:
        return {"matched": True, "match_type": "exact", **exact}
    rules = cartigsfm.load_claim_safety_classifier()
    lowered = claim.casefold()
    keyword_matches = []
    for rule in rules:
        rule_claim = str(rule.get("claim", ""))
        tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", rule_claim.casefold()) if len(token) >= 5]
        overlap = sorted({token for token in tokens if token in lowered})
        if overlap:
            keyword_matches.append({"claim": rule_claim, "overlap": overlap[:8], "recommendation": rule.get("recommendation", "")})
    return {
        "matched": False,
        "match_type": "keyword_hint" if keyword_matches else "none",
        "claim": claim,
        "recommendation": (
            "No exact bundled safety rule matched. Use conservative wording and cite the CartiGM evidence table "
            "rather than claiming causal, clinical, or fully trained LLM performance."
        ),
        "related_rules": keyword_matches[:5],
    }


def _join_powershell_command(pieces: list[str]) -> str:
    return " `\n".join(pieces)


def _p4_command(req: P4CommandRequest) -> dict[str, Any]:
    pieces = [
        "cartigsfm p4-project",
        f'  --h5ad "{req.h5ad_path}"',
        f'  --outdir "{req.outdir}"',
        f"  --sample-col {req.sample_col}",
        f"  --tissue-col {req.tissue_col}",
        f"  --cluster-col {req.cluster_col}",
    ]
    if req.celltype_col:
        pieces.append(f"  --celltype-col {req.celltype_col}")
    if req.layer:
        pieces.append(f"  --layer {req.layer}")
    if req.no_celltype_filter:
        pieces.append("  --no-celltype-filter")
    if req.min_cells:
        pieces.append(f"  --min-cells {req.min_cells}")
    if req.streaming == "force":
        pieces.append("  --streaming")
    elif req.streaming == "off":
        pieces.append("  --no-streaming")
    if req.chunk_size:
        pieces.append(f"  --chunk-size {req.chunk_size}")
    return {
        "command": _join_powershell_command(pieces),
        "note": "The web portal generates the command but does not upload or process large h5ad files.",
    }


def _inspect_command(req: InspectCommandRequest) -> dict[str, Any]:
    output_format = req.output_format if req.output_format in {"markdown", "json"} else "markdown"
    pieces = [
        "cartigsfm inspect-h5ad",
        f'  --h5ad "{req.h5ad_path}"',
        f"  --format {output_format}",
    ]
    return {
        "command": _join_powershell_command(pieces),
        "note": "Run this locally before p4-project to confirm obs columns, layers, obsm keys, and candidate metadata columns.",
    }


def _cs_predict_command(req: CSPredictCommandRequest) -> dict[str, Any]:
    mode = req.mode if req.mode in {"v1", "v2", "ensemble"} else "ensemble"
    pieces = [
        "cartigsfm cs-predict",
        f'  --h5ad "{req.h5ad_path}"',
        f'  --out "{req.out}"',
        f"  --mode {mode}",
        f"  --batch-size {req.batch_size}",
    ]
    if req.layer:
        pieces.append(f"  --layer {req.layer}")
    if req.device:
        pieces.append(f"  --device {req.device}")
    return {
        "command": _join_powershell_command(pieces),
        "note": "The bundled cs_classifier predicts 10 CartiGM cell_subtype labels per cell. Use --layer log1p_norm when .X is not clean log-normalized expression.",
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="CartiGM Web Portal",
        version="0.2.0",
        description="Local web interface for CartiGM dictionary scoring, evidence-constrained interpretation, and command generation.",
    )
    static_dir = resources.files("cartigsfm_web").joinpath("static")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(static_dir.joinpath("index.html")))

    @app.get("/api/health")
    def health():
        dictionary = cartigsfm.load_cartilage_dictionary_v1()
        return {
            "ok": True,
            "cartigsfm_version": getattr(cartigsfm, "__version__", "unknown"),
            "dictionary_version": dictionary.get("version", ""),
            "axis_count": sum((layer.get("count", 0) for layer in (dictionary.get("layers") or {}).values())),
        }

    @app.get("/api/dictionary")
    def dictionary():
        return _dictionary_summary()

    @app.post("/api/score")
    def score(req: GeneScoreRequest):
        try:
            return _score_gene_list(req.genes, top=req.top, anti_penalty=req.anti_penalty, no_alias=req.no_alias)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/interpret")
    def interpret(req: InterpretRequest):
        try:
            return _interpret_gene_list(
                req.genes,
                top_per_layer=req.top_per_layer,
                overall_top=req.overall_top,
                claims=req.claims,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/claim-check")
    def claim_check(req: ClaimCheckRequest):
        return _claim_check(req.claim)

    @app.post("/api/p4-command")
    def p4_command(req: P4CommandRequest):
        return _p4_command(req)

    @app.post("/api/inspect-command")
    def inspect_command(req: InspectCommandRequest):
        return _inspect_command(req)

    @app.post("/api/cs-predict-command")
    def cs_predict_command(req: CSPredictCommandRequest):
        return _cs_predict_command(req)

    return app


app = create_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "cartigsfm-web requires uvicorn. Install with: pip install 'cartigsfm[web]' "
            "or pip install uvicorn"
        ) from exc
    host = os.environ.get("CARTIGSFM_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("CARTIGSFM_WEB_PORT", "8000")))
    uvicorn.run("cartigsfm_web.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
