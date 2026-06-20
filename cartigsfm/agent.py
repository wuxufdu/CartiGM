"""CartiGM Agent: tool wrappers + LLM-driven dispatcher.

Three first-class tools, all backed by the existing CartiGM package:

  * cartigm_score(genes)           -> top cartilage axes + safety + evidence + experiment
  * p4_project(h5ad_path, outdir)  -> P4 self-validation outdir
  * rag_evidence_lookup(query)     -> P6 RAG evidence cards + claim safety

The module exposes a stable Python API and a CLI. A small
keyword-based dispatcher ships first; an LLM-driven ReAct loop
(Qwen2.5-7B-Instruct native function calling) is layered on top.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .interpret import (
    apply_safety_filter,
    interpret_gene_list,
    interpret_p4_dir,
    render_json,
)
from .gsfm import tool_gsfm_score
from .scgpt import tool_scgpt_encode


# ---------------------------------------------------------------------------
# Tool schema (OpenAI-compatible function calling shape)
# ---------------------------------------------------------------------------

TOOL_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cartigm_score",
            "description": (
                "Score a list of HGNC gene symbols against the CartiGM 42-axis "
                "cartilage dictionary. Returns the top axes per layer, per-axis "
                "scores, safety classification, confidence, evidence snippets, "
                "and a suggested wet-lab / dry-lab validation experiment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "HGNC gene symbols (uppercase recommended).",
                    },
                    "top_per_layer": {
                        "type": "integer",
                        "description": "How many top axes to keep per dictionary layer.",
                        "default": 3,
                    },
                    "overall_top": {
                        "type": "integer",
                        "description": "How many top axes to keep in the overall ranking.",
                        "default": 5,
                    },
                },
                "required": ["genes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "p4_project",
            "description": (
                "Run P4 self-validation: project an h5ad or pseudobulk matrix "
                "onto all 42 cartilage_dictionary_v1 axes and write the standard "
                "P4 delivery (scores, top assignments, tissue summary, marker "
                "validation, Markdown report)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "h5ad_path": {
                        "type": "string",
                        "description": "Path to an .h5ad file. Requires anndata.",
                    },
                    "outdir": {
                        "type": "string",
                        "description": "Output directory for the P4 delivery.",
                    },
                    "sample_col": {"type": "string", "default": "sample"},
                    "tissue_col": {"type": "string", "default": "tissue"},
                    "cluster_col": {"type": "string", "default": "cluster"},
                },
                "required": ["h5ad_path", "outdir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_evidence_lookup",
            "description": (
                "Look up P6 RAG evidence for a cartilage axis or a free-text "
                "claim. Returns axis evidence cards, claim safety classification, "
                "and the relevant knowledge-base snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Axis id (e.g. 'functional_axis::Avascular_Antimineralization') or free-text topic.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gsfm_score",
            "description": (
                "GSFM-branch frozen feature extractor: score a marker list "
                "against the CartiGM 42-axis dictionary using a weighted "
                "Jaccard similarity on each axis's core_genes. Returns the "
                "top axes and (optionally) a single axis similarity number. "
                "Does NOT use bulk expression; complement the scGPT branch "
                "when only a gene list is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "HGNC gene symbols (uppercase recommended).",
                    },
                    "axis_id": {
                        "type": "string",
                        "description": "Optional axis id; if given, returns the similarity to that single axis.",
                    },
                    "top_n": {
                        "type": "integer",
                        "default": 5,
                    },
                },
                "required": ["genes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scgpt_encode",
            "description": (
                "scGPT-branch frozen feature extractor: encode an h5ad (or a "
                "raw genes x samples DataFrame) into per-cluster axis "
                "embeddings (cluster x axis_id). Each cluster is a column of "
                "mean core-gene expression. Use this branch whenever "
                "expression data is available; complement the GSFM branch "
                "which only takes a marker list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "h5ad_path": {
                        "type": "string",
                        "description": "Path to an .h5ad file. Requires anndata.",
                    },
                    "cluster_col": {"type": "string", "default": "cluster"},
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_cartigm_score(genes, *, top_per_layer: int = 3, overall_top: int = 5) -> Dict[str, Any]:
    """Wrap interpret_gene_list + apply_safety_filter into a tool result."""
    if isinstance(genes, str):
        genes = [g.strip() for g in re.split(r"[\s,;]+", genes) if g.strip()]
    result = interpret_gene_list(list(genes), top_per_layer=top_per_layer, overall_top=overall_top)
    safe = apply_safety_filter(result)
    return json.loads(render_json(safe))


def tool_p4_project(h5ad_path: str, outdir: str, *,
                    sample_col: str = "sample",
                    tissue_col: str = "tissue",
                    cluster_col: str = "cluster") -> Dict[str, Any]:
    """Run P4 projection on an h5ad and return the resulting outdir paths."""
    from .p4 import run_p4_project
    out = Path(outdir)
    outputs = run_p4_project(
        outdir=out,
        h5ad=h5ad_path,
        sample_col=sample_col,
        tissue_col=tissue_col,
        cluster_col=cluster_col,
    )
    return {
        "outdir": str(out),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }


def tool_rag_evidence_lookup(query: str) -> Dict[str, Any]:
    """Look up P6 RAG evidence for an axis id or free-text query."""
    from .assets import (
        load_axis_evidence_cards,
        load_claim_safety_classifier,
        load_rag_knowledge_base,
    )
    from .interpret import classify_claim
    out: Dict[str, Any] = {"query": query}
    q = (query or "").strip()
    ql = q.lower()
    cards = load_axis_evidence_cards() or []
    norm_cards: Dict[str, Dict[str, Any]] = {}
    for c in cards:
        if isinstance(c, dict) and c.get("axis_id"):
            norm_cards[str(c["axis_id"])] = c
    out["axis_cards"] = {q: norm_cards[q]} if q in norm_cards else {}
    if not out["axis_cards"]:
        out["axis_cards"] = {
            k: v for k, v in norm_cards.items()
            if ql in json.dumps(v, default=str).lower()
        }
    classifier = load_claim_safety_classifier() or []
    out["claim_matches"] = [
        dict(c) for c in classifier
        if ql and ql in json.dumps(c, default=str).lower()
    ][:10]
    if not out["claim_matches"] and q:
        out["claim_classification"] = classify_claim(q)
    kb = load_rag_knowledge_base() or {}
    atlas = kb.get("atlas_result_knowledge") or {}
    out["kb_snippets"] = {k: atlas[k] for k in list(atlas.keys())[:3]}
    return out


TOOL_DISPATCH = {
    "cartigm_score": tool_cartigm_score,
    "p4_project": tool_p4_project,
    "rag_evidence_lookup": tool_rag_evidence_lookup,
    "gsfm_score": tool_gsfm_score,
    "scgpt_encode": tool_scgpt_encode,
}


# ---------------------------------------------------------------------------
# Keyword-based dispatcher (baseline; bypasses LLM entirely)
# ---------------------------------------------------------------------------

_GENE_RE = re.compile(r"^[A-Z][A-Z0-9-]{1,15}$")


def _detect_genes(text: str) -> List[str]:
    """Return probable HGNC symbols in ``text`` (uppercased tokens)."""
    toks = re.split(r"[^A-Za-z0-9-]+", text)
    return [t.upper() for t in toks if t and _GENE_RE.match(t)]


def _keyword_route(query: str) -> Optional[str]:
    """Pick a tool by simple keyword match; None if ambiguous."""
    q = query.lower()
    if "scgpt" in q or "sc gpt" in q or "cluster encode" in q or "encode cluster" in q:
        return "scgpt_encode"
    if "gsfm" in q or "axis similarity" in q or "axis embedding" in q:
        return "gsfm_score"
    if "p4" in q or "h5ad" in q or "self-validation" in q or "self validation" in q:
        return "p4_project"
    if "evidence" in q or "rag" in q or "claim" in q or "safe" in q or "p6" in q:
        return "rag_evidence_lookup"
    if _detect_genes(query):
        return "cartigm_score"
    return None


def run_query_keyword(query: str) -> Dict[str, Any]:
    """Dispatch ``query`` to a single tool using keyword rules."""
    tool_name = _keyword_route(query)
    if tool_name is None:
        return {
            "tool": None,
            "result": None,
            "note": (
                "could not auto-route the query; please specify either a gene "
                "list (cartigm_score), an h5ad path (p4_project), or an axis "
                "id / claim (rag_evidence_lookup)."
            ),
        }
    fn = TOOL_DISPATCH[tool_name]
    if tool_name == "cartigm_score":
        result = fn(_detect_genes(query))
    elif tool_name == "gsfm_score":
        result = fn(_detect_genes(query))
    elif tool_name == "scgpt_encode":
        m = re.search(r"(\S+\.h5ad)", query)
        h5ad = m.group(1) if m else None
        if h5ad and Path(h5ad).exists():
            result = fn(h5ad_path=h5ad)
        else:
            return {
                "tool": tool_name, "result": None,
                "note": (
                    "scgpt_encode needs an existing .h5ad path in the query; "
                    "none was found, so no expression-based embedding was run."
                ),
            }
    elif tool_name == "p4_project":
        m = re.search(r"(\S+\.h5ad)", query)
        m_out = re.search(r"--outdir\s+(\S+)", query) or re.search(r"to\s+(\S+_p4\S*)", query)
        if not m:
            return {"tool": tool_name, "result": None,
                    "note": "p4_project needs an .h5ad path in the query"}
        h5ad = m.group(1)
        outdir = m_out.group(1) if m_out else str(Path(h5ad).with_suffix("").as_posix() + "_p4")
        if not Path(h5ad).exists():
            return {"tool": tool_name, "result": None,
                    "note": f"h5ad not found at {h5ad} (would error if executed)"}
        result = fn(h5ad, outdir)
    else:
        result = fn(query)
    return {"tool": tool_name, "result": result}


# ---------------------------------------------------------------------------
# LLM-driven ReAct loop (Qwen2.5 native function calling)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are CartiAgent, a careful cartilage-biology assistant backed by the "
    "CartiGM Python package. You must use one of the provided tools whenever "
    "the user asks about cartilage axes, gene interpretation, P4 self-"
    "validation, or evidence / claim safety. Never invent gene names, p-values, "
    "or sample sizes. If a tool returns safety_classification=NOT_SUPPORTED, "
    "report it as such and do not paraphrase as fact."
)


def _format_tool_result(name: str, payload: Any) -> str:
    """Stringify a tool result for the chat template."""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)[:6000]
    except Exception:
        return str(payload)[:6000]


def run_query_llm(query: str, model_path: str, *, max_iterations: int = 4,
                  max_new_tokens: int = 1024) -> Dict[str, Any]:
    """Drive Qwen2.5-7B-Instruct native function calling against the tool set."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", trust_remote_code=True,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    trace: List[Dict[str, Any]] = []
    for i in range(max_iterations):
        text = tok.apply_chat_template(
            messages, tools=TOOL_SCHEMA, tokenize=False,
            add_generation_prompt=True,
        )
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inp, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tok.eos_token_id, use_cache=True,
            )
        delta = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=False)
        trace.append({"step": i, "raw": delta[:1500]})
        messages.append({"role": "assistant", "content": delta})
        tool_calls = _parse_qwen_tool_calls(delta)
        if not tool_calls:
            return {"answer": delta, "trace": trace, "iterations": i + 1}
        for name, args in tool_calls:
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = {"error": f"unknown tool {name!r}"}
            else:
                try:
                    result = fn(**args)
                except TypeError as exc:
                    result = {"error": f"bad args for {name}: {exc}"}
            trace.append({"step": i, "tool": name, "args": args,
                          "result_preview": _format_tool_result(name, result)[:1200]})
            messages.append({
                "role": "tool", "name": name,
                "content": _format_tool_result(name, result),
            })
    return {"answer": "(max iterations reached)", "trace": trace, "iterations": max_iterations}


_QWEN_TOOL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL,
)


def _parse_qwen_tool_calls(delta: str) -> List[Any]:
    """Pull <tool_call>{...}</tool_call> JSON blobs out of a Qwen delta."""
    out: List[Any] = []
    for blob in _QWEN_TOOL_RE.findall(delta):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if "name" in obj and "arguments" in obj:
            args = obj["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            out.append((obj["name"], args or {}))
    return out
