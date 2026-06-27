"""CartiGSFM: cartilage-domain gene-set foundation model utilities.

Public API:
   load_dictionary(version="v0.3.1") -> dict
   load_specificity(version="v0.3.1") -> dict
   load_function_dictionary(version="v0.6.5") -> dict
   load_function_specificity(version="v0.6.5") -> dict
   score_query(genes, dictionary, anti_penalty=1.0) -> pandas.DataFrame
   score_function_query(genes, specificity, dictionary=None) -> pandas.DataFrame
   project_bulk(expr_df, dictionary, sample_col=None) -> pandas.DataFrame
   project_function_bulk(expr_df, specificity, dictionary=None) -> pandas.DataFrame
   load_cartilage_dictionary_v1() -> dict
   load_rag_knowledge_base(version="v1") -> dict
   interpret_gene_list(genes, ...) -> dict
   interpret_p4_dir(outdir, ...) -> dict
   interpret_p4_csv(path, ...) -> dict
   classify_claim(text) -> dict
   apply_safety_filter(interpretation, ...) -> dict
   render_markdown(interpretation) -> str
   render_json(interpretation) -> str
   axis_safety_class(axis) -> str
"""
from .dictionary import (
   load_alias_map,
   load_dictionary,
   load_function_dictionary,
   load_function_specificity,
   load_specificity,
   list_function_versions,
   list_versions,
   panel_genes,
   anti_panel_genes,
)
from .scoring import resolve_aliases, score_function_query, score_query
from .projection import project_bulk, project_dictionary_v1_bulk, project_function_bulk
from .p4 import run_p4_project
from .p4 import (
   _autostream_threshold_bytes,
   _resolve_chunk_size,
   auto_detect_obs_columns,
   pseudobulk_from_h5ad,
   pseudobulk_streaming,
)
from .assets import (
   device_summary,
   find_claim_safety,
   get_p9_adapter_path,
   list_cartilage_dictionary_versions,
   list_p9_versions,
   list_rag_versions,
   load_axis_evidence_cards,
   load_cartilage_dictionary_v1,
   load_claim_safety_classifier,
   load_p9_claim_safety_eval,
   load_p9_hallucination_audit,
   load_p9_model_card,
   load_p9_model_comparison,
   load_p9_p4_case_eval,
   load_p9_training_config,
   load_p9_training_report,
   load_prompt_templates,
   load_rag_knowledge_base,
   prefer_device,
   p9_is_adapter_available,
)
from . import interpret as _interpret
from .interpret import (
   apply_safety_filter,
   axis_safety_class,
   build_interpretation,
   classify_claim,
   interpret_gene_list,
   interpret_p4_csv,
   interpret_p4_dir,
   render_json,
   render_markdown,
)
from . import interpret as _interpret
from . import agent as _agent
from .interpret import (
   apply_safety_filter,
   axis_safety_class,
   build_interpretation,
   classify_claim,
   interpret_gene_list,
   interpret_p4_csv,
   interpret_p4_dir,
   render_json,
   render_markdown,
   suggested_validation_experiment,
)
from .agent import (
   TOOL_SCHEMA,
   run_query_keyword,
   run_query_llm,
   tool_cartigm_score,
   tool_p4_project,
   tool_rag_evidence_lookup,
)
from .gsfm import (
   gsfm_axis_embedding,
   gsfm_axis_similarity,
   gsfm_axis_table,
   gsfm_marker_axes,
   tool_gsfm_score,
)
from .scgpt import (
   scgpt_encode_cluster,
   scgpt_encode_dataframe,
   scgpt_encode_h5ad,
   tool_scgpt_encode,
)
from .ablation import render_ablation_markdown, run_ablation
from .ablation import (
   DEFAULT_CELLTYPE_AXIS_MAP,
   DEFAULT_REFUSAL_CLAIMS,
   DEFAULT_TISSUE_AXIS_MAP,
   run_ablation_real,
)

__version__ = "0.6.1"
__all__ = [
   "load_dictionary",
   "load_specificity",
   "load_function_dictionary",
   "load_function_specificity",
   "load_alias_map",
   "list_versions",
   "list_function_versions",
   "panel_genes",
   "anti_panel_genes",
   "resolve_aliases",
   "score_query",
   "score_function_query",
   "project_bulk",
   "project_function_bulk",
   "project_dictionary_v1_bulk",
   "run_p4_project",
   "auto_detect_obs_columns",
   "pseudobulk_from_h5ad",
   "pseudobulk_streaming",
   "_resolve_chunk_size",
   "_autostream_threshold_bytes",
   "list_cartilage_dictionary_versions",
   "load_cartilage_dictionary_v1",
   "list_rag_versions",
   "load_rag_knowledge_base",
   "load_axis_evidence_cards",
   "load_claim_safety_classifier",
   "load_prompt_templates",
   "find_claim_safety",
   "list_p9_versions",
   "load_p9_training_config",
   "load_p9_model_comparison",
   "load_p9_claim_safety_eval",
   "load_p9_hallucination_audit",
   "load_p9_p4_case_eval",
   "load_p9_training_report",
   "load_p9_model_card",
   "get_p9_adapter_path",
   "p9_is_adapter_available",
   "interpret_gene_list",
   "interpret_p4_dir",
   "interpret_p4_csv",
   "classify_claim",
   "apply_safety_filter",
   "build_interpretation",
   "axis_safety_class",
   "render_markdown",
   "render_json",
   "suggested_validation_experiment",
   "TOOL_SCHEMA",
   "run_query_keyword",
   "run_query_llm",
   "tool_cartigm_score",
   "tool_p4_project",
   "tool_rag_evidence_lookup",
   "gsfm_axis_embedding",
   "gsfm_axis_similarity",
   "gsfm_axis_table",
   "gsfm_marker_axes",
   "tool_gsfm_score",
   "scgpt_encode_cluster",
   "scgpt_encode_dataframe",
   "scgpt_encode_h5ad",
   "tool_scgpt_encode",
   "run_ablation",
   "run_ablation_real",
   "DEFAULT_TISSUE_AXIS_MAP",
   "DEFAULT_CELLTYPE_AXIS_MAP",
   "DEFAULT_REFUSAL_CLAIMS",
   "render_ablation_markdown",
]
from .annotate import (
    ACC_CHONDROCYTE_SUBTYPE_TO_V1,
    annotate_with_cartigsm,
    annotate_with_cellassign,
    annotate_with_celltypist,
    annotate_with_gptcelltype,
    annotate_with_marker_rule,
    annotate_with_scmap,
    annotate_with_scgpt,
    annotate_with_singler,
    annotate_with_symphony,
    build_gptcelltype_prompt,
    compare_annotations,
)
