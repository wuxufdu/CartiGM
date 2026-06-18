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
from .assets import (
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
    p9_is_adapter_available,
)

__version__ = "0.4.0"
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
]
