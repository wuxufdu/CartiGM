"""Bundled CartiGSFM v1 dictionary and RAG resource loaders."""
from __future__ import annotations

import json
import os
import csv
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List


_DICTIONARY_VERSIONS = ("v1",)
_RAG_VERSIONS = ("v1",)
_P9_VERSIONS = ("v1",)


def _load_json(relative_path: str) -> Any:
    path = resources.files("cartigsfm").joinpath("resources", *relative_path.split("/"))
    return json.loads(path.read_text(encoding="utf-8"))


def _load_text(relative_path: str) -> str:
    path = resources.files("cartigsfm").joinpath("resources", *relative_path.split("/"))
    return path.read_text(encoding="utf-8")


def _load_tsv(relative_path: str) -> List[Dict[str, str]]:
    text = _load_text(relative_path)
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def list_cartilage_dictionary_versions() -> List[str]:
    """List bundled three-layer cartilage dictionary versions."""
    return list(_DICTIONARY_VERSIONS)


def load_cartilage_dictionary_v1() -> Dict[str, Any]:
    """Load the bundled three-layer ``cartilage_dictionary_v1.json`` resource."""
    return _load_json("dictionary_v1/cartilage_dictionary_v1.json")


def list_rag_versions() -> List[str]:
    """List bundled CartiGSFM-RAG resource versions."""
    return list(_RAG_VERSIONS)


def _check_rag_version(version: str) -> None:
    if version not in _RAG_VERSIONS:
        raise ValueError(f"Unknown RAG version {version!r}; known: {list(_RAG_VERSIONS)}")


def load_rag_knowledge_base(version: str = "v1") -> Dict[str, Any]:
    """Load the bundled P6 CartiGSFM-RAG knowledge base."""
    _check_rag_version(version)
    return _load_json(f"rag_{version}/p6_cartigsfm_knowledge_base.json")


def load_axis_evidence_cards(version: str = "v1") -> List[Dict[str, Any]]:
    """Load bundled P6 axis evidence cards."""
    _check_rag_version(version)
    return _load_json(f"rag_{version}/p6_axis_evidence_cards.json")


def load_claim_safety_classifier(version: str = "v1") -> List[Dict[str, Any]]:
    """Load bundled P6 claim safety classifier rules."""
    _check_rag_version(version)
    return _load_json(f"rag_{version}/p6_claim_safety_classifier.json")


def load_prompt_templates(version: str = "v1") -> Dict[str, Any]:
    """Load bundled P6 LLM/RAG prompt templates."""
    _check_rag_version(version)
    return _load_json(f"rag_{version}/p6_prompt_templates.json")


def find_claim_safety(claim: str, version: str = "v1") -> Dict[str, Any] | None:
    """Return the first bundled claim-safety entry matching ``claim`` exactly."""
    claim_norm = claim.strip().casefold()
    for entry in load_claim_safety_classifier(version):
        if str(entry.get("claim", "")).strip().casefold() == claim_norm:
            return entry
    return None


def list_p9_versions() -> List[str]:
    """List bundled P9 LoRA prototype metadata versions."""
    return list(_P9_VERSIONS)


def _check_p9_version(version: str) -> None:
    if version not in _P9_VERSIONS:
        raise ValueError(f"Unknown P9 version {version!r}; known: {list(_P9_VERSIONS)}")


def load_p9_training_config(version: str = "v1") -> Dict[str, Any]:
    """Load bundled P9 LoRA prototype training configuration metadata."""
    _check_p9_version(version)
    return _load_json(f"p9_{version}/config/p9_lora_training_config.json")


def load_p9_model_comparison(version: str = "v1") -> List[Dict[str, str]]:
    """Load bundled P9 four-system model comparison metrics."""
    _check_p9_version(version)
    return _load_tsv(f"p9_{version}/tsv/p9_model_comparison_results.tsv")


def load_p9_claim_safety_eval(version: str = "v1") -> List[Dict[str, str]]:
    """Load bundled P9 claim-safety evaluation metrics."""
    _check_p9_version(version)
    return _load_tsv(f"p9_{version}/tsv/p9_claim_safety_eval.tsv")


def load_p9_hallucination_audit(version: str = "v1") -> List[Dict[str, str]]:
    """Load bundled P9 hallucination audit rows."""
    _check_p9_version(version)
    return _load_tsv(f"p9_{version}/tsv/p9_hallucination_audit.tsv")


def load_p9_p4_case_eval(version: str = "v1") -> List[Dict[str, str]]:
    """Load bundled P9 P4-shaped case evaluation rows."""
    _check_p9_version(version)
    return _load_tsv(f"p9_{version}/tsv/p9_p4_case_eval.tsv")


def load_p9_training_report(version: str = "v1") -> str:
    """Load bundled P9 training report markdown text."""
    _check_p9_version(version)
    return _load_text(f"p9_{version}/docs/P9_CARTIGSFM_LLM_TRAINING_REPORT.md")


def load_p9_model_card(version: str = "v1") -> str:
    """Load bundled P9 model card markdown text."""
    _check_p9_version(version)
    return _load_text(f"p9_{version}/docs/P9_MODEL_CARD.md")


def get_p9_adapter_path(adapter_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the preferred local P9 adapter directory path.

    The package does not bundle adapter weights. Resolution order:
    explicit ``adapter_dir``; ``CARTIGSFM_P9_ADAPTER_DIR``; common workspace
    artifact locations next to the current working directory.
    """
    if adapter_dir:
        return Path(adapter_dir)
    env_path = os.environ.get("CARTIGSFM_P9_ADAPTER_DIR")
    if env_path:
        return Path(env_path)
    candidates = [
        Path.cwd() / "cartigsfm_p9_lora_training_delivery" / "adapter",
        Path.cwd() / "review_p9_delivery" / "cartigsfm_p9_lora_training_delivery" / "adapter",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def p9_is_adapter_available(adapter_dir: str | os.PathLike[str] | None = None) -> bool:
    """Return whether a local P9 LoRA adapter appears available."""
    path = get_p9_adapter_path(adapter_dir)
    return (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )
def prefer_device(explicit: Optional[str] = None) -> str:
    """Pick the best available torch device, preferring CUDA.

    The user wants all training to run on GPU when possible. This helper
    resolves to ``"cuda:0"`` when a CUDA device is reachable, falls back
    to MPS on Apple Silicon, and finally to ``"cpu"``. The ``explicit``
    argument wins, so callers can still force ``"cpu"`` in tests.
    """
    if explicit:
        return str(explicit)
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def device_summary(device: Optional[str] = None) -> Dict[str, Any]:
    """Tiny summary dict for the chosen device, useful in tool notes."""
    chosen = prefer_device(device)
    info: Dict[str, Any] = {"device": chosen}
    try:
        import torch
        info["torch_version"] = getattr(torch, "__version__", "")
        if chosen.startswith("cuda"):
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info
