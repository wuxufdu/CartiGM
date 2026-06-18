"""Load CartiGSFM subtype and function dictionaries."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

# Resolve from environment or fall back to repo-relative path
import os

_DEFAULT_PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC = Path(os.environ.get("CARTIGSFM_PROC_DIR", str(_DEFAULT_PROC)))


_VERSION_ALIASES = {
    "v0.5": "v0.5",
    "v050": "v0.5",
    "v0.4": "v0.4",
    "v0.3.2": "v0.3.2",
    "v032": "v0.3.2",
    "v0.3.1": "v0.3.1",
    "v031": "v0.3.1",
    "v0.3": "v0.3",
    "v0.2": "v0.2",
    "v0.1": "v0.1",
}

_FUNCTION_VERSION_ALIASES = {
    "v0.6.5": "v0.6.5",
    "v065": "v0.6.5",
    "v0.6.4": "v0.6.4",
    "v064": "v0.6.4",
    "v0.6.3": "v0.6.3",
    "v063": "v0.6.3",
    "v0.6.2": "v0.6.2",
    "v062": "v0.6.2",
    "v0.6.1": "v0.6.1",
    "v061": "v0.6.1",
    "v0.6": "v0.6",
    "v060": "v0.6",
    "v0.5": "v0.5",
    "v050": "v0.5",
    "v0.2": "v0.2",
    "v020": "v0.2",
}


def list_versions() -> List[str]:
    """List dictionary versions available on disk."""
    out = []
    for stem in PROC.glob("cgrm_v*_subtype_dictionary.json"):
        v = stem.stem.split("_")[1]
        out.append(v)
    return sorted(set(out))


def list_function_versions() -> List[str]:
    """List function dictionary versions available on disk."""
    out = []
    for stem in PROC.glob("v*_function_dictionary.json"):
        out.append(stem.stem.replace("_function_dictionary", ""))
    return sorted(set(out))


def _resolve_version(version: str) -> str:
    if version in _VERSION_ALIASES:
        return _VERSION_ALIASES[version]
    raise ValueError(f"Unknown cgrm version {version!r}; known: {sorted(_VERSION_ALIASES)}")


def _resolve_function_version(version: str) -> str:
    if version in _FUNCTION_VERSION_ALIASES:
        return _FUNCTION_VERSION_ALIASES[version]
    raise ValueError(
        f"Unknown function version {version!r}; known: {sorted(_FUNCTION_VERSION_ALIASES)}"
    )


def load_dictionary(version: str = "v0.3.1") -> Dict:
    """Load the cgrm subtype dictionary JSON for the given version."""
    v = _resolve_version(version)
    path = PROC / f"cgrm_{v}_subtype_dictionary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; available: {list_versions()}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_specificity(version: str = "v0.3.1") -> Dict:
    """Load the cgrm subtype specificity JSON for the given version."""
    v = _resolve_version(version)
    path = PROC / f"cgrm_{v}_subtype_specificity.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_function_dictionary(version: str = "v0.6.5") -> Dict:
    """Load the function dictionary JSON for the given version."""
    v = _resolve_function_version(version)
    path = PROC / f"{v}_function_dictionary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; available: {list_function_versions()}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_function_specificity(version: str = "v0.6.5") -> Dict:
    """Load the function specificity JSON for the given version."""
    v = _resolve_function_version(version)
    path = PROC / f"{v}_function_specificity.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_alias_map() -> Dict[str, str]:
    """Load HGNC alias-to-current-symbol mappings when bundled."""
    path = PROC / "hgnc_alias_to_current.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def panel_genes(dictionary: Dict, subtype: str) -> Dict[str, float]:
    """Return panel_weights dict for one subtype (auto-prefix with cgrm:: if needed)."""
    if subtype not in dictionary and f"cgrm::{subtype}" in dictionary:
        subtype = f"cgrm::{subtype}"
    return dictionary[subtype].get("panel_weights", {})


def anti_panel_genes(dictionary: Dict, subtype: str) -> Dict[str, float]:
    """Return anti_panel_weights dict for one subtype."""
    if subtype not in dictionary and f"cgrm::{subtype}" in dictionary:
        subtype = f"cgrm::{subtype}"
    return dictionary[subtype].get("anti_panel_weights", {})
