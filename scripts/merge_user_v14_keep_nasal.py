"""Merge user-supplied dictionary v1.4 with the bundled v1.2 by keeping
the Nasal_Septum_Cartilage axis we added on top of v1.1.

Inputs:
  USER:    F:/cartifm/cartilage_dictionary_v1.json (v1.4)
  BUNDLED: F:/cartifm/CartiGM/cartigsfm/resources/dictionary_v1/cartilage_dictionary_v1.json (v1.2 with nasal axis)

Output:
  - Bundled file overwritten in-place with the merged dictionary (version "1.5").
  - Backup of the prior bundled file at *.pre_v1_5.bak (only created if not present).
  - Updated changelog block describing the merge.

The merge takes:
  - cell_subtype layer:  USER (atlas-rebuilt markers + literature-aligned rename + aliases)
  - tissue_developmental_state: USER's 3 axes + bundled's Nasal_Septum_Cartilage
  - functional_axis: USER (unchanged from v1.1)
  - top-level: USER (panel_qc, evidence_policy, candidate_axes), but we re-add
    a changelog entry and recompute axis_count_total like v1.2.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

USER_PATH = Path(r"F:\cartifm\cartilage_dictionary_v1.json")
BUNDLED_PATH = Path(r"F:\cartifm\CartiGM\cartigsfm\resources\dictionary_v1\cartilage_dictionary_v1.json")
BACKUP = BUNDLED_PATH.with_suffix(".json.pre_v1_5.bak")
NEW_VERSION = "1.5"


def find_axis(axes, axis_id):
    for a in axes:
        if a.get("axis_id") == axis_id:
            return a
    return None


def main() -> None:
    user = json.loads(USER_PATH.read_text(encoding="utf-8"))
    bundled = json.loads(BUNDLED_PATH.read_text(encoding="utf-8"))

    print(f"USER    : v{user.get('version')} ({user.get('generated_at')})")
    print(f"BUNDLED : v{bundled.get('version')} ({bundled.get('generated_at')})")

    # Locate the nasal septum axis we want to preserve from bundled.
    bundled_tds = bundled["layers"]["tissue_developmental_state"]["axes"]
    nasal_axis = find_axis(bundled_tds, "tissue_developmental_state::Nasal_Septum_Cartilage")
    if nasal_axis is None:
        raise RuntimeError("Nasal_Septum_Cartilage not found in bundled dictionary; nothing to preserve.")
    print(f"Preserving axis: {nasal_axis['axis_id']} (status={nasal_axis.get('status')})")

    # Start from a deep copy of USER so we keep its panel_qc / candidate_axes / evidence_policy etc.
    merged = json.loads(json.dumps(user))

    # Re-attach the nasal septum axis into tissue_developmental_state.
    merged_tds = merged["layers"]["tissue_developmental_state"]
    # Drop any pre-existing nasal axis (defensive; user v1.4 doesn't have one)
    merged_tds["axes"] = [a for a in merged_tds.get("axes", [])
                          if a.get("axis_id") != nasal_axis["axis_id"]]
    merged_tds["axes"].append(nasal_axis)
    merged_tds["count"] = len(merged_tds["axes"])

    # Refresh per-layer counts and the running total.
    total = 0
    for k, v in merged["layers"].items():
        if isinstance(v, dict) and "axes" in v:
            v["count"] = len(v["axes"])
            total += v["count"]
    merged["axis_count_total"] = total

    # Bump version + generated_at; keep description, append a brief note.
    merged["version"] = NEW_VERSION
    merged["generated_at"] = time.strftime("%Y-%m-%d")
    desc = merged.get("description") or ""
    note = " v1.5 merges user-supplied v1.4 (atlas-rebuilt cell_subtype markers + literature-aligned names + aliases) with the v1.2 nasal-septum tissue axis."
    if note.strip() not in desc:
        merged["description"] = desc + note

    # Carry-forward changelog from bundled (USER v1.4 had none).
    cl = merged.get("changelog")
    if cl is None:
        cl = bundled.get("changelog") or []
    cl.append({
        "version": NEW_VERSION,
        "generated_at": merged["generated_at"],
        "changes": [
            "Merged user-supplied dictionary v1.4 (atlas-rebuilt cell_subtype markers + literature-aligned name_en/name_cn with aliases + expanded panel_qc) with the v1.2 bundled dictionary.",
            "Preserved tissue_developmental_state::Nasal_Septum_Cartilage axis added in v1.2 (cell-level Mann-Whitney U on EBR.h5ad log1p_norm).",
            "Layer counts after merge: cell_subtype=10, tissue_developmental_state=4, functional_axis=29 (43 total).",
            "v1.4-only top-level fields (panel_qc with artifact_gene_filters / cell_subtype_marker_policy / cell_subtype_name_version) carried in unchanged.",
        ],
        "scripts": ["scripts/merge_user_v14_keep_nasal.py"],
    })
    merged["changelog"] = cl

    # Sanity: marker / anti overlap should remain zero across the whole dictionary.
    def gene_set(value):
        out = set()
        if isinstance(value, list):
            for it in value:
                if isinstance(it, str):
                    out.add(it)
                elif isinstance(it, dict):
                    g = it.get("gene") or it.get("symbol")
                    if g:
                        out.add(g)
        elif isinstance(value, dict):
            out.update(value.keys())
        return out

    overlap_axes = []
    for lk, layer in merged["layers"].items():
        for a in layer["axes"]:
            markers = gene_set(a.get("core_genes")) | gene_set(a.get("panel_genes"))
            anti = gene_set(a.get("anti_genes"))
            ovl = markers & anti
            if ovl:
                overlap_axes.append((lk, a.get("axis_id"), sorted(ovl)))
    if overlap_axes:
        for o in overlap_axes:
            print("  marker/anti overlap:", o)
        raise RuntimeError("Marker/anti overlap detected after merge; aborting before write.")
    print("marker/anti overlap (whole dict): 0 axes")

    # Backup before write.
    if not BACKUP.exists():
        shutil.copy2(BUNDLED_PATH, BACKUP)
        print(f"backed up bundled -> {BACKUP.name}")
    else:
        print(f"backup already present, leaving in place: {BACKUP.name}")

    BUNDLED_PATH.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote merged dictionary: {BUNDLED_PATH.stat().st_size} bytes")
    print(f"  version={merged['version']} axis_count_total={merged['axis_count_total']}")
    for k, v in merged["layers"].items():
        ids = [a.get("axis_id") for a in v["axes"]]
        print(f"  layer {k}: {len(ids)} axes")
        for aid in ids:
            print(f"    - {aid}")


if __name__ == "__main__":
    main()
