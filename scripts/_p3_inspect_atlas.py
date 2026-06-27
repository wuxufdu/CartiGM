"""One-shot diagnostic: load EBR.h5ad and report structure + panel coverage."""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _metabolism_axis_panels as MP  # noqa: E402


def main() -> int:
    h5 = Path("F:/cartifm/outputs/EBR/EBR.h5ad")
    a = ad.read_h5ad(h5, backed="r")
    print("shape:", a.shape)
    print("obs cols:", list(a.obs.columns))
    print("var cols:", list(a.var.columns))
    print("layers:", list(a.layers.keys()))
    print("uns keys:", list(a.uns.keys())[:10])
    print("obsm keys:", list(a.obsm.keys()))
    print("X dtype:", getattr(a.X, "dtype", type(a.X)))
    vn = a.var_names.astype(str).str.upper()
    var_set = set(vn.tolist())
    print("first 10 var:", list(vn[:10]))

    print("--- panel coverage in EBR ---")
    for s in MP.axis_specs():
        panel_set = {g.upper() for g in s["panel"]}
        core_set = {g.upper() for (g, _) in s["core"]}
        overlap_panel = panel_set & var_set
        overlap_core = core_set & var_set
        print(
            "  {:28s}  core={:2d}/{:2d}  panel={:3d}/{:3d}".format(
                s["axis_id"], len(overlap_core), len(core_set),
                len(overlap_panel), len(panel_set),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
