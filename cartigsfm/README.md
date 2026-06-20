# cartigsfm

Cartilage-domain gene-set foundation model utilities.

## Install (development mode)

```bash
pip install -e .
# or, using a tsinghua mirror in mainland China:
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Quick start (Python)

```python
import cartigsfm

# Load production dictionary
d = cartigsfm.load_dictionary("v0.3.1")
print(cartigsfm.list_versions())
print(cartigsfm.list_function_versions())

# Score a DEG list against every cgrm subtype
genes = open("my_oa_deg.txt").read().split()
df = cartigsfm.score_query(genes, d, anti_penalty=1.0)
print(df.head())

# Score a marker list against function axes, including the
# avascular / anti-mineralization cartilage identity axis.
fn_spec = cartigsfm.load_function_specificity("v0.6.5")
fn_dict = cartigsfm.load_function_dictionary("v0.6.5")
avam_genes = "MGP CNMD LECT1 TIMP3 ANKH ENPP1 TNFRSF11B FRZB SOX9 ACAN".split()
avam_genes = cartigsfm.resolve_aliases(avam_genes, cartigsfm.load_alias_map())
fn_df = cartigsfm.score_function_query(avam_genes, fn_spec, fn_dict)
print(fn_df.head())

# Project bulk RNA-seq onto subtypes
import pandas as pd
expr = pd.read_csv("bulk.tsv", sep="\t", index_col=0)  # genes x samples
proj = cartigsfm.project_bulk(expr, d, anti_lambda=0.5)
proj.to_csv("projection.tsv", sep="\t", index=False)

# Project bulk or pseudobulk RNA-seq onto function axes
fn_proj = cartigsfm.project_function_bulk(expr, fn_spec, fn_dict)
fn_proj.to_csv("function_projection.tsv", sep="\t", index=False)
```

## CLI

```bash
cartigsfm versions
cartigsfm score --query my_oa_deg.txt --version v0.3.1 --top 10
cartigsfm score --kind function --query avam_genes.txt --function-version v0.6.5 --top 10
cartigsfm score --kind both --query marker_genes.txt --version v0.3.1 --function-version v0.6.5 --top 20
cartigsfm project --matrix bulk.tsv --version v0.3.1 --out projection.tsv \
    --gene-col gene_symbol --samples sample1,sample2,sample3
cartigsfm project --kind function --matrix bulk.tsv --function-version v0.6.5 \
    --gene-col gene_symbol --out function_projection.tsv
cartigsfm project --kind both --matrix pseudobulk.tsv --version v0.3.1 \
    --function-version v0.6.5 --gene-col gene_symbol --out combined_projection.tsv
```

For single-cell h5ad validation, export cluster pseudobulk and run subtype plus
function projection in one step:

```bash
python3 scripts/62_project_h5ad_pseudobulk.py \
    --h5ad auricular.h5ad \
    --cluster-key leiden \
    --kind both \
    --out-prefix data/processed/scrna/auricular_cartigsfm
```

This writes:

- `*_cluster_pseudobulk.tsv`: gene x cluster mean expression matrix.
- `*_cluster_meta.tsv`: cluster cell counts.
- `*_projection.tsv`: full subtype/function projection scores.
- `*_top_assignments.tsv`: top subtype/function per cluster.
- `*_figure_summary.tsv`: one-row-per-cluster summary with subtype top hit,
  function top hit, AvAm score, AvAm rank, and support counts.
- `*_report.md`: human-readable summary.

`score --kind subtype` ranks cgrm single-cell-derived cartilage subtype panels.
`score --kind function` ranks functional gene-set axes. `score --kind both`
returns both channels in one table. HGNC alias resolution is enabled by default;
use `--no-alias` only when symbols have already been normalized.

`project --kind function` applies the same function axes to expression matrices
after gene-wise z-scoring. This is intended for external bulk RNA-seq and
single-cell pseudobulk validation, including direct AvAm module scoring in
healthy auricular, nasal, and costal cartilage.

## Production version

`v0.3.1` is the data-driven dictionary built from the integrated 77-sample
single-cell atlas; it currently delivers OA top-10 = 0.417 on 12 paired
holdout x direction runs (vs cart-Enrichr 0.242, broad Enrichr 0.025; see
`data/processed/BASELINE_COMPARISON_VERSIONS.md`). v0.2 (manual reviewer
panel) is preserved for backward compatibility.

`v0.6.5` is the current function dictionary. It includes
`Avascular_Antimineralization`, a literature-prior cartilage identity axis
represented by genes such as `MGP`, `CNMD`, `TNMD`, `TIMP3`, `TNFRSF11B`,
`ANKH`, `ENPP1`, `FRZB`, `SOX9`, and `ACAN`.

## P15: cluster-level annotation with cross-method comparison

`cartigsfm.annotate` ships one entry point that runs a query h5ad through
seven backends and writes a pairwise agreement matrix. Real CellTypist,
the bundled deterministic scGPT proxy, the CartiGM P4 dictionary
projection, the marker-only dictionary re-projection, a GPTcelltype
prompt builder, and four R-only placeholders (SingleR / scmap / Symphony
/ CellAssign) are all surfaced through the same function signature.

```bash
# annotate EBR using every working backend, with the bundled v1
# dictionary as the ground truth and acc.chongdrocyte_subtype as the
# CellTypist reference labels
python -m cartigsfm annotate --method all \
    --p4-outdir F:/cartifm/outputs/P4_EBR_real_validation \
    --query-h5ad F:/cartifm/outputs/EBR/EBR.h5ad \
    --reference-h5ad F:/cartifm/acc.h5ad \
    --reference-label-col chongdrocyte_subtype \
    --cluster-col leiden_res0_5 \
    --out F:/cartifm/CartiGM/reports/P15_ANNOTATION_COMPARISON.md \
    --max-reference-cells 20000 \
    --device cuda:0

# single-method quick run (e.g. only the bundled scGPT proxy)
python -m cartigsfm annotate --method scgpt \
    --query-h5ad F:/cartifm/outputs/EBR/EBR.h5ad \
    --cluster-col leiden_res0_5 \
    --per-cluster-tsv F:/cartifm/outputs/EBR/ebr_scgpt.tsv
```

The `--method all` run writes:

- `annotation_comparison_long.tsv` long-form `(method, cluster, label)`
- `annotation_comparison_wide.tsv` per-cluster wide table
- `annotation_comparison_pairwise.tsv` pairwise agreement matrix
- `annotation_comparison_summary.json` n_clusters_per_method + agreement
- `P15_ANNOTATION_COMPARISON.md` human-readable report (also at
  `reports/P15_ANNOTATION_COMPARISON.md` for the canonical EBR run)
- `annotation_gptcelltype_prompts.tsv` prompt-only fallback for GPTcelltype

EBR does not ship with curator cell-type labels, so the comparison is
reported as **pairwise agreement with the CartiGM `cartigsm` branch as
the reference**. For a true head-to-head benchmark against a labelled
atlas, run `cartigsfm annotate ... --reference-h5ad <labelled_atlas>`
on the labelled atlas instead of on EBR.

### GPU note

All training-capable entry points accept `--device` (default: `cuda:0`
if a CUDA device is reachable, `mps` on Apple Silicon, `cpu` otherwise).
`cartigsfm.prefer_device()` and `cartigsfm.device_summary()` are the
underlying helpers. CellTypist v1.7.1 uses scikit-learn's `SGDClassifier`
under the hood, which is CPU-only; the device argument is recorded in
the returned note but training stays on CPU. For a GPU-runnable
cell-typing path, load real scGPT-human weights and call them directly;
the bundled scGPT branch is a deterministic proxy marked `fallback=True`.
