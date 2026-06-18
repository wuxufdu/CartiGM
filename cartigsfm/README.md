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
