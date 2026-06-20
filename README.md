# CartiGSFM - Cartilage Gene-Set Foundation Model (work in progress)

## cartigsfm package v0.6.1 (this directory)

The `cartigsfm/` Python package in this repository is the runtime artifact;
the `data/processed/` corpus described further down is the offline research
layer that feeds it. After `pip install -e .` you get a CLI plus a small
inference stack:

- **Three-layer dictionary v1.8.6 (53 axes)** at
  `cartigsfm/resources/dictionary_v1/cartilage_dictionary_v1.json`
  - 10 `cell_subtype` axes (Effector_Metabolic / Progenitor / Homeostatic /
    Hypoxic / Metabolic_Stress / Inflammatory_Response /
    Prehypertrophic_Matrix / Fibrocartilage / Superficial_Zone /
    Reparative_Stress Chondrocytes), rebuilt from a balanced acc_new
    wilcoxon DE pass (v1.8.4), refit on in-domain EBR DE for the 7
    EBR-present axes (v1.8.5), and cross-injected with mutual anti
    markers (v1.8.6 P-D triangulation).
  - 4 `tissue_developmental_state` axes including
    `Nasal_Septum_Cartilage`, built from EBR nose-vs-(ear+rib) GPU
    Mann-Whitney U with 10 anatomical anchors + 10 data-driven markers.
  - 39 `functional_axis` entries spanning cartilage development,
    signalling, OA biology, and a 10-axis metabolism block (Glycolysis,
    OxPhos, TCA, PPP, FAO, Lipogenesis, Cholesterol, Lipid_Droplet,
    Glutaminolysis, Mitochondrial_Biogenesis).
- **Bundled cs_classifier_v1** at
  `cartigsfm/resources/cs_classifier_v1/`:
  `classifier.pt` (v1, 0.85M params, 384/192 MLP) +
  `classifier_v2.pt` (v2, 1.7M params, 512/256 MLP with Gaussian-noise +
  MixUp augmentation) + `hvg_genes.tsv` (2000 HVG basis on
  acc_new var ∩ EBR var).
  - EBR within-cluster cell-level holdout: v1 76.6% / v2 76.9% /
    **ensemble 77.6%**; within-cluster cluster top-1: v1 76.2% /
    ensemble 76.2%.
  - EBR leave-batch-out (held-out tissue) mean cell accuracy: v1 48.8% /
    **v2 57.5%**. See [reports/P_F_TRAINING_REPORT.md](reports/P_F_TRAINING_REPORT.md).
- **CLI commands** (all also importable from `cartigsfm.*`):
  ```
  cartigsfm dictionary-v1                                  # axis summary
  cartigsfm score --query genes.txt                        # subtype/function score from a gene list
  cartigsfm function-score --genes COL2A1,ACAN,SOX9
  cartigsfm project --matrix bulk.tsv --out scores.tsv     # bulk projection
  cartigsfm cs-predict --h5ad your.h5ad --out preds.tsv \  # P-F per-cell prediction
      --layer log1p_norm --mode ensemble --device cuda
  cartigsfm p4-project --h5ad your.h5ad --outdir P4_out \
      --sample-col sample --tissue-col tissue --cluster-col cluster
  cartigsfm annotate --p4-outdir P4_out --method all       # P15 cross-method
  cartigsfm interpret --p4-outdir P4_out                   # evidence-constrained
  cartigsfm scgpt-pretrain ...                             # P16 small-transformer MLM
  cartigsfm train-fusion ...                               # P17 fusion ablation
  cartigsfm p6-info | rag-info | claim-check | p9-info | p9-eval
  cartigsfm agent                                          # LLM tool-use over CartiGM
  ```
- **Tests**: `python -m unittest discover -s tests` -> 78 unit tests pass
  (skipped=18 for R-backend / heavy-GPU paths). The bundled checkpoint and
  HVG basis are exercised by `tests/test_cs_classifier.py`.
- **Changelog (most recent first)**:
  - v0.6.1 - cs_classifier v2 + softmax-average ensemble; LBO
    cross-tissue 48.8% -> 57.5%; within-cluster cell 76.6% -> 77.6%.
  - v0.6.0 - dictionary v1.8.6 + bundled cs_classifier_v1 (v1) +
    `cartigsfm cs-predict` CLI; new modules ablation / agent / annotate /
    fusion / gsfm / interpret / scgpt / scgpt_pretrain.
  - v0.5.0 - tissue_developmental_state::Nasal_Septum_Cartilage axis.
  - v0.4.0 - initial bundled three-layer dictionary + P4/P6/P9 RAG /
    LoRA metadata loaders.

The text below this section describes the offline research corpus
(`data/processed/`) that the dictionary was mined from; it is preserved
for traceability and is not required to use the installed package.

---

A cartilage-domain foundation model in the spirit of GSFM
([Patterns 2026, doi:10.1016/j.patter.2026.101565](https://doi.org/10.1016/j.patter.2026.101565)),
specialised to chondrocyte / cartilage / OA biology. This repo is the
public-data half of the project: a curated, HGNC-normalised human
cartilage gene-set corpus, a 45-category subtype + function dictionary
mined from it, and a small two-stage embedding model on top. A
one-command interface ingests user scRNA cluster markers and rebuilds
the dictionary; the scRNA-derived half is being integrated on a
separate machine and merges in via that interface.

## Latest version: v0.8.3

### What changed since v0.8

- **OA dictionary rebuilt** (`scripts/22_repair_oa_dictionary.py`): the
  v0.6 `Osteoarthritis` consensus came from DisGeNET / ChEA OA-risk
  sets and held only 27 genes, much thinner than its functional
  neighbours. v0.6.1 unions that with anchor-driven OA-cartilage
  dysregulation sets (IL1/TNF cytokines, MMPs, ADAMTS, COL2A1/ACAN/COMP
  loss, COL10A1/RUNX2 hypertrophy, CDKN1A/2A senescence). Consensus
  27 -> 105; specificity markers 12 -> 50, top-weighted by PTGS2 / MMP3
  / MMP1 / IL1B / IL1A / IL1RN / MMP2 / TNF.
- **Log-space hypergeometric enrichment**: bulk DEG queries (4k+ genes)
  used to saturate `-log10(p)` at the cap (=12) for nearly every
  category, leaving combined ranking up to embedding alone. v0.8.2
  switches to log-space, raises the cap to 60, and tightens the
  normaliser. Canonical 13/13 + stress 3/3 unchanged.
- **Leave-one-GSE-out external validation** (`scripts/21`): retrains
  PPMI+SVD and the contrastive projection on a corpus that has the
  target GSE removed, then classifies the held-out study's union DEG
  query. Reports an `fn::Osteoarthritis` rank, an
  `fn::Inflammation_NFkB` rank, and a constellation rank over the OA
  biology axes (Inflammation_NFkB, IL1, TNF, Senescence, Apoptosis,
  Hypoxia, Ferroptosis, MMP, ADAMTS, Hypertrophy, ECM_Organization).
- **Holdout matrix**: `data/processed/HOLDOUT_MATRIX.md` aggregates two
  cross-study runs:

  | holdout    | study type                    | OA-up constellation rank | OA-dn constellation rank |
  |------------|-------------------------------|--------------------------|--------------------------|
  | GSE114007  | OA vs Normal cartilage        | 10 (fn::Hypoxia)         | 2 (fn::Ferroptosis)      |
  | GSE236122  | synovial fibroblast stress    | 3 (fn::IL1_Signaling)    | 7 (fn::IL1_Signaling)    |

  See `data/processed/holdout_GSE114007/REPORT.md` and
  `data/processed/holdout_GSE236122/REPORT.md` for full top-15 tables
  and interpretation.

| layer | what | files |
|---|---|---|
| corpus | 13,227 human cartilage gene sets (RummaGEO + Rummagene + Enrichr + 15 curated paper sets) | `data/processed/cartilage_genesets_v03.parquet` |
| dictionary | 19 subtypes + 26 functions, specificity-ranked top-50 markers per category, shared cartilage/ECM core separated | `data/processed/v0.4_subtype_v05_specificity.json`, `data/processed/v0.6_function_specificity.json` |
| provenance | per-category set_id evidence files with source / term / DOI | `data/processed/provenance/INDEX.tsv` + 45 per-category TSVs |
| model v1 (default) | PPMI+SVD 128-dim gene+set embeddings | `data/processed/embeddings/` |
| model v2 (opt-in) | supervised-contrastive projection trained on the 45 dictionary categories | `data/processed/embeddings_v08/` |
| classifier | marker list -> ranked subtypes/functions, overlap + embedding + signature boost | `scripts/16_classify_marker_list.py` |
| scRNA ingest | one-command rebuild from a user TSV of cluster markers | `scripts/19_seed_from_scrna.py` |
| HGNC alias map | 58,409 alias -> current symbol mappings used everywhere | `data/processed/hgnc_alias_to_current.json` |

Earlier versions (v0.1-v0.6) are still on disk for audit; see
`data/processed/DATASET_CARD.md` for the full version history.

## Python package resources

`cartigsfm` 0.4.0 bundles lightweight manuscript-facing resources:

- `cartilage_dictionary_v1.json`: three-layer cartilage dictionary with 10 cell-subtype axes, 3 tissue/developmental-state axes, and 29 functional axes.
- P6 CartiGSFM-RAG JSON resources: knowledge base, axis evidence cards, prompt templates, and claim-safety classifier.
- P9 CartiGSFM-LoRA prototype metadata: training config, model card, report text, and evaluation tables. Adapter weights are not bundled in the base package; use `CARTIGSFM_P9_ADAPTER_DIR` or a local P9 delivery folder.

Useful Python entry points:

```python
import cartigsfm

dictionary = cartigsfm.load_cartilage_dictionary_v1()
knowledge_base = cartigsfm.load_rag_knowledge_base()
claim = cartigsfm.find_claim_safety(
    "CartiGSFM is a trained cartilage large language model (LLM)"
)
p9 = cartigsfm.load_p9_training_config()
p9_metrics = cartigsfm.load_p9_model_comparison()
adapter_available = cartigsfm.p9_is_adapter_available()
interp = cartigsfm.interpret_gene_list(["MGP", "CNMD", "LECT1", "TIMP3",
                                         "ANKH", "ENPP1", "TNFRSF11B",
                                         "FRZB", "SOX9", "ACAN"])
safe = cartigsfm.apply_safety_filter(interp)
print(cartigsfm.render_markdown(safe))
```

Useful CLI commands:

```bash
cartigsfm dictionary-v1 --show-axes
cartigsfm rag-info
cartigsfm claim-check --claim "CartiGSFM is a trained cartilage large language model (LLM)"
cartigsfm p9-info
cartigsfm p9-eval
cartigsfm p9-adapter-path --check
```

P9 improves language-layer behavior (unsupported-claim refusal and evidence-citation style) but does not change gene-set scoring or single-cell projection scores.

For independent in-house single-cell validation, `p4-project` projects either
an `.h5ad` file or an existing pseudobulk matrix onto all 42
`cartilage_dictionary_v1` axes:

```bash
cartigsfm p4-project \
  --h5ad self_data.h5ad \
  --sample-col sample \
  --tissue-col tissue \
  --cluster-col cluster \
  --celltype-col celltype \
  --outdir cartigsfm_p4_independent_validation_delivery
```

If `scanpy/anndata` is not available, first create a genes x sample-cluster
pseudobulk matrix and run:

```bash
cartigsfm p4-project \
  --pseudobulk p4_self_sample_cluster_pseudobulk.tsv \
  --meta p4_self_sample_cluster_meta.tsv \
  --outdir cartigsfm_p4_independent_validation_delivery
```

The command writes all three-layer scores, top assignments, tissue summaries,
marker validation tables, and a conservative P4 report.

## Evidence-constrained interpretation

`cartigsfm interpret` turns gene-list scores or P4 score tables into
evidence-constrained biological interpretations. Every output is
anchored in `cartilage_dictionary_v1`, the P6 CartiGSFM-RAG claim
safety classifier, and the P9 hard constraints from the LoRA model
card, so overclaim about external validation, LLM training, or
therapeutic targets is blocked at the source.

Three input modes are supported:

- `--mode genes` (with `--genes` or `--gene-file`) scores a gene list
  against the curated `core_genes` of every v1 axis. Useful for
  one-off queries such as "what does this set of 10 AvAm genes look
  like in the dictionary".
- `--mode p4-dir` reads `tsv/p4_sample_cluster_three_layer_scores.tsv`
  from a previous `p4-project` outdir and groups the highest scoring
  sample-cluster per axis.
- `--mode p4-csv` reads an arbitrary long-form score table with
  `axis_id` and `score` columns (and optional `layer`, `sample`).

Every interpretation surfaces, per layer, the safety classification of
each axis (`PENDING_INDEPENDENT_VALIDATION` for production axes,
`SUPPLEMENTARY_ONLY` for reference, `EXPLORATORY` for
literature-prior). It also reports, for each axis, the recommended and
forbidden wording pulled from the bundled evidence cards.

Free-text claims can be audited against the same P6 / P9 safety
metadata with `--claim` (repeatable). Each claim is classified as one
of `NOT_SUPPORTED`, `UNREVIEWED`, or matched against an exact entry in
`p6_claim_safety_classifier.json`. Claims that fail the safety filter
are listed under `Cannot Claim` and trigger a warning.

Example: a 10-gene AvAm panel

```bash
cartigsfm interpret \
  --mode genes \
  --genes "MGP,CNMD,LECT1,TIMP3,ANKH,ENPP1,TNFRSF11B,FRZB,SOX9,ACAN" \
  --claim "MGP is a therapeutic target for OA" \
  --claim "CartiGSFM is a trained cartilage LLM" \
  --claim "Avascular_Antimineralization is increased in OA" \
  --format markdown
```

Example: a real P4 self-validation outdir

```bash
cartigsfm p4-project \
  --h5ad self_data.h5ad \
  --sample-col sample --tissue-col tissue --cluster-col cluster \
  --outdir cartigsfm_p4_independent_validation_delivery

cartigsfm interpret \
  --mode p4-dir \
  --input cartigsfm_p4_independent_validation_delivery \
  --format markdown \
  --out p4_interpretation.md
```

The same flow is available programmatically via
`interpret_gene_list`, `interpret_p4_dir`, `interpret_p4_csv`,
`classify_claim`, and `apply_safety_filter`.

## CartiAgent: LLM-driven tool use (P11)

`cartigsfm.agent` is the small LLM agent that fronts the package. It
exposes a stable 5-tool schema (`TOOL_SCHEMA` in Python) and a CLI:

| tool | what it does |
| --- | --- |
| `cartigm_score(genes, top_per_layer, overall_top)` | score a gene list against the 42 axes, return safety + evidence + experiment |
| `p4_project(h5ad_path, outdir, ...)` | run a P4 self-validation projection and return the file paths |
| `rag_evidence_lookup(query)` | P6 RAG evidence cards + claim safety for an axis or free-text topic |
| `gsfm_score(genes, axis_id=None, top_n=5)` | GSFM branch: gene-set / axis similarity (weighted Jaccard on `core_genes`) |
| `scgpt_encode(h5ad_path, cluster_col="cluster")` | scGPT branch: per-cluster axis embedding from h5ad or DataFrame |

The keyword dispatcher (`run_query_keyword`) routes by simple tokens
("p4" / "h5ad" / "self-validation" -> p4_project, "gsfm" /
"axis similarity" -> gsfm_score, "scgpt" / "cluster encode" ->
scgpt_encode, "evidence" / "rag" / "claim" -> rag_evidence_lookup,
otherwise detected gene symbols -> cartigm_score). The LLM ReAct
loop (`run_query_llm`) drives a Qwen2.5-7B-Instruct model with
OpenAI-compatible native function calling, max 4 iterations.

```bash
# keyword mode (default; no GPU needed)
python -m cartigsfm agent --query "MGP CNMD ACAN enriched in cartilage"
python -m cartigsfm agent --query "gsfm score MGP CNMD ACAN"
python -m cartigsfm agent --query "scgpt encode my_clusters.h5ad"
python -m cartigsfm agent --query "evidence for Avascular_Antimineralization"

# LLM mode (requires a local Qwen2.5-7B-Instruct directory)
python -m cartigsfm agent \
    --query "Which cartilage axis best explains this AvAm panel?" \
    --mode llm \
    --model F:\cartifm\models\Qwen2.5-7B-Instruct
```

The agent deliberately **never invents** gene names, p-values,
sample sizes, or therapeutic conclusions. Every tool result is
run through `apply_safety_filter` (P9 hard-constraint list + P6
claim-safety classifier) before the LLM sees it.

## GSFM / scGPT fusion (P12 + P13)

The package ships two **frozen** branches alongside the existing
CartiGM projection:

- `cartigsfm.gsfm` is the gene-set / axis embedding branch. It
  scores a marker list against every v1 axis with a weighted
  Jaccard coefficient on each axis's `marker_weights`, returns the
  top axes, and exposes `gsfm_axis_embedding(axis_id)` as a
  JSON-serializable feature dict. Use it whenever only a marker
  list is available (no expression matrix).
- `cartigsfm.scgpt` is the cluster expression encoder. It accepts
  an h5ad (with `anndata`) or a raw genes-by-samples DataFrame,
  builds a genes x cluster pseudobulk, and returns a per-cluster
  axis embedding (mean expression of axis core_genes per cluster)
  plus a per-cluster `top_axis_id`. Use it whenever expression
  data is available.

Both branches share the bundled v1 dictionary, the P6 axis
evidence cards, and the P9 hard-constraint list, so the agent
cannot make a fabricated-axis claim. Neither branch calls the
LLM; both are deterministic feature extractors ready to be
swapped for real GSFM PPMI+SVD or scGPT-human weights when
available in the sandbox.

```python
import cartigsfm

# GSFM branch
top = cartigsfm.gsfm_marker_axes(
    ["MGP", "CNMD", "TIMP3", "ANKH", "ENPP1",
     "TNFRSF11B", "FRZB", "SOX9", "ACAN"],
    top_n=5,
)
avam = cartigsfm.gsfm_axis_embedding(
    "functional_axis::Avascular_Antimineralization"
)

# scGPT branch
result = cartigsfm.scgpt_encode_dataframe(
    expr_df, gene_col="gene",
)
for cluster_summary in result["per_cluster_summary"]:
    print(cluster_summary["cluster"],
          cluster_summary["top_axis_id"],
          cluster_summary["top_score"])
```

The four-way ablation runner
(`cartigsfm.run_ablation(outdir)` + `cartigsfm.render_ablation_markdown`)
compares CartiGM-only, CartiGM+GSFM, CartiGM+scGPT, and the full
stack (all three + the LLM keyword dispatch) on any P4 outdir. It
reports top-axis accuracy against a per-tissue ground-truth set,
evidence citation rate, hallucination rate, and P4 self-data
consistency. CLI:

```bash
python -m cartigsfm ablation \
    --outdir cartigsfm_p4_independent_validation_delivery \
    --format markdown > reports/ablation_$(date +%Y%m%d).md
```

See `reports/P10_P13_INTEGRATION.md` for the full model
architecture, the ablation table, and the limitations of the
current proxies.



## Real-data P4 + ablate-real (P14)

The package can be driven against a real cartilage single-cell experiment end-to-end.
The P4 projection turns an h5ad or pseudobulk into a three-layer
CartiGM score table, and ablate-real compares the four
configurations (CartiGM only, CartiGM + GSFM, CartiGM + scGPT, full
stack + LLM agent) on top of that P4 outdir using tissue /
celltype-annotation-based ground truth and a P6 / P9 LLM refusal
audit.

```powershell
# 1. Inspect an h5ad before projecting; auto-detect sample / tissue / cluster columns
python -m cartigsfm inspect-h5ad --h5ad F:\cartifm\acc.h5ad

# 2. Project onto cartilage_dictionary_v1 (auto-streams if h5ad > 2 GB)
python -m cartigsfm p4-project --h5ad F:\cartifm\acc.h5ad --sample-col orig.ident --tissue-col group --cluster-col seurat_clusters --no-celltype-filter --min-cells 50 --chunk-size 10000 --outdir F:\cartifm\outputs\P14_acc_atlas_projection

# 3. Real-data ablation with annotation-based ground truth + LLM refusal audit
python -m cartigsfm ablate-real --outdir F:\cartifm\outputs\P14_acc_atlas_projection --meta-col tissue --celltype-col cluster --out F:\cartifm\outputs\ablation_acc_real.md --json-out F:\cartifm\outputs\ablation_acc_real.json
```

p4-project writes the standard P4 delivery (pseudobulk, meta,
three-layer scores, top assignments, tissue summary, marker
validation, Markdown report) and ablate-real adds the four-way
metrics + per-config top-1 + LLM refusal table.

Streaming pseudobulk is the default for any h5ad > 2 GB on disk; the
chunk size is auto-resolved from the gene panel size. Override with
--streaming / --no-streaming / --chunk-size. The full
real-data report is at reports/P14_REAL_DATA_ABLATION.md.

## Branch provenance: real weights vs lightweight fallback

The cartigsfm.gsfm (P12) and cartigsfm.scgpt (P13) branches in
this sandbox are lightweight deterministic proxies, not the real
foundation-model encoders. Both use the bundled
cartilage_dictionary_v1 core_genes as the embedding basis.

- cartigsfm.gsfm: weighted Jaccard on axis core_genes. Real
  GSFM is PPMI + SVD on 13,227 cartilage gene sets; weights are not
  bundled in this sandbox.
- cartigsfm.scgpt: per-cluster mean core_gene expression as
  the cluster embedding. Real scGPT-human is a transformer pretrained
  on 33M human cells; weights are not bundled in this sandbox.

ablate-real always reports which branch is a fallback. To use real
weights when they become available, pass --use-real-scgpt-gsfm; the
labels flip automatically and the report stops saying "lightweight
fallback".

## Tests

```bash
python -m unittest discover -s tests
```

On a fresh GitHub clone the suite runs as **17 OK / 18 skip / 0 fail / 0 error** (35 tests in total). The 18 skipped tests are guarded with `unittest.skipUnless` and require files that are intentionally not bundled in the public package because they live on the source machine only:

- `data/processed/cgrm_v0.3.1_subtype_dictionary.json`
- `data/processed/v0.6.5_function_specificity.json`
- `scripts/63_summarize_projection_for_figures.py`
- `scripts/64_plot_cross_tissue_projection.py`
- `scripts/65_annotate_marker_table.py`
- `review_p9_delivery/cartigsfm_p9_lora_training_delivery/adapter/` (LoRA adapter weights, ~16 MB)

The 4 always-bundled tests cover the package version, the v0.3.1 unknown-version error path, the three-layer `cartilage_dictionary_v1` shape, and the P6 RAG / claim-safety resources. The 13 always-bundled tests in `tests/test_cartigsfm_interpret.py` cover the new evidence-constrained interpretation module: axis safety classification, claim classifier (exact-match + regex overclaim guard), gene-list scoring, P4 score-CSV interpretation, JSON and Markdown renderers.

On the source machine (the original Windows workstation that produced the handoff), the same suite runs as **35 OK / 0 skip / 0 fail / 0 error** because all of the source-only files are present. The source-machine 22/22 figure in `NEXT_CODEX_HANDOFF.md` therefore remains the source of truth for the legacy tests, while the GitHub clone gives a deterministic, runnable baseline.

With the P10-P13 fusion pass, the suite expands to **53 OK / 18 skip / 0 fail / 0 error** (71 tests in total): the original 4 + 18 interpret + 13 agent + 11 GSFM + 7 scGPT. The 18 skip count is unchanged (still source-machine files only). The 4-way ablation runner (`cartigsfm.ablation`) is exercised via the CLI rather than unit tests because it depends on a real P4 outdir.

## Benchmark

`scripts/17_evaluate_modelv1.py` runs three classifiers (overlap,
embedding-only cosine, combined) against 13 held-out canonical marker
queries plus 3 stress queries with deliberately reduced overlap to
the dictionary.

|                          | v0.7.1 (PPMI+SVD) | v0.8 (contrastive) |
|--------------------------|-------------------|--------------------|
| canonical overlap        | 12/13             | 12/13              |
| canonical embedding-only | 13/13             | 13/13              |
| canonical combined       | 13/13             | 13/13              |
| stress overlap           | 2/3               | 2/3                |
| stress embedding-only    | 0/3               | 2/3                |
| stress combined          | 3/3               | 3/3                |

v0.8 lifts the stress embedding-only axis from 0/3 to 2/3 (Mechano
low-overlap and Hypoxia downstream now resolved by embedding alone)
without regressing any other axis. Senescence SASP is still pulled by
Autophagy on the embedding-only axis but combined still recovers via
the SIGNATURE_BOOST table in `scripts/16`.

v0.7.1 PPMI+SVD remains the default; v0.8 is opt-in via a single env
var because it requires PyTorch.

## Quick start

Classify a marker list:

```bash
# default v0.7.1 embedding (no torch dependency)
python3 scripts/16_classify_marker_list.py PRG4 CILP CLU S100A4 TNC

# v0.8 contrastive embedding
CARTI_EMB_DIR=data/processed/embeddings_v08 \
    python3 scripts/16_classify_marker_list.py PRG4 CILP CLU S100A4 TNC

# read from a file, top-5 only, function categories
python3 scripts/16_classify_marker_list.py --file my_markers.txt --kind function --topk 5
```

Evaluate end-to-end against the held-out and stress queries:

```bash
python3 scripts/17_evaluate_modelv1.py
CARTI_EMB_DIR=data/processed/embeddings_v08 python3 scripts/17_evaluate_modelv1.py
```

Older hypergeometric-only query interface (still useful for batch reports):

```bash
python3 scripts/06_query_marker_list.py CLUSTER_MARKERS.txt --top-k 25 --out results/cluster_X
```

## Plug your scRNA clusters in

Format your cluster markers as a TSV with header `cluster_id`,
`cell_type_label`, `markers` (comma / pipe / semicolon / whitespace
separated). Optional columns: `doi`, `study`, `function_label`. Then:

```bash
# full rebuild: scRNA -> v0.8 corpus -> dictionary -> embeddings
python3 scripts/19_seed_from_scrna.py path/to/markers.tsv

# stop after corpus merge (handy while debugging label normalisation)
python3 scripts/19_seed_from_scrna.py path/to/markers.tsv --skip-pipeline
```

Labels are normalised to the 45 canonical keywords through
`SUBTYPE_LABEL_MAP` and `FUNCTION_LABEL_MAP` in `scripts/19`; unmapped
labels are flagged with a warning instead of silently misrouted. The
pipeline rebuilds the v0.8 corpus, re-seeds the dictionary by reusing
`scripts/09`, then re-runs scripts 11, 13, 10, 14, 15 to refresh the
specificity tables and gene/set embeddings.

## How it was built

1. v0.1 - keyword search across RummaGEO, Rummagene and Enrichr; anchor
   seeding from 60 cartilage genes (`scripts/01-04`, 13,212 sets).
2. v0.2 - hypergeometric feature-based expansion of v0.1 dictionary
   entries (`scripts/07`).
3. v0.3 - 15 curated paper-attributed marker sets (Ji 2019, Decker 2017,
   Mizuhashi 2018, Tam 2020, Sun 2020) seed the formerly-empty subtypes;
   expansion runs again (`scripts/08, 09`). Corpus reaches 13,227 sets.
4. v0.4 - TF-IDF specificity replaces absolute-frequency consensus
   (`scripts/10`): per-category top-50 markers ranked by
   `freq * log2((freq + alpha) / (bg_freq + alpha))`, shared-core gene
   list separated.
5. v0.5 - IHH, FGF, Hypoxia rebuilt from narrow canonical anchors after
   v0.4 specificity exposed label pollution; MSC_Progenitor trimmed
   from 10,324-gene blow-out to focused surface markers (`scripts/11`).
6. v0.6 - ADAMTS, Ferroptosis, Mechanotransduction filled
   (`scripts/13`); MMP/ADAMTS family-prefix disambiguation enforced
   (`scripts/14`). 13/13 canonical validation.
7. v0.7 / v0.7.1 - PPMI+SVD gene + set embeddings (`scripts/15`);
   classifier with HGNC alias resolver (`scripts/18`) and
   pathway-signature TF boost. Canonical 13/13, stress 3/3 combined.
8. v0.8 - supervised-contrastive projection trained on the dictionary
   (`scripts/20`). Stress embedding-only 0/3 -> 2/3. scRNA ingest
   end-to-end pipeline (`scripts/19`).

`data/processed/DATASET_CARD.md` carries the long-form version history,
schemas and build commands.

## Pipeline at a glance

```
scripts/
  01-04   public-data fetch + HGNC normalisation
  05-09   subtype/function dictionary scaffolding + curated seeding
  10-14   specificity, anchor repairs, function fills, family disambiguation
  15      PPMI+SVD gene + set embeddings (v0.7 baseline)
  16      marker-list classifier (overlap + embedding + boost)
  17      held-out + stress benchmark
  18      HGNC alias map builder
  19      scRNA cluster-marker ingest + one-command v0.8 rebuild
  20      supervised-contrastive embedding (v0.8 model layer, opt-in)
```

## Roadmap

Next, in order:

1. Re-run `scripts/19` once real scRNA cluster markers from the other
   machine arrive; produce a v0.8.1 corpus + dictionary + retrained
   contrastive embeddings.
2. Extend the supervised-contrastive head to a true GSFM-style masked
   gene modelling pre-training pass on the corpus, then fine-tune the
   45-category supervised head on top. This is the "foundation model"
   step proper.
3. External validation on OA bulk cohorts (GSE114007, GSE57218,
   GSE169077, etc.).

## Caveats

- Human-only by design; ortholog mapping (mouse, etc.) is deferred.
- Some Enrichr libraries pulled by the anchor-gene step are TF-target
  or miRNA-target rather than functional pathways; they are kept for
  downstream multi-view modelling but excluded from pure "function" use
  by `scripts/14`.
- Senescence specificity is thin in the public corpus; v0.8 still
  cannot separate Senescence from Autophagy on the embedding-only
  axis. Combined axis recovers via signature boost; a richer Senescence
  cluster from the user scRNA atlas should fix this in v0.8.1.
