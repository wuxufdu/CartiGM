# CartiGSFM - Cartilage Gene-Set Foundation Model (work in progress)

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
