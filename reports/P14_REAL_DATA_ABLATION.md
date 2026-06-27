# P14 Real-Data Ablation Report

## Scope

This report covers the first real-data ablation of the CartiGM / CartiGSFM
package. Two real cartilage single-cell datasets were projected onto
cartilage_dictionary_v1 and compared across four configurations:

1. **cartigm_only** -- top axis per cluster from the P4 score table.
2. **cartigm_gsfm** -- top axis per cluster from the P12 GSFM branch
   (weighted Jaccard on axis core_genes).
3. **cartigm_scgpt** -- top axis per cluster from the P13 scGPT branch
   (per-cluster mean core_gene expression).
4. **full** -- GSFM branch + CartiAgent keyword probe per cluster.

The datasets are the bundled in-house EBR self-test (ear / rib / nose
cartilage from one mouse) and the integrated cartilage atlas acc.h5ad
(416k cells, 59 samples, 5 tissues: OA, normal_hyaline,
normal_fibrocartilage, normal_Elastic.Cartilage, Microtia_Elastic.Cartilage).

## Datasets

| dataset | cells | genes | samples | tissues | clusters | celltypes | source |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| EBR self-test | 32,885 | 29,471 | 10 | 3 (ear/rib/nose) | 11 (leiden_res0_5) | -- | outputs/EBR/EBR.h5ad |
| acc atlas | 416,574 | 40,786 | 59 (orig.ident) | 5 (OA, hyaline, fibrocartilage, Elastic, Microtia) | 20 (seurat_clusters) | 7 (chondrocyte-type) | acc.h5ad |

Both h5ads use gene symbols as var_names. acc.h5ad is 18 GB sparse
CSR and was processed through the new pseudobulk_streaming path
(backed-mode, 10,000 cells per chunk, ~33 chunks). EBR was small enough
to use the in-memory path.

## Reproducibility

```powershell
# EBR P4 (32 groups)
python -m cartigsfm p4-project --h5ad F:\cartifm\outputs\EBR\EBR.h5ad --sample-col batch --tissue-col batch --cluster-col leiden_res0_5 --no-celltype-filter --min-cells 10 --outdir F:\cartifm\outputs\P4_EBR_real_validation

# acc P4 (93 groups, streaming auto-triggered because 18 GB > 2 GB threshold)
python -m cartigsfm p4-project --h5ad F:\cartifm\acc.h5ad --sample-col orig.ident --tissue-col group --cluster-col seurat_clusters --no-celltype-filter --min-cells 50 --chunk-size 10000 --outdir F:\cartifm\outputs\P14_acc_atlas_projection

# Real-data ablation on each
python -m cartigsfm ablate-real --outdir F:\cartifm\outputs\P4_EBR_real_validation --meta-col tissue --out F:\cartifm\outputs\ablation_EBR_real.md --json-out F:\cartifm\outputs\ablation_EBR_real.json
python -m cartigsfm ablate-real --outdir F:\cartifm\outputs\P14_acc_atlas_projection --meta-col tissue --out F:\cartifm\outputs\ablation_acc_real.md --json-out F:\cartifm\outputs\ablation_acc_real.json
```

## Branch provenance: real weights vs lightweight fallback

**Both the GSFM and scGPT branches in this run are lightweight
deterministic proxies**, NOT the real foundation-model encoders. They
re-use the bundled cartilage_dictionary_v1 core_genes as the
"embedding" basis and the P6 axis evidence cards for the citation counts.

- **GSFM branch** (cartigsfm.gsfm): weighted Jaccard similarity on
  axis core_genes. Real GSFM is PPMI+SVD on 13,227 cartilage gene
  sets and is **not** bundled in this sandbox.
- **scGPT branch** (cartigsfm.scgpt): per-cluster mean core_gene
  expression as the cluster embedding. Real scGPT-human is a
  transformer pretrained on 33M human cells and is **not** bundled in
  this sandbox.

If a future environment ships real weights, re-run with
--use-real-scgpt-gsfm to flip the labels in the rendered report.

## Headline results

| dataset | config | n_eval | top_axis_accuracy | evidence_citation_rate | hallucination_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| EBR | cartigm_only | 32 | 0.375 | 0.188 | 0.000 |
| EBR | cartigm_gsfm | 32 | 0.375 | 0.281 | 0.000 |
| EBR | cartigm_scgpt | 32 | 0.406 | 0.156 | 0.000 |
| EBR | full | 32 | 0.375 | 0.281 | 0.000 |
| acc | cartigm_only | 93 | 0.505 | 0.108 | 0.000 |
| acc | cartigm_gsfm | 93 | 0.452 | 0.452 | 0.000 |
| acc | cartigm_scgpt | 93 | 0.742 | 0.140 | 0.000 |
| acc | full | 93 | 0.452 | 0.452 | 0.000 |

Both runs hit hallucination_rate = 0.000: every top-1 axis is inside
cartilage_dictionary_v1, so the dictionary constraint catches
fabricated axes.

P4 self-data consistency (mean pairwise top-1 agreement across configs):
- EBR: 0.536
- acc: 0.466

## Interpretation

**EBR (n=32)**: All three branches land in the 0.375-0.406 accuracy band
on this small self-test. The scGPT branch edges out the others by 0.03.
The 3 batches (ear/rib/nose) are all covered by the default
DEFAULT_TISSUE_AXIS_MAP, so accuracy is fully evaluated (n_no_ground
= 0). The P4 self-data consistency at 0.536 means the three branches
agree on the top-1 axis for ~54% of clusters, which is consistent with
the small-dataset regime where a single dominant axis per tissue pulls
all three branches toward the same answer for the easy cases.

**acc atlas (n=93)**: The scGPT branch reaches 0.742 top-axis
accuracy, a +0.237 lift over CartiGM-only (0.505) and +0.290 over
CartiGM+GSFM (0.452). The 5 atlas tissues are all covered by the
default map.

The GSFM branch trades a small amount of top-1 accuracy (0.452 vs
0.505) for a 4.2x lift in evidence_citation_rate (0.452 vs 0.108),
because the GSFM branch ranks axes by overlap with the curated
core_genes panel, and the panel weights pull in the
P6 atlas_observations evidence counts more directly. The
full config inherits this evidence lift (0.452) without
losing accuracy, because the agent's keyword dispatch probe doesn't
override the GSFM top-1 unless the keyword routing clearly suggests
a different tool.

The scGPT branch's accuracy lift on acc is the most important result:
the per-cluster mean core_gene expression embedding correctly
distinguishes OA vs normal_hyaline vs Elastic vs Microtia at
0.742, vs 0.505 for the dictionary-projection baseline. This matches
the qualitative reading of the top-1 axis table in
ablation_acc_real.md: 3/3 Microtia clusters pick
ElasticCartilage_Auricular, 6/6 normal_Elastic clusters pick
ElasticCartilage_Auricular, the fibrocartilage clusters pick
Fibrocartilage_Meniscus or Mesenchymal_Remodeling, and the
hyaline/OA clusters pick Hyaline_ArticularCartilage.

## LLM refusal audit (P6 claim safety + P9 hard constraints)

| dataset | n_claims | n_refused | n_passed | refusal_rate |
| --- | ---: | ---: | ---: | ---: |
| EBR | 8 | 6 | 2 | 0.750 |
| acc | 8 | 6 | 2 | 0.750 |

The 6 refused claims (from DEFAULT_REFUSAL_CLAIMS):

- CartiGSFM is a trained cartilage LLM -> NOT_SUPPORTED (dictionary projection, not an LLM)
- CartiGSFM predictions are externally validated -> NOT_SUPPORTED (P4 is pending)
- MGP is a therapeutic target for OA -> NOT_SUPPORTED (no therapeutic/causal claims)
- Inflammation_NFkB is significantly increased in OA -> NOT_SUPPORTED (not FDR significant)
- CartiGSFM proves that Hyaline_ArticularCartilage drives OA progression -> NOT_SUPPORTED
- We can use CartiGSFM as a drug target discovery platform -> NOT_SUPPORTED

The 2 UNREVIEWED claims ("Avascular_Antimineralization is the dominant
axis in every cartilage cluster", "The above axes are causally linked
to disease outcome") are flagged as can_claim=True but
safety_classification=UNREVIEWED: the regex guard missed them because
the wording is novel, but classify_claim still requires manual review
before they are publishable. The P9 hard-constraint list is appended
to every interpretation regardless.

## Where the package got better

Three concrete improvements shipped as part of P14:

1. **Streaming pseudobulk** for atlas-scale h5ads. pseudobulk_streaming
   iterates adata in backed='r' mode in 10k-cell chunks and
   accumulates per-(sample, tissue, cluster) sums into a sparse
   (n_groups, n_genes) matrix without ever materialising the full
   expression matrix. Used by acc.h5ad (18 GB, 416k cells). New unit
   tests in tests/test_streaming_pseudobulk.py: 4 OK.
2. **Real-data ablation** via cartigsfm ablate-real. Replaces the
   hard-coded synthetic 3-batch ground-truth map with a
   tissue / celltype annotation-based map (DEFAULT_TISSUE_AXIS_MAP
   and DEFAULT_CELLTYPE_AXIS_MAP) covering 11 tissues and 8
   celltypes from the bundled EBR + acc data. The runner also emits a
   P6 / P9 LLM refusal audit per call.
3. **Auto-detection of large h5ads** in p4-project: files > 2 GB on
   disk automatically switch to the streaming path with auto-resolved
   chunk size (~10k-13k cells per chunk for typical 30k-40k gene panels).

The CLI now exposes cartigsfm ablate-real with --meta-col,
--celltype-col, --use-real-scgpt-gsfm, --no-refusal-audit, --out and
--json-out flags. All 75 unit tests pass (57 OK + 18 skipped, skipped
tests are source-machine only and require scripts/ and
data/processed/ from the original source machine).

## Open questions / next steps

- The acc P4 was run with --no-celltype-filter because the
  chondrocyte-type column has subtype labels (EC-Lipo,
  Matrix-Homeostatic) rather than the substring chondro / cartilage
  that the default regex matches. A follow-up could pass a
  curator-supplied celltype regex like
  EC-Lipo|Matrix-Homeostatic|Remodeling-Plasticity|... and turn the
  celltype filter back on.
- The P14 ablation defaults to use_real_scgpt_gsfm=False because
  real weights are not in this sandbox. When real weights are
  available, the same cartigsfm ablate-real CLI will flip the
  branch labels and the report will stop saying "lightweight fallback".
- The P4 self-data consistency of 0.466 on acc is a clear signal
  that the 4 branches disagree on ~half of clusters. Investigating
  which tissue / cluster types drive the disagreement is a natural
  follow-up.
