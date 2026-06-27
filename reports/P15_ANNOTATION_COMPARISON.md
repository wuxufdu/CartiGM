# P15 - Annotation Back-end Comparison on EBR

Query: EBR.h5ad (32,885 cells, 11 leiden clusters)  
Reference: acc.h5ad (416,574 cells; `chongdrocyte_subtype` -> 10 cartilage v1 axes)  
CartiGSFM package: 0.4.0  

EBR does not ship with curator cell-type labels, so the comparison below is
reported as **pairwise agreement with the CartiGM `cartigsm` branch as the
reference**. Where multiple labels agree on a cluster we treat that as a
consensus annotation; where they disagree the row is flagged.

## 1. Per-method availability

| method | available | n_clusters | source |
| --- | --- | --- | --- |
| cartigsm | true | 11 | CartiGSFM v0.4.0 P4 + cartilage_dictionary_v1 |
| marker_rule | true | 11 | CartiGSFM v0.4.0 P4 pseudobulk + cartilage_dictionary_v1 |
| scgpt | true | 11 | bundled lightweight proxy (fallback=True) |
| celltypist | true | 11 | CellTypist 1.7.1 trained on acc.chongdrocyte_subtype (50k cells) |

R-only placeholders (SingleR / scmap / Symphony / CellAssign) and
GPTcelltype (requires `OPENAI_API_KEY`) are reported separately in section 4.

## 2. Per-cluster wide table

| cluster | cartigsm | celltypist | marker_rule | scgpt |
| --- | --- | --- | --- | --- |
| 0 | Maturation_Matrix | Hypoxia_Adaptive | Maturation_Matrix | Maturation_Matrix |
| 1 | Hypoxia_Metabolic_Stress | Hypoxia_Metabolic_Stress | Hypoxia_Metabolic_Stress | EC_Lipo_Plasticity |
| 10 | Hypoxia_Metabolic_Stress | Fibro_Matrix | Hypoxia_Metabolic_Stress | Hypoxia_Adaptive |
| 2 | Stress_IEG | Hypoxia_Adaptive | Stress_IEG | Stress_IEG |
| 3 | Hypoxia_Metabolic_Stress | EC_Lipo_Plasticity | Hypoxia_Metabolic_Stress | EC_Lipo_Plasticity |
| 4 | EC_Lipo_Plasticity | Hypoxia_Metabolic_Stress | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 5 | Stress_IEG | PRG4_Interface | Stress_IEG | Maturation_Matrix |
| 6 | Maturation_Matrix | Hypoxia_Adaptive | Maturation_Matrix | Homeostatic_Matrix |
| 7 | Stress_IEG | Fibro_Matrix | Stress_IEG | Fibro_Matrix |
| 8 | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling |
| 9 | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling |

## 3. Pairwise agreement matrix

| method | cartigsm | celltypist | marker_rule | scgpt |
| --- | --- | --- | --- | --- |
| cartigsm | 1.00 | 0.27 | 1.00 | 0.45 |
| celltypist | 0.27 | 1.00 | 0.27 | 0.36 |
| marker_rule | 1.00 | 0.27 | 1.00 | 0.45 |
| scgpt | 0.45 | 0.36 | 0.45 | 1.00 |

Cell-by-cell counts per pair: agreement is computed on the intersection of
clusters where both methods produced a label, so the denominator can differ
from the headline cluster count when one method skipped a cluster.

## 4. Other backends (R-only and prompt-only)

### 4.1 SingleR

- available: **False**
- note: singler is an R/Bioconductor package; not callable from this Python entry point. To enable it, install the R package 'SingleR' and bridge via rpy2.

### 4.2 scmap

- available: **False**
- note: scmap is an R/Bioconductor package; not callable from this Python entry point. To enable it, install the R package 'scmap' and bridge via rpy2.

### 4.3 symphony

- available: **False**
- note: symphony is an R/Bioconductor package; not callable from this Python entry point. To enable it, install the R package 'symphony' and bridge via rpy2.

### 4.4 cellassign

- available: **False**
- note: cellassign is an R/Bioconductor package; not callable from this Python entry point. To enable it, install the R package 'cellassign' and bridge via rpy2.

### 4.5 GPTcelltype (OpenAI prompt-only fallback)

- status: prompts built, no model call (OPENAI_API_KEY not set)
- n_prompts: **11**
- prompt TSV: F:\cartifm\outputs\annotation_P15\annotation_gptcelltype_prompts.tsv
- example prompt (cluster 0):

```
You are a careful cartilage-biology curator. Below is a list of marker genes ranked by specificity, followed by a closed-set of candidate cell-type labels. Choose exactly one label from the candidate set that best matches the marker list. Reply with a single line: label: <chosen_label> then a short rationale.
Tissue / context: mixed (ear/nose/rib)
Top markers (n=20): DCLK1, SLC7A2, BCAT1, IGFBP7, COL2A1, SPARC, SFRP2, HOXC8, LOXL4, PPP1R14C, GREM1, CTNND2, NDUFA4L2, DDIT4L, COLGALT2, SCRG1, CDA, CDK6, FGFBP2, STC2 or N/A
Candidate labels: Homeostatic_Matrix, Hypoxia_Adaptive, EC_Lipo_Plasticit ...(truncated)
```

## 5. How to read the agreement

- `cartigsm` is the CartiGSFM P4 dictionary projection.
- `marker_rule` is a marker-only re-run of the same dictionary on the P4 pseudobulk;
  it ignores the P4 score table and the anti-marker panel, so any difference between
  `cartigsm` and `marker_rule` reflects the value of the anti-marker + score-table
  evidence weighting.
- `scgpt` is the bundled lightweight proxy; the real scGPT-human weights are NOT
  loaded in this sandbox, so the row is labelled `fallback=True`.
- `celltypist` was trained on a 50k-cell subsample of `acc.h5ad` with the v1.7.1
  SGD recipe (full-batch would have taken longer than the time budget).
- EBR is real data, not a held-out atlas fold; agreement with `cartigsm` is the
  practical figure of merit here. For a true head-to-head benchmark against a
  labelled dataset, run `cartigsfm annotate ... --reference-h5ad <labelled_atlas>`
  on the labelled atlas instead of on EBR.

## 6. Files

- F:\cartifm\outputs\annotation_P15\annotation_comparison_long.tsv
- F:\cartifm\outputs\annotation_P15\annotation_comparison_wide.tsv
- F:\cartifm\outputs\annotation_P15\annotation_comparison_pairwise.tsv
- F:\cartifm\CartiGM\reports\P15_ANNOTATION_COMPARISON.md
- F:\cartifm\outputs\annotation_P15\annotation_gptcelltype_prompts.tsv
- F:\cartifm\outputs\annotation_P15\annotation_celltypist_per_cluster.tsv
