# CartiGM / CartiGSFM

**CartiGM** is a cartilage-focused gene-set and single-cell interpretation framework for chondrocyte biology, cartilage tissue states, and disease-associated functional programs. It provides a curated three-layer cartilage dictionary, command-line utilities, and a lightweight local web portal for gene-list scoring, h5ad projection, annotation benchmarking, and evidence-constrained biological interpretation.

The project is designed for cartilage single-cell and transcriptomic studies, especially analyses involving chondrocyte subtypes, auricular / hyaline / fibrocartilage tissue states, osteoarthritis-related biology, metabolic stress, ferroptosis, matrix remodeling, and cartilage-specific functional axes.

---

## Overview

CartiGM organizes cartilage biology into a **three-layer dictionary**:

| Layer | Content |
|---|---|
| **Cell subtype** | Chondrocyte states such as Effector_Metabolic, Homeostatic, Progenitor, Inflammatory_Response, Prehypertrophic_Matrix, Fibrocartilage, and Reparative_Stress chondrocytes. |
| **Tissue / developmental state** | Cartilage tissue contexts including auricular elastic cartilage, articular hyaline cartilage, meniscus fibrocartilage, and nasal septum cartilage. |
| **Functional axis** | 39 biological axes covering cartilage development, ECM remodeling, signaling, inflammation, OA programs, ferroptosis, senescence, hypoxia, and metabolic pathways. |

The package includes:

- `cartilage_dictionary_v1`: a curated 53-axis cartilage dictionary.
- `cartigsfm score`: gene-list scoring against cartilage cell states, tissue states, and functional axes.
- `cartigsfm p4-project`: projection of independent single-cell or pseudobulk data onto the dictionary.
- `cartigsfm annotate`: cluster-level annotation using CartiGM and optional reference methods.
- `cartigsfm cs-predict`: bundled chondrocyte subtype classifier.
- `cartigsfm interpret`: evidence-constrained biological interpretation with claim-safety rules.
- `cartigsfm-web`: a lightweight local web interface for open-source use.

---

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
pip install -e .
```

For the local web portal:

```bash
pip install -e ".[web]"
```

The core package requires Python 3.9+ and depends mainly on `numpy` and `pandas`. Some workflows, such as h5ad projection or annotation backends, may require additional scientific Python or R packages depending on the selected method.

---

## Quick usage

### 1. Show the bundled cartilage dictionary

```bash
cartigsfm dictionary-v1
```

This summarizes the three-layer CartiGM dictionary and available axes.

---

### 2. Score a gene list

Create a gene list file:

```text
genes.txt
```

Example content:

```text
COL2A1
ACAN
SOX9
MMP13
ADAMTS5
IL1B
TNF
GPX4
SOD2
TXNRD1
```

Run:

```bash
cartigsfm score --query genes.txt
```

This returns ranked cartilage cell states, tissue states, and functional axes matching the input genes.

---

### 3. Project single-cell data onto CartiGM axes

For an `.h5ad` file:

```bash
cartigsfm p4-project \
  --h5ad your_data.h5ad \
  --sample-col sample \
  --tissue-col tissue \
  --cluster-col cluster \
  --celltype-col celltype \
  --layer log1p_norm \
  --outdir cartigm_p4_output
```

For a precomputed pseudobulk matrix:

```bash
cartigsfm p4-project \
  --pseudobulk p4_self_sample_cluster_pseudobulk.tsv \
  --meta p4_self_sample_cluster_meta.tsv \
  --outdir cartigm_p4_output
```

Main outputs include:

```text
cartigm_p4_output/tsv/p4_sample_cluster_three_layer_scores.tsv
cartigm_p4_output/tsv/p4_sample_cluster_top_assignments.tsv
cartigm_p4_output/tsv/p4_tissue_axis_summary.tsv
cartigm_p4_output/docs/P4_INDEPENDENT_VALIDATION_REPORT.md
```

---

### 4. Annotate clusters

After running `p4-project`, annotate clusters with CartiGM:

```bash
cartigsfm annotate \
  --method cartigsm \
  --p4-outdir cartigm_p4_output \
  --out annotation_report.md \
  --per-cluster-tsv annotation_per_cluster.tsv
```

To run all available annotation backends:

```bash
cartigsfm annotate \
  --method all \
  --p4-outdir cartigm_p4_output \
  --query-h5ad your_data.h5ad \
  --reference-h5ad reference_data.h5ad \
  --reference-label-col celltype \
  --cluster-col cluster \
  --out annotation_comparison.md
```

---

### 5. Predict chondrocyte subtypes per cell

```bash
cartigsfm cs-predict \
  --h5ad your_data.h5ad \
  --layer log1p_norm \
  --mode ensemble \
  --device cuda \
  --out cell_subtype_predictions.tsv
```

The bundled classifier is intended for chondrocyte subtype prediction and should be interpreted together with marker expression, tissue context, and manual review.

---

### 6. Generate evidence-constrained interpretation

```bash
cartigsfm interpret \
  --mode genes \
  --genes "COL2A1,ACAN,SOX9,MMP13,ADAMTS5,IL1B,TNF,SOD2,TXNRD1" \
  --format markdown
```

This produces a conservative biological interpretation anchored in the bundled dictionary and claim-safety rules.

---

## Local web portal

Start the web interface:

```bash
cartigsfm-web
```

Open:

```text
http://127.0.0.1:8000
```

The web portal supports:

- gene-list scoring;
- dictionary browsing;
- claim checking;
- evidence-constrained interpretation;
- command generation for h5ad projection workflows.

Large `.h5ad` files are not uploaded through the web interface. For large single-cell data, use the command-line workflow locally.

---

## Example benchmark use case

A typical validation workflow is:

```bash
# 1. Project manually annotated h5ad-derived pseudobulk onto CartiGM
cartigsfm p4-project \
  --pseudobulk ebr_leiden_res0_5_pseudobulk.tsv \
  --meta ebr_leiden_res0_5_meta.tsv \
  --outdir p4_cartigm_latest

# 2. Annotate clusters
cartigsfm annotate \
  --method cartigsm \
  --p4-outdir p4_cartigm_latest \
  --out annotation_cartigsm.md \
  --per-cluster-tsv annotation_cartigsm.tsv
```

In the current EBR manual-annotation benchmark, rerunning CartiGM on the current manually annotated `EBR.h5ad` reached:

| Metric | Value |
|---|---:|
| Cell-weighted accuracy | 0.671 |
| Cluster-majority accuracy | 0.800 |
| Macro F1 | 0.714 |
| Weighted F1 | 0.642 |

These values are provided as an in-project benchmark and should not be interpreted as clinical validation.

---

## Project structure

```text
cartigsfm/                  Python package and CLI
cartigsfm_web/              Local FastAPI web portal
cartigsfm/resources/        Bundled dictionary, RAG resources, and classifiers
data/processed/             Processed research corpus and gene-set resources
figures/                    Generated figures and benchmark plots
reports/                    Project reports and validation summaries
scripts/                    Analysis and figure-generation scripts
tests/                      Unit tests
```

---

## Interpretation boundaries

CartiGM is a research tool for cartilage gene-set interpretation and single-cell annotation support. It should be used as an evidence-prioritization and hypothesis-generation framework, not as a clinical diagnostic system. Functional-axis scores, cell-state assignments, and disease-associated interpretations require biological validation and expert review.

---

## Author

**YuJie Shen**  
PhD candidate in Otorhinolaryngology, Eye & ENT Hospital of Fudan University.  
Research interests include inner-ear regeneration, cartilage single-cell biology, cartilage gene-set modeling, and medical AI applications in otolaryngology.

Project focus: developing cartilage-domain computational models that connect public transcriptomic resources, single-cell atlases, literature-derived gene sets, and interpretable biological axes for cartilage research.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
