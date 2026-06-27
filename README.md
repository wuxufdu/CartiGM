# CartiGM / CartiGSFM

CartiGM is a cartilage-focused gene-set and single-cell interpretation framework for chondrocyte biology, cartilage tissue states, and disease-associated functional programs.

The project provides a bundled three-layer cartilage dictionary, command-line utilities, and a lightweight local web portal for:

- cartilage gene-list scoring;
- independent `.h5ad` or pseudobulk projection;
- chondrocyte subtype prediction;
- cluster-level annotation support;
- evidence-constrained biological interpretation and claim checking.

The current bundled dictionary is `cartilage_dictionary_v1` version `v1.8.6`, containing 53 axes:

| Layer | Axes |
|---|---:|
| Cell subtype | 10 |
| Tissue / developmental state | 4 |
| Functional axis | 39 |

CartiGM is a research tool for hypothesis generation and biological interpretation. It is not a clinical diagnostic system.

---

## Installation

```bash
git clone https://github.com/wuxufdu/CartiGM.git
cd CartiGM
pip install -e .
```

For the local web portal:

```bash
pip install -e ".[web]"
```

Python 3.9+ is recommended. The core package mainly depends on `numpy` and `pandas`. Optional workflows such as `.h5ad` projection, classifier inference, or R-backed annotation methods may require additional packages.

---

## Usage

### Show the bundled dictionary

```bash
cartigsfm dictionary-v1
cartigsfm dictionary-v1 --show-axes
```

### Score a gene list

Create `genes.txt`:

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
cartigsfm score --query genes.txt --kind both --top 20
```

### Project single-cell or pseudobulk data

For `.h5ad` input:

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

For pseudobulk input:

```bash
cartigsfm p4-project \
  --pseudobulk p4_self_sample_cluster_pseudobulk.tsv \
  --meta p4_self_sample_cluster_meta.tsv \
  --outdir cartigm_p4_output
```

### Annotate clusters

```bash
cartigsfm annotate \
  --method cartigsm \
  --p4-outdir cartigm_p4_output \
  --out annotation_report.md \
  --per-cluster-tsv annotation_per_cluster.tsv
```

### Predict chondrocyte subtypes

```bash
cartigsfm cs-predict \
  --h5ad your_data.h5ad \
  --layer log1p_norm \
  --mode ensemble \
  --device cpu \
  --out cell_subtype_predictions.tsv
```

### Generate evidence-constrained interpretation

```bash
cartigsfm interpret \
  --mode genes \
  --genes "COL2A1,ACAN,SOX9,MMP13,ADAMTS5,IL1B,TNF,SOD2,TXNRD1" \
  --format markdown
```

### Run the local web portal

```bash
cartigsfm-web
```

Open:

```text
http://127.0.0.1:8000
```

The web portal supports gene-list scoring, dictionary browsing, claim checking, and local command generation for `.h5ad` workflows. Large `.h5ad` files are not uploaded through the web interface.

---

## Author

**Xu Wu**  
复旦大学附属眼耳鼻喉科医院眼耳鼻整形外科博士后。  
Postdoctoral Fellow, Department of Ophthalmology, Otolaryngology and Plastic Surgery, Eye & ENT Hospital of Fudan University.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
