# P-F Training Report - CartiGSFM cell_subtype classifier

## 1. Dataset and feature basis

Two atlases jointly supervise a 10-class cell_subtype classifier:

| atlas | cells | label column | classes seen |
|---|---|---|---|
| acc_new.h5ad | 416,574 (subsampled stratified by sample x label to ~63k) | `celltype_new` | 10 (full cs panel) |
| EBR.h5ad | 32,281 (cell-level 70/30 train/test within each batch x cluster unit) | `celltype` | 7 EBR-present cs classes |

Feature basis: 2000 highly variable genes selected from acc_new on the
intersection `acc_new.var_names` ∩ `EBR.var_names` (28,451 / 40,786 acc genes
pass the intersection filter), so every HVG column has a non-padded value in
both atlases. acc_new uses its native `.X` (probed log1p-norm, max ~7.83);
EBR uses `layers["log1p_norm"]` because its native `.X` is scVelo-corrupted
(has negative values).

## 2. Model

`cartigsfm.cs_classifier.CSClassifier`:

- input LayerNorm (per-cell scale alignment across acc / EBR)
- input dropout 0.1
- Linear(2000 -> 384) + LayerNorm + GELU + Dropout(0.4)
- Linear(384 -> 192)  + LayerNorm + GELU + Dropout(0.4)
- Linear(192 -> 10)
- 0.85M parameters, AdamW lr=5e-4 cosine 60 epochs, weight_decay=1e-3

Class-balanced cross-entropy with EBR rows up-weighted 3x to bias the
decision boundary toward the target domain.

## 3. Headline accuracy (best by acc-val accuracy)

| split | n_cells | accuracy | macro_f1 | balanced_acc |
|---|---|---|---|---|
| acc_sample_stratified_val | 12,165 | **56.4%** | 54.6% | 53.4% |
| EBR cell-level holdout (within-cluster 30%) | 9,687 | **76.6%** | n/a | n/a |

Cluster-level (batch x cluster) majority-vote agreement on EBR test = **76.2%**
(32 / 42 clusters). For comparison, the v1.8.5 dictionary panel projection
P4 (no learning) achieves cs top-1 cell 68.1% and cs top-1 cluster 45.2%.
The trained classifier improves cluster top-1 by **+31 pp**.

## 4. Per-celltype recall on EBR test

| celltype | n_cells | recall |
|---|---|---|
| Fibrocartilage_Chondrocytes | 231 | 90.9% |
| Effector_Metabolic_Chondrocytes | 3026 | 78.3% |
| Homeostatic_Chondrocytes | 2849 | 78.4% |
| Prehypertrophic_Matrix_Chondrocytes | 913 | 77.0% |
| Progenitor_Chondrocytes | 987 | 72.8% |
| Reparative_Stress_Chondrocytes | 750 | 70.7% |
| Inflammatory_Response_Chondrocytes | 931 | 70.3% |

No celltype falls below 70%; Homeostatic - the residual confusion in v1.8.5
(27% top-1 cell) - climbs to 78.4%.

## 5. Cross-batch (LBO) and v2 augmentation

The within-cluster cell-level holdout reuses cells from each EBR
(batch, cluster) unit; it does *not* test whether the classifier extrapolates
to a previously unseen tissue. To stress that we also ran a leave-batch-out
evaluation that holds out ear / nose / rib in turn:

| held-out batch | n_test | v1 cell acc | v2 cell acc |
|---|---|---|---|
| ear | 7641 | 53.6% | 67.8% |
| nose | 16436 | 43.8% | 52.6% |
| rib | 8204 | 49.1% | 52.1% |
| **mean** |   | **48.8%** | **57.5%** |

The v2 model adds Gaussian input noise (sigma=0.15), MixUp (alpha=0.2), a
slightly wider hidden stack (512/256), higher dropout (0.5 / input 0.15),
and weight_decay=3e-3 on top of an extended 100-epoch cosine schedule. It
lifts cross-batch (LBO) accuracy by **+8.7 pp** while preserving the
within-cluster cell-level number (76.9% vs v1 76.6%); the within-cluster
cluster-majority drops from 76.2% to 69.0% because v2 spreads probability
more evenly between subtypes.

The bundled package therefore ships two checkpoints (`classifier.pt` =
v1, `classifier_v2.pt` = v2) and a default ensemble that averages their
softmaxes:

| model | within-cluster cell | within-cluster cluster top-1 | LBO mean cell |
|---|---|---|---|
| v1 | 76.6% | 76.2% | 48.8% |
| v2 | 76.9% | 69.0% | 57.5% |
| **ensemble (default)** | **77.6%** | **76.2%** | n/a |

`cartigsfm cs-predict` defaults to `--mode ensemble`; pass `--mode v1` for
the older behaviour or `--mode v2` for cross-tissue queries.

## 6. Caveats

- The within-cluster hold-out is cell-level inside each (batch, cluster)
  unit; the model still relies on having *seen* a few cells from each EBR
  cluster during training. The LBO column above is the authoritative
  cross-tissue generalization measurement.
- The 3 atlas-only cs classes (Hypoxic / Metabolic_Stress / Superficial_Zone)
  retain the v1.8.4 acc_new-supervised dictionary panels but were not
  separately validated on EBR (EBR has none of them).
- The model is intentionally compact (0.85M params); pairing it with the
  scGPT-pretrain encoder (P16) is left as a follow-up because the remote
  RTX 5070 needs PyTorch >=2.7 nightly for sm_120 support and the local 4090
  was preferred for fast iteration.

## 7. Artifacts

- `outputs/training_local/classifier.pt` - state_dict + class/gene order + config
- `outputs/training_local/train_log.tsv` - per-epoch loss/acc/F1
- `outputs/training_local/acc_val_metrics.tsv` - acc held-out
- `outputs/training_local/ebr_metrics.tsv` - EBR cross-atlas
- `outputs/training_local/ebr_by_celltype.tsv` - per-celltype recall
- `outputs/training_local/ebr_per_cluster.tsv` - per-(batch,cluster) majority
- `outputs/training_local/ebr_confusion.tsv` - confusion matrix
- `outputs/training_local/summary.json` - headline numbers
- `outputs/training_local/lbo_v1_metrics.tsv` - v1 leave-batch-out
- `outputs/training_local/lbo_v1_per_celltype.tsv`
- `outputs/training_local/v2/classifier_v2.pt` - v2 augmentation checkpoint
- `outputs/training_local/v2/lbo_metrics.tsv` - v2 leave-batch-out
- `outputs/training_local/v2/lbo_per_celltype.tsv`
- `outputs/training_local/v2/within_cluster_metrics.tsv`
- `outputs/training_local/v2/within_cluster_per_celltype.tsv`
- `outputs/training_local/v2/within_cluster_per_cluster.tsv`
- `outputs/training_local/v2/summary.json`
- `outputs/training_local/ensemble/within_cluster_summary.tsv` - v1 / v2 / ensemble side-by-side
- `outputs/training_local/ensemble/within_cluster_*_per_celltype.tsv`
- `cartigsfm/resources/cs_classifier_v1/classifier.pt` - bundled v1
- `cartigsfm/resources/cs_classifier_v1/classifier_v2.pt` - bundled v2
- `cartigsfm/resources/cs_classifier_v1/hvg_genes.tsv` - bundled HVG basis
- `outputs/training_subset/acc_train.npz`, `ebr_eval.npz`, `hvg_genes.tsv` - feature dumps
