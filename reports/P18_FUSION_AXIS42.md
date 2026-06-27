# P17 - CartiGM + scGPT-proxy + GSFM Fusion Ablation

Reference atlas: acc.h5ad (29691 cells, sample-stratified 80/20 split)
External validation: EBR.h5ad (32885 cells)
Train samples: 59; Val samples: 15
Classes: 10 (EC_Lipo_Plasticity, Fibro_Matrix, Homeostatic_Matrix, Hypoxia_Adaptive, Hypoxia_Metabolic_Stress, Inflammatory_Remodeling, Maturation_Matrix, Mesenchymal_Remodeling, PRG4_Interface, Stress_IEG)
Elapsed: 253.2s

## 1. On the acc sample-stratified held-out split

| config | features | in_dim | accuracy | macro_f1 | balanced_acc | top_axis_consistency | evidence_citation | hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartigm_only | cartigm | 42 | 0.6646 | 0.5728 | 0.5736 | 0.617 | 1.0 | 0.0 |
| scgpt_only | scgpt | 42 | 0.2007 | 0.0573 | 0.1214 | 0.1021 | 0.9998 | 0.0002 |
| gsfm_only | gsfm | 42 | 0.5397 | 0.404 | 0.429 | 0.5644 | 1.0 | 0.0 |
| cartigm_scgpt | cartigm+scgpt | 84 | 0.6629 | 0.5547 | 0.5552 | 0.6108 | 1.0 | 0.0 |
| cartigm_gsfm | cartigm+gsfm | 84 | 0.6663 | 0.5778 | 0.5797 | 0.6165 | 1.0 | 0.0 |
| full_fusion | cartigm+scgpt+gsfm | 126 | 0.6605 | 0.5546 | 0.5544 | 0.6097 | 1.0 | 0.0 |

## 2. External validation on EBR (no curator celltype labels)

EBR has no curator cell-type labels, so accuracy / macro-F1 / balanced
accuracy are not directly computable. We report top-axis consistency
(does the predicted top axis agree with the per-cell CartiGM top axis?),
evidence citation (does the predicted axis have non-zero CartiGM
support?), hallucination rate, and the per-config dominant prediction.

| config | features | n_cells | n_pred_classes | top_pred_class | top_pred_frac | top_axis_consistency | evidence_citation | hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartigm_only | cartigm | 32885 | 10 | EC_Lipo_Plasticity | 0.5079 | 0.3196 | 0.6903 | 0.3097 |
| scgpt_only | scgpt | 32885 | 2 | Maturation_Matrix | 0.987 | 0.1336 | 0.4459 | 0.5541 |
| gsfm_only | gsfm | 32885 | 4 | Mesenchymal_Remodeling | 0.8727 | 0.0721 | 0.292 | 0.708 |
| cartigm_scgpt | cartigm+scgpt | 32885 | 7 | EC_Lipo_Plasticity | 0.7914 | 0.3151 | 0.6175 | 0.3825 |
| cartigm_gsfm | cartigm+gsfm | 32885 | 10 | EC_Lipo_Plasticity | 0.5189 | 0.3549 | 0.7147 | 0.2853 |
| full_fusion | cartigm+scgpt+gsfm | 32885 | 7 | EC_Lipo_Plasticity | 0.7035 | 0.3192 | 0.6449 | 0.3551 |

## 3. Per-cluster majority prediction on EBR

| cluster | n_cells | cartigm_only | scgpt_only | gsfm_only | cartigm_scgpt | cartigm_gsfm | full_fusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6358 | EC_Lipo_Plasticity | Maturation_Matrix | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 1 | 5993 | EC_Lipo_Plasticity | Maturation_Matrix | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 10 | 184 | PRG4_Interface | Maturation_Matrix | Mesenchymal_Remodeling | EC_Lipo_Plasticity | PRG4_Interface | EC_Lipo_Plasticity |
| 2 | 4309 | Fibro_Matrix | Maturation_Matrix | Mesenchymal_Remodeling | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix |
| 3 | 4287 | EC_Lipo_Plasticity | Maturation_Matrix | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 4 | 3187 | EC_Lipo_Plasticity | Maturation_Matrix | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 5 | 2694 | EC_Lipo_Plasticity | Maturation_Matrix | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 6 | 2439 | EC_Lipo_Plasticity | Maturation_Matrix | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 7 | 2292 | Fibro_Matrix | Maturation_Matrix | Mesenchymal_Remodeling | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix |
| 8 | 772 | Mesenchymal_Remodeling | Maturation_Matrix | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling |
| 9 | 370 | EC_Lipo_Plasticity | Maturation_Matrix | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |

## 4. Caveats

- The scGPT branch is the bundled 42-axis proxy (real scGPT-human
  weights are not downloadable in this sandbox). Every result that
  depends on it carries `fallback=True`. Swap in real weights by
  editing `cartigsfm.fusion.build_scgpt_per_cell_embedding`.
- GSFM here is the per-cell Jaccard between top-50 expressed genes
  and each axis's `panel_genes` (no real GSFM weights loaded).
- acc was subsampled to 29691 cells stratified by sample
  to keep the run in memory; the split is then sample-stratified
  80/20 so no cell-level leakage is possible.