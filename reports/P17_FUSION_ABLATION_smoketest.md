# P17 - CartiGM + scGPT-proxy + GSFM Fusion Ablation

Reference atlas: acc.h5ad (2960 cells, sample-stratified 80/20 split)
External validation: EBR.h5ad (32885 cells)
Train samples: 59; Val samples: 15
Classes: 10 (EC_Lipo_Plasticity, Fibro_Matrix, Homeostatic_Matrix, Hypoxia_Adaptive, Hypoxia_Metabolic_Stress, Inflammatory_Remodeling, Maturation_Matrix, Mesenchymal_Remodeling, PRG4_Interface, Stress_IEG)
Elapsed: 138.2s

## 1. On the acc sample-stratified held-out split

| config | features | in_dim | accuracy | macro_f1 | balanced_acc | top_axis_consistency | evidence_citation | hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartigm_only | cartigm | 42 | 0.1683 | 0.0381 | 0.1066 | 0.11 | 1.0 | 0.0 |
| scgpt_only | scgpt | 42 | 0.255 | 0.0813 | 0.1558 | 0.125 | 1.0 | 0.0 |
| gsfm_only | gsfm | 42 | 0.1767 | 0.03 | 0.1 | 0.0167 | 0.9883 | 0.0117 |
| cartigm_scgpt | cartigm+scgpt | 84 | 0.1917 | 0.0626 | 0.129 | 0.3783 | 1.0 | 0.0 |
| cartigm_gsfm | cartigm+gsfm | 84 | 0.1467 | 0.0256 | 0.1 | 0.35 | 0.98 | 0.02 |
| full_fusion | cartigm+scgpt+gsfm | 126 | 0.1583 | 0.0293 | 0.1009 | 0.105 | 1.0 | 0.0 |

## 2. External validation on EBR (no curator celltype labels)

EBR has no curator cell-type labels, so accuracy / macro-F1 / balanced
accuracy are not directly computable. We report top-axis consistency
(does the predicted top axis agree with the per-cell CartiGM top axis?),
evidence citation (does the predicted axis have non-zero CartiGM
support?), hallucination rate, and the per-config dominant prediction.

| config | features | n_cells | n_pred_classes | top_pred_class | top_pred_frac | top_axis_consistency | evidence_citation | hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartigm_only | cartigm | 32885 | 5 | Fibro_Matrix | 0.6936 | 0.1167 | 0.434 | 0.566 |
| scgpt_only | scgpt | 32885 | 9 | Fibro_Matrix | 0.4611 | 0.1767 | 0.5871 | 0.4129 |
| gsfm_only | gsfm | 32885 | 1 | Fibro_Matrix | 1.0 | 0.0585 | 0.2829 | 0.7171 |
| cartigm_scgpt | cartigm+scgpt | 32885 | 7 | Homeostatic_Matrix | 0.3845 | 0.3035 | 0.711 | 0.289 |
| cartigm_gsfm | cartigm+gsfm | 32885 | 6 | Homeostatic_Matrix | 0.6048 | 0.1655 | 0.5557 | 0.4443 |
| full_fusion | cartigm+scgpt+gsfm | 32885 | 9 | Maturation_Matrix | 0.507 | 0.2333 | 0.6643 | 0.3357 |

## 3. Per-cluster majority prediction on EBR

| cluster | n_cells | cartigm_only | scgpt_only | gsfm_only | cartigm_scgpt | cartigm_gsfm | full_fusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6358 | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix | Maturation_Matrix | Homeostatic_Matrix | Maturation_Matrix |
| 1 | 5993 | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix | Maturation_Matrix | Homeostatic_Matrix | Maturation_Matrix |
| 10 | 184 | Hypoxia_Metabolic_Stress | Hypoxia_Metabolic_Stress | Fibro_Matrix | Hypoxia_Metabolic_Stress | Hypoxia_Metabolic_Stress | PRG4_Interface |
| 2 | 4309 | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix | Homeostatic_Matrix | Homeostatic_Matrix | Maturation_Matrix |
| 3 | 4287 | Fibro_Matrix | Hypoxia_Metabolic_Stress | Fibro_Matrix | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity |
| 4 | 3187 | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix | Maturation_Matrix | Homeostatic_Matrix | Maturation_Matrix |
| 5 | 2694 | Fibro_Matrix | Maturation_Matrix | Fibro_Matrix | Homeostatic_Matrix | Homeostatic_Matrix | Maturation_Matrix |
| 6 | 2439 | Fibro_Matrix | Maturation_Matrix | Fibro_Matrix | Homeostatic_Matrix | Homeostatic_Matrix | Maturation_Matrix |
| 7 | 2292 | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix | Homeostatic_Matrix | Homeostatic_Matrix | PRG4_Interface |
| 8 | 772 | Fibro_Matrix | Mesenchymal_Remodeling | Fibro_Matrix | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Fibro_Matrix |
| 9 | 370 | Hypoxia_Metabolic_Stress | Hypoxia_Metabolic_Stress | Fibro_Matrix | Mesenchymal_Remodeling | Mesenchymal_Remodeling | PRG4_Interface |

## 4. Caveats

- The scGPT branch is the bundled 42-axis proxy (real scGPT-human
  weights are not downloadable in this sandbox). Every result that
  depends on it carries `fallback=True`. Swap in real weights by
  editing `cartigsfm.fusion.build_scgpt_per_cell_embedding`.
- GSFM here is the per-cell Jaccard between top-50 expressed genes
  and each axis's `panel_genes` (no real GSFM weights loaded).
- acc was subsampled to 2960 cells stratified by sample
  to keep the run in memory; the split is then sample-stratified
  80/20 so no cell-level leakage is possible.