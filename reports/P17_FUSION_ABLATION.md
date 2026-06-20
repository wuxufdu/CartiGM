# P17 - CartiGM + scGPT-proxy + GSFM Fusion Ablation

Reference atlas: acc.h5ad (29691 cells, sample-stratified 80/20 split)
External validation: EBR.h5ad (32885 cells)
Train samples: 59; Val samples: 15
Classes: 10 (EC_Lipo_Plasticity, Fibro_Matrix, Homeostatic_Matrix, Hypoxia_Adaptive, Hypoxia_Metabolic_Stress, Inflammatory_Remodeling, Maturation_Matrix, Mesenchymal_Remodeling, PRG4_Interface, Stress_IEG)
Elapsed: 210.0s

## 1. On the acc sample-stratified held-out split

| config | features | in_dim | accuracy | macro_f1 | balanced_acc | top_axis_consistency | evidence_citation | hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartigm_only | cartigm | 42 | 0.6686 | 0.5765 | 0.5785 | 0.6132 | 1.0 | 0.0 |
| scgpt_only | scgpt | 42 | 0.6439 | 0.564 | 0.5634 | 0.5944 | 1.0 | 0.0 |
| gsfm_only | gsfm | 42 | 0.5492 | 0.4209 | 0.4398 | 0.5633 | 1.0 | 0.0 |
| cartigm_scgpt | cartigm+scgpt | 84 | 0.6729 | 0.584 | 0.5877 | 0.6021 | 1.0 | 0.0 |
| cartigm_gsfm | cartigm+gsfm | 84 | 0.6698 | 0.5849 | 0.5873 | 0.6121 | 1.0 | 0.0 |
| full_fusion | cartigm+scgpt+gsfm | 126 | 0.6774 | 0.5925 | 0.599 | 0.602 | 1.0 | 0.0 |

## 2. External validation on EBR (no curator celltype labels)

EBR has no curator cell-type labels, so accuracy / macro-F1 / balanced
accuracy are not directly computable. We report top-axis consistency
(does the predicted top axis agree with the per-cell CartiGM top axis?),
evidence citation (does the predicted axis have non-zero CartiGM
support?), hallucination rate, and the per-config dominant prediction.

| config | features | n_cells | n_pred_classes | top_pred_class | top_pred_frac | top_axis_consistency | evidence_citation | hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartigm_only | cartigm | 32885 | 10 | EC_Lipo_Plasticity | 0.4667 | 0.3087 | 0.6974 | 0.3026 |
| scgpt_only | scgpt | 32885 | 10 | EC_Lipo_Plasticity | 0.5583 | 0.2997 | 0.6268 | 0.3732 |
| gsfm_only | gsfm | 32885 | 4 | Mesenchymal_Remodeling | 0.8233 | 0.0839 | 0.3026 | 0.6974 |
| cartigm_scgpt | cartigm+scgpt | 32885 | 10 | EC_Lipo_Plasticity | 0.3956 | 0.3745 | 0.7325 | 0.2675 |
| cartigm_gsfm | cartigm+gsfm | 32885 | 10 | EC_Lipo_Plasticity | 0.4756 | 0.3432 | 0.7156 | 0.2844 |
| full_fusion | cartigm+scgpt+gsfm | 32885 | 10 | EC_Lipo_Plasticity | 0.3916 | 0.3162 | 0.6895 | 0.3105 |

## 3. Per-cluster majority prediction on EBR

| cluster | n_cells | cartigm_only | scgpt_only | gsfm_only | cartigm_scgpt | cartigm_gsfm | full_fusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6358 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 1 | 5993 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 10 | 184 | PRG4_Interface | Hypoxia_Adaptive | Mesenchymal_Remodeling | Hypoxia_Adaptive | PRG4_Interface | PRG4_Interface |
| 2 | 4309 | Fibro_Matrix | EC_Lipo_Plasticity | Mesenchymal_Remodeling | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix |
| 3 | 4287 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 4 | 3187 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 5 | 2694 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Fibro_Matrix |
| 6 | 2439 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Fibro_Matrix |
| 7 | 2292 | PRG4_Interface | Maturation_Matrix | Mesenchymal_Remodeling | Homeostatic_Matrix | PRG4_Interface | PRG4_Interface |
| 8 | 772 | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling |
| 9 | 370 | PRG4_Interface | PRG4_Interface | Mesenchymal_Remodeling | PRG4_Interface | EC_Lipo_Plasticity | PRG4_Interface |

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