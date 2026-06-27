# P17 - CartiGM + scGPT-proxy + GSFM Fusion Ablation

Reference atlas: acc.h5ad (29691 cells, sample-stratified 80/20 split)
External validation: EBR.h5ad (32885 cells)
Train samples: 59; Val samples: 15
Classes: 10 (EC_Lipo_Plasticity, Fibro_Matrix, Homeostatic_Matrix, Hypoxia_Adaptive, Hypoxia_Metabolic_Stress, Inflammatory_Remodeling, Maturation_Matrix, Mesenchymal_Remodeling, PRG4_Interface, Stress_IEG)
Elapsed: 154.4s

## 1. On the acc sample-stratified held-out split

| config | features | in_dim | accuracy | macro_f1 | balanced_acc | top_axis_consistency | evidence_citation | hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartigm_only | cartigm | 42 | 0.6646 | 0.5728 | 0.5736 | 0.617 | 1.0 | 0.0 |
| scgpt_only | scgpt | 192 | 0.3401 | 0.2279 | 0.2544 | 0.3219 | 0.9993 | 0.0007 |
| gsfm_only | gsfm | 42 | 0.5397 | 0.404 | 0.429 | 0.5644 | 1.0 | 0.0 |
| cartigm_scgpt | cartigm+scgpt | 234 | 0.6599 | 0.5577 | 0.5595 | 0.6178 | 1.0 | 0.0 |
| cartigm_gsfm | cartigm+gsfm | 84 | 0.6663 | 0.5778 | 0.5797 | 0.6165 | 1.0 | 0.0 |
| full_fusion | cartigm+scgpt+gsfm | 276 | 0.6618 | 0.5629 | 0.564 | 0.6196 | 1.0 | 0.0 |

## 2. External validation on EBR (no curator celltype labels)

EBR has no curator cell-type labels, so accuracy / macro-F1 / balanced
accuracy are not directly computable. We report top-axis consistency
(does the predicted top axis agree with the per-cell CartiGM top axis?),
evidence citation (does the predicted axis have non-zero CartiGM
support?), hallucination rate, and the per-config dominant prediction.

| config | features | n_cells | n_pred_classes | top_pred_class | top_pred_frac | top_axis_consistency | evidence_citation | hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartigm_only | cartigm | 32885 | 10 | EC_Lipo_Plasticity | 0.5079 | 0.3196 | 0.6903 | 0.3097 |
| scgpt_only | scgpt | 32885 | 8 | EC_Lipo_Plasticity | 0.442 | 0.1916 | 0.4213 | 0.5787 |
| gsfm_only | gsfm | 32885 | 4 | Mesenchymal_Remodeling | 0.8727 | 0.0721 | 0.292 | 0.708 |
| cartigm_scgpt | cartigm+scgpt | 32885 | 10 | EC_Lipo_Plasticity | 0.6052 | 0.3455 | 0.6736 | 0.3264 |
| cartigm_gsfm | cartigm+gsfm | 32885 | 10 | EC_Lipo_Plasticity | 0.5189 | 0.3549 | 0.7147 | 0.2853 |
| full_fusion | cartigm+scgpt+gsfm | 32885 | 10 | EC_Lipo_Plasticity | 0.6308 | 0.3454 | 0.673 | 0.327 |

## 3. Per-cluster majority prediction on EBR

| cluster | n_cells | cartigm_only | scgpt_only | gsfm_only | cartigm_scgpt | cartigm_gsfm | full_fusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6358 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 1 | 5993 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 10 | 184 | PRG4_Interface | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | PRG4_Interface | EC_Lipo_Plasticity |
| 2 | 4309 | Fibro_Matrix | EC_Lipo_Plasticity | Mesenchymal_Remodeling | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix |
| 3 | 4287 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 4 | 3187 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 5 | 2694 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | Fibro_Matrix | EC_Lipo_Plasticity | Fibro_Matrix |
| 6 | 2439 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | Fibro_Matrix | EC_Lipo_Plasticity | Fibro_Matrix |
| 7 | 2292 | Fibro_Matrix | EC_Lipo_Plasticity | Mesenchymal_Remodeling | Fibro_Matrix | Fibro_Matrix | Fibro_Matrix |
| 8 | 772 | Mesenchymal_Remodeling | EC_Lipo_Plasticity | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling |
| 9 | 370 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Mesenchymal_Remodeling | Mesenchymal_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity |

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