# cartigsfm annotate (--method all) - cross-method summary

n_clusters_per_method: {"cartigsm": 11, "cellassign": 11, "celltypist": 11, "scgpt": 11, "scmap": 11, "singler": 11, "symphony": 11}

pairwise_agreement: {"cartigsm__cellassign": 0.3636, "cartigsm__celltypist": 0.1818, "cartigsm__scgpt": 0.4545, "cartigsm__scmap": 0.3636, "cartigsm__singler": 0.3636, "cartigsm__symphony": 0.2727, "cellassign__celltypist": 0.2727, "cellassign__scgpt": 0.4545, "cellassign__scmap": 0.4545, "cellassign__singler": 0.6364, "cellassign__symphony": 0.2727, "celltypist__scgpt": 0.4545, "celltypist__scmap": 0.3636, "celltypist__singler": 0.4545, "celltypist__symphony": 0.2727, "scgpt__scmap": 0.3636, "scgpt__singler": 0.7273, "scgpt__symphony": 0.3636, "scmap__singler": 0.5455, "scmap__symphony": 0.3636, "singler__symphony": 0.4545}

## per-cluster wide table

| cluster | cartigsm | cellassign | celltypist | scgpt | scmap | singler | symphony |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Maturation_Matrix | EC_Lipo_Plasticity | Hypoxia_Adaptive | Maturation_Matrix | unassigned | Inflammatory_Remodeling | Hypoxia_Metabolic_Stress |
| 1 | Hypoxia_Metabolic_Stress | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Hypoxia_Adaptive | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 10 | Hypoxia_Metabolic_Stress | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Hypoxia_Adaptive | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Hypoxia_Metabolic_Stress |
| 2 | Stress_IEG | Stress_IEG | Hypoxia_Adaptive | Stress_IEG | Stress_IEG | Stress_IEG | Stress_IEG |
| 3 | Hypoxia_Metabolic_Stress | Inflammatory_Remodeling | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity | EC_Lipo_Plasticity |
| 4 | EC_Lipo_Plasticity | EC_Lipo_Plasticity | Hypoxia_Adaptive | EC_Lipo_Plasticity | Maturation_Matrix | EC_Lipo_Plasticity | Hypoxia_Metabolic_Stress |
| 5 | Stress_IEG | Maturation_Matrix | PRG4_Interface | Maturation_Matrix | Hypoxia_Adaptive | Maturation_Matrix | Inflammatory_Remodeling |
| 6 | Maturation_Matrix | Maturation_Matrix | Hypoxia_Adaptive | Homeostatic_Matrix | Maturation_Matrix | Homeostatic_Matrix | Hypoxia_Metabolic_Stress |
| 7 | Stress_IEG | Homeostatic_Matrix | Fibro_Matrix | Fibro_Matrix | Homeostatic_Matrix | Homeostatic_Matrix | Homeostatic_Matrix |
| 8 | Mesenchymal_Remodeling | EC_Lipo_Plasticity | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling |
| 9 | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Mesenchymal_Remodeling | Hypoxia_Metabolic_Stress |

per_method notes:
- cartigsm: CartiGSFM P4 + cartilage_dictionary_v1
- marker_rule: marker-only re-run of cartilage_dictionary_v1 on the P4 pseudobulk
- scgpt: Fallback: weighted-mean of per-cluster mean expression of each v1 axis marker genes. Real scGPT-human weights are not bundled in this sandbox. Device: cuda:0 (the proxy ignores device; this field is recorded for API uniformity so callers can route real weights to GPU).
- celltypist: Real CellTypist v1.7.1; trained on the reference using chongdrocyte_subtype, then per-cell predictions aggregated per leiden_res0_5 with majority vote and mean confidence. Trainer: cartigsfm.annotate_torch.train_logreg_torch on cuda:0 (celltypist's sklearn SGD trainer is CPU-only; the bundled torch trainer is the GPU-runnable path)
- gptcelltype: OPENAI_API_KEY not set; prompts returned without model call.
- singler: Real SingleR via rpy2 on cluster-level pseudo-bulk. Reference subsampled to 8000 cells (max 800 per label). Common genes after intersection: 28451.
- scmap: Real scmap-cluster via rpy2. Reference subsampled to 8000 cells (max 800 per label). selectFeatures n_features=500. Common genes after intersection: 28451.
- symphony: Real Symphony via rpy2. buildReference K=100 d=20. Reference subsampled to 8000 cells (max 800 per label). Common genes after intersection: 28451. Per-cell labels aggregated to per-cluster majority.
- cellassign: Real CellAssign via cartigsfm.cellassign_torch (PyTorch port of Irrationone/cellassign EM, runs on cuda:0, 8 EM iters) with 171 marker genes from cartilage_dictionary_v1 layer 'cell_subtype' (10 cell types). Query subsampled to 6000 cells. Per-cell labels aggregated to per-cluster majority.