# CartiGM 项目报告

**项目名称**: CartiGSFM (Cartilage Gene-Set Foundation Model)
**包名**: `cartigsfm`
**当前版本**: 0.6.1
**字典版本**: v1.8.6
**GitHub**: https://github.com/wuxufdu/CartiGM
**日期**: 2026-06-21

---

## 1. 项目背景与目标

CartiGSFM 是一个面向软骨 / 软骨细胞 / 骨关节炎 (OA) 生物学的领域专用基础模型,灵感来源于 GSFM (Gene-Set Foundation Model, Patterns 2026)。项目的核心目标是:将公共单细胞软骨 atlas 数据与文献证据整合,构建一个可落地为 Python 包的计算工具,将软骨 marker 基因或表达矩阵转化为实用的下游输出——软骨亚型鉴定、组织/发育状态解读、功能轴打分、独立验证报告、RAG 证据摘要与 LLM 安全元数据。

设计原则:每一个计算、训练、验证步骤都必须能提升 Python 包的实际输出能力。离线报告本身不算交付,除非通过包函数或 CLI 暴露出来。

---

## 2. 数据基础

项目整合了两套核心单细胞数据:

- **acc atlas (acc_new.h5ad)**:公共软骨单细胞 atlas,约 416,574 细胞,74 个可用样本,覆盖耳软骨、关节软骨、半月板、椎间盘,以及微耳畸形 / OA 等疾病-正常对比。`celltype_new` 列标注 10 个软骨细胞亚型。X 为 log1p-normalized (max ~7.83)。
- **EBR atlas (EBR.h5ad)**:自建耳 / 鼻 / 肋软骨单细胞数据,32,281 细胞,3 个 batch (ear / nose / rib),14 个 leiden cluster,`celltype` 列标注 7 个软骨细胞亚型。`layers["log1p_norm"]` 为干净的对数归一化层 (原始 .X 被 scVelo 污染含负值)。

两个 atlas 在基因层面有 28,451 个共同基因,作为 HVG 选择的范围。

---

## 3. 三层软骨字典 (v1.8.6)

字典是项目的知识核心,共 53 个轴,分三层:

| 层 | 轴数 | 说明 |
|---|---|---|
| `cell_subtype` | 10 | 软骨细胞亚型 |
| `tissue_developmental_state` | 4 | 组织 / 发育状态 |
| `functional_axis` | 39 | 功能轴 (信号通路、代谢、疾病等) |

每个 cs 轴包含 30 core genes / 50 panel genes / 30+ anti-markers。

### 3.1 cell_subtype 层 (10 轴)

| 轴 ID | 中文名 |
|---|---|
| Effector_Metabolic_Chondrocytes | 效应代谢型软骨细胞 |
| Progenitor_Chondrocytes | 祖细胞型软骨细胞 |
| Homeostatic_Chondrocytes | 稳态型软骨细胞 |
| Hypoxic_Chondrocytes | 缺氧适应型软骨细胞 |
| Metabolic_Stress_Chondrocytes | 代谢应激型软骨细胞 |
| Inflammatory_Response_Chondrocytes | 炎症应答型软骨细胞 |
| Prehypertrophic_Matrix_Chondrocytes | 肥大前基质型软骨细胞 |
| Fibrocartilage_Chondrocytes | 纤维软骨型软骨细胞 |
| Superficial_Zone_Chondrocytes | 浅层带软骨细胞 |
| Reparative_Stress_Chondrocytes | 修复应激型软骨细胞 |

每个轴保留了旧 axis_id 的别名映射 (alias),保证向后兼容。

### 3.2 tissue_developmental_state 层 (4 轴)

ElasticCartilage_Auricular (弹性软骨-耳廓), Hyaline_ArticularCartilage (透明软骨-关节), Fibrocartilage_Meniscus (纤维软骨-半月板), Nasal_Septum_Cartilage (鼻中隔软骨)。

其中鼻中隔软骨轴基于 EBR scRNA 数据 (n_nose=16,706 vs n_ear+rib=16,179) 构建:GPU Mann-Whitney U + FDR-BH,再以 10 个解剖学锚点 + 10 个数据驱动基因构 panel 60。Core 包含 CNMD, MGP, MATN1, ENPP1, ACAN, MATN3, COL11A2, COL11A1, COL9A1, ANKH, MMP1, TNFRSF11B 等。

### 3.3 functional_axis 层 (39 轴)

涵盖软骨发育 (CartilageDevelopment, Chondrogenesis, EndochondralOssification, Hypertrophy)、ECM 代谢 (ECM_Organization, CollagenMetabolism, Proteoglycan, MMP_Activity, ADAMTS_Activity)、信号通路 (Wnt, BMP, TGF-beta, IHH, FGF)、炎症 (Inflammation_NFkB, IL1, TNF)、细胞命运 (Apoptosis, Ferroptosis, Autophagy, Senescence)、力学 (Mechanotransduction)、缺氧 (Hypoxia)、疾病 (Osteoarthritis, RheumatoidArthritis, OA_cartilage_intrinsic, OA_synovium)、血管化 (AngiogenesisInCartilage, Avascular_Antimineralization),以及 10 个代谢轴 (Glycolysis, OxidativePhosphorylation, TCA_Cycle, PentosePhosphatePathway, FattyAcidOxidation, Lipogenesis, CholesterolHomeostasis, LipidDroplet, Glutaminolysis, MitochondrialBiogenesis)。

### 3.4 字典演进历程

| 版本 | 主要变化 |
|---|---|
| v1.1 | marker / anti-marker 重叠修复 |
| v1.2 | 新增鼻中隔软骨轴 |
| v1.5 | 合并用户 v1.4 atlas marker 表 |
| v1.6 | 10 个 cs axis_id 改为文献名 |
| v1.7 | 新增 10 个代谢功能轴 |
| v1.8.1 | Stromal_Matrix -> Progenitor 改名 |
| v1.8.2 | Matrix_Maintenance -> Homeostatic 改名 |
| v1.8.3 | EBR 监督重建 Homeostatic panel |
| v1.8.4 | acc_new 一次性重建全部 10 个 cs panel |
| v1.8.5 | 7 个 EBR 在场轴用 EBR in-domain DE 重新拟合 |
| v1.8.6 | P-D triangulation,10 轴互掐 anti-marker 防泄漏 |

---

## 4. 分类器模型 (cs_classifier_v1)

### 4.1 训练目标

训练一个 10 类软骨细胞亚型分类器,输入为 log1p-normalized 表达矩阵 (2000 HVG),输出为亚型标签。

### 4.2 训练数据

- acc_new: 416k -> 平衡抽样 63k (每类 5000),`celltype_new` 10 类
- EBR: 32k,`celltype` 7 类 (cell-level 70/30 split within (batch, cluster) unit)
- 特征: 2000 HVG,选自 acc ∩ EBR 28,451 基因交集

### 4.3 模型架构

`cartigsfm.cs_classifier.CSClassifier`:

```
LayerNorm(2000) -> Dropout(0.1)
-> Linear(2000,384) + LayerNorm + GELU + Dropout(0.4)
-> Linear(384,192) + LayerNorm + GELU + Dropout(0.4)
-> Linear(192,10)
```

0.85M 参数,AdamW lr=5e-4 cosine,类平衡交叉熵,EBR rows 权重 3.0。

### 4.4 两个 checkpoint

- **v1**: 60 epochs,基础训练。EBR within-cluster cell 76.6% / cluster top-1 76.2%。
- **v2**: 100 epochs,加 Gaussian noise (sigma=0.15) + MixUp (alpha=0.2) + 更强正则 (dropout 0.5, wd=3e-3, hidden 512/256)。EBR within-cluster cell 76.9% / cluster top-1 69.0%,但 leave-batch-out 平均 57.5% (v1 仅 48.8%,+8.7pp)。
- **ensemble (默认)**: v1 + v2 softmax 平均。EBR within-cluster cell 77.6% / cluster top-1 76.2%。

### 4.5 评估结果

**Within-cluster cell-level holdout (EBR,同协议对比)**

| 模型 | cell accuracy | cluster top-1 |
|---|---|---|
| v1 | 76.6% | 76.2% |
| v2 | 76.9% | 69.0% |
| ensemble | 77.6% | 76.2% |
| (字典投影 v1.8.5, 无学习) | 68.1% | 45.2% |

训练后的 ensemble 相比字典投影,cluster top-1 提升 +31pp。

**Leave-batch-out (跨批次泛化)**

| held-out batch | n_cells | v1 cell acc | v2 cell acc |
|---|---|---|---|
| ear | 7,641 | 53.6% | 67.8% |
| nose | 16,436 | 43.8% | 52.6% |
| rib | 8,204 | 49.1% | 52.1% |
| 均值 | | 48.8% | 57.5% |

**EBR per-celltype recall (ensemble)**

| celltype | n | recall |
|---|---|---|
| Fibrocartilage_Chondrocytes | 231 | 90.9% |
| Homeostatic_Chondrocytes | 2,849 | 79.3% |
| Prehypertrophic_Matrix_Chondrocytes | 913 | 79.5% |
| Effector_Metabolic_Chondrocytes | 3,026 | 76.6% |
| Progenitor_Chondrocytes | 987 | 76.6% |
| Reparative_Stress_Chondrocytes | 750 | 74.7% |
| Inflammatory_Response_Chondrocytes | 931 | 66.3% |

所有亚型 recall >= 66%,无低于 60% 的类别。

### 4.6 关键发现与局限

- 当前 77.6% 是 **within-cluster generalization**:测试 cell 来自训练时见过的 cluster,只是不同 cell。
- **Cross-cluster / cross-batch generalization** 仍弱 (LBO 平均 57.5%):acc 和 EBR 的 batch 间存在系统性偏移,模型仍需在训练时见过目标 cluster 的部分 cell 才能可靠预测。
- 3 个 atlas-only 亚型 (Hypoxic / Metabolic_Stress / Superficial_Zone) 保留 v1.8.4 acc 监督字典 panel,但未在 EBR 上单独验证 (EBR 无这三类)。
- 模型刻意紧凑 (0.85M params),与 scGPT 预训练 encoder 的融合 (P16/P17 路径) 留作后续。

---

## 5. 训练流程 (P-A 到 P-F)

| 阶段 | 内容 | 状态 |
|---|---|---|
| P-A | acc_new 全 10 cs 轴 wilcoxon DE | 完成 |
| P-D | 10 轴 anti-marker triangulation 防泄漏 | 完成 |
| P-B | EBR 7 个在场轴 in-domain DE 重拟合 | 完成 |
| P-C | P3 代谢轴 atlas 校准 | 完成 |
| P-E | EBR P4 独立验证 + 准确率报告 | 完成 |
| P-F | GPU 训练 cs 分类器 (v1 + v2 + ensemble) | 完成 |

训练全部在本地 RTX 4090 (48G, torch 2.5.1+cu121) 完成。远端 RTX 5070 因 sm_120 架构不被 torch 2.6 支持而无法用于 GPU 训练,仅用于 DE 计算和特征提取。

---

## 6. 软件包能力

`cartigsfm` 0.6.1 提供以下能力:

- **字典查询**:三层软骨字典加载、轴汇总、基因列表打分
- **P4 独立验证**:将任意 h5ad / pseudobulk 投影到字典,生成 per-cluster top-axis 报告
- **cs-predict 分类器**:对任意 h5ad 做端到端软骨亚型预测 (bundled v1+v2 ensemble,CUDA)
- **RAG 证据**:P6 知识库、轴证据卡、claim 安全检查
- **P9 LoRA 元数据**:LLM 微调安全审计
- **多后端注释**:cartigsm / marker_rule / scgpt / celltypist / gptcelltype / singler / scmap / symphony / cellassign
- **消融实验**:四路消融 (cartigm_only / cartigm_gsfm / cartigm_scgpt / full)
- **scGPT 预训练**:小 transformer encoder MLM 预训练
- **Fusion 训练**:CartiGM + scGPT + GSFM 融合消融

---

## 7. 技术栈

- Python 3.11,torch 2.5.1+cu121 (本地 4090)
- anndata, scanpy, numpy, pandas, scipy
- 远端:Ubuntu, conda squidp311, torch 2.6+cu126 (DE / P4 评估用)
- Git,GitHub (https://github.com/wuxufdu/CartiGM)

---

## 8. 交付物

| 类型 | 路径 |
|---|---|
| Python 包 | `cartigsfm/` |
| 字典 | `cartigsfm/resources/dictionary_v1/cartilage_dictionary_v1.json` |
| RAG 知识库 | `cartigsfm/resources/rag_v1/p6_cartigsfm_knowledge_base.json` |
| 分类器 v1 | `cartigsfm/resources/cs_classifier_v1/classifier.pt` |
| 分类器 v2 | `cartigsfm/resources/cs_classifier_v1/classifier_v2.pt` |
| HVG 基因表 | `cartigsfm/resources/cs_classifier_v1/hvg_genes.tsv` |
| 训练脚本 | `scripts/train_cartigm_classifier.py`, `scripts/train_classifier_v2.py` |
| 评估脚本 | `scripts/eval_classifier_lbo.py`, `scripts/eval_ensemble.py` |
| 测试 | `tests/` (78 tests, 18 skipped) |
| 报告 | `reports/` |

---

## 9. 后续方向

1. **Cross-batch 泛化**:当前 LBO 平均 57.5% 是主要瓶颈,需更多 batch-diverse 训练数据或 batch correction 预处理 (Harmony / scVI)。
2. **scGPT encoder 融合**:将 P16 预训练的 transformer encoder 作为 cs_classifier 的前端,潜在提升表征能力。
3. **3 个 atlas-only 亚型验证**:Hypoxic / Metabolic_Stress / Superficial_Zone 需在含这些亚型的独立数据上验证。
4. **字典扩展**:functional_axis 可继续扩展 (如自噬亚型、铁死亡细分、表观调控)。
5. **CLI 集成深化**:将 fusion / scgpt-pretrain 训练流程封装为端到端 CLI。
