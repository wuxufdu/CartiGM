# CartiGSFM 用法指南 (cartigsfm v0.6.1)

## 1. 安装

```bash
# 从 GitHub 安装（含训练好的 cs_classifier 权重）
git clone https://github.com/wuxufdu/CartiGM.git
cd CartiGM
pip install -e .

# 推理 + 训练需要这些额外依赖
pip install torch anndata scanpy scipy
```

安装后命令行入口 `cartigsfm` 可用，也可通过 `python -m cartigsfm` 调用。

```bash
cartigsfm --help           # 查看所有子命令
python -c "import cartigsfm; print(cartigsfm.__version__)"  # 0.6.1
```

---

## 2. 三大核心功能

| 功能 | 命令 | 输入 | 输出 |
|---|---|---|---|
| **基因列表打分** | `score` | marker 基因列表 | 轴排序 + 分数 |
| **独立数据验证** | `p4-project` | h5ad / pseudobulk | 逐 cluster 轴投影 + 报告 |
| **单细胞分类** | `cs-predict` | h5ad | 逐细胞亚型 + 概率 |

---

## 3. cs-predict：单细胞亚型分类（P-F 模型）

这是本项目的核心产出：一个在 acc atlas（41.6 万细胞）+ EBR 自测数据（3.2 万细胞）上联合训练的 10 类软骨细胞亚型分类器。

### 3.1 基本用法

```bash
# 默认 ensemble 模式（v1+v2 平均 softmax），自动用 CUDA
cartigsfm cs-predict --h5ad your_data.h5ad --out predictions.tsv
```

输出 TSV 每行一个细胞，包含预测亚型 + 10 类概率：

```
cell_index  predicted_cell_subtype                   max_prob   Effector_Metabolic_Chondrocytes  ...  Superficial_Chondrocytes
0           Homeostatic_Chondrocytes                 0.847      0.021                            ...  0.003
1           Effector_Metabolic_Chondrocytes           0.623      0.623                            ...  0.001
```

### 3.2 参数

```bash
cartigsfm cs-predict \
  --h5ad data.h5ad \
  --out preds.tsv \
  --mode ensemble          # v1 / v2 / ensemble（默认）
  --layer log1p_norm       # 指定 adata layer（默认 .X）
  --device cuda            # cuda / cpu（默认自动）
  --batch-size 4096        # 推理 batch
  --ckpt /path/to/own.pt   # 用自己的 checkpoint 覆盖
```

### 3.3 模型选择

| 模式 | 模型 | within-cluster cell acc | LBO 平均 | 适用场景 |
|---|---|---|---|---|
| `--mode v1` | 60 epoch MLP | 76.6% | 48.8% | 老基准 |
| `--mode v2` | 100 epoch + MixUp + noise | 76.9% | 57.5% | 跨批次数据 |
| `--mode ensemble`（默认） | v1+v2 softmax 平均 | 77.6% | — | 推荐默认 |

模型自动将输入基因 align 到内置 2000 HVG basis（缺失基因补零），所以任意基因集的 h5ad 都能直接跑。

### 3.4 Python API

```python
import anndata as ad
import numpy as np
from cartigsfm.cs_classifier import (
    load_bundled_classifier,
    bundled_classifier_v2_path,
    load_classifier,
    align_to_genes,
    predict_from_array,
    predict_ensemble,
)

# 加载内置 ensemble
m1, classes, genes, cfg = load_bundled_classifier()          # v1
m2, _, _, _ = load_classifier(bundled_classifier_v2_path())  # v2

# 读数据并 align
adata = ad.read_h5ad("data.h5ad")
X = adata.X  # log1p normalized; 若 .X 有负值用 adata.layers["log1p_norm"]
X = np.asarray(X.todense() if hasattr(X, "todense") else X)
Xa, n_hit = align_to_genes(X, list(adata.var_names), genes)
print(f"HVG hit: {n_hit}/{len(genes)}")

# 推理
idx, probs = predict_ensemble(Xa, [m1, m2], classes, device="cuda")
pred_labels = [classes[i] for i in idx]
```

### 3.5 自己的数据注意事项

- 输入必须是 **log1p-normalized** 表达矩阵（与训练数据同尺度）
- 若 .X 被 scVelo 等污染含负值，用 `--layer log1p_norm`
- 基因符号需在 var_names（HGNC symbol）；非符号 ID 会自动跳过
- HVG 命中率低于 80% 时建议检查基因命名

---

## 4. p4-project：独立数据轴投影验证

把任意单细胞数据投影到三层字典（53 轴），得到每个 cluster 的 top 轴打分。用于验证字典在你的数据上的区分度。

### 4.1 h5ad 输入

```bash
cartigsfm p4-project \
  --h5ad your_data.h5ad \
  --outdir P4_results \
  --sample-col sample \
  --tissue-col tissue \
  --cluster-col leiden_res1 \
  --layer log1p_norm
```

自动检测 obs 列名（也可手动指定）。大文件（>2GB）自动切到 streaming backed 模式。

### 4.2 pseudobulk TSV 输入

```bash
cartigsfm p4-project \
  --pseudobulk genes_x_cluster.tsv \
  --meta cluster_meta.tsv \
  --outdir P4_results
```

### 4.3 输出

`outdir` 下包含：
- `tsv/top_axis_per_cluster.tsv`：每个 cluster 的 top 轴 + 分数
- `tsv/celltype_crosstab.tsv`：celltype × top axis 交叉表（若有 celltype 列）
- `docs/P4_REPORT.md`：markdown 报告

---

## 5. score：基因列表打分

输入一组 marker 基因，输出与字典 53 轴的匹配排序。

```bash
# 准备基因列表文件（每行一个基因）
echo "COL2A1
ACAN
SOX9
COL11A1
COMP" > markers.txt

cartigsfm score --query markers.txt --kind both --top 10
```

`--kind subtype` 只跑 10 个亚型轴；`--kind function` 只跑 39 个功能轴；`--kind both` 全跑。

---

## 6. dictionary-v1：查看字典结构

```bash
# 总览
cartigsfm dictionary-v1
# 输出: version v1.8.6, 53 axes (cell_subtype=10, tissue_developmental_state=4, functional_axis=39)

# 查看某层全部轴
cartigsfm dictionary-v1 --layer cell_subtype --show-axes

# 导出轴表
cartigsfm dictionary-v1 --show-axes --out axis_table.tsv
```

---

## 7. interpret：证据约束解释

对基因列表或 P4 打分表做证据约束的生物学解释，输出带 RAG 引用的解读。

```bash
cartigsfm interpret --query markers.txt --out interpretation.md
cartigsfm interpret --p4-table P4_results/tsv/top_axis_per_cluster.tsv --out interp.md
```

---

## 8. inspect-h5ad：自动识别 obs 列

在跑 p4-project 之前快速看一下 h5ad 结构，自动推荐 sample/tissue/cluster/celltype 列名：

```bash
cartigsfm inspect-h5ad --h5ad your_data.h5ad
```

---

## 9. 10 个细胞亚型一览

| axis_id | 名称 | 角色 |
|---|---|---|
| Effector_Metabolic_Chondrocytes | 效应代谢型 | 活跃代谢、脂质可塑性 |
| Progenitor_Chondrocytes | 前体型 | 基质重塑、间质特征 |
| Homeostatic_Chondrocytes | 稳态型 | II 型胶原 / 聚集蛋白聚糖维持 |
| Hypoxic_Chondrocytes | 低氧适应型 | 低氧应答、血管化抵抗 |
| Metabolic_Stress_Chondrocytes | 代谢应激型 | 核糖体应激、翻译失衡 |
| Inflammatory_Response_Chondrocytes | 炎症应答型 | NF-κB / IL1 / TNF |
| Prehypertrophic_Matrix_Chondrocytes | 肥厚前型 | RUNX2 / COL10A1 过渡 |
| Fibrocartilage_Chondrocytes | 纤维软骨型 | I 型胶原、纤维特征 |
| Superficial_Zone_Chondrocytes | 表层带型 | PRG4 / 滑液界面 |
| Reparative_Stress_Chondrocytes | 修复应激型 | IEG / 修复信号 |

---

## 10. 训练自己的分类器

```bash
# 训练脚本（需 GPU）
python scripts/train_cartigm_classifier.py        # v1: 60 epoch 基准
python scripts/train_classifier_v2.py             # v2: 100 epoch + MixUp + noise

# 评估
python scripts/eval_classifier_lbo.py             # leave-batch-out 评估
python scripts/eval_ensemble.py                   # ensemble 评估
```

训练数据准备见 `scripts/_remote_extract_train_subset.py`（从 atlas + EBR 抽 HVG 子集到 npz）。

---

## 11. 常见问题

**Q: cs-predict 报 "checkpoint not found"**
A: 内置权重在 `cartigsfm/resources/cs_classifier_v1/`，确认 pip install 时打包了 .pt 文件（`include_package_data=True`）。

**Q: HVG 命中率低**
A: 检查 var_names 是不是 HGNC gene symbol；若是 Ensembl ID 需先转换。

**Q: GPU 不可用**
A: `--device cpu` 降级到 CPU 推理，10 万细胞约 30 秒。

**Q: .X 含负值**
A: 用 `--layer log1p_norm` 指定正确的归一化层。
