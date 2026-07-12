# Data Preparation

本仓库不包含 OpenSiteRec 原始数据、官方划分、内部验证集或严格测试集。

## 1. 获取 OpenSiteRec

请根据 OpenSiteRec 官方仓库的说明下载数据，并在仓库根目录创建：

```text
OpenSiteRec/
├── Chicago/
├── NYC/
├── Singapore/
└── Tokyo/
```

每个城市至少需要官方提供的 `split/train.pkl` 与 `split/test.pkl`，
以及 baseline 训练所需的品牌、区域与图结构文件。

## 2. 建立 exp_clean 官方划分

最终代码预期以下结构：

```text
exp_clean/data_splits/official/
├── Chicago/train.pkl, test.pkl
├── NYC/train.pkl, test.pkl
├── Singapore/train.pkl, test.pkl
└── Tokyo/train.pkl, test.pkl
```

## 3. 生成内部验证集和 strict unseen-pair

使用：

```text
exp_clean/scripts/00_data_checks/create_internal_valid_splits.py
exp_clean/scripts/00_data_checks/create_strict_unseen_test_splits.py
```

数据文件被 `.gitignore` 排除，不应提交到 GitHub。
