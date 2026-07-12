# RD-Safe+HR-OTC

本仓库包含对 **Optimal Transport Enhanced Cross-City Site Recommendation
(OTC)** 的复现，以及面向负迁移问题提出的改进方法。

## 方法概述

- **RD-Safe**：利用内部验证集估计源城市可靠性，并通过安全门控抑制负迁移。
- **HR-Rerank**：使用三类商业结构信号对候选区域进行重排序：
  - Category–Region Affinity（类别–区域亲和力）
  - Region–Region Diffusion（区域共现扩散）
  - Region Popularity Prior（区域热度先验）
- **Strict unseen-pair evaluation**：评估训练集中未见品牌–区域组合的泛化能力。

## 实验设置

- Dataset: OpenSiteRec
- Cities: Chicago, NYC, Singapore, Tokyo
- Backbones: VanillaMF, LightGCN
- Metrics: Recall@20, nDCG@20
- Seeds: 2024, 2025, 2026

## 仓库结构

```text
.
├── exp_clean/
│   ├── scripts/       # 数据检查、backbone、OTC、最终方法、消融
│   ├── configs/       # 最终实验配置
│   └── results/       # 公开的 CSV、表格和报告
├── OpenSiteRec/
│   └── baseline/      # 复现所需的上游 baseline 代码
├── data/
│   └── README.md      # 数据下载与放置说明
├── paper/             # 最终论文源文件与图片（如已提供）
├── requirements.txt
└── RELEASE_MANIFEST.csv
```

## 数据说明

OpenSiteRec 原始数据、数据划分、模型权重、embedding 和分数矩阵未上传到仓库。
请阅读 [`data/README.md`](data/README.md)。

## 推荐复现流程

1. 准备 OpenSiteRec 官方数据。
2. 运行 `exp_clean/scripts/00_data_checks/` 中的数据审计与划分脚本。
3. 导出 VanillaMF 与 LightGCN 的多 seed backbone artifacts。
4. 运行多 seed OTC-GW。
5. 运行 RD-Safe+HR-OTC。
6. 运行完整消融实验。
7. 汇总三次随机种子的 mean±std。

各脚本的具体参数请执行：

```bash
python path/to/script.py --help
```

## 主要入口

```text
exp_clean/scripts/01_baselines/run_backbone_artifact_export_multiseed.py
exp_clean/scripts/02_otc_reproduction/run_otc_gw_multiseed.py
exp_clean/scripts/03_main_methods/run_hr_rerank_multiseed.py
exp_clean/scripts/04_ablation/run_complete_ablation_multiseed.py
```

## 已排除内容

- OpenSiteRec 城市原始数据与 `OpenSiteRec.zip`
- `exp_clean/data_splits/`
- `exp_clean/checkpoints/`
- `exp_clean/scores/`
- `.pkl`、`.npy`、`.npz`、`.pt` 等大文件
- 调试、smoke、fast、Sparse 探索和旧版本结果

## 致谢

本项目使用 OpenSiteRec 数据与 baseline 代码。上游项目的 LICENSE 与 README
保存在 `OpenSiteRec/` 下。请同时遵守上游项目的许可要求。
