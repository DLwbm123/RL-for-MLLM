# Medical image evidence-dependence pilots

包含已结束的 BUS-BRA pilot 和独立 RSNA 胸片 pilot，检验医学短答案对候选病灶区域的依赖。当前是方法可行性与工程验证，未证明 RL 必要、方法有效或具有临床安全性。

来源：[BUS-BRA 官方数据](https://zenodo.org/records/8231412)、[原始论文](https://doi.org/10.1002/mp.16812)、[Evidence-RL 作者项目](https://evidencerl.github.io/)、[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)。数据使用需引用 Gómez-Flores、Gregorio-Calas、Pereira，Medical Physics 51, 3110–3123 (2024)。Zenodo 记录为 CC BY 4.0；压缩包另带引用及许可文本，一并保留，不用作者代码许可证替代数据许可。

## 路径与运行

代码在当前仓库；数据、权重、缓存、checkpoint、图像审核页与逐样本输出在指定数据盘。设置 `DATA_ROOT`、`MODEL_ROOT`、`CACHE_ROOT`、`OUTPUT_ROOT`。远程启动还需 `REMOTE_ROOT`、`REMOTE_BASE_ROOT`、`REMOTE_PYTHON`、`GPU_INDEX`，本次部署的 host 为用户指定的 pro5000。真实部署值保存在被 Git 排除的 `private/runtime.json`。

```bash
PYTHONPATH=. python -m scripts.download_busbra
PYTHONPATH=. python -m scripts.prepare_manifest
PYTHONPATH=. python -m scripts.prepare_regions
PYTHONPATH=. python -m pytest -q tests
python3 scripts/sync.py
python3 scripts/dispatch.py smoke
python3 scripts/dispatch.py pilot
python3 scripts/status.py
```

远程 GPU 工作进程的命令行为中性的 `python -u -`；方法和路径通过环境和 stdin 传递。后台流程为 B0 冻结审计 → 32 Case 短拟合 → B1 共同 SFT → B1 冻结审计 → gate → B2–B6 各最多 100 步。任何失败或 gate 不通过均停止受影响后续步骤并保留日志。没有自动启动正式实验或定时监测。

## 产物

- `reports/data_audit.md`：实际统计、异常及协议边界；完整清单/QA/划分在私有数据盘 `outputs/protocol/`，本地副本在 `private/protocol/`。
- `reports/reproduction_notes.md`：原文、代码语义和医学适配差异。
- `reports/environment.md`、`requirements.lock`：实际环境及版本。
- `reports/pilot_report.md`、`reports/frozen_audit.md`：BUS-BRA 已完成的聚合结果及 gate 失败记录；RSNA 单独记录。
- `outputs/pipeline_status.json`、分阶段日志、逐样本 JSONL 和 checkpoint：可恢复的私有运行记录。
- `reports/formal_run_plan.md`：未授权的正式阶段计划。

分类使用候选标签条件似然；主计分排除 EOS。SFT 监督包括 EOS。任务 A、B、C 独立构建，首轮只训练 A+B。测试集未开启；使用图像级指标及 Case 聚类 bootstrap，不把尚未确认的多图关系拼成患者诊断。

`scripts/expert_review.py` 可生成本地专家审核工具，标签初始全部为空。没有足够临床标注前，不训练或声称具备“证据不足识别”。

公开边界：仅代码、配置、聚合报告及必要小图可发布；原始图像、患者级清单、逐样本预测、模型与 checkpoint、第三方全文和原始日志不得提交。实际实验完成后按用户项目交付规则整理公开结果；运行中不提前声称完成交付。

## RSNA independent pilot

配置 `configs/rsna_pilot.json`。使用 RSNA 2018 adjudicated MD.ai archive 的 Calculated 标签及 NIH 文件名前缀患者键，重新建立患者互斥的 70/15/15 划分；不是 Kaggle 官方划分复现。固定 seed 17，选取 1,024 名训练患者及 256 名验证患者，每名患者一个预定图像。测试图像不解码、不评估。仅准备这些开发图像，保留原始下载包。

设置 `DATASET_NAME=rsna`、`DATA_ROOT` 为 RSNA 下载目录、独立 `OUTPUT_ROOT`、`RUN_CONFIG` 指向 RSNA 配置，执行 `scripts.prepare_rsna`，再用 `scripts/dispatch_rsna.py` 启动真实模型 smoke 与 `scripts.pipeline` 的独立后台链。固定流程 B0 → 32 例短拟合 → 一轮 B1 SFT → B1 证据审计。`diagnostic_only=true` 禁止自动执行 B2–B6；后续需另行授权。

no/yes 表示是否有提示肺炎的肺部不透明影，不是病原确诊标签。无最终裁定标签的图像不当作阴性；无框阴性仅参与分类；多框阳性用于分类和包围全部标注的定位，机制分析限单框病例。几何对照未由临床专家确认为正常肺区，亦未完成肺野分割；结果必须带上此局限及资格覆盖率。
