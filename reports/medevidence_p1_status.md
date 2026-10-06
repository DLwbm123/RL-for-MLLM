# MedEvidence pilot-1 完成状态

实验状态：**completed**。GPU阶段于2026-10-06 21:35:53（北京时间）结束，已核验实际摘要和输出；从正式启动到GPU阶段结束约44分29秒。累计保守计费1.822674 GPU小时（含加载、smoke、两次0更新preflight失败及全部评价），上限6小时，最多3个独立单卡作业。

| 项目 | 状态 | 实际覆盖与输出 |
|---|---|---|
| 数据与视图冻结 | completed | 1024训练、256开发；各3名AI辅助阳性证据患者、各32名阴性假干预；`protocol/medevidence_p1/freeze_summary.json` |
| CPU与工程smoke | completed | 8步smoke、恢复与梯度审计；`medevidence_p1_engineering.md` |
| 两次smoke preflight | failed | 均首个optimizer更新前停止；失败原因与39.006秒计费保留，非正式训练重试 |
| M0 | completed | 512/512；512固定最终checkpoint；`medevidence_p1_training.json` |
| W | completed | 256/256；只训练一次，保留128/256 checkpoint |
| M1/M2 | completed | 各256/256续训、累计512；实际相同W恢复及暴露一致性PASS |
| 统一评价 | completed | M0/M1/M2各256/256；固定checkpoint、阈值和视图；`medevidence_p1_results.md/json` |
| 统计与训练审计 | completed | 患者配对2000次seed42；多框/多视图成簇；`medevidence_p1_training.md/json` |
| 研发决策 | completed | dep增量效用尚未支持，定位仍严重不足；`medevidence_p1_decision.md` |
| 后续扩展 | skipped | 未增加seed、更新、数据下载、RL或超参数搜索 |

本轮新监督/evidence训练及公开发布已获授权。代码、冻结配置、聚合审计和最终报告作为本次公开提交内容；实际推送与匿名访问验证以交付回执为准。患者数据、图像、标识、坐标、逐例审核/排程/预测、模型权重及原始日志保持私有。

私有输出相对运行目录：`outputs/{M0,W,M1,M2}/step_*`、`outputs/eval_{M0,M1,M2}/predictions.jsonl`、`outputs/protocol/`、`outputs/gpu_ledger.json`和`logs/`。本报告不给出私有绝对路径。实际可执行入口见协议；重复正式启动被拒绝。
