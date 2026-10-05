# Pilot 启动快照（2026-10-05）

已完成：BUSBRA 下载与校验、真实 schema 审计、744/160/160 Case 划分、QA、固定区域与协议锁、50 张开发图像审核；11 项 CPU 测试通过。真实 7B 模型的 16 图单卡工程检查通过，5,046,272 个 LoRA 参数，峰值已分配显存 17.21 GiB。生成与 teacher forcing 最大 logprob 差 0.000318，checkpoint 恢复后计分相同。

| 方法 | 本轮启动时状态 | 性能 |
|---|---|---|
| B0 | 已在 GPU 0 后台运行，启动检查完成 28/120 Case | 尚未完成 |
| B1 | 审计与 32 Case 短拟合后执行共同 SFT | 尚未完成 |
| B2–B6 | 仅在 SFT 后证据 gate 通过时，各最多 100 步 | 尚未运行 |

这只能说明接口和梯度可运行，不能证明证据依赖改善、RL 必要或临床有效。固定 120 Case 审计只有 18 图通过区域资格，需同时呈现覆盖偏差及负结果。测试集保持封存；正式多 seed 尚未授权。

当前文件是启动快照。后台实时状态为 outputs/pipeline_status.json；流程退出后自动生成 outputs/reports/pilot_report.md 和 frozen_audit.md，包含真实结果或失败原因。运行、阶段日志与 checkpoint 均在 pro5000 的指定 data 盘。默认不创建持续监测任务。

运行标识：pilot-2862649。启动检查确认 supervisor 已脱离 SSH 会话、日志正常、GPU 0 正在计算，无立即失败。代码版本：3735bd5（codex/r812）。

下一条命令为 `python3 scripts/status.py`，只查询一次实时状态；不重复启动 pilot。
