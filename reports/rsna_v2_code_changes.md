# RSNA diagnostic v2 实现差距与变更

| 历史问题/要求 | 实际修改 | 验证 |
|---|---|---|
| 排程硬编码A/A/A/B | experiment.schedule接收task_pattern；v2固定分类排程 | N两轮、B逐步2+2、仅负类不替换，CPU通过 |
| 生成参数写死 | Model.generate构造并保存实际GenerationConfig | 实际传入参数检查通过 |
| 封存与阶段门禁 | DevelopmentData白名单及open审计；P1门禁；旧训练入口拒绝v2 | 实际读阻断、冻结更新阻断通过 |
| 阈值/奖励诊断缺失 | src/v2.py统一统计、患者fold权重和奖励组比较 | 边界/ties、NA、配对、epsilon检查通过 |
| 分阶段实验缺失 | src/v2_run.py实现P0/P1/P2/P3/P4；R0条件不足blocked | GPU科学结果待运行 |
| 并行预算与可靠后台 | rsna_v2_pipeline.py两独立单卡worker、累计6GPU小时、禁止重试 | 单/双卡失败计费及重复启动拒绝通过 |
| 协议和报告 | prepare_rsna_v2、report_rsna_v2、dispatch_rsna_v2 | 预检和24项CPU测试通过，完整输出留存 |

未更改候选标签的历史logprob计算。P1用历史argmax gate，.5阈值单列，不覆写v1。复用标准库subprocess及既有模型/区域函数，无新增依赖、DDP、RL或发布自动化。

真实可执行命令清单在私有rsna_v2_launch_commands.md，配置/执行源码身份在私有协议锁。重复启动会被拒绝，不能用于未经授权的续跑。
