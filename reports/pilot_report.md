# BUS-BRA pilot 完成报告

状态：completed_with_gate_stop。B0、32 Case 短拟合、B1 SFT 和 B1 审计完成；预先冻结的证据 gate 失败，B2–B6 未运行。测试集未开启。

| 项目 | 实际结果 |
|---|---:|
| 32 Case 拟合损失（前 → 后） | 1.285333 → 0.190434 |
| B1 SFT 更新数 | 186 |
| B1 完整 validation checkpoint 选择 AUROC | 0.667489 |
| B0 固定 120 Case 审计 AUROC | 0.670149 |
| B1 同一 120 Case 审计 AUROC | 0.719848 |
| 两次审计合格区域数 | 18/120 |

完整 validation 与 120 Case 审计的分母不同，不能混用。单次 seed 结果不证明 RL 必要或临床有效。B1 在固定 0.5 阈值下 sensitivity=0、specificity=1、balanced accuracy=0.5；AUROC 提升没有解决默认阈值下的类别偏置。

| B1 证据指标 | 均值 | 95% Case bootstrap 区间 |
|---|---:|---|
| feature M | 0.007183 | [-0.008010090266664823, 0.02218355035271357] |
| feature evidence_minus_wrong | 0.006531 | [-0.007850694273495011, 0.02595605860567754] |
| pixel M | 0.011087 | [-0.001269422662961813, 0.0245557245850149] |
| pixel evidence_minus_wrong | -0.009241 | [-0.021129802407489885, 0.0016113675665110308] |

上述四个区间下界均未大于 0，依预先规则停止后训练，保留负结果。工程几何检查不等于临床专家确认对照区域正常。

全部聚合统计见 busbra_results.json。原始图像、病例清单、逐样本预测、checkpoint 和日志保留于私有数据盘，不公开。执行代码版本 3735bd5；后续 RSNA 适配未用于这些历史结果。
