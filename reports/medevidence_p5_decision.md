# MedEvidence P5 实际诊断与决策

诊断状态：completed；主SFT/RL对照：not_started_gate_failed。入选坐标格式：None。

本轮固定比较两种坐标接口，实际均从同一历史COV初始化；Fit32与Calibration64不重合，但两组都来自历史训练池，不是独立验证。原开发集未新增生成或图像读取。

P4已有D3训练rollout：32个阳性组中26组奖励可区分，其中23组没有任何IoU≥0.5匹配；只有3组含匹配区域。空/非空状态间方差占总组内奖励方差70.7062%。另有20组存在固定非空框数下的几何奖励差异；该差异不等于正确定位。47个无匹配completion获得正优势。阴性奖励由-1改-5后，组内标准化优势不变的数值检查通过。

| 指标 | normalized | absolute |
|---|---:|---:|
| 初始Fit阳性严格成功 | 0/16 | 1/16 |
| 初始Fit阴性出框 | 0/16 | 0/16 |
| 最终Fit阳性严格成功 | 3/16 | 7/16 |
| 最终Fit阴性出框 | 0/16 | 0/16 |
| 初始Calibration阳性严格成功 | 0/32 | 0/32 |
| 初始Calibration阴性出框 | 0/32 | 0/32 |
| 最终Calibration阳性严格成功 | 2/32 | 2/32 |
| 最终Calibration阴性出框 | 7/32 | 6/32 |
| 最终Calibration单框严格成功 | 1/16 | 2/16 |
| 最终Calibration单框平均IoU | 0.222193 | 0.191306 |
| 最终Calibration匹配GT区域 | 5/49 | 3/49 |
| Calibration自然best-of8单框成功 | 5/16 | 3/16 |
| Calibration阳性几何可区分组 | 29/32 | 28/32 |

门槛逐项结果：

- normalized：fit_positive_strict_at_least12_of16=False；fit_negative_nonempty_at_most2_of16=True；cal_negative_nonempty_at_most4_of32=False；cal_single_best8_at_least4_of16=True；cal_positive_geometric_groups_at_least8_of32=True；no_invalid_or_truncated=True。
- absolute：fit_positive_strict_at_least12_of16=False；fit_negative_nonempty_at_most2_of16=True；cal_negative_nonempty_at_most4_of32=False；cal_single_best8_at_least4_of16=False；cal_positive_geometric_groups_at_least8_of32=True；no_invalid_or_truncated=True。

Calibration配对差值（absolute−normalized），2000次患者bootstrap：

- single_strict_rate：+0.062500，95%区间[+0.000000, +0.200000]，有效2000次。
- positive_strict_rate：+0.000000，95%区间[-0.090909, +0.093750]，有效2000次。
- single_mean_IoU：-0.030887，95%区间[-0.110169, +0.047839]，有效2000次。
- negative_nonempty_rate：-0.031250，95%区间[-0.107143, +0.000000]，有效2000次。
- region_recall：-0.040816，95%区间[-0.166667, +0.070207]，有效2000次。

实际训练与资源：

- normalized：completed，256步，1024患者暴露，训练rollout 0，诊断rollout 512。
- absolute：completed，256步，1024患者暴露，训练rollout 0，诊断rollout 512。

P5新增GPU占用0.373980小时；包含P4后累计0.887491/3小时。新增影像读取：{"normalized": {"train": 96}, "absolute": {"train": 96}}；test实际读取0。模型权重、患者标识、GT、逐例输出、原始日志保持私有。

两个坐标分支均未通过执行前固定的组合门槛，因此本轮在诊断结束；主SFT/RL均0步、0暴露、0rollout，冻结分类分支方案和原生RL校验未执行。没有根据结果降门槛、增加步数、追加seed或使用开发集选优。不能据此判断RL是否有效。

**机制解释。** 两组最终Fit阳性均16/16非空，但严格成功仅3/16与7/16，说明在本次设置下消除空输出还没有解决坐标/多框准确性。Calibration阳性两组均由23/32空输出降至4/32空输出，严格成功仅2/32，同时阴性出框增加到7/32与6/32。绝对坐标提高了训练内严格成功，却没有在校准总体严格成功、区域匹配或自然best-of8上形成一致收益；本轮不支持仅替换坐标格式足以解决问题。

| Calibration教师强制坐标token NLL | 初始 | 最终 |
|---|---:|---:|
| normalized | 1.882244 | 1.906142 |
| absolute | 1.932972 | 1.884796 |

该NLL条件于GT前缀，不是自由生成成功概率；token跨语义分组存在边界限制。存在性、坐标及多框继续输出需要进一步区分，不能只增加出框倾向。EOS损失约1e-5，初始8人梯度检查也很小，当前证据不支持EOS学习不足是主要瓶颈。

token梯度分组为计算诊断，分组范数不可直接相加，也不能单凭范数认定根因。多框syntax组包含继续输出下一框的分隔符，不能把它等同于JSON格式错误。坐标接口效果受旧COV适配历史限制，不代表原生base模型能力。旧pilot2的训练拟合成功采用不同初始化、仅单框阳性及4倍学习率，本次未拟合不能证明该模型没有拟合能力。

本轮无科学执行偏离，无自动重试或追加训练。条件性分类评价未执行，因此没有新的AP或分类能力保留实测声明。完整token分组聚合、初末指标、资源和工程门槛见相邻JSON。
