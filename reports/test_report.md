# 实际验证记录

CPU 工程测试：11 项通过。覆盖官方校验失败、ZIP CRC 损坏与路径穿越/符号链接、断点下载 Range 校验、病例及重复组隔离、半开框/归一化逆变换、token 覆盖、同面积/同 token 对照、非原地替换、答案 span 与 padding、严格格式奖励、零方差 GRPO、辅助梯度和零门控、KL stop-gradient、非法框以及 Case 聚类 bootstrap。

真实 Qwen2.5-VL-7B 单卡检查：16 张训练图，视觉 merger/window 逆排列与 processor grid 一致；重复计分和空干预结果一致；监督、可微 evidence 和 stability 均产生 LoRA 梯度，零质量门控为零梯度；保存/重新加载 PEFT checkpoint 后答案分数相同。约 5,046,272 个可训练参数，峰值已分配显存约 17.21 GiB。大 hinge margin 只用于工程梯度检查，不是训练超参数，也不是医学实验结果。

增加了 cached generation 与 teacher forcing 的分数对照，以及完整前向次数计数检查。初版 root module hook 没有捕获 PEFT 直接调用的 teacher-forcing forward，已修正为明确计数 score 路径、hook 计数 generation 路径，旧记录保留。

尚未验证：真实模型多卡数值一致性、正式多 seed 复现、测试集泛化、临床证据充分性。单卡验证不代替这些检查。

最后一次检查实际统计 148 次 teacher-forcing forward + 3 次生成 forward = 151；生成/teacher-forcing 最大 logprob 差 0.0003181167，检查通过。
