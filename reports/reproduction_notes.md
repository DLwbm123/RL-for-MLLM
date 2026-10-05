# 原论文与医学适配边界

核对资料：用户提供的 Evidence-RL PDF，arXiv:2608.08021v2，2026-09-28；阅读第 3、4.1、A.1、A.2、A.9、A.11 节。作者代码固定在 `808a092fe87dd8be4b7e88c2d44aff76cec81b47`，从[作者项目](https://evidencerl.github.io/)所链接的[代码仓库](https://github.com/evidencerl/code)获取。第三方全文和代码克隆保存在被 Git 排除的 private 目录。

原文提供的接口：固定图像、问题和候选答案，在空间合并之后用邻近 token 均值替换选定区域；比较证据区域与 K 个对照区域的答案支持度下降；把标准化 margin 经 tanh 门控组合到 GRPO 奖励中。

## 计分语义核对

作者 `src/reward/action_logprob_ate.py` 的 `_mean_logprob_from_logits` 实际计算被选答案 token 的平均对数概率。该代码支持在原响应前缀条件下取 final-answer span，也支持隔离答案字符串；其平均与正文 Eq.2 简写的整段 log likelihood 不能混为逐字相同的定义。代码将孤立答案用 tokenizer 编码，不自动附加 EOS；负对照标准差采用总体标准差，epsilon 为 1e-6。

本项目第一轮无 CoT、无答案前缀，模式 A 固定采用作者代码的 isolated-answer mean、排除 EOS；模式 B 同样排除 EOS、按有效答案 token 平均，并保留可微计算图。两模式在这个受限场景数值可相同，接口及梯度用途不同。每次另存序列 logprob 总和、长度和均值。SFT 单独把 EOS 纳入监督目标，以学习终止；所有可比方法的监督项一致。

## 明确属于适配的部分

- 使用 BUS-BRA 金标准掩膜及框，替代 COCO 弱目标建议；E2 为主。不是重现原训练数据或九个通用基准。
- 严格标签正确=1、错误=0、格式非法=-0.1。正文附录的复合 R_ans、公开代码的错误=-0.25 和二元任务路由均不等同于此设计。B4 是 CED-adapted 基线，主动检验二元病理任务上的适用性，不能称为原文完整复现。
- 对照匹配像素面积、矩形形状、视觉 token 数和深度；排除病灶安全边界、后方列、黑背景及文字/标尺候选。不采用原代码中的任意 token 子集回退。
- 语言部分所有层 q_proj/v_proj 的 LoRA r=16、alpha=32；冻结视觉编码器和 merger。与原论文最后四层更新不一致。
- 可微相对/绝对 evidence hinge、对照稳定性 KL、共同监督 anchor 均为本项目待验证设计。CED reward 本身 detach；GRPO 梯度来自策略目标。参考模型 KL 系数统一为 0，未实现另一套隐含 KL。
- 相同图像、问题、答案、参数和固定区域的奖励必须相同。二元组零方差时 advantage 为零，只能由独立监督/辅助项更新，不能套用原文 same-answer-different-reward 的统计解释。

## 工程核对与解释限制

固定 Transformers 4.51.3 的 Qwen2.5-VL visual.forward 先 window_index 重排，merger 后用 argsort(window_index) 恢复空间顺序。替换发生在恢复后的输出；真实模型 smoke 比较 merger 钩子输出与恢复排列，另外核对 processor grid、占位数及特征数。

全图编码器已经混合上下文，局部替换不等于完全删除病灶信息。像素强模糊只是压力测试，不产生“病灶已消失”真值。原图保留的测量符号和文字仍是潜在捷径。临床充分性标签、跨图同病灶身份、医学因果性及临床安全性均未获验证。

Pilot 的进入门槛在模型审计前冻结：几何/工程检查通过、32 Case 拟合分类监督损失下降超过 10%；SFT 审计至少 10 个合格 Case，特征和像素 M 及相对偏移区域效应的配对 95% 下界均大于 0，面积/token 数相关绝对值小于 0.5。它们仅用于有限 pilot 的可行性筛查，未声称是医学标准或正式研究所需样本量；不把 Pr(M>0)=0.5 当作随机零假设。未通过时停止 B2–B6 并保留失败，不调整测试集或反复搜索参数。代码起草时的 30 Case 占位门槛在仅有数据几何信息、尚未进行模型计分时改为 10，并随最终配置锁定；这一变化没有改变划分或审计病例选择。

真实分词已确认 benign=2、malignant=3 tokens。第一轮主计分保持原标签平均 logprob，不按测试选择编码；在审计前固定首 8 个审计 Case 的 A=benign/B=malignant 单 token 格式诊断。它同时改变答案格式，不能解释成只改变长度的完全受控实验。
