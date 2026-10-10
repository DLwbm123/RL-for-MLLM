# 七个 RL 方法的原文与作者代码核对

本文件保留修复前提交的审阅证据；其中五项问题及后续作者对齐已在[2026-10-10 修复说明](rl_methods_source_fixes.md)中处理，当前实现以该说明和新验证回执为准。

2026-10-10。核对对象为提交 `8d98fba74ed76eefecf8206671b810d97e1567f9` 的七方法实现及其 scorer、参考适配器、训练调用和测试。结论是：**不能确认七个方法均已正确实现。AXPO 有两处核心逻辑错误；CFPO 有可复现的数值问题，且与公开作者实现存在多项实质差异。其余方法的已检查机制基本成立，但其中一些只是任务适配。**

本轮只做源码审阅及人工 CPU 反例检查，没有加载模型、采样真实轨迹、执行 optimizer、读取患者数据或使用 GPU。算法、配置和原测试保持原样；本报告列出修复建议，不表示问题已修复。上一轮 31 项检查证明工程路径能执行，不能证明与原算法一致。

## 作者代码的可核对范围

| 方法 | 原文及作者来源 | 本次取得的代码范围 |
|---|---|---|
| ViSurf | [论文](https://arxiv.org/abs/2510.10606)、[作者模型卡](https://huggingface.co/Ricky06662/Visurf-7B-Best-on-gRefCOCO)提供的 [仓库链接](https://github.com/dvlab-research/ViSurf) | 公共仓库 API 返回 404，未取得作者实现；不能据此断言仓库不存在或从未发布 |
| RL-ZVP | [论文](https://arxiv.org/abs/2509.21880)、[作者项目页](https://bltnynk.github.io/publications/rl-zvp/) | 页面源码仍标注 Code To be released，未取得公开训练代码 |
| PAPO | [论文](https://arxiv.org/abs/2507.06448)、[作者仓库](https://github.com/mikewangwzhl/PAPO/tree/fb63bf4533b78da5a2bb17bc812375dcb620887d) | 固定 revision `fb63bf4533b78da5a2bb17bc812375dcb620887d`；核对 core_algos、dp_actor、papo_utils、ray_trainer、dataset 和配置 |
| CFPO | [论文](https://arxiv.org/abs/2606.23206)、[作者仓库](https://github.com/Raven-July/CFPO/tree/98ca5a4123be2e85e94855fead46c9a385611136) | 固定 revision `98ca5a4123be2e85e94855fead46c9a385611136`；核对 value 干预、actor、缓存概率调用、损失和运行配置 |
| ACTIVE-o3 | [论文](https://arxiv.org/abs/2505.21457)、[作者仓库](https://github.com/aim-uofa/Active-o3/tree/a862f0e0951e082d8af4adb3ca4c79e2809077ae) | 固定 revision `a862f0e0951e082d8af4adb3ca4c79e2809077ae`；树中只有 7 个文件，没有训练实现 |
| AXPO | [论文](https://arxiv.org/abs/2605.28774)、[作者项目页](https://byungkwanlee.github.io/AXPO-page/) | 页面把代码标为 NVIDIA Internal Use Only、Coming soon to public；未访问内部仓库 |
| DeFacto | [论文](https://arxiv.org/abs/2509.20912)、[作者仓库](https://github.com/tinnel123666888/defacto/tree/471a747bc416a59e8ddad22de2e9b038c0c0aeba) | 固定 revision `471a747bc416a59e8ddad22de2e9b038c0c0aeba`；核对奖励插件、GRPO trainer 和训练脚本，没有复刻数据构造流水线 |

仓库、raw 文件和 GitHub Pages 获取均通过本机代理。第三方源码仅保存在私有核对目录，未复制到公开项目源码。

## 逐方法结论与适配差异

| 方法 | 核心逻辑判断 | 必要适配及不能混同的差异 |
|---|---|---|
| ViSurf | Eq.6–7 的标签加入组优势、达到标签奖励后平滑为原组均值，与 §3.4 对齐；单次采样、单次更新也成立 | 真实框替代分割或其他标签是任务适配；没有推理标注，所以纯框接口不加 thinking reward 合理。作者强调标签输出风格对齐，本地只用 compact JSON，没有验证它与模型偏好格式一致。论文明确在实作中移除 KL，本地所有轨迹仍加 beta=.01 的参考 KL，包括标签轨迹；这不是必须的任务适配。论文使用抽象序列概率写目标，本地沿用 token 平均；作者代码不可得，不能确认其实际归约规则 |
| RL-ZVP | Eq.5 的三分支、逐响应最大熵、完整词表熵及负组方向正确；不是把负组简单写成 `-alpha*entropy` | 二元 reward 与纯框输出是任务适配。当前每组只更新一次，旧策略下冻结熵与当前策略的起点相同；不能将此实现直接推广成多轮 PPO 更新而不重新核对熵语义。单 token 或所有 token 熵相同的全错响应仍无该优势信号，这是公式本身的退化，不是遗漏。原文脚注说明省略 KL 展示，因此不能仅因正文未写 KL 就判定本地参考项错误 |
| PAPO | 概率比 `corrupt-factual`、low_var_kl 的 ±20 和 k3 限幅、最大化 perception KL 与降低双侧 sampled-surprisal 的符号均对齐作者代码 | 本地两侧重算并回传，对应作者可选 `RECOMPUTE_AUG_LOG_PROBS=True`，而非作者默认缓存扰动概率路径。不能把 sampled negative logprob 称为完整分布熵。作者在原图上按 14×14 像素 patch 做独立 Bernoulli 遮挡；本地按合并视觉网格选择固定数量，且强制至少遮一个、至少留一个。这不是同一遮挡分布，也不是必须的任务变化 |
| CFPO | 把高注意力边的 value 换成均值、保持非干预边，是论文 Eq.9–13 的合理实现；但本地不是公开作者代码的同一目标或同一干预路径 | 明确的论文／代码冲突及本地差异见下一节。Qwen2.5-VL 4.51.3、单图、无 cache 的 scoring 限制已标明；双 partial backward 对本地联合目标成立，不能证明这个目标就是作者目标 |
| ACTIVE-o3 | 可训练观察策略与固定任务模块分离、单步并行裁剪、heuristic 加 task reward，符合论文主要结构 | 本地固定模块是初始 adapter 的 yes/no scorer，论文检测模块会输出裁剪内定位结果再映射评估。本地直接把裁剪框当作最终证据框并做严格 GT 匹配，是更强的任务改变；启发式权重、二元回答合并和阳性必须有框都是项目定义。作者训练代码不可得，不能声称奖励逐项一致。两框上限与多病灶严格成功有输入支持问题 |
| AXPO | 全错工具子组触发、前缀包含开头 tool_call tag、每前缀续写独立标准化、恢复标志、重复前缀与来源续写屏蔽，符合 §3.1–3.3 | 候选排序和未选中轨迹优势有错误，见发现 1–2。单步裁剪代替多轮工具、无模型生成最终工具后回答，是项目范围变化。默认 G=4 加 K=4 的额外成本可达 100%，原文按 batch 全局预算并 breadth first 分配且 r<1；本地逐样本上限不是其全局分配。把 G+K 条有效 loss 等权平均也不等同于原文 Eq.5 在标准 GRPO 上逐前缀加入贡献的写法 |
| DeFacto | pos/cf/rand 三种视图、cf 奖励 Unknown 且对“猜中原答案”额外惩罚、非 cf 的正／负区域 IoU 与 cf 无 selection reward，保留主要机制 | 真实框替代 DINO-X、只对登记完整证据的阳性赋 Unknown，以及阴性空框接口，是医学任务的必要边界。JSON 取代 think/bbox/answer、只有 yes/no/unknown、没有逐步区域探索，是任务缩减。奖励权重并非作者默认：本地 gamma_unk=1、gamma_guess=1、gamma_corr=1、负区域系数=.5、format=.1、selection=.25；作者插件默认 .2/.6/.9、负区域系数=1，脚本组合权重为 1/.2/.2。作者答案判定支持语义 API，本地严格枚举也不同。这些不能统称“忠实复现” |

## CFPO 的论文与作者代码并不完全一致

[作者干预函数](https://github.com/Raven-July/CFPO/blob/98ca5a4123be2e85e94855fead46c9a385611136/verl/models/transformers/qwen2_vl_cmve.py#L257)和[作者损失函数](https://github.com/Raven-July/CFPO/blob/98ca5a4123be2e85e94855fead46c9a385611136/verl/trainer/core_algos.py#L551)显示以下差异。它们是源码事实，不应擅自解释成作者意图。

| 项目 | 本地实现 | 公开作者代码 |
|---|---|---|
| k3 中的 log-ratio | `factual-corrupt`，按正文 Eq.14–15 | `corrupt-factual`，复用 low_var_kl |
| 数值限幅 | CFPO 分支没有限幅 | log-ratio 限制 ±20，k3 限制 [-10,10] |
| 均值 prior | 每个 head、每个 feature 维度，在 image token 轴取均值 | 在 image token 与 head_dim 两轴求和再除以 token_count×head_dim，得到每个 head 的一个标量后广播 |
| attention 统计 | 每个 head 单独计算人口标准差 | 对当前样本图像后的所有 query 与各 head 的有效权重共同统计，`torch.std` 默认样本标准差，排除 ≤1e-8 的值 |
| 被干预的 query | 仅 prompt 中图像之后至用户消息终止符之前的文字；completion 补 false | 从 image_token_end 起的所有后续 query，包括 teacher-forced completion 位置 |
| 扰动概率梯度 | 当步重新评分，并对事实、干预两侧回传 | [调用链](https://github.com/Raven-July/CFPO/blob/98ca5a4123be2e85e94855fead46c9a385611136/verl/workers/fsdp_workers.py#L700)调用 `@torch.no_grad()` 的 compute_log_prob，把 cmve_log_probs 缓存给 actor，更新时不回传扰动侧 |

例如 logp_factual=-1、logp_corrupt=-3，本地 k3=4.389056，作者 k3=1.135335；并非等价的符号改写。视觉 values `[1,3]`、`[5,7]` 时，本地 prior 为 `[3,5]`，作者 prior 为标量 4。正文 Eq.12 的向量 token 均值与作者实现也存在差别。

正文 Appendix A 定义的 query 是输入问题文字，本地选择与此接近；作者代码使用更宽的范围。因此“匹配正文公式”和“匹配作者基线”须作为两个明确目标。建议作者基线按已固定 revision 对齐；若保留当前正文解释与联合梯度，另列为项目变体，不能把这几个差别藏在共同的 CFPO 名称下。

## 必须修复的问题

1. **P1 AXPO 排序使用错误的 token 范围**。位置：`src/rl_methods_run.py:127`。原文 §3.2 使用来源轨迹的工具调用 token 平均概率，本地用了从开头到 tool_call 开头 tag 的思考前缀概率。人工例子中 A 的前缀概率=.9、工具概率=.1，B 的前缀=.2、工具=.8；代码选择 B，原文选择 A。最小修复是记录完整工具调用 token span，并仅用该 span 排序；修复后增加这个排序翻转反例。否则预算花在与原方法不同的候选上。

2. **P1 AXPO 恢复奖励改写了未选中轨迹的优势**。位置：`src/rl_methods.py:95`、`src/rl_methods_run.py:134–135`。原文 §3.3 要求未选中轨迹保留标准 GRPO loss，恢复后的组标准化用于被选中来源的前缀。全错 `[0,0,0,0]`，只恢复来源 0 时，代码给其余三条轨迹 -0.577350，而普通 GRPO 的原优势都是 0。最小修复是先保留原组优势，仅覆盖 selected 来源的 prefix advantage；独立续写组不变。否则一次成功恢复会额外惩罚原文未要求惩罚的轨迹。

3. **P1 CFPO 有限输入也会指数溢出**。位置：`src/rl_methods.py:72`。人工输入 logp_factual=-1、logp_corrupt=-101 均有限，但 FP32 `exp(100)` 溢出，当前函数报 FloatingPointError；作者已有 log-ratio 与 k3 限幅。最小修复先明确采用哪种目标，再补其数值稳定规则及大 log-ratio 反例，不能只去掉错误检查。否则差异较大的事实／干预概率会中断训练。

4. **P2 ACTIVE-o3 和 AXPO 的两框上限与严格多病灶成功不兼容**。位置：`configs/rl_methods.json` 的 observation.max_regions、`src/rl_observation.py:49`、`scripts/run_rl_methods.py:71–74`。三 GT 病灶属于当前加载接口接受的输入，但最多两个预测框不可能满足 strict 匹配要求；即使两个框完全正确、回答为 yes，correct 仍为 0。最小修复是注册时验证样本范围与 cap、在载入时拒绝不支持的病例；若研究需要多病灶，再明确增大动作预算或按任务模型定位输出评估。不能默默更改历史样本或成功标准。否则部分阳性先天没有成功轨迹。

## 应当修复的核对缺口

5. **P2 现有测试把本地选择当成原算法定义**。`tests/test_rl_methods.py:76–78` 直接要求恢复后未选中优势为负，反而固定了发现 2 的行为；CFPO 的公式和 paired-gradient 测试只验证本地目标自洽，未对照作者实现。修复 1–3 后，应增加原文独立预期和可获取作者函数的数值对照，保留已有效的概率回放、冻结 reference、mask 与 checkpoint 测试。无需扩大为新的医学实验。

此外，ViSurf 的参考 KL、PAPO 的遮挡分布、AXPO 的预算与 loss 汇总、DeFacto 的额外系数，都不是由医学标签替换必然导致的改动。后续必须把“为任务所需的适配”和“自行选择的算法变体”分开标注；并在公平比较中统一共享设置。

## 反例与执行边界

[机器可读核对记录](rl_methods_source_review.json)保存来源 revision、裁定和人工反例结果。反例执行采用本地当前函数的 AST 提取；作者函数也仅提取已阅读的纯 tensor／图像函数，没有运行第三方模块顶层、训练入口或外部判题 API。

确定性 CPU 检查确认：AXPO 排序反转、未选中优势被改写、CFPO 两种概率比的差异、有限输入溢出、vector/scalar prior 差异、三 GT 与两框 cap 不兼容，以及 PAPO 独立 Bernoulli 与强制固定数量遮挡的分布差异。CUDA 未初始化。没有重新跑上一轮 31 项连通测试，因为它们不能回答本次原法对齐问题。

裁定：先修复 1–4，并补齐 5；CFPO 若作为作者基线，还需先对齐来源目标与干预语义。ViSurf、RL-ZVP、ACTIVE-o3 尚缺可取得的公开作者训练代码，AXPO 作者代码仍为内部使用；这些方法的结论限于已取得的原文。7B BF16 的 GPU 数值、显存、吞吐与科学效果本轮均未验证，实验仍未启动。
