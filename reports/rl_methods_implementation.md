# 通用 RL 方法实现

2026-10-10 更新。按用户指定范围实现通用 RL 方法；医学方法只用于文献对比。本次依据[源码核对](rl_methods_source_review.md)修复问题：PAPO、CFPO、DeFacto 以固定作者 revision 为准，其余依据原文。没有正式训练、患者图像读取、GPU 使用或历史阶段续跑。实现是本项目的任务适配，不是论文原始任务与完整训练系统的复现，也不是科学效果验证。当前复核和必要适配详见[72 项复核报告](rl_methods_source_recheck.md)，上一轮见[修复说明](rl_methods_source_fixes.md)。

## 实现范围与优先级

| 优先级 | 方法与来源 | 已接通的机制 | 本项目适配与限制 |
|---|---|---|---|
| 1 | [ViSurf](https://arxiv.org/abs/2510.10606) | G 个自行采样回答与一个真实框回答共同计算优势；达到标签奖励后把标签奖励平滑到原组均值；去掉参考 KL；按 Eq.9 的序列 logprob 求和梯度 | 外部回答来自已授权真实框，包含 EOS；没有额外无权重 SFT 项。使用同一严格 JSON 框接口，没有额外推理标签；这不表示已测得模型偏好的输出风格 |
| 2 | [RL-ZVP](https://arxiv.org/abs/2509.21880) | 非零方差组沿用 GRPO；全对组用正 token 熵优势；全错组用 `-alpha * (max_entropy - token_entropy)` | 熵是完整词表的分布熵，并在旧策略下冻结；单 token 全错组仍可能为零，不能凭空提供正确框 |
| 3 | [PAPO](https://arxiv.org/abs/2507.06448) | 原图上 14×14 像素 patch 独立 Bernoulli(.6) 遮挡；相同回答的事实／遮挡概率差与 sampled-surprisal 项 | [固定作者代码](https://github.com/mikewangwzhl/PAPO/tree/fb63bf4533b78da5a2bb17bc812375dcb620887d)：缓存无梯度遮挡概率；`d=corrupt-factual` 限幅 ±20，k3 限幅 [-10,10]。按 7B 脚本 gamma=.02、surprisal 系数均 .05；缓存侧项只贡献损失数值。组优势用样本标准差，loss 按整个更新批的有效 token 平均 |
| 4 | [CFPO](https://arxiv.org/abs/2606.23206) | 各样本所有 post-image query／head 的有效 attention 共同计算均值和样本标准差；高于 mean+2std 的边替换为每 head 标量视觉均值 prior，包括 completion query | [固定作者代码](https://github.com/Raven-July/CFPO/tree/98ca5a4123be2e85e94855fead46c9a385611136)：缓存无梯度干预概率；与 PAPO 相同的有界 `corrupt-factual` k3；按 CFPO-G 脚本 gamma=.02，两个 entropy 开关均关闭。选择作者实现而非冲突的正文解释。限定 Transformers 4.51.3 的 Qwen2.5-VL、单图、无 cache 完整序列；生成仍走原生路径 |
| 5 | [ACTIVE-o3](https://arxiv.org/abs/2505.21457) | 可训练策略生成有限个裁剪框；冻结初始适配器识别每个新裁剪；合并回答、区域成功、面积和覆盖奖励 | 单步并行裁剪任务适配；不运行任意 Python、网页或外部工具，不另加 Grounding DINO。观察者不接收真实框。阳性正确必须同时有正确回答和严格匹配证据，单独说 yes 不算成功 |
| 6 | [AXPO](https://arxiv.org/abs/2605.28774) | 非空工具子组全错触发；按完整工具调用 token 的平均概率排序；整批 r=.25、K=4，广度优先分配；未选中轨迹保留原优势；每个被选来源单独做 Eq.4 奖励替换 | 原来源只训练前缀，分支只训练续写，重复前缀不进入 policy/KL。普通未选中 loss 保留 1/G，Eq.5 的 prefix+continuation 和逐前缀加入，再按问题批平均；不再按 G+K 稀释。按附录采用上裁剪 .4、KL=.001。闭合、可解析的有限裁剪接口代替原多轮工具系统；小批预算不足 K 时不向上取整加预算 |
| 7 | [DeFacto](https://arxiv.org/abs/2509.20912) | pos/cf/rand 视图；答案、格式、selection 三项独立计分；恢复作者默认 .2/1/.6/.9 系数、正负 IoU 权重 1、空框 -.5、总组合 1/.2/.2、参考 KL=0 | 使用注册真实框与 control 框，不复刻 DINO-X 与作者大规模数据流水线。只有显式登记完整证据支持的阳性才生成 Unknown；无关遮挡保留 yes，阴性只使用原图 no。严格 JSON 代替标签与语义 API，不把格式失败伪装成答案判定。每视图独立样本标准差加 1e-4 归一化。两组遮挡框在归一化与实际像素层面都要求不重叠、面积相等；取整破坏条件时拒绝 |

RL-ZVP 已按附录 A／Table 5 采用无参考 KL、上裁剪 .28 和整批 token 平均。ACTIVE-o3 的四项启发式权重均为 1、overlap 阈值 .3、coverage 按每个 GT 有任一匹配计算；医学任务成功仍为严格证据与回答联合成功。AXPO 的纯 prefix token 边界允许开标签后空白，不包含动作。

## 接口与更新路径

- `src/rl_methods.py`：优势、masked PPO／固定参考 k3 KL、配对感知目标、CFPO attention 干预。
- `src/rl_observation.py`：严格动作解析、裁剪、随机 patch 遮挡、三视图构造和任务奖励。
- `src/rl_methods_run.py`：采样、旧策略和参考策略评分、额外续写、组装优势和 backward。
- `scripts/run_rl_methods.py`：新注册协议的 preflight／train 入口；只读取训练白名单，固定 schedule，拒绝覆盖输出目录，执行 deadline／预算 reserve，保存独立完成或失败回执。
- `configs/rl_methods.json`：工程默认值，标记 `implementation_only`。没有新授权协议时不能借用旧预算启动。

调用 `UpdateGroup(model, method, config, reference_context, deadline_check, rng)` 后，通过 `groups(image, normalized_gt, question, controls, complete_evidence)` 收集轨迹，再调用 `backward(groups)` 累积梯度，由调用者执行 optimizer step。传入的参考 context 必须保持参考参数冻结；注册入口复用项目已有的 `add_reference`／`reference` 实现，读取同一个初始适配器。所有框的接口单位为 0–1000。

注册入口现在调用 `batch_groups(cases)`，每个 case 是上述五个参数组成的 tuple。先收集整批原轨迹，再统一分配 AXPO 预算，或按 PAPO／CFPO 的整批有效 token 数归约。ACTIVE-o3／AXPO 会在加载白名单 manifest、读取任何图像或模型之前检查病灶数，在直接接口中也会于采样前检查；GT 数超过固定两框上限就拒绝整批，不筛掉病例、不改标注或成功阈值。

每条自行采样轨迹都要通过原生 generation／完整回放概率检查；配置要求 temperature=1、top_p=1、top_k=0、repetition_penalty=1，拒绝强制 token、抑制词表和其他采样变换。每次 backward 前检查旧策略／当前策略一致性。最大误差与平均误差容差保持既有 P6 数值，没有放宽来掩盖真实模型失败。ViSurf 的真实框回答是外部标签轨迹，仅做旧／当前与参考评分，不伪称由生成器采样。

PAPO／CFPO 在采样阶段一次性缓存无梯度扰动概率，更新只通过原生事实路径回传。CFPO 干预 context 仅用于这些 no-grad 评分，不再把扰动图送入 checkpoint backward；上下文退出后恢复原 attention。

注册入口通过环境变量 `RUN_SPEC` 读取私有 JSON。必填内容包括：`registered=true`、`gpu_authorized=true`、新 `protocol_id`、`execution_mode`（preflight 或 train）、`method`、GPU 编号、绝对截止时间、预算和 reserve 秒数、seed、学习率、模型与初始适配器路径、数据根目录、manifest、训练白名单、固定 schedule、新输出目录、任务 question，以及阳性证据成功／阴性误报标准。manifest 延用现有 `image_id,case_id,split,image_path,mask_path,boxes`；DeFacto 另需归一化 `control_boxes` 与 `complete_evidence`。AdamW 使用标准 betas／eps，weight_decay=0，gradient_clip=1。零优势产生的有限零梯度允许存在，不把它伪称为计算图断开。

真实运行前还需按项目规则检查显存、挂载与预算，并使用中性 `python -u -` stdin 入口调用 `main()`；不能直接把项目／方法路径放进可见命令行。本次没有创建授权 JSON，也没有调用真实模型训练入口。preflight 只执行第一组 schedule 的评分和 backward，不做 optimizer step；train 完成后保存固定末步适配器，失败时保留已更新的停止适配器。评估与检查点选择不在该入口中自动执行。

## 工程检查

2026-10-09 的 31 项检查和 P6 自检属于修复前记录，保留在[历史回执](rl_methods_validation.json)，不能作为本次对齐证据。上一轮定向测试、核心回归和独立作者函数对照见[54 项修复检查回执](rl_methods_source_fix_validation.json)。后续补齐附录及医学边界的当前结果见[72 项复核报告](rl_methods_source_recheck.md)和[复核回执](rl_methods_source_recheck_validation.json)。未安装新依赖。

检查覆盖独立原文预期、作者函数的数值结果、极端概率限幅、AXPO 排序／预算／优势／mask、输入支持、完整证据门槛、原生随机初始化小型 Qwen2.5-VL 的生成与 replay、冻结参考适配器、缓存教师与 checkpoint 事实梯度、七个方法的梯度及一次人工夹具 optimizer 更新。七方法集成检查使用人工指定动作和原生概率评分；真正的原生随机采样由单独检查覆盖，不能把人工轨迹称为真实 on-policy 数据。

PEFT 在随机初始化的小模型保存时提示找不到预训练配置，属于夹具提醒；断言与退出码均通过。CUDA 未初始化，患者数据未读取。

未检查 7B BF16 实际 GPU 概率一致性、峰值显存、吞吐和医学效果；小模型 CPU 通过不能证明历史 P6 的真实模型数值问题已解决。实验待另行授权与注册，P4–P9 的停止决定不因此解除。
