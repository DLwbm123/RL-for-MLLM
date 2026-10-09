# 通用 RL 方法实现

2026-10-09。按用户指定范围实现通用 RL 方法；医学方法只用于文献对比。用户明确允许实验暂不开展，因此本次没有正式训练、患者图像读取、GPU 使用或历史阶段续跑。实现是本项目的任务适配，不是论文原始数据与完整训练系统的复现，也不是科学效果验证。

## 实现范围与优先级

| 优先级 | 方法与来源 | 已接通的机制 | 本项目适配与限制 |
|---|---|---|---|
| 1 | [ViSurf](https://arxiv.org/abs/2510.10606) | G 个自行采样回答与一个真实框回答共同计算优势；自行采样出现正确回答时，把标签奖励平滑到原组均值 | 外部回答来自已授权真实框，包含 EOS；没有额外无权重 SFT 项。奖励采用合法输出且完整匹配框的二元结果 |
| 2 | [RL-ZVP](https://arxiv.org/abs/2509.21880) | 非零方差组沿用 GRPO；全对组用正 token 熵优势；全错组用 `-alpha * (max_entropy - token_entropy)` | 熵是完整词表的分布熵，并在旧策略下冻结；单 token 全错组仍可能为零，不能凭空提供正确框 |
| 3 | [PAPO](https://arxiv.org/abs/2507.06448) | 随机遮挡图像的合并视觉 patch；相同回答的事实／遮挡概率差与双侧 sampled-surprisal 稳定项 | 依据[作者代码](https://github.com/mikewangwzhl/PAPO)的 `low_var_kl`：`d=logp_corrupt-logp_factual`，限制 d 为 ±20，k3 为 [-10,10]。采用两侧重新评分和联合梯度，对应作者可选重算路径；并非默认缓存遮挡概率路径 |
| 4 | [CFPO](https://arxiv.org/abs/2606.23206) | 按每个 head 的跨模态 attention 均值与标准差识别高值边；只把这些 query→image 边的 value 改为视觉 value 均值；用事实／干预概率差训练 | 使用论文 Eq.15 的 `d=logp_factual-logp_corrupt`。稳定项采用 PAPO 作者代码的负 sampled logprob 约定，作为显式项目选择；不能等同于完整词表熵。限定 Transformers 4.51.3 的 Qwen2.5-VL 文本 attention、单图、无 cache 的完整序列评分；生成仍走原生路径 |
| 5 | [ACTIVE-o3](https://arxiv.org/abs/2505.21457) | 可训练策略生成有限个裁剪框；冻结初始适配器识别每个新裁剪；合并回答、区域成功、面积和覆盖奖励 | 单步并行裁剪任务适配；不运行任意 Python、网页或外部工具，不另加 Grounding DINO。观察者不接收真实框。阳性正确必须同时有正确回答和严格匹配证据，单独说 yes 不算成功 |
| 6 | [AXPO](https://arxiv.org/abs/2605.28774) | 仅在非空工具子组全部错误时触发；按前缀平均 token 概率从低到高选择；固定思考前缀后重采样动作；对恢复前缀和续写分别算组优势 | 有限裁剪接口的适配。原来源只训练前缀；分支只训练续写，重复前缀不进入 policy/KL。默认最多一个来源、四条额外续写；各分支分别标准化，最终对原组和分支的有效 completion loss 等权平均。这是明确的项目汇总规则，不声称复现作者多轮工具系统 |
| 7 | [DeFacto](https://arxiv.org/abs/2509.20912) | 原图、完整证据遮挡、无关区域遮挡三种视图；答案、格式、证据一致性奖励；每个视图独立组标准化 | 使用注册真实框与 control 框，没有复刻 DINO-X 与作者大规模数据流水线。只有显式登记完整证据支持的阳性才生成 Unknown 标签；无关遮挡保留 yes，阴性仅使用原图 no。两组遮挡框均要求内部不重叠、互相不重叠、归一化总面积相等；像素取整可能造成小面积差异 |

## 接口与更新路径

- `src/rl_methods.py`：优势、masked PPO／固定参考 k3 KL、配对感知目标、CFPO attention 干预。
- `src/rl_observation.py`：严格动作解析、裁剪、随机 patch 遮挡、三视图构造和任务奖励。
- `src/rl_methods_run.py`：采样、旧策略和参考策略评分、额外续写、组装优势和 backward。
- `scripts/run_rl_methods.py`：新注册协议的 preflight／train 入口；只读取训练白名单，固定 schedule，拒绝覆盖输出目录，执行 deadline／预算 reserve，保存独立完成或失败回执。
- `configs/rl_methods.json`：工程默认值，标记 `implementation_only`。没有新授权协议时不能借用旧预算启动。

调用 `UpdateGroup(model, method, config, reference_context, deadline_check, rng)` 后，通过 `groups(image, normalized_gt, question, controls, complete_evidence)` 收集轨迹，再调用 `backward(groups)` 累积梯度，由调用者执行 optimizer step。传入的参考 context 必须保持参考参数冻结；注册入口复用项目已有的 `add_reference`／`reference` 实现，读取同一个初始适配器。所有框的接口单位为 0–1000。

每条自行采样轨迹都要通过原生 generation／完整回放概率检查；配置要求 temperature=1、top_p=1、top_k=0、repetition_penalty=1，拒绝强制 token、抑制词表和其他采样变换。每次 backward 前检查旧策略／当前策略一致性。最大误差与平均误差容差保持既有 P6 数值，没有放宽来掩盖真实模型失败。ViSurf 的真实框回答是外部标签轨迹，仅做旧／当前与参考评分，不伪称由生成器采样。

PAPO／CFPO 用两次 partial backward 得到两侧联合梯度。CFPO 的 checkpoint 重计算必须处于对应事实／干预 attention context 内，不能在切回原路径后回传。

注册入口通过环境变量 `RUN_SPEC` 读取私有 JSON。必填内容包括：`registered=true`、`gpu_authorized=true`、新 `protocol_id`、`execution_mode`（preflight 或 train）、`method`、GPU 编号、绝对截止时间、预算和 reserve 秒数、seed、学习率、模型与初始适配器路径、数据根目录、manifest、训练白名单、固定 schedule、新输出目录、任务 question，以及阳性证据成功／阴性误报标准。manifest 延用现有 `image_id,case_id,split,image_path,mask_path,boxes`；DeFacto 另需归一化 `control_boxes` 与 `complete_evidence`。AdamW 使用标准 betas／eps，weight_decay=0，gradient_clip=1。零优势产生的有限零梯度允许存在，不把它伪称为计算图断开。

真实运行前还需按项目规则检查显存、挂载与预算，并使用中性 `python -u -` stdin 入口调用 `main()`；不能直接把项目／方法路径放进可见命令行。本次没有创建授权 JSON，也没有调用真实模型训练入口。preflight 只执行第一组 schedule 的评分和 backward，不做 optimizer step；train 完成后保存固定末步适配器，失败时保留已更新的停止适配器。评估与检查点选择不在该入口中自动执行。

## 工程检查

31 项 pytest 检查通过：20 项新方法检查，11 项既有 core 回归。另在独立 CPU 进程通过现有 P6 自检。详见 [回执](rl_methods_validation.json)。使用已有 Torch 2.7.1+cu128、Transformers 4.51.3、PEFT 0.15.2，未安装新依赖。

检查覆盖公式与梯度方向、零方差退化、重复字段／非法裁剪拒绝、完整证据支持门槛、原生随机初始化小型 Qwen2.5-VL 的生成与 replay、独立冻结参考适配器、七个方法的梯度及一次人工夹具 optimizer 更新。另比较了 CFPO checkpoint 下两次 partial backward 与关闭 checkpoint 的直接联合梯度，结果一致。七方法集成检查使用人工指定动作和原生概率评分；真正的原生随机采样由单独检查覆盖，不能把人工轨迹称为真实 on-policy 数据。

PEFT 在随机初始化的小模型保存时提示找不到预训练配置，属于夹具提醒；断言与退出码均通过。CUDA 未初始化，患者数据未读取。

未检查 7B BF16 实际 GPU 概率一致性、峰值显存、吞吐和医学效果；小模型 CPU 通过不能证明历史 P6 的真实模型数值问题已解决。实验待另行授权与注册，P4–P9 的停止决定不因此解除。
