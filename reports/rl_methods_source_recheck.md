# 修复后七方法复核与遗漏修正

2026-10-10，基于 `fc36bf1849d90980819a59369adbf46810795458`。复核当前调用链、作者固定 revision、原文正文与附录、梯度和医学接口。没有启动实验。上一轮 [54 项回执](rl_methods_source_fix_validation.json)保留为历史记录；本轮结果见 [72 项回执](rl_methods_source_recheck_validation.json)。

## 本轮发现并修复

| 方法 | 遗漏及依据 | 当前行为与针对性检查 |
|---|---|---|
| RL-ZVP | 初审只看正文省略 KL 的脚注，漏看附录 A／Table 5：实际实现明确去掉 KL、clip-low=.20／high=.28、token 级损失 | 不再添加 reference KL；upper clip=.28；整批有效 token 平均，避免短响应与长响应按相同权重。独立反例检查 reference=-101 时仍无 KL、ratio=1.5 时 loss=-1.28；长度 1／3、优势 .1／.2 时权重 .25／.75，梯度 -.175 |
| ACTIVE-o3 | 附录 B 给出 overlap threshold=.3、四项权重均 1；本地 .1 阈值、先除以 4 再乘 .25 和格式失败 -1 均缺少原文依据。coverage 也误用了严格一对一匹配计数 | 四项启发式直接相加、阈值 .3；每个 GT 有任一 IoU≥.5 提案即计入 coverage。完整医学成功仍需严格一对一匹配与正确答案；无效动作奖励 0。IoU=1/7 的两提案检查得到总奖励 5；一提案覆盖两 GT 可得 coverage=1、但任务正确仍为 0 |
| DeFacto | 归一化框等面积／不相交不能保证取整后的像素遮挡仍等面积／不相交；control 可能擦除目标像素，损坏 rand 标签前提 | 同时检查实际像素矩形在视图内与跨视图不相交，并且像素面积相等；直接使用验证后的矩形遮挡。存在像素交叠或面积不等则拒绝，不移动 control、不改标签、不筛选病例。10×10 合成图复现交叠与 4／9 像素面积反例，合法配对保留 |
| AXPO | 当 tokenizer 把 `<tool_call>` 后的换行并入同一 token 时，完整合法调用被排除，额外采样静默退化为普通 GRPO | 分叉边界允许开标签后只带空白，不包含工具动作；合并换行 token 检查能找到 prefix 与完整工具 span。若 token 同时含第一段动作内容，仍不把部分动作当成纯思考前缀 |

原初审的 RL-ZVP KL 判断现已由附录证据纠正；这不是必须保留的医学适配。AXPO 的换行修复属于 token 边界适配，不扩大工具集或额外预算。CFPO context 的说明也更新为实际缓存教师用途。

## 逐方法复核结论

| 方法 | 原文／作者对齐与实际梯度路径 | 医学与工程边界 |
|---|---|---|
| [ViSurf](https://arxiv.org/abs/2510.10606) | Eq.6–7 标签加入与奖励平滑；单次更新按 Eq.9 的序列梯度求和；无参考 KL。标签和旧概率冻结，标签轨迹仍给当前语言 LoRA 梯度 | 真实框 compact JSON + EOS 为外部监督，不能称为 on-policy 采样。其风格是否为实际模型偏好尚未验证。此实现针对单次更新的 Eq.9，不宣称通用多轮更新与 Eq.8 序列概率裁剪等价 |
| [RL-ZVP](https://arxiv.org/abs/2509.21880) | 三分支、逐响应最大熵、完整词表熵与冻结熵均正确；补齐附录裁剪、无 KL 和 token 归约。全错／全对组梯度方向及 entropy stop-gradient 均有检查 | 医学严格框成功替代数学答案；保留 G=4、128 token 与一次更新。无训练代码时 std 的数值惯例沿用 population std 与 1e-8；不能据公式未指明的惯例声称逐值复刻作者 verl 归一化 |
| [PAPO](https://github.com/mikewangwzhl/PAPO/tree/fb63bf4533b78da5a2bb17bc812375dcb620887d) | 原图 14 像素 Bernoulli(.6) 遮挡、缓存 no-grad teacher、作者 `corrupt-factual` 有界 k3、事实侧梯度、sample std+1e-6、token 平均、双 sampled-surprisal 与默认系数均保留 | 原始视觉编码和 reference 冻结，只更新语言 LoRA。扰动侧 surprisal 为常数项；不能称为完整词表熵或双侧参数更新 |
| [CFPO](https://github.com/Raven-July/CFPO/tree/98ca5a4123be2e85e94855fead46c9a385611136) | 池化有效权重／sample std、标量图像 prior、vision_end 之后全部 query（含 completion）、缓存干预教师与事实侧回传对齐固定代码。补读作者图像 span 构造，确认 vision_start 后至 vision_end 前的索引 | Qwen2.5-VL 的 RoPE／原生事实路径适配作者 Qwen2-VL 干预实现，限定 4.51.3、单图和无 cache 评分。保留作者代码优先于冲突的正文向量 prior／KL 方向 |
| [ACTIVE-o3](https://arxiv.org/abs/2505.21457) | 训练观察策略、冻结任务模块、有限并行裁剪和启发式加任务奖励结构成立；本轮恢复附录可确定的启发式规则 | yes/no 观察者与裁剪作为证据替代原检测／SAM task model；task 项使用联合成功二值值。normalized 半开框面积替代原像素 inclusive `+1`；阴性可不裁剪，空集合的 area／overlap 为真、coverage 按无误报定义。不能称为原检测奖励复现 |
| [AXPO](https://arxiv.org/abs/2605.28774) | 全错工具子组触发、完整工具调用平均概率排序、r=.25／K=4 的整批广度优先预算、逐来源 Eq.4 恢复、未选中优势保留、Eq.5 分段权重均保留。来源只训练 prefix，分支只训练 continuation；policy／KL 共用 mask | 使用可闭合的单步医学裁剪。未闭合调用或不存在不含动作的 token 分叉边界时不作 resampling 候选；这一有限接口边界仍区别于作者多轮 agent。G=4 小批预算不足 K 不借预算 |
| [DeFacto](https://github.com/tinnel123666888/defacto/tree/471a747bc416a59e8ddad22de2e9b038c0c0aeba) | pos／cf／rand、Unknown／猜中原答案惩罚、独立 answer／format／selection、作者系数、sample std+1e-4、无 KL 正确。视图与标签冻结，奖励与优势不参与梯度 | 完整证据登记仍为生成 Unknown 的前提；真实框／control 与严格 JSON 替代 detector／语义 API。新增实际像素 guard；登记完整支持不是医学可回答性已被实验验证的证明 |

取得的 PAPO／CFPO／DeFacto 固定训练源码作为实现依据。ViSurf、RL-ZVP、ACTIVE-o3、AXPO 继续依已保存原文；前次获取的 ACTIVE 仓库无训练实现，其他方法亦未取得可用作者训练代码。本次不是最新仓库发布状况调查。

## 梯度与检查证据

- 本轮 **72 项 pytest 通过**：方法检查 61 项、既有 core 回归 11 项。七方法各在显式 eager 和 SDPA 的随机小型 Qwen2.5-VL／PEFT 路径运行，验证梯度、一次合成 optimizer 更新和冻结 visual／reference；原生生成概率回放单独检查，不把指定动作的集成夹具称为真实采样。
- old/current 门槛检查整批后才做首次 backward；后部样本失败时前部也不留下梯度。长度不同的 RL-ZVP／PAPO／CFPO fixture 检查整批 token 权重及梯度，entropy、teacher 没有梯度。
- 从两个固定作者 `core_algos.py` 仅 AST 提取 `compute_policy_loss`／`average_loss`／`compute_kl`，未执行第三方模块顶层；混合正负／零优势、mask、上下／dual clip、极端有限 ratio／reference 组合的本地 loss 与参数梯度均与作者函数相同，本 fixture 最大误差均为 0。旧轮遮挡、CFPO 选边、组优势和 selection 对照保留其原回执，不伪称本轮重跑。
- 新增行为反例在修复前已复现失败。检查全部在现有本机 CPU 环境完成：Torch 2.6.0、Transformers 4.51.3、PEFT 0.19.1；CUDA 未初始化，没有新增依赖。20 条 PEFT 提示来自随机夹具保存时缺预训练配置，不是失败。

未访问患者图像、未启动正式训练／评估／历史续跑。未验证服务器固定依赖、7B BF16 GPU 的概率门槛、显存／吞吐、真实 tokenizer 的全部边界或医学效果；CPU 和合成 BPE 反例不能证明这些已解决。当前修改不解除历史科学停止条件。
