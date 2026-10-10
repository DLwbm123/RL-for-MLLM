# 七方法源码对齐与五项问题修复

2026-10-10。已修复[审阅报告](rl_methods_source_review.md)的五项问题，并按用户要求对齐：取得训练代码的 PAPO、CFPO、DeFacto 以固定作者 revision 为准；ViSurf、RL-ZVP、ACTIVE-o3、AXPO 依据已取得原文。没有启动实验，历史停止决定、数据、checkpoint、样本和预算保持原样。

## 五项问题的处理

| 原发现 | 修复 | 针对性验证 |
|---|---|---|
| 1：AXPO 候选排序范围错误 | 保存完整序列化工具调用 span，包括开／闭标签；只用该 span 的平均 token 概率排序。分叉点仍位于开标签之后，与原文一致 | 思考概率 .9／工具概率 .1 与思考 .2／工具 .8 的排序翻转反例；工具 span 不包含思考和 EOS；思考内出现字面 tool tag 不误作分叉点 |
| 2：恢复后改写未选中优势 | 原组优势保留；每个 selected 来源独立在原始奖励组中替换自己的恢复奖励，仅覆盖该来源 prefix advantage。分支仍独立标准化 | 全错组其余优势保持 0；存在成功 no-tool 时保留原优势；多个 selected 来源也各自按 Eq.4 计算；来源续写和重复分支前缀 mask 回归 |
| 3：CFPO 有限输入溢出 | 采用作者 `corrupt-factual` low_var_kl，log-ratio 限幅 ±20、k3 限幅 [-10,10]；保留有限性校验 | (-1,-101) 和反向输入均有限、梯度有限，k3=10；原文／作者方向反例现在得到作者 k3=1.135335；扰动侧无梯度 |
| 4：两框动作不支持三 GT 严格成功 | 两框 cap 保持不变；白名单 manifest 在图像／模型载入前检查，直接 groups／batch_groups 在任何采样前检查。超限拒绝整批，不删除病例、不降低成功要求 | 三 GT 拒绝且 tick／采样次数为 0；整批后部出现不支持病例也不能先采前部；零、一、两 GT 和原始像素坐标的合法输入通过支持检查 |
| 5：测试把本地错误当算法定义 | 删除“未选中优势应为负”和 CFPO 联合双侧梯度预期；使用原文独立预期、作者函数数值和缓存教师梯度预期 | 54 项 pytest 通过，其中方法检查 43 项、既有 core 回归 11 项；另独立提取并对照已审阅作者纯函数 |

## 作者实现对齐

PAPO 固定 [fb63bf4](https://github.com/mikewangwzhl/PAPO/tree/fb63bf4533b78da5a2bb17bc812375dcb620887d)：原图 14×14 像素 patch 按独立 Bernoulli(.6) 遮挡，不再强制固定遮挡数量或必须留下 patch；可以全留或全遮。恢复默认 `RECOMPUTE_AUG_LOG_PROBS=False` 的缓存教师路径。按 7B 脚本设置 gamma=.02、双 sampled-surprisal 系数 .05；缓存侧项仅贡献损失数值，没有参数梯度。组优势使用样本标准差加 1e-6；默认 token 归约在整个更新批的有效 token 上平均。

CFPO 固定 [98ca5a4](https://github.com/Raven-July/CFPO/tree/98ca5a4123be2e85e94855fead46c9a385611136)：采用 CFPO-G 脚本的 gamma=.02 和两个 entropy 开关关闭。干预概率在原始轨迹采样阶段缓存为 no-grad teacher，更新只回传事实侧。图像后所有 query 都参与，包括 vision_end 位置、assistant 前缀和 teacher-forced completion；每个样本跨 head 汇总 >1e-8 的有效 attention，以样本标准差计算 mean+2std。prior 使用每 head 的图像 token×feature 标量均值广播，避免把正文的向量均值误称为作者实现。零有效边不干预；单有效边保持作者 NaN 阈值对应的“不选边”结果，但不产生 std 警告。

两者也对齐作者 PPO 的 upper clip=.3、negative-advantage dual clip=3 和有界 reference k3，未修改历史阶段的共享 objective 实现。

DeFacto 固定 [471a747](https://github.com/tinnel123666888/defacto/tree/471a747bc416a59e8ddad22de2e9b038c0c0aeba)：恢复 `gamma_unk=.2, rho_unk=1, gamma_guess=.6, gamma_corr=.9`，selection 的正／负 IoU 权重均 1、空框 -.5，组合权重 `answer/format/selection=1/.2/.2`，参考 KL=0。答案、格式、selection 分别计分；例如合法答案 yes 但空框可取得答案项，却没有格式项，也不算有证据成功。每个视图用样本标准差加 1e-4 归一化。

## 无公开训练代码的方法

[ViSurf](https://arxiv.org/abs/2510.10606)保留 Eq.6–7 标签优势与平滑，移除参考 KL，并依据 Eq.9 对序列 logprob 求和，而非按长度平均。[RL-ZVP](https://arxiv.org/abs/2509.21880)三分支与完整词表熵本来正确，保持单次 on-policy 更新及旧策略冻结熵；不擅自推广到多次 PPO 更新。

[AXPO](https://arxiv.org/abs/2605.28774)除发现 1–2，还恢复附录的 r=.25、K=4 整批额外预算和广度优先分配：每个触发问题先获得首个候选，再分配第二个；总预算向下取整到完整 K，不超额。普通未选中轨迹保留标准 GRPO 的 1/G 系数；Eq.5 的 prefix loss 与 K 个 continuation loss 相加，每个 selected prefix 加入一次，然后按问题批平均，不再把所有条目按 G+K 稀释。上裁剪=.4、参考 KL=.001。通用工程 G=4 时，B<4 的整批预算不足一个 K，会保留普通 GRPO，不跨 step 积攒或借预算。

[ACTIVE-o3](https://arxiv.org/abs/2505.21457)保留可训练观察策略与冻结任务模块、有限并行裁剪和启发式加任务奖励；增加发现 4 的输入支持检查。作者公共仓库没有训练实现，不能声称奖励逐项代码复刻。

## 保留的医学任务适配与工程边界

1. 框和病例答案替代原数学／开放 VQA 标签，所有框统一为 0–1000；阳性成功必须同时答对并完整严格匹配证据，阴性必须没有误报框。真实框来自登记训练标注，避免引入额外未核验 detector。
2. DeFacto 只有登记“完整证据支持”的阳性才能遮挡后标 Unknown；缺证据支持就拒绝，不能把局部遮挡伪称为不可回答。阴性只用原图 no；control 与 GT 内部及相互不重叠、面积匹配。selection 仍使用作者空框系数，不再额外给阴性加自定 bonus。
3. 可审计的框／答案 JSON 接口代替自由语义判题和作者输出标签；不调用外部判题 API，不传出医学图像。ViSurf 标签使用相同输出契约，但没有实验验证其为模型偏好格式；DeFacto 不复刻逐步区域探索或推理过程格式奖励。
4. ACTIVE-o3／AXPO 使用固定初始 adapter 在裁剪图上回答 yes/no，并把动作框作为证据框；这是医学观察任务的明确定义，区别于原论文任务模型输出局部检测框再映射到全图。有限单步裁剪替代 Python／网页等多轮工具，不能据此宣称完整 agentic 系统复现。
5. 训练参数仍限定语言 LoRA，视觉与 reference 冻结；共用原生无变换采样、G=4、128 token 上限和单次更新，保留概率回放门槛。这些是现有工程比较条件，与作者组大小、温度、全参数训练和长度配置并不都相同，需在后续新实验协议中明确登记。更改方法机制不解除旧科学停止条件。

注册入口使用 `UpdateGroup.batch_groups(cases)`，先整批采样再 backward，支持 AXPO 的全局预算与 PAPO／CFPO 的 token 归约。直接单病例 groups 接口仍可用，但 AXPO 不再默认获赠四条额外轨迹。整批收集的显存需求尚未用 7B GPU 验证。

## 验证及未验证内容

[当前回执](rl_methods_source_fix_validation.json)记录 54 项 CPU pytest、独立作者函数数值对照及运行版本。人工小模型覆盖七方法的梯度、冻结 reference、原生 generation/replay 和单次 optimizer fixture；这些是工程测试，不是患者训练或效果实验。作者函数仅 AST 提取所审阅的纯 tensor／图像部分，没有运行第三方训练模块顶层、语义 API 或数据流水线。

CFPO 的四种阈值模式在 FP32／BF16 人工张量上选边一致；FP32 输出最大差异 5.96e-8，来自取图像子矩阵后的浮点归约，BF16 本次输出一致。PAPO 相同随机数下的像素遮挡、PAPO／CFPO 的 low_var_kl 与组归一化、DeFacto 三视图 selection 函数均与固定作者代码一致。源码对照中的严格逐值断言因 FP32 5.96e-8 舍入差异改为该 dtype 的一个 epsilon×结果量级，不修改训练概率 gate。

pro5000 的 SSH 连续两次关闭，远端检查没有成功启动；改用已有本机 CPU 环境，Torch 2.6.0、Transformers 4.51.3、PEFT 0.19.1。与登记服务器的 Torch 2.7.1／PEFT 0.15.2 不同，未安装或改动依赖。随机初始化小模型保存时的 PEFT 配置提示属于 fixture 警告。

未读取患者数据，未使用 GPU，未运行正式训练／评估／历史续跑。本次 CPU BF16 张量对照不等于 7B BF16 GPU 验证；真实模型概率门槛、显存、吞吐和医学效果仍未验证。
