# RSNA v3：作者接口来源核对与共享候选归因设计

> 后续授权更新（2026-10-06）：用户已授权 GPU 0/1、发布，并指定 GPT Astra 替代本轮人工审核。原 CPU 重放配置/协议锁保留为历史记录；当前权限与状态以 `rsna_v3_execution_amendment.json`、`rsna_v3_stage_status.json` 为准。AI 审核不改称真实专家意见；历史未授权描述仅对应初始阶段。

本轮完成的是**源码/文献审计及 CPU 合同检查**，没有新增冻结模型测量、没有训练。G2 为 `not_authorized`，同时其材料前置条件为 `blocked_source_recipe`。不能将以下设计或 synthetic 检查视为原机制复现成功。

核对来源为本地论文全文 `private/paper.txt`（历史记录标识 arXiv:2608.08021v2）及作者私有只读副本。已实际阅读 §3、§4.1、A.1、A.2、A.8、A.9、A.11；本轮未定位到原上传 PDF，故未做 PDF 页图核验，引用采用章节与全文文本行号，不把历史记录的 PDF 版本身份当作新确认。作者仓库 origin 为 `https://github.com/evidencerl/code.git`，本地 HEAD 是 `808a092fe87dd8be4b7e88c2d44aff76cec81b47`，读取时 `git status --short` 为空。已核对关键文件的 Git blob 身份，记录于同名 JSON；本轮未访问 main、未更新副本、未请求外部服务或下载模型/数据。

## 最重要的来源差异

1. 论文 A.1 将 `R_ans` 描述为 `0.30*answer_score + 0.70*response_score`，各分数由 log probability 与温度产生；但固定公开代码同时包含多条奖励路径。不能把它们拼接成一个已经确认的论文 recipe。
2. `ActionLogProbATEReward.compute` 的 `score_base = .70*tanh(m/.20) + .30*tanh(delta_ans_margin/1.0)` 已包含干预信息；`delta_ans_margin` 是 yes/no logit-margin 在原图和 evidence 干预间的变化，并不是严格正确性。此分支中的 “response” 也随 `reward_logprob_scope` 改变，并非必然整条 response logprob。
3. `train_main_experiment.py` 构建 `MiniGRPO(... reward_mix_mode='main_experiment', allow_ground_truth_for_main_rewards=True)`。其实际可追踪代码调用使用 `reward_main_*` 字段：主 correctness 为正确 `+1`、错误 `−0.25`、无 GT `0`；门控主路径以这个离散奖励为基础，**不消费 `score_base` 作为基础 B**。`supervised_shaped_reward` 是另一个带 bias/scale/clipping 的路径。
4. 作者发布的启动脚本是 Visual Genome `vg_brutal` 数据；论文 A.1 是 COCO val2017 派生 train/validation/probe。没有该论文诊断对应的冻结 checkpoint、运行收据及 probe manifest，不能仅凭脚本名或 README 的成品模型证明 §4.1/A.8/A.9 的实际配置。
5. 作者主路径还包含默认样式惩罚、hard-template veto 和 KL 后的优势标准化。公开脚本写了 min/max clip 参数，但这些不能证明 `reward_main_*` 在最终使用前一定裁剪到 `[-1.25,1]`。CPU 源码方法检查确认，主路径的正确奖励在 `m=1` 时大于 1，而独立 soft-shaping 路径裁剪到 1。

6. 作者 `MiniGRPO.train_step` 生成时使用构造后的 `short_evidence_v1` prompt，但 `_compute_reward_detail` 的 action-logprob 分支传入 `question=sample["question"]`，未使用其收到的 `prompt` 参数。因此该公开调用的生成问题模板与 CED scorer 问题文本不自动相同；作者实际运行是否另有预处理仍缺 receipt，G2须分别冻结并保存这两项。

上述差异修正的是本轮对来源的理解，不改写 v2 文件，不推定论文结果错误，也不证明哪个公开路径就是论文实际运行路径。

## 逐项接口对照

“实际调用配置”一栏区分**已发布脚本的可追踪调用**与**论文实际运行收据**；前者不是后者。作者文件路径均相对于上述固定 commit，RSNA 文件相对于本项目。

| 项目 | 论文描述 | 作者实现 | 实际调用配置及证据范围 | RSNA v2 适配 |
|---|---|---|---|---|
| 冻结模型/checkpoint | §4 主匹配基座 Qwen2.5-VL-7B；A.11 的 Answer/CoT 比较为 Qwen3.5-9B；§4.1 对应精确权重 revision 未给出可复算标识（全文 318–325、374–379、1162–1169） | README 发布 `hhj-ai/Evidence-RL-9B` 成品模型；不是自动等同于 pre-RL 信号探针 checkpoint | `scripts/train_qwen35_9b_single.sh` 默认 `Qwen/Qwen3.5-9B`，无固定模型 revision；§4.1/A.8/A.9 的冻结 checkpoint、receipt **unresolved** | `configs/rsna_v2.json` 基座 Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`；P4 仅历史 B1 step_0256，非作者权重 |
| 数据与任务来源 | A.1：COCO val2017 派生 train 17,502 / validation 2,352 / probe 1,904；在线筛选后 15,314（834–849） | README 和 data 目录提供 Visual Genome brutal 标注/下载脚本 | 发布脚本 `dataset_name=vg_brutal`；本地仅标注、无配套 image bundle、无原 COCO probe split/recipe；实际论文诊断 manifest **unresolved** | 原 RSNA 开发范围；不具有作者任务复现身份，不扩展患者或使用测试 |
| question/response 模板 | A.1 Table A.1 `short_evidence_v1`；A.11 最终答案与 CoT 范围有别 | `answer_format_utils.build_generation_prompt`：任务规则 + Evidence 1–8 words + Final answer 两行；`render_question_text` 可把 bbox 文本嵌入问题；`ced_core.prepare_inputs` 使用 processor chat template | 发布脚本显式short_evidence_v1；MiniGRPO.train_step生成用扩展prompt，但_compute_reward_detail传sample[question]给scorer；两种文本须分别记录，论文信号实验实际模板 **unresolved** | `src/data.py` DATASET_NAME=rsna 的 A 问句；`src/model.Model.prompt` chat template；只要求 no/yes，无候选特有 Evidence/CoT 前缀 |
| final-answer 提取 | A.11 仅 final answer 计敏感性；无完整 tokenizer/span 算法（1162–1195） | `extract_final_answer` 按 final tag、answer tag、两行、tag-next-line、末短行/单短行等规则提取；`extract_final_answer_with_prefix` 返回 char span 和此前 prefix；`MiniGRPO._select_policy_scope_and_span` 增量 decode 映射与字符区间重叠的 tokens | `answer_text_source=final_answer`、policy/reward scope=final_answer、strict scope=True 为主入口 defaults；没有真实候选 span/截断收据，不能证明所有历史候选对齐 | `src/model.Model.score` 直接 tokenize 固定标签；P4 raw 回答经 `src/objectives.correctness` 严格匹配；不是作者 span 提取路径 |
| 保留响应前缀 | Answer-CED“仅答案计分”不等于“删除此前 prefix” | `ActionLogProbATEReward.compute` 的 `reward_condition_on_prefix=True` 把完整 response teacher-force，仅汇总 final span；False 则只 tokenize answer | 主入口 default=True 且传至 MiniGRPO；作者实际 §4.1 日志 **unresolved**；支持 isolated 切换不是论文实际使用的证据 | 只计 isolated yes/no：score(a given I,q)，没有 h_i |
| 干预间文本固定 | §3 固定候选 y，比较局部干预支持度（237–269） | `compute` 仅一次构造 `response_ids`/`extended`，原图、evidence、三个 controls 均复用；没有干预后重新生成前缀 | 已在该函数源码中确认；实际调用是否使用此路径缺 receipt | `src/v2_run.py` P4 缓存每病例两标签；`Model.score` 固定答案 tokens，所有干预共享该答案 |
| 平均/序列 logprob | Eq.2 用序列 log likelihood 简写；A.11 讨论 span 范围/长度 | `_mean_logprob_from_logits` 对所选 token 的 logprob **平均**；返回 vector 与 count；不是序列总和 | 该 compute 原图/所有干预调用同一 mean helper；无法用 Eq.2 简写替代实际 reduction | `Model.score` 保存 mean/sum/length，`classes`/CED 用 mean；v2 冻结定义不变 |
| EOS/结束标记 | A.1 未独立给出 EOS 计分规则；A.11 长度诊断不能推出 EOS 合同 | `compute` encode(... add_special_tokens=False) 不额外追加 EOS；候选生成有效长度截在 EOS/PAD 前；解码 skip_special_tokens；final span排除后缀标记 | 主作用路径可追踪；若 raw 内显式含特殊标记需实际 token 记录确认，不能泛称一切 EOS 都已排除 | classification_score_eos=false；SFT historical include_eos=true 是监督协议，不是本轮训练；P4 reward排除EOS |
| 长度、格式、截断 | A.1 evidence max_new_tokens=32，full max_response_tokens=64；format penalty −.75；A.11 测长另有诊断（797–831、1236–1259） | compute response IDs截至max_response_tokens；MiniGRPO span映射截至max_new_tokens；strict无span返回invalid。边界token相交计入span，故需真实tokenizer复核 | 发布脚本32/64；候选生成decode/重新encode可能影响长度，需逐候选记录；A.11 held-out自然长回答不能当作同一32-token采样收据 | P4最大16新tokens，temperature1/top_p.95/top_k0；合法回答2048/2048是v2已有结果，非本轮新增 |
| R_ans 复合分数 | A.1 `alpha_ans=.30, alpha_resp=.70, tau_ans=1, tau_resp=.20`；复合基础值，不是严格c | compute 的 `score_resp=tanh(m/.20)`；`score_ans=tanh(delta_ans_margin/1)`；score_base为加权和，yes/no margin 默认启用，非yes/no parse令delta_ans=0 | 主入口构造reward_fn采用soft_shaping，但主trainer另消费reward_main字段；论文复合B与实际运行字段映射 **unresolved** | R_H直接c*g(m)+.1m；不得称为作者复合R_ans |
| correctness-only | §4.1/A.8/A.9名称及分组统计不能单独定义0/1编码 | `_correctness_reward` task-aware score>=1给+1，否则−.25；缺GT为0。`reward_main_correctness_only`即该值 | MiniGRPO main路径明确消费对应字段；此外样式和KL可再改变最终reward；actual receipt缺失 | c正确1、错误0、非法−.1（`src/objectives.correctness`），不等于作者+1/−.25 |
| 非法格式与异常 | A.1 format penalty −.75 | `_make_result`失败结果、strict scope、`invalid_final_answer_penalty`、style parse-fail和hard-template veto属于不同分支 | `MiniGRPO.step`先改r['reward']，之后main mix从reward_main字段取training_reward_total，因此不能断言−.75直接传递到最终main reward；还需完整receipt与style状态 | P4全部合法；非法−.1规则存在但实测无非法分母；不能由此验证作者异常路径 |
| gate/tie/clipping | Eq.4 g=.5(1+tanh(m/.20)), tie=.10m；A.1 clip[-1.25,1] | `_compose_main_rewards`门控离散correctness，无最终clip；`_soft_shaping_reward`另有bias/scale与clip；legacy/raw还有不同规则 | main入口的min/max只送soft-shaping构造；CPU pure-method检查main在正确m=1时>1、soft分支clip=1。style/veto/KL后才形成最终标准化奖励；论文实际clip路径 **unresolved** | `src/objectives.ced_reward` detach，c*g+.1m无额外clip；eps_CED=1e-6、controls ddof=0 |
| evidence/control proposals | §3 COCO弱对象框；A.2 priority proposal/target/argument/metadata/center，K=3 area match、IoU≤.05，允许token-subset回退 | `proposal_utils.resolve_primary_proposal_bbox/build_intervention_proposals`实现优先级、random bbox及token-subset fallback | 发布脚本K3、mean；各论文run具体proposal/fallback计数 **unresolved** | `src/regions.py` 固定标注框联合区域，三个几何匹配controls；禁止任意token fallback；93例仅几何合格，human review pending |
| replacement来源/层位 | A.2 post spatial-merge，邻近非目标token均值（879–901） | `ced_core.TokenReplacer`在首LLM层输入hook替换；source由`visual_token_map.surrounding_indices`默认ring2；空ring可回退全局非目标，最后可回退all visual；TokenReplacer空sur甚至用整序列hidden均值 | 作者源码支持的fallback，不能说每个论文样本实际用过；需要来源索引和hook运行证据 | `Model.supplied_features`在visual恢复空间顺序后替换；`source_ring`ring3/valid/forbidden约束，源与目标不交且至少4源tokens；语义合格未获人工确认 |
| 跨候选区域固定性 | 论文按sample构建references，未给seed字段规则 | 每次compute可重建proposal，但_seed_from_metadata按sample_uid/pair_id/sample_id/image_id优先，MD5前8位seed局部random.Random；同metadata/grid/框条件确定 | MiniGRPO传metadata=sample，同case多个候选通常共享同sample；不得把“每次调用重建”误报为“每候选独立随机”。若metadata身份变动则不能保证，需freeze并对照indices | v2私有regions manifest按病例固定；同标签缓存奖励，不按候选重抽 |
| presence/binary routing | §3.2、A.9 presence进入audit；二元任务组方差限制（1127–1138） | answer_format_utils evidence families=counting/attribute/spatial；existence probe-only；_compose_main_rewards不允许的任务退correctness-only | 单卡脚本task_family_filter=counting,attribute,spatial，probe_filter=existence；G2 presence只能作为CED诊断，不能强推训练 | P4对RSNA binary进行离线诊断；RL从未运行，当前isolated yes/no CED-GRPO维持停止 |
| advantage/近零分支 | §3.2组内(R−均值)/std，正文省略数值分支 | MiniGRPO先减KL，再np.std(ddof=0)；std<1e-4时仅中心化，否则除std；all-identical telemetry阈值1e-8；style惩罚先于此 | 主脚本kl_coeff=.01；style_debias默认True、长度/abstention/template/parse等系数，hard veto默认True；论文该诊断是否启用每项 **unresolved** | v2.reward_group为float32、std<1e-8置零，否则除std+1e-8；不可把作者near-zero路径与本项目公式混同 |
| 候选数量/采样参数 | A.1 group32、rerank6、T1/top_p.95；A.8不同分析分母而非一套通用数量 | train入口及候选生成分别有group/rerank/max sampling rounds；可选去重/补采路径 | 发布单卡脚本group32/T1/top_p.95/max_new32/max_response64；§4.1原始group收据 **unresolved**；v3不可直接沿用作者补采逻辑 | P4每病例4组、每组8、64病例；G2仅预注册16 counting+16 presence，每题一次16候选，最多512、不补采 |

## B、prefix 和 CED 的归因不能混为一项

定义严格正确性 `C_i`、事先核验且**只计算一次**的基础 `B_i`。I为 `score(a_i | I,q)`，P为 `score(a_i | I,q,h_i)`。P只对最终答案span取均值，但 `h_i` 必须在原图、evidence、每个control之间逐token不变。完全相同图像、问题、模型、前缀、答案及区域/来源应重现相同分数；干预后重生前缀会改变比较对象。

在同一图像—问题、规范化答案及正确性组内，I若使用完全相同答案token/条件，其margin应相同；若不相同，先查tokenization、源索引和文本条件。P可能因前缀不同产生不同条件分数。这种非零方差既可能来自原图信息携带，也可能来自语言断言或长度/格式差异，**不是可靠grounding已成立的证明**。

复合B若本身已含CED式干预（固定代码的score_base如此），必须记录其输入、scope和来源；把它固定于I/P间只能隔离“新增门控/前缀变化”的边际贡献，不能称B为不含证据的正确性基线。作者主路径离散基值、score_base、supervised-shaped三个值要分别存，禁止重命名后混合。论文B到运行字段映射缺失，现阶段真实B=NA；新增CPU接口要求调用方显式提供B，**不自动拿c或作者score_base填补**。

## 已实现的最小 CPU 接口检查

`src/v3_interfaces.py` 只实现冻结token/最终span合同、原图及干预复用同一条件对象、非有限/截断span拒绝和七个纯代数组合：`C`、`B`、`C_I`、`C_P`、`B_I`、`B_P`、`B_fixed_m0=.5B`。B不因I/P切换而重算，没有模型或训练入口，没有“自动author_exact”默认替身。

`tests/test_v3_interfaces.py` **4项通过、0跳过**。synthetic测试检查重复固定条件、span越界/重复/缺失、非有限分数和C/B分离；另从固定作者源码AST提取四个纯方法执行，避免导入模型或训练模块，确认作者错误奖励−.25、presence路由、主奖励与soft-shaping裁剪差异。读取轻量format模块核对真实作者final-prefix提取逻辑。这不验证真实tokenizer跨字符span的对齐，也不验证真实模型后端、视觉hook或医学证据质量。

复现检查命令：`python3 -m unittest discover -s tests -p test_v3_interfaces.py -v`。只运行CPU；具体结果和文件身份见聚合JSON。

## G2待执行设计与阻塞条件

G2现已获得本轮GPU授权；材料状态仍为 `blocked_source_recipe`。通过配置代理复查官方main仍为历史固定提交；两个官方标注文件各21,758条，作者原始任务路由均给出2,720个counting、0个existence，不能选出要求的16个presence任务。官方模型卡未补充缺失probe manifest。详见`rsna_v3_G2_readiness.json`。缺少：论文信号诊断对应、合法可用的train/probe任务与图像manifest及分割/图像身份去重证据；冻结模型checkpoint/revision及receipt；论文复合B与公开代码路径的可复算映射；完整模板/tokenizer/generation/reward/style/veto/KL有效配置；真实候选span和截断验证。作者本地标注存在不能代替这些条件。禁止下载大模型/大数据或临时制造题目后宣称复现。

其余条件满足后，在已有授权和预算内先冻结16 counting+16 presence，记录32任务对应的独立图像数；一次每题16候选，总计最多512，I/P及所有奖励共享候选池，不补采非法或截断候选。使用核实后的生成长度；缺精确recipe则保持blocked，不能把32/64 defaults冒充已核实运行参数。若只用本项目backbone，明示adapted-control，不称严格复现。

先冻结image/question/model/模板/tokenizer/候选IDs、char/token spans、evidence/3controls及replacement来源，按图像—问题固定，不让候选决定proposal seed。再一次计算并封存B，计算I/P margin及上述六组合+固定m=0；分别记录格式非法、截断、span失败与全部分母，不删除失败来凑结果。`author_exact`必须另行消费核验后的实际字段、路由、clip、非法/样式惩罚及归一化，不等于任一诊断组合；当前为NA/未实现完整模型运行路径，不输出虚构author_exact数值。

按图像聚簇，报告同答案组内B差异、I差异、P差异、固定B后的偏好变化、仅B已有差异、correct/wrong反转，以及与前缀长度/格式/截断/置信度的关系。m=0对B正比缩放在无稳定项的非退化标准化下优势不变；实际近零分支/epsilon另核对，不能强行套用恒等式。预先冻结预算内的不相关proposal负对照；若无独立证据质量标注，最多报告候选区分/区域敏感性。不得将同答案候选对当独立样本，不因非零prefix方差宣布成功。

当前可保留的是来源补全后的原任务机制探针及另立可验证医学证据接口；本次R3不能提供“作者复合基础奖励、前缀与CED各自贡献多少”的实测数值。孤立yes/no CED-GRPO停止状态不变，任何后续方向都不自动转成训练权限。
