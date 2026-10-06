# RSNA v3：奖励归因与证据接口审计协议

> 后续授权更新（2026-10-06）：用户已授权 GPU 0/1、发布，并指定 GPT Astra 替代本轮人工审核。原 CPU 重放配置/协议锁保留为历史记录；当前权限与状态以 `rsna_v3_execution_amendment.json`、`rsna_v3_stage_status.json` 为准。AI 审核不改称真实专家意见；历史未授权描述仅对应初始阶段。

## 范围与身份

依据v2提交76816e51c98f6ec2e41260d3784bc6e11b596f5a；开始时HEAD与该提交一致、工作树干净。本轮仅新增v3文件，v1/v2代码、报告、gate和私有产物只读保留。独立configs/rsna_v3.json和私有v3 protocol/stage_status/budget。

training_enabled、posttraining_enabled、resume_P1、run_P2、run_RL始终全部false。初始R1配置GPU预算0且未授权push；后续独立修订已授权最多两卡、累计1 GPU小时及公开交付，未改写R1锁。R1只读取既有缓存、tokenizer和身份信息，不读取图像、不运行模型。R2只能读取原开发范围；G1/G2仍分别受区域资格与作者材料门禁约束。旧孤立yes/no分支维持停止。

## 冻结数值约定

真实重放前锁定：eps_CED=1e-6，advantage epsilon=1e-8、近零std=1e-8；gate温度.20、tie权重.10。float32奖励/分量重放atol1e-6，优势重放atol1e-5，float64代数残差atol1e-10；相对误差分母floor1e-12。历史Delta_A<1e-4另列，不能与实现一致性容差混用。描述性small-control-std为0<std<1e-4，不是资格筛选；gate饱和<.001或>.999，与std零/很小分开。

R-C=c；R-H=c*g+.1m；R-G=c*g；R-A=c+.1m。相同候选、区域和来源，无参数扫描或补采。无资格病例legacy置零与revised回退分别核验；四版本离线分析对无资格使用correctness回退，不能用于证明证据机制。若有非法格式保留历史-.1并排除二元代数，但实际2048条均合法。

三种优势路径：历史float32；float64保留epsilon；float64去epsilon但仍保留近零分支。数学恒等式只用于合法混合、b非零且不触发近零的组。全同浮点舍入伪优势单列，不强制修正历史实现。患者聚簇bootstrap2000次、seed42，保留每患者四组；空分母重复无效，不填0。

## 分阶段范围

R1校验14个既有产物哈希、冻结源码/config、B1小adapter身份及基座metadata；64患者/256组/2048候选，逐token解码检查文本、标签、正确性、截断、seed和缓存复用；全部32几何病例两标签分解及四版本重放。组/病例细节私有，聚合公开候选。缓存以前的logits/精度未保留者为NA。

R2审核全部93例旧区域。seed42首批全部32名P4几何阳性，次批61名；无结果筛选。图像、区域、每个replacement source、坐标/token统计及930行待审核表私有。wrong改展示名为shifted-overlap probe，历史标识保留。原人工审核表保留pending；后续按用户授权另存Astra AI审核结果，明确无临床资质，肺野coverage unknown。全部旧区域AI审核完成后，按预先冻结的单次语义修订规则生成新区域；计分前冻结最终manifest，保留全部排除。旧区域文件不覆盖。

R3以固定作者commit808a092fe87dd8be4b7e88c2d44aff76cec81b47和可读论文全文审计，分开论文、代码、公开脚本调用和实际运行receipt。真实复合B、prefix/CED贡献未测，缺recipe时blocked_source_recipe；synthetic合同不能冒充模型复现。

原G1要求真实人工审核；后续用户明确改为Astra AI筛查，因此后续只能称AI审核后的适配G1。GPU授权已取得，仍要求R1通过、新区域实际审核通过、manifest冻结及B1身份一致。G2独立要求作者数据/模型/recipe，当前缺失presence任务等材料。任何结果不自动授权训练。

## 检查和保留记录

R1合成边界13项通过；R2合成1项通过；R3四项通过（含固定作者纯方法检查，无跳过），初始共18项；授权修订新增AI审核合并2项、语义区域1项、G1冻结重计分门禁4项，累计26项通过（含新增进程组预算/失败联动检查）。首次R1合成检查暴露constant-float32舍入现象，保留失败日志后修正不成立的理想化测试假设并新增明确检查；历史实现和容差未改。随后冻结R1源码/配置，再真实重放一次，完成后未重算模型。

实际状态、误差、源哈希和测试摘要见rsna_v3_provenance.json。启动命令见rsna_v3_commands.md。人工审核待定不等于否决，来源审计完成不等于author_exact可复现。禁止将v3结果追认为v2 gate通过。
