# MedEvidence pilot-3：患者覆盖扩展与答案能力保留

本轮两组监督训练，不运行RL、dep、control、crop训练、蒸馏、KL、首token加权或新模型/头。历史依据21e7914370081f504d8efd4f6795a8f24785732b；均从pilot1原M1最终权重开始，绝不继续L1。旧结果只读保留。

用户本轮明确允许自行审定执行，随后指定“你可以用3个gpu，尽快完成”。最多3个独立单卡作业、累计3 GPU小时，包括加载、预检、前向、训练、评价、失败和退出。各卡启动前至少24000MiB可用显存，不改变其他进程。公开push仍须本轮授权。进程入口为中性python -u -。

## 患者与完全共享排程

原1024训练/256开发患者、一患者一图。原71名单框阳性全部入选，不限旧dep面积资格；78名多框训练阳性不加入。阴性保留旧Fit16，再对剩余训练阴性按case_id稳定排序、numpy.default_rng(17)排列选240。总327患者，选择不用分数、误报、奖励或审核难度。

排程用单独numpy.default_rng(17)，先阳性再阴性排列；每步positive,negative,positive,negative，类内冻结队列无放回循环。256步1024定位暴露，正负各512；56阳性7次、15阳性8次，256阴性各2次。两分支同一患者/图像/目标顺序。旧Fit32全部进入，旧Ref16阳性+4阴性进入，不能继续称本轮不更新参照。

冻结私有患者、图像、原框、标签、GT/token、选择与顺序哈希。395幅训练/局部供体原图核对维度和身份；全1280个GT自匹配、坐标往返误差≤每轴尺寸/2000。327例实际tokenizer A/L含EOS编码核验。DevelopmentData真实ID/路径白名单与audit hook持续封存测试。GT线条、真值类别和审核不进入输入；所有开发患者调用L，无分类门控。

## 模型、初始化和目标

Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5，真实冻结分片bytes/mtime及adapter张量身份核验。语言q/v LoRA r16/alpha32/dropout0，5,046,272可训练参数；视觉及merger冻结，bf16、576–1024视觉tokens、原processor/chat template。

各分支新AdamW、seed17重置，lr2e-5、betas(.9,.999)、eps1e-8、weight_decay.01、clip1、固定学习率。任一更新前实际adapter/optimizer/RNG/lr/排程位置/训练模块必须精确匹配预检初始身份；不继承L1或COV状态启动COV-A。

ell_L/ell_A分别完整目标有效token平均NLL、含EOS。COV=sum4 ell_L/4；COV-A=sum4 ell_L/4+sum4 ell_A/4，不再除2，不额外除累积次数。同4原图，微批量1、累积4后统一clip/step一次。COV同图A仅no_grad诊断，COV-A保留LLM/LoRA A梯度；冻结视觉特征正确复用。L/A反向时参数不变，最终才更新。唯一目标差异A系数0/1，多出A backward成本和剪裁影响如实报告，不称FLOPs完全匹配。

唯一无更新preflight用首4排程病例：有/无A no_grad的L梯度比较、A独立LoRA连接、视觉无梯度、adapter/optimizer不变及保存/恢复/RNG精确回原状态。浮点容差预定2个bf16 epsilon=.015625，用于相对L2和参考最大梯度缩放的最大绝对差；旧实际重复反向差.006211作为精度依据，不因新失败放宽。精确身份检查不用浮点容差。无optimizer更新式smoke或第三组训练。

保存0/64/128/256的权重、optimizer、RNG和位置；固定256最终，不选最佳、不追加。日志含L/A正负loss、EOS、总梯度/剪裁、暴露、逻辑forward/vision/backward调用数、显存及时间。逻辑调用数不包含所有checkpoint内部重算，不等于FLOPs；主日志不额外逐任务反向重算。

## 预算与并行

预约preflight500秒、COV2600、COV-A3700、各最终开发800、三个局部诊断各700；合计10500、余量300，累计10800秒。无更新首4病例测L forward/backward、A forward及A forward/backward、A/L自由评价计时。训练估计乘1.5并各加120秒；任何更新前冻结budget_plan，不适配保护配额则停止、不增加预算。共享账本累计各GPU作业墙钟秒数。

预检后COV/COV-A用GPU0/1，M1局部诊断用GPU2。分支完成即同卡一次开发评价，GPU2空闲时评价已有最终权重局部视图；最多3作业，全部单卡。工程/预算失败保存state并停止，无静默重复或预算重置。后台启动后简短检查，默认不持续监测。

## 开发指标与预固定比较

原256例全部保留，37阳性/219阴性，25单框/12多框、51区域。greedy80 token、严格最多64框JSON、0–1000 xyxy、原EOS/截断合同。匹配先最大化IoU≥.5个数，再IoU及固定索引ties。invalid不转[]。区域precision以正负全部已知有效预测框为分母，未知数量/invalid另列。

首页先单框严格、阴性错误框、阳性空列表、区域召回/precision、多框覆盖、无效/截断。A报告average_precision AP、AUROC、BA、macro-F1、原p≥.5 TP/FN/TN/FP、敏感度/特异度及精确ties。原固定五折训练四折特异度90%交叉阈值仅补充。A×L交叉表、额外/重复框及始终no+[]的219/256参照保留；后者无视觉定位能力。

固定比较COV-A−COV、COV−L1、COV-A−M1，所有绝对值并列。2000患者配对bootstrap、seed42；重复患者保留原fold，每次重拟合阈值，不可估计记NA。框不独立扩样，单seed区间不含训练随机性。旧开发多次使用，不是独立测试。COV−L1同时改变两类覆盖和重复分配、COV另增加A no_grad诊断前向（不反传），保留历史工程差异，不隔离单一机制。

## 局部可读性

旧crop资格34阳性（22单框、12多框），旧3上下文排除保留，不用dep3人替代。阳性case_id排序，阴性排序后numpy.default_rng(42)排列分配唯一供体，两尺度共享供体。2倍主视图完全复用旧crop_box，2.5倍补充；原边界处理/map_box给阴性同归一化位置/形状，192个crop、68患者、34匹配簇，实际归一化几何误差0。几何失败留原因不补选，不按分数挑图。

统一可见区域问题、no/yes顺序。每crop平均token logprob不含EOS、softmax候选支持、原.5预测。多框先患者内平均支持，报告正检出、负误报、配对均正确、患者内逐crop平均正确率与全部crop严格正确、分布。匹配簇2000次seed42 bootstrap保留供体、全部框/模型/两尺度，不拆crop扩样或调阈值。候选支持不是校准疾病概率。

外部给定区域读取不证明自主选框、全图答案因果依赖、临床可靠性或阴性肺完全正常；跨患者配对不是医学反事实，资格保留旧标注引导与AI上下文排除来源，不冒称专家因果验证。

## 警戒、数据准备与交付

相对原M1的AP下降>.02、阴性出框率增加>5pp仅研发警戒，不是临床标准/非劣界或自动成功gate。不从结果自动追加seed/步数/RL；无有用收益时停止追加同配方，下一步另预指定数据/接口。

PadChest-GR本轮不混入RSNA。当前权限未证实，官方入口/条款已核对，已准备本地申请草稿、缺字段、schema和患者级划分方案，状态pending_user_submission。未替用户接受协议、发送申请、下载或计分；无局部框不能自动成为阴性，详见data_readiness。

CPU检查入口：

```sh
python - <<'PY'
import runpy
runpy.run_path('tests/test_medevidence_p3.py',run_name='__main__')
PY
```

唯一冻结链入口：

```sh
python - <<'PY'
import runpy
runpy.run_path('scripts/launch_medevidence_p3.py',run_name='__main__')
PY
```

患者、视图、逐例框/token/预测/排程、权重及原始日志私有。公开仅独立pilot3代码/配置/聚合/去标识溯源，需本轮授权。源码存在不等于训练完成，以实际收据为准；已有pilot3不得重复启动，旧pilot2报告不作为本轮新结果。
