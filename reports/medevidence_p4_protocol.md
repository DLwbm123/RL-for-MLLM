# MedEvidence P4 冻结执行协议（计划，不是运行结果）

目标：在同一个冻结 COV step256、相同原图输入、患者排程、SFT 暴露和正式贪心策略下，比较 FULL-SFT 与 FULL-GRL（几何 GRPO 加固定 KL）。单 seed17，累计最多3.00 GPU小时，GPU0/1/2单作业单卡，最多3卡并行。最新P4请求授权预算内实现与运行；未收到P4单独公开推送授权，仅本地提交。

## 数据与输入

原始71名单框阳性、78名多框阳性，加P3同一批256名阴性，共405名训练患者。保持所有原GT；不按crop/dep资格筛选。沿用原256名开发患者（37阳性、219阴性；25单框、12多框、51区域）与原5fold。患者、影像、标签、GT和训练/开发无重叠均由CPU预检验证。旧Ref进入训练者不再作未训练对照。

三层独立seed17 RNG，对各层影像ID稳定排序后循环无放回重新排列；每步single,multi,negative,negative。预生成256步主排程，两分支用相同前缀；256步1024次SFT暴露，128步512次。实际独立患者和暴露次数按层核算；128步时部分阴性只有一次，不能声称完整256步暴露。

沿用历史原图L提示词、0–1000坐标、64框严格JSON上限和完整目标含EOS的每患者平均NLL。问题中没有GT坐标、病灶提示、叠图或crop。最长训练目标含EOS长度由实际冻结tokenizer审计，正式cap=max(80,max_length+16)；超过256即阻断GPU。所有新模型和本轮COV重评共用该cap；P3历史80-token记录独立标记。数据读取白名单缩小为405训练加256开发，sealed test读取0。

## 自然诊断与启动门槛

D1：COV、COV-A各评价全部71训练单框及固定seed17选64训练阴性。输出严格成功、全样本与非空条件IoU及条件n、空输出、阴性出框、无效、截断。

D2：COV在全部25开发单框及seed17固定32开发阴性自然采样8次，temperature0.7、top_p0.95、top_k0。使用独立seed1702采样流；相同cap与提示词，无GT提示或强制非空。另生成同策略贪心。best-of8必须选一份完整合法未截断恰一框输出；不拼框。分开报告阴性单次出框与至少一次出框。GT选优结果只作诊断。

D3：seed17固定16训练单框、16多框、32阴性，每人4份自然completion，temperature1、top_p1、top_k0，独立seed1703采样流。报告ddof0方差、std>1e-6比例、全空、全同奖励、无效、截断以及分层候选质量。

D4：复用25个D2单框预测组，无额外生成；统一归一化空间，对GT作seed42无自配对随机置换1000次，真实与打乱均用合法未截断恰一框的最佳IoU；不合格输出IoU0。

RL三门槛固定：D2自然best-of8成功>=8/25；D3阳性可区分奖励组比例>=30%；D4真实平均最佳IoU减打乱均值>=0.05。三者全通过才进入RL标定，否则FULL-GRL=not_started_gate_failed，仅FULL-SFT。不可改变提示词、阈值、采样数、IoU，或拿SFT后模型重启RL。

不计划可选中间checkpoint曲线、强制非空诊断；保留主阶段预算。没有预验证定位置信标量，因此可选成功—阴性出框曲线记not_completed，不临时发明分数。

## 几何奖励、策略目标与数值

负患者合法未截断[]为+1，其余−1。正患者空/无效/截断为−1；其余用大小min(pred,GT)一对一Hungarian匹配最大化GIoU总和，R=clip((sumGIoU−漏GT−额外框)/GT数,−1,+1)。重复、额外、漏框均保留惩罚；无修框、NMS、格式奖励、答案奖励、裁判或开发GT训练奖励。奖励与匹配detach，策略梯度优化token概率，不对离散坐标直接反传。

FULL-SFT=L_SFT。FULL-GRL=L_SFT−lambda*J_GRPO+0.01*KL。每步4患者，每患者4自然completion，temperature1、top_p1、top_k0；每份按真实生成token（含实际EOS，不加假EOS、不含prompt/padding）平均，再completion平均，再4患者平均。采样前策略的原始FP32 token logprob detach为old，独立reference不是old；每批一次update，不多epoch复用。

组内advantage=(r−mean)/std，ddof0；std<=1e-6则advantage0，但保留SFT和KL。PPO epsilon0.2，min(ratio*A,clip(ratio,0.8,1.2)*A)。关键reward/advantage/logsoftmax/ratio/KL均FP32。d=log_pi_ref−log_pi_current；KL=exp(d)−d−1；无非有限静默钳制。old/current初始化token对齐使用预声明2个bf16 epsilon的atol与rtol=0.015625。

固定reference采用同一冻结base上的独立PEFT reference adapter；每次切换检查训练adapter及gradient版本、恢复训练flags和RNG，标定时另验证optimizer与reference digest。参考分支冻结且无梯度，不随训练更新。

固定首4训练minibatch，零optimizer更新，分别计算四批平均SFT梯度与负GRPO梯度。lambda=min(1,0.25*norm(gSFT)/norm(gRL))；gRL<1e-8或非有限停止RL；恢复模型、空AdamW与主RNG，单独保存初始采样RNG，记录实际梯度比例和是否触顶。全程不开发调参。

## 初始化与训练

两组同一个COV step256，从真实权重digest核验，不用COV-A、L1、M1。冻结Qwen2.5VL7B历史revision；bf16、视觉576–1024token；语言q/v LoRA r16 alpha32 dropout0，5,046,272可训练参数，视觉/merger冻结。

seed17；主模型与优化器初始RNG/状态精确核验，RL独立采样seed1704并保存。AdamW重置，lr5e-6，betas0.9/0.999、eps1e-8、weight_decay0.01，clip1。A系数0，不训练分类。不重复除梯度累积次数。

正式训练前实测首4固定训练minibatch的前向、rollout、old/reference计算和反向，不作optimizer update。共同先尝试256步预算；否则128；仍不可行则stopped_budget。若RL门槛不通过只估SFT。实际吞吐乘预声明1.4安全系数，每训练worker另180秒加载/保存余量，每最终评价预留540秒，共240秒退出缓冲；在总账下冻结具体步数、lambda、预算和门槛/标定哈希后才正式更新。

checkpoint0/64/128，计划256时另存256。固定最终checkpoint主评价，未到最终不提升中间点为主结果。所有GPUworker墙钟包括模型加载、采样、标定、GPU诊断、失败与保存；并行相加，不减免。初始诊断阶段保留最终评价预算，各阶段上限只作工程资源约束，不按结果选模型。

非有限、视觉/参考意外梯度、数据/模型/优化器身份不符、未授权数据、old/reference校验失败或预算不足均停止相关阶段；保存实际步数与失败，不重试、不改算法、超参或预算。

## 最终评价与解释

正式贪心重评COV和完成的新分支，原图、相同cap、相同严格解析、IoU>=0.5与历史一对一匹配。报告全部单/多框成功、IoU条件n、空输出、region recall/precision、阴性出框患者和框总数/每图、无效/截断/超框数、额外/重复框及配对新/丢失/共同成功失败。

原图A仅评价：AP/AUROC，固定0.5全混淆矩阵、敏感度、特异度、平衡准确率、MacroF1；原fold约90%特异度交叉拟合补充，实际达到值如实展示。标签归一化支持度不是校准疾病概率；分类FP和定位阴性出框分开。

主要FULL-GRL−FULL-SFT，辅助各自−同策略COV，固定seed42患者配对bootstrap2000次。所有框随患者一起重采样，重复患者保留原fold并重新拟合阈值；NA不当0，报告有效重复次数。P3 M1/COV-A历史不是本轮重评。单seed、多轮使用过的开发集不是独立泛化或训练随机性验证。

继续研发门槛：RL单框>=6/25且净增>=3；阴性出框<=11/219且不高于新策略COV；region recall高于SFT且多框匹配数不下降；AP下降<=0.02；无效/截断不能隐藏。达到也只值得独立多seed确认，不证明RL必要性、单项几何/KL贡献、等算力优势、因果忠实性或临床效果。

## 交付与未完成项

协议、数据/源码/权重/tokenizer哈希在GPU前冻结；具体正式训练方案依据既定训练吞吐规则在第一正式更新前二次冻结。计划报告与实际聚合报告分开。源码、配置、合成单测和聚合报告可公开；真实清单、GT、身份、影像、患者预测、token、权重和原始日志保留私有。没有本轮推送授权则只本地提交并明确未推送，不阻碍实验完成。

本请求明确要求本次实际完成而不以后台承诺代替交付，因此会话持续到本轮完成或真实阻断；不创建定时监测，不自动下一轮。
