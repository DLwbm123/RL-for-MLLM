# MedEvidence pilot-2：定位学习诊断与证据覆盖分析

本轮从`a94f1e2db2ca0243ec8e78c3c2783125a1f40e6a`的干净`codex/r812`工作区建立独立配置、源码与输出。目的为定位失败分解、token路径诊断、小样本可学习性和覆盖审计，非论文配置复现或临床验证。历史源码、报告、gate及私有产物不覆盖。

## 授权与范围

D0/E0和CPU准备已获本轮任务授权。D1/L1与公开push需要本轮单独授权，旧授权不继承；初始资源方案为单卡、累计2 GPU小时，GPU与公开发布分别取得本轮授权。运行器硬性检查私有授权收据`private/medevidence_p2/authorization.json`；用户后续明确“我授权你使用所有 gpu”，已获得本轮GPU授权；全部GPU编号可选，独立冻结诊断最多3作业并行，L1仍单卡。累计2GPU小时不增加。公开发布已另获用户明确回复“授权推送”，仅含代码、配置和聚合报告。源码存在不等于已训练，真实阶段状态以stage_status和GPU账本为准。

仅D0历史CPU重放、E0旧审核覆盖审计、D1三个冻结checkpoint的64例诊断以及唯一L1定位探针。禁止重跑M0/W/M1/M2，禁止dep/control/crop训练、RL、换backbone/LoRA范围/视觉解冻、提示/参数/seed搜索、追加更新和新数据扩展。不跑有optimizer更新的smoke。

## 历史重放与样本

D0读取真实256例M1/M2私有文本与原manifest，重新严格解析、原一对一匹配并逐字段核对历史结果。主分类阈值0.5，历史五折只按保存阈值和fold作补充，不重新拟合。展示阳性空输出、区域召回、严格单框成功、阴性非空和分类—定位不一致，总体联合与始终no+[]为补充。invalid保留，不能转换成[]。每预测框最高IoU的GT近邻（tie按中心距离）只诊断，不计一对一检出；模板频数和中心/尺寸散度不单独证明模板机制。

Fit32和Ref32各16名单框阳性、16名阴性，只从原1024训练池选。在每类别按case_id排序，使用同一个numpy.default_rng(17)，先阳性随机排列再阴性；各前16为Fit、接下16为Ref。未使用分数、正确性、AI资格或奖励。64患者互斥，保持原归属。Ref是本轮不更新参照，可能被历史模型看过；原256例开发集继续是已使用开发数据。

冻结患者/图像/原始框、GT JSON、选择顺序、token目标及排程。所选64图像按本轮明确要求记录SHA256、大小和mtime；基础模型复用冻结revision和分片bytes/mtime，不全权重哈希。旧adapter CPU张量身份核对，GPU加载再核对实际参数。

## 编码与工程

保持原A/L提示、greedy、最大80 token、严格最多64框JSON、0–1000排序xyxy、576–1024视觉token和processor。图像使用原冻结派生原图，不翻转、转置或加入GT；4例固定Fit阳性原图/GT叠图仅私有工程对齐，不产生新疾病真值或重审证据资格。全部1280个GT自匹配及坐标round-trip误差≤每轴尺寸/2000像素；64所选图像核对尺寸和实际processor网格。继承真实DevelopmentData文件系统白名单，CPU探针实际拒绝未授权数据路径。

实际processor tokenizer核验目标完整解码和token offsets或逐步decode边界。所选单框GT含EOS为18 token，[]含EOS为2 token；初始[[与[]为不同token，共同答案token前缀0。跨度互斥分为首分歧、数字、结构、mixed、EOS；首分歧优先，混合字符token不重计。逐例边界私有。

浮点对齐容差在模型测试前固定token logprob atol=.001、rtol=.001，并报告实际差异；adapter/optimizer/RNG身份精确比较，不随失败不断放宽。CPU回归覆盖固定选择、2+2 L-only排程、每患者32次、invalid/空/额外框分母及token跨度。评分器通过不等于模型定位有效；若发现历史实质评分错误，保留旧结果并暂停对应训练。

## D1冻结诊断

W step256、M1/M2固定最终累计512，同Fit32/Ref32测A、自由L及教师强制。M1为主，其他缺失checkpoint单列blocked；L1不按诊断更换初始化。保留自由生成完整token/文本、EOS位置、长度、截断、匹配、前4位置top8原始logprob支持和实际generation_config；缓存包含model、任务和图像身份。输入不含标签或GT。

阳性计真实完整GT与[]，分别含及不含EOS的sum/mean/length/逐token分数；阴性仅真实[]。首分歧两种token在真实共同prompt+答案前缀下比较，并记录自由路径是否进入该前缀/分支。两个字符串分数不是所有非空列表概率；总分与均分都不能消除全部长度效应，GT路径条件数字支持不等于自由定位能力。

## 唯一L1训练

固定M1最终权重，重置AdamW与本轮seed17状态，optimizer_reset=true。固定基础revision、语言q/v LoRA r16/alpha32/dropout0、视觉/merger冻结、bf16和processor。AdamW lr2e−5、betas(.9,.999)、eps1e−8、weight_decay.01、clip1、固定lr。

Fit32每步阳性/阴性/阳性/阴性，按冻结类内队列循环、无重抽。微批量1，四例完整目标含EOS平均token NLL各除4，一次optimizer.step。仅L，没有A/crop/control/dep、额外负样本、首token或数字权重。固定256更新、1024暴露、正负各512，每患者32次。患者权重相同但首token在阳性/阴性患者平均loss的权重分别1/18和1/2；这仅诊断，不自动改loss。

固定首4例做无optimizer更新的梯度检查，视觉无梯度、adapter/optimizer未变，恢复RNG。保存0/32/64/128/256，主结果固定256，无最佳点选择。0可复用D1_M1完全相同model/任务/图像身份的缓存；各点Fit/Ref自由L与教师读数，A仅0/256。eval/no_grad结束恢复训练模式和Python/NumPy/Torch/CUDA RNG，要求身份精确相等。记录总/正负NLL、token跨度、梯度、实际暴露、checkpoint及GPU时间。

## GPU预算与停止

全部GPU编号可选，每作业单卡、最多3个独立冻结诊断并行，累计7200秒，含加载、D1、无更新梯度检查、训练、评价、失败及退出。名义配额D1_M1 800秒、D1_W 800秒、D1_M2 800秒、L1及Fit/Ref评价4000秒、可选最终开发700秒，共享安全余量100秒。D1_M1固定前8例测吞吐；任何L1更新前保存完整budget_plan，检查保护的训练/小集合评价配额可行性。科学效果差不增加步数。

三个D1冻结模型可在不同GPU并行；D1_M1完成且预算分配冻结后，等待其他既定D1作业结束，再执行唯一单卡L1→可选固定最终开发。未增加独立诊断样本或模型，启动前各卡至少24000 MiB显存，保留峰值余量。已完成阶段的未用预约只能释放给最终开发评价，不增加训练或重复诊断。预算触及时保存模型/optimizer/RNG/位置，标stopped_budget，保留partial与不利病例；工程异常按阶段停止，科学表现差记completed_diagnostic。默认可靠后台启动后一次简短检查，不持续监测。

## 评价、覆盖与判断

开发只在step256生成一次A/L，全部256例含12名多框阳性，不改阈值或解析器；与缓存M1患者配对2000次bootstrap seed42，所有框成簇，区间不含训练seed随机性。Fit32只训练内可学习性；Ref32只不更新参照。无条件IoU含空和失败0，条件非空IoU仅补充，阴性误报与分类代价并列。

E0重建原149/37阳性→单/多框→面积/边长→几何→提议→旧审核→接受，区分首32提议与更多候选。旧源码为所有面积/边长合格者尝试几何，上限外不保存逐例对照；旧聚合几何失败0，仅据源码和收据推断通过，未审核不能称语义合格。reject_or_unknown不虚拆，未记录多标签保持unknown。crop、dep与原图L资格分开，原始区域监督、区域—发现对应、成对证据受限比较不用总gate。本轮不扩旧3+3，不重算旧主结果。

依据实际计数、曲线、首分歧、GT路径、阴性误报和分类代价给单一优先建议，不用自动总gate或任意成功阈值。后续接口、监督或证据方案均需新授权。

## 已实现入口与公开边界

CPU回归：

```sh
python - <<'PY'
import runpy
runpy.run_path('tests/test_medevidence_p2.py',run_name='__main__')
PY
```

CPU检查冻结且本轮GPU授权收据存在后，唯一链入口：

```sh
python - <<'PY'
import runpy
runpy.run_path('scripts/launch_medevidence_p2.py',run_name='__main__')
PY
```

读取私有runtime和授权，拒绝重复launch/ledger。远程controller/子进程均中性python -u -，启动后核对ps/nvidia-smi。私有标识、图像、GT、逐例文本/token/叠图/预测/排程、权重及原始日志不公开；本次已运行代码、配置和聚合报告可公开；本轮push已获得明确授权。

资源授权修订发生在任何GPU模型计分前：仅并发与候选GPU范围变更，CPU初版冻结保留为私有版本记录；患者、选择、排程、GT/token、任务目标、训练步数和数据归属完全保持原冻结摘要。后续锁更新保留来源，未重复选择或启动。
