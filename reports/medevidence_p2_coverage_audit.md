# MedEvidence pilot-2 证据覆盖审计

E0只读取元数据、实际源码流程与旧审核记录，未重新采样对照、未重审。

| 阶段 | 训练 | 开发 |
|---|---:|---:|
| original_positive_patients | 149 | 37 |
| original_single_box | 71 | 25 |
| original_multi_box | 78 | 12 |
| area_side_eligible | 69 | 25 |
| geometry_attempted_source_loop | 69 | 25 |
| geometry_success_inferred_from_source_and_aggregate | 69 | 25 |
| actual_proposals | 32 | 25 |
| review_completed | 32 | 25 |
| accepted | 3 | 3 |
| eligible_not_proposed_or_reviewed | 37 | 0 |

实际旧源码先为每个面积/边长合格病例调用control_box，再检查训练提议上限32。因此未进入提议的病例不是完全未尝试几何：它们做过几何搜索但结果没有逐例保存。旧全量几何失败记录为0，可以据执行源码和聚合收据推断所有满足面积/边长者通过几何；对未提议者不能声称完成语义审核。

主要缩减来自单框/局部面积的任务定义、训练首32提议范围以及已审核病例中的语义适用性。不同阶段和unknown原因见JSON；旧reject_or_unknown无法再拆成纯拒绝与不确定。37个总计框外/非局部信息原因有记录，其余目标/control/解剖上下文之间不能精确归因；多标签重叠未记录，不把推测补成审核事实。

crop资格与dep资格不同：dep未通过不删除原图L监督。原始区域监督、区域—发现对应、成对证据受限比较分别保留各自适用边界，不共用总gate。

后续证据任务的首选建议：改为较少依赖“框内目标信息完全消失”的区域—发现对应任务。依据是当前最明显排除原因为框外/非局部信息，增加相同删除式候选不能消除定义限制；需要验证区域与发现的对应标注和可观察上下文。该任务只能证明局部对应，不能证明删除目标后的因果答案变化。本轮不执行新证据训练或数据扩展；定位探针后的总研发优先级另见最终decision。
