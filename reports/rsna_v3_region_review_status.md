# RSNA v3：区域与替换来源的 AI 盲审

用户在初始 CPU 阶段后明确授权 GPT Astra 替代本轮人工审核条件。**全部 93 例旧区域的 AI 审核已完成，930 行无缺失或重复；旧接口完整通过 0/93 例。** 这是 AI 语义筛查，审核者不具备人类临床资质；原人工表仍为 human_review_pending，不能把 AI 判断改称临床专家结论。

## 旧区域审核结果

| 组件 | acceptable | uncertain | rejected |
|---|---:|---:|---:|
| control_1:region | 5 | 10 | 78 |
| control_1:replacement_source | 0 | 0 | 93 |
| control_2:region | 5 | 15 | 73 |
| control_2:replacement_source | 0 | 1 | 92 |
| control_3:region | 2 | 10 | 81 |
| control_3:replacement_source | 0 | 0 | 93 |
| evidence:region | 64 | 28 | 1 |
| evidence:replacement_source | 0 | 0 | 93 |
| wrong:region | 53 | 38 | 2 |
| wrong:replacement_source | 0 | 1 | 92 |

合计 129 acceptable、103 uncertain、698 rejected。每例资格要求 evidence 区域及来源、三个 controls 区域及来源共8项全部 acceptable；任一 uncertain/rejected 均排除。历史 wrong 只作为 shifted-overlap probe 单独记录，不参与此资格门禁。第一批原 P4 32例：0通过、32排除；第二批61例：0通过、61排除。单个 evidence 可接受不代表其替换来源或整个接口可接受。

全部 evidence 来源和 control_1、control_3 来源均被判 rejected；control_2 来源为92 rejected、1 uncertain。主要视觉理由涉及心脏/纵隔、胸壁、膈下、标注或可疑异常、文字或器械等。该结论只描述这批具体来源，不能推断所有 feature replacement 方法均无效，也不把影像判断视作新的疾病诊断。

## 范围、盲法与实际检查

沿用 v2 全部93名几何合格患者：训练69、开发24。seed42盲化；首批原P4全部32例，次批61例，不按奖励或反转挑选。三个同型号Astra实例分担互不重叠病例，每例旧接口由一个实例审核；不是三位专家共识，也没有独立重复审核的一致性估计。

每例12个原分辨率面板：原图、全部已有标注、evidence、三个control、历史wrong各自的区域与来源。1116个面板均有实际查看记录；组合图辅助浏览，疑难处另开原分辨率图。来源显示实际离散视觉token单元，不用包围框填平空白。审核者只读取盲号、train/development用途、图像、几何及模板；奖励、正确性、反转、干预分数及身份映射不提供。审核CSV、逐例视觉理由和查看记录均私有。

肺野coverage始终unknown；未见标注不等于正常肺。wrong的约0.42 token IoU是相对evidence的历史错位重叠，不属于三个controls的共同重叠指标。原人工pending表和旧区域未覆盖，AI结果另存。

## 唯一一次语义修订

全部旧审核完成后，依据事前保存的 configs/rsna_v3_ai_region_revision.json 开始修订。仅处理原P4的32例：22例原evidence区域acceptable，10例不acceptable，后者直接保留排除，不能缩小evidence来取得资格。

Astra仅从原图/标注提出保守的非证据支持多边形。确定性程序保留evidence框与tokens，要求三个controls同面积/形状、同token形状，垂直中心差≤.10、相互重叠≤.75，避开扩张10%的evidence边界；来源为局部三格ring与支持区的交集，至少4个token，所有来源和control单元完整位于支持区。支持区是AI提议，非真实肺分割。

新生成的8项区域/来源须再次由Astra实际看图且全部acceptable。任何不合格保留原因，不做第二轮重抽，不读取奖励来调区。最终manifest在任何模型重计分前冻结。本轮单次修订已完成：19例非空支持提议，3例虽evidence可接受但无可确认支持，10例原evidence排除；进入几何生成的22例全部无法容纳一个完整同形control块，故最终0/32合格、32排除，其中19例同时缺少局部evidence来源。没有可进入最终8项视觉审核的候选接口，G1不启动。全部排除已冻结；详见rsna_v3_G1_status.md和JSON。

## 私有边界与验证

审核包生成时复用DevelopmentData白名单与文件访问钩子，核验冻结v2协议；实际图像读取为训练69、开发24、测试0。原始生成检查确认1116张PNG、930条pending原表及盲号/结果字段边界。区域包1项、AI合并2项、单次区域生成1项合成检查已通过；这些工程检查不代替视觉判断。无新依赖、无在线审核平台、无公开患者内容。
