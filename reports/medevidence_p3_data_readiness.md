# MedEvidence pilot-3 声明—区域数据接入

状态：`pending_user_submission`。核对日期：2026-10-07（Asia/Shanghai）。当前项目私有材料文件名索引、指定远端项目data顶层及旧pilot1记录未提供PadChest-GR访问收据或合法数据材料；不能据此声称已获权限。本轮实际读取新数据元数据0条、下载0、提交申请0、模型计分0。

已核对[官方项目页](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)与[官方申请表](https://docs.google.com/forms/d/e/1FAIpQLScTnT0aUCQERfiHet7QZ7mQeED-V-veKlmEWRMEpiU26ePQag/viewform)：表单要求Email、Name、Affiliation、Objective。申请草稿已在本地准备，姓名、邮箱和正式机构信息待用户填写。未接受协议、发送邮件或提交申请。

[官方研究使用条款](https://bimcv.cipf.es/bimcv-projects/padchest/padchest-dataset-research-use-agreement/)规定研究用途，不允许未经书面许可分发数据，禁止尝试重新识别患者，不可用于患者诊疗。这里仅记录条款，未替用户接受。

[当前论文v2](https://arxiv.org/abs/2411.05085v2)描述英语/西班牙语发现句，阳性句最多有两个独立读者的框标注集。下面是拟接入的规范化schema，原始字段名和数据版本要在权限及实际材料到位后验证，不能冒称已完成schema检查：

| 层级 | 所需字段与区别 |
|---|---|
| patient / study / image | 患者键、study键、image键、view、原始划分、图像尺寸及版本 |
| finding / statement | 原文、语言、阳性/阴性/不确定、概念与映射来源/版本 |
| grounding | 区域列表、坐标单位、local/global/unlocalized/missing、读者ID与标注来源/版本 |
| provenance | 发布版本、合法访问记录、原始记录键、映射和处理版本 |

没有局部框不能自动变成阴性；全局阳性、缺失定位、不确定与真正不存在分别保留。读者标注集不自动取并集，不复制RSNA的[]合同。

下一阶段划分方案：先验证官方split是否患者互斥；按患者汇总其所有studies/images/statements，检查跨study与跨image重复、患者键缺失和读者标注重复；可用的官方患者级划分优先保留，无法确认时阻塞划分，不用图像级随机划分替代。图像重复先检查标识与已提供的校验字段，发现疑点后才作定向像素比较。概念映射仅依据训练材料/预指定词表建立，评价数据不参与筛选方法。下一阶段训练、模型计分和新划分需要单独预指定授权。
