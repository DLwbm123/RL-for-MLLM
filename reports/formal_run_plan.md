# 正式运行计划（尚未授权启动）

本轮只执行 seed 17 的有限 pilot。正式关键比较使用 seeds 17/42/123、相同 split seed 42，以及 B3/B4/B5/B6，保留 B2 继续训练对照和共同 B1 起点。E1/E3、去除各辅助项、随机对照、噪声/偏移区域、未参与训练的像素干预列为后续冻结消融，不在本轮扩充。

当前训练集合为 744 Case。有效病例批量 4 时，一遍病例为 186 个 optimizer updates；正式后训练 1/2/3 遍分别为 186/372/558 次。每种方法保持相同病例顺序、A:B=3:1、相同 LoRA、相同监督 anchor；RL 每次更新在首个 A 样本生成 8 条候选，辅助约束也只取该 25% 子批，资格不足时如实跳过。应同时汇报完整 forward、vision encode、生成 token 及 GPU 小时，不能把相同步数称为同算力。

单 GPU pilot 先测吞吐和显存，不预设占用四卡。正式资源估算以实际 `summary.json` 中每步耗时乘以更新数和 seeds；尚无实测值时不填写虚构 GPU 小时。另列共同 SFT、冻结审计及像素重新编码成本。分布式真实模型一致性尚未验证，正式多卡前须补检。

执行前必须确认：pilot gate、原图性能、合格覆盖率及良恶性构成、文字/测量标记捷径、资源预算；病例聚合继续关闭，除非同 Case 同目标身份获得依据。测试集只在协议及 checkpoint 选择冻结后开启。温度/阈值只能用 validation。

现有可运行 pilot 命令（服务器路径与设备由环境变量提供）：

```bash
python3 scripts/sync.py
python3 scripts/dispatch.py smoke
python3 scripts/dispatch.py pilot
```

这三条命令分别同步代码、做单卡实测检查、启动有限后台流程；不要重复启动正在运行的 job。`configs/formal.json` 当前故意关闭正式执行。正式授权后，先根据 pilot 实测写入正式配置和版本锁，再开启正式入口，不使用本轮未验证的批量或后端。未通过证据 gate 时，不启动正式方法矩阵。
