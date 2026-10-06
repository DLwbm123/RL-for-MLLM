# RSNA diagnostic v2: P2-summary

状态：blocked；原因：P1 did not pass; P2 cannot start。

## 实际测得

```json
{
  "actual_samples": 0,
  "steps": 0,
  "result": "not_measured"
}
```

## 根据结果的解释

该阶段尚无测量结果；状态不代表科学 gate 成功或失败。

## 尚未验证的假设

独立测试泛化、临床有效性和可靠医学证据使用均未验证。单 seed 开发结果不证明普遍改进。

## 阻塞项

clinical_review_pending；R0 作者数据/配置未齐备。若预算、P1 或工程前提不满足，按 stage_status 保留 stopped_budget、failed 或 blocked。

口径：标签顺序 no/yes；分类为不含 EOS 的候选平均 token logprob；SFT 含 EOS 并分项记录；原始阈值 p>=0.5。分母及 checkpoint 见实际测量记录。区间为 2,000 次患者级 bootstrap；阈值比较在重复采样中重新拟合且保留原 fold。训练内 P1 不计算泛化区间。测试图像未访问；RL 未运行。
