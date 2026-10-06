"""Aggregate-only local reports; no synchronization, upload, or push."""
import json
import os
from pathlib import Path
from src.v2 import save,STAGES

REPORTS={'P0':'score_audit','P1':'overfit32','P2-summary':'sft_ablation','P3':'region_audit','P4':'reward_shadow','R0':'original_task_probe'}


def public_boundary(value):
    if isinstance(value,dict):
        forbidden={'image_id','case_id','patient_id','predictions','samples','text','tokens','parameter_versions','sampling_seeds'}
        if forbidden&set(value):raise ValueError('Private fields in aggregate report: '+str(forbidden&set(value)))
        for v in value.values():public_boundary(v)
    elif isinstance(value,list):
        for v in value:public_boundary(v)
    elif isinstance(value,str):
        if any(s in value for s in ['/data/bmw/','/Users/','wangbomin']):raise ValueError('Private deployment path in aggregate report')


def render(root):
    root=Path(root);out=root/'reports';out.mkdir(exist_ok=True)
    statuses=json.loads((root/'stage_status.json').read_text());budget=json.loads((root/'budget.json').read_text());aggregate={'stages':statuses,'budget':budget,'results':{}}
    # Raw attempt PID/host provenance remains private; aggregate durations include failures.
    aggregate['budget']={k:v for k,v in budget.items() if k!='attempts'}
    aggregate['budget']['attempts']=[{k:v for k,v in a.items() if k!='pid'} for a in budget['attempts']]
    for stage in STAGES:
        p=root/stage/'summary.json'
        if p.exists():aggregate['results'][stage]=json.loads(p.read_text())
    public_boundary(aggregate);save(out/'rsna_v2_results.json',aggregate)
    for stage,name in REPORTS.items():
        state=statuses[stage];result=aggregate['results'].get(stage)
        sections=[f'# RSNA diagnostic v2: {stage}','',f"状态：{state['status']}；原因：{state['reason']}。",'',
                  '## 实际测得','', '```json',json.dumps(result if result is not None else {'actual_samples':0,'steps':0,'result':'not_measured'},indent=2,ensure_ascii=False),'```','',
                  '## 根据结果的解释','']
        if result is None:sections+=['该阶段尚无测量结果；状态不代表科学 gate 成功或失败。']
        elif stage=='P0':sections+=['所有结果来自已参与历史 checkpoint 选择的开发集。五折交叉阈值不消除该依赖；候选 softmax 不是校准后的疾病概率。原始 0.5 阈值与折外结果分别保留。黑图/错配图属于图像依赖压力测试。']
        elif stage=='P1':sections+=['仅连续两次满足阳性、阴性各至少 15/16 才通过训练内可学习性条件；loss 下降或进程退出码 0 不构成通过。']
        elif stage=='P2-summary':sections+=['主比较固定为 step512 的 SFT-N 与 SFT-B；旧 B1 仅作历史参考。S* 只是开发筛选，不是独立泛化证据。']
        elif stage=='P3':sections+=['当前没有合格胸片人工语义审核，也没有可验证肺野信息。仅为几何探索；主要特异性指标为 D_E-D_wrong。正 M 可能被负 D_N 推高，wrong-region 可能仍与病灶重叠。']
        elif stage=='P4':sections+=['候选只生成一次并用于全部奖励比较。合法二元答案在固定孤立计分下只有两个确定奖励值；正序仿射变换的标准化优势相同。报告具体分母、数值分支和反转；无可用组报告 no_usable_groups。该阶段没有反向传播或参数更新。']
        else:sections+=['缺少作者合法本地 counting/presence 训练或探针数据及可核验运行配置时保持 blocked，不制造替代复现。']
        sections+=['','## 尚未验证的假设','','独立测试泛化、临床有效性和可靠医学证据使用均未验证。单 seed 开发结果不证明普遍改进。','',
                   '## 阻塞项','', 'clinical_review_pending；R0 作者数据/配置未齐备。若预算、P1 或工程前提不满足，按 stage_status 保留 stopped_budget、failed 或 blocked。','',
                   '口径：标签顺序 no/yes；分类为不含 EOS 的候选平均 token logprob；SFT 含 EOS 并分项记录；原始阈值 p>=0.5。分母及 checkpoint 见实际测量记录。区间为 2,000 次患者级 bootstrap；阈值比较在重复采样中重新拟合且保留原 fold。训练内 P1 不计算泛化区间。测试图像未访问；RL 未运行。','']
        (out/('rsna_v2_'+name+'.md')).write_text('\n'.join(sections))
    return aggregate

if __name__=='__main__':render(os.environ['OUTPUT_ROOT'])
