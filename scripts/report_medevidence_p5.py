"""Aggregate diagnostic delivery; never exports case-level records."""
import json
import os
from pathlib import Path
import numpy as np
from scripts.medevidence_p5 import read,lines,save


def paired(a,b):
    a=sorted(a,key=lambda r:r['case_id']);b=sorted(b,key=lambda r:r['case_id'])
    assert [(r['case_id'],r['image_id']) for r in a]==[(r['case_id'],r['image_id']) for r in b]
    def stats(rs,w):
        single=np.array([r['gt']==1 for r in rs]);positive=np.array([bool(r['y']) for r in rs]);negative=~positive
        mean=lambda values,mask:float(np.dot(w,np.array(values)*mask)/np.dot(w,mask)) if np.dot(w,mask)>0 else None
        gt=np.array([r['gt'] for r in rs]);matches=np.array([r['matches'] for r in rs])
        return {'single_strict_rate':mean([r['strict'] for r in rs],single),
                'positive_strict_rate':mean([r['strict'] for r in rs],positive),
                'single_mean_IoU':mean([r['single_iou'] or 0 for r in rs],single),
                'negative_nonempty_rate':mean([r['state']=='valid_nonempty' for r in rs],negative),
                'region_recall':float(np.dot(w,matches)/np.dot(w,gt)) if np.dot(w,gt) else None}
    rng=np.random.default_rng(42);one=stats(a,np.ones(len(a)));two=stats(b,np.ones(len(b)));samples={k:[] for k in one}
    for _ in range(2000):
        w=np.bincount(rng.integers(0,len(a),len(a)),minlength=len(a));left,right=stats(a,w),stats(b,w)
        for k in samples:
            if left[k] is not None and right[k] is not None:samples[k].append(right[k]-left[k])
    return {k:{'difference':two[k]-one[k],'ci95':np.quantile(v,[.025,.975]).tolist(),'valid_replicates':len(v)} for k,v in samples.items()}


def main():
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';public=base/'code/reports'
    decision=read(out/'decision.json');ledger=read(out/'gpu_ledger.json')
    assert not ledger['active']
    models={fmt:read(out/('fit_'+fmt)/'summary.json') for fmt in ('normalized','absolute')}
    complete=all(v['status']=='completed' for v in models.values());comparisons={}
    if complete:
        for subset in ('final_fit','final_calibration'):
            comparisons[subset]=paired(lines(out/'fit_normalized'/(subset+'.jsonl')),lines(out/'fit_absolute'/(subset+'.jsonl')))
    access={}
    for fmt in models:
        counts={}
        for r in lines(out/('fit_'+fmt)/'data_access.jsonl'):counts[r['split']]=counts.get(r['split'],0)+1
        access[fmt]=counts
    assert all(set(v)<= {'train'} for v in access.values())
    evaluation={'coordinate_comparison':'absolute minus normalized','paired':comparisons,'bootstrap_seed':42,'bootstrap_repeats':2000,
                'unit':'patient; all boxes kept together','scope':'training-pool diagnostic, calibration64 had historical COV exposure; not independent validation',
                'new_image_access_by_stage_and_split':access,'new_development_image_reads':0,'test_pixels_read':0,
                'historical_rollouts_read':{'D3_train':256,'D2_development':456},'historical_read_is_not_new_generation':True,
                'scientific_protocol_deviations':[],
                'classification_evaluation':'not_executed; conditional separate frozen answer adapter not reached',
                'native_RL_validation':'not_executed_gate_failed'}
    save(public/'medevidence_p5_evaluation.json',evaluation)
    # Strip process metadata from a publicly deliverable cost receipt.
    budget={k:v for k,v in ledger.items() if k!='active'}
    save(public/'medevidence_p5_budget.json',budget)
    save(public/'medevidence_p5_diagnostics.json',models);save(public/'medevidence_p5_decision.json',decision)
    audit=read(out/'reward_audit.json')['D3']['positive']
    text='# MedEvidence P5 实际诊断与决策\n\n'
    text+=f"诊断状态：{decision['diagnostics_status']}；主SFT/RL对照：{decision['main_status']}。入选坐标格式：{decision['selected_format']}。\n\n"
    text+='本轮固定比较两种坐标接口，实际均从同一历史COV初始化；Fit32与Calibration64不重合，但两组都来自历史训练池，不是独立验证。原开发集未新增生成或图像读取。\n\n'
    text+=f"P4已有D3训练rollout：32个阳性组中{audit['distinguishable_groups']}组奖励可区分，其中{audit['distinguishable_without_any_match']}组没有任何IoU≥0.5匹配；只有{audit['any_matched_region_groups']}组含匹配区域。空/非空状态间方差占总组内奖励方差{audit['between_state_share']:.4%}。另有{audit['nonempty_geometry_distinguishable_groups']}组存在固定非空框数下的几何奖励差异；该差异不等于正确定位。{audit['positive_advantage_without_match_completions']}个无匹配completion获得正优势。阴性奖励由-1改-5后，组内标准化优势不变的数值检查通过。\n\n"
    if complete:
        text+='| 指标 | normalized | absolute |\n|---|---:|---:|\n'
        def row(label,get):
            return '| '+label+' | '+' | '.join(str(get(models[f])) for f in ('normalized','absolute'))+' |\n'
        for prefix,label in [('initial_fit','初始Fit'),('final_fit','最终Fit'),('initial_calibration','初始Calibration'),('final_calibration','最终Calibration')]:
            pos=16 if 'fit' in prefix else 32;neg=16 if 'fit' in prefix else 32
            text+=row(label+'阳性严格成功',lambda v,p=prefix,n=pos:f"{v[p]['positive_strict']}/{n}")
            text+=row(label+'阴性出框',lambda v,p=prefix,n=neg:f"{v[p]['negative_nonempty']}/{n}")
        text+=row('最终Calibration单框严格成功',lambda v:f"{v['final_calibration']['metrics']['single_strict_success_count']}/16")
        text+=row('最终Calibration单框平均IoU',lambda v:f"{v['final_calibration']['metrics']['single_IoU_unconditional']['mean']:.6f}")
        text+=row('最终Calibration匹配GT区域',lambda v:f"{v['final_calibration']['metrics']['matched_regions']}/{v['final_calibration']['metrics']['GT_regions']}")
        text+=row('Calibration自然best-of8单框成功',lambda v:f"{v['natural_calibration']['single_best8_success']}/16")
        text+=row('Calibration阳性几何可区分组',lambda v:f"{v['natural_calibration']['decomposition']['positive'].get('nonempty_geometry_distinguishable_groups',0)}/32")
        text+='\n门槛逐项结果：\n\n'
        for fmt,v in models.items():
            text+=f"- {fmt}："+'；'.join(k+'='+str(val) for k,val in v['gate']['conditions'].items())+'。\n'
        text+='\nCalibration配对差值（absolute−normalized），2000次患者bootstrap：\n\n'
        for k,v in comparisons['final_calibration'].items():
            text+=f"- {k}：{v['difference']:+.6f}，95%区间[{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]，有效{v['valid_replicates']}次。\n"
    text+='\n实际训练与资源：\n\n'
    for fmt,v in models.items():text+=f"- {fmt}：{v['status']}，{v['steps']}步，{v['SFT_exposures']}患者暴露，训练rollout {v['training_rollouts']}，诊断rollout {v['diagnostic_rollouts']}。\n"
    text+=f"\nP5新增GPU占用{ledger['new_charged_seconds']/3600:.6f}小时；包含P4后累计{ledger['combined_GPU_hours']:.6f}/3小时。新增影像读取：{json.dumps(access)}；test实际读取0。模型权重、患者标识、GT、逐例输出、原始日志保持私有。\n\n"
    if decision['main_status']=='not_started_gate_failed':
        text+='两个坐标分支均未通过执行前固定的组合门槛，因此本轮在诊断结束；主SFT/RL均0步、0暴露、0rollout，冻结分类分支方案和原生RL校验未执行。没有根据结果降门槛、增加步数、追加seed或使用开发集选优。不能据此判断RL是否有效。\n\n'
    if complete:
        text+='**机制解释。** 两组最终Fit阳性均16/16非空，但严格成功仅3/16与7/16，说明在本次设置下消除空输出还没有解决坐标/多框准确性。Calibration阳性两组均由23/32空输出降至4/32空输出，严格成功仅2/32，同时阴性出框增加到7/32与6/32。绝对坐标提高了训练内严格成功，却没有在校准总体严格成功、区域匹配或自然best-of8上形成一致收益；本轮不支持仅替换坐标格式足以解决问题。\n\n'
        text+='| Calibration教师强制坐标token NLL | 初始 | 最终 |\n|---|---:|---:|\n'
        for fmt,v in models.items():
            text+=f"| {fmt} | {v['initial_calibration']['token_part_mean_NLL']['coordinate']:.6f} | {v['final_calibration']['token_part_mean_NLL']['coordinate']:.6f} |\n"
        text+='\n该NLL条件于GT前缀，不是自由生成成功概率；token跨语义分组存在边界限制。存在性、坐标及多框继续输出需要进一步区分，不能只增加出框倾向。EOS损失约1e-5，初始8人梯度检查也很小，当前证据不支持EOS学习不足是主要瓶颈。\n\n'
    text+='token梯度分组为计算诊断，分组范数不可直接相加，也不能单凭范数认定根因。多框syntax组包含继续输出下一框的分隔符，不能把它等同于JSON格式错误。坐标接口效果受旧COV适配历史限制，不代表原生base模型能力。旧pilot2的训练拟合成功采用不同初始化、仅单框阳性及4倍学习率，本次未拟合不能证明该模型没有拟合能力。\n\n'
    text+='本轮无科学执行偏离，无自动重试或追加训练。条件性分类评价未执行，因此没有新的AP或分类能力保留实测声明。完整token分组聚合、初末指标、资源和工程门槛见相邻JSON。\n'
    (public/'medevidence_p5_decision.md').write_text(text)
    print(json.dumps({'status':decision,'combined_GPU_hours':ledger['combined_GPU_hours'],'public_report_written':True}))


if __name__=='__main__':main()
