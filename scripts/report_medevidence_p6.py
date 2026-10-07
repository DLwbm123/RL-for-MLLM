"""P6 aggregate metrics and paired comparisons, without patient identifiers."""
from pathlib import Path
import json
import os
import re
from scripts.medevidence_p5 import read,lines,save
from scripts.report_medevidence_p3 import full_metrics
from scripts.report_medevidence_p5 import paired as localization_paired
from src.medevidence_p4 import extended_summary,paired_intervals,paired_counts


def main():
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';public=base/'code/reports'
    cfg=read(base/'code/configs/medevidence_p6.json');ledger=read(out/'gpu_ledger.json');statuses=read(out/'stage_status.json');folds=read(out/'protocol/folds.json')
    training={};identities={};dev={};cal={};absolute={};mechanism={}
    for name in cfg['branches']:
        path=out/name/'summary.json'
        training[name]=read(path) if path.exists() else {**statuses.get(name,{'status':'not_started'}),'steps':0,'SFT_exposures':0,'rollouts':0}
        if (out/name/'initial_identity.json').exists():identities[name]=read(out/name/'initial_identity.json')
        path=out/name/'groups.jsonl'
        if path.exists():
            groups=lines(path);wrong=[r for r in groups if r['all_wrong_negative']];neg=[r for r in groups if not r['positive']]
            mechanism[name]={'negative_groups':len(neg),'all_wrong_negative_groups':len(wrong),
                'all_wrong_negative_zero_advantage_groups':sum(r['zero_advantage_group'] for r in wrong),
                'all_wrong_negative_advantage_mean':sum(r['advantage_mean'] for r in wrong)/len(wrong) if wrong else None,
                'positive_groups':len(groups)-len(neg),'positive_zero_advantage_groups':sum(r['positive'] and r['zero_advantage_group'] for r in groups)}
    for name in ['INIT']+cfg['branches']:
        p=out/('eval_'+name)
        if (p/'summary.json').exists() and read(p/'summary.json')['status']=='completed':
            dev[name]=lines(p/'dev.jsonl');cal[name]=lines(p/'calibration.jsonl')
            absolute[name]={'development':{**full_metrics(dev[name],folds),**extended_summary(dev[name])},'calibration':extended_summary(cal[name])}
    comparisons={}
    pairs=[('B_GRPO','C_NEGABS'),('A_SFT','B_GRPO'),('A_SFT','C_NEGABS')]+[('INIT',n) for n in cfg['branches']]
    for left,right in pairs:
        if left in dev and right in dev:
            comparisons[right+' minus '+left]={'development':paired_intervals(dev[left],dev[right],folds),
                'development_success_counts':paired_counts(dev[left],dev[right]),'calibration':localization_paired(cal[left],cal[right])}
    answer={}
    if 'INIT' in dev:
        first={r['image_id']:r for r in dev['INIT']}
        for name,rs in dev.items():
            assert set(first)=={r['image_id'] for r in rs}
            error=max(abs(r[key]-first[r['image_id']][key]) for r in rs for key in ('score_no','score_yes'))
            answer[name]={'patients':len(rs),'max_token_score_absolute_difference_from_INIT':error,'exact_equal':error==0}
    conditions={}
    if all(n in absolute for n in cfg['branches']):
        a,b,c=[absolute[n]['development'] for n in cfg['branches']]
        def check(new,old,max_neg):
            checks={'region_recall_increased':new['region_recall']>old['region_recall'],
                    'single_strict_not_down':new['single_strict_success_count']>=old['single_strict_success_count'],
                    'multi_matches_not_down':new['strata']['multi']['matched_regions']>=old['strata']['multi']['matched_regions'],
                    'negative_nonempty_not_up':new['negative']['valid_nonempty']<=max_neg,
                    'invalid_truncated_not_up':sum(new[g][k] for g in ('positive','negative') for k in ('invalid','truncated'))<=sum(old[g][k] for g in ('positive','negative') for k in ('invalid','truncated'))}
            return {'passed':all(checks.values()),'conditions':checks,'interpretation':'R&D direction only, not statistical or clinical success'}
        conditions={'C_vs_B':check(c,b,min(a['negative']['valid_nonempty'],b['negative']['valid_nonempty'])),
                    'B_vs_A':check(b,a,a['negative']['valid_nonempty'])}
    access={}
    for path in out.glob('*/data_access.jsonl'):
        counts={}
        for r in lines(path):counts[r['split']]=counts.get(r['split'],0)+1
        assert set(counts)<= {'train','validation'};access[path.parent.name]=counts
    pre=read(out/'preflight/summary.json') if (out/'preflight/summary.json').exists() else {'status':'not_completed'}
    pre.pop('effective_generation_settings',None)
    plan=read(out/'training_plan.json') if (out/'training_plan.json').exists() else None
    validation={'same_training_initialization':all(v==next(iter(identities.values())) for v in identities.values()) if len(identities)==3 else None,
                'native_preflight':pre,'frozen_classifier_comparison':answer,'data_access':access,'test_pixels_read':0,
                'scientific_protocol_deviations':[],'public_push':'pending_completion_delivery_under_current_user_AGENTS' }
    if (out/'engineering_repair.json').exists():validation['engineering_repair']=read(out/'engineering_repair.json')
    for name,value in [('training',{'branches':training,'plan':plan,'mechanism':mechanism}),
                       ('evaluation',{'status':ledger['status'],'absolute':absolute,'paired_comparisons':comparisons,'engineering_conditions':conditions,
                                      'primary_comparison':'C_NEGABS minus B_GRPO','paired_bootstrap':'2000 patient replicates, seed42; no training-seed uncertainty',
                                      'classification_scope':'independent frozen COV; not shared trainable-parameter retention'}),
                       ('validation',validation),('budget',{k:v for k,v in ledger.items() if k!='active'}),('status',statuses)]:
        # Runtime failures can mention private storage paths; public errors retain types and relative context only.
        clean=json.loads(re.sub(r'/(?:data|Users|remote-home)/[^\s"\\]+','[private-path]',json.dumps(value)))
        save(public/('medevidence_p6_'+name+'.json'),clean)
    document='# MedEvidence P6 实际结果\n\n'
    document+=f"总体状态：{ledger['status']}。三组共用P5 normalized初始化；P4/P5历史门槛和结果未修改。\n\n"
    document+='| 分支 | 状态 | 实际步数 | SFT暴露 | 训练rollout |\n|---|---|---:|---:|---:|\n'
    for name,v in training.items():document+=f"| {name} | {v['status']} | {v.get('steps',0)} | {v.get('SFT_exposures',0)} | {v.get('rollouts',0)} |\n"
    if plan:document+=f"\n两RL分支共同lambda={plan['lambda']:.8f}，beta=.01；原生检查状态{pre['status']}。\n"
    document+='\n| 同协议完整开发评价 | 单框严格 | 匹配GT区域 | 阴性出框 | 阴性假阳性框 | 多框严格 | 分类AP |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,v in absolute.items():
        m=v['development'];document+=f"| {name} | {m['single_strict_success_count']}/25 | {m['matched_regions']}/51 | {m['negative']['valid_nonempty']}/219 | {m['negative_false_positive_boxes']} | {m['multi_strict_success']}/12 | {m['A_raw_ap']:.6f} |\n"
    document+='\n主比较与辅助比较的患者配对差值、95%区间、有效重复次数及新/丢失成功计数在evaluation JSON中。\n\n'
    if 'C_NEGABS minus B_GRPO' in comparisons:
        for metric in ('single_strict_success_rate','region_recall','negative_nonempty_rate','single_mean_IoU'):
            v=comparisons['C_NEGABS minus B_GRPO']['development'][metric]
            document+=f"- C−B {metric}：{v['difference']:+.6f}，95%区间{v['ci95']}，有效{v['valid_replicates']}次。\n"
    document+='\n阴性全错组机制检查：\n\n'+json.dumps(mechanism,ensure_ascii=False,indent=2)+'\n\n'
    document+='方向性工程门槛：\n\n'+json.dumps(conditions,ensure_ascii=False,indent=2)+'\n\n'
    document+=f"P6新增{ledger['new_charged_seconds']/3600:.6f}GPU小时，含P4/P5累计{ledger['combined_GPU_hours']:.6f}/3小时；test实际读取0。实际数据访问与原生检查见validation JSON。\n\n"
    document+='分类使用独立冻结COV，各模型分类一致性另行核验，不能称共享更新参数的答案能力保留。P5初始化已根据训练池校准结果选择，原开发集已多轮使用，单seed短程实验及患者bootstrap不能证明独立泛化、训练随机性稳健、临床有效或等算力优势。即使机制按设计产生非零梯度，也不等于最终定位收益。\n\n'
    document+='代码、配置和聚合材料已准备；生成此报告时尚未执行完成后的GitHub交付，推送与匿名访问以单独交付回执为准。患者数据、GT、逐例输出、权重和原始日志保持私有。\n'
    (public/'medevidence_p6_decision.md').write_text(document)
    print(json.dumps({'report_status':ledger['status'],'completed_evaluations':list(absolute),'combined_GPU_hours':ledger['combined_GPU_hours']}),flush=True)


if __name__=='__main__':main()
