"""Aggregate historical replay and new probe measurements without private rows."""
from collections import Counter
from pathlib import Path
import json
import os
import re
import numpy as np
import pandas as pd
from src.v2 import save, binary_stats
from src.medevidence_p2 import distribution, summarize, enrich


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(v) for v in p.read_text().splitlines()]


def token_summary(records):
    output={}
    for positive in (True,False):
        selected=[r for r in records if bool(r['y'])==positive]
        groups={}
        for mode in ('including_EOS','excluding_EOS'):
            groups[mode]={key:distribution([r['teacher_GT'][mode][key] for r in selected]) for key in ('length','sum_logprob','mean_logprob')}
        groups['token_parts']={category:{'patient_mean_NLL':distribution([r['teacher_GT']['token_parts'][category]['mean_NLL'] for r in selected if r['teacher_GT']['token_parts'][category]['tokens']]),
                                        'contribution_to_patient_mean_NLL':distribution([r['teacher_GT']['token_parts'][category]['contribution_to_sequence_mean_NLL'] for r in selected])}
                               for category in ('first_divergence','digits','structure','mixed','EOS')}
        groups['free_generation_lengths']=distribution([r['generated_length'] for r in selected])
        groups['empty_free_generations']=sum(r['state']=='valid_empty' for r in selected)
        groups['EOS_at_position_histogram']=dict(Counter(i for r in selected for i in r['EOS_positions']))
        if positive:
            groups['GT_minus_empty_specific_string_scores']={mode:{key:distribution([r['teacher_GT'][mode][key]-r['teacher_empty'][mode][key] for r in selected]) for key in ('sum_logprob','mean_logprob')} for mode in ('including_EOS','excluding_EOS')}
            groups['first_divergence']={'patients':len(selected),'actual_common_prefix_lengths':dict(Counter(r['divergence']['position'] for r in selected)),
                'GT_minus_empty_logprob':distribution([r['divergence']['GT_minus_empty_logprob'] for r in selected]),
                'empty_token_higher_teacher_logprob':sum(r['divergence']['GT_minus_empty_logprob']<0 for r in selected),
                'free_prefix_entered':sum(r['divergence']['free_generation_entered_shared_prefix'] for r in selected),
                'free_next_matches_GT':sum(r['divergence']['free_next_matches_GT'] for r in selected),
                'free_next_matches_empty':sum(r['divergence']['free_next_matches_empty'] for r in selected),
                'GT_coordinate_NLL_while_free_empty':distribution([r['teacher_GT']['token_parts']['digits']['mean_NLL'] for r in selected if r['state']=='valid_empty' and r['teacher_GT']['token_parts']['digits']['mean_NLL'] is not None]),
                'interpretation':'Actual tokenizer/shared prompt prefix; zero common-token prefix means first generated-token decision. Teacher coordinates are conditioned on GT path, not evidence of usable free localization.'}
        output['positive' if positive else 'negative']=groups
    output['score_scope']='Two specific strings, never the marginal probability of any nonempty box list; sum and mean both retain length effects'
    output['numeric_prefix_max_error']=max(r['teacher_GT']['prefix_logprob_max_error'] for r in records)
    return output


def diagnostic_metrics(records):
    output=summarize(records)
    if all('p' in r for r in records):output['classification_raw_0_5']=binary_stats([r['y'] for r in records],[r['p'] for r in records])
    return output


def comparison(a,b,repeats=2000):
    assert [r['case_id'] for r in a]==[r['case_id'] for r in b]
    def values(records):
        y=np.array([r['y'] for r in records]);p=np.array([r['p'] for r in records]);cl=binary_stats(y,p)
        pos=[r for r in records if r['y']];neg=[r for r in records if not r['y']];single=[r for r in pos if r['gt']==1]
        return {'positive_empty_rate':sum(r['state']=='valid_empty' for r in pos)/len(pos) if pos else None,
                'positive_region_recall':sum(r['matches'] for r in pos)/sum(r['gt'] for r in pos) if pos else None,
                'single_strict_success_rate':sum(r['strict'] for r in single)/len(single) if single else None,
                'negative_nonempty_rate':sum(r['state']=='valid_nonempty' for r in neg)/len(neg) if neg else None,
                'class_loc_inconsistent_all_patient_rate':sum(r['state']!='invalid' and (r['p']>=.5)!=(r['state']=='valid_nonempty') for r in records)/len(records),
                'overall_joint_supplement':sum((r['p']>=.5)==bool(r['y']) and r['strict'] for r in records)/len(records),
                **{k:cl[k] for k in ('auroc','ap','sensitivity','specificity','balanced_accuracy','macro_f1')}}
    left,right=values(a),values(b);draws={key:[] for key in left};rng=np.random.default_rng(42)
    for _ in range(repeats):
        idx=rng.integers(0,len(a),len(a));va=values([a[i] for i in idx]);vb=values([b[i] for i in idx])
        for key in draws:
            if va[key] is not None and vb[key] is not None:draws[key].append(vb[key]-va[key])
    return {key:{'L1_minus_cached_M1':right[key]-left[key] if right[key] is not None and left[key] is not None else None,
                 'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,'valid_replicates':len(v),'requested_replicates':repeats,'patients':len(a)} for key,v in draws.items()}


def main():
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';public=base/'code/reports'
    status=read(root/'stage_status.json');d0=read(public/'medevidence_p2_empty_output_audit.json');coverage=read(public/'medevidence_p2_coverage_audit.json')
    status={k:{key:(re.sub(r'/(?:Users|data|remote-home)/[^\s]+','[private path]',value) if isinstance(value,str) else value)
               for key,value in v.items() if key!='pid'} for k,v in status.items()}
    token_results={};d1={}
    for name in ('M1','W','M2'):
        p=root/('D1_'+name)
        if not (p/'summary.json').exists():continue
        summary=read(p/'summary.json')
        if summary['status']!='completed':d1[name]={'status':summary['status']};continue
        records=rows(p/'predictions.jsonl');assert len(records)==64
        d1[name]={'status':'completed','sets':{s:diagnostic_metrics([r for r in records if r['set']==s]) for s in ('Fit32','Ref32')}}
        token_results[name]={s:token_summary([r for r in records if r['set']==s]) for s in ('Fit32','Ref32')}
    tokens={'status':'completed' if 'M1' in token_results else 'not_authorized_or_not_completed','models':token_results,
            'CPU_token_contract':{'GT_single_box_including_EOS_tokens':18,'negative_empty_including_EOS_tokens':2,
                'positive_first_token_patient_loss_weight':1/18,'negative_first_token_patient_loss_weight':1/2,
                'GT_vs_empty_common_token_prefix':0,'first_divergence_at_generated_token_position':0,
                'encoding':'Actual processor tokenizer verified on all64 selected patients; GT initial [[ and [] are different tokens; no character/token identity assumption'},
            'limitation':'Token teacher diagnostics condition on a prescribed prefix; no coordinate support threshold was searched, and no new loss weighting was introduced'}
    save(public/'medevidence_p2_token_diagnostic.json',tokens)
    lines=['# MedEvidence pilot-2 token与冻结模型诊断','',f"状态：{tokens['status']}。",'',
           'CPU已核对实际编码：单框GT含EOS为18 token，[]含EOS为2 token；GT与空列表没有共同答案token，首个分歧在位置0。同一患者平均NLL中首token权重分别1/18和1/2；本轮未更改权重。这只是编码事实，不能直接认定空列表偏置原因。']
    for model,sets in token_results.items():
        lines+=['',f'## {model} 实际冻结测量','','| 集合 | 阳性人数 | 首分歧空分支分数更高 | 实际选择空分支token | 自由生成[] | GT坐标条件NLL均值 |','|---|---:|---:|---:|---:|---:|']
        for name,v in sets.items():
            x=v['positive'];div=x['first_divergence'];coords=x['token_parts']['digits']['patient_mean_NLL']['mean']
            lines.append(f"| {name} | {div['patients']} | {div['empty_token_higher_teacher_logprob']} | {div['free_next_matches_empty']} | {x['empty_free_generations']} | {coords} |")
    lines+=['','逐例完整token IDs、解码、EOS、截断、前四位置top支持、实际生成设置、GT/[]含及不含EOS的sum/mean/逐token分数和字符跨度保持私有。相邻JSON给出聚合分布和数值误差。共享前缀是否进入实际生成路径已逐例记录；本次共享前缀长度0为共同问题后的直接首token选择。GT分支后的数字支持不等于自由生成已经会定位；GT/[]二选字符串计分不是所有非空列表总概率。']
    (public/'medevidence_p2_token_diagnostic.md').write_text('\n'.join(lines)+'\n')
    probe={'status':'not_authorized_or_not_completed','curves':{},'development':None};p=root/'L1_loc_probe'
    if (p/'summary.json').exists():
        summary=read(p/'summary.json');probe.update({k:v for k,v in summary.items() if k not in ('actual_patient_exposures','curves')})
        actual=summary.get('actual_patient_exposures',{});probe['patient_exposure_histogram']=dict(Counter(actual.values()))
        probe['unexposed_Fit_patients']=32-len(actual)
        for step in (0,32,64,128,256):
            path=p/f'step_{step:04d}'
            if not (path/'predictions.jsonl').exists():continue
            records=rows(path/'predictions.jsonl')
            if len(records)!=64:probe['curves'][str(step)]={'status':'incomplete_evaluation','patients':len(records)};continue
            probe['curves'][str(step)]={s:diagnostic_metrics([r for r in records if r['set']==s]) for s in ('Fit32','Ref32')}
            probe['curves'][str(step)]['checkpoint_identity']=read(path/'identity.json')
        if (p/'train.jsonl').exists():
            train=rows(p/'train.jsonl');probe['training']={k:distribution([v[k] for v in train]) for k in ('NLL','positive_NLL','negative_NLL','gradient_norm_pre_clip','update_seconds')}
            probe['token_part_NLL_over_training']={label:{category:distribution([r['token_parts'][category]['mean_NLL'] for v in train for r in v['patient_records'] if r['positive']==(label=='positive') and r['token_parts'][category]['mean_NLL'] is not None]) for category in ('first_divergence','digits','structure','mixed','EOS')} for label in ('positive','negative')}
        if (p/'gradient_check.json').exists():probe['no_update_gradient_check']=read(p/'gradient_check.json')
        if (p/'initial_identity.json').exists():probe['initial_reset_and_restore']=read(p/'initial_identity.json')
    dev=root/'eval_L1'
    if (dev/'summary.json').exists():
        ds=read(dev/'summary.json');probe['development']={'status':ds['status'],'patients':ds.get('patients')}
        if ds['status']=='completed':
            generated=sorted(rows(dev/'predictions.jsonl'),key=lambda r:r['case_id'])
            replay=read(root/'D0_patient_replay.json')['M1'];cached=sorted(replay,key=lambda r:r['case_id'])
            assert len(generated)==256
            probe['development'].update(metrics=diagnostic_metrics(generated),cached_M1_metrics=diagnostic_metrics(cached),paired_bootstrap=comparison(cached,generated))
    ledger=read(root/'gpu_ledger.json') if (root/'gpu_ledger.json').exists() else {'gpu_hours':0.,'records':[],'status':'not_authorized'}
    probe['GPU_ledger_aggregate']={'status':ledger['status'],'charged_gpu_hours':ledger['gpu_hours'],
        'stages':[{'stage':v['stage'],'status':v['status'],'charged_seconds':v['charged_seconds'],'exit_code':v['exit_code']} for v in ledger['records']]}
    save(public/'medevidence_p2_loc_probe.json',probe)
    lines=['# MedEvidence pilot-2 唯一定位专训探针','',f"状态：{probe['status']}；累计GPU计费{ledger['gpu_hours']:.6f}小时。",'',
           'L1固定从M1最终权重开始，AdamW与seed17状态重置，仅L完整目标含EOS的患者平均NLL。每步2阳性2阴性、微批量1、累积4，固定256更新；没有optimizer更新式smoke、dep/crop/control/A辅助训练或最佳checkpoint选择。', '',
           '| checkpoint | 集合 | 阳性[] | 阳性非空 | 单框IoU≥.5 | 严格单框成功 | 阳性无条件IoU均值 | 阴性非空 | 阴性非法 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for step,values in probe['curves'].items():
        if values.get('status')=='incomplete_evaluation':continue
        for name in ('Fit32','Ref32'):
            v=values[name];lines.append(f"| {step} | {name} | {v['positive']['valid_empty']}/16 | {v['positive']['valid_nonempty']}/16 | {v['single_IoU_ge_05_count']}/16 | {v['single_strict_success_count']}/16 | {v['single_IoU_unconditional']['mean']} | {v['negative']['valid_nonempty']}/16 | {v['negative']['invalid']}/16 |")
    lines+=['','Fit32是本轮训练内可学习性；Ref32是旧训练池中的本轮不更新参照，不是独立未见测试。未完成项NA。分类仅step0/256检查；实际eval/no_grad及RNG/训练模式恢复受源码检查。所有invalid和空输出保留无条件分母，条件非空IoU仅补充。','',
            '最终开发评价状态、原始0.5分类混淆矩阵、阴性误报、多框层及与已缓存M1的2000次患者配对区间见相邻JSON。预算允许时只生成一次固定step256开发输出，不据结果追加步数。训练loss/跨度诊断、实际暴露、梯度范数和checkpoint身份也在JSON；私有路径和逐例原始记录不公开。']
    (public/'medevidence_p2_loc_probe.md').write_text('\n'.join(lines)+'\n')
    m1=d0['M1'];m2=d0['M2']
    count=lambda v:next(r['patients'] for r in v['classification_loc_cross_table'] if r['true'] and r['class_yes'] and r['loc_state']=='valid_empty')
    decisions=[f"1. 历史重放：M1/M2阳性空列表分别{m1['positive']['valid_empty']}/37、{m2['positive']['valid_empty']}/37。",
               f"2. 历史重放：阳性分类yes但定位[]分别{count(m1)}、{count(m2)}人。",
               '3. 首分歧偏向空输出：'+('已进行新冻结测量，按token报告中的逐集合计数判断；不能由CPU编码事实代替模型偏好。' if 'M1' in token_results else 'NA，未获GPU授权或尚未完成D1；CPU只确认分歧在位置0。'),
               '4. 教师强制与自由生成脱节：'+('条件GT数字NLL与自由生成计数已分别记录；只支持条件化路径诊断，不把较低NLL当作可用定位能力。' if 'M1' in token_results else 'NA，尚无本轮模型测量。')]
    if 'M1' in token_results:
        detail=[]
        for name,value in token_results['M1'].items():
            x=value['positive'];div=x['first_divergence']
            detail.append(f"{name}：16名阳性中首分歧空分支支持更高{div['empty_token_higher_teacher_logprob']}名、实际选空token {div['free_next_matches_empty']}名，自由[] {x['empty_free_generations']}名；GT条件数字NLL均值{x['token_parts']['digits']['patient_mean_NLL']['mean']:.4f}")
        decisions[2]='3. 新冻结M1测量：'+'；'.join(detail)+'。'
        decisions[3]='4. GT教师强制与自由生成：上述条件数字NLL和空输出计数分别测得；是否存在脱节据共同问题后首分歧及完整自由路径解释，不能把条件数字得分当实际定位成功。'
    final=probe['curves'].get('256',{})
    if 'Fit32' in final:
        fit,ref=final['Fit32'],final['Ref32'];decisions += [f"5. Fit32最终：阳性非空{fit['positive']['valid_nonempty']}/16，严格单框成功{fit['single_strict_success_count']}/16，平均IoU{fit['single_IoU_unconditional']['mean']:.4f}；阴性非空{fit['negative']['valid_nonempty']}/16。",
          f"6. Ref32最终：严格单框成功{ref['single_strict_success_count']}/16，阴性非空{ref['negative']['valid_nonempty']}/16。开发状态{probe['development']['status'] if probe['development'] else 'NA'}；分类代价及多框结果按loc_probe JSON原始0.5结果报告。"]
        initial_fit=probe['curves']['0']['Fit32']
        if fit['single_strict_success_count']>initial_fit['single_strict_success_count'] and fit['single_strict_success_count']>ref['single_strict_success_count']:
            recommendation='优先扩大定位监督覆盖，另行预指定独立患者评价；本轮训练内严格成功有所增加而本轮参照较少，属于探索性拟合差距。'
        elif fit['positive']['valid_nonempty']>initial_fit['positive']['valid_nonempty'] and fit['single_strict_success_count']<=initial_fit['single_strict_success_count']:
            recommendation='优先诊断坐标生成接口；非空输出增加而训练内严格定位未增加，不增加dep。'
        else:
            recommendation='优先诊断定位的存在性/终止决策与坐标生成接口；先结合首分歧及条件坐标读数查失败路径，不自动改loss或更换模型。'
    else:
        decisions += ['5. Fit32可学习性：NA，L1未授权或未完成；不能由CPU评分正控推出模型学会定位。','6. Ref32/开发效果及分类代价：NA，未进行本轮训练及最终评价。']
        recommendation='当前历史证据只能把定位输出接口作为诊断优先项；最终单一改进环节须等固定L1结果，不能提前把待检验假设写成原因。'
    if probe['development'] and probe['development']['status']=='completed':
        devmetrics=probe['development']['metrics'];oldmetrics=probe['development']['cached_M1_metrics'];cl=devmetrics['classification_raw_0_5'];oldcl=oldmetrics['classification_raw_0_5']
        decisions[5]+=f" 开发阳性空列表{devmetrics['positive']['valid_empty']}/37，区域匹配{int(devmetrics['positive_region_recall']*51+.5)}/51，单框严格{devmetrics['single_strict_success_count']}/25，多框完整覆盖{devmetrics['multi_all_covered']}/12，阴性非空{devmetrics['negative']['valid_nonempty']}/219；分类TP/FN/TN/FP={cl['TP']}/{cl['FN']}/{cl['TN']}/{cl['FP']}，AP变化{cl['ap']-oldcl['ap']:+.6f}。患者配对区间见JSON。"
    decisions += [f"7. 覆盖缩减：训练149→71单框→69面积/边长合格→首32提议→3接受；开发37→25单框→25提议→3接受。训练仍有{coverage['train']['eligible_not_proposed_or_reviewed']}个合格几何未提议/未做语义审核。语义原因和unknown保持旧记录，不重选旧3+3。",
                  '8. 本轮总研发优先项：'+recommendation+' E0的未来证据任务只推荐区域—发现对应，适用边界见coverage审计；本轮不执行该扩展。',
                  f"9. 阶段状态见下方；当前累计{ledger['gpu_hours']:.6f} GPU小时，无授权阶段保持not_authorized/NA，不作为科学无效。"]
    output={'status':probe['status'],'answers':decisions,'stage_status':status,'GPU_hours':ledger['gpu_hours'],'recommendation':recommendation,
            'limits':['One localization-only probe, one seed; not a mechanism-isolated matched comparison','Ref32 is from historical training pool; no independent unseen test','Development set reused; no clinical validation','No automatic second training, evidence expansion, external patient upload, new data or RL']}
    save(public/'medevidence_p2_decision.json',output)
    lines=['# MedEvidence pilot-2 判断与状态','','历史重放、编码检查、新冻结模型测量、训练内拟合和开发结果分别报告；尚未执行或不足支持的部分记NA。','']
    lines += decisions
    lines += ['','阶段状态：','', '```json',json.dumps(status,ensure_ascii=False,indent=2),'```','',
              '此建议不构成下一轮授权，不自动启动存在性头、首token加权、不同坐标编码、视觉解冻、第二个训练方案、数据扩展或RL。']
    (public/'medevidence_p2_decision.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps({'report_status':probe['status'],'GPU_hours':ledger['gpu_hours'],'D1_completed_models':list(token_results)}))


if __name__=='__main__':main()
