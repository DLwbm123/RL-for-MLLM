"""Public aggregates, fixed matched comparisons and bounded research conclusions."""
from collections import Counter
import json
import os
from pathlib import Path
import re
import numpy as np
from src.v2 import save,binary_stats,crossfit
from src.medevidence_p2 import summarize,distribution
from src.medevidence_p3 import metrics,paired_intervals,local_metrics


def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]


def full_metrics(records,folds):
    out=summarize(records);out.update(metrics(records,folds));y=[r['y'] for r in records];p=[r['p'] for r in records]
    prediction,thresholds=crossfit(y,p,[folds[r['image_id']] for r in records]);out['classification_raw_0_5']=binary_stats(y,p)
    out['classification_crossfit']=binary_stats(y,p,prediction) if prediction is not None else None;out['fold_thresholds']=thresholds
    out['exact_p_0_5_ties']=sum(r['p']==.5 for r in records);out['exact_candidate_score_ties']=sum(r['score_no']==r['score_yes'] for r in records)
    out['parse_errors']=dict(Counter(r['parse_error'] for r in records if r['parse_error']))
    out['predicted_regions_known_all_patients']=sum(r['pred'] or 0 for r in records);out['unknown_prediction_count_patients']=sum(r['pred'] is None for r in records)
    return out


def local_intervals(models,repeats=2000,seed=42):
    vectors={};clusters=None;keys=[]
    for scale in (2.,2.5):
        keys += [str(scale)+'/'+k for k in ('positive_detected','negative_false_positive','pair_correct','within_patient_crop_correctness','all_views_correct','positive_support','negative_support')]
    for name,records in models.items():
        ids=sorted({r['cluster'] for r in records})
        if clusters is None:clusters=ids
        assert ids==clusters
        values=[]
        for k in ids:
            row=[]
            for scale in (2.,2.5):
                pos=[r['p'] for r in records if r['cluster']==k and r['scale']==scale and r['y']==1]
                neg=[r['p'] for r in records if r['cluster']==k and r['scale']==scale and r['y']==0]
                assert len(pos)==len(neg)>0
                a,b=float(np.mean(pos)),float(np.mean(neg))
                row += [a>=.5,b>=.5,a>=.5 and b<.5,np.mean([p>=.5 for p in pos]+[p<.5 for p in neg]),all(p>=.5 for p in pos) and all(p<.5 for p in neg),a,b]
            values.append(row)
        vectors[name]=np.asarray(values,float)
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(clusters),(repeats,len(clusters)));out={}
    for a,b in (('COV','COV-A'),('M1','COV'),('M1','COV-A')):
        if a not in vectors or b not in vectors:continue
        delta=vectors[b]-vectors[a];boot=delta[idx].mean(axis=1)
        out[b+' minus '+a]={key:{'difference':float(delta[:,j].mean()),'ci95':np.quantile(boot[:,j],[.025,.975]).tolist(),'valid_replicates':repeats} for j,key in enumerate(keys)}
    return {'matched_clusters':len(clusters),'independent_patients':2*len(clusters),'paired_donor_and_all_crops_and_scales_retained':True,'comparisons':out}


def main():
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';public=base/'code/reports';cfg=read(base/'code/configs/medevidence_p3.json')
    audit=read(public/'medevidence_p3_data_schedule_audit.json');status=read(root/'stage_status.json');folds=read(root/'protocol/folds.json')
    status={k:{key:re.sub(r'/(?:data|Users|remote-home)/[^\s]+','[private path]',v) if isinstance(v,str) else v for key,v in row.items() if key!='pid'} for k,row in status.items()}
    save(public/'medevidence_p3_status.json',status)
    stages={};initial={};training={};records={name:read(root/(name+'_historical_predictions.json')) for name in ('M1','L1')}
    for name in ('COV','COV-A'):
        p=root/name
        if (p/'summary.json').exists():
            summary=read(p/'summary.json');stages[name]={k:v for k,v in summary.items() if k!='actual_patient_exposures'}
            stages[name]['patient_exposure_histogram']=dict(Counter(summary.get('actual_patient_exposures',{}).values()))
        if (p/'initial_identity.json').exists():initial[name]=read(p/'initial_identity.json')
        if (p/'train.jsonl').exists():
            rows=lines(p/'train.jsonl');training[name]={'updates':len(rows),'per_step_statistics':{k:distribution([r[k] for r in rows]) for k in ('L_NLL','A_NLL','L_positive_NLL','L_negative_NLL','A_positive_NLL','A_negative_NLL','L_EOS_NLL','A_EOS_NLL','gradient_norm_pre_clip','update_seconds')},
                'forward_calls':sum(r['forward_calls'] for r in rows),'vision_calls':sum(r['vision_calls'] for r in rows),'backward_calls':sum(r['backward_calls'] for r in rows),
                'clip_count':sum(r['gradient_clipped'] for r in rows),'peak_memory_gib':max((r['peak_memory_gib'] for r in rows),default=None),
                'curve_block64':[{key:float(np.mean([r[key] for r in rows[start:start+64]])) for key in ('L_NLL','A_NLL','L_positive_NLL','L_negative_NLL','A_positive_NLL','A_negative_NLL')} for start in range(0,len(rows),64)]}
        p=root/('eval_'+name)
        if (p/'summary.json').exists() and read(p/'summary.json')['status']=='completed':
            rs=lines(p/'predictions.jsonl');assert len(rs)==256;records[name]=rs
    equal=initial.get('COV')==initial.get('COV-A') if len(initial)==2 else None
    if len(initial)==2 and not equal:raise ValueError('Actual matched branch initialization differs')
    preflight=read(root/'preflight/summary.json') if (root/'preflight/summary.json').exists() else {'status':'pending'}
    engineering=read(public/'medevidence_p3_engineering.json');engineering.update(GPU_no_update_preflight=preflight,branch_initialization_exact_equal=equal,initial_identities=initial)
    save(public/'medevidence_p3_engineering.json',engineering)
    save(public/'medevidence_p3_training.json',{'stages':stages,'training':training,'initialization_exact_equal':equal,'sole_objective_difference':{'COV_A_coefficient':0,'COV-A_A_coefficient':1},'shared_L_schedule':True})
    absolute={name:full_metrics(rs,folds) for name,rs in records.items()};comparisons={}
    for a,b in cfg['bootstrap_pairs']:
        if a in records and b in records:comparisons[b+' minus '+a]=paired_intervals(records[a],records[b],folds)
    alarms={name:{'AP_drop_over_0_02':v['A_raw_ap']<absolute['M1']['A_raw_ap']-.02,
                  'negative_box_increase_over_5pp':v['negative_nonempty_rate']>absolute['M1']['negative_nonempty_rate']+.05} for name,v in absolute.items() if name in ('COV','COV-A')}
    evaluation={'status':'completed' if all(k in absolute for k in ('COV','COV-A')) else 'pending_or_partial','absolute':absolute,'paired_comparisons':comparisons,'research_alarms_only':alarms,
        'CI_method':'2000 patient-paired bootstrap, seed42; repeated patients retain original fold; specificity90% thresholds refit each replicate; no training-seed uncertainty',
        'precision_scope':'Matched GT regions divided by known valid predicted regions across positive and negative patients; invalid predictions retain unknown count and failure denominator',
        'historical_COV_L1_limit':'Historical probe comparison changes patient coverage and repetition allocation of both classes; COV additionally computes A no-grad diagnostic (no A backward); no synchronous multiseed replication'}
    save(public/'medevidence_p3_evaluation.json',evaluation)
    local_records={};local_results={}
    for name in ('M1','COV','COV-A'):
        p=root/('local_'+name)
        if (p/'summary.json').exists() and read(p/'summary.json')['status']=='completed':
            local_records[name]=lines(p/'predictions.jsonl');local_results[name]=local_metrics(local_records[name])
            for scale,value in local_results[name].items():
                selected=[r for r in local_records[name] if r['scale']==float(scale)]
                value['candidate_score_distributions']={k:distribution([r[k] for r in selected]) for k in ('score_no','score_yes','p')}
                value['candidate_length_histogram']=dict(Counter(str(r['candidate_lengths']) for r in selected))
    local={'status':'completed' if len(local_results)==3 else 'pending_or_partial','models':local_results,
           'cluster_bootstrap':local_intervals(local_records) if local_records else None,'qualification':'legacy annotation-guided geometry and AI context exclusions; not new expert causal review',
           'limits':['External annotated crop, no self-selected localization','Cross-patient matched geometry is not a medical counterfactual','Dataset target-negative is not completely normal lung','No evidence-dependence or clinical validation']}
    save(public/'medevidence_p3_local_evidence.json',local)
    ledger=read(root/'gpu_ledger.json') if (root/'gpu_ledger.json').exists() else {'status':'ready_to_launch','gpu_hours':0,'records':[]}
    budget={'status':ledger['status'],'gpu_hours':ledger['gpu_hours'],'gpu_limit_hours':3,'max_concurrent_gpus':3,'records':[{k:v for k,v in r.items() if k in ('stage','status','charged_seconds','exit_code')} for r in ledger['records']]}
    save(public/'medevidence_p3_budget.json',budget)
    answers=[f"1. 冻结{audit['positive_patients']}阳性+{audit['negative_patients']}阴性=327患者，每分支1024次L暴露、正负各512；56名阳性7次、15名阳性8次、256名阴性各2次。旧Ref重叠{audit['old_Ref_overlap']}，不再称本轮不更新参照。",
             f"2. 两分支原M1/optimizer/RNG/lr/排程位置/训练模块初始身份精确相等：{equal if equal is not None else 'NA，尚未完成GPU初始检查'}。绝不使用L1权重或优化器。",
             f"3. 实际步数：{ {k:v.get('steps') for k,v in stages.items()} }；唯一训练目标差异为A系数0/1，同图同L排程，不将总目标除2。额外A backward与实际剪裁情况分别报告。"]
    def compare_text(a,b):
        if a not in absolute or b not in absolute:return 'NA，对应固定最终评价尚未完成。'
        x,y=absolute[a],absolute[b]
        return f"{a}→{b}：单框严格{x['single_strict_success_count']}→{y['single_strict_success_count']}/25，阴性出框{x['negative']['valid_nonempty']}→{y['negative']['valid_nonempty']}/219，区域匹配{int(x['region_recall']*51+.5)}→{int(y['region_recall']*51+.5)}/51，AP{x['A_raw_ap']:.4f}→{y['A_raw_ap']:.4f}；配对区间见evaluation JSON。"
    answers += ['4. 覆盖比较：'+compare_text('L1','COV'),'5. 答案能力保留：'+compare_text('COV','COV-A'),'6. 原M1综合参照：'+compare_text('M1','COV-A')]
    answers += [f"7. 局部读取诊断状态{local['status']}，已完成模型{list(local_results)}；{audit['local_view_pairs']}匹配簇、{audit['local_independent_patients']}患者、{audit['local_crops_all_scales']}crop，两尺度和同一供体成簇。局部支持与自由定位分别展示，不把局部识别当模型自主选框。",
                '8. 本轮只检验grounding、答案监督保留与外部给定局部视图的可读性；无dep、因果证据忠实性验证、临床验证或RL训练。',
                '9. PadChest-GR：pending_user_submission；官方入口/条款已核对，本地申请草稿和schema/患者级划分方案已准备。访问权限未验证，实际新数据记录读取0、申请提交0、下载0、模型计分0；姓名/邮箱/机构等待用户确认。',
                f"10. 阶段状态{status}；累计{budget['gpu_hours']:.6f}/3 GPU小时。未完成项NA，未审核候选和独立机制保持未知；无自动下一轮。公开push需要本轮单独授权。"]
    recommendation='等待两组完整匹配结果，不根据未完成分支选择胜者。'
    if all(k in absolute for k in ('COV','COV-A')):
        m,a=absolute['M1'],absolute['COV-A'];c=absolute['COV']
        if a['single_strict_success_count']>m['single_strict_success_count'] and a['negative_nonempty_rate']<c['negative_nonempty_rate'] and a['A_raw_ap']>c['A_raw_ap']:
            recommendation='COV-A作为后续重复验证的grounded候选；结合警戒线和配对区间，不能自动称为忠实证据增强或独立泛化成功。'
        elif a['single_strict_success_count']<=m['single_strict_success_count'] and c['single_strict_success_count']<=m['single_strict_success_count']:
            recommendation='本轮两组未增加单框严格成功；不继续追加同一覆盖/步数配方，下一步单独预指定数据或任务接口研究。'
        else:recommendation='指标存在取舍，依据绝对计数、阴性误报、AP警戒和配对区间保留不确定性，不强制选择胜者。'
    save(public/'medevidence_p3_decision.json',{'status':ledger['status'],'answers':answers,'recommendation':recommendation,'alarms':alarms,'stage_status':status,'delivery':'pending_current_public_release_authorization'})
    (public/'medevidence_p3_decision.md').write_text('# MedEvidence pilot-3 结果与决策\n\n'+'\n\n'.join(answers)+'\n\n'+recommendation+'\n\n所有开发结果属于多轮使用过的开发集；单seed患者区间不包含训练随机性。\n')
    (public/'medevidence_p3_data_schedule_audit.md').write_text('# MedEvidence pilot-3 患者与共享排程\n\n```json\n'+json.dumps(audit,indent=2,ensure_ascii=False)+'\n```\n\n选择不使用模型分数，旧Fit全部32名重叠；旧Ref重叠者进入训练，不再作本轮不更新参照。原78名多框训练阳性未加入。局部34患者资格来自旧crop规则，保留3名旧排除病例，无新审核、补选或训练。\n')
    (public/'medevidence_p3_engineering.md').write_text('# MedEvidence pilot-3 工程检查\n\nCPU身份/GT/排程/封存检查、GPU无更新梯度与恢复、初始状态比较见相邻JSON。\n\nGPU预检状态：'+str(preflight['status'])+'；两分支实际初始状态相等：'+str(equal)+'。浮点容差预定2个bf16 epsilon=.015625；源码和adapter/RNG身份使用精确比较。无optimizer更新式smoke。\n')
    (public/'medevidence_p3_training.md').write_text('# MedEvidence pilot-3 实际训练\n\n每组固定256步；checkpoint0/64/128/256，最终固定256。COV仅L反传，同时同图A仅no_grad诊断；COV-A为L+1*A，患者平均，不除2。训练曲线和实际forward/vision/backward/剪裁计数见JSON。\n\n```json\n'+json.dumps(stages,indent=2,ensure_ascii=False)+'\n```\n')
    text='# MedEvidence pilot-3 完整开发评价\n\n| 模型 | 单框严格 | 阴性错误框 | 阳性空列表 | 区域召回 | 区域precision | 多框全覆盖 | AP | 分类FP |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for name in ('M1','L1','COV','COV-A'):
        if name not in absolute:continue
        v=absolute[name];precision=v['region_precision_known_predictions']
        text+=f"| {name} | {v['single_strict_success_count']}/25 | {v['negative']['valid_nonempty']}/219 | {v['positive']['valid_empty']}/37 | {int(v['region_recall']*51+.5)}/51 | {precision} | {v['multi_all_covered']}/12 | {v['A_raw_ap']:.4f} | {v['classification_raw_0_5']['FP']}/219 |\n"
    text+='\n无效JSON/坐标/截断、阴性严格空输出、额外/重复框、0.5阈值混淆矩阵与ties、A×L交叉表及患者配对区间见JSON。始终no+[]为219/256，仅多数类参照，无视觉定位能力。交叉阈值每次bootstrap重新拟合训练四折，重复患者保留原fold。COV−L1是历史覆盖和重复分配比较，COV新增同图A no_grad诊断计算，不冒称只有覆盖一个工程变化。\n'
    (public/'medevidence_p3_evaluation.md').write_text(text)
    text='# MedEvidence pilot-3 局部证据可读性\n\n状态：'+local['status']+'。\n\n| 模型 | 尺度 | 匹配簇 | 阳性局部检出 | 阴性局部误报 | 配对均正确 | 全部crop严格正确 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,scales in local_results.items():
        for scale,v in scales.items():text+=f"| {name} | {scale} | {v['pairs']} | {v['positive_detected']} | {v['negative_false_positive']} | {v['pair_correct']} | {v['all_views_correct']} |\n"
    text+='\n34个独立正负匹配簇，两尺度共享供体；多框先患者内平均，原始0.5判定。候选支持不是校准疾病概率。跨患者几何匹配不是医学反事实，RSNA目标阴性不代表完全正常肺；外部给定框局部识别不证明自主定位、答案因果依赖或临床可靠性。匹配簇bootstrap保留全部框、模型和尺度。\n'
    (public/'medevidence_p3_local_evidence.md').write_text(text)
    print(json.dumps({'report_status':ledger['status'],'GPU_hours':budget['gpu_hours'],'new_completed_models':[k for k in records if k not in ('M1','L1')]}))


if __name__=='__main__':main()
