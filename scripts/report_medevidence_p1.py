"""Aggregate private predictions into public patient-level exploratory results."""
import json
import os
from pathlib import Path
import numpy as np
from src.v2 import binary_stats, crossfit, summarize_scores, paired_bootstrap


def ratio(a,b):return float(a/b) if b else None


def loc_metrics(records,pred,w=None):
    w=np.ones(len(records)) if w is None else np.asarray(w);positive=np.array([r['y']==1 for r in records]);negative=~positive
    valid=np.array([r['parse_error'] is None for r in records]);gt=np.array([r['gt'] for r in records]);matches=np.array([r['matches'] for r in records])
    counts=np.array([r['pred'] or 0 for r in records]);strict=np.array([r['strict'] for r in records]);covered=np.array([r['all_covered'] for r in records])
    class_correct=np.asarray(pred)==positive
    single=gt==1;multi=gt>1
    joint_strict=class_correct&strict
    joint_coverage=class_correct&np.where(positive,covered,strict)
    known=positive&valid
    out={'strict_joint_all_patients':ratio((w*joint_strict).sum(),w.sum()),
         'coverage_joint_all_patients':ratio((w*joint_coverage).sum(),w.sum()),
         'single_joint':ratio((w*single*joint_strict).sum(),(w*single).sum()),
         'multi_coverage_joint':ratio((w*multi*joint_coverage).sum(),(w*multi).sum()),
         'multi_strict_joint':ratio((w*multi*joint_strict).sum(),(w*multi).sum()),
         'single_iou_mean':ratio(sum(weight*(r['single_iou'] or 0.) for weight,r in zip(w,records) if r['gt']==1),(w*single).sum()),
         'single_iou_ge_05':ratio(sum(weight*(r['single_iou']>=.5) for weight,r in zip(w,records) if r['gt']==1),(w*single).sum()),
         'multi_region_recall':ratio((w*multi*matches).sum(),(w*multi*gt).sum()),
         'multi_region_precision_known_outputs':ratio((w*multi*valid*matches).sum(),(w*multi*valid*counts).sum()),
         'multi_all_target_coverage':ratio((w*multi*covered).sum(),(w*multi).sum()),
         'positive_region_recall':ratio((w*positive*matches).sum(),(w*positive*gt).sum()),
         'positive_region_precision_known_outputs':ratio((w*known*matches).sum(),(w*known*counts).sum()),
         'negative_nonempty_known_rate_all_negative_denominator':ratio((w*negative*valid*(counts>0)).sum(),(w*negative).sum()),
         'invalid_output_rate':ratio((w*~valid).sum(),w.sum())}
    return out


def evidence(records,kind,positive=True):
    subset=[r for r in records if 'views' in r and r['views']['positive']==positive]
    fields=['keep','hide','D_E','D_N','S']
    out={'n':len(subset),'u_original':float(np.mean([r['p'] for r in subset])) if subset else None}
    out.update({key:float(np.mean([r['views'][kind][key] for r in subset])) if subset else None for key in fields})
    out['S_positive_rate']=float(np.mean([r['views'][kind]['S']>0 for r in subset])) if subset else None
    if not positive:
        out['view_false_positive_rate']=float(np.mean([(r['views'][kind][key]>=.5) for r in subset for key in ('keep','hide')])) if subset else None
    return out


def main():
    base=Path(os.environ['PILOT_ROOT']);root=base/'outputs';code=base/'code';public=code/'reports';public.mkdir(exist_ok=True)
    models={}
    for name in ('M0','M1','M2'):
        summary=root/('eval_'+name)/'summary.json'
        if summary.exists() and json.loads(summary.read_text())['status']=='completed':
            rows=[json.loads(line) for line in (summary.parent/'predictions.jsonl').read_text().splitlines()]
            if len(rows)!=256 or len({r['case_id'] for r in rows})!=256:raise ValueError('Incomplete/duplicate evaluation population')
            models[name]=sorted(rows,key=lambda r:r['case_id'])
    if not all(k in models for k in ('M1','M2')):raise ValueError('Matched final evaluations required')
    folds=json.loads((root/'protocol/folds.json').read_text());cfg=json.loads((code/'configs/medevidence_p1.json').read_text())
    ids=[r['image_id'] for r in models['M1']];y=np.array([r['y'] for r in models['M1']]);fold=np.array([folds[k] for k in ids]);scores={}
    for m,rows in models.items():
        if [r['image_id'] for r in rows]!=ids:raise ValueError('Unpaired patients')
        scores[m]=np.array([r['p'] for r in rows])
    results={};bootstrap={};rng=np.random.default_rng(42)
    for name,rows in models.items():
        cf,thresholds=crossfit(y,scores[name],fold)
        single=[r['single_iou'] for r in rows if r['gt']==1]
        results[name]={'classification':summarize_scores(rows,folds),
                       'localization_raw_0_5':loc_metrics(rows,scores[name]>=.5),
                       'localization_crossfit':loc_metrics(rows,cf),
                       'single_IoU_quantiles_0_25_50_75_100':np.quantile(single,[0,.25,.5,.75,1]).tolist() if single else None,
                       'denominators':{'all_patients':len(rows),'positive':int(y.sum()),'negative':int((y==0).sum()),
                           'single_box_patients':sum(r['gt']==1 for r in rows),'multi_box_patients':sum(r['gt']>1 for r in rows),
                           'positive_GT_regions':sum(r['gt'] for r in rows),'valid_predicted_boxes_positive':sum((r['pred'] or 0) for r in rows if r['y']==1 and r['parse_error'] is None),
                           'positive_unknown_box_count_patients':sum(r['parse_error'] is not None and r['y']==1 for r in rows),
                           'negative_unknown_box_count_patients':sum(r['parse_error'] is not None and r['y']==0 for r in rows)},
                       'errors':{'parse_reasons':{reason:sum(r['parse_error']==reason for r in rows) for reason in ('invalid_json','invalid_list','invalid_coordinates','truncated')},
                                 'truncated':sum(r['truncated'] for r in rows),'missing_GT_regions':sum(r['gt']-r['matches'] for r in rows),
                                 'extra_known_boxes':sum(r['pred']-r['matches'] for r in rows if r['pred'] is not None),
                                 'unknown_box_counts':'Retained failures; precision over known output counts is conditional, full precision has no identifiable denominator'},
                       'evidence':{kind:evidence(rows,kind) for kind in ('gray','blur','blank')},
                       'negative_sham':{kind:evidence(rows,kind,False) for kind in ('gray','blur','blank')},
                       'crop':{'positive_patients':sum('crops' in r for r in rows),
                               'patient_mean_u_2':float(np.mean([np.mean([b['2.0'] for b in r['crops']]) for r in rows if 'crops' in r])),
                               'patient_mean_u_2_5':float(np.mean([np.mean([b['2.5'] for b in r['crops']]) for r in rows if 'crops' in r])),
                               'negative_u_2':float(np.mean([r['negative_crop_p_2.0'] for r in rows if 'negative_crop_p_2.0' in r])),
                               'negative_u_2_5':float(np.mean([r['negative_crop_p_2.5'] for r in rows if 'negative_crop_p_2.5' in r]))}}
    # Joint thresholds are fitted again with duplicate-patient multiplicities kept inside original folds.
    samples={}
    for _ in range(cfg['bootstrap_repeats']):
        w=np.bincount(rng.integers(0,256,256),minlength=256);replicate={}
        for name,rows in models.items():
            cf,_=crossfit(y,scores[name],fold,w)
            for mode,pred in [('raw',scores[name]>=.5),('crossfit',cf)]:
                if pred is None:continue
                values=loc_metrics(rows,pred,w)
                for key,val in values.items():
                    if val is not None:samples.setdefault((name,mode,key),[]).append(val);replicate[(name,mode,key)]=val
        for mode in ('raw','crossfit'):
            for key in results['M1']['localization_raw_0_5']:
                a=replicate.get(('M1',mode,key));b=replicate.get(('M2',mode,key))
                if a is not None and b is not None:samples.setdefault(('M2 minus M1',mode,key),[]).append(b-a)
    for key,vals in samples.items():bootstrap[' / '.join(key)]={'ci95':np.quantile(vals,[.025,.975]).tolist(),'valid_replicates':len(vals),'requested_replicates':2000}
    view_bootstrap={};common_correct={}
    for positive in (True,False):
        subset=[i for i,r in enumerate(models['M1']) if 'views' in r and r['views']['positive']==positive]
        if any(('views' not in models['M2'][i]) or models['M2'][i]['views']['positive']!=positive for i in subset):raise ValueError('Unpaired evidence cohort')
        for kind in ('gray','blur','blank'):
            keys=('S','D_E','D_N','keep','hide')
            for key in keys:
                values={m:np.array([models[m][i]['views'][kind][key] for i in subset]) for m in ('M1','M2')}
                delta=values['M2']-values['M1'];draws=[]
                for _ in range(2000):
                    if len(subset):draws.append(float(delta[rng.integers(0,len(subset),len(subset))].mean()))
                view_bootstrap[f"{'positive' if positive else 'negative'} / {kind} / {key}"]={'n_patients':len(subset),'mean_difference':float(delta.mean()) if len(subset) else None,
                     'ci95':np.quantile(draws,[.025,.975]).tolist() if draws else None,'valid_replicates':len(draws),'requested_replicates':2000}
            if positive:
                correct=[i for i in subset if (scores['M1'][i]>=.5)==bool(y[i]) and (scores['M2'][i]>=.5)==bool(y[i])]
                common_correct[kind]={'n':len(correct),'Delta_S':float(np.mean([models['M2'][i]['views'][kind]['S']-models['M1'][i]['views'][kind]['S'] for i in correct])) if correct else None,'supplement_only':True}
    paired_changes=[{'case_id':models['M1'][i]['case_id'],'image_id':ids[i],
                     'gray_Delta_S':models['M2'][i]['views']['gray']['S']-models['M1'][i]['views']['gray']['S']} for i in range(256) if 'views' in models['M1'][i] and models['M1'][i]['views']['positive']]
    (root/'paired_patient_changes.json').write_text(json.dumps(paired_changes,indent=2))
    delta_ap=results['M2']['classification']['raw_0_5']['ap']-results['M1']['classification']['raw_0_5']['ap']
    output={'status':'completed_evaluation','primary':'M2 minus M1 Delta_S, all frozen evaluable positive patients; joint primary raw0.5',
            'models':results,'classification_bootstrap':paired_bootstrap(models,folds,2000,42),
            'localization_joint_bootstrap':bootstrap,'evidence_paired_bootstrap':view_bootstrap,'common_correct_supplement':common_correct,
            'AP_difference_M2_M1':delta_ap,'AP_drop_warning_beyond_0_02':delta_ap < -.02,
            'limits':['Reused exploratory development set; no independent test or clinical validation','Single training seed; bootstrap excludes training randomness',
                      'AI-assisted annotation-guided views; no clinical causal ground truth','Normalized candidate support is not calibrated probability',
                      'M0 is a reference with different supervision/compute; primary matched comparison M2-M1'],
            'next_data_requirements':'New seeds and held-out patients; PadChest-GR/MS-CXR access/annotation review only, no download authorized'}
    (public/'medevidence_p1_results.json').write_text(json.dumps(output,indent=2))
    lines=['# MedEvidence pilot-1：探索性开发结果','',f"主要比较 M2−M1；AP 差值 {delta_ap:.6f}。预定 AP 下降警戒是否触发：{delta_ap < -.02}。",'',
           '| 模型 | AUROC | AP | 严格联合正确（原始0.5） | 交叉阈值严格联合正确 |','|---|---:|---:|---:|---:|']
    for name,res in results.items():
        cl=res['classification']['raw_0_5'];lines.append(f"| {name} | {cl['auroc']:.4f} | {cl['ap']:.4f} | {res['localization_raw_0_5']['strict_joint_all_patients']:.4f} | {res['localization_crossfit']['strict_joint_all_patients']:.4f} |")
    lines+=['','全部256名开发患者均进入分类和定位分母；解析失败计联合失败、区域未检出。未知预测框数不伪造，区域精度以已知输出数量条件定义。','',
            '## 证据分解（全部冻结合格开发阳性）','','| 条件 | 患者数 | M1 S | M2 S | 配对 ΔS | 95%区间 |','|---|---:|---:|---:|---:|---|']
    for kind in ('gray','blur','blank'):
        d=view_bootstrap[f'positive / {kind} / S'];lines.append(f"| {kind} | {d['n_patients']} | {results['M1']['evidence'][kind]['S']} | {results['M2']['evidence'][kind]['S']} | {d['mean_difference']} | {d['ci95']} |")
    lines+=['','实际测得的数据与2000次患者配对区间详见相邻 JSON。灰色视图、模糊压力测试、可见遮挡几何的黑背景、阴性假干预和2/2.5倍裁剪一起解释；不凭单项margin宣布机制成立。','',
            '重复使用开发集、单训练seed、AI辅助视图审核均限制结论。尚未验证临床因果机制、独立患者泛化及训练随机性；后续需新seed、新数据。没有自动增加实验或下载数据。']
    (public/'medevidence_p1_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'aggregate_status':'completed','public_reports':['reports/medevidence_p1_results.md','reports/medevidence_p1_results.json']}),flush=True)


if __name__=='__main__':main()
