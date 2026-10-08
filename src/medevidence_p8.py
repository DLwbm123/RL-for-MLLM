"""One frozen-detector selection experiment; matched coverage is descriptive evaluation."""
import os
from pathlib import Path
import numpy as np
import torch
from src.medevidence_p7 import Run as PreviousRun, read, action_targets, selection_stats
from src.medevidence_p7 import worker as previous_worker
from src.v2 import save, specificity_threshold


def matched_stats(records, coverage, weights=None):
    """Uniform random retention within a confidence tie, reported as expected counts."""
    if not 0 < coverage <= 1:raise ValueError('Coverage must be in (0, 1]')
    w=np.ones(len(records)) if weights is None else np.asarray(weights,dtype=float)
    if len(w)!=len(records) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Invalid patient weights')
    action=np.array([r['action'] for r in records]);eligible=action!=9
    confidence=np.array([r['confidence'] for r in records]);assert np.isfinite(confidence).all()
    target=coverage*w.sum();capacity=float(w@eligible/w.sum())
    if w@eligible+1e-9<target:return {'available':False,'maximum_coverage':capacity,'requested_coverage':coverage}
    keep=np.zeros(len(w));remaining=target
    for score in np.unique(confidence[eligible])[::-1]:
        group=eligible&(confidence==score);mass=w@group
        if mass==0:continue
        fraction=min(1.,remaining/mass);keep[group]=fraction;remaining-=fraction*mass
        if remaining<1e-9:break
    positive=np.array([r['positive'] for r in records]);correct=np.array([r['correct'] for r in records]);yes=action<8
    mean=lambda x,mask:float(w@(np.asarray(x)*mask)/(w@mask)) if w@mask else None
    accepted=w*keep
    return {'available':True,'coverage':float(accepted.sum()/w.sum()),'maximum_coverage':capacity,
        'answered_risk':float(accepted@(~correct)/accepted.sum()),
        'positive_supported_success_rate':mean(keep*correct*yes,positive),
        'negative_wrong_yes_rate':mean(keep*yes,~positive),
        'positive_supported_success':float(accepted@(correct&yes&positive)),
        'negative_wrong_yes':float(accepted@(yes&~positive)),
        'mean_utility':float((accepted@np.array([r['reward'] for r in records])-.2*(w.sum()-accepted.sum()))/w.sum())}


def matched_pair(a,b,coverage,repeats):
    assert [r['image_id'] for r in a]==[r['image_id'] for r in b]
    x,y=matched_stats(a,coverage),matched_stats(b,coverage)
    if not (x['available'] and y['available']):return {'available':False,'baseline':x,'comparison':y}
    keys=['answered_risk','positive_supported_success_rate','negative_wrong_yes_rate','mean_utility']
    rng=np.random.default_rng(42);samples={k:[] for k in keys}
    for _ in range(repeats):
        w=np.bincount(rng.integers(len(a),size=len(a)),minlength=len(a))
        xx,yy=matched_stats(a,coverage,w),matched_stats(b,coverage,w)
        if not (xx['available'] and yy['available']):continue
        for k in keys:
            if xx[k] is not None and yy[k] is not None:samples[k].append(yy[k]-xx[k])
    return {'available':True,'coverage':coverage,'comparison_minus_baseline':{
        k:{'difference':y[k]-x[k] if x[k] is not None and y[k] is not None else None,
           'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,'valid_replicates':len(v)} for k,v in samples.items()}}


def evidence(pair,repeats):
    if not pair['available']:return False
    d=pair['comparison_minus_baseline'];risk=d['answered_risk']
    return bool(risk['valid_replicates']>=.95*repeats and risk['ci95'][1]<0
        and d['positive_supported_success_rate']['difference'] is not None
        and d['positive_supported_success_rate']['difference']>=0 and d['negative_wrong_yes_rate']['difference']<=0)


class Run(PreviousRun):
    def __init__(self):super().__init__('medevidence_p8.json')

    def evaluate(self,records):
        cal=self.sets['calibration'];threshold=specificity_threshold([bool(self.rows[k]['boxes']) for k in cal],
            [max(self.candidates[k]['scores'],default=0.) for k in cal],target=.9)
        outputs={}
        for name in ['detector_threshold','always_no','always_abstain']+list(self.actors):
            rs=[]
            for k in self.sets['dev']:
                self.tick()
                if name=='always_no':action,confidence=8,1.
                elif name=='always_abstain':action,confidence=9,1.
                elif name=='detector_threshold':
                    score=max(self.candidates[k]['scores'],default=0.)
                    action=0 if score>=threshold and self.candidates[k]['boxes'] else 8
                    confidence=abs(score-threshold)
                else:
                    with torch.no_grad():p=self.actors[name](records[k]['x'].float()[None],records[k]['valid'][None]).exp()[0]
                    confidence,action=p.max(0);action=int(action);confidence=float(confidence)
                _,reward,_=action_targets(self.candidates[k]['boxes'],self.rows[k]['boxes'])
                rs.append({'image_id':k,'positive':bool(self.rows[k]['boxes']),'action':action,'confidence':float(confidence),
                    'correct':bool(reward[action]==1),'reward':float(reward[action])})
            outputs[name]=rs
        save(self.out/'predictions.json',outputs)
        comparisons={name:matched_pair(outputs[a],outputs[b],self.cfg['primary_coverage'],self.cfg['bootstrap_repeats'])
            for name,a,b in [('SFT_minus_detector','detector_threshold','SFT'),('bandit_minus_detector','detector_threshold','bandit'),('bandit_minus_SFT','SFT','bandit')]}
        gates={name:evidence(pair,self.cfg['bootstrap_repeats']) for name,pair in comparisons.items()}
        decision=('KEEP_BANDIT_FOR_FUTURE_CONFIRMATION' if gates['bandit_minus_detector'] and gates['bandit_minus_SFT']
            else 'KEEP_SFT_NO_REWARD_OPTIMIZATION_EVIDENCE' if gates['SFT_minus_detector'] else 'STOP_CURRENT_SELECTOR_BRANCH')
        result={'metrics':{n:selection_stats(rs) for n,rs in outputs.items()},
            'matched_coverage':{n:[matched_stats(rs,c) for c in self.cfg['matched_coverages']] for n,rs in outputs.items()},
            'primary_coverage':self.cfg['primary_coverage'],'comparisons':comparisons,'engineering_evidence':gates,'decision':decision,
            'detector_threshold':float(threshold),'tie_rule':'expected uniform retention within tied confidence; no label-based tie breaking',
            'scope':'Retrospective confidence-ranked risk at fixed coverage, not deployed calibrated abstention; single seed and reused development set; unadjusted exploratory bootstrap intervals.'}
        save(self.out/'evaluation.json',result);return result

    def execute(self):
        prior=self.base.parent/'medevidence_p7/outputs'
        self.update('loading_frozen_detector',reused_detector_steps=800,new_detector_steps=0)
        self.detector_model=self.build_detector()
        self.detector_model.load_state_dict(torch.load(prior/'detector_final.pt',map_location='cpu',weights_only=True))
        self.detector_model.eval().requires_grad_(False).cuda()
        self.candidates=read(prior/'candidates.json');assert set(self.candidates)==set(self.sets['calibration'])
        save(self.out/'calibration_candidate_audit.json',read(prior/'candidate_gate.json'))
        self.update('training_candidates');self.predict(self.sets['train'])
        self.detector_model.cpu();torch.cuda.empty_cache()
        features=self.features(self.sets['train']);torch.save(features,self.out/'train_features.pt')
        self.fit_policy(features);del features;self.allow_dev=True
        self.update('final_development_candidates');self.detector_model.cuda();self.predict(self.sets['dev'])
        self.detector_model.cpu();torch.cuda.empty_cache()
        save(self.out/'development_candidate_audit.json',self.candidate_audit(self.sets['dev']))
        features=self.features(self.sets['dev']);torch.save(features,self.out/'dev_features.pt')
        self.update('evaluation');self.evaluate(features);return 'completed'


def worker():previous_worker(Run)


def report():
    import re
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';public=base/'code/reports'
    names={'final':'FINAL.json','budget':'gpu_ledger.json','training':'actor_training.json','evaluation':'evaluation.json',
        'development_candidates':'development_candidate_audit.json','preparation':'protocol/CPU_checks.json'}
    values={n:read(out/p) for n,p in names.items() if (out/p).exists()}
    for n,v in values.items():save(public/('medevidence_p8_'+n+'.json'),json.loads(re.sub(r'/(?:data|Users|remote-home)/[^\s"\\]+','[private-path]',json.dumps(v))))
    f=values['final'];text='# MedEvidence P8 实际结果\n\n'+f"状态：{f['status']}；新增检测器与语言模型更新均0步。复用P7最终检测器。\n\n"
    if 'evaluation' in values:
        e=values['evaluation'];text+=f"冻结的75%覆盖率决策：`{e['decision']}`。\n\n"
        text+='| 策略 | 自然覆盖率 | 75%覆盖风险 | 75%覆盖阳性有证据成功率 | 75%覆盖阴性误报率 |\n|---|---:|---:|---:|---:|\n'
        fmt=lambda x:'NA' if x is None else f'{x:.4f}'
        for n,m in e['metrics'].items():
            v=e['matched_coverage'][n][2]
            text+='| '+n+' | '+fmt(m['coverage'])+' | '+' | '.join(fmt(v.get(k)) for k in ['answered_risk','positive_supported_success_rate','negative_wrong_yes_rate'])+' |\n'
    else:text+='未形成完整选择器比较，不支持效能或奖励优化收益声明。\n'
    text+=f"\n本轮GPU小时：{values['budget']['new_charged_seconds']/3600:.6f}/0.5；累计：{values['budget']['combined_GPU_hours']:.6f}/3。\n\n"
    text+='并列置信度按等概率保留的期望风险计算，非已部署拒答阈值。P7候选门槛失败结论保持不变；P8不以该门槛拦截实验。固定视觉特征上的小型有限动作actor不是语言模型RL。单seed、反复使用的开发集、不完整候选与未调整的探索性置信区间限制结论；不证明临床安全、独立泛化或因果忠实。原图、GT、患者级预测、特征及权重保持私有。无自动重试或扩展。\n'
    (public/'medevidence_p8_decision.md').write_text(text)
