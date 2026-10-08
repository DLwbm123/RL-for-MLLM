"""Out-of-fold proposal supervision and independent answer-correctness probabilities."""
import json
import os
from pathlib import Path
import warnings
import numpy as np
import torch
from scipy.special import expit
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from src.medevidence_p7 import Run as PreviousRun, read, worker as previous_worker, selection_stats, action_targets
from src.medevidence_p8 import matched_stats, matched_pair, evidence
from src.v2 import save, specificity_threshold


def input_features(records,ids):
    x=torch.stack([records[k]['x'] for k in ids]).float();v=torch.stack([records[k]['valid'] for k in ids])
    d=(x.shape[-1]-8)//2
    # No/yes correctness share observations, but have separate coefficients and targets.
    features=torch.cat([x[:,8,:d],x[:,0,d:2*d+5],v[:,0:1].float(),v[:,:8].sum(-1,keepdim=True)/8],-1).numpy()
    rewards=torch.stack([records[k]['rewards'] for k in ids])
    targets=torch.stack([rewards[:,0]==1,rewards[:,8]==1],-1).numpy().astype(float)
    return features,targets,v[:,0].numpy()


def fit_heads(x,y,cfg):
    assert np.isfinite(x).all() and y.shape==(len(x),2)
    mean=x.mean(0);scale=np.maximum(x.std(0),cfg['feature_std_floor']);z=(x-mean)/scale
    coefficients=[];intercepts=[];checks=[]
    for j in range(2):
        assert set(y[:,j])=={0.,1.},'Correctness target lacks a class'
        model=LogisticRegression(C=cfg['head_C'],solver='lbfgs',max_iter=cfg['head_max_iter'],tol=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);model.fit(z,y[:,j])
        coef=model.coef_[0];bias=float(model.intercept_[0]);logit=z@coef+bias
        loss=float(np.mean(np.logaddexp(0,logit)-y[:,j]*logit))
        prior=float(y[:,j].mean());initial=float(-prior*np.log(prior)-(1-prior)*np.log1p(-prior))
        if not np.isfinite(loss) or loss>initial+1e-6:raise RuntimeError('Full training loss failed to improve on intercept baseline')
        coefficients.append(coef);intercepts.append(bias)
        checks.append({'iterations':int(model.n_iter_[0]),'positive_targets':int(y[:,j].sum()),'initial_full_loss':initial,
            'final_full_loss':loss,'regularized_objective':loss+float(coef@coef)/(2*cfg['head_C']*len(x))})
    return {'mean':mean,'scale':scale,'coef':np.stack(coefficients),'bias':np.array(intercepts)},checks


def logits(model,x):return ((x-model['mean'])/model['scale'])@model['coef'].T+model['bias']


def calibrated_probabilities(z,shift,has_candidate):
    p=expit(z+shift);p[:,0]*=has_candidate
    return p


def calibrate(z,y,weights,has_candidate,l2):
    shifts=[]
    for j in range(2):
        mask=has_candidate if j==0 else np.ones(len(y),bool);w=weights[mask]
        if not len(w) or set(y[mask,j])!={0.,1.}:raise ValueError('Calibration target lacks usable classes')
        def gradient(b):return float(w@(expit(z[mask,j]+b)-y[mask,j])/w.sum()+l2*b)
        shifts.append(brentq(gradient,-30,30))
    return np.array(shifts)


def decisions(p,has_candidate,minimum_correctness=.4):
    choice=(p[:,0]>p[:,1])&has_candidate;confidence=np.where(choice,p[:,0],p[:,1])
    action=np.where(choice,0,8);action[confidence<minimum_correctness]=9
    return action,confidence


def weighted_cutoff(confidence,weights,coverage):
    assert 0<coverage<=1 and np.isfinite(confidence).all() and (weights>0).all()
    order=np.argsort(confidence);cumulative=np.cumsum(weights[order])
    index=min(np.searchsorted(cumulative,(1-coverage)*weights.sum(),side='right'),len(order)-1)
    # Inclusive ties are retained; report actual coverage instead of claiming an exact 75%.
    return float(confidence[order[index]])


class Run(PreviousRun):
    def __init__(self):super().__init__('medevidence_p9.json')

    def execute(self):
        from src.experiment import seed_all
        previous=self.base.parent/'medevidence_p8/outputs'
        original=list(self.sets['train']);folds=read(self.out/'protocol/folds.json');fold_receipts=[]
        self.candidates={k:v for k,v in read(previous/'candidates.json').items() if k in self.sets['calibration']}
        for i,heldout in enumerate(folds):
            self.tick();seed_all(17);training=sorted(set(original)-set(heldout));assert not set(training)&set(heldout)
            self.sets['train']=training;folder=self.out/f'fold_{i}';folder.mkdir()
            self.update('out_of_fold_training',fold=i+1,total_folds=len(folds),heldout_patients=len(heldout))
            self.detector(output_dir=folder,evaluate_calibration=False)
            used=set(sum(read(folder/'detector_schedule.json'),[]))
            assert used==set(training) and not used&set(heldout)
            fold_receipts.append({'fold':i+1,'fit_patients':len(used),'heldout_patients':len(heldout),'overlap':0,'steps':self.cfg['detector_steps']})
            self.update('out_of_fold_candidates',fold=i+1);self.predict(heldout)
            self.detector_model.cpu();del self.detector_model;torch.cuda.empty_cache()
        self.sets['train']=original
        positives=[k for k in original if self.rows[k]['boxes']]
        audit={'patients':len(original),'positive':len(positives),'top1_supported_positive':0,'positive_without_supported_candidate':0,
            'new_detector_steps':self.detector_steps,'folds':len(folds),'patient_leakage':0,'fold_receipts':fold_receipts}
        for k in positives:
            _,r,_=action_targets(self.candidates[k]['boxes'],self.rows[k]['boxes'])
            audit['top1_supported_positive']+=int(r[0]==1);audit['positive_without_supported_candidate']+=int(not r[:8].eq(1).any())
        save(self.out/'oof_audit.json',audit)
        if not audit['top1_supported_positive'] or not audit['positive_without_supported_candidate']:
            return 'stopped_oof_target_support'
        features=self.features(original+self.sets['calibration']);torch.save(features,self.out/'training_calibration_features.pt')
        self.update('correctness_training');x,y,has=input_features(features,original)
        model,checks=fit_heads(x,y,self.cfg);save(self.out/'head_training.json',{'heads':['supported_top1_yes','correct_no'],'patients':len(x),
            'positive_prior':len(positives)/len(x),'weighting':'natural patient distribution; no class-balanced resampling','checks':checks})
        self.tick();cal=self.sets['calibration'];cx,cy,ch=input_features(features,cal)
        labels=np.array([bool(self.rows[k]['boxes']) for k in cal]);prior=len(positives)/len(original)
        weights=np.where(labels,prior/labels.mean(),(1-prior)/(1-labels.mean()))
        cz=logits(model,cx);shift=calibrate(cz,cy,weights,ch,self.cfg['calibration_bias_l2'])
        p=calibrated_probabilities(cz,shift,ch);_,confidence=decisions(p,ch)
        threshold=max(.4,weighted_cutoff(confidence,weights,self.cfg['primary_coverage']))
        detector_scores=np.array([max(self.candidates[k]['scores'],default=0.) for k in cal])
        detector_threshold=specificity_threshold(labels,detector_scores,target=.9)
        detector_reject=weighted_cutoff(abs(detector_scores-detector_threshold),weights,self.cfg['primary_coverage'])
        calibration={'target_positive_prior':prior,'calibration_positive_fraction':float(labels.mean()),'bias_shift':shift.tolist(),
            'utility_minimum_correctness':.4,'correctness_reject_threshold':threshold,
            'weighted_calibration_coverage':float(weights@(confidence>=threshold)/weights.sum()),
            'detector_threshold':float(detector_threshold),'detector_reject_threshold':detector_reject,
            'weighted_brier_before':float(np.sum(weights[:,None]*(calibrated_probabilities(cz,np.zeros(2),ch)-cy)**2)/(2*weights.sum())),
            'weighted_brier_after':float(np.sum(weights[:,None]*(p-cy)**2)/(2*weights.sum())),
            'scope':'Two regularized intercept corrections on 64 calibration patients; not a clinical calibration guarantee; no development-based threshold choice'}
        save(self.out/'calibration.json',calibration)
        torch.save({**{k:torch.as_tensor(v) for k,v in model.items()},'calibration_shift':torch.as_tensor(shift)},self.out/'correctness_heads.pt')
        del features;self.tick();self.allow_dev=True;self.update('final_cached_development_evaluation')
        self.candidates.update({k:v for k,v in read(previous/'candidates.json').items() if k in self.sets['dev']})
        dev_features=torch.load(previous/'dev_features.pt',map_location='cpu',weights_only=True)
        assert set(dev_features)==set(self.sets['dev'])
        self.evaluate_heads(model,shift,dev_features,calibration);return 'completed'

    def evaluate_heads(self,model,shift,features,cal):
        ids=self.sets['dev'];x,y,has=input_features(features,ids);p=calibrated_probabilities(logits(model,x),shift,has)
        qa,qc=decisions(p,has);scores=np.array([max(self.candidates[k]['scores'],default=0.) for k in ids])
        da=np.where((scores>=cal['detector_threshold'])&has,0,8);dc=abs(scores-cal['detector_threshold'])
        methods={'detector_threshold':(da,dc),'detector_calibrated75':(np.where(dc>=cal['detector_reject_threshold'],da,9),dc),
            'correctness_utility':(qa,qc),'correctness_calibrated75':(np.where(qc>=cal['correctness_reject_threshold'],qa,9),qc),
            'always_no':(np.full(len(ids),8),np.ones(len(ids))),'always_abstain':(np.full(len(ids),9),np.ones(len(ids)))}
        outputs={}
        for name,(actions,confidence) in methods.items():
            rs=[]
            for i,k in enumerate(ids):
                _,r,_=action_targets(self.candidates[k]['boxes'],self.rows[k]['boxes']);a=int(actions[i])
                rs.append({'image_id':k,'positive':bool(self.rows[k]['boxes']),'action':a,'confidence':float(confidence[i]),'correct':bool(r[a]==1),'reward':float(r[a])})
            outputs[name]=rs
        save(self.out/'predictions.json',outputs)
        comparison=matched_pair(outputs['detector_threshold'],outputs['correctness_utility'],self.cfg['primary_coverage'],self.cfg['bootstrap_repeats'])
        passed=evidence(comparison,self.cfg['bootstrap_repeats'])
        save(self.out/'evaluation.json',{'metrics':{n:selection_stats(rs) for n,rs in outputs.items()},
            'matched_coverage':{n:[matched_stats(rs,c) for c in self.cfg['matched_coverages']] for n,rs in outputs.items()},
            'correctness_minus_detector':comparison,'decision':'KEEP_SUPERVISED_BASELINE_FOR_CONFIRMATION' if passed else 'STOP_CURRENT_CORRECTNESS_BASELINE',
            'primary_coverage':self.cfg['primary_coverage'],'primary_scope':'Retrospective matched coverage; deployment-style calibration thresholds reported separately at their actual coverage',
            'development_brier':np.mean((p-y)**2,axis=0).tolist(),'development_cached_patients':len(ids),'new_development_pixel_reads':0,
            'RL_updates':0,'test_pixels_read':0})


def worker():previous_worker(Run)


def report():
    import re
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';public=base/'code/reports';public.mkdir(exist_ok=True)
    names={'final':'FINAL.json','budget':'gpu_ledger.json','training':'head_training.json','calibration':'calibration.json','oof':'oof_audit.json',
        'evaluation':'evaluation.json','preparation':'protocol/CPU_checks.json'}
    values={n:read(out/p) for n,p in names.items() if (out/p).exists()}
    for n,v in values.items():save(public/('medevidence_p9_'+n+'.json'),json.loads(re.sub(r'/(?:data|Users|remote-home)/[^\s"\\]+','[private-path]',json.dumps(v))))
    f=values['final'];text='# MedEvidence P9 实际结果\n\n'+f"状态：{f['status']}；折外检测器总更新{f['detector_steps']}步；RL及语言模型更新0。\n\n"
    if 'evaluation' in values:
        e=values['evaluation'];text+=f"冻结决策：`{e['decision']}`。\n\n"
        text+='| 方法 | 实际回答覆盖率 | 已回答错误风险 | 阳性有证据成功率 | 阴性误报率 |\n|---|---:|---:|---:|---:|\n'
        fmt=lambda v:'NA' if v is None else f'{v:.4f}'
        for n,m in e['metrics'].items():text+='| '+n+' | '+' | '.join(fmt(m[k]) for k in ['coverage','answered_risk','positive_supported_success_rate','negative_wrong_yes_rate'])+' |\n'
        text+='\n主要75%相同覆盖率比较及配对区间见evaluation JSON；表中为校准规则实际覆盖率，二者不可混同。\n'
    else:text+='尚无完整评价，不支持有效性结论。\n'
    text+=f"\n本轮GPU小时：{values['budget']['new_charged_seconds']/3600:.6f}/0.5；累计：{values['budget']['combined_GPU_hours']:.6f}/3。\n\n"
    text+='三折训练候选与全960人最终检测器仍可能存在分布差异；64例校准仅作有限研发校准。开发集已多次使用，本轮不能证明独立泛化、临床安全或MLLM的RL收益。原图、患者级记录、特征和权重保持私有；不自动重试或扩展。\n'
    (public/'medevidence_p9_decision.md').write_text(text)
