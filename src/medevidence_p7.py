"""Frozen proposals, visual-feature selection and an exact finite-action bandit pilot."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import torch
from torch import nn
from src.v2 import save,DevelopmentData,specificity_threshold


def read(path):return json.loads(Path(path).read_text())


def overlaps(boxes,gt):
    a=np.asarray(boxes,float).reshape(-1,4);b=np.asarray(gt,float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(-1)
    area=lambda x:np.maximum(x[:,2:]-x[:,:2],0).prod(-1)
    return intersection/np.maximum(area(a)[:,None]+area(b)[None,:]-intersection,1e-12)


def action_targets(boxes,gt,k=8):
    """GT is used only here for supervision/evaluation, never in proposal generation/features."""
    if len(boxes)>k:raise ValueError('Too many candidates')
    valid=torch.zeros(k+2,dtype=torch.bool);valid[:len(boxes)]=True;valid[k:]=True
    rewards=torch.full((k+2,),-1.);rewards[k+1]=-.2
    good=torch.zeros(k+2,dtype=torch.bool)
    if len(gt):
        hit=overlaps(boxes,gt).max(1)>=.5 if len(boxes) else np.zeros(0,bool)
        good[:len(boxes)]=torch.as_tensor(hit);rewards[:len(boxes)][good[:len(boxes)]]=1.
        if not good.any():good[k+1]=True
    else:good[k]=True;rewards[k]=1.
    return valid,rewards,good


class Actor(nn.Module):
    def __init__(self,dimension,hidden=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(dimension,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,x,valid):return self.net(x).squeeze(-1).masked_fill(~valid,-torch.inf).log_softmax(-1)


def actor_loss(logp,valid,rewards,good,reference=None,beta=.01):
    if reference is None:return -logp.masked_fill(~good,-torch.inf).logsumexp(-1).mean()
    # Full action enumeration is exact here; no sampled old-policy estimate or clipping is needed.
    safe=torch.where(valid,logp,torch.zeros_like(logp));ref=torch.where(valid,reference,torch.zeros_like(reference))
    return (-(logp.exp()*rewards).sum(-1)+beta*(logp.exp()*(safe-ref)).sum(-1)).mean()


def selection_stats(records,weights=None):
    w=np.ones(len(records)) if weights is None else np.asarray(weights);n=w.sum()
    y=np.array([r['positive'] for r in records]);answer=np.array([r['action']!=9 for r in records])
    correct=np.array([r['correct'] for r in records]);yes=np.array([r['action']<8 for r in records])
    mean=lambda x,mask:float(np.dot(w,np.asarray(x)*mask)/np.dot(w,mask)) if np.dot(w,mask)>0 else None
    return {'patients':int(n),'positive_supported_success_rate':mean(correct&yes,y),
        'negative_wrong_yes_rate':mean(yes,~y),'coverage':mean(answer,np.ones(len(y),bool)),
        'answered_risk':mean(~correct,answer),'mean_utility':float(np.dot(w,[r['reward'] for r in records])/n),
        'positive_supported_success':int(np.dot(w,correct&yes&y)),'negative_wrong_yes':int(np.dot(w,yes&~y)),
        'wrong_no':int(np.dot(w,(~yes)&answer&y)),'abstentions':int(np.dot(w,~answer))}


def paired(a,b,repeats=2000):
    assert [r['image_id'] for r in a]==[r['image_id'] for r in b]
    keys=['positive_supported_success_rate','negative_wrong_yes_rate','coverage','answered_risk','mean_utility']
    x,y=selection_stats(a),selection_stats(b);samples={k:[] for k in keys};rng=np.random.default_rng(42)
    for _ in range(repeats):
        w=np.bincount(rng.integers(len(a),size=len(a)),minlength=len(a));xx,yy=selection_stats(a,w),selection_stats(b,w)
        for k in keys:
            if xx[k] is not None and yy[k] is not None:samples[k].append(yy[k]-xx[k])
    return {k:{'difference':y[k]-x[k] if x[k] is not None and y[k] is not None else None,
        'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,'valid_replicates':len(v)} for k,v in samples.items()}


class Run:
    def __init__(self,config_name="medevidence_p7.json"):
        import pandas as pd
        from src.experiment import seed_all
        self.start=time.time();self.base=Path(os.environ['PILOT_ROOT']);self.out=self.base/'outputs';self.cfg=read(self.base/'code/configs'/config_name)
        self.deadline=float(os.environ['JOB_DEADLINE']);self.stop=False
        signal.signal(signal.SIGUSR1,lambda *_:setattr(self,'stop',True))
        auth=read(self.out/'authorization.json');assert auth['gpu_authorized'] and auth['source_commit']
        self.frame=pd.read_csv(self.out/'protocol/manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
        self.rows=self.frame.set_index('image_id').to_dict('index')
        for r in self.rows.values():r['boxes']=json.loads(r['boxes'])
        self.sets=read(self.out/'protocol/sets.json');self.cache={};self.candidates={};self.status={};self.detector_steps=0;self.stage='initialization';self.allow_dev=False
        self.data=DevelopmentData(os.environ['DATA_ROOT'],self.frame,self.out/'data_access.jsonl');self.data.install_guard()
        seed_all(self.cfg['seed']);torch.set_num_threads(4)
        argv=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip()
        assert not any(s in argv.lower() for s in ['wangbomin','medevidence','mllm','qwen','sft','grpo'])
        self.update('initialization',argv_verified=True)

    def tick(self):
        if self.stop or time.time()>self.deadline-60:raise TimeoutError('Cumulative GPU budget reserve')

    def update(self,stage,**values):
        self.stage=stage;self.status.update(stage=stage,elapsed_seconds=time.time()-self.start,**values)
        save(self.out/'progress.json',self.status);print(json.dumps(self.status),flush=True)

    def image(self,k):
        self.tick()
        if k in self.sets['dev'] and not self.allow_dev:raise PermissionError('Development pixels sealed until both policy branches finish')
        if k not in self.cache:
            self.cache[k]=self.data.image(k)
            if self.cache[k].size!=(self.rows[k]['width'],self.rows[k]['height']):raise ValueError('Manifest/image geometry mismatch')
        return self.cache[k]

    def schedule(self,steps,per_class,seed):
        rng=np.random.default_rng(seed);pools={y:sorted(k for k in self.sets['train'] if bool(self.rows[k]['boxes'])==y) for y in [True,False]};queues={True:[],False:[]};plan=[]
        for _ in range(steps):
            batch=[]
            for y in [True,False]:
                for _ in range(per_class):
                    if not queues[y]:queues[y]=list(rng.permutation(pools[y]))
                    batch.append(str(queues[y].pop()))
            plan.append(batch)
        return plan

    def build_detector(self):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        c=self.cfg
        m=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=c['detector_min_size'],max_size=c['detector_max_size'],
            box_score_thresh=c['candidate_score_threshold'],box_nms_thresh=c['candidate_nms_iou'],box_detections_per_img=c['candidate_limit'])
        m.load_state_dict(torch.load(Path(os.environ['MODEL_ROOT'])/c['weights_filename'],map_location='cpu',weights_only=True))
        m.roi_heads.box_predictor=FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,2)
        for name,p in m.backbone.body.named_parameters():p.requires_grad_(name.startswith(('layer2.','layer3.','layer4.')))
        return m

    def detector(self,output_dir=None,evaluate_calibration=True):
        from torchvision.transforms.functional import to_tensor
        out=Path(output_dir) if output_dir is not None else self.out
        previous_steps=self.detector_steps
        m=self.build_detector().cuda();m.train()
        for module in m.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=self.cfg['detector_lr'],momentum=.9,weight_decay=.0005)
        plan=self.schedule(self.cfg['detector_steps'],2,17);save(out/'detector_schedule.json',plan)
        self.update('detector_training',planned_steps=len(plan),steps=0)
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        own=[s for s in gpu.splitlines() if s.split(',')[0].strip()==str(os.getpid())]
        assert own and not any(t in ' '.join(own).lower() for t in ('wangbomin','medevidence','mllm','qwen','sft','grpo'))
        save(out/'gpu_process_receipt.json',{'neutral_process':True,'GPU_process':own})
        with (out/'detector_training.jsonl').open('x') as log:
            for step,ids in enumerate(plan):
                self.tick()
                if time.time()-self.start>self.cfg['detector_phase_limit_seconds']-180:raise TimeoutError('Detector phase reserve')
                images=[to_tensor(self.image(k)).cuda() for k in ids]
                targets=[{'boxes':torch.tensor(self.rows[k]['boxes'],device='cuda',dtype=torch.float32).reshape(-1,4),
                    'labels':torch.ones(len(self.rows[k]['boxes']),device='cuda',dtype=torch.int64)} for k in ids]
                ratio=(.1+.9*(step+1)/50) if step<50 else .1+.9*(1+math.cos(math.pi*(step-50)/max(1,len(plan)-50)))/2
                for g in opt.param_groups:g['lr']=self.cfg['detector_lr']*ratio
                opt.zero_grad(set_to_none=True);losses=m(images,targets);loss=sum(losses.values())
                if not torch.isfinite(loss):raise FloatingPointError('Detector loss nonfinite')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),10.)
                if not torch.isfinite(norm):raise FloatingPointError('Detector gradient nonfinite')
                opt.step();self.detector_steps=previous_steps+step+1
                r={'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm),'patients':ids};log.write(json.dumps(r)+'\n');log.flush()
                if (step+1)%40==0:self.update('detector_training',steps=step+1,loss=r['loss'])
        torch.save(m.state_dict(),out/'detector_final.pt');del opt
        m.eval().requires_grad_(False);self.detector_model=m
        if not evaluate_calibration:return True
        self.predict(self.sets['calibration'])
        gate=self.candidate_audit(self.sets['calibration']);save(self.out/'candidate_gate.json',gate)
        self.update('candidate_gate',candidate_gate_passed=gate['passed'])
        return gate['passed']

    def predict(self,ids):
        from torchvision.transforms.functional import to_tensor
        # This inference path never consumes the row's GT boxes.
        with torch.no_grad():
            for k in ids:
                if k in self.candidates:continue
                self.tick();im=self.image(k);r=self.detector_model([to_tensor(im).cuda()])[0]
                boxes=r['boxes'].float().cpu().numpy();scores=r['scores'].float().cpu().tolist()
                if len(boxes)>8 or not np.isfinite(boxes).all():raise ValueError('Candidate output invalid')
                for box in boxes:
                    if not (0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height):raise ValueError('Candidate outside image')
                self.candidates[k]={'boxes':boxes.tolist(),'scores':scores}
        save(self.out/'candidates.json',self.candidates)

    def candidate_audit(self,ids):
        from scipy.optimize import linear_sum_assignment
        def stats(source_ids):
            pos=hits=regions=matches=0
            for k,source in zip(ids,source_ids):
                gt=self.rows[k]['boxes']
                if not gt:continue
                pos+=1;regions+=len(gt);r=self.rows[k];s=self.rows[source]
                boxes=np.array(self.candidates[source]['boxes']).reshape(-1,4)*np.array([r['width']/s['width'],r['height']/s['height']]*2)
                a=overlaps(boxes,gt);hits+=int(bool(a.size and a.max()>=.5))
                if a.size:
                    x,y=linear_sum_assignment(-(a>=.5).astype(float)-a*.000001);matches+=int((a[x,y]>=.5).sum())
            return {'positive_patients':pos,'covered_positive_patients':hits,'positive_any_hit':hits/pos,'GT_regions':regions,'matched_regions':matches,'region_recall':matches/regions}
        actual=stats(ids);rng=np.random.default_rng(42);shuffled=[]
        for _ in range(100):
            for attempt in range(1000):
                other=list(rng.permutation(ids))
                if all(a!=b for a,b in zip(ids,other)):break
            else:raise RuntimeError('No derangement')
            shuffled.append(stats(other)['positive_any_hit'])
        actual.update(shuffled_any_hit_mean=float(np.mean(shuffled)),real_minus_shuffled_any_hit=actual['positive_any_hit']-float(np.mean(shuffled)),permutation_repeats=100)
        actual['conditions']={k:actual[k]>=v for k,v in self.cfg['candidate_gate'].items()};actual['passed']=all(actual['conditions'].values())
        actual['scope']='Candidate support only; not diagnostic reliability';return actual

    def features(self,ids):
        from src.model import Model
        from src.regions import region_tokens
        from src.data import QUESTIONS
        self.update('frozen_visual_features',feature_patients=len(ids))
        model=Model(Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct',None,False,576,1024)
        QUESTIONS['A']='Does this chest radiograph contain lung opacity suspicious for pneumonia?'
        records={}
        with torch.no_grad():
            for i,k in enumerate(ids):
                self.tick();im=self.image(k);inp,(gh,gw)=model.prompt(im,'A');visual=model.encode(inp).float();d=visual.shape[1]
                global_feature=nn.functional.layer_norm(visual.mean(0),(d,)).cpu();x=torch.zeros(10,d*2+8)
                for j,(box,score) in enumerate(zip(self.candidates[k]['boxes'],self.candidates[k]['scores'])):
                    selected=region_tokens(box,im.width,im.height,gh,gw)
                    if not selected:raise ValueError('Candidate lacks visual tokens')
                    local=nn.functional.layer_norm(visual[selected].mean(0),(d,)).cpu()
                    geometry=torch.tensor([box[0]/im.width,box[1]/im.height,box[2]/im.width,box[3]/im.height,score,1,0,0])
                    x[j]=torch.cat([global_feature,local,geometry])
                x[8]=torch.cat([global_feature,torch.zeros(d),torch.tensor([0.,0,0,0,0,0,1,0])])
                x[9]=torch.cat([global_feature,torch.zeros(d),torch.tensor([0.,0,0,0,0,0,0,1])])
                valid,rewards,good=action_targets(self.candidates[k]['boxes'],self.rows[k]['boxes'])
                assert torch.isfinite(x).all();records[k]={'x':x.half(),'valid':valid,'rewards':rewards,'good':good}
                if (i+1)%128==0:self.update('frozen_visual_features',features_done=i+1,feature_patients=len(ids))
        assert not any(p.requires_grad for p in model.model.parameters());del model;torch.cuda.empty_cache()
        return records

    def fit_policy(self,records):
        from src.experiment import seed_all
        seed_all(17);ids=self.sets['train'];pos={k:i for i,k in enumerate(ids)}
        tensors={key:torch.stack([records[k][key] for k in ids]) for key in ['x','valid','rewards','good']};tensors['x']=tensors['x'].float()
        actor=Actor(tensors['x'].shape[-1],self.cfg['policy_hidden_size']);self.update('actor_warmup')
        warm_plan=self.schedule(self.cfg['policy_warmup_steps'],8,17);plan=self.schedule(self.cfg['policy_continuation_steps'],8,18)
        save(self.out/'actor_schedule.json',{'warmup':warm_plan,'continuation':plan})
        def fit(model,batches,reference=None):
            opt=torch.optim.AdamW(model.parameters(),lr=self.cfg['policy_lr']);history=[]
            for batch in batches:
                self.tick();idx=[pos[k] for k in batch];x,v,r,g=[tensors[key][idx] for key in ['x','valid','rewards','good']]
                opt.zero_grad(set_to_none=True);logp=model(x,v)
                with torch.no_grad():ref=reference(x,v) if reference is not None else None
                loss=actor_loss(logp,v,r,g,ref,self.cfg['policy_kl_beta']);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Actor update nonfinite')
                opt.step();history.append(float(loss.detach()))
            return {'steps':len(history),'first_loss':history[0],'last_loss':history[-1],'patient_exposures':sum(map(len,batches))}
        training={'warmup':fit(actor,warm_plan)};reference=deepcopy(actor).eval().requires_grad_(False)
        a,b=deepcopy(actor),deepcopy(actor)
        assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))
        self.update('actor_SFT');training['SFT']=fit(a,plan)
        self.update('actor_bandit');training['bandit']=fit(b,plan,reference)
        assert all(torch.equal(p,q) for p,q in zip(actor.parameters(),reference.parameters()))
        self.actors={'warmup':actor.eval(),'SFT':a.eval(),'bandit':b.eval()}
        torch.save({n:m.state_dict() for n,m in self.actors.items()},self.out/'actors.pt');save(self.out/'actor_training.json',training)

    def evaluate(self,records):
        ids=self.sets['dev'];cal=self.sets['calibration'];scores=[max(self.candidates[k]['scores'],default=0.) for k in cal]
        threshold=specificity_threshold([bool(self.rows[k]['boxes']) for k in cal],scores,target=.9)
        outputs={};metrics={};curves={}
        def record(k,action,confidence):
            _,reward,_=action_targets(self.candidates[k]['boxes'],self.rows[k]['boxes'])
            return {'image_id':k,'positive':bool(self.rows[k]['boxes']),'action':int(action),'confidence':float(confidence),'correct':bool(reward[action]==1),'reward':float(reward[action])}
        for name in ['detector_top1','always_no','always_abstain']+list(self.actors):
            rs=[]
            for k in ids:
                self.tick()
                if name=='always_no':action,confidence=8,1.
                elif name=='always_abstain':action,confidence=9,1.
                elif name=='detector_top1':
                    score=max(self.candidates[k]['scores'],default=0.);action,confidence=(0 if score>=threshold and self.candidates[k]['boxes'] else 8),max(score,1-score)
                else:
                    with torch.no_grad():p=self.actors[name](records[k]['x'].float()[None],records[k]['valid'][None]).exp()[0]
                    confidence,action=p.max(0);action=int(action);confidence=float(confidence)
                rs.append(record(k,action,confidence))
            outputs[name]=rs;metrics[name]=selection_stats(rs)
            curves[name]=[]
            for t in [0,.5,.6,.7,.8,.9]:
                rr=[record(r['image_id'],r['action'] if r['confidence']>=t else 9,r['confidence']) for r in rs]
                curves[name].append({'confidence_threshold':t,**selection_stats(rr)})
        save(self.out/'predictions.json',outputs)
        a,b=metrics['SFT'],metrics['bandit'];conditions={'supported_success_increased':b['positive_supported_success']>a['positive_supported_success'],
            'negative_wrong_yes_not_up':b['negative_wrong_yes']<=a['negative_wrong_yes'],'coverage_not_down':b['coverage']>=a['coverage'],'utility_increased':b['mean_utility']>a['mean_utility']}
        result={'metrics':metrics,'risk_coverage':curves,'detector_threshold':float(threshold),'bandit_minus_SFT':paired(outputs['SFT'],outputs['bandit'],self.cfg['bootstrap_repeats']),
            'engineering_conditions':conditions,'directional_success':all(conditions.values()),'classification_scope':'target lung opacity; no clinical abstention labels or causal faithfulness validation'}
        save(self.out/'evaluation.json',result);return result

    def execute(self):
        if not self.detector():return 'stopped_candidate_gate'
        self.update('training_candidates');self.predict(self.sets['train'])
        # Keep the detector on CPU while Qwen encodes; no concurrent GPU models are necessary.
        self.detector_model.cpu();torch.cuda.empty_cache();train_features=self.features(self.sets['train'])
        torch.save(train_features,self.out/'train_features.pt');self.fit_policy(train_features);del train_features;self.allow_dev=True
        self.update('final_development_candidates');self.detector_model.cuda();self.predict(self.sets['dev']);self.detector_model.cpu();torch.cuda.empty_cache()
        save(self.out/'development_candidate_audit.json',self.candidate_audit(self.sets['dev']))
        dev_features=self.features(self.sets['dev']);torch.save(dev_features,self.out/'dev_features.pt');self.evaluate(dev_features)
        return 'completed'


def worker(run_class=Run):
    run=None
    try:
        run=run_class();status=run.execute();reason=None
    except Exception as exc:
        status='stopped_budget' if isinstance(exc,TimeoutError) else 'failed';reason=type(exc).__name__+': '+str(exc)
        if run:save(run.out/'failure.json',{'reason':reason,'stage':run.stage,'detector_steps':run.detector_steps})
        else:raise
    save(run.out/'FINAL.json',{'status':status,'reason':reason,'stage':run.stage,'detector_steps':run.detector_steps,
        'test_pixels_read':0,'data_access':dict(run.data.access_counts),'runtime_seconds':time.time()-run.start,
        'language_model_updates':0,'scope':run.cfg['scope']})


def report():
    import re
    base=Path(os.environ['PILOT_ROOT']);out=base/'outputs';public=base/'code/reports'
    names={'final':'FINAL.json','budget':'gpu_ledger.json','candidates':'candidate_gate.json','training':'actor_training.json','evaluation':'evaluation.json','development_candidates':'development_candidate_audit.json','preparation':'protocol/CPU_checks.json'}
    values={name:read(out/path) for name,path in names.items() if (out/path).exists()}
    for name,value in values.items():
        clean=json.loads(re.sub(r'/(?:data|Users|remote-home)/[^\s"\\]+','[private-path]',json.dumps(value)))
        save(public/('medevidence_p7_'+name+'.json'),clean)
    f=values['final'];text='# MedEvidence P7 实际结果\n\n'+f"状态：{f['status']}。检测器实际{f['detector_steps']}步；语言模型更新0步。\n\n"
    if f['reason']:text+='停止原因见final JSON；不自动重试。\n\n'
    if 'candidates' in values:
        c=values['candidates'];text+=f"校准候选覆盖：{c['covered_positive_patients']}/{c['positive_patients']}阳性；GT区域匹配{c['matched_regions']}/{c['GT_regions']}；相对患者错配覆盖差{c['real_minus_shuffled_any_hit']:+.4f}。候选准入={c['passed']}。\n\n"
    if 'evaluation' in values:
        text+='| 策略 | 正例有证据成功率 | 阴性错误yes率 | 回答覆盖率 | 已回答错误风险 | 平均效用 |\n|---|---:|---:|---:|---:|---:|\n'
        for name,v in values['evaluation']['metrics'].items():
            fmt=lambda x:'NA' if x is None else f'{x:.4f}'
            text+='| '+name+' | '+' | '.join(fmt(v[k]) for k in ['positive_supported_success_rate','negative_wrong_yes_rate','coverage','answered_risk','mean_utility'])+' |\n'
    else:text+='策略训练/比较没有形成完整评价，不能声称RL收益。\n'
    text+=f"\n累计GPU小时：{values['budget']['combined_GPU_hours']:.6f}/3；test像素读取：{f.get('test_pixels_read','NA')}。\n\n"
    text+='候选由冻结检测器产生，GT只用于训练与评价；策略是冻结Qwen视觉特征上的小型有限动作actor，非自回归语言模型RL。拒答基于标注支持，不等同临床安全；引用区域只是结构约束，不证明因果忠实性。单seed、已多轮使用的开发集，不证明独立泛化。权重、GT、逐例记录与特征保持私有。\n'
    (public/'medevidence_p7_decision.md').write_text(text);print(json.dumps({'report_status':f['status']}),flush=True)
