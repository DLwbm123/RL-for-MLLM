"""Engineering-only P9 continuation: durable feature chunks, unchanged scientific stages."""
import os
from pathlib import Path
import numpy as np
import torch
from src.medevidence_p7 import read,worker as previous_worker
from src.medevidence_p9 import Run as OriginalRun,report as original_report,input_features,logits,calibrated_probabilities,decisions
from src.v2 import save


class FeatureChunks:
    def __init__(self,path,ids,identity,chunk_size=32):
        self.path=Path(path);self.path.mkdir(exist_ok=True);self.ids=list(ids);self.identity=identity;self.chunk_size=chunk_size
        assert self.ids and len(set(self.ids))==len(self.ids) and chunk_size>0
        self.blocks=[self.ids[i:i+chunk_size] for i in range(0,len(ids),chunk_size)]
        self.block_of={k:i for i,block in enumerate(self.blocks) for k in block};self.records={};self.dirty=set();self.committed=set()
        meta={'ids':self.ids,'identity':identity,'chunk_size':chunk_size};p=self.path/'manifest.json'
        if p.exists():
            if read(p)!=meta:raise ValueError('Feature cache provenance or patient set changed')
        else:
            if list(self.path.glob('chunk_*.pt')):raise ValueError('Chunks exist without their manifest')
            save(p,meta)
        for i,block in enumerate(self.blocks):
            p=self.path/f'chunk_{i:04d}.pt'
            if not p.exists():continue
            payload=torch.load(p,map_location='cpu',weights_only=True)
            if payload['identity']!=identity:raise ValueError('Chunk provenance mismatch')
            if set(payload['records'])-set(block) or payload['ids']!=[k for k in block if k in payload['records']]:raise ValueError('Chunk patient membership mismatch')
            for k,record in payload['records'].items():self.validate(record);self.records[k]=record;self.committed.add(k)
        self.state()

    @property
    def persisted(self):return len(self.committed)

    def validate(self,record):
        if set(record)!={'x','valid','rewards','good'}:raise ValueError('Unexpected feature fields')
        if record['x'].shape!=(10,self.identity['feature_dimension']) or record['x'].dtype!=torch.float16:raise ValueError('Feature layout changed')
        for key in ['valid','rewards','good']:
            if record[key].shape!=(10,):raise ValueError('Action layout changed')
        if record['valid'].dtype!=torch.bool or record['good'].dtype!=torch.bool:raise ValueError('Invalid masks')
        if not torch.isfinite(record['x']).all() or not torch.isfinite(record['rewards']).all():raise ValueError('Nonfinite cached values')
        if not record['valid'][8:].all() or not record['good'].any() or (record['good']&~record['valid']).any():raise ValueError('Invalid action support')

    def add(self,k,record):
        if k not in self.block_of or k in self.records:raise ValueError('Unexpected or duplicate feature patient')
        self.validate(record);self.records[k]=record;i=self.block_of[k];self.dirty.add(i)
        if all(j in self.records for j in self.blocks[i]):self.flush()

    def flush(self):
        for i in sorted(self.dirty):
            ids=[k for k in self.blocks[i] if k in self.records];p=self.path/f'chunk_{i:04d}.pt';tmp=p.with_suffix('.pt.tmp')
            with tmp.open('wb') as stream:
                torch.save({'identity':self.identity,'ids':ids,'records':{k:self.records[k] for k in ids}},stream)
                stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,p);self.committed.update(ids)
        self.dirty.clear();self.state()

    def state(self):
        save(self.path.parent/'cache_state.json',{'schema':'p9-features-v1','patients_planned':len(self.ids),'patients_persisted':self.persisted,
            'chunk_size':self.chunk_size,'complete':self.persisted==len(self.ids),'atomic_chunk_writes':True})


def file_receipt(path):
    path=Path(path);s=path.stat()
    return {'path':str(path.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}


def feature_identity(base,model_root):
    from src.model import REVISION
    old=Path(base).parent/'medevidence_p9';model=Path(model_root)/'Qwen2.5-VL-7B-Instruct'
    index=read(model/'model.safetensors.index.json');names=sorted(set(index['weight_map'].values()))
    return {'schema':'p9-features-v1','feature_dimension':7176,'candidate_source_commit':read(old/'outputs/authorization.json')['source_commit'],
        'candidates':file_receipt(old/'outputs/candidates.json'),'patient_manifest':file_receipt(old/'outputs/protocol/manifest.csv'),
        'visual_model_revision':REVISION,'model_files':[file_receipt(model/n) for n in names+['config.json','preprocessor_config.json','model.safetensors.index.json']],
        'preprocessing':{'min_tokens':576,'max_tokens':1024,'model_dtype':'bfloat16','cache_dtype':'float16',
            'pooling':'P7 global and candidate region mean with separate layer norm','feature_contract_source':'5672cd293a7e4eaba97b0673f814b1328e3c321e'}}


def diagnose(records,ids,p,cutoff,weights=None):
    from sklearn.metrics import roc_auc_score,average_precision_score
    w=np.ones(len(ids)) if weights is None else np.asarray(weights)
    rewards=torch.stack([records[k]['rewards'] for k in ids]).numpy();has=np.array([bool(records[k]['valid'][0]) for k in ids])
    y=np.stack([rewards[:,0]==1,rewards[:,8]==1],-1);action,confidence=decisions(p,has)
    actions={'utility':action}
    if cutoff is not None:actions['calibrated_threshold']=np.where(confidence>=cutoff,action,9)
    groups={'negative':y[:,1],'top1_supported_positive':y[:,0],
        'other_candidate_supported_positive':~y[:,1]&~y[:,0]&(rewards[:,:8]==1).any(-1),
        'no_supported_candidate_positive':~y[:,1]&~(rewards[:,:8]==1).any(-1)}
    summary={}
    for name,mask in groups.items():
        summary[name]={'patients':int(mask.sum()),'rules':{rule:{'yes':int((mask&(a==0)).sum()),'no':int((mask&(a==8)).sum()),
            'abstain':int((mask&(a==9)).sum()),'correct':int((mask&(rewards[np.arange(len(ids)),a]==1)).sum())} for rule,a in actions.items()}}
    excess=np.maximum(p.sum(-1)-1,0);heads={}
    for j,name in enumerate(['supported_yes','correct_no']):
        bins=[]
        for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            mask=(p[:,j]>=lo)&(p[:,j]<(hi if hi<1 else 1.0000001));mass=w[mask].sum()
            bins.append({'lower':float(lo),'upper':float(hi),'patients':int(mask.sum()),
                'mean_probability':float(w[mask]@p[mask,j]/mass) if mass else None,
                'observed_correct_fraction':float(w[mask]@y[mask,j]/mass) if mass else None})
        both=len(np.unique(y[:,j]))==2
        heads[name]={'brier':float(w@((p[:,j]-y[:,j])**2)/w.sum()),'auroc':float(roc_auc_score(y[:,j],p[:,j],sample_weight=w)) if both else None,
            'average_precision':float(average_precision_score(y[:,j],p[:,j],sample_weight=w)) if both else None,'reliability_bins':bins}
    return {'patients':len(ids),'groups':summary,'probability_conflict':{'tolerance':1e-6,'count':int((excess>1e-6).sum()),
        'fraction':float((excess>1e-6).mean()),'mean_excess':float(excess.mean()),'max_excess':float(excess.max())},'heads':heads}


class Run(OriginalRun):
    def __init__(self):super().__init__('medevidence_p9_resume.json')

    def execute(self):
        old=self.base.parent/'medevidence_p9/outputs'
        assert read(old/'FINAL.json')['status']=='stopped_budget' and not read(old/'gpu_ledger.json')['active']
        self.candidates=read(old/'candidates.json');ids=self.sets['train']+self.sets['calibration']
        assert set(self.candidates)==set(ids)
        identity=feature_identity(self.base,os.environ['MODEL_ROOT'])
        if identity!=read(self.out/'protocol/feature_identity.json'):raise ValueError('Sources changed after continuation preparation')
        cache=FeatureChunks(self.out/'feature_chunks',ids,identity,self.cfg['feature_cache_chunk_size'])
        self.update('resume_frozen_visual_features',reused_detector_steps=2400,new_detector_steps=0,reused_candidate_patients=960,
            persisted_features=cache.persisted)
        features=self.features(ids,cache=cache)
        assert cache.persisted==len(ids)
        return self.finish(features)

    def finish(self,features):
        reused=read(self.out/'protocol/reused_development_sources.json')
        for name,receipt in reused.items():
            if file_receipt(self.base.parent/'medevidence_p8/outputs'/name)!=receipt:raise ValueError('Frozen development source changed')
        self.diagnostic_features=features
        return super().finish(features)

    def evaluate_heads(self,model,shift,features,cal):
        super().evaluate_heads(model,shift,features,cal)
        result={'scope':'Explanatory diagnostics only; frozen main decision and predictions are unchanged','splits':{}}
        for split in ['train','calibration','dev']:
            ids=self.sets[split];records=features if split=='dev' else self.diagnostic_features
            x,y,has=input_features(records,ids);z=logits(model,x);weights=None
            if split=='calibration':
                positive=~y[:,1].astype(bool);prior=cal['target_positive_prior']
                weights=np.where(positive,prior/positive.mean(),(1-prior)/(1-positive.mean()))
            result['splits'][split]={name:diagnose(records,ids,calibrated_probabilities(z,s,has),cal['correctness_reject_threshold'] if name=='calibrated' else None,weights)
                for name,s in [('raw',np.zeros(2)),('calibrated',shift)]}
        save(self.out/'diagnostics.json',result)
        del self.diagnostic_features


def worker():previous_worker(Run)


def report():original_report('medevidence_p9_resume')
