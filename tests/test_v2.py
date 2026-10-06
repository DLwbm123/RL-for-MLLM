import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
import pandas as pd
import pytest
import torch
from src.v2 import (classification_schedule,exposure_summary,folds_for,mismatch_permutations,DevelopmentData,
                    specificity_threshold,crossfit,binary_stats,paired_bootstrap,reward_group,require_stage,require_safe_config)


def frame(npos=149,nneg=875):
    return pd.DataFrame([{'image_id':f'i{i:04d}','case_id':f'p{i:04d}','pathology':'yes' if i<npos else 'no',
                          'split':'train','bbox_xyxy':'[1,2,3,4]' if i<npos else 'null','mask_components':1 if i<npos else 0}
                         for i in range(npos+nneg)])


def test_schedules_exact_exposures_and_no_localization_replacement():
    f=frame();natural=classification_schedule(f,512,17,False);balanced=classification_schedule(f,512,17,True)
    n=exposure_summary(natural,f,{});b=exposure_summary(balanced,f,{})
    assert n['label_exposures']=={'yes':298,'no':1750} and n['unique_patients']==1024
    assert b['label_exposures']=={'yes':1024,'no':1024} and b['repeat_exposures']==1024
    lookup=f.set_index('image_id').pathology
    for i in range(0,len(balanced),4):assert sorted(lookup[x['image_id']] for x in balanced[i:i+4])==['no','no','yes','yes']
    for epoch in [natural[:1024],natural[1024:]]:assert len({x['image_id'] for x in epoch})==1024
    from src.experiment import schedule
    only_negative=f[f.pathology=='no'].iloc[:8]
    legacy=schedule(only_negative,2,4,17,task_pattern=['A'])
    assert {r['task'] for r in legacy}=={'A'} and len({r['image_id'] for r in legacy})==8


def test_patient_folds_and_fixed_mismatches():
    f=frame(32,32);f['split']='validation';folds=folds_for(f,42)
    assert folds==folds_for(f.sample(frac=1,random_state=3),42)
    perms=mismatch_permutations(f.image_id.tolist(),f,3,17)
    assert len({tuple(p) for p in perms})==3
    assert all(a!=b for p in perms for a,b in zip(f.image_id,p))


def test_sealed_loader_blocks_ids_and_actual_file_open(tmp_path):
    from PIL import Image
    root=tmp_path/'data';root.mkdir();Image.new('RGB',(2,2)).save(root/'ok.png');Image.new('L',(2,2)).save(root/'mask.png');(root/'test.png').write_bytes(b'sealed')
    f=pd.DataFrame([{'image_id':'ok','case_id':'p','split':'train','image_path':'ok.png','mask_path':'mask.png'}])
    data=DevelopmentData(root,f)
    assert data.image('ok').size==(2,2)
    with pytest.raises(PermissionError):data.image('test')
    bad=f.copy();bad['split']='test'
    with pytest.raises(PermissionError):DevelopmentData(root,bad)
    code="""import pandas as pd
from src.v2 import DevelopmentData
from pathlib import Path
root=Path(ROOT)
f=pd.DataFrame([{'image_id':'ok','case_id':'p','split':'train','image_path':'ok.png','mask_path':'mask.png'}])
d=DevelopmentData(root,f);d.install_guard();d.image('ok')
try: (root/'test.png').read_bytes()
except PermissionError: print('blocked')
else: raise AssertionError('unsealed read')
""".replace('ROOT',repr(str(root)))
    result=subprocess.run([sys.executable,'-u','-'],input=code,text=True,capture_output=True)
    assert result.returncode==0 and result.stdout.strip()=='blocked',result.stderr


def test_threshold_minimum_boundary_and_ties():
    y=np.array([0]*10+[1]*3);p=np.array([.1]*8+[.2,.2,.15,.25,.9]);w=np.array([1]*13)
    t=specificity_threshold(y,p,w);assert t==np.nextafter(.2,np.inf)
    candidates=np.unique(np.r_[0,p,np.nextafter(p,np.inf),np.nextafter(1.,np.inf)])
    feasible=[x for x in candidates if np.mean(p[y==0]<x)>=.9];assert t==min(feasible)
    # No finite p<=1 can be classified positive when all negative support is at 1.
    assert specificity_threshold([0,0],[1.,1.])>1
    assert specificity_threshold([1],[.5]) is None
    met=binary_stats([0,1],[.5,.5]);assert met['TP']==1 and met['FP']==1
    assert np.argmax([0.,0.])==0  # Historical argmax uses a different tie convention.


def test_crossfit_weighted_duplicates_remain_in_original_fold():
    f=frame(10,20);f['split']='validation';fm=folds_for(f,42);fold=np.array([fm[k] for k in f.image_id]);y=(f.pathology=='yes').astype(int).to_numpy();p=np.linspace(.05,.95,len(f));w=np.ones(len(f),int);w[0]=4;w[1]=0
    pred,thresholds=crossfit(y,p,fold,w)
    for i in range(5):
        idx=np.repeat(np.where(fold!=i)[0],w[fold!=i]);assert thresholds[i]==specificity_threshold(y[idx],p[idx])
    # Compare full weighted re-fit against literal duplicated patients retaining their fold.
    idx=np.repeat(np.arange(len(f)),w);expanded,other=crossfit(y[idx],p[idx],fold[idx]);assert thresholds==other
    assert np.array_equal(expanded,pred[idx])


def test_bootstrap_pairing_and_undefined_not_zero():
    f=frame(10,20);f['split']='validation';fold=folds_for(f,42)
    rows=[{'image_id':r.image_id,'case_id':r.case_id,'y':int(r.pathology=='yes'),'p':(i+.5)/len(f)} for i,r in enumerate(f.itertuples())]
    result=paired_bootstrap({'a':rows,'b':list(reversed(rows))},fold,20,42)
    interval=result['intervals']['b minus a / crossfit / sensitivity'];assert interval['ci95']==[0.,0.] and interval['valid_replicates']==20
    assert result['intervals']['a / crossfit_minus_raw / ap']['ci95']==[0.,0.]
    m=binary_stats([0,0],[.1,.2]);assert m['sensitivity'] is None and m['auroc'] is None and m['ap'] is None


def test_affine_binary_advantage_and_epsilon_branch():
    c=np.array([0,1,0,1,1,0,1,0],dtype=float)
    for a,b in [(2.,3.),(-1.,.001),(.1,.7)]:
        r=a+b*c
        assert np.max(np.abs((r-r.mean())/r.std()-(c-c.mean())/c.std()))<1e-12
        actual=reward_group(c,r);assert actual['usable'] and actual['delta_A']<1e-4
    same=reward_group(np.ones(8),np.ones(8));assert same['zero_advantage'] and same['near_zero_std']
    tiny=reward_group(c,1e-10*c);assert tiny['near_zero_std'] and tiny['delta_A']>.9
    reverse=reward_group(c,1-c);assert reverse['ordering_reversed'] and not reverse['usable']


def test_actual_generation_config_is_used():
    from src.model import Model
    from transformers import GenerationConfig
    model=Model.__new__(Model);captured={}
    class Fake:
        training=False
        generation_config=GenerationConfig(eos_token_id=3,pad_token_id=0)
        def generate(self,**kwargs):
            captured.update(kwargs['generation_config'].to_dict());return torch.tensor([[9,8,2,3]])
    model.model=Fake();model.supplied_features=lambda _:nullcontext();model.set_training=lambda flag:None
    model.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,skip_special_tokens:'yes'))
    model.generation_calls=0;model.generated_tokens=0
    cfg={'temperature':.7,'top_p':.8,'top_k':5,'max_new_tokens':7}
    text,tokens=model.generate({'input_ids':torch.tensor([[9,8]])},None,sampling_config=cfg)
    assert text=='yes' and tokens==[2,3]
    assert all(captured[k]==v and model.last_generation_settings[k]==v for k,v in cfg.items())


def test_frozen_guard_and_stage_dependencies():
    from src.v2_run import frozen
    net=torch.nn.Linear(2,2);net.requires_grad_(False);wrapper=SimpleNamespace(model=net)
    before={n:p.clone() for n,p in net.named_parameters()}
    with frozen(wrapper):
        with pytest.raises(PermissionError):torch.tensor(1.,requires_grad=True).backward()
        optimizer=torch.optim.AdamW(net.parameters())
        with pytest.raises(PermissionError):optimizer.step()
    assert wrapper.v2_identity_check['unchanged']
    assert all(torch.equal(before[n],p) for n,p in net.named_parameters())
    for stage in ['P4','P3','R0']:
        with pytest.raises(PermissionError):require_stage(stage,{},training=True)
    with pytest.raises(PermissionError):require_stage('SFT-N',{'P1':{'status':'failed','scientific_passed':False}},training=True)
    require_stage('SFT-B',{'P1':{'status':'completed','scientific_passed':True}},training=True)


@pytest.mark.parametrize('gpu_indices',['0','0,1'])
def test_supervisor_counts_failed_attempt_and_blocks_dependents(tmp_path,monkeypatch,gpu_indices):
    from scripts import rsna_v2_pipeline as pipeline
    cfg=json.loads(Path('configs/rsna_v2.json').read_text());(tmp_path/'config.json').write_text(json.dumps(cfg))
    (tmp_path/'auth.json').write_text(json.dumps({'authorized':True,'max_gpu_hours':6,'max_concurrent_gpus':2}))
    from src.v2 import STAGES
    statuses={s:{'status':'ready','reason':'test','actual_samples':0,'steps':0,'runtime_seconds':0,'output':s+'/'} for s in STAGES}
    (tmp_path/'stage_status.json').write_text(json.dumps(statuses));(tmp_path/'budget.json').write_text(json.dumps({'limit_seconds':21600.,'consumed_seconds':0.,'attempts':[]}))
    for k,v in {'OUTPUT_ROOT':str(tmp_path),'RUN_CONFIG':str(tmp_path/'config.json'),'V2_GPU_AUTH_FILE':str(tmp_path/'auth.json'),'CUDA_VISIBLE_DEVICES':'0','V2_GPU_INDICES':gpu_indices}.items():monkeypatch.setenv(k,v)
    class Worker:
        pid=123
        def __init__(self,*args,**kwargs):self.stdin=io.BytesIO()
        def wait(self,timeout):return 1
    monkeypatch.setattr(pipeline.subprocess,'Popen',Worker);monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *a,**k:'72000')
    pipeline.main();budget=json.loads((tmp_path/'budget.json').read_text());state=json.loads((tmp_path/'stage_status.json').read_text())
    assert len(budget['attempts'])==len(gpu_indices.split(',')) and all(a['exit_code']==1 for a in budget['attempts']) and budget['consumed_seconds']>0
    assert budget['consumed_seconds']==sum(a['gpu_seconds'] for a in budget['attempts'])
    assert state['P0']['status']=='failed' and state['P1']['status']=='blocked' and state['SFT-N']['status']=='blocked'
    with pytest.raises(FileExistsError):pipeline.main()


def test_p3_review_rejects_bad_geometry_without_claiming_clinical_review():
    from scripts.prepare_rsna_v2 import region_review
    from src.regions import build_regions
    from PIL import Image
    im=Image.fromarray(np.random.default_rng(1).integers(25,180,(300,400),dtype=np.uint8)).convert('RGB');mask=np.zeros((300,400),bool);mask[120:145,175:205]=True
    region=build_regions(im,mask,[175,120,205,145],chest=True)
    f=pd.DataFrame([{'image_id':'a','case_id':'p','split':'train','width':400,'height':300,'boxes':'[[175,120,205,145]]'}])
    region['controls'][0]=region['evidence'].copy()
    fixed,rows=region_review(f,{'a':region})
    assert not fixed['a']['eligible'] and 'control_intersects_annotation_margin' in rows[0]['reasons']
    assert fixed['a']['clinical_review']=='pending' and fixed['a']['lung_coverage'] is None
