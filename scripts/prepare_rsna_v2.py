"""CPU-only v2 preflight and irreversible protocol freeze; never starts a GPU worker."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from src.data import sha,write_json,QUESTIONS
from src.regions import region_tokens
from src.v2 import (require_safe_config,select_balanced,classification_schedule,exposure_summary,folds_for,
                    mismatch_permutations,DevelopmentData,save,STAGES)


def region_review(frame,regions):
    fixed={};details=[]
    for row in frame.to_dict('records'):
        region=regions.get(row['image_id'],{})
        if not region.get('eligible'):continue
        r=json.loads(json.dumps(region));reasons=[];width=float(row['width']);height=float(row['height']);gh,gw=r['grid']
        boxes=json.loads(row['boxes']);forbidden=set()
        for x1,y1,x2,y2 in boxes:
            expanded=[max(0,x1-.1*(x2-x1)),max(0,y1-.1*(y2-y1)),min(width,x2+.1*(x2-x1)),min(height,y2+.1*(y2-y1))]
            forbidden.update(region_tokens(expanded,width,height,gh,gw))
        e=r['evidence'];eb=e['box'];area=(eb[2]-eb[0])*(eb[3]-eb[1])
        if len(r['controls'])!=3:reasons.append('not_three_controls')
        for control in r['controls']:
            b=control['box'];a=(b[2]-b[0])*(b[3]-b[1])
            if not np.isclose(a,area,rtol=1e-6) or not np.allclose([b[2]-b[0],b[3]-b[1]],[eb[2]-eb[0],eb[3]-eb[1]],rtol=1e-6):reasons.append('area_or_shape_mismatch')
            if len(control['tokens'])!=len(e['tokens']):reasons.append('token_count_mismatch')
            if set(control['tokens'])&forbidden:reasons.append('control_intersects_annotation_margin')
        for item in [e,*r['controls'],r['wrong']]:
            if len(item['source'])<4 or set(item['source'])&set(item['tokens']):reasons.append('replacement_source_invalid')
            if set(item['source'])&forbidden:reasons.append('replacement_source_intersects_annotation_margin')
            if set(region_tokens(item['box'],width,height,gh,gw))!=set(item['tokens']):reasons.append('box_token_mapping_mismatch')
        wrong=set(r['wrong']['tokens']);evidence=set(e['tokens']);overlap=len(wrong&evidence)/len(wrong|evidence)
        r.update(eligible=not reasons,clinical_review='pending',geometry_review='passed' if not reasons else 'failed',lung_coverage=None,
                 exclusion_reasons=sorted(set(reasons)),wrong_evidence_token_iou=overlap)
        fixed[row['image_id']]=r
        details.append({'image_id':row['image_id'],'case_id':row['case_id'],'split':row['split'],'geometry_eligible':not reasons,
                        'reasons':sorted(set(reasons)),'clinical_review':'pending','lung_coverage':None,'wrong_evidence_token_iou':overlap})
    return fixed,details


def main():
    cfgpath=Path(os.environ['RUN_CONFIG']);cfg=json.loads(cfgpath.read_text());require_safe_config(cfg)
    old=Path(os.environ['V1_OUTPUT_ROOT']);root=Path(os.environ['OUTPUT_ROOT']);protocol=root/'protocol'
    if (protocol/'protocol_lock.json').exists():raise FileExistsError('v2 protocol already frozen; never overwrite it')
    protocol.mkdir(parents=True,exist_ok=True)
    lock=json.loads((old/'protocol/protocol_lock.json').read_text());old_cfg=old.parent/'code/configs/rsna_pilot.json'
    for name,digest in lock['hashes'].items():
        if sha((old/'protocol'/name).read_bytes())!=digest:raise ValueError('v1 protocol identity mismatch: '+name)
    if sha(old_cfg.read_bytes())!=lock['config_sha256']:raise ValueError('v1 config mismatch')
    full=pd.read_csv(old/'protocol/manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str,'duplicate_group':str})
    if full.groupby('case_id').partition.nunique().max()!=1:raise ValueError('Patient partition leakage')
    frame=full[full.split.isin(['train','validation'])].copy().sort_values('image_id')
    if frame.case_id.duplicated().any():raise ValueError('Predetermined patient image changed')
    train=frame[frame.split=='train'];val=frame[frame.split=='validation'];diff=[]
    observed={'train':len(train),'validation':len(val),'train_positive':int((train.pathology=='yes').sum()),'validation_positive':int((val.pathology=='yes').sum())}
    expected={'train':1024,'validation':256,'train_positive':149,'validation_positive':37}
    for key,want in expected.items():
        diff.append({'item':key,'expected':want,'observed':observed[key],'matches':want==observed[key]})
    old_regions=json.loads((old/'protocol/regions.json').read_text())
    for name,subset,want in [('eligible_train',train,69),('eligible_validation',val,24)]:
        actual=sum(old_regions.get(k,{}).get('eligible',False) for k in subset.image_id)
        diff.append({'item':name,'expected':want,'observed':actual,'matches':want==actual})
    selected=json.loads((old/'B1/selected.json').read_text())
    if selected['checkpoint']!='step_0256' or not (old/'B1'/selected['checkpoint']/'adapter_model.safetensors').is_file():raise ValueError('Historical adapter identity mismatch')
    if not all(x['matches'] for x in diff):
        save(root/'preflight_differences.json',diff);raise ValueError('Resolve baseline differences before GPU execution')
    data=DevelopmentData(os.environ['DATA_ROOT'],frame)
    # Stat availability only; no test/nonselected image can enter this list.
    if any(not p.is_file() for p in data.allowed):raise FileNotFoundError('Missing selected development image or mask')
    regions,details=region_review(frame,old_regions)
    frame.to_csv(protocol/'manifest.csv',index=False)
    write_json(protocol/'regions.json',regions);write_json(protocol/'region_review.json',details)
    write_json(protocol/'templates.json',QUESTIONS)
    folds=folds_for(val,cfg['fold_seed']);write_json(protocol/'folds.json',folds)
    p1=select_balanced(train,16,16,cfg['seed']);diagnostic=select_balanced(val,32,32,cfg['seed'])
    subsets={'P1':p1,'P0':diagnostic,'mismatch':mismatch_permutations(diagnostic,val,3,cfg['seed'])}
    pool=train[train.image_id.map(lambda k:regions.get(k,{}).get('eligible',False))]
    ineligible=train[(train.pathology=='yes') & ~train.image_id.isin(pool.image_id)]
    p4=[]
    for name,subset,n in [('geometry_only_positive',pool,32),('ineligible_positive',ineligible,16),('negative',train[train.pathology=='no'],16)]:
        ids=subset.sort_values('case_id').sample(n=min(n,len(subset)),random_state=cfg['seed']).image_id.tolist()
        p4.extend({'image_id':k,'stratum':name,'semantic_eligible':False,'sampling_seeds':[cfg['seed']+100000+len(p4)*100+i*8+j for i in range(4) for j in range(8)]} for k in ids)
    # Each case receives its own independent, immutable group/slot seeds.
    for i,row in enumerate(p4):row['sampling_seeds']=[cfg['seed']+100000+i*100+j for j in range(32)]
    subsets['P4']=p4;write_json(protocol/'subsets.json',subsets)
    schedules={}
    for name,pool,steps,balanced in [('P1',train[train.image_id.isin(p1)],256,True),('SFT-N',train,512,False),('SFT-B',train,512,True)]:
        plan=classification_schedule(pool,steps,cfg['seed'],balanced);write_json(protocol/(name+'_schedule.json'),plan)
        schedules[name]=exposure_summary(plan,frame,regions)
    if schedules['SFT-N']['label_exposures']!={'yes':298,'no':1750}:raise ValueError('Natural-distribution schedule mismatch')
    if schedules['SFT-B']['label_exposures']!={'yes':1024,'no':1024}:raise ValueError('Balanced schedule mismatch')
    model=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct';metadata=model/'.cache/huggingface/download'
    receipts={p.name:p.read_text().splitlines()[0] for p in metadata.glob('*.metadata')}
    if not receipts or set(receipts.values())!={cfg['revision']}:raise ValueError('Model download revision receipts disagree')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    ids={label:tokenizer.encode(label,add_special_tokens=False) for label in ['no','yes']}
    if any(len(v)!=1 for v in ids.values()):raise ValueError('Expected historical single-token labels')
    adapter=old/'B1'/selected['checkpoint'];adapter_identity={'checkpoint':selected['checkpoint'],'sha256':sha((adapter/'adapter_model.safetensors').read_bytes()),'config_sha256':sha((adapter/'adapter_config.json').read_bytes())}
    # Explicit v2 identity request: hash this small adapter, not the multi-GB base shards.
    base_identity={'revision':cfg['revision'],'receipts':receipts,'files':{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in model.glob('*.safetensors')}}
    write_json(protocol/'identity.json',{'base':base_identity,'historical_B1':adapter_identity,'label_ids':ids,'eos_id':tokenizer.eos_token_id,'score_eos':False,'train_eos':True})
    geometry={}
    from collections import Counter
    for split in ['train','validation']:
        rows=[r for r in details if r['split']==split];geometry[split]={'candidates':len(rows),'geometry_eligible':sum(r['geometry_eligible'] for r in rows),'semantic_approved':0,'clinical_review_pending':len(rows),'exclusions':dict(Counter(reason for r in rows for reason in r['reasons']))}
    review=root/'review';review.mkdir(exist_ok=True)
    # Blind reviewer packet: no model scores, predictions, or outcome-dependent selection.
    for split in ['train','validation']:
        rows=[r for r in details if r['split']==split]
        for page in range((len(rows)+11)//12):
            canvas=Image.new('RGB',(1280,1008),'white')
            for i,entry in enumerate(rows[page*12:(page+1)*12]):
                row=data.rows[entry['image_id']];im=data.image(entry['image_id']);im.thumbnail((320,320));draw=ImageDraw.Draw(im);r=regions[entry['image_id']]
                for item,color in [(r['evidence'],'red'),*[(c,'lime') for c in r['controls']],(r['wrong'],'blue')]:
                    b=item['box'];draw.rectangle([b[0]*im.width/float(row['width']),b[1]*im.height/float(row['height']),b[2]*im.width/float(row['width']),b[3]*im.height/float(row['height'])],outline=color,width=2)
                canvas.paste(im,((i%4)*320,(i//4)*336));ImageDraw.Draw(canvas).text(((i%4)*320,(i//4)*336+320),entry['image_id'],fill='black')
            canvas.save(review/(split+'_'+str(page+1)+'.jpg'))
    pd.DataFrame([{'image_id':r['image_id'],'split':r['split'],'reviewer_qualification':'','reviewer_blinded_to_scores':'','controls_semantically_valid':'','replacement_sources_valid':'','notes':''} for r in details]).to_csv(review/'clinical_review_pending.csv',index=False)
    source=Path(__file__).resolve().parents[1]
    source_hashes={str(p.relative_to(source)):sha(p.read_bytes()) for directory in ['src','scripts','tests'] for p in (source/directory).glob('*.py')}
    write_json(protocol/'protocol_lock.json',{'version':cfg['protocol_version'],'config_sha256':sha(cfgpath.read_bytes()),'hashes':{p.name:sha(p.read_bytes()) for p in protocol.iterdir() if p.is_file()},'source_hashes':source_hashes,'origin_HEAD':os.environ.get('V2_ORIGIN_HEAD'),'v1_config_sha256':lock['config_sha256'],'test_access':'denied by whitelist and filesystem audit hook','all_regions_frozen_before_v2_model_scoring':True})
    preflight={'status':'passed','differences':diff,'schedule_exposures':schedules,'geometry':geometry,'label_ids':ids,'eos_id':tokenizer.eos_token_id,'model_revision':cfg['revision'],'v1_checkpoint':selected['checkpoint'],'v1_selected_validation_auroc':selected['auroc'],'test_images_read':0,'clinical_review':'pending','python':platform.python_version()}
    write_json(root/'preflight.json',preflight)
    write_json(root/'budget.json',{'limit_seconds':cfg['gpu_hours_limit']*3600,'consumed_seconds':0.,'attempts':[],'accounting':'sum of independent GPU-worker wall times, including load/evaluation/failures; <=2 concurrent GPUs, single GPU per trial; CPU-only aggregation excluded'})
    write_json(root/'stage_status.json',{s:{'status':'ready' if s in ['P0','P1','P3','P4'] else 'blocked','reason':('requires_P1_pass' if s in ['SFT-N','SFT-B','P2-summary'] else cfg['R0']['reason'] if s=='R0' else 'protocol_frozen_pending_GPU_authorization'),'actual_samples':0,'steps':0,'runtime_seconds':0,'output':s+'/'} for s in STAGES})
    print(json.dumps(preflight),flush=True)

if __name__=='__main__':main()
