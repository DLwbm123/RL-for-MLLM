"""Recover selected original annotations, construct score-blind views, then freeze once."""
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from src.medevidence import normalized, crop_box, map_box, control_box, intervention, schedules, exposures
from src.v2 import DevelopmentData, save


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path(os.environ['OUTPUT_ROOT']);data_root=Path(os.environ['DATA_ROOT'])
    cfgpath=Path(os.environ['RUN_CONFIG']);cfg=json.loads(cfgpath.read_text());p=root/'protocol';p.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).resolve().parents[1]
    if (p/'lock.json').exists(): raise FileExistsError('Already frozen')
    if os.environ.get('PREPARE_ACTION')=='freeze':
        views=json.loads((p/'views_draft.json').read_text());review=json.loads((p/'AI_review.json').read_text())
        reviews=review['cases'];proposals=views.pop('positive_proposals')
        if set(reviews)!=set(proposals): raise ValueError('Every positive proposal needs one blinded review')
        views['pairs'].update({k:v for k,v in proposals.items() if reviews[k]['decision']=='accept'})
        crop_exclusions={k:r['crop_reason'] for k,r in reviews.items() if not r.get('crop_usable',True)}
        for k in crop_exclusions:views['crops'].pop(k,None)
        views['review_level']='AI-assisted annotation-guided exploratory; no clinical expert review'
        save(p/'views.json',views)
        frame=pd.read_csv(p/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
        rows=frame.set_index('image_id').to_dict('index')
        for r in rows.values():r['boxes']=json.loads(r['boxes'])
        # Match negative crops to the actually usable positive geometry distribution before any scoring.
        for split in ('train','validation'):
            templates=[(k,v) for k in sorted(views['crops']) if rows[k]['split']==split and rows[k]['pathology']=='yes' for v in views['crops'][k]]
            negatives=sorted(k for k,r in rows.items() if r['split']==split and r['pathology']=='no')
            for i,k in enumerate(negatives):
                donor,v=templates[i%len(templates)];src=rows[donor];r=rows[k]
                views['crops'][k]=[{'box':map_box(v['box'],(src['width'],src['height']),(r['width'],r['height'])),
                    'evaluation_box':map_box(crop_box(v['target_box'],src['width'],src['height'],2.5),(src['width'],src['height']),(r['width'],r['height'])),
                    'matched_positive':donor,'target_box_index':None,'source_same_split':True,'different_patient':True}]
        save(p/'views.json',views)
        plan=schedules(rows,views,512,cfg['seed']);save(p/'schedule.json',plan)
        exposure={method:exposures(plan[256:] if method in ('M1','M2') else plan[:cfg['steps'][method]],rows,views,method) for method in ('M0','W','M1','M2')}
        save(p/'planned_exposures.json',exposure)
        audit=json.loads((p/'data_audit.json').read_text())
        audit.update(positive_pairs_accepted={s:sum(rows[k]['split']==s for k in views['pairs'] if views['pairs'][k]['positive']) for s in ('train','validation')},
                     AI_exclusions=dict(Counter(r['reason'] for r in reviews.values() if r['decision']!='accept')),clinical_expert_review=False,
                     crop_exclusions=dict(Counter(crop_exclusions.values())),crop_excluded_patients=len(crop_exclusions))
        distribution={}
        for split in ('train','validation'):
            for label in ('yes','no'):
                values=[]
                for k,vv in views['crops'].items():
                    r=rows[k]
                    if r['split']!=split or r['pathology']!=label:continue
                    for v in vv:
                        a,b,c,d=v['box'];values.append([a/r['width'],b/r['height'],(c-a)/r['width'],(d-b)/r['height'],(c-a)/(d-b)])
                distribution[split+'_'+label]={'n_geometries':len(values),'fields':['x','y','width','height','aspect'],
                    'quantiles_0_25_50_75_100':np.quantile(values,[0,.25,.5,.75,1],axis=0).tolist() if values else None}
        audit['crop_geometry_distributions']=distribution
        audit['crop_patients']={s:sum(rows[k]['split']==s for k in views['crops']) for s in ('train','validation')}
        save(p/'data_audit.json',audit)
        public=code/'protocol/medevidence_p1';public.mkdir(parents=True,exist_ok=True)
        save(public/'implementation_resolution.json',{'config':cfg,'localization_lengths':json.loads((p/'token_contract.json').read_text()),
              'primary':'M2-M1 Delta_S on all frozen evaluable development positives; strict joint raw0.5',
              'schedule':'512 full balanced updates; W first256, M1/M2 identical last256',
              'budget':'smoke counted; protect complete matched branches plus evaluation before M0 if throughput shows insufficiency; fixed quotas, no branch shortening to claim completion',
              'AI_review':'Current Codex AI source-pixel screening; annotation-guided, no clinical expert; unknown rejected, one candidate per case, no score selection'})
        names=['manifest.csv','views.json','schedule.json','folds.json','identity.json','token_contract.json','AI_review.json']
        sources=['src/medevidence.py','src/medevidence_run.py','src/model.py','src/data.py','src/experiment.py','src/v2.py',
                 'src/objectives.py','src/regions.py','src/evaluation.py','scripts/medevidence_pipeline.py','scripts/report_medevidence_p1.py','tests/test_medevidence.py']
        lock={'config_sha256':digest(cfgpath),'hashes':{n:digest(p/n) for n in names},
              'source_hashes':{n:digest(code/n) for n in sources},'prompts_sha256':hashlib.sha256(json.dumps(cfg['prompts'],sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
              'seed':17,'test_pixels_read':0,'RL':False}
        save(p/'lock.json',lock)
        save(public/'freeze_summary.json',{'lock':lock,'data_audit':audit,'planned_exposures':{m:{t:{k:v for k,v in a.items() if k!='patient_repetitions'} for t,a in tasks.items()} for m,tasks in exposure.items()}})
        print(json.dumps({'status':'ready','audit':audit,'lock':lock}),flush=True);return
    old=Path(os.environ['OLD_PROTOCOL'])
    frame=pd.read_csv(old/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
    if set(frame.split)!={'train','validation'} or frame.case_id.duplicated().any():raise ValueError('Development identity/split error')
    selected=set(frame.sop);raw=json.loads((data_root/'pneumonia-challenge-annotations-adjudicated-kaggle_2018.json').read_text())
    annotations=defaultdict(list)
    for a in raw['datasets'][0]['annotations']:
        if a.get('SOPInstanceUID') in selected and a.get('labelId') in ('L_o8w','L_yd0','L_v8n'):annotations[a['SOPInstanceUID']].append(a)
    originals={};rows={}
    for row in frame.to_dict('records'):
        aa=annotations[row['sop']]
        if not aa or len({a['labelId'] for a in aa})!=1:raise ValueError('Missing/conflicting original annotations')
        recovered=[];raw_boxes=[]
        for a in aa:
            if a['labelId']!='L_v8n':continue
            if (a['width'],a['height'])!=(row['width'],row['height']):raise ValueError('Annotation dimension mismatch')
            d=a['data'];x,y,w,h=[d[k] for k in ('x','y','width','height')]
            if w<=0 or h<=0:raise ValueError('Invalid original xywh')
            b=[max(0,math.floor(x)),max(0,math.floor(y)),min(a['width'],math.ceil(x+w)),min(a['height'],math.ceil(y+h))]
            recovered.append(b);raw_boxes.append({'xywh':[x,y,w,h],'xyxy':b})
        if sorted(recovered)!=sorted(json.loads(row['boxes'])):raise ValueError('Independent source boxes differ from frozen manifest')
        if bool(recovered)!=(row['pathology']=='yes'):raise ValueError('Label/box mismatch')
        row['boxes']=sorted(recovered);row['loc_target']=json.dumps(normalized(recovered,row['width'],row['height']),separators=(',',':'))
        rows[row['image_id']]=row;originals[row['image_id']]=raw_boxes
    save(p/'original_boxes.json',originals)
    records=[]
    for row in rows.values():
        copy=dict(row);copy['boxes']=json.dumps(row['boxes']);records.append(copy)
    frame=pd.DataFrame(records);frame.to_csv(p/'manifest.csv',index=False)
    save(p/'folds.json',json.loads((old/'folds.json').read_text()))
    identity=json.loads((old/'identity.json').read_text());base=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    for name,info in identity['base']['files'].items():
        st=(base/name).stat()
        if st.st_size!=info['bytes'] or st.st_mtime_ns!=info['mtime_ns']:raise ValueError('Local base model changed')
    save(p/'identity.json',{'base':identity['base'],'label_ids':identity['label_ids'],'eos_id':identity['eos_id'],'revision':cfg['revision']})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    lengths=[len(tokenizer.encode(r['loc_target'],add_special_tokens=False))+1 for r in rows.values() if r['split']=='train']
    maximum=max(64,math.ceil((max(lengths)+16)/16)*16)
    save(p/'token_contract.json',{'max_new_tokens':maximum,'max_training_target_including_eos':max(lengths),'training_lengths':dict(Counter(lengths)),
                                 'truncation':'generated last token is not EOS at cap; retain failure and never enlarge cap','parser':'strict JSON list <=64 finite ordered-valid xyxy boxes, no fences'})
    data=DevelopmentData(data_root,frame);data.install_guard()
    views={'crops':{},'pairs':{},'positive_proposals':{}};rng=np.random.default_rng(42);geometry_reasons=Counter()
    # Read selected pixels only. Positive boxes remain separate; no mask or annotated display enters model inputs.
    for k,r in rows.items():
        im=data.image(k)
        if im.size!=(r['width'],r['height']):raise ValueError('Image dimensions differ from annotations')
        if r['boxes']:
            views['crops'][k]=[]
            for index,b in enumerate(r['boxes']):
                crop=crop_box(b,*im.size,2.)
                views['crops'][k].append({'box':crop,'target_box_index':index,'target_box':b,
                    'scale_xy':[(crop[2]-crop[0])/(b[2]-b[0]),(crop[3]-crop[1])/(b[3]-b[1])],
                    'target_fraction':(b[2]-b[0])*(b[3]-b[1])/((crop[2]-crop[0])*(crop[3]-crop[1])),
                    'source_to_crop_offset':[-crop[0],-crop[1]]})
    for split in ('train','validation'):
        positive=sorted(k for k,r in rows.items() if r['split']==split and r['pathology']=='yes')
        ordered=list(rng.permutation(positive));accepted=0
        for k in ordered:
            r=rows[k];boxes=r['boxes'];rule=cfg['dep_applicability']
            if len(boxes)!=1:geometry_reasons['multiple_boxes']+=1;continue
            target=boxes[0];area=(target[2]-target[0])*(target[3]-target[1])/(r['width']*r['height'])
            if not rule['min_target_area_fraction']<=area<=rule['max_target_area_fraction'] or min(target[2]-target[0],target[3]-target[1])<rule['min_target_side_pixels']:
                geometry_reasons['nonlocal_or_too_small']+=1;continue
            image=data.image(k);control,index=control_box(image,target,cfg['views'],rng)
            if control is None:geometry_reasons['no_matched_control']+=1;continue
            if split=='train' and accepted>=cfg['train_evidence_candidate_limit']:continue
            views['positive_proposals'][k]={'positive':True,'target':target,'control':control,'candidate_index':index,'split':split,'source_same_patient':True}
            accepted+=1
        negative=sorted(k for k,r in rows.items() if r['split']==split and r['pathology']=='no')
        crop_templates=[(k,v) for k in positive for v in views['crops'][k]]
        for i,k in enumerate(negative):
            donor,v=crop_templates[i%len(crop_templates)];src=rows[donor];r=rows[k]
            views['crops'][k]=[{'box':map_box(v['box'],(src['width'],src['height']),(r['width'],r['height'])),
                'matched_positive':donor,'target_box_index':None,'source_same_split':True,'different_patient':True}]
        templates=[(k,v) for k,v in views['positive_proposals'].items() if v['split']==split]
        for i,k in enumerate(list(rng.permutation(negative))[:cfg['negative_pair_limit']]):
            if not templates:break
            donor,v=templates[i%len(templates)];src=rows[donor];r=rows[k]
            a=map_box(v['target'],(src['width'],src['height']),(r['width'],r['height']))
            b=map_box(v['control'],(src['width'],src['height']),(r['width'],r['height']))
            if (a[2]-a[0],a[3]-a[1])!=(b[2]-b[0],b[3]-b[1]):raise ValueError('Negative pair shape mismatch')
            views['pairs'][k]={'positive':False,'target':a,'control':b,'split':split,'matched_positive_geometry':donor}
    save(p/'views_draft.json',views)
    review_dir=root/'review';review_dir.mkdir(exist_ok=True);index=[]
    proposals=list(views['positive_proposals'].items())
    for page in range(math.ceil(len(proposals)/3)):
        canvas=Image.new('RGB',(1600,1020),'white')
        for j,(k,v) in enumerate(proposals[page*3:page*3+3]):
            im=data.image(k);marked=im.copy();draw=ImageDraw.Draw(marked)
            draw.rectangle(v['target'],outline='red',width=5);draw.rectangle(v['control'],outline='lime',width=5)
            panels=[im,marked,intervention(im,v['control'],'gray',cfg['views']),intervention(im,v['target'],'gray',cfg['views']),im.crop(views['crops'][k][0]['box'])]
            for column,panel in enumerate(panels):canvas.paste(panel.resize((320,320),Image.Resampling.BICUBIC),(column*320,j*340+20))
            ImageDraw.Draw(canvas).text((5,j*340+3),f"proposal {len(index)} | original / geometry / keep / hide / crop",fill='black')
            index.append({'review_index':len(index),'image_id':k,'split':v['split'],'page':page})
        canvas.save(review_dir/f'page_{page:02d}.jpg',quality=95)
    save(p/'review_index.json',index)
    audit={'patients':{s:dict(Counter(r['pathology'] for r in rows.values() if r['split']==s)) for s in ('train','validation')},
           'independent_box_counts':dict(Counter(len(r['boxes']) for r in rows.values())),
           'raw_boxes_recovered_and_verified':True,'union_box_guessing':False,'patient_split_disjoint':True,
           'duplicate_patient_ids':0,'pixel_duplicate_scan':'not performed; preserved historical patient split',
           'geometry_exclusions':dict(geometry_reasons),'positive_pair_proposals':{s:sum(v['split']==s for v in views['positive_proposals'].values()) for s in ('train','validation')},
           'negative_sham_pairs':{s:sum(v['split']==s for v in views['pairs'].values()) for s in ('train','validation')},
           'crop_patients':{s:sum(rows[k]['split']==s for k in views['crops']) for s in ('train','validation')},
           'test_pixels_read':0,'model_bytes_mtime_match_frozen_identity':True,'revision':cfg['revision']}
    save(p/'data_audit.json',audit);print(json.dumps({'status':'awaiting_blinded_AI_review','audit':audit,'review_pages':math.ceil(len(proposals)/3),'token_cap':maximum}),flush=True)


if __name__=='__main__':main()
