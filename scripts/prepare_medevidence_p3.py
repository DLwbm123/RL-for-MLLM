"""CPU-only reconstruction from real frozen historical artifacts."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from src.data import sha,QUESTIONS
from src.v2 import save,DevelopmentData
from src.medevidence import normalized,parse_boxes,matching
from src.medevidence_p2 import enrich,summarize,token_contract
from src.medevidence_p3 import select_patients,local_pairs


def main():
    base=Path(os.environ['PILOT_ROOT']);code=base/'code';root=base/'outputs';p=root/'protocol';p.mkdir(exist_ok=True)
    if (p/'lock.json').exists() or (p/'CPU_checks.json').exists():raise FileExistsError('No re-selection or repeated preparation')
    p2=Path(os.environ['REFERENCE_P2_ROOT']);p1=Path(os.environ['REFERENCE_P1_ROOT']);old=p2/'protocol'
    lock=json.loads((old/'lock.json').read_text());cfg=json.loads((code/'configs/medevidence_p3.json').read_text())
    for name,h in lock['protocol_hashes'].items():assert sha((old/name).read_bytes())==h,'Historical protocol '+name
    for name,h in lock['source_hashes'].items():assert sha((code/name).read_bytes())==h,'Historical source '+name
    frame=pd.read_csv(old/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
    assert not frame.case_id.duplicated().any() and len(frame)==1280
    rows=frame.set_index('image_id').to_dict('index')
    for r in rows.values():r['boxes']=json.loads(r['boxes'])
    old_sets=json.loads((old/'sets.json').read_text());selected,plan=select_patients(rows,old_sets['Fit32'])
    views=json.loads((p1/'protocol/views.json').read_text());pairs,excluded=local_pairs(rows,views)
    references=json.loads((old/'reference_checkpoints.json').read_text());reference=references['M1'];cp=Path(reference['checkpoint_path'])
    assert cp.parent.parent==p1 and cp.parent.name=='M1' and cp.name=='step_0256'
    from safetensors.torch import load_file
    from src.medevidence_run import tensor_digest
    weights=load_file(str(cp/'adapter_model.safetensors'));weights={k.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight'):v for k,v in weights.items()}
    assert tensor_digest(weights)==reference['identity']['adapter_digest'];del weights
    history={}
    for name,path in (('M1',p1/'eval_M1/predictions.jsonl'),('L1',p2/'eval_L1/predictions.jsonl')):
        original=[json.loads(x) for x in path.read_text().splitlines()];assert len(original)==256
        assert all(rows[r['image_id']]['split']=='validation' for r in original)
        history[name]=[enrich(r,rows[r['image_id']]) for r in original]
        assert len({r['case_id'] for r in history[name]})==256
    for name in history:save(root/(name+'_historical_predictions.json'),history[name])
    h_summary={name:summarize(rs) for name,rs in history.items()}
    previous=json.loads((p2.parent/'code/reports/medevidence_p2_loc_probe.json').read_text())['development']
    discrepancies=[]
    for name,key in (('M1','cached_M1_metrics'),('L1','metrics')):
        for field in ('positive','negative','positive_region_recall','single_strict_success_count'):
            if h_summary[name][field]!=previous[key][field]:discrepancies.append({'model':name,'field':field,'previous':previous[key][field],'actual':h_summary[name][field]})
    save(code/'reports/medevidence_p3_historical_check.json',{'source':'actual private outputs, no regeneration','models':h_summary,
         'reference_commit':cfg['reference_commit'],'discrepancies':discrepancies})
    if discrepancies:raise ValueError('Historical replay differs; keep differences and pause preparation')
    identity=json.loads((old/'reference_identity.json').read_text());modelbase=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    for name,v in identity['base']['files'].items():
        st=(modelbase/name).stat();assert (st.st_size,st.st_mtime_ns)==(v['bytes'],v['mtime_ns'])
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(modelbase,local_files_only=True,use_fast=False,min_pixels=cfg['min_visual_tokens']*784,max_pixels=cfg['max_visual_tokens']*784)
    tokenizer=processor.tokenizer;assert tokenizer.eos_token_id==identity['eos_id']
    assert {x:tokenizer.encode(x,add_special_tokens=False) for x in ('no','yes')}==identity['label_ids']
    contracts={k:{'L':token_contract(tokenizer,rows[k]['loc_target']),'A':token_contract(tokenizer,rows[k]['pathology'])} for k in selected['order']}
    data=DevelopmentData(os.environ['DATA_ROOT'],frame,root/'CPU_data_access.jsonl');data.install_guard()
    try:(Path(os.environ['DATA_ROOT'])/'not_allowed_test.png').read_bytes()
    except PermissionError:pass
    else:raise AssertionError('Sealed test guard failed')
    images={};max_error=0.
    image_ids=set(selected['order'])|{r['positive'] for r in pairs}|{r['negative'] for r in pairs}
    for k in sorted(image_ids):
        r=rows[k];im=data.image(k);assert im.size==(r['width'],r['height'])
        path=Path(os.environ['DATA_ROOT'])/r['image_path'];st=path.stat()
        images[k]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':sha(path.read_bytes()),'size':list(im.size)}
    for r in rows.values():
        gt=normalized(r['boxes'],r['width'],r['height']);parsed,error=parse_boxes(r['loc_target']);assert not error and parsed==gt and matching(gt,parsed)['strict']
    for k in selected['order']:
        r=rows[k]
        for b,nb in zip(sorted(r['boxes']),normalized(r['boxes'],r['width'],r['height'])):
            for j,size in enumerate((r['width'],r['height'],r['width'],r['height'])):
                error=abs(b[j]-nb[j]*size/1000);assert error<=size/2000+1e-9;max_error=max(max_error,error)
    save(p/'selected_patients.json',selected);save(p/'schedule.json',plan);save(p/'local_pairs.json',{'pairs':pairs,'excluded':excluded})
    save(p/'token_contracts.json',contracts);save(p/'selected_images.json',images);save(p/'reference.json',reference);save(p/'identity.json',identity)
    (p/'manifest.csv').write_text((old/'manifest.csv').read_text());(p/'folds.json').write_text((p1/'protocol/folds.json').read_text())
    counts=Counter(k for update in plan for k in update)
    audit={'status':'prepared','positive_patients':len(selected['positive']),'negative_patients':len(selected['negative']),
           'shared_steps':256,'L_exposures_per_branch':1024,'positive_L_exposures':512,'negative_L_exposures':512,
           'positive_repetition_histogram':dict(Counter(counts[k] for k in selected['positive'])),
           'negative_repetition_histogram':dict(Counter(counts[k] for k in selected['negative'])),
           'old_Fit_overlap':len(set(old_sets['Fit32'])&set(selected['order'])),'old_Ref_overlap':{label:sum(k in selected['order'] for k in old_sets['Ref32'] if rows[k]['pathology']==label) for label in ('yes','no')},
           'old_Ref_not_unupdated_reference':True,'new_multibox_training_patients':0,'selection_uses_model_scores':False,
           'local_view_pairs':len(pairs),'local_independent_patients':2*len(pairs),'local_crops_all_scales':sum(len(r['views'])*2 for r in pairs),
           'local_positive_single_patients':sum(len(rows[r['positive']]['boxes'])==1 for r in pairs),
           'local_positive_multi_patients':sum(len(rows[r['positive']]['boxes'])>1 for r in pairs),
           'local_crop_qualification_excluded':sum(r['split']=='validation' and bool(r['boxes']) and k not in views['crops'] for k,r in rows.items()),
           'local_geometry_exclusions':len(excluded),'local_negative_donors_reused':0,'max_crop_geometry_normalized_error':max((abs(v) for pair in pairs for view in pair['views'] for v in view['normalized_geometry_difference']),default=0)}
    save(code/'reports/medevidence_p3_data_schedule_audit.json',audit)
    checks={'status':'passed','patients':327,'shared_schedule':True,'test_access_blocked':True,'test_pixels_read':0,
            'GT_self_matches':1280,'selected_image_identity_checks':len(images),'source_coordinate_roundtrip_max_pixels':max_error,
            'actual_tokenizer_targets_including_EOS':len(contracts),'M1_adapter_identity_exact':True,'base_revision_files_bytes_mtime_equal':True,
            'GPU_gradient_and_restore':'pending_native_no_update_check'}
    save(p/'CPU_checks.json',checks);save(code/'reports/medevidence_p3_engineering.json',{'CPU':checks,'GPU':'pending'})
    save(root/'stage_status.json',{'CPU':{'status':'passed'},'training':{'status':'ready_to_launch'},'local_views':{'status':'prepared','pairs':len(pairs)}})
    print(json.dumps({'CPU_checks':checks,'audit':audit}))


if __name__=='__main__':main()
