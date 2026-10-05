"""Real-image engineering checks; not a medical efficacy experiment."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from src.experiment import context,get_model,fixed_cases,sample_image
from src.data import write_json,LABELS
from src.regions import replace_features
from src.objectives import evidence_loss,stability_loss


def main():
    root,data,path,cfg,frame,regions=context();seed=cfg['seed'];started=time.time()
    model=get_model(path,cfg,trainable=True)
    samples=fixed_cases(frame,'train',cfg['smoke_images'],seed).to_dict('records')
    records=[]
    for row in samples:
        image=sample_image(data,row);inputs,grid=model.prompt(image)
        captured=[]
        hook=model.visual.merger.register_forward_hook(lambda module,args,out:captured.append(out.detach()))
        features=model.encode(inputs);hook.remove()
        window_index,_=model.visual.get_window_index(inputs['image_grid_thw'])
        assert torch.equal(features,captured[0][torch.argsort(window_index)])
        with torch.no_grad():
            a,_=model.classes(inputs,features);b,_=model.classes(inputs,features)
            c,_=model.classes(inputs,replace_features(features,[],[]))
        if not torch.equal(a,b) or not torch.equal(a,c):raise AssertionError('Repeated or empty-intervention scores changed')
        model.model.zero_grad(set_to_none=True)
        loss=-model.score(inputs,features,answer=row['pathology'],include_eos=True)['mean'];loss.backward()
        grad=sum(float(p.grad.float().square().sum()) for p in model.model.parameters() if p.grad is not None)**.5
        assert np.isfinite(grad) and grad>0
        records.append({'image_id':row['image_id'],'grid':grid,'loss':float(loss.detach()),'gradient_norm':grad,
                        'class_token_lengths':[len(model.processor.tokenizer.encode(x,add_special_tokens=False)) for x in LABELS]})
    eligible=frame[(frame.split=='train') & frame.image_id.map(lambda x:regions.get(x,{}).get('eligible',False))]
    if len(eligible)==0:raise RuntimeError('No eligible training examples for intervention gradient check')
    row=eligible.sort_values('image_id').iloc[0].to_dict();region=regions[row['image_id']]
    image=sample_image(data,row);inputs,grid=model.prompt(image);features=model.encode(inputs)
    assert list(grid)==region['grid']
    y=LABELS.index(row['pathology']);gradient_checks={}
    for kind in ['evidence','stability','zero_gate']:
        model.model.zero_grad(set_to_none=True)
        original,_=model.classes(inputs,features)
        evidence,_=model.classes(inputs,replace_features(features,region['evidence']['tokens'],region['evidence']['source']))
        controls=torch.stack([model.classes(inputs,replace_features(features,r['tokens'],r['source']))[0] for r in region['controls']])
        if kind=='stability':loss=stability_loss(original,controls)
        else:
            # Large diagnostic margins make hinges active; not used for training or performance claims.
            loss,_=evidence_loss(original,evidence,controls,y,weight=0 if kind=='zero_gate' else 1,delta_rel=100,delta_abs=100)
        loss.backward()
        grad=sum(float(p.grad.float().square().sum()) for p in model.model.parameters() if p.grad is not None)**.5
        if kind=='zero_gate':assert grad==0
        else:assert np.isfinite(grad) and grad>0
        gradient_checks[kind]={'loss':float(loss.detach()),'gradient_norm':grad}
    dest=root/'smoke';dest.mkdir(exist_ok=True)
    # Actual PEFT checkpoint save/load, plus a nonzero optimizer update and restoration.
    model.model.zero_grad(set_to_none=True)
    opt=torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad],lr=cfg['sft_lr'])
    (-model.score(inputs,features,answer=row['pathology'],include_eos=True)['mean']).backward();opt.step()
    with torch.no_grad():before=model.classes(inputs,features)[0]
    model.save(dest/'checkpoint')
    from safetensors.torch import load_file
    state=load_file(str(dest/'checkpoint/adapter_model.safetensors'))
    with torch.no_grad():
        for n,p in model.model.named_parameters():
            if p.requires_grad:p.zero_()
    set_peft_model_state_dict(model.model,state)
    with torch.no_grad():after=model.classes(inputs,features)[0]
    assert torch.equal(before,after),'Checkpoint restore changed scores'
    answer,ids=model.generate(inputs,features,max_tokens=16,sample=False,diagnostic=True)
    with torch.no_grad():teacher=model.score(inputs,features,ids=ids)['tokens'].float().cpu().numpy()
    generation_delta=float(np.max(np.abs(teacher-np.asarray(model.last_generation_logps))))
    assert generation_delta<.15,'Cached generation and teacher forcing disagree beyond BF16 tolerance'
    assert model.forward_count==model.score_count+model.generated_tokens,'Forward counter misses scoring or generation'
    result={'status':'passed','scope':'engineering_only_real_training_images','images':len(records),'samples':records,
            'gradient_checks':gradient_checks,'checkpoint_score_exact':True,'window_restore_mapping_verified':True,'multi_gpu':'not_run_single_gpu_pilot',
            'elapsed_s':time.time()-started,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,
            'trainable_parameters':sum(p.numel() for p in model.model.parameters() if p.requires_grad)}
    result['generation_teacher_forcing_max_abs_logprob_delta']=generation_delta
    result['generation_teacher_forcing_tolerance']=.15
    result['forward_count']=model.forward_count;result['score_count']=model.score_count;result['generated_tokens']=model.generated_tokens
    write_json(dest/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
