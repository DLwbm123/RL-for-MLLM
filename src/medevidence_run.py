"""Independent supervised/evidence pilot. Neutral stdin entry; no legacy RL gates."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import pandas as pd
import torch
from src.data import QUESTIONS, sha
from src.model import Model
from src.experiment import seed_all, optimizer_snapshot
from src.v2 import DevelopmentData, save
from src.medevidence import intervention, crop_box, parse_boxes, normalized, matching, dependency, exposures


class BudgetStop(Exception): pass


def tensor_digest(state):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for item in x:visit(item)
        elif isinstance(x,np.ndarray):h.update(x.tobytes())
        else:h.update(repr(x).encode())
    visit(state);return h.hexdigest()


class Context:
    def __init__(self):
        self.started=time.time();self.deadline=float(os.environ['JOB_DEADLINE']);self.stop=False
        signal.signal(signal.SIGUSR1,lambda *_:setattr(self,'stop',True))
        self.root=Path(os.environ['OUTPUT_ROOT']);self.code=Path(__file__).resolve().parents[1]
        self.cfgpath=Path(os.environ['RUN_CONFIG']);self.cfg=json.loads(self.cfgpath.read_text());self.stage=os.environ['JOB_STAGE']
        if not self.cfg['supervised_training_enabled'] or not self.cfg['evidence_ft_enabled'] or self.cfg['rl_enabled'] is not False:raise PermissionError('New training authorization contract violated')
        if self.stage not in ('smoke','M0','W','M1','M2','eval_M0','eval_M1','eval_M2'):raise PermissionError('Unauthorized stage')
        self.dest=self.root/self.stage;self.dest.mkdir(exist_ok=True);self.protocol=self.root/'protocol'
        lock=json.loads((self.protocol/'lock.json').read_text())
        if sha(self.cfgpath.read_bytes())!=lock['config_sha256']:raise ValueError('Config changed after freeze')
        for n,h in lock['hashes'].items():
            if sha((self.protocol/n).read_bytes())!=h:raise ValueError('Protocol changed '+n)
        for n,h in lock['source_hashes'].items():
            if sha((self.code/n).read_bytes())!=h:raise ValueError('Execution source changed '+n)
        self.views=json.loads((self.protocol/'views.json').read_text());self.plan=json.loads((self.protocol/'schedule.json').read_text())
        self.frame=pd.read_csv(self.protocol/'manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
        self.rows=self.frame.set_index('image_id').to_dict('index')
        for r in self.rows.values():r['boxes']=json.loads(r['boxes'])
        self.data=DevelopmentData(os.environ['DATA_ROOT'],self.frame,self.dest/'data_access.jsonl');self.data.install_guard()
        self.identity=json.loads((self.protocol/'identity.json').read_text());self.base=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
        for n,r in self.identity['base']['files'].items():
            st=(self.base/n).stat()
            if (st.st_size,st.st_mtime_ns)!=(r['bytes'],r['mtime_ns']):raise ValueError('Base identity changed')
        QUESTIONS.update(self.cfg['prompts']);self.tick()

    def tick(self):
        if self.stop or time.time()>=self.deadline-90:raise BudgetStop('Frozen stage quota or shared budget reached')

    def image(self,k,view='original',scale=2.,crop_index=0):
        im=self.data.image(k)
        if view=='crop':
            b=self.views['crops'][k][crop_index]['box'] if scale==2. else (crop_box(self.rows[k]['boxes'][crop_index],*im.size,scale) if self.rows[k]['boxes'] else self.views['crops'][k][crop_index]['evaluation_box'])
            return im.crop(b)
        if view=='original':return im
        pair=self.views['pairs'][k];kind,which=view.split('_');b=pair['control'] if which=='keep' else pair['target']
        return intervention(im,b,kind,self.cfg['views'])

    def model(self,adapter=None,train=True):
        seed_all(self.cfg['seed']);self.tick()
        m=Model(self.base,adapter,train,self.cfg['min_visual_tokens'],self.cfg['max_visual_tokens'])
        ids={lab:m.processor.tokenizer.encode(lab,add_special_tokens=False) for lab in ('no','yes')}
        if ids!=self.identity['label_ids'] or m.processor.tokenizer.eos_token_id!=self.identity['eos_id']:raise ValueError('Tokenizer identity changed')
        if train:
            params={n:p for n,p in m.model.named_parameters() if p.requires_grad}
            if sum(p.numel() for p in params.values())!=5046272 or any('lora_' not in n or '.visual.' in n for n in params):raise ValueError('LoRA scope changed')
        self.tick();return m


def adapter_digest(m):return tensor_digest({n:p for n,p in m.model.named_parameters() if 'lora_' in n})


def restore(m,opt,path):
    saved=torch.load(Path(path)/'training_state.pt',map_location='cpu',weights_only=False)
    opt.load_state_dict(saved['optimizer'])
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved


def train(c):
    method=c.stage;branch=method in ('M1','M2');offset=256 if branch else 0;steps=c.cfg['steps'][method]
    if method=='M2' and not any(v['positive'] and v['split']=='train' for v in c.views['pairs'].values()):
        save(c.dest/'summary.json',{'status':'blocked_no_valid_evidence_pairs','steps':0});return
    if (c.dest/'train.jsonl').exists():raise FileExistsError('Existing attempt; no silent retry')
    adapter=c.root/'W/step_0256' if branch else None;m=c.model(adapter,True)
    params=[p for p in m.model.parameters() if p.requires_grad];oc=c.cfg['optimizer']
    opt=torch.optim.AdamW(params,lr=oc['lr'],betas=tuple(oc['betas']),eps=oc['eps'],weight_decay=oc['weight_decay'])
    if branch:
        state=restore(m,opt,adapter)
        if state['step']!=256:raise ValueError('Wrong W split')
    initial={'adapter_digest':adapter_digest(m),'optimizer_digest':tensor_digest(opt.state_dict()),
             'RNG_digest':tensor_digest([random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()]),
             'schedule_position':offset,'lr':opt.param_groups[0]['lr']}
    save(c.dest/'initial_identity.json',initial)
    if branch:
        expected=json.loads((adapter/'identity.json').read_text())
        if initial!=expected:raise ValueError('W model/optimizer/RNG/schedule fork mismatch')
    elif method in ('M0','W'):
        other=c.root/('W' if method=='M0' else 'M0')/'initial_identity.json'
        if other.exists() and json.loads(other.read_text())['adapter_digest']!=initial['adapter_digest']:raise ValueError('Fresh B0 initialization differs')
    completed=0;status='running';loss_sums={};hinge_count=0;positive_updates=0;dep_gradient_norms=[];step_times=[]
    save(c.dest/'status.json',{'status':'running','step':0,'pid':os.getpid()})
    try:
        with (c.dest/'train.jsonl').open('w') as log:
            for local in range(steps):
                c.tick();begin=time.time();item=c.plan[offset+local];opt.zero_grad(set_to_none=True);stats={'local_step':local+1,'cumulative_step':offset+local+1,'dep_applicable':False,'dep_loss':0.}
                cache={}
                def encoded(k,view='original',task='A',index=0):
                    key=(k,view,index)
                    image=c.image(k,view,crop_index=index)
                    inputs,_=m.prompt(image,task)
                    if key not in cache:cache[key]=m.encode(inputs)
                    return inputs,cache[key]
                def nll(k,task='A',view='original',index=0,label=None):
                    inputs,features=encoded(k,view,task,index)
                    target=c.rows[k]['loc_target'] if task=='L' else (label or c.rows[k]['pathology'])
                    result=m.score(inputs,features,answer=target,include_eos=True)
                    if result['ids'][-1].item()!=m.processor.tokenizer.eos_token_id or result['length']!=len(result['ids']):raise ValueError('EOS/token mask alignment')
                    value=-result['mean'];eos=-result['tokens'][-1]
                    if not torch.isfinite(value):raise FloatingPointError('Nonfinite token NLL')
                    return value,eos
                full=0.;full_eos=0.
                for k in item['full']:
                    value,eos=nll(k);(value/4).backward();full+=float(value.detach())/4;full_eos+=float(eos.detach())/4
                stats.update(full_loss=full,full_eos=full_eos,loc_loss=0.,crop_loss=0.,control_loss=0.,full_seconds=time.time()-begin)
                aux_begin=time.time()
                if method!='M0':
                    for task,weight in [('loc',.5),('crop',.5)]:
                        k=item[task]
                        if k:
                            value,eos=nll(k,'L' if task=='loc' else 'crop','crop' if task=='crop' else 'original',item['crop_index'] or 0)
                            (weight*value).backward();stats[task+'_loss']=float(value.detach());stats[task+'_eos']=float(eos.detach())
                stats['aux_seconds']=time.time()-aux_begin;pair_begin=time.time()
                if method in ('M1','M2','smoke') and item['pair']:
                    k=item['pair'];positive=c.views['pairs'][k]['positive']
                    if positive:
                        oi,of=encoded(k)
                        with torch.no_grad(): original=m.classes(oi,of)[0].float().softmax(-1)
                        ki,kf=encoded(k,'gray_keep');hi,hf=encoded(k,'gray_hide')
                        if method=='smoke' and (kf is hf or kf is of or hf is of or torch.equal(kf,hf)):
                            raise RuntimeError('Different pixel views reused one visual encoding')
                        keep=m.classes(ki,kf)[0].float().softmax(-1);hide=m.classes(hi,hf)[0].float().softmax(-1)
                        control=(original*(original.clamp_min(1e-12).log()-keep.clamp_min(1e-12).log())).sum()
                        dep=dependency(keep[1],hide[1],.1);coefficient=.2 if method in ('M2','smoke') else 0.
                        if not keep.requires_grad or not hide.requires_grad or dep.grad_fn is None:raise RuntimeError('Dependency graph disconnected')
                        if method=='smoke':
                            audit_started=time.time()
                            gradients=torch.autograd.grad(dep,params,retain_graph=True,allow_unused=True)
                            connected=[g for g in gradients if g is not None]
                            if not connected or not all(torch.isfinite(g).all() for g in connected):raise RuntimeError('Dependency LoRA gradient disconnected/nonfinite')
                            dn=float(torch.sqrt(sum(g.float().square().sum() for g in connected)));dep_gradient_norms.append(dn)
                            stats['dep_gradient_norm']=dn
                            if local==0:
                                g1=torch.autograd.grad(.1*control,params,retain_graph=True,allow_unused=True)
                                # Use the production zero-weight branch: zero multiplication still traverses a BF16 graph.
                                zero_weight=c.cfg['loss']['dep_M1']
                                zero_target=.1*control+zero_weight*dep if zero_weight else .1*control
                                g2=torch.autograd.grad(zero_target,params,retain_graph=True,allow_unused=True)
                                delta=max(float((a-b).abs().max()) for a,b in zip(g1,g2) if a is not None and b is not None)
                                pairs=[(a.float(),b.float()) for a,b in zip(g1,g2) if a is not None and b is not None]
                                difference_norm=float(torch.sqrt(sum((a-b).square().sum() for a,b in pairs)))
                                reference_norm=float(torch.sqrt(sum(a.square().sum() for a,b in pairs)))
                                maximum=max(float(a.abs().max()) for a,b in pairs)
                                relative=difference_norm/max(reference_norm,1e-12)
                                tolerance=2*torch.finfo(torch.bfloat16).eps
                                stats.update(zero_dep_gradient_relative_L2=relative,zero_dep_gradient_tolerance=tolerance,
                                             zero_dep_gradient_reference_max=maximum)
                                if relative>tolerance or delta>tolerance*max(maximum,1e-12):
                                    raise RuntimeError(f'Zero-dep gradient mismatch: max_abs={delta}, relative_L2={relative}, max_reference={maximum}, tolerance={tolerance}')
                                stats['zero_dep_gradient_max_difference']=delta
                                label=c.rows[k]['pathology'];result=m.score(oi,of,answer=label,include_eos=True)
                                answer=m.score(oi,of,answer=label,include_eos=False)
                                if not torch.allclose(result['tokens'][:-1],answer['tokens'],atol=1e-4,rtol=1e-4):raise ValueError('Next-token alignment inconsistency')
                            stats['dep_gradient_audit_seconds']=time.time()-audit_started
                        total=.1*control+coefficient*dep if coefficient else .1*control
                        if not torch.isfinite(total):raise FloatingPointError('Nonfinite control/dependency')
                        backward_start=time.time();total.backward();stats['pair_backward_s']=time.time()-backward_start
                        stats.update(dep_applicable=True,dep_loss=float(dep.detach()),control_loss=float(control.detach()),
                                     u_original=float(original[1]),u_keep=float(keep[1].detach()),u_hide=float(hide[1].detach()),hinge_active=bool(dep.detach()>0))
                        positive_updates+=1;hinge_count+=int(stats['hinge_active'])
                    else:
                        control_value=0.
                        for view in ('gray_keep','gray_hide'):
                            value,eos=nll(k,view=view,label='no');(.05*value).backward();control_value+=float(value.detach())/2
                        stats['control_loss']=control_value
                stats['pair_seconds']=time.time()-pair_begin
                if any(p.grad is not None for p in m.visual.parameters()):raise RuntimeError('Frozen visual/merger received gradient')
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm) or not any(p.grad is not None for p in params):raise FloatingPointError('Nonfinite/missing LoRA gradients')
                opt.step();completed=local+1
                stats.update(gradient_norm=float(norm),update_seconds=time.time()-begin,elapsed_s=time.time()-c.started,
                             peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,forward_count=m.forward_count,vision_count=m.vision_count)
                step_times.append(stats['update_seconds'])
                for key in ('full_loss','loc_loss','crop_loss','control_loss','dep_loss'):loss_sums[key]=loss_sums.get(key,0.)+stats[key]
                log.write(json.dumps(stats)+'\n');log.flush();print(json.dumps(stats),flush=True)
                save(c.dest/'status.json',{'status':'running','step':completed,'cumulative_step':offset+completed,'pid':os.getpid()})
                if completed in c.cfg['checkpoints'].get(method,[]) or (method=='smoke' and completed==steps):
                    checkpoint=c.dest/f'step_{completed:04d}';optimizer_snapshot(m,opt,checkpoint,offset+completed)
                    identity={'adapter_digest':adapter_digest(m),'optimizer_digest':tensor_digest(opt.state_dict()),
                              'RNG_digest':tensor_digest([random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()]),
                              'schedule_position':offset+completed,'lr':opt.param_groups[0]['lr']}
                    save(checkpoint/'identity.json',identity)
                    if method=='smoke':
                        expected_draw=[random.random(),float(np.random.rand()),float(torch.rand(1)),float(torch.rand(1,device='cuda'))]
                        restore(m,opt,checkpoint)
                        actual_draw=[random.random(),float(np.random.rand()),float(torch.rand(1)),float(torch.rand(1,device='cuda'))]
                        if actual_draw!=expected_draw or tensor_digest(opt.state_dict())!=identity['optimizer_digest']:raise RuntimeError('Optimizer/RNG restore mismatch')
                        from safetensors.torch import load_file
                        from peft import set_peft_model_state_dict
                        with torch.no_grad():params[0].add_(1.)
                        set_peft_model_state_dict(m.model,load_file(str(checkpoint/'adapter_model.safetensors')))
                        if adapter_digest(m)!=identity['adapter_digest']:raise RuntimeError('Adapter save/load identity mismatch')
                        save(c.dest/'restore_check.json',{'status':'passed','optimizer_RNG_exact':True,'adapter_exact':True})
                        (checkpoint/'adapter_model.safetensors').unlink();(checkpoint/'training_state.pt').unlink()
            status='completed'
    except BudgetStop:
        optimizer_snapshot(m,opt,c.dest/f'stopped_step_{completed:04d}',offset+completed);status='stopped_budget'
    summary={'status':status,'steps':completed,'planned_steps':steps,'cumulative_step':offset+completed,
             'mean_losses':{k:v/completed if completed else None for k,v in loss_sums.items()},
             'hinge_active_rate':hinge_count/positive_updates if positive_updates else None,'dep_applicable_updates':positive_updates,
             'dep_gradient_norms_smoke':dep_gradient_norms,'mean_update_seconds':float(np.mean(step_times)) if step_times else None,
             'update_time_quantiles':np.quantile(step_times,[.1,.5,.9]).tolist() if step_times else None,
             'runtime_seconds':time.time()-c.started,'gpu_hours':(time.time()-c.started)/3600,
             'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,
             'forward_count':m.forward_count,'vision_count':m.vision_count,
             'exposures':exposures(c.plan[offset:offset+completed],c.rows,c.views,method),'test_pixels_read':0,
             'smoke_weights':'deleted after successful adapter/optimizer/RNG restore audit; never used for formal initialization' if method=='smoke' else None}
    save(c.dest/'summary.json',summary);save(c.dest/'status.json',{'status':status,'step':completed});print(json.dumps(summary),flush=True)


@torch.no_grad()
def evaluate(c):
    method=c.stage[5:];checkpoint=c.root/method/('step_0512' if method=='M0' else 'step_0256')
    if json.loads((c.root/method/'summary.json').read_text())['status']!='completed':raise PermissionError('Fixed final checkpoint incomplete')
    if (c.dest/'predictions.jsonl').exists():raise FileExistsError('No silent repeated evaluation')
    m=c.model(checkpoint,False);cap=json.loads((c.protocol/'token_contract.json').read_text())['max_new_tokens'];done=0;status='running'
    with (c.dest/'predictions.jsonl').open('w') as out:
        try:
            for k,r in sorted(c.rows.items()):
                if r['split']!='validation':continue
                c.tick();im=c.image(k);inp,_=m.prompt(im,'A');feat=m.encode(inp);scores=m.classes(inp,feat)[0].float()
                p=float(scores.softmax(-1)[1]);rec={'image_id':k,'case_id':r['case_id'],'y':int(r['pathology']=='yes'),'p':p,
                    'score_no':float(scores[0]),'score_yes':float(scores[1]),'margin':float(scores[1]-scores[0])}
                loc,_=m.prompt(im,'L');answer,tokens=m.generate(loc,feat,max_tokens=cap,sample=False)
                boxes,error=parse_boxes(answer,c.cfg['localization']['max_parsed_boxes']);truncated=len(tokens)>=cap and (not tokens or tokens[-1]!=m.processor.tokenizer.eos_token_id)
                if truncated:boxes=None;error='truncated'
                gt=normalized(r['boxes'],r['width'],r['height']);rec.update(loc_text=answer,parse_error=error,truncated=truncated,**matching(gt,boxes))
                if k in c.views['pairs']:
                    v=c.views['pairs'][k];rec['views']={'positive':v['positive']}
                    for kind in ('gray','blur','blank'):
                        values={}
                        for which in ('keep','hide'):
                            c.tick();vi,_=m.prompt(c.image(k,kind+'_'+which),'A');vf=m.encode(vi)
                            values[which]=float(m.classes(vi,vf)[0].float().softmax(-1)[1])
                        values.update(S=values['keep']-values['hide'],D_E=p-values['hide'],D_N=p-values['keep'])
                        rec['views'][kind]=values
                if r['boxes'] and k in c.views['crops']:
                    rec['crops']=[]
                    for index in range(len(r['boxes'])):
                        values={}
                        for scale in (2.,2.5):
                            c.tick();ci,_=m.prompt(c.image(k,'crop',scale,index),'crop');cf=m.encode(ci)
                            values[str(scale)]=float(m.classes(ci,cf)[0].float().softmax(-1)[1])
                        rec['crops'].append(values)
                elif not r['boxes']:
                    for scale in (2.,2.5):
                        ci,_=m.prompt(c.image(k,'crop',scale),'crop');cf=m.encode(ci)
                        rec['negative_crop_p_'+str(scale)]=float(m.classes(ci,cf)[0].float().softmax(-1)[1])
                else:rec['crop_skipped']='frozen AI crop-context exclusion'
                out.write(json.dumps(rec)+'\n');out.flush();done+=1;print(json.dumps({'stage':c.stage,'done':done,'elapsed_s':time.time()-c.started}),flush=True)
            status='completed'
        except BudgetStop:status='stopped_budget'
    summary={'status':status,'patients':done,'planned_patients':256,'runtime_seconds':time.time()-c.started,
             'gpu_hours':(time.time()-c.started)/3600,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,
             'forward_count':m.forward_count,'vision_count':m.vision_count,'test_pixels_read':0,'max_new_tokens':cap}
    save(c.dest/'summary.json',summary);print(json.dumps(summary),flush=True)


def main():
    c=None
    try:
        c=Context();evaluate(c) if c.stage.startswith('eval_') else train(c)
    except Exception as e:
        if c:
            failed={'status':'failed','error_type':type(e).__name__,'reason':str(e),'runtime_seconds':time.time()-c.started}
            save(c.dest/'summary.json',failed);save(c.dest/'status.json',failed)
        raise


if __name__=='__main__':main()
