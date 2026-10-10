"""One on-policy update group for the seven registered project adaptations."""
from contextlib import nullcontext
import json
import torch
from src.medevidence import parse_boxes,matching
from src.objectives import advantages
from src.rl_methods import (PRIORITY,visurf_advantages,zvp_advantages,axpo_advantages,author_advantages,
                            completion_loss,perception_loss,cfpo_attention)
from src.rl_observation import (observation_prompt,parse_observation,tool_prefix,tool_span,pixel_box,validate_observation_support,
                               active_reward,random_patch_mask,defacto_views,evidence_prompt,defacto_reward)


def probability_audit(generated,replayed,cfg):
    generated=torch.as_tensor(generated,device=replayed.device,dtype=torch.float32)
    if generated.shape!=replayed.shape or not len(generated) or not torch.isfinite(generated).all() or not torch.isfinite(replayed).all():
        raise ValueError('Invalid generation/replay probability span')
    error=(generated-replayed).abs()
    stats={'max_error':float(error.max()),'mean_error':float(error.mean())}
    if stats['max_error']>cfg['generation_replay_max_logprob_error'] or stats['mean_error']>cfg['generation_replay_mean_logprob_error']:
        raise RuntimeError('Native generation/replay probability gate failed: '+json.dumps(stats))
    return stats


def native_sampling(settings):
    expected={'temperature':1.,'top_p':1.,'top_k':0,'repetition_penalty':1.,'num_beams':1,
              'num_beam_groups':1,'typical_p':1.,'min_p':None,'epsilon_cutoff':0.,'eta_cutoff':0.,
              'no_repeat_ngram_size':0,'encoder_no_repeat_ngram_size':0,'penalty_alpha':None,
              'forced_bos_token_id':None,'forced_eos_token_id':None,'forced_decoder_ids':None,
              'suppress_tokens':None,'begin_suppress_tokens':None,'sequence_bias':None,
              'bad_words_ids':None,'constraints':None,'force_words_ids':None,'watermarking_config':None,
              'renormalize_logits':False,'remove_invalid_values':False,'min_length':0,'min_new_tokens':None}
    if not settings.get('do_sample') or any(settings.get(k,v)!=v for k,v in expected.items()):
        raise ValueError('Rollout must use untransformed native on-policy sampling')


def query_masks(m,inputs):
    ids=inputs['input_ids'][0];image=ids==m.model.config.image_token_id
    end=m.model.config.vision_end_token_id
    locations=(ids==end).nonzero().flatten()
    if len(locations)!=1:raise ValueError('Single image and explicit image boundary required')
    after=int(locations[0]);start=(ids==m.model.config.vision_start_token_id).nonzero().flatten()
    if len(start)!=1 or not image[int(start[0])+1:after].all() or int(start[0])+1>=after:
        raise ValueError('Contiguous single-image span required')
    query=torch.zeros_like(image);query[after:]=True
    if not image.any() or not query.any():raise ValueError('Missing CFPO image/question span')
    return image,query


class UpdateGroup:
    """Caller supplies a frozen-reference context and a deadline check; no data access."""
    def __init__(self,m,method,cfg,reference,tick,rng):
        if method not in PRIORITY:raise ValueError('Unknown RL adaptation')
        if cfg['sampling']['completions']<2:raise ValueError('At least two rollouts required')
        native_sampling(cfg['sampling']|{'do_sample':True})
        if method=='axpo':
            a=cfg['axpo']
            if type(a['continuations']) is not int or a['continuations']<2 or not 0<a['extra_rollout_ratio']<1:
                raise ValueError('AXPO requires K>=2 and a global extra-rollout ratio in (0,1)')
        self.m=m;self.method=method;self.cfg=cfg;self.reference=reference;self.tick=tick;self.rng=rng
        self.observer_calls=0;self.extra_rollouts=0

    def frozen_answer(self,image,question):
        self.tick()
        with self.reference():
            inputs,_=self.m.prompt(image,question=question+' Answer only yes or no.')
            features=self.m.encode(inputs)
            scores,_=self.m.classes(inputs,features,labels=('no','yes'))
            answer=('no','yes')[int(scores.argmax())]
        self.observer_calls+=1
        return answer

    def outcome(self,text,truncated,image,gt,controls,variant,question):
        if self.method in ('active_o3','axpo'):
            o=self.cfg['observation'];action=parse_observation(text,o['max_regions'],truncated)
            answer=action['answer']
            if action['valid'] and action['tool']:
                answers=[self.frozen_answer(image.crop(pixel_box(b,image.size)),question) for b in action['boxes']]
                answer='yes' if 'yes' in answers else 'no'
            return active_reward(action,gt,answer,o['min_area'],o['max_area'])
        if self.method=='defacto':return defacto_reward(text,gt,controls,variant,truncated)
        boxes,error=parse_boxes(text)
        correct=float(not truncated and not error and matching(gt,boxes)['strict'])
        return {'reward':correct,'correct':correct}

    def collect(self,inputs,features,image,gt,controls,variant,question,prefix=None):
        self.tick();m=self.m;cfg=self.cfg
        prefix=prefix or {'tokens':[],'generated':[]}
        n=len(prefix['tokens']);remaining=cfg['sampling']['max_new_tokens']-n
        if remaining<=0:raise ValueError('No continuation token budget')
        extended=dict(inputs)
        if n:
            ids=torch.as_tensor(prefix['tokens'],device=inputs['input_ids'].device,dtype=torch.long)[None]
            extended['input_ids']=torch.cat([inputs['input_ids'],ids],1)
            extended['attention_mask']=torch.ones_like(extended['input_ids'])
        _,suffix=m.generate(extended,features,diagnostic=True,sampling_config=cfg['sampling']|{'max_new_tokens':remaining})
        native_sampling(m.last_generation_settings)
        tokens=prefix['tokens']+suffix;generated=prefix['generated']+m.last_generation_logps
        if not suffix:raise ValueError('Empty generated continuation')
        text=m.processor.tokenizer.decode(tokens,skip_special_tokens=True)
        truncated=len(tokens)>=cfg['sampling']['max_new_tokens'] and tokens[-1]!=m.processor.tokenizer.eos_token_id
        before=m.model.training;m.set_training(False)
        try:
            with torch.no_grad():old=m.score(inputs,features,ids=tokens,include_entropy=self.method=='rl_zvp')
            alignment=probability_audit(generated,old['tokens'],cfg)
            with self.reference():ref=m.score(inputs,features,ids=tokens)['tokens'].detach()
        finally:m.set_training(before)
        span=tool_span(tokens,m.processor.tokenizer) if self.method=='axpo' else None
        result={'tokens':tokens,'old':old['tokens'].detach(),'reference':ref,'generated':generated,'alignment':alignment,
                'entropy':old.get('entropy'),'mask':torch.ones(len(tokens),dtype=torch.bool),
                'prefix':tool_prefix(tokens,m.processor.tokenizer) if span else None,'tool_span':span}
        result.update(self.outcome(text,truncated,image,gt,controls,variant,question))
        return result

    def groups(self,image,gt,question,controls=(),complete_evidence=False,resample=True):
        m=self.m;cfg=self.cfg
        if self.method in ('active_o3','axpo'):validate_observation_support(gt,cfg['observation']['max_regions'])
        if self.method=='defacto':views=defacto_views(image,gt,controls,complete_evidence);prompt=evidence_prompt(question)
        else:
            views=[('pos',image,'yes' if gt else 'no')]
            prompt=(observation_prompt(question,cfg['observation']['max_regions']) if self.method in ('active_o3','axpo') else
                    question+' Return only [[x1,y1,x2,y2],...] with coordinates normalized to 0–1000, or [] when no target is visible.')
        out=[]
        for variant,view,_ in views:
            self.tick();inputs,_=m.prompt(view,question=prompt);features=m.encode(inputs)
            samples=[self.collect(inputs,features,view,gt,controls,variant,question) for _ in range(cfg['sampling']['completions'])]
            rewards=[s['correct'] if self.method in ('rl_zvp','axpo') else s['reward'] for s in samples]
            if self.method=='axpo':
                adv=advantages(torch.tensor(rewards,dtype=torch.float32)).tolist()
            elif self.method=='visurf':
                target=json.dumps(gt,separators=(',',':'))
                ids=m.processor.tokenizer.encode(target,add_special_tokens=False)+[m.processor.tokenizer.eos_token_id]
                with torch.no_grad():old=m.score(inputs,features,ids=ids)['tokens'].detach()
                with self.reference():ref=m.score(inputs,features,ids=ids)['tokens'].detach()
                samples.append({'tokens':ids,'old':old,'reference':ref,'mask':torch.ones(len(ids),dtype=torch.bool),'reward':1.,'correct':1.})
                adv=visurf_advantages(rewards,1.).tolist()
            elif self.method=='rl_zvp':adv=zvp_advantages(rewards,[s['entropy'] for s in samples],cfg['zvp_alpha'])
            elif self.method in ('papo','cfpo','defacto'):adv=author_advantages(rewards,self.method).tolist()
            else:adv=advantages(torch.tensor(rewards,dtype=torch.float32)).tolist()
            group={'inputs':inputs,'features':features,'samples':samples,'advantages':adv,'view':variant,'selected_prefixes':[],
                   'original_count':len(samples)}
            if self.method=='axpo':group['case']=(view,gt,controls,question)
            if self.method=='papo':
                masked,indices=random_patch_mask(view,cfg['papo']['patch_size'],self.rng,cfg['papo']['mask_ratio'])
                altered,_=m.prompt(masked,question=prompt);group['corrupted']=(altered,m.encode(altered));group['masked_patches']=indices
            if self.method=='cfpo':group['spans']=query_masks(m,inputs)
            if self.method in ('papo','cfpo'):
                ci,cf=group.get('corrupted',(inputs,features))
                context=cfpo_attention(m.model,*group['spans'],sigma=cfg['cfpo']['sigma']) if self.method=='cfpo' else nullcontext()
                with torch.no_grad(),context:
                    for sample in samples:sample['corrupted_old']=m.score(ci,cf,ids=sample['tokens'])['tokens'].detach()
            out.append(group)
        if self.method=='axpo' and resample:self.resample(out)
        return out

    def batch_groups(self,cases):
        """Collect all original groups before allocating AXPO's step-wide budget."""
        if not cases:raise ValueError('Nonempty update batch required')
        if self.method in ('active_o3','axpo'):
            for case in cases:validate_observation_support(case[1],self.cfg['observation']['max_regions'])
        groups=[g for case in cases for g in self.groups(*case,resample=False)]
        if self.method=='axpo':self.resample(groups)
        return groups

    def resample(self,groups):
        a=self.cfg['axpo'];k=a['continuations'];slots=int(a['extra_rollout_ratio']*sum(g['original_count'] for g in groups))//k
        candidates=[]
        for g in groups:
            tool=[i for i,s in enumerate(g['samples']) if s['prefix'] is not None]
            if tool and all(g['samples'][i]['correct']==0 for i in tool):
                tool.sort(key=lambda i:float(g['samples'][i]['old'][slice(*g['samples'][i]['tool_span'])].exp().mean()))
            else:tool=[]
            candidates.append(tool)
        # Breadth first: each admitted question gets its first prefix before any gets a second.
        for rank in range(max(map(len,candidates),default=0)):
            for g,tool in zip(groups,candidates):
                if slots and rank<len(tool):g['selected_prefixes'].append(tool[rank]);slots-=1
        for g in groups:
            original=g['samples'][:g['original_count']];branches={}
            image,gt,controls,question=g['case']
            for i in g['selected_prefixes']:
                s=original[i];n=s['prefix'];prefix={'tokens':s['tokens'][:n],'generated':s['generated'][:n]}
                branches[i]=[self.collect(g['inputs'],g['features'],image,gt,controls,g['view'],question,prefix) for _ in range(k)]
                self.extra_rollouts+=k;s['mask'][n:]=False
                for branch in branches[i]:branch['mask'][:n]=False
            base,extra=axpo_advantages([s['correct'] for s in original],[s['prefix'] for s in original],
                                       {i:[s['correct'] for s in ss] for i,ss in branches.items()})
            g['advantages']=base.tolist()
            for i,ss in branches.items():g['samples'].extend(ss);g['advantages'].extend(extra[i].tolist())

    def backward(self,groups,scale=1.):
        if not groups or not 0<scale<=1:raise ValueError('Nonempty groups and positive averaging scale required')
        m=self.m;cfg=self.cfg;logs=[];weighted_loss=0.
        token_count=sum(int(s['mask'].sum()) for g in groups for s in g['samples'])
        # Check every old/current pair before accumulating any gradient.
        for g in groups:
            for s in g['samples']:
                self.tick()
                with torch.no_grad():current=m.score(g['inputs'],g['features'],ids=s['tokens'])['tokens']
                if not torch.allclose(current,s['old'],atol=cfg['old_current_logprob_atol'],rtol=cfg['old_current_logprob_rtol']):
                    raise RuntimeError('Old/current on-policy probability gate failed')
        for g in groups:
            for index,(s,a) in enumerate(zip(g['samples'],g['advantages'])):
                self.tick();paired=self.method in ('papo','cfpo');trained=int(s['mask'].sum())
                weight=scale/len(groups)/len(g['samples'])
                if paired:weight=scale*trained/token_count
                if self.method=='axpo':
                    weight=scale/len(groups)*(1. if index in g['selected_prefixes'] or index>=g['original_count'] else 1/g['original_count'])
                factual=m.score(g['inputs'],g['features'],ids=s['tokens'])['tokens']
                beta=cfg[self.method].get('beta',cfg['beta']) if self.method in cfg else cfg['beta']
                loss,stats=completion_loss(factual,s['old'],s['reference'],a,s['mask'],cfg['epsilon'],beta,self.method)
                if paired:
                    p=cfg[self.method]
                    extra,perception=perception_loss(factual,s['corrupted_old'],self.method,p['gamma'],p['entropy_factual'],p['entropy_corrupted'])
                    loss=loss+extra;stats.update(perception)
                value=float(loss.detach());(weight*loss).backward();weighted_loss+=weight*value
                logs.append(stats|{'loss':value,'weight':weight,'view':g['view'],'trained_tokens':trained})
        return {'method':self.method,'mean_loss':weighted_loss/scale,'completions':len(logs),
                'extra_rollouts':self.extra_rollouts,'observer_calls':self.observer_calls,'terms':logs,
                'generation_replay_max_error':max(s['alignment']['max_error'] for g in groups for s in g['samples'] if 'alignment' in s)}
