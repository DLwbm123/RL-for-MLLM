"""General RL mechanisms for the existing completion scorer; no dataset access."""
from contextlib import contextmanager
import math
import types
import torch
from src.objectives import advantages


PRIORITY=('visurf','rl_zvp','papo','cfpo','active_o3','axpo','defacto')


def reward_vector(rewards):
    r=torch.as_tensor(rewards,dtype=torch.float32).detach()
    if r.ndim!=1 or len(r)<2 or not torch.isfinite(r).all():
        raise ValueError('At least two finite scalar rewards required')
    return r


def visurf_advantages(rewards,target_reward):
    """ViSurf Eqs. 6–7, including label-reward smoothing from section 3.4."""
    r=reward_vector(rewards)
    if not math.isfinite(target_reward):raise ValueError('Nonfinite label reward')
    y=r.mean() if r.max()>=target_reward else r.new_tensor(target_reward)
    return advantages(torch.cat([r,y.reshape(1)]))


def zvp_advantages(rewards,entropies,alpha):
    """RL-ZVP Eq. 5. Entropy is over the vocabulary, not the sampled token."""
    r=reward_vector(rewards)
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('Positive finite ZVP scale required')
    if len(entropies)!=len(r) or not torch.all((r==-1)|(r==0)|(r==1)):
        raise ValueError('ZVP requires binary correctness rewards and one entropy span per response')
    out=[];relative=advantages(r)
    for i,h in enumerate(entropies):
        h=torch.as_tensor(h,dtype=torch.float32).detach()
        if h.ndim!=1 or not len(h) or not torch.isfinite(h).all() or (h<0).any():
            raise ValueError('Invalid unpadded token entropies')
        if r.std(unbiased=False)>1e-8:a=torch.full_like(h,relative[i])
        elif r[i]>0:a=alpha*h
        else:a=-alpha*(h.max()-h)
        out.append(a)
    return out


def author_advantages(rewards,method):
    """Pinned PAPO/CFPO and DeFacto trainers use sample standard deviation."""
    r=reward_vector(rewards)
    if method not in ('papo','cfpo','defacto'):raise ValueError('Unknown author normalization')
    return (r-r.mean())/(r.std()+ (1e-4 if method=='defacto' else 1e-6))


def completion_loss(current,old,reference,advantage,mask,epsilon,beta,method=None):
    if not 0<epsilon<1 or not math.isfinite(beta) or beta<0:
        raise ValueError('Invalid PPO/KL coefficients')
    mask=torch.as_tensor(mask,device=current.device,dtype=torch.bool)
    if current.ndim!=1 or current.shape!=old.shape or current.shape!=reference.shape or current.shape!=mask.shape:
        raise ValueError('Completion probability/mask shape mismatch')
    a=torch.as_tensor(advantage,device=current.device,dtype=torch.float32).detach()
    if a.ndim:
        if a.shape!=current.shape:raise ValueError('Token advantage span mismatch')
        a=a[mask]
    if not torch.isfinite(a).all():raise ValueError('Nonfinite advantage')
    current=current[mask].float();old=old[mask].detach().float();reference=reference[mask].detach().float()
    if not len(current) or not all(torch.isfinite(x).all() for x in (current,old,reference)):
        raise FloatingPointError('Empty/nonfinite completion logprobs')
    d=current-old
    if method in ('papo','cfpo'):d=d.clamp(-20.,20.)
    high=.3 if method in ('papo','cfpo') else .4 if method=='axpo' else .28 if method=='rl_zvp' else epsilon
    ratio=d.exp();surrogate=torch.minimum(ratio*a,ratio.clamp(1-epsilon,1+high)*a)
    if method in ('papo','cfpo'):surrogate=torch.where(a<0,torch.maximum(surrogate,3.*a),surrogate)
    objective=surrogate.sum() if method=='visurf' else surrogate.mean()
    if method in ('visurf','rl_zvp','defacto') or beta==0:kl=current.new_zeros(())
    else:
        delta=reference-current
        if method in ('papo','cfpo'):delta=delta.clamp(-20.,20.)
        k3=delta.exp()-delta-1
        if method in ('papo','cfpo'):k3=k3.clamp(-10.,10.)
        kl=k3.mean()
    if not all(torch.isfinite(x).all() for x in (ratio,objective,kl)):raise FloatingPointError('Nonfinite ratio/KL/objective')
    return -objective+beta*kl,{'policy_objective':float(objective.detach()),'reference_kl':float(kl.detach())}


def perception_loss(factual,corrupted,method,gamma,entropy_factual,entropy_corrupted):
    """Pinned author low_var_kl and cached, no-grad corrupted teacher."""
    if method not in ('papo','cfpo'):raise ValueError('Unknown perception objective')
    if factual.ndim!=1 or factual.shape!=corrupted.shape or not len(factual):raise ValueError('Paired response span mismatch')
    if not all(math.isfinite(v) and v>=0 for v in (gamma,entropy_factual,entropy_corrupted)):
        raise ValueError('Invalid perception coefficients')
    if not torch.isfinite(factual).all() or not torch.isfinite(corrupted).all():raise FloatingPointError('Nonfinite paired logprobs')
    corrupted=corrupted.detach().float()
    d=(corrupted-factual.float()).clamp(-20.,20.)
    k3=(d.exp()-d-1).clamp(-10.,10.)
    # The released PAPO code uses negative sampled log-probability as its entropy surrogate.
    loss=-gamma*k3.mean()-entropy_factual*factual.mean()-entropy_corrupted*corrupted.mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite perception objective')
    return loss,{'perception_k3':float(k3.mean().detach()),'factual_surprisal':float(-factual.mean().detach()),
                 'corrupted_surprisal':float(-corrupted.mean().detach())}


def axpo_advantages(rewards,prefix_lengths,continuation_rewards):
    """AXPO Eqs. 2–4; selected source continuation and repeated prefixes are masked."""
    r=reward_vector(rewards)
    if not torch.all((r==0)|(r==1)) or len(prefix_lengths)!=len(r):raise ValueError('AXPO requires binary outcome rewards')
    selected=set(continuation_rewards)
    tool={i for i,n in enumerate(prefix_lengths) if n is not None}
    if selected and (not selected<=tool or any(r[i]!=0 for i in tool)):
        raise ValueError('Resampling requires an entirely wrong, nonempty tool subgroup')
    base=advantages(r);branches={}
    for i,values in continuation_rewards.items():
        if not isinstance(i,int) or not 0<=i<len(r) or not isinstance(prefix_lengths[i],int) or prefix_lengths[i]<=0:
            raise ValueError('Invalid source prefix')
        values=reward_vector(values)
        if not torch.all((values==0)|(values==1)):raise ValueError('Binary continuation rewards required')
        recovered=r.clone();recovered[i]=float((values==1).any())
        base[i]=advantages(recovered)[i];branches[i]=advantages(values)
    return base,branches


def crossmodal_values(attention,values,image_mask,query_mask,sigma):
    """Pinned author image-mean prior and pooled valid-weight GMM statistics."""
    if attention.ndim!=4 or values.ndim!=4 or attention.shape[:2]!=values.shape[:2] or attention.shape[-1]!=values.shape[-2]:
        raise ValueError('Attention/value shape mismatch')
    length=attention.shape[-1]
    if attention.shape[-2]!=length or len(image_mask)!=length or len(query_mask)!=length:
        raise ValueError('CFPO supports full-sequence, no-cache scoring only')
    image_mask=torch.as_tensor(image_mask,device=attention.device,dtype=torch.bool)
    query_mask=torch.as_tensor(query_mask,device=attention.device,dtype=torch.bool)
    if not image_mask.any() or not query_mask.any() or (image_mask&query_mask).any() or not math.isfinite(sigma) or sigma<0:
        raise ValueError('Disjoint nonempty image/query spans required')
    qi=query_mask.nonzero().flatten();ii=image_mask.nonzero().flatten()
    cross=attention.index_select(-2,qi).index_select(-1,ii)
    salient=torch.zeros_like(cross)
    for b in range(len(cross)):
        valid=cross[b].detach()>1e-8;data=cross[b].detach()[valid]
        # Author singleton sample std is NaN and selects no edges; avoid its warning.
        if len(data)>1:salient[b]=((cross[b].detach()>data.mean()+sigma*data.std())&valid).to(cross.dtype)
    visual=values.index_select(-2,ii)
    delta=visual.sum((-2,-1),keepdim=True)/(visual.shape[-2]*visual.shape[-1])-visual
    correction=(cross*salient)@delta
    out=attention@values
    return out.index_add(-2,qi,correction),int(salient.sum())


@contextmanager
def cfpo_attention(model,image_mask,query_mask,intervene=True,sigma=2.):
    """Qwen2.5-VL 4.51 teacher scoring; keep open through backward if gradients are enabled."""
    # shortcut: full-sequence Qwen2.5-VL only, add a backbone-specific path before changing models.
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention,apply_multimodal_rotary_pos_emb,repeat_kv
    masks=[torch.as_tensor(v,dtype=torch.bool) for v in (image_mask,query_mask)]
    if masks[0].ndim!=1 or masks[0].shape!=masks[1].shape:raise ValueError('Prompt mask shape mismatch')
    layers=[m for m in model.modules() if isinstance(m,Qwen2_5_VLAttention)]
    if not layers:raise TypeError('CFPO requires supported Qwen2.5-VL text attention')
    stats={'salient_edges':0,'forwards':0};original=[]
    def forward(layer,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,output_attentions=False,
                use_cache=False,cache_position=None,position_embeddings=None):
        if past_key_value is not None or use_cache or position_embeddings is None or layer.attention_dropout!=0:
            raise ValueError('CFPO requires no cache, explicit RoPE and zero attention dropout')
        b,n,_=hidden_states.shape
        if n<len(masks[0]):raise ValueError('Prompt spans exceed score sequence')
        q=layer.q_proj(hidden_states).view(b,n,-1,layer.head_dim).transpose(1,2)
        k=layer.k_proj(hidden_states).view(b,n,-1,layer.head_dim).transpose(1,2)
        v=layer.v_proj(hidden_states).view(b,n,-1,layer.head_dim).transpose(1,2)
        q,k=apply_multimodal_rotary_pos_emb(q,k,*position_embeddings,layer.rope_scaling['mrope_section'])
        k=repeat_kv(k,layer.num_key_value_groups);v=repeat_kv(v,layer.num_key_value_groups)
        scores=q@k.transpose(-2,-1)/math.sqrt(layer.head_dim)
        if attention_mask is not None:scores=scores+attention_mask[:,:,:,:n]
        else:scores=scores.masked_fill(torch.ones(n,n,device=q.device,dtype=torch.bool).triu(1),float('-inf'))
        weights=torch.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
        if intervene:
            image=torch.cat([masks[0].to(q.device),torch.zeros(n-len(masks[0]),device=q.device,dtype=torch.bool)])
            query=torch.cat([masks[1].to(q.device),torch.ones(n-len(masks[1]),device=q.device,dtype=torch.bool)])
            out,count=crossmodal_values(weights,v,image,query,sigma);stats['salient_edges']+=count
        else:out=weights@v
        stats['forwards']+=1
        out=layer.o_proj(out.transpose(1,2).contiguous().reshape(b,n,-1))
        return out,weights if output_attentions else None,past_key_value
    try:
        for layer in layers:
            original.append((layer,layer.__dict__.get('forward')));layer.forward=types.MethodType(forward,layer)
        yield stats
    finally:
        for layer,value in original:
            if value is None:del layer.forward
            else:layer.forward=value
