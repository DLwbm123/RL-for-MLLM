import torch
from src.data import LABELS
import torch.nn.functional as F


def token_logps(logits, ids, answer_mask):
    """Mask excludes prompt, image placeholders, padding; predicts next token."""
    mask=answer_mask[:,1:].bool()
    values=F.log_softmax(logits[:,:-1].float(),dim=-1).gather(-1,ids[:,1:,None]).squeeze(-1)
    count=mask.sum(-1)
    if (count==0).any(): raise ValueError('Empty answer span')
    total=(values*mask).sum(-1)
    return {'sum':total,'mean':total/count,'length':count,'tokens':values,'mask':mask}


def correctness(answer, target):
    label=answer.strip()
    if label not in LABELS:return -.1
    return float(label==target)


def ced_reward(correct, de, dn, eps=1e-6):
    """Detached teacher-forced author-code mean-token score; population std."""
    with torch.no_grad():
        std=dn.float().std(unbiased=False)
        m=torch.tanh((de-dn.mean())/(std+eps))
        gate=.5*(1+torch.tanh(m/.20))
        reward=correct*gate+.10*m
    return reward,{'m':float(m),'std':float(std),'gate':float(gate),'saturated':bool(gate<.001 or gate>.999)}


def advantages(rewards):
    rewards=rewards.float()
    std=rewards.std(unbiased=False)
    if float(std)<1e-8:return torch.zeros_like(rewards)
    return (rewards-rewards.mean())/(std+1e-8)


def grpo_loss(new_logps, old_logps, advantage, clip=.2):
    ratio=torch.exp(new_logps-old_logps.detach())
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()


def evidence_loss(original, evidence, controls, target_index, weight=1., delta_rel=.05, delta_abs=.02):
    de=original[target_index]-evidence[target_index]
    dn=original[target_index]-controls[:,target_index]
    margin=de-dn.mean()
    loss=weight*(F.relu(delta_rel-margin)+F.relu(delta_abs-de))
    return loss,{'D_E':de,'D_N':dn.mean(),'M':margin}


def stability_loss(original, controls):
    target=F.softmax(original.detach(),dim=-1).expand_as(controls)
    return F.kl_div(F.log_softmax(controls,dim=-1),target,reduction='batchmean')
