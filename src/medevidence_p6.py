"""One experimental change: absolute negative advantages, relative positive ones."""
import torch
from src.medevidence_p4 import advantages


def training_advantages(rewards, positive, branch):
    if branch not in ('B_GRPO','C_NEGABS'):
        raise ValueError('No RL objective for this branch')
    r=torch.as_tensor(rewards,dtype=torch.float32).detach()
    if r.ndim!=1 or len(r)!=4 or not torch.isfinite(r).all():
        raise ValueError('Exactly four finite rewards required')
    if not positive and not torch.all((r==1)|(r==-1)):
        raise ValueError('Negative reward must be exact valid-empty +1 or error -1')
    return r if branch=='C_NEGABS' and not positive else advantages(r)


def common_coefficient(sft_norm, b_norm, c_norm):
    import math
    if not all(math.isfinite(x) and x>1e-8 for x in (sft_norm,b_norm,c_norm)):
        raise ValueError('Disconnected or nonfinite calibration gradient')
    return min(1.,.25*sft_norm/max(b_norm,c_norm))
