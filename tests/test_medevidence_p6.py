"""Verify the all-wrong mechanism and the shared-gradient cap on synthetic data."""
import torch
from src.medevidence_p6 import training_advantages,common_coefficient
from src.medevidence_p4 import policy_terms


def main():
    for rewards in ([-1.]*4,[1.]*4,[-1.,1.,-1.,1.]):
        b=training_advantages(rewards,False,'B_GRPO');c=training_advantages(rewards,False,'C_NEGABS')
        assert torch.equal(c,torch.tensor(rewards))
        if len(set(rewards))==1:assert torch.equal(b,torch.zeros(4))
    r=[-.8,-.6,-.2,.1]
    assert torch.equal(training_advantages(r,True,'B_GRPO'),training_advantages(r,True,'C_NEGABS'))
    # A scalar log-probability surrogate: descending this loss decreases an all-wrong response's log probability.
    current=torch.tensor([-.7,-1.2],requires_grad=True);old=current.detach().clone();reference=old.clone()
    obj,kl=policy_terms(current,old,reference,-1.,torch.ones(2,dtype=torch.bool))
    (-obj).backward();assert torch.all(current.grad>0) and float(kl)==0
    current.grad=None;obj,_=policy_terms(current,old,reference,0.,torch.ones(2,dtype=torch.bool));(-obj).backward()
    assert torch.equal(current.grad,torch.zeros_like(current))
    lam=common_coefficient(2.,3.,5.);assert lam==.1 and max(lam*3/2,lam*5/2)<=.25
    try:training_advantages([-.5]*4,False,'C_NEGABS')
    except ValueError:pass
    else:raise AssertionError('Nonbinary negative reward accepted')
    print('P6 synthetic objective checks passed')


if __name__=='__main__':main()
