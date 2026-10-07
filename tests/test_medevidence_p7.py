"""Check finite-action rewards, masking, gradients and abstention accounting on CPU."""
import torch
from src.medevidence_p7 import action_targets,actor_loss,selection_stats,Actor,overlaps


def main():
    boxes=[[0,0,10,10],[20,20,30,30]];gt=[[0,0,10,10]]
    v,r,g=action_targets(boxes,gt);assert v.sum()==4 and r[0]==1 and r[1]==-1 and g.tolist()==[True]+[False]*9
    _,r2,g2=action_targets(boxes,[]);assert r2[8]==1 and g2[8] and not g2[:8].any()
    _,r3,g3=action_targets([],gt);assert g3[9] and r3[8]==-1 and abs(float(r3[9])+.2)<1e-6
    assert overlaps(boxes,gt)[0,0]==1 and overlaps(boxes,gt)[1,0]==0
    logits=torch.zeros(1,10,requires_grad=True);valid=v[None];lp=logits.masked_fill(~valid,-torch.inf).log_softmax(-1)
    loss=actor_loss(lp,valid,r[None],g[None],lp.detach());loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0,0]<0 and logits.grad[0,1]>0 and logits.grad[0,2]==0
    # Exact expected-return differentiation equals the enumerated score-function gradient.
    exact=logits.grad.clone();logits.grad=None;lp=logits.masked_fill(~valid,-torch.inf).log_softmax(-1)
    safe=torch.where(valid,lp,torch.zeros_like(lp));surrogate=-(lp.exp().detach()*r[None]*safe).sum();surrogate.backward()
    assert torch.allclose(exact,logits.grad,atol=1e-7)
    model=Actor(12);x=torch.randn(1,10,12);lp=model(x,valid);actor_loss(lp,valid,r[None],g[None]).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    before=model(x,valid).detach();x[:,2:8]+=1000;assert torch.equal(before,model(x,valid).detach())
    all_reject=selection_stats([{'positive':True,'action':9,'correct':False,'reward':-.2},{'positive':False,'action':9,'correct':False,'reward':-.2}])
    assert all_reject['coverage']==0 and all_reject['answered_risk'] is None and all_reject['positive_supported_success_rate']==0
    assert not torch.cuda.is_initialized();print('P7 candidate/actor/abstention CPU checks passed')


if __name__=='__main__':main()
