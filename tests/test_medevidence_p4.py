"""Synthetic only: geometry, strict denominators, PPO masks/direction and schedule."""
import json
import numpy as np
import torch
from src.medevidence import normalized,matching
from src.medevidence_p2 import enrich
from src.medevidence_p4 import reward,giou,advantages,policy_terms,schedule,best_single,paired_counts,gate


def main():
    a=[0,0,100,100];b=[200,200,300,300];dump=json.dumps
    assert reward(dump([a]),[a])==1 and reward(dump([a,b]),[a,b])==1
    assert reward('[]',[a])==-1 and reward('[]',[])==1 and reward(dump([a]),[])==-1
    assert reward(dump([a,a]),[a])==0 and reward(dump([a,b]),[a])==0 and reward(dump([a]),[a,b])==0
    assert giou(a,b)<0 and reward(dump([b]),[a])<0
    for text,truncated in [('bad',False),('[[]]',False),('[[0,0,0,2]]',False),('[[0,0,1001,20]]',False),(dump([a]*65),False),(dump([a]),True)]:assert reward(text,[a],truncated)==-1
    assert normalized([[0,0,50,100]],100,200)==[[0,0,500,500]]
    assert reward('[[0,0,1000,1000]]',[[0,0,1000,1000]])==1
    assert matching([a,b],[a,a])['matches']==1 and not matching([a],[a,a])['strict']
    row={'boxes':[a],'width':1000,'height':1000};invalid=enrich({'image_id':'a','case_id':'a','loc_text':'bad','truncated':False},row)
    assert invalid['gt']==1 and not invalid['strict'] and invalid['pred'] is None
    assert best_single([{'loc_text':dump([a,a]),'truncated':False}], [a])['best_iou']==0
    assert torch.equal(advantages([1,1,1,1]),torch.zeros(4))
    assert torch.allclose(advantages([-1,-1,1,1]),torch.tensor([-1.,-1.,1.,1.]))
    current=torch.tensor([-.2,-.7,float('nan')],requires_grad=True);old=torch.tensor([-.3,-.8,0.]);ref=torch.tensor([-.4,-.5,0.]);mask=torch.tensor([True,True,False])
    j,kl=policy_terms(current,old,ref,1.,mask)
    d=ref[:2]-current[:2];assert torch.allclose(kl,(d.exp()-d-1).mean()) and kl>=0
    assert torch.allclose(j,torch.tensor(.1).exp())
    (-j+.01*kl).backward();assert current.grad[2]==0 and torch.isfinite(current.grad).all()
    j,kl=policy_terms(torch.tensor([0.]),torch.tensor([-1.]),torch.tensor([0.]),1.,torch.tensor([True]));assert abs(float(j)-1.2)<1e-6 and kl==0
    j,_=policy_terms(torch.tensor([-1.]),torch.tensor([0.]),torch.tensor([0.]),-1.,torch.tensor([True]));assert abs(float(j)+.8)<1e-6
    rows={str(i):{'split':'train','pathology':'yes' if i<149 else 'no','boxes':[a]*(1 if i<71 else 2) if i<149 else [],'case_id':str(i)} for i in range(405)}
    pools,plan=schedule(rows,[str(i) for i in range(149,405)]);assert len(plan)==256 and len({k for ids in plan for k in ids})==405
    assert all([len(rows[k]['boxes']) for k in ids]==[1,2,0,0] for ids in plan)
    assert plan==schedule(dict(reversed(list(rows.items()))),list(reversed(pools['negative'])))[1]
    assert not gate({'positive_best_of8_success':7},{'positive_distinguishable_fraction':1.},{'real_minus_shuffled_mean':1.})['passed']
    assert gate({'positive_best_of8_success':8},{'positive_distinguishable_fraction':.3},{'real_minus_shuffled_mean':.05})['passed']
    print('PASS: detached GIoU one-to-one penalties, strict failure denominator, fp32 PPO/KL/mask, fixed405-patient stratified schedule and gate')


if __name__=='__main__':main()
