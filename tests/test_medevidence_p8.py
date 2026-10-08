"""Check tied confidence, insufficient coverage and paired inference on CPU."""
import torch
from src.medevidence_p8 import matched_stats, matched_pair, evidence


def main():
    rs=[{'image_id':str(i),'positive':i<2,'action':0 if i<2 else 8,'confidence':1.,'correct':i!=1,'reward':-1. if i==1 else 1.} for i in range(4)]
    x=matched_stats(rs,.5)
    assert x['coverage']==.5 and x['answered_risk']==.25 and x['positive_supported_success_rate']==.25
    assert x==matched_stats(rs[::-1],.5)
    rs[0]['confidence']=2.;assert matched_stats(rs,.25)['answered_risk']==0
    rs[0]['action']=9;assert not matched_stats(rs,1.)['available']
    assert matched_stats(rs,.75)['coverage']==.75
    z=matched_pair(rs,rs,.5,30);assert z['comparison_minus_baseline']['answered_risk']['difference']==0 and not evidence(z,30)
    assert not evidence({'available':False},30)
    for r in rs:r['action']=9
    assert not matched_stats(rs,.25)['available']
    assert not torch.cuda.is_initialized();print('P8 matched coverage CPU checks passed')


if __name__=='__main__':main()
