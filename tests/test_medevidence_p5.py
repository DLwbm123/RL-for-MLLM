"""Coordinate frame safety and reward-decomposition regression checks."""
import json
from src.medevidence_p5 import canonical, target, decompose, diagnostic_gate
from src.medevidence_p4 import reward


def main():
    box = [[100,50,400,200]]
    for fmt in ('normalized','absolute'):
        text=target(box,1000,500,fmt,(840,420));b,e=canonical(text,fmt,(840,420))
        assert e is None and b==[[100.,100.,400.,400.]]
    assert canonical('[[0,0,841,1]]','absolute',(840,420))[1]
    assert canonical('[[false,0,10,10]]','absolute',(840,420))[1]
    assert canonical('[[0,0,NaN,10]]','absolute',(840,420))[1]
    assert canonical('[]','absolute',(840,420))==([],None)
    groups=[]
    for gt, texts in [(box,['[]','[[700,700,800,800]]','[[600,600,800,800]]','[]']),
                      ([],['[]','[[0,0,1,1]]','[]','[]'])]:
        groups.append({'gt':gt,'outputs':[{'loc_text':s,'truncated':False,'reward':reward(s,gt)} for s in texts]})
    d=decompose(groups)
    assert d['positive']['distinguishable_without_any_match']==1
    assert d['positive']['nonempty_geometry_distinguishable_groups']==1
    assert d['negative_penalty_rescaling_advantage_invariance_checked']
    assert d['positive']['positive_advantage_without_match_completions']>0
    print(json.dumps({'P5_synthetic_checks':'passed'}))


if __name__=='__main__':main()
