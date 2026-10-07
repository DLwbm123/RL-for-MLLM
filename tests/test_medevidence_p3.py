"""CPU check: coverage, shared exposure, strict errors and crop cluster contract."""
from collections import Counter
from src.medevidence_p3 import select_patients,local_pairs,local_metrics
from src.medevidence import parse_boxes,matching
from src.medevidence_p2 import enrich


def main():
    rows={str(i):{'case_id':f'{i:04d}','split':'train','pathology':'yes' if i<71 else 'no','boxes':[[10,20,30,40]] if i<71 else [],'width':100,'height':100} for i in range(500)}
    old_fit=[str(i) for i in range(16)]+[str(i) for i in range(71,87)]
    selected,plan=select_patients(rows,old_fit);assert (selected,plan)==select_patients(dict(reversed(list(rows.items()))),old_fit)
    counts=Counter(k for step in plan for k in step);assert len(counts)==327 and sum(counts.values())==1024
    assert all(sum(rows[k]['pathology']=='yes' for k in step)==2 for step in plan)
    assert all(k in selected['negative'] for k in old_fit[16:])
    assert parse_boxes('[[0,0,1001,20]]')[1]=='invalid_coordinates' and parse_boxes('bad')[1]=='invalid_json'
    assert matching([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]])['strict'] is False
    assert enrich({'loc_text':'[]','truncated':True},rows['0'])['state']=='invalid'
    crops=[{'cluster':str(i),'scale':scale,'y':y,'p':p} for i in range(2) for scale in (2.,2.5) for y,p in ((1,.8),(0,.2))]
    result=local_metrics(crops);assert result['2.0']['pairs']==2 and result['2.0']['pair_correct']==1 and result['2.0']['independent_patients']==4
    import torch
    from types import SimpleNamespace
    from src.medevidence_p3_run import patient
    class TinyModel:
        def __init__(self):self.weight=torch.nn.Parameter(torch.tensor(1.));self.answer_grad_modes=[]
        def prompt(self,image,task):return task,None
        def encode(self,inputs):return None
        def score(self,inputs,features,answer,include_eos):
            if inputs=='A':self.answer_grad_modes.append(torch.is_grad_enabled())
            loss=self.weight*(3 if inputs=='L' else 5)
            return {'mean':-loss,'tokens':torch.stack([-loss,-loss]),'ids':torch.tensor([7,99])}
    context=SimpleNamespace(tick=lambda:None,rows={'x':{'boxes':[],'loc_target':'[]','pathology':'no'}},data=SimpleNamespace(image=lambda k:None),contracts={'x':{'A':{'target_ids':[7,99]},'L':{'target_ids':[7,99]}}})
    m=TinyModel();patient(context,m,'x',0);assert float(m.weight.grad)==3/4 and m.answer_grad_modes==[False] and float(m.weight)==1
    m=TinyModel();patient(context,m,'x',1);assert float(m.weight.grad)==(3+5)/4 and m.answer_grad_modes==[True] and float(m.weight)==1
    print('passed:327 identity-stable patients,shared256step 512/512 exposures,strict errors and patient-cluster crop aggregation')


if __name__=='__main__':main()
