"""One runnable CPU check for the independent pilot contracts."""
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from src.medevidence import *


def main():
    assert normalized([[1,2,10,20],[0,0,1,1]],100,100)==[[0,0,10,10],[10,20,100,200]]
    assert parse_boxes('[]') == ([],None)
    for bad in ('null','[1,2,3,4]','[[0,0,0,1]]','[[true,0,1,2]]','[[0,0,NaN,1]]','```[]```'):
        assert parse_boxes(bad)[0] is None
    assert parse_boxes('[[20,20,30,30],[0,0,10,10]]')[0][0]==[0,0,10,10]
    assert matching([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]])['strict'] is False
    assert matching([[0,0,10,10],[20,20,30,30]],[[20,20,30,30],[0,0,10,10]])['matches']==2
    assert matching([[0,0,10,10]],None)['single_iou']==0
    cfg={'safety_margin':.1,'control_candidates_max':128,'nonblack_intensity_threshold':5,
         'nonblack_fraction_min':.95,'blank_rgb':[0,0,0],'mask_rgb':[128]*3,'blur_radius':12}
    image=Image.new('RGB',(100,100),'white');target=[5,10,25,40]
    control,_=control_box(image,target,cfg,np.random.default_rng(42));assert control==[75,10,95,40]
    a=intervention(image,target,'gray',cfg);b=intervention(image,control,'gray',cfg)
    assert ((np.asarray(a)==128).all(-1)).sum()==((np.asarray(b)==128).all(-1)).sum()==600
    assert not np.array_equal(np.asarray(intervention(image,target,'blank',cfg)),np.asarray(intervention(image,control,'blank',cfg)))
    for box in ([0,0,10,10],[90,90,100,100],[0,0,100,100]):
        crop=crop_box(box,100,100);assert crop[0]<=box[0] and crop[2]>=box[2]
    rows={str(i):{'split':'train','case_id':str(i),'pathology':'yes' if i<4 else 'no','boxes':[[5,10,25,40]] if i<4 else []} for i in range(8)}
    views={'crops':{k:[{}] for k in rows},'pairs':{k:{} for k in rows}}
    plan=schedules(rows,views,512,17);assert plan==schedules(rows,views,512,17)
    assert all(sum(rows[k]['pathology']=='yes' for k in p['full'])==2 for p in plan)
    assert all(rows[p['loc']]['pathology']==('yes' if i%2==0 else 'no') for i,p in enumerate(plan))
    assert exposures(plan[256:],rows,views,'M1')==exposures(plan[256:],rows,views,'M2')
    import torch
    keep=torch.tensor(.4,requires_grad=True);hide=torch.tensor(.4,requires_grad=True)
    dependency(keep,hide).backward();assert keep.grad==-1 and hide.grad==1
    x=torch.tensor(.2,requires_grad=True);y=torch.tensor(.2,requires_grad=True);common=x*x+y*y
    g1=torch.autograd.grad(common,(x,y),retain_graph=True);g2=torch.autograd.grad(common+0*dependency(x,y),(x,y));assert all(a==b for a,b in zip(g1,g2))
    # Real filesystem boundary, not merely an ID check.
    import pandas as pd
    from src.v2 import DevelopmentData
    d=tempfile.mkdtemp()
    try:
        root=Path(d);image.save(root/'image.png');image.save(root/'mask.png')
        frame=pd.DataFrame([{'image_id':'a','case_id':'a','split':'train','image_path':'image.png','mask_path':'mask.png'}])
        data=DevelopmentData(root,frame);data.install_guard();data.image('a')
        try: (root/'test.png').read_bytes()
        except PermissionError: pass
        else: raise AssertionError('Test path was not sealed')
    finally:
        # Audit hook seals directory opens too; remove the two known synthetic files without scanning.
        (root/'image.png').unlink();(root/'mask.png').unlink();root.rmdir()
    print(json.dumps({'status':'passed','checks':'geometry, parsing, matching, balanced schedules, loss gradient direction, zero dep, sealed filesystem'}))


if __name__=='__main__':main()
