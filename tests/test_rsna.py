import pandas as pd
from scripts.prepare_rsna import records_from_release, patient_split


def test_adjudication_and_patient_partition():
    mappings=[{'SOPInstanceUID':str(i),'subset_img_id':str(i),'img_id':f'{i//2:08d}_{i%2:03d}.png','subset_group':0} for i in range(120)]
    annotations=[{'SOPInstanceUID':str(i),'labelId':'L_v8n' if i%4==1 else 'L_o8w','width':1024,'height':1024,'data':{'x':1,'y':2,'width':10,'height':20}} for i in range(119)]
    annotations.append(dict(annotations[1],data={'x':40,'y':50,'width':10,'height':20}))
    annotations.append(dict(annotations[0],labelId='reader_label'))
    frame=records_from_release(annotations,mappings)
    assert len(frame)==119 and frame.iloc[0].pathology=='no' and frame.iloc[0].boxes==[]
    assert frame.iloc[1].pathology=='yes' and len(frame.iloc[1].boxes)==2
    frame['split']=patient_split(frame,42)
    assert frame.groupby('case_id').split.nunique().max()==1
    assert set(frame.split)=={'train','validation','test'}
