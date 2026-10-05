"""Independent patient-grouped RSNA development pilot; not a Kaggle split reproduction."""
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from src.data import QUESTIONS, write_json, sha, bbox
from src.regions import build_regions


def records_from_release(annotations, mappings):
    final=defaultdict(list)
    for a in annotations:
        if a['labelId'] in ['L_o8w','L_yd0','L_v8n']:
            final[a['SOPInstanceUID']].append(a)
    rows=[]
    for m in mappings:
        aa=final[m['SOPInstanceUID']]
        if not aa:continue  # Unadjudicated images have no target; never treat them as negatives.
        labels={a['labelId'] for a in aa}
        if len(labels)!=1:raise ValueError('Conflicting calculated labels')
        boxes=[]
        for a in aa:
            if a['labelId']!='L_v8n':continue
            d=a['data'];x,y,w,h=[d[k] for k in ['x','y','width','height']]
            if w<=0 or h<=0:raise ValueError('Invalid box')
            boxes.append([max(0,int(np.floor(x))),max(0,int(np.floor(y))),min(a['width'],int(np.ceil(x+w))),min(a['height'],int(np.ceil(y+h)))])
        rows.append({'image_id':m['subset_img_id'],'case_id':m['img_id'].split('_')[0],
                     'sop':m['SOPInstanceUID'],'pathology':'yes' if boxes else 'no',
                     'boxes':boxes,'width':aa[0]['width'],'height':aa[0]['height'],
                     'source_subset_group':m['subset_group']})
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Duplicate image id')
    return pd.DataFrame(rows)


def patient_split(frame,seed):
    # Mixed labels across longitudinal studies are valid; any-positive stratification is patient-level only.
    target=frame.groupby('case_id').pathology.agg(lambda x:int('yes' in set(x)))
    train,held=train_test_split(target.index,test_size=.30,stratify=target,random_state=seed)
    val,test=train_test_split(held,test_size=.50,stratify=target.loc[held],random_state=seed)
    assignment={c:s for s,ids in [('train',train),('validation',val),('test',test)] for c in ids}
    return frame.case_id.map(assignment)


def main():
    import pydicom
    root=Path(os.environ['DATA_ROOT']);out=Path(os.environ['OUTPUT_ROOT']);cfg=Path(os.environ['RUN_CONFIG'])
    config=json.loads(cfg.read_text());protocol=out/'protocol';protocol.mkdir(parents=True,exist_ok=True)
    if (protocol/'protocol_lock.json').exists():raise FileExistsError('Protocol already frozen')
    j=json.loads(next(root.glob('*kaggle_2018.json')).read_text())
    m=json.loads(next(root.glob('*mappings*.json')).read_text())
    frame=records_from_release(j['datasets'][0]['annotations'],m)
    frame['partition']=patient_split(frame,config['split_seed']);frame['split']=frame.partition
    # Pick patients independently of image appearance and model scores; one predetermined image per patient keeps compute bounded.
    for split,limit in [('train',config['train_cases']),('validation',config['validation_cases'])]:
        subset=frame[frame.partition==split].sort_values('image_id').drop_duplicates('case_id')
        selected,_=train_test_split(subset,train_size=limit,stratify=subset.pathology,random_state=config['seed'])
        frame.loc[frame.partition==split,'split']='unused_'+split
        frame.loc[selected.index,'split']=split
    assert frame.groupby('case_id').partition.nunique().max()==1
    archive=ZipFile(next(root.glob('*.zip')));members={Path(n).stem:n for n in archive.namelist() if n.endswith('.dcm')}
    derived=root/'derived_pilot_v1';derived.mkdir(exist_ok=True);regions={};review=[];syntaxes=set()
    for i,row in frame.iterrows():
        boxes=row.boxes;union=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)] if boxes else None
        frame.at[i,'bbox_xyxy']=json.dumps(union);frame.at[i,'mask_components']=len(boxes)
        frame.at[i,'evidence_eligible']=bool(len(boxes)==1);frame.at[i,'duplicate_group']=row.case_id
        frame.at[i,'mask_area_ratio']=0.;frame.at[i,'image_path']='';frame.at[i,'mask_path']=''
        if row.split not in ['train','validation']:continue
        ds=pydicom.dcmread(io.BytesIO(archive.read(members[row.sop])));syntaxes.add(str(ds.file_meta.TransferSyntaxUID))
        a=ds.pixel_array
        if boxes and a.shape!=(row.height,row.width):raise ValueError('DICOM annotation geometry mismatch')
        frame.at[i,'height'],frame.at[i,'width']=a.shape
        if a.dtype!=np.uint8 or ds.PhotometricInterpretation not in ['MONOCHROME1','MONOCHROME2']:
            raise ValueError('Unexpected pixel encoding; specify intensity processing before proceeding')
        if ds.PhotometricInterpretation=='MONOCHROME1':a=255-a
        image=Image.fromarray(a).convert('RGB');mask=np.zeros(a.shape,dtype=np.uint8)
        for x1,y1,x2,y2 in boxes:mask[y1:y2,x1:x2]=255
        image.save(derived/(row.image_id+'.png'));Image.fromarray(mask).save(derived/(row.image_id+'_mask.png'))
        frame.at[i,'image_path']=str((derived/(row.image_id+'.png')).relative_to(root))
        frame.at[i,'mask_path']=str((derived/(row.image_id+'_mask.png')).relative_to(root));frame.at[i,'mask_area_ratio']=float((mask>0).mean())
        if len(boxes)==1:
            reg=build_regions(image,mask>0,union,chest=True);regions[row.image_id]=reg
            if row.split=='train' and reg['eligible'] and len(review)<12:
                thumb=image.resize((320,320));draw=ImageDraw.Draw(thumb)
                for r,color in [(reg['evidence'],'red')]+[(c,'lime') for c in reg['controls']]:
                    draw.rectangle([v*320/1024 for v in r['box']],outline=color,width=2)
                review.append(thumb)
    frame['boxes']=frame.boxes.map(json.dumps)
    frame.to_csv(protocol/'manifest.csv',index=False)
    frame[['image_id','case_id','partition','split']].to_csv(protocol/'splits.csv',index=False)
    write_json(protocol/'templates.json',QUESTIONS);write_json(protocol/'regions.json',regions)
    write_json(protocol/'schema_mapping.json',{'dataset':'RSNA 2018 adjudicated MD.ai archive','target':'Calculated final labels only; Lung Opacity vs two no-opacity classes','labels':{'L_o8w':'no','L_yd0':'no','L_v8n':'yes'},'patient_key':'NIH image filename prefix in mapping','source_subset_group':'retained but not interpreted as an official train/test split','split':'new patient-grouped 70/15/15, not competition reproduction','multiple_boxes':'enclosing union for localization; excluded from mechanism audit','negative_images':'classification only; no synthetic boxes','duplicates':'No pixel duplicate scan performed','pixel_processing':'8-bit native intensities; MONOCHROME1 inverted; no per-image normalization','transfer_syntaxes':sorted(syntaxes)})
    summary={'mapped_images':len(m),'unadjudicated_excluded':len(m)-len(frame),'images':len(frame),'patients':int(frame.case_id.nunique()),'split_images':frame.split.value_counts().to_dict(),'active_labels':{str(k):int(v) for k,v in frame[frame.split.isin(['train','validation'])].groupby(['split','pathology']).size().items()},'eligible':{s:sum(regions.get(k,{}).get('eligible',False) for k in frame[frame.split==s].image_id) for s in ['train','validation']},'test_pixels_read':False,'patient_partition_disjoint':True}
    write_json(protocol/'data_audit.json',summary)
    canvas=Image.new('RGB',(1280,960),'white')
    for i,im in enumerate(review):canvas.paste(im,((i%4)*320,(i//4)*320))
    (out/'review').mkdir(exist_ok=True);canvas.save(out/'review/geometry.jpg')
    write_json(protocol/'protocol_lock.json',{'version':'rsna-pilot-1','hashes':{n:sha((protocol/n).read_bytes()) for n in ['manifest.csv','splits.csv','templates.json','schema_mapping.json','regions.json']},'config_sha256':sha(cfg.read_bytes()),'test_access':'sealed; metadata partition only','region_rules':{'K':3,'posterior_column_excluded':False,'exclusion':'annotated single box plus 10% margin','valid':'nonblack grid cells, excluding outer grid border','controls':'geometric candidates, not expert-confirmed normal lung','multibox_evidence':'excluded','depth_tolerance':.1,'pixel_blur_radius':12}})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
