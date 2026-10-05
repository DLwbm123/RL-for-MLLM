"""Deterministic geometric controls; these are not expert-confirmed negatives."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from scipy import ndimage
from src.data import bbox, smart_size, write_json, sha


def region_tokens(box, width, height, gh, gw):
    """All merged cells with positive-area intersection; half-open geometry."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return []
    xx, yy = np.meshgrid(np.arange(gw), np.arange(gh))
    selected = (xx * width / gw < x2) & ((xx + 1) * width / gw > x1) & (yy * height / gh < y2) & ((yy + 1) * height / gh > y1)
    return np.flatnonzero(selected).tolist()


def tissue_grid(image, gh, gw):
    a = np.asarray(image.convert('L'))
    h, w = a.shape
    # Conservative annotation candidates: thin bright connected components.
    # This excludes possible text/calipers from controls and replacement sources;
    # it is a heuristic, so development overlays still require visual review.
    components,n=ndimage.label(a>=220)
    annotation=np.zeros_like(a,dtype=bool)
    for label,sl in enumerate(ndimage.find_objects(components),1):
        if sl is None:continue
        patch=components[sl]==label;area=int(patch.sum());ph,pw=patch.shape
        if 6<=area<=1200 and max(ph,pw)>=4 and (area/(ph*pw)<.7 or max(ph,pw)/max(1,min(ph,pw))>3):
            annotation[sl]|=patch
    annotation=ndimage.binary_dilation(annotation,iterations=1)
    valid = np.zeros((gh, gw), bool)
    for y in range(gh):
        for x in range(gw):
            p = a[int(y*h/gh):max(int((y+1)*h/gh),int(y*h/gh)+1), int(x*w/gw):max(int((x+1)*w/gw),int(x*w/gw)+1)]
            valid[y,x] = p.mean() > 15 and p.std() > 5 and (p < 8).mean() < .5 and (p > 245).mean() < .08
            ap=annotation[int(y*h/gh):max(int((y+1)*h/gh),int(y*h/gh)+1),int(x*w/gw):max(int((x+1)*w/gw),int(x*w/gw)+1)]
            valid[y,x] &= not ap.any()
    valid[[0,-1],:] = False
    valid[:,[0,-1]] = False
    return valid


def source_ring(selected, valid, forbidden, gh, gw):
    target = np.zeros(gh * gw, bool); target[selected] = True; target = target.reshape(gh,gw)
    ring = ndimage.binary_dilation(target, structure=np.ones((3,3)), iterations=3) & ~target
    return np.flatnonzero(ring & valid & ~forbidden).tolist()


def build_regions(image, mask, box, seed=42, min_tokens=576, max_tokens=1024):
    width, height = image.size
    rh, rw = smart_size(height, width, min_tokens, max_tokens)
    gh, gw = rh//28, rw//28
    ids = region_tokens(box, width, height, gh, gw)
    yy, xx = np.divmod(ids, gw)
    eh, ew = int(yy.max()-yy.min()+1), int(xx.max()-xx.min()+1)
    valid = tissue_grid(image, gh, gw)
    forbidden = np.zeros((gh,gw),bool)
    # Exclude lesion + 10% box margin and the posterior column below it.
    x1,y1,x2,y2 = box
    expanded = [max(0,x1-.1*(x2-x1)), max(0,y1-.1*(y2-y1)), min(width,x2+.1*(x2-x1)), height]
    forbidden.reshape(-1)[region_tokens(expanded,width,height,gh,gw)] = True
    rng = np.random.default_rng(seed)
    candidates = [(y,x) for y in range(1,gh-eh) for x in range(1,gw-ew)]
    rng.shuffle(candidates)
    controls, diagnostics = [], []
    target_cy = (y1+y2)/2/height
    target_area = (x2-x1)*(y2-y1)/(width*height)
    for y,x in candidates:
        region = np.zeros((gh,gw),bool);region[y:y+eh,x:x+ew] = True
        cy = (y+eh/2)/gh
        if abs(cy-target_cy) > .10 or (region & forbidden).any(): continue
        if (region & valid).sum()/region.sum() < .9: continue
        # Distinct controls may overlap, but not almost coincide; all have equal token shape/count.
        idx = np.flatnonzero(region).tolist()
        if any(len(set(idx)&set(c['tokens']))/len(idx) > .75 for c in controls): continue
        source = source_ring(idx,valid,forbidden,gh,gw)
        if len(source)<4:continue
        # Translate the ORIGINAL box by integer merged cells, preserving fractional
        # cell offsets: equal pixel area/shape AND equal touched-token shape/count.
        dx=(x-int(xx.min()))*width/gw;dy=(y-int(yy.min()))*height/gh
        cb = [x1+dx,y1+dy,x2+dx,y2+dy]
        if set(region_tokens(cb,width,height,gh,gw))!=set(idx):continue
        controls.append({'box':cb,'tokens':idx,'source':source})
        diagnostics.append({'depth_error':float(abs(cy-target_cy)), 'token_error':0,
                            'area_relative_error':float(abs((cb[2]-cb[0])*(cb[3]-cb[1])/(width*height)-target_area)/target_area),
                            'valid_tissue_fraction':float((region&valid).sum()/region.sum())})
        if len(controls)==3:break
    source = source_ring(ids,valid,forbidden,gh,gw)
    reasons=[]
    if len(ids)<4: reasons.append('fewer_than_4_evidence_tokens')
    if len(controls)<3: reasons.append('fewer_than_3_matched_controls')
    if len(source)<4: reasons.append('insufficient_valid_local_replacement_source')
    if any(d['area_relative_error']>.25 for d in diagnostics):reasons.append('control_area_error_over_25pct')
    # E1/E3 are fixed definitions; no per-example model-score selection.
    mask_grid=np.asarray(Image.fromarray(mask.astype('uint8')*255).resize((gw,gh),Image.Resampling.NEAREST))>0
    e3=[max(0,x1-.1*(x2-x1)),max(0,y1-.1*(y2-y1)),min(width,x2+.1*(x2-x1)),min(height,y2+.1*(y2-y1))]
    # Wrong region is deterministic 50%-width displacement, not reward-selected.
    dx=(x2-x1)*.5*(1 if x2+(x2-x1)*.5<=width else -1)
    wrong=[max(0,x1+dx),y1,min(width,x2+dx),y2]
    wrong_ids=region_tokens(wrong,width,height,gh,gw)
    wrong_source=source_ring(wrong_ids,valid,forbidden,gh,gw)
    if len(wrong_source)<4: reasons.append('wrong_region_source_insufficient')
    return {'eligible':not reasons,'reasons':reasons,'grid':[gh,gw],'resized':[rh,rw],
            'evidence':{'box':box,'tokens':ids,'source':source}, 'controls':controls,
            'wrong':{'box':wrong,'tokens':wrong_ids,'source':wrong_source},
            'E1':np.flatnonzero(mask_grid).tolist(),'E3':region_tokens(e3,width,height,gh,gw),
            'matching':diagnostics,'mask_area_ratio':float(mask.mean()),'seed':seed}


def replace_features(features, selected, source):
    if not selected:
        return features.clone()
    if not source or set(selected)&set(source):
        raise ValueError('Replacement requires disjoint valid source tokens')
    result=features.clone()
    result[selected]=features[source].mean(dim=0)
    return result


def pixel_blur(image, box, radius=12):
    # Same fixed full-image Gaussian kernel followed by rectangular replacement for E and N.
    out=image.copy(); b=tuple(map(int,[np.floor(box[0]),np.floor(box[1]),np.ceil(box[2]),np.ceil(box[3])]))
    out.paste(image.filter(ImageFilter.GaussianBlur(radius)).crop(b),b)
    return out


def prepare_regions(root, output):
    import pandas as pd
    root,output=Path(root),Path(output)
    frame=pd.read_csv(output/'protocol/manifest.csv',dtype={'case_id':str})
    regions={}
    # Test region construction deferred until frozen rules and final authorized evaluation.
    for r in frame.loc[frame.split.isin(['train','validation'])].to_dict('records'):
        if not r['evidence_eligible']:continue
        im=Image.open(root/r['image_path']).convert('RGB')
        mask=np.asarray(Image.open(root/r['mask_path']))>0
        regions[r['image_id']]=build_regions(im,mask,json.loads(r['bbox_xyxy']))
    write_json(output/'protocol/regions.json',regions)
    summary={}
    for s in ['train','validation']:
        subset=frame[frame.split==s]
        summary[s]={'images':len(subset),'eligible':sum(regions.get(k,{}).get('eligible',False) for k in subset.image_id)}
    write_json(output/'protocol/region_summary.json',summary)
    # Development-only review: 50 images balanced across pathology and area quartiles.
    dev=frame[frame.split=='train'].copy();dev['area_bin']=pd.qcut(dev.mask_area_ratio,4,labels=False,duplicates='drop')
    selected=dev.groupby(['pathology','area_bin'],group_keys=False).sample(n=5,random_state=42)
    special=dev.loc[(dev.mask_components!=1)|dev.metadata_geometry_mismatch]
    selected=pd.concat([selected,special]).drop_duplicates('image_id').iloc[:50]
    extra=dev.loc[~dev.image_id.isin(selected.image_id)].sample(n=50-len(selected),random_state=42)
    selected=pd.concat([selected,extra]);review=output/'review';review.mkdir(exist_ok=True)
    selected[['image_id','case_id']].to_csv(review/'review_50.csv',index=False)
    for page in range(5):
        canvas=Image.new('RGB',(1500,1500),'white'); draw=ImageDraw.Draw(canvas)
        for cell,r in enumerate(selected.iloc[page*10:(page+1)*10].to_dict('records')):
            im=Image.open(root/r['image_path']).convert('RGB'); im.thumbnail((280,310))
            x=(cell%5)*300;y=(cell//5)*750
            canvas.paste(im,(x,y+45)); draw.text((x+2,y+5),r['image_id']+' '+r['pathology'],fill='black')
            region=regions.get(r['image_id'])
            anno=im.copy();d=ImageDraw.Draw(anno)
            if region:
                sx=im.width/r['width'];sy=im.height/r['height']
                for item,color in [(region['evidence'],'red')]+[(c,'lime') for c in region['controls']]:
                    b=item['box'];d.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],outline=color,width=2)
                gh,gw=region['grid']
                for i in range(1,gw):d.line((i*im.width/gw,0,i*im.width/gw,im.height),fill=(70,70,100))
                for i in range(1,gh):d.line((0,i*im.height/gh,im.width,i*im.height/gh),fill=(70,70,100))
            mask=Image.open(root/r['mask_path']).resize(im.size,Image.Resampling.NEAREST)
            edge=mask.filter(ImageFilter.FIND_EDGES);anno.paste((255,180,0),(0,0,im.width,im.height),edge)
            canvas.paste(anno,(x,y+390))
        canvas.save(review/f'page_{page+1}.jpg',quality=90)
    lock=json.loads((output/'protocol/protocol_lock.json').read_text())
    lock['hashes']['regions.json']=sha((output/'protocol/regions.json').read_bytes())
    lock['region_rules']={'K':3,'depth_tolerance':.10,'max_area_relative_error':.25,'min_source_tokens':4,'posterior_column_excluded':True,'pixel_blur_radius':12,'overlap':'positive_area','test_regions':'deferred'}
    write_json(output/'protocol/protocol_lock.json',lock)
    return summary
