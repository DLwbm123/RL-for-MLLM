"""CPU diagnostics for localization output, token spans and fixed patient samples."""
from collections import Counter
import json
import math
import numpy as np
from src.medevidence import normalized, parse_boxes, matching


def select_sets(rows, seed=17):
    rng = np.random.default_rng(seed)
    selected = {'Fit32': [], 'Ref32': []}
    for label in ('yes', 'no'):
        pool = sorted((k for k,r in rows.items() if r['split']=='train' and r['pathology']==label
                       and (label=='no' or len(r['boxes'])==1)), key=lambda k:rows[k]['case_id'])
        if len(pool)<32:
            raise ValueError(f'Insufficient fixed {label} stratum: {len(pool)}')
        pool = list(rng.permutation(pool))
        selected['Fit32'].extend(str(k) for k in pool[:16])
        selected['Ref32'].extend(str(k) for k in pool[16:32])
    assert not set(selected['Fit32']) & set(selected['Ref32'])
    return selected


def loc_schedule(fit, rows, steps=256):
    pools = {label:[k for k in fit if rows[k]['pathology']==label] for label in ('yes','no')}
    position = Counter()
    result = []
    for _ in range(steps):
        update=[]
        for label in ('yes','no','yes','no'):
            pool=pools[label]
            update.append(pool[position[label] % len(pool)])
            position[label]+=1
        result.append(update)
    return result


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def enrich(record, row):
    rec=dict(record)
    gt=normalized(row['boxes'],row['width'],row['height'])
    pred,error=parse_boxes(rec['loc_text'])
    if rec.get('truncated'): pred,error=None,'truncated'
    checked=matching(gt,pred)
    for key in ('matches','gt','pred','strict','all_covered','single_iou'):
        if key in rec and rec[key]!=checked[key]:
            if isinstance(rec[key],float) and math.isclose(rec[key],checked[key],abs_tol=1e-12):continue
            raise ValueError('Historical scorer disagreement: '+key)
    if 'parse_error' in rec and rec['parse_error']!=error:
        raise ValueError('Historical parser disagreement')
    rec.update(checked,parse_error=error,boxes=pred,y=int(bool(gt)),state='invalid' if error else ('valid_nonempty' if pred else 'valid_empty'))
    if gt:
        if not pred: category='no_valid_prediction'
        elif not rec['matches']: category='nonempty_all_IoU_below_05'
        elif rec['matches']<len(gt): category='partial_GT_matches'
        elif not rec['strict']: category='all_GT_with_extras'
        else: category='all_GT_without_extras'
        rec['failure_category']=category
    rec['coordinate_diagnostic']=[]
    # Diagnostic only: each predicted box chooses its highest-IoU GT, ties by center distance.
    # This can reuse one GT and is never counted as a formal one-to-one detection.
    for b in pred or []:
        if not gt:continue
        distance=lambda a:math.hypot((b[0]+b[2]-a[0]-a[2])/2000,(b[1]+b[3]-a[1]-a[3])/2000)
        a=max(gt,key=lambda a:(iou(a,b),-distance(a)))
        wr=(b[2]-b[0])/(a[2]-a[0]);hr=(b[3]-b[1])/(a[3]-a[1])
        rec['coordinate_diagnostic'].append({'best_IoU':iou(a,b),'center_distance_image_diagonal_units':distance(a)/math.sqrt(2),
                                            'width_ratio':wr,'height_ratio':hr,'area_ratio':wr*hr})
    return rec


def distribution(values):
    return {'n':len(values),'mean':float(np.mean(values)) if values else None,
            'quantiles_0_25_50_75_100':np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else None}


def summarize(records):
    n=len(records);positive=[r for r in records if r['y']];negative=[r for r in records if not r['y']]
    ratio=lambda a,b:a/b if b else None
    def group(rows):
        return {'n':len(rows),**{s:sum(r['state']==s for r in rows) for s in ('invalid','valid_empty','valid_nonempty')},
                'truncated':sum(r.get('truncated',False) for r in rows)}
    single=[r for r in positive if r['gt']==1]
    groups={name:[r for r in positive if r['failure_category']==name] for name in
            ('no_valid_prediction','nonempty_all_IoU_below_05','partial_GT_matches','all_GT_with_extras','all_GT_without_extras')}
    scored=all('p' in r for r in records)
    predicted=lambda r:bool(r.get('class_prediction',r.get('p',0)>=.5))
    cross=Counter((r['y'],int(predicted(r)),r['state']) for r in records) if scored else None
    valid=[r for r in records if r['state']!='invalid']
    inconsistent=[r for r in valid if ('p' in r) and predicted(r)!=(r['state']=='valid_nonempty')]
    output={'all_patients':n,'positive':group(positive),'negative':group(negative),
            'positive_empty_rate':ratio(sum(r['state']=='valid_empty' for r in positive),len(positive)),
            'positive_region_recall':ratio(sum(r['matches'] for r in positive),sum(r['gt'] for r in positive)),
            'GT_regions':sum(r['gt'] for r in positive),'positive_predicted_boxes_known':sum(r['pred'] or 0 for r in positive),
            'single_IoU_unconditional':distribution([r['single_iou'] for r in single]),
            'single_IoU_ge_05_count':sum(r['single_iou']>=.5 for r in single),
            'single_strict_success_count':sum(r['strict'] for r in single),
            'single_IoU_nonempty_supplement':distribution([r['single_iou'] for r in single if r['state']=='valid_nonempty']),
            'negative_nonempty_rate':ratio(sum(r['state']=='valid_nonempty' for r in negative),len(negative)),
            'negative_invalid_rate':ratio(sum(r['state']=='invalid' for r in negative),len(negative)),
            'positive_failure_categories':{k:{'patients':len(v),'GT_regions':sum(r['gt'] for r in v),
                                            'missed_GT_regions':sum(r['gt']-r['matches'] for r in v)} for k,v in groups.items()},
            'multi_patient_count':sum(r['gt']>1 for r in positive),'multi_only_one_prediction':sum(r['gt']>1 and r['pred']==1 for r in positive),
            'multi_all_covered':sum(r['gt']>1 and r['all_covered'] for r in positive),
            'multi_strict_success':sum(r['gt']>1 and r['strict'] for r in positive),
            'class_loc_inconsistent_valid_count':len(inconsistent) if scored else None,
            'class_loc_inconsistent_valid_rate':ratio(len(inconsistent),len(valid)) if scored else None,
            'class_loc_inconsistent_all_patient_denominator':ratio(len(inconsistent),n) if scored else None,
            'overall_joint_supplement':ratio(sum(predicted(r)==bool(r['y']) and r['strict'] for r in records),n) if scored else None,
            'always_no_empty_joint_reference':ratio(len(negative),n),
            'classification_loc_cross_table':[{'true':bool(y),'class_yes':bool(p),'loc_state':s,'patients':cross[y,p,s]} for y in (0,1) for p in (0,1) for s in ('invalid','valid_empty','valid_nonempty')] if scored else None}
    diagnostics=[v for r in positive for v in r['coordinate_diagnostic']]
    output['coordinate_diagnostics']={k:distribution([v[k] for v in diagnostics]) for k in ('best_IoU','center_distance_image_diagonal_units','width_ratio','height_ratio','area_ratio')}
    boxes=[tuple(b) for r in positive for b in r['boxes'] or []]
    output['template_diagnostic']={'predicted_boxes':len(boxes),'distinct_exact_boxes':len(set(boxes)),
                                   'largest_exact_template_count':max(Counter(boxes).values(),default=0),
                                   'normalized_center_width_height_std':np.std([[(b[0]+b[2])/2000,(b[1]+b[3])/2000,(b[2]-b[0])/1000,(b[3]-b[1])/1000] for b in boxes],axis=0).tolist() if boxes else None,
                                   'interpretation':'Descriptive output spread; insufficient alone to prove a fixed template or explain empty outputs'}
    return output


def token_contract(tokenizer, target):
    ids=tokenizer.encode(target,add_special_tokens=False)
    empty=tokenizer.encode('[]',add_special_tokens=False)
    eos=tokenizer.eos_token_id
    gt,alternative=ids+[eos],empty+[eos]
    prefix=0
    if target!='[]':
        while prefix<min(len(gt),len(alternative)) and gt[prefix]==alternative[prefix]:prefix+=1
        if prefix==min(len(gt),len(alternative)):raise ValueError('GT/empty have no EOS-inclusive divergence')
    else:prefix=None
    try:
        offsets=tokenizer(target,add_special_tokens=False,return_offsets_mapping=True)
        if offsets['input_ids']!=ids:raise ValueError('Tokenizer offset IDs differ from actual teacher IDs')
        spans=offsets['offset_mapping']
        alignment='actual_tokenizer_offsets'
    except (NotImplementedError,TypeError):
        spans=[];previous=''
        for i in range(len(ids)):
            text=tokenizer.decode(ids[:i+1],skip_special_tokens=False)
            if not text.startswith(previous) or not target.startswith(text):raise ValueError('Cannot verify decoded token boundary')
            spans.append((len(previous),len(text)));previous=text
        alignment='verified_incremental_decode_offsets'
    if tokenizer.decode(ids,skip_special_tokens=False)!=target:raise ValueError('Teacher target does not decode exactly')
    categories=[];tokens=[]
    for i,(a,b) in enumerate(spans):
        chars=target[a:b]
        kinds=set('digits' if c.isdigit() else 'structure' if c in '[], ' else 'other' for c in chars)
        raw=next(iter(kinds)) if len(kinds)==1 else 'mixed'
        category='first_divergence' if prefix==i else raw
        if category=='other':category='mixed'
        categories.append(category);tokens.append({'index':i,'id':ids[i],'char_span':[a,b],'characters':chars,'category':category,'raw_character_category':raw})
    categories.append('EOS');tokens.append({'index':len(ids),'id':eos,'char_span':None,'characters':None,'category':'EOS'})
    assert len(categories)==len(gt) and sum(Counter(categories).values())==len(gt)
    return {'target_ids':gt,'empty_ids':alternative,'common_prefix_length':prefix,'common_prefix_ids':gt[:prefix] if prefix is not None else None,
            'GT_divergence_token':gt[prefix] if prefix is not None else None,'empty_divergence_token':alternative[prefix] if prefix is not None else None,
            'tokens':tokens,'categories':categories,'alignment':alignment,'patient_loss_weight':1.,'per_token_weight_in_patient_NLL':1/len(gt)}


def token_loss_parts(logps, contract):
    values=np.asarray(logps,float)
    if len(values)!=len(contract['categories']):raise ValueError('Token mask length mismatch')
    return {category:{'tokens':sum(c==category for c in contract['categories']),
                      'mean_NLL':float(-values[[c==category for c in contract['categories']]].mean()) if category in contract['categories'] else None,
                      'contribution_to_sequence_mean_NLL':float(-values[[c==category for c in contract['categories']]].sum()/len(values))}
            for category in ('first_divergence','digits','structure','mixed','EOS')}
