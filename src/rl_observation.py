"""Bounded crop actions and evidence views; no arbitrary tool execution."""
import json
import math
import re
import numpy as np
from src.medevidence import parse_boxes,matching,overlaps
from src.medevidence_p2 import iou


def observation_prompt(question,max_regions):
    return (question+' Decide whether a closer observation is needed. Return exactly '
            '<think>brief reasoning</think><tool_call>[[x1,y1,x2,y2],...]</tool_call> '
            f'for at most {max_regions} image crops with coordinates normalized to 0–1000, '
            'or <think>brief reasoning</think><answer>no</answer> or '
            '<think>brief reasoning</think><answer>yes</answer> without a crop. Do not output other tool types.')


def parse_observation(text,max_regions,truncated=False):
    if type(max_regions) is not int or max_regions<1:raise ValueError('Positive crop limit required')
    match=re.fullmatch(r'\s*<think>(.*?)</think>\s*(?:<tool_call>(.*?)</tool_call>|<answer>(no|yes)</answer>)\s*',text,re.S)
    if truncated or match is None:return {'valid':False,'boxes':None,'answer':None,'tool':bool('<tool_call>' in text)}
    if match[2] is None:return {'valid':True,'boxes':[],'answer':match[3],'tool':False}
    boxes,error=parse_boxes(match[2],limit=max_regions)
    return {'valid':not error and bool(boxes),'boxes':boxes if not error else None,'answer':None,'tool':True}


def tool_prefix(tokens,tokenizer):
    """Fork only at an exact token boundary following the opening tool tag."""
    for n in range(1,len(tokens)+1):
        if tokenizer.decode(tokens[:n],skip_special_tokens=False).endswith('<tool_call>'):return n
    return None


def pixel_box(box,size):
    boxes,error=parse_boxes(json.dumps([box]))
    if error:raise ValueError('Illegal normalized box')
    w,h=size;b=boxes[0]
    out=(math.floor(b[0]*w/1000),math.floor(b[1]*h/1000),math.ceil(b[2]*w/1000),math.ceil(b[3]*h/1000))
    if not 0<=out[0]<out[2]<=w or not 0<=out[1]<out[3]<=h:raise ValueError('Crop lost pixel support')
    return out


def active_reward(action,gt,observed_answer,min_area,max_area):
    """ACTIVE-o3-inspired heuristics plus verified joint answer/region outcome."""
    if not 0<min_area<max_area<=1:raise ValueError('Invalid crop area bounds')
    if not action['valid']:return {'reward':-1.,'correct':0.,'answer_correct':False,'evidence_success':False}
    boxes=action['boxes'];matched=matching(gt,boxes)
    answer_correct=observed_answer==('yes' if gt else 'no')
    evidence_success=matched['strict'] if gt else not boxes
    correct=float(answer_correct and evidence_success)
    areas=[(b[2]-b[0])*(b[3]-b[1])/1e6 for b in boxes]
    area_ok=all(min_area<=a<=max_area for a in areas)
    nonoverlap=all(iou(a,b)<=.1 for i,a in enumerate(boxes) for b in boxes[i+1:])
    coverage=matched['matches']/len(gt) if gt else float(not boxes)
    heuristic=(1.+float(area_ok)+float(nonoverlap)+coverage)/4
    return {'reward':correct+.25*heuristic,'correct':correct,'answer_correct':answer_correct,
            'evidence_success':evidence_success,'coverage':coverage,'regions':len(boxes)}


def random_patch_mask(image,grid,rng,ratio):
    gh,gw=grid
    if type(gh) is not int or type(gw) is not int or gh*gw<2 or not 0<ratio<1:
        raise ValueError('Invalid random patch masking geometry')
    count=min(gh*gw-1,max(1,math.ceil(gh*gw*ratio)))
    selected=rng.choice(gh*gw,size=count,replace=False).tolist();out=image.copy();w,h=image.size
    for index in selected:
        y,x=divmod(index,gw)
        out.paste((0,0,0),(math.floor(x*w/gw),math.floor(y*h/gh),math.ceil((x+1)*w/gw),math.ceil((y+1)*h/gh)))
    return out,selected


def defacto_views(image,gt,controls,complete_evidence):
    """Use registered target boxes instead of introducing an external detector."""
    for boxes in (gt,controls):
        if parse_boxes(json.dumps(boxes))[1]:raise ValueError('Illegal evidence/control geometry')
    if not gt:return [('pos',image.copy(),'no')]
    if complete_evidence is not True:raise PermissionError('Counterfactual Unknown requires registered complete target-evidence support')
    if not controls or any(overlaps(a,b) for a in gt for b in controls):raise ValueError('Disjoint control boxes required')
    if any(overlaps(a,b) for boxes in (gt,controls) for i,a in enumerate(boxes) for b in boxes[i+1:]):
        raise ValueError('Area-matched masks require nonoverlapping rectangles within each view')
    area=lambda boxes:sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
    if not np.isclose(area(gt),area(controls),rtol=0,atol=1e-6):raise ValueError('Control mask area must match evidence mask area')
    evidence=image.copy();control=image.copy()
    for b in gt:evidence.paste((0,0,0),pixel_box(b,image.size))
    for b in controls:control.paste((0,0,0),pixel_box(b,image.size))
    return [('pos',image.copy(),'yes'),('cf',evidence,'unknown'),('rand',control,'yes')]


def evidence_prompt(question):
    return (question+' Return only a JSON object with keys "answer" and "boxes". '
            'The answer must be "yes", "no", or "unknown". Supply supporting boxes for "yes", '
            'normalized to 0–1000 as [[x1,y1,x2,y2],...]. Use [] for "no" or "unknown". '
            'Answer "unknown" when the required visual evidence is missing.')


def parse_evidence(text,truncated=False):
    try:
        def unique(pairs):
            out={}
            for k,v in pairs:
                if k in out:raise ValueError('Duplicate evidence field')
                out[k]=v
            return out
        value=json.loads(text,object_pairs_hook=unique)
        if truncated or not isinstance(value,dict) or set(value)!={'answer','boxes'} or value['answer'] not in ('yes','no','unknown'):
            raise ValueError('Invalid evidence schema')
        boxes,error=parse_boxes(json.dumps(value['boxes']))
        if error or bool(boxes)!=(value['answer']=='yes'):raise ValueError('Answer/evidence disagreement')
        return {'valid':True,'answer':value['answer'],'boxes':boxes}
    except (ValueError,TypeError):return {'valid':False,'answer':None,'boxes':None}


def defacto_reward(text,gt,controls,variant,truncated=False):
    if variant not in ('pos','cf','rand'):raise ValueError('Unknown evidence view')
    value=parse_evidence(text,truncated)
    if not value['valid']:return {'reward':-1.,'correct':0.,'answer_correct':False,'evidence_success':False}
    answer=value['answer'];boxes=value['boxes'];target='yes' if gt else 'no'
    if variant=='cf':
        if not gt:raise ValueError('No counterfactual label for target-free image')
        correct=float(answer=='unknown');answer_reward=1. if correct else -1.-float(answer==target)
        coherence=0.;evidence_success=bool(correct)
    else:
        answer_reward=float(answer==target)-float(answer=='unknown')
        evidence_success=matching(gt,boxes)['strict'];correct=float(answer==target and evidence_success)
        coherence=(sum(max(iou(b,g) for g in gt)-.5*max((iou(b,c) for c in controls),default=0.) for b in boxes)/len(boxes)
                   if boxes and gt else -.5 if gt else float(not boxes))
    return {'reward':answer_reward+.1+.25*coherence,'correct':correct,'answer_correct':answer==('unknown' if variant=='cf' else target),
            'evidence_success':evidence_success,'view':variant,'coherence':coherence}
