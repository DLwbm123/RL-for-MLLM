"""Pilot-1 geometry, strict multi-box contract and patient schedules (no model calls)."""
import json
import math
from collections import Counter
import numpy as np
from PIL import Image, ImageFilter


def normalized(boxes, width, height):
    result = []
    for x1, y1, x2, y2 in boxes:
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('Invalid source box')
        b = [round(x1*1000/width), round(y1*1000/height),
             round(x2*1000/width), round(y2*1000/height)]
        if b[0] >= b[2] or b[1] >= b[3]:
            raise ValueError('Box collapsed after normalization')
        result.append(b)
    return sorted(result)


def parse_boxes(text, limit=64):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None, 'invalid_json'
    if not isinstance(value, list) or len(value) > limit:
        return None, 'invalid_list'
    for b in value:
        if (not isinstance(b, list) or len(b) != 4 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in b) or
                not 0 <= b[0] < b[2] <= 1000 or not 0 <= b[1] < b[3] <= 1000):
            return None, 'invalid_coordinates'
    return sorted(value), None


def crop_box(box, width, height, scale=2.):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    a, b = math.floor(cx-(x2-x1)*scale/2), math.floor(cy-(y2-y1)*scale/2)
    c, d = math.ceil(cx+(x2-x1)*scale/2), math.ceil(cy+(y2-y1)*scale/2)
    if c-a >= width: a, c = 0, width
    else:
        shift = max(0, -a)-max(0, c-width); a += shift; c += shift
    if d-b >= height: b, d = 0, height
    else:
        shift = max(0, -b)-max(0, d-height); b += shift; d += shift
    if not (a <= x1 < x2 <= c and b <= y1 < y2 <= d):
        raise ValueError('Crop lost target')
    return [a, b, c, d]


def map_box(box, source_size, target_size):
    w, h = source_size; tw, th = target_size
    b = [round(box[0]*tw/w), round(box[1]*th/h),
         round(box[2]*tw/w), round(box[3]*th/h)]
    if not 0 <= b[0] < b[2] <= tw or not 0 <= b[1] < b[3] <= th:
        raise ValueError('Invalid mapped geometry')
    return b


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def control_box(image, target, cfg, rng):
    w, h = image.size; x1, y1, x2, y2 = target; bw, bh = x2-x1, y2-y1
    margin = cfg['safety_margin']
    safety = [x1-bw*margin, y1-bh*margin, x2+bw*margin, y2+bh*margin]
    candidates = [[w-x2, y1, w-x1, y2]]
    for _ in range(cfg['control_candidates_max']-1):
        a, b = int(rng.integers(w-bw+1)), int(rng.integers(h-bh+1))
        candidates.append([a, b, a+bw, b+bh])
    for index, box in enumerate(candidates):
        if overlaps(box, safety): continue
        pixels = np.asarray(image.crop(box).convert('L'))
        if np.mean(pixels > cfg['nonblack_intensity_threshold']) >= cfg['nonblack_fraction_min']:
            return box, index
    return None, None


def intervention(image, box, kind, cfg):
    out = Image.new('RGB', image.size, tuple(cfg['blank_rgb'])) if kind == 'blank' else image.copy()
    if kind == 'blur': out.paste(image.crop(box).filter(ImageFilter.GaussianBlur(cfg['blur_radius'])), box[:2])
    elif kind in ('gray', 'blank'): out.paste(tuple(cfg['mask_rgb']), tuple(box))
    else: raise ValueError(kind)
    return out


def schedules(rows, views, steps, seed):
    rng = np.random.default_rng(seed); queues = {}; occurrences = Counter()
    train = {k:r for k,r in rows.items() if r['split'] == 'train'}
    pools = {label:sorted(k for k,r in train.items() if r['pathology'] == label) for label in ('yes','no')}
    def take(name, pool):
        if not pool: return None
        if not queues.get(name): queues[name] = list(rng.permutation(pool))
        return str(queues[name].pop())
    result = []
    for step in range(steps):
        label = 'yes' if step % 2 == 0 else 'no'
        full = [take('full_'+lab, pools[lab]) for lab in ('yes','no','yes','no')]
        loc = take('loc_'+label, pools[label])
        crop = take('crop_'+label, sorted(k for k in pools[label] if k in views['crops']))
        crop_index = None
        if crop:
            crop_index = occurrences[crop] % len(views['crops'][crop]); occurrences[crop] += 1
        pool = sorted(k for k in pools[label] if k in views['pairs'])
        pair = take('pair_'+label, pool)
        result.append({'step':step+1, 'full':full, 'loc':loc, 'crop':crop,
                       'crop_index':crop_index, 'pair':pair, 'pair_positive':label == 'yes'})
    return result


def exposures(plan, rows, views, method):
    tasks = {k:[] for k in ('full','loc','crop','pair')}
    for item in plan:
        tasks['full'].extend((k,None) for k in item['full'])
        if method != 'M0':
            for task in ('loc','crop'):
                if item[task]: tasks[task].append((item[task],item.get('crop_index') if task == 'crop' else None))
        if method in ('M1','M2','smoke') and item['pair']: tasks['pair'].append((item['pair'],None))
    out = {}
    for task, entries in tasks.items():
        counts = Counter(k for k,_ in entries)
        boxes = set()
        for k,i in entries:
            if task == 'loc': boxes.update((k,j) for j in range(len(rows[k]['boxes'])))
            elif task == 'crop' and rows[k]['pathology'] == 'yes': boxes.add((k,i))
            elif task == 'pair' and rows[k]['pathology'] == 'yes': boxes.add((k,0))
        out[task] = {'exposures':len(entries), 'positive':sum(rows[k]['pathology']=='yes' for k,_ in entries),
                     'negative':sum(rows[k]['pathology']=='no' for k,_ in entries), 'unique_patients':len(counts),
                     'patient_repetitions':dict(counts), 'distinct_boxes':len(boxes),
                     'distinct_views':len(set(entries))*(2 if task=='pair' else 1),
                     'skipped':len(plan)-len(entries) if task in ('loc','crop','pair') and (method!='M0' and (task!='pair' or method in ('M1','M2','smoke'))) else 0}
    return out


def matching(gt, prediction):
    from scipy.optimize import linear_sum_assignment
    if prediction is None: return {'matches':0, 'gt':len(gt), 'pred':None, 'all_covered':False, 'strict':False, 'single_iou':0. if len(gt)==1 else None}
    ious = np.zeros((len(gt),len(prediction)))
    for i,a in enumerate(gt):
        for j,b in enumerate(prediction):
            inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
            ious[i,j] = inter/union
    matches = 0
    if ious.size:
        # Count dominates any possible sum of <=64 IoUs. Stable index perturbation resolves residual ties.
        reward = 1000*(ious>=.5)+ious-1e-10*np.arange(ious.size).reshape(ious.shape)
        ii,jj = linear_sum_assignment(-reward); matches = int((ious[ii,jj]>=.5).sum())
    covered = matches == len(gt)
    return {'matches':matches,'gt':len(gt),'pred':len(prediction),'all_covered':covered,
            'strict':covered and len(prediction)==len(gt),
            'single_iou':float(ious.max()) if len(gt)==1 and ious.size else (0. if len(gt)==1 else None)}


def dependency(keep, hide, margin=.1):
    import torch
    return torch.relu(margin-(keep.float()-hide.float()))
