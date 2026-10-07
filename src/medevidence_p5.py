"""P5 coordinate contracts and aggregate analysis of existing sampled groups."""
import json
import math
from collections import Counter


def target(boxes, width, height, fmt, displayed):
    from src.medevidence import normalized, map_box
    if fmt == 'normalized':
        value = normalized(boxes, width, height)
    elif fmt == 'absolute':
        value = sorted(map_box(b, (width, height), displayed) for b in boxes)
    else:
        raise ValueError('Unknown coordinate format')
    return json.dumps(value, separators=(',', ':'))


def canonical(text, fmt, displayed):
    """Strictly validate in the declared frame, then map to the old metric frame."""
    from src.medevidence import parse_boxes
    if fmt == 'normalized':
        return parse_boxes(text)
    if fmt != 'absolute':
        raise ValueError('Unknown coordinate format')
    try:
        value = json.loads(text)
        if not isinstance(value, list) or len(value) > 64:
            return None, 'invalid_list'
        w, h = displayed
        for b in value:
            if (not isinstance(b, list) or len(b) != 4 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in b) or
                not 0 <= b[0] < b[2] <= w or not 0 <= b[1] < b[3] <= h):
                return None, 'invalid_coordinates'
        return sorted([[x * 1000 / (w if i % 2 == 0 else h) for i, x in enumerate(b)] for b in value]), None
    except (ValueError, TypeError):
        return None, 'invalid_json'


def prompt(fmt, displayed):
    w, h = displayed
    units = 'coordinates normalized to 0–1000' if fmt == 'normalized' else 'pixel coordinates in the displayed image'
    return (f'Locate any lung opacity suspicious for pneumonia in this chest radiograph. '
            f'The displayed image is {w} pixels wide and {h} pixels high. '
            f'Return only a JSON list of bounding boxes [[x1,y1,x2,y2],...], with {units}. '
            'Return [] when no target finding is visible.')


def token_parts(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    empty = tokenizer.encode('[]', add_special_tokens=False) + [tokenizer.eos_token_id]
    divergence = 0
    if text != '[]':
        while divergence < min(len(ids), len(empty)) and ids[divergence] == empty[divergence]:
            divergence += 1
    parts = []
    for i, token in enumerate(ids):
        decoded = tokenizer.decode([token], skip_special_tokens=False)
        parts.append('eos' if i == len(ids)-1 else 'existence' if i == divergence else
                     'coordinate' if any(c.isdigit() for c in decoded) else 'syntax')
    return ids, parts


def decompose(groups):
    """Counts are descriptive. A reward distinction need not improve IoU matching."""
    import numpy as np
    from src.medevidence import parse_boxes, matching
    from src.medevidence_p4 import reward, advantages
    out = {}
    for name, chosen in [('positive', [g for g in groups if g['gt']]),
                         ('negative', [g for g in groups if not g['gt']])]:
        counts = Counter(groups=len(chosen)); between = within = total = 0.
        for g in chosen:
            rs = g['outputs']; rewards = []; states = []; matches = []; numbers = []
            for r in rs:
                boxes, error = parse_boxes(r['loc_text'])
                valid = not error and not r['truncated']
                state = 'invalid' if not valid else 'nonempty' if boxes else 'empty'
                rr = reward(r['loc_text'], g['gt'], r['truncated'])
                assert abs(rr-r['reward']) < 1e-10
                rewards.append(rr); states.append(state)
                numbers.append(len(boxes) if valid else -1)
                matches.append(matching(g['gt'], boxes if valid else None)['matches'])
                counts['completions'] += 1; counts[state+'_completions'] += 1
            rr = np.array(rewards); variance = float(rr.var()); total += variance
            within_group = sum(sum(s == state for s in states)/len(states) *
                               float(rr[np.array(states) == state].var()) for state in set(states))
            within += within_group; between += max(0., variance-within_group)
            counts['distinguishable_groups'] += variance > 1e-12
            counts['all_empty_groups'] += all(s == 'empty' for s in states)
            counts['mixed_empty_nonempty_groups'] += 'empty' in states and 'nonempty' in states
            counts['any_matched_region_groups'] += any(matches)
            counts['distinguishable_without_any_match'] += variance > 1e-12 and not any(matches)
            counts['nonempty_geometry_distinguishable_groups'] += any(
                states[i] == states[j] == 'nonempty' and numbers[i] == numbers[j] and
                abs(rewards[i]-rewards[j]) > 1e-6 for i in range(len(rs)) for j in range(i))
            counts['positive_advantage_without_match_completions'] += sum(
                a > 0 and m == 0 for a, m in zip(advantages(rewards).tolist(), matches)) if g['gt'] else 0
            if not g['gt']:
                stronger = [1. if x == 1. else -5. for x in rewards]
                assert np.allclose(advantages(rewards).numpy(), advantages(stronger).numpy(), atol=1e-6)
        out[name] = {**{k:int(v) for k,v in counts.items()},
                     'sum_group_reward_variance':total, 'between_output_state_variance':between,
                     'within_output_state_variance':within,
                     'between_state_share':between/total if total else None}
    out['negative_penalty_rescaling_advantage_invariance_checked'] = True
    out['interpretation'] = 'Equal-count nonempty reward variation is geometric, not proof of correct boxes; no causal attribution.'
    return out


def diagnostic_gate(summary):
    fit, cal, sampled = summary['final_fit'], summary['final_calibration'], summary['natural_calibration']
    checks = {
        'fit_positive_strict_at_least12_of16':fit['positive_strict'] >= 12,
        'fit_negative_nonempty_at_most2_of16':fit['negative_nonempty'] <= 2,
        'cal_negative_nonempty_at_most4_of32':cal['negative_nonempty'] <= 4,
        'cal_single_best8_at_least4_of16':sampled['single_best8_success'] >= 4,
        'cal_positive_geometric_groups_at_least8_of32':sampled['decomposition']['positive'].get('nonempty_geometry_distinguishable_groups', 0) >= 8,
        'no_invalid_or_truncated':sum(x['invalid']+x['truncated'] for x in (fit,cal)) == 0 and sampled['invalid_or_truncated'] == 0,
    }
    return {'passed':all(checks.values()), 'conditions':checks,
            'meaning':'Prespecified P5 engineering gate on training-pool calibration, not a revised P4 gate or independent validation.'}
