"""Frozen B1 paired rescoring; launch via neutral stdin, never samples or trains."""
import csv
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

VERSION = 'v3-g1-ai-reviewed-1'
VARIANT = 'AI-screened intervention interface, not human-validated'
ARTIFACTS = ['P4/B1/ced_diagnostics.json', 'P4/B1/groups.json',
             'P4/B1/frozen_identity.json', 'protocol/manifest.csv',
             'protocol/regions.json', 'protocol/subsets.json', 'protocol/identity.json']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, list):
        for item in value: finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite protocol or measurement')


def write_new(path, value):
    finite(value)
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_hashes(root, hashes, required=()):
    root = Path(root).resolve()
    if not set(required) <= set(hashes):
        raise ValueError('Missing frozen artifact/source hashes')
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Unsafe frozen path or digest')
        if sha(path) != digest:
            raise ValueError('Frozen identity mismatch: ' + name)


def validate_region(region, old, row):
    if region.get('eligible_geometry') is not True or region.get('semantic_review_status') != 'AI_acceptable':
        raise PermissionError('New region lacks frozen AI qualification')
    if region['grid'] != old['grid'] or len(region['controls']) != 3:
        raise ValueError('Grid/control count changed')
    gh, gw = region['grid']; width, height = float(row['width']), float(row['height'])
    if not all(type(n) is int and n > 0 for n in [gh, gw]):
        raise ValueError('Invalid grid')
    if any(region['evidence'][key] != old['evidence'][key] for key in ['box', 'tokens']):
        raise ValueError('Original evidence region must remain fixed')
    support_ids = region['AI_support_tokens']
    if len(set(support_ids)) != len(support_ids) or any(type(i) is not int or not 0 <= i < gh * gw for i in support_ids):
        raise ValueError('Invalid AI support token indices')
    support = set(support_ids)
    def box_tokens(box):
        return {y * gw + x for y in range(gh) for x in range(gw)
                if x * width / gw < box[2] and (x + 1) * width / gw > box[0]
                and y * height / gh < box[3] and (y + 1) * height / gh > box[1]}
    x1, y1, x2, y2 = region['evidence']['box']
    forbidden = box_tokens([max(0, x1-.1*(x2-x1)), max(0, y1-.1*(y2-y1)),
                            min(width, x2+.1*(x2-x1)), min(height, y2+.1*(y2-y1))])
    evidence_ids = region['evidence']['tokens']
    def token_shape(ids):
        ys, xs = [i // gw for i in ids], [i % gw for i in ids]
        return {(y-min(ys), x-min(xs)) for y,x in zip(ys, xs)}
    previous_controls = []
    for item in [region['evidence'], *region['controls']]:
        box = item['box']
        if len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError('Invalid region box')
        for key in ['tokens', 'source']:
            ids = item[key]
            if len(ids) < 4 or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < gh * gw for i in ids):
                raise ValueError('Invalid region/source token indices')
        if set(item['tokens']) & set(item['source']):
            raise ValueError('Replacement source overlaps target')
        expected = box_tokens(box)
        if set(item['tokens']) != expected:
            raise ValueError('Box/token mismatch')
        if len(item['tokens']) != len(region['evidence']['tokens']):
            raise ValueError('Matched token count changed')
        target = set(item['tokens']); sources = set(item['source'])
        local = {yy * gw + xx for token in target for y,x in [divmod(token, gw)]
                 for yy in range(max(0, y-3), min(gh, y+4))
                 for xx in range(max(0, x-3), min(gw, x+4))}
        if sources != (local & support) - target - forbidden:
            raise ValueError('Source differs from frozen local support ring or overlaps evidence margin')
        if item is not region['evidence']:
            if not target <= support or target & forbidden:
                raise ValueError('Control outside AI support or inside evidence margin')
            if (not math.isclose(box[2]-box[0], x2-x1, rel_tol=1e-9, abs_tol=1e-9)
                    or not math.isclose(box[3]-box[1], y2-y1, rel_tol=1e-9, abs_tol=1e-9)
                    or token_shape(item['tokens']) != token_shape(evidence_ids)):
                raise ValueError('Control pixel/token shape differs from evidence')
            ys = [i // gw for i in target]
            if abs((min(ys)+max(ys)+1)/2/gh - (y1+y2)/2/height) > .10:
                raise ValueError('Control vertical center exceeds frozen tolerance')
            if any(len(target & previous)/len(target) > .75 for previous in previous_controls):
                raise ValueError('Control pairwise overlap exceeds frozen tolerance')
            previous_controls.append(target)


@contextmanager
def forbid_training(torch):
    """Close autograd/optimizer entry points in addition to historical frozen()."""
    targets = [(torch.optim.Optimizer, '__init__'), (torch.autograd, 'backward'), (torch.autograd, 'grad')]
    originals = [getattr(obj, name) for obj, name in targets]
    def denied(*args, **kwargs): raise PermissionError('Training/autograd is forbidden in G1')
    try:
        for obj, name in targets: setattr(obj, name, denied)
        yield
    finally:
        for (obj, name), original in zip(targets, originals): setattr(obj, name, original)


def validate_selection(manifest, selections, rows, regions, diagnostics, groups):
    """Pure CPU gate, before importing any model implementation."""
    finite(manifest); finite(diagnostics); finite(groups)
    if (manifest.get('version') != VERSION or manifest.get('protocol_variant') != VARIANT
            or manifest.get('original_geometry_patients') != 32 or manifest.get('old_reviewed_patients') != 93
            or manifest.get('sampling') != 'reuse existing v2 P4 groups exactly; no sampling'):
        raise PermissionError('Missing independently frozen G1 protocol')
    for key, value in [('ced_epsilon', 1e-6), ('advantage_epsilon', 1e-8),
                       ('near_zero_std', 1e-8), ('score_atol', 1e-5)]:
        if manifest.get(key) != value: raise ValueError('Frozen tolerance changed: ' + key)
    eligible = {r['image_id'] for r in selections if regions.get(r['image_id'], {}).get('eligible')}
    if len(eligible) != 32: raise ValueError('Original P4 geometry denominator changed')
    coverage = manifest['selected'] + manifest['excluded']
    if len(coverage) != 32 or {r['image_id'] for r in coverage} != eligible:
        raise ValueError('Selected/excluded must partition all original 32 patients')
    if len({r['blind_id'] for r in coverage}) != 32 or len({r['case_id'] for r in coverage}) != 32:
        raise ValueError('Duplicated blind or patient identity')
    by_id = {r['image_id']: r for r in diagnostics}
    if len(by_id) != len(diagnostics): raise ValueError('Duplicate old diagnostics')
    grouped = {}
    for group in groups: grouped.setdefault(group['image_id'], []).append(group)
    for item in coverage:
        key = item['image_id']; row = rows[key]; old = by_id[key]
        if not re.fullmatch(r'R[0-9]{3}', item['blind_id']): raise ValueError('Invalid blind ID')
        if row['case_id'] != item['case_id'] or old['case_id'] != item['case_id'] or row['split'] != 'train':
            raise ValueError('Original development patient identity mismatch')
        if not old['eligible_geometry'] or set(old['labels']) != {'no', 'yes'}:
            raise ValueError('Missing original eligible label cache')
        batch = grouped[key]
        if len(batch) != 4 or {g['group'] for g in batch} != set(range(4)):
            raise ValueError('Original four groups missing or duplicated')
        for group in batch:
            answers = group['answers']; correct = group['correctness_only']['correctness_rewards']
            expected = [float(a == row['pathology']) if a in ['no', 'yes'] else -.1 for a in answers]
            if (len(answers) != 8 or len(correct) != 8
                    or any(abs(a-b) > manifest['score_atol'] for a,b in zip(correct, expected))
                    or group['case_id'] != row['case_id']
                    or group['revised_CED']['correctness_rewards'] != correct):
                raise ValueError('Original eight-answer/correctness mapping changed')
            rewards = [old['labels'][a]['reward'] if a in ['no', 'yes'] else -.1 for a in answers]
            if group['revised_CED']['ced_rewards'] != rewards:
                # Stored rewards are float32 in reward_group; allow only the frozen score tolerance.
                if len(group['revised_CED']['ced_rewards']) != 8 or max(abs(a-b) for a,b in zip(group['revised_CED']['ced_rewards'], rewards)) > manifest['score_atol']:
                    raise ValueError('Stored group/cache reward mismatch')
    for item in manifest['excluded']:
        if not item.get('reason'): raise ValueError('Excluded patient needs a frozen reason')
    for item in manifest['selected']:
        validate_region(item['new_region'], regions[item['image_id']], rows[item['image_id']])
    return by_id, grouped


def compare_cache(actual, reference, tolerance):
    finite(actual); finite(reference)
    errors = []
    for label in ['no', 'yes']:
        a, b = actual[label], reference[label]
        for key in ['reward', 'D_E', 'raw_M', 'std', 'm', 'gate']:
            errors.append(abs(a[key] - b[key]))
        if len(a['D_N']) != 3 or len(b['D_N']) != 3 or a['saturated'] != b['saturated']:
            raise ValueError('Old control/saturation identity mismatch')
        errors.extend(abs(x-y) for x,y in zip(a['D_N'], b['D_N']))
    maximum = max(errors)
    if maximum > tolerance: raise ValueError('Old P4 recomputed cache exceeds fixed score tolerance')
    return maximum


def replay_groups(groups, cache, target, manifest, reward_group):
    wrong = 'no' if target == 'yes' else 'yes'
    gap = cache[target]['reward'] - cache[wrong]['reward']
    sign = (gap > 0) - (gap < 0)
    result = []
    for group in sorted(groups, key=lambda g: g['group']):
        correct = group['correctness_only']['correctness_rewards']
        rewards = [cache[a]['reward'] if a in cache else -.1 for a in group['answers']]
        detail = reward_group(correct, rewards, manifest['advantage_epsilon'], manifest['near_zero_std'])
        detail.update(group=group['group'], answers=group['answers'], reward_gap=gap,
                      sign_residual=max(abs(r-sign*c) for r,c in zip(detail['ced_advantage'], detail['correctness_advantage'])),
                      sign_reference_applicable=detail['all_legal_mixed'] and gap != 0 and not detail['near_zero_std'])
        result.append(detail)
    finite(result)
    return result


def run():
    started = time.time(); model = data = None; completed = []; durations = []
    shard = int(os.environ['V3_G1_SHARD']); shards = int(os.environ['V3_G1_SHARDS'])
    if shards != 2 or shard not in [0, 1]: raise ValueError('Exactly two prespecified shards required')
    root = Path(os.environ['OUTPUT_ROOT']).resolve(); old = Path(os.environ['V2_OUTPUT_ROOT']).resolve()
    v1 = Path(os.environ['V1_OUTPUT_ROOT']).resolve()
    if any(root == p or root.is_relative_to(p) for p in [old, v1]): raise PermissionError('Independent v3 output root required')
    dest = root / 'G1' / f'shard_{shard}'; dest.mkdir(parents=True, exist_ok=False)
    result = {'status': 'failed', 'shard': shard, 'completed_blind_ids': completed,
              'protocol_variant': VARIANT, 'test_images_read': 0, 'training_run': False, 'generation_calls': 0}
    try:
        deadline = float(os.environ['V3_G1_DEADLINE'])
        if not math.isfinite(deadline): raise ValueError('Invalid absolute budget deadline')
        manifest_path = Path(os.environ['V3_G1_MANIFEST']).resolve()
        if manifest_path != root / 'G1/protocol/manifest.json': raise ValueError('Unexpected G1 manifest location')
        digest = os.environ['V3_G1_MANIFEST_SHA256']
        if not re.fullmatch(r'[0-9a-f]{64}', digest) or sha(manifest_path) != digest:
            raise PermissionError('Missing or changed externally frozen manifest digest')
        manifest = read(manifest_path); result['manifest_sha256'] = digest
        if os.environ['DATASET_NAME'] != 'rsna': raise PermissionError('RSNA namespace required')
        code = Path(__file__).resolve().parents[1]
        check_hashes(code, manifest['source_hashes'], ['scripts/run_rsna_v3_g1.py'])
        oldcode = old.parent / 'code'; cfgpath = oldcode / 'configs/rsna_v2.json'
        lockpath = old / 'protocol/protocol_lock.json'; lock = read(lockpath); cfg = read(cfgpath)
        if sha(cfgpath) != manifest['v2_config_sha256'] or sha(cfgpath) != lock['config_sha256'] or sha(lockpath) != manifest['v2_protocol_lock_sha256']:
            raise ValueError('Frozen v2 config/protocol changed')
        check_hashes(old / 'protocol', lock['hashes'])
        check_hashes(oldcode, lock['source_hashes'])
        # Imported historical helpers must be byte-identical to their frozen v2 implementation.
        check_hashes(code, lock['source_hashes'])
        check_hashes(old, manifest['artifact_hashes'], ARTIFACTS)
        identity = read(old / 'protocol/identity.json'); regions = read(old / 'protocol/regions.json')
        with (old / 'protocol/manifest.csv').open() as stream: rows_list = list(csv.DictReader(stream))
        rows = {r['image_id']: r for r in rows_list}
        if len(rows) != len(rows_list): raise ValueError('Duplicate historical image identity')
        by_id, grouped = validate_selection(manifest, read(old / 'protocol/subsets.json')['P4'], rows, regions,
                                             read(old / 'P4/B1/ced_diagnostics.json'), read(old / 'P4/B1/groups.json'))
        for key in ['ced_epsilon', 'advantage_epsilon', 'near_zero_std']:
            if manifest[key] != cfg['P4'][key]: raise ValueError('Historical numeric contract differs')
        if manifest['score_atol'] != cfg['P4']['numerical_score_tolerance'] or cfg['P4']['groups'] != 4 or cfg['P4']['group_size'] != 8:
            raise ValueError('Historical group/score contract differs')
        prior = read(old / 'P4/B1/frozen_identity.json')
        if not prior['unchanged'] or prior['before'] != prior['after'] or prior['before']['requires_grad_count'] != 0:
            raise ValueError('Historical frozen parameter receipt invalid')
        assigned = sorted(manifest['selected'], key=lambda r: r['blind_id'])[shard::shards]
        result.update(assigned_blind_ids=[r['blind_id'] for r in assigned], selected_patients=len(manifest['selected']),
                      original_geometry_patients=32, excluded_patients=len(manifest['excluded']))
        write_new(dest / 'coverage.json', {'selected': manifest['selected'], 'excluded': manifest['excluded'],
                                           'assigned_blind_ids': result['assigned_blind_ids'], 'manifest_sha256': digest})
        if not assigned:
            result.update(status='completed', reason='no eligible cases assigned; model not loaded'); return result
        if time.time() >= deadline - 30:
            result.update(status='partial_budget', reason='budget exhausted before model load'); return result
        # Heavy imports and dataset access begin only after the full frozen CPU gate.
        import pandas as pd
        import torch
        from src.model import Model, REVISION
        from src.experiment import seed_all
        from src.v2 import DevelopmentData, reward_group
        from src.v2_run import frozen, parameter_identity, ced_label_cache
        base = Path(os.environ['MODEL_ROOT']) / 'Qwen2.5-VL-7B-Instruct'
        if identity['base']['revision'] != REVISION or manifest['source_identity']['base_revision'] != REVISION:
            raise ValueError('Base revision changed')
        for name, info in identity['base']['files'].items():
            stat = (base / name).stat()
            if stat.st_size != info['bytes'] or stat.st_mtime_ns != info['mtime_ns']: raise ValueError('Base file identity changed')
        adapter = v1 / 'B1' / identity['historical_B1']['checkpoint']
        if (identity['historical_B1']['checkpoint'] != 'step_0256'
                or sha(adapter / 'adapter_model.safetensors') != identity['historical_B1']['sha256']
                or manifest['source_identity']['B1_adapter_sha256'] != identity['historical_B1']['sha256']
                or sha(adapter / 'adapter_config.json') != identity['historical_B1']['config_sha256']):
            raise ValueError('Historical B1 adapter changed')
        data = DevelopmentData(os.environ['DATA_ROOT'], pd.DataFrame([rows[r['image_id']] for r in assigned]), dest / 'data_access.jsonl')
        data.install_guard(); seed_all(cfg['seed'])
        model = Model(base, adapter=adapter, trainable=False, min_tokens=cfg['min_visual_tokens'], max_tokens=cfg['max_visual_tokens'])
        ids = {label: model.processor.tokenizer.encode(label, add_special_tokens=False) for label in ['no', 'yes']}
        if ids != identity['label_ids'] or model.processor.tokenizer.eos_token_id != identity['eos_id']: raise ValueError('Label token identity changed')
        before = parameter_identity(model)
        if before['adapter_parameter_digest'] != prior['before']['adapter_parameter_digest']:
            raise ValueError('Loaded adapter parameters differ from historical B1')
        ctx = SimpleNamespace(cfg=cfg)
        with forbid_training(torch), frozen(model):
            for item in assigned:
                if time.time() >= deadline - 30 or (durations and max(durations) * 1.2 > deadline - time.time() - 30):
                    result.update(status='partial_budget', reason='only complete preordered case pairs within remaining budget'); break
                case_start = time.time(); key = item['image_id']; target = rows[key]['pathology']
                inputs, grid = model.prompt(data.image(key), 'A'); features = model.encode(inputs)
                if list(grid) != regions[key]['grid'] or list(grid) != item['new_region']['grid']: raise ValueError('Preprocessing grid mismatch')
                oldcache = ced_label_cache(ctx, model, inputs, features, regions[key], target)
                maximum = compare_cache(oldcache, by_id[key]['labels'], manifest['score_atol'])
                newcache = ced_label_cache(ctx, model, inputs, features, item['new_region'], target); finite(newcache)
                paired = {'blind_id': item['blind_id'], 'image_id': key, 'case_id': item['case_id'], 'target': target,
                          'old_cache_max_abs_error': maximum, 'old': {'labels': oldcache}, 'new': {'labels': newcache}}
                for name, cache in [('old', oldcache), ('new', newcache)]:
                    wrong = 'no' if target == 'yes' else 'yes'
                    paired[name]['correct_minus_wrong_reward_gap'] = cache[target]['reward'] - cache[wrong]['reward']
                    paired[name]['groups'] = replay_groups(grouped[key], cache, target, manifest, reward_group)
                paired['runtime_seconds'] = time.time() - case_start
                write_new(dest / (item['blind_id'] + '.json'), paired)
                completed.append(item['blind_id']); durations.append(paired['runtime_seconds'])
                throughput = {'completed': len(completed), 'case_seconds': durations[-1],
                              'fixed_order_remaining': len(assigned) - len(completed),
                              'estimated_remaining_seconds': sum(durations)/len(durations)*(len(assigned)-len(completed)),
                              'budget_remaining_seconds': max(0, deadline-time.time())}
                write_new(dest / f'throughput_{len(completed):02d}.json', throughput)
                print(json.dumps(throughput), flush=True)
            else: result['status'] = 'completed'
        write_new(dest / 'frozen_identity.json', model.v2_identity_check)
        result['frozen_parameters_unchanged'] = model.v2_identity_check['unchanged']
        if model.generation_calls or model.generated_tokens: raise RuntimeError('Generation is forbidden')
        return result
    except Exception as error:
        result.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if data is not None: result['data_access_counts'] = dict(data.access_counts)
        if model is not None:
            result['generation_calls'] = model.generation_calls
            result['score_calls'] = model.score_count
            if hasattr(model, 'v2_identity_check') and not (dest / 'frozen_identity.json').exists():
                write_new(dest / 'frozen_identity.json', model.v2_identity_check)
        result['runtime_seconds'] = time.time() - started
        write_new(dest / 'result.json', result)
        print(json.dumps({k: result[k] for k in ['status', 'shard', 'completed_blind_ids', 'runtime_seconds']}), flush=True)


if __name__ == '__main__':
    run()
