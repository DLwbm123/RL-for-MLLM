"""Stdlib synthetic gates: no patient data, torch import, or model execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_rsna_v3_g1 import (VERSION, VARIANT, compare_cache, forbid_training, run,
                                   validate_selection)


def fixture():
    support = {y*20+x for y in range(1, 19) for x in range(1, 19)}
    forbidden = {y*20+x for y in range(8, 12) for x in range(1, 5)}
    def rectangle(x):
        tokens = {y*20+xx for y in [9, 10] for xx in [x, x+1]}
        local = {yy*20+xx for token in tokens for y,xx0 in [divmod(token, 20)]
                 for yy in range(y-3, y+4) for xx in range(max(0, xx0-3), min(20, xx0+4))}
        return {'box': [x, 9, x+2, 11], 'tokens': sorted(tokens),
                'source': sorted((local & support)-tokens-forbidden)}
    region = {'eligible': True, 'grid': [20, 20], 'evidence': rectangle(2),
              'controls': [rectangle(x) for x in [7, 11, 15]], 'AI_support_tokens': sorted(support)}
    cache = {label: {'reward': float(label == 'yes'), 'D_E': 0., 'D_N': [0., 0., 0.],
                     'raw_M': 0., 'std': 0., 'm': 0., 'gate': .5, 'saturated': False}
             for label in ['no', 'yes']}
    rows = {}; regions = {}; diagnostics = []; groups = []; coverage = []; selections = []
    for i in range(32):
        key = f'synthetic_{i}'
        rows[key] = {'case_id': key, 'split': 'train', 'pathology': 'yes', 'width': '20', 'height': '20'}
        regions[key] = deepcopy(region)
        diagnostics.append({'image_id': key, 'case_id': key, 'eligible_geometry': True, 'labels': deepcopy(cache)})
        selections.append({'image_id': key})
        coverage.append({'blind_id': f'R{i+1:03d}', 'image_id': key, 'case_id': key, 'reason': 'synthetic exclusion'})
        for group in range(4):
            groups.append({'image_id': key, 'case_id': key, 'group': group, 'answers': ['no', 'yes'] * 4,
                           'correctness_only': {'correctness_rewards': [0., 1.] * 4},
                           'revised_CED': {'correctness_rewards': [0., 1.] * 4, 'ced_rewards': [0., 1.] * 4}})
    selected = coverage[:1]
    selected[0]['new_region'] = {**deepcopy(region), 'eligible_geometry': True, 'semantic_review_status': 'AI_acceptable'}
    manifest = {'version': VERSION, 'protocol_variant': VARIANT, 'original_geometry_patients': 32,
                'old_reviewed_patients': 93, 'sampling': 'reuse existing v2 P4 groups exactly; no sampling',
                'ced_epsilon': 1e-6, 'advantage_epsilon': 1e-8, 'near_zero_std': 1e-8,
                'score_atol': 1e-5, 'selected': selected, 'excluded': coverage[1:]}
    return manifest, selections, rows, regions, diagnostics, groups


class G1FrozenGateTests(unittest.TestCase):
    def test_accepts_exact_partition_and_rejects_scope_or_protocol_mutations(self):
        args = fixture()
        by_id, groups = validate_selection(*args)
        self.assertEqual(len(by_id), 32)
        self.assertEqual(len(groups['synthetic_0']), 4)
        mutations = [lambda a: a[0]['excluded'].pop(),
                     lambda a: a[0].update(score_atol=1e-3),
                     lambda a: a[0]['selected'][0]['new_region'].update(semantic_review_status='pending'),
                     lambda a: a[0]['selected'][0]['new_region']['evidence'].update(source=[0, 1, 6, 7]),
                     lambda a: a[0]['selected'][0]['new_region']['AI_support_tokens'].remove(187),
                     lambda a: a[0]['selected'][0]['new_region']['controls'].__setitem__(1, deepcopy(a[0]['selected'][0]['new_region']['controls'][0])),
                     lambda a: a[2]['synthetic_0'].update(split='test'),
                     lambda a: a[5][0]['answers'].__setitem__(0, 'yes')]
        for mutate in mutations:
            changed = deepcopy(args); mutate(changed)
            with self.assertRaises((ValueError, PermissionError)): validate_selection(*changed)

    def test_training_entry_points_are_denied_and_restored(self):
        class Optimizer:
            def __init__(self): self.ready = True
        autograd = SimpleNamespace(backward=lambda: 'backward', grad=lambda: 'grad')
        fake_torch = SimpleNamespace(optim=SimpleNamespace(Optimizer=Optimizer), autograd=autograd)
        with forbid_training(fake_torch):
            for operation in [Optimizer, autograd.backward, autograd.grad]:
                with self.assertRaises(PermissionError): operation()
        self.assertTrue(Optimizer().ready)
        self.assertEqual(autograd.backward(), 'backward')
        self.assertEqual(autograd.grad(), 'grad')

    def test_fixed_cache_tolerance_and_nonfinite_stop(self):
        cache = fixture()[4][0]['labels']
        self.assertEqual(compare_cache(cache, cache, 1e-5), 0.)
        changed = deepcopy(cache); changed['yes']['D_E'] = 2e-5
        with self.assertRaises(ValueError): compare_cache(changed, cache, 1e-5)
        changed['yes']['D_E'] = float('nan')
        with self.assertRaises(ValueError): compare_cache(changed, cache, 1e-5)

    def test_absent_freeze_fails_before_any_model_import_or_dataset_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = {'V3_G1_SHARD': '0', 'V3_G1_SHARDS': '2', 'OUTPUT_ROOT': str(root/'new'),
                   'V2_OUTPUT_ROOT': str(root/'old'), 'V1_OUTPUT_ROOT': str(root/'v1'),
                   'V3_G1_DEADLINE': '9999999999', 'V3_G1_MANIFEST': str(root/'new/G1/protocol/manifest.json')}
            imported = set(sys.modules)
            with patch.dict(os.environ, env, clear=True), self.assertRaises(KeyError): run()
            result = json.loads((root/'new/G1/shard_0/result.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['test_images_read'], 0)
            self.assertEqual(result['completed_blind_ids'], [])
            self.assertFalse({'torch', 'src.model', 'src.v2_run'} & (set(sys.modules)-imported))
            with patch.dict(os.environ, env, clear=True), self.assertRaises(FileExistsError): run()


if __name__ == '__main__':
    unittest.main()
