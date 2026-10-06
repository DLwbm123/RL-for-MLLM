"""Synthetic-only contracts and CPU extraction of pinned author pure methods."""
import ast
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

from src.v3_interfaces import AnswerCondition, diagnostic_rewards, score_fixed_conditions


class InterfaceContracts(unittest.TestCase):
    def test_fixed_prefix_and_repeated_conditions(self):
        condition = AnswerCondition((21, 22, 23, 24), (3,))
        seen = []
        def synthetic_score(item, region):
            seen.append(item)
            return -2.0 if region == 'original' else -2.5
        first = score_fixed_conditions(condition, ['evidence', 'control1', 'control2', 'control3'], synthetic_score)
        second = score_fixed_conditions(condition, ['evidence', 'control1', 'control2', 'control3'], synthetic_score)
        self.assertEqual(first, second)
        self.assertTrue(all(x is condition for x in seen))
        with self.assertRaises(ValueError):
            score_fixed_conditions(condition, ['evidence'], lambda *_: float('nan'))
        with self.assertRaises(ValueError):
            score_fixed_conditions(condition, ['original'], synthetic_score)

    def test_span_and_no_silent_truncation(self):
        for ids, positions in [((1, 2), (2,)), ((1, 2), (1, 1)), ((1, 2), ()), ((1, 2), (-1,)), ([1, 2], (1,))]:
            with self.assertRaises(ValueError):
                AnswerCondition(ids, positions)
        self.assertEqual(AnswerCondition((7,), (0,)).token_ids, (7,))

    def test_separate_base_and_margin(self):
        low = diagnostic_rewards(1, .2, 0, .5)
        high = diagnostic_rewards(1, .8, 0, .5)
        self.assertEqual(set(low), {'C', 'B', 'C_I', 'C_P', 'B_I', 'B_P', 'B_fixed_m0'})
        self.assertEqual(low['C_I'], high['C_I'])
        self.assertEqual(low['C_P'], high['C_P'])
        self.assertNotEqual(low['B_I'], high['B_I'])
        self.assertEqual(low['B_I'], low['B_fixed_m0'])
        self.assertEqual(high['B_I'], high['B_fixed_m0'])
        self.assertLess(diagnostic_rewards(1, .2, -1, -1)['C_I'],
                        diagnostic_rewards(0, .2, 1, 1)['C_I'])
        for values in [(2, .2, 0, 0), (1, float('nan'), 0, 0), (1, .2, 1.01, 0)]:
            with self.assertRaises(ValueError):
                diagnostic_rewards(*values)

    def test_pinned_author_pure_methods(self):
        root = Path(__file__).resolve().parents[1] / 'private' / 'author_code'
        if not root.exists():
            self.skipTest('Private author checkout absent; source check not verified')
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        self.assertEqual(head, '808a092fe87dd8be4b7e88c2d44aff76cec81b47')
        # Import only stdlib formatting modules; never import model/train modules.
        old_yesno = sys.modules.get('yesno_utils')
        modules = {}
        try:
            for name in ('yesno_utils', 'answer_format_utils'):
                spec = importlib.util.spec_from_file_location(name, root / 'src' / (name + '.py'))
                module = importlib.util.module_from_spec(spec)
                if name == 'yesno_utils':
                    sys.modules[name] = module
                spec.loader.exec_module(module)
                modules[name] = module
            fmt = modules['answer_format_utils']
            raw = 'Evidence: two objects\nFinal answer: 2'
            ext = fmt.extract_final_answer_with_prefix(raw, task_type='counting')
            self.assertEqual(ext['final_answer'], '2')
            self.assertEqual(ext['prefix_text_before_answer'], raw[:-1])
            self.assertFalse(fmt.task_family_allows_evidence_training('existence'))
            # Execute only selected pure method ASTs, with no constructor or model.
            source = ast.parse((root / 'src/reward/action_logprob_ate.py').read_text())
            cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ActionLogProbATEReward')
            wanted = {'_evidence_gate', '_correctness_reward', '_compose_main_rewards', '_soft_shaping_reward'}
            body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
            scope = dict(vars(fmt), math=math, Dict=dict, Any=object, Optional=__import__('typing').Optional)
            selected = ast.ClassDef(name='PureAuthor', bases=[], keywords=[], body=body, decorator_list=[])
            module_ast = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
            exec(compile(module_ast, '<pinned-author-pure-methods>', 'exec'), scope)
            author = scope['PureAuthor']()
            author.tau_resp, author.evidence_eps, author.main_reward_mode = .2, .1, 'routed_gated_evidence'
            author.min_reward, author.max_reward = -1.25, 1.0
            self.assertEqual(author._correctness_reward(answer_text='3', question='How many?', task_type='counting',
                             gt_answer='2', gt_present=None)['correctness_reward'], -.25)
            pack = author._compose_main_rewards(task_family='existence', correctness_reward=1,
                                                correctness_known=True, relative_margin=-1)
            self.assertEqual(pack['main_reward_active'], 1)
            self.assertEqual(pack['main_reward_mode_effective'], 'correctness_only')
            pack = author._compose_main_rewards(task_family='counting', correctness_reward=1,
                                                correctness_known=True, relative_margin=1)
            self.assertGreater(pack['main_reward_active'], 1)  # main path has no +1 clip
            self.assertEqual(author._soft_shaping_reward(2, True, 'other')[0], 1)  # separate path clips
        finally:
            if old_yesno is None:
                sys.modules.pop('yesno_utils', None)
            else:
                sys.modules['yesno_utils'] = old_yesno


if __name__ == '__main__':
    unittest.main()
