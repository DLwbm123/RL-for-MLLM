"""Synthetic geometry checks; no clinical acceptance claim."""
import unittest
from scripts.prepare_rsna_v3_ai_regions import build_proposal, box_tokens, support_tokens, local_source


class ProposalGeometry(unittest.TestCase):
    def test_cells_sources_and_exclusion_without_fallback(self):
        grid = [16, 16]
        box = [40, 60, 50, 70]
        ids = box_tokens(box, 160, 160, grid)
        geometry = {'image_size': [160, 160], 'grid': grid,
                    'regions': {'evidence_region': {'box_xyxy': box, 'token_ids': ids}}}
        p = build_proposal(geometry, [[[10, 10], [150, 10], [150, 150], [10, 150]]])
        self.assertTrue(p['eligible_geometry'])
        self.assertEqual(len(p['controls']), 3)
        for item in [p['evidence'], *p['controls']]:
            self.assertFalse(set(item['tokens']) & set(item['source']))
            self.assertTrue(set(item['source']) <= set(p['AI_support_tokens']))
            self.assertGreaterEqual(len(item['source']), 4)
        for c in p['controls']:
            self.assertEqual(len(c['tokens']), len(ids))
            self.assertEqual((c['box'][2] - c['box'][0]) * (c['box'][3] - c['box'][1]), 100)
        empty = build_proposal(geometry, [])
        self.assertFalse(empty['eligible_geometry'])
        self.assertEqual(empty['controls'], [])
        self.assertEqual(empty['evidence']['source'], [])
        # A partial cell must not be admitted based only on its center.
        self.assertNotIn(0, support_tokens([[[1, 1], [9, 1], [9, 9], [1, 9]]], 160, 160, grid))
        with self.assertRaises(ValueError):
            support_tokens([[[0, 0], [200, 0], [0, 20]]], 160, 160, grid)


if __name__ == '__main__':
    unittest.main()
