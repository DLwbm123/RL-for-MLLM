"""Synthetic checks of coverage and the predeclared AI qualification rule."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('review_aggregate', Path(__file__).parents[1] / 'scripts/summarize_rsna_v3_ai_review.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return [dict(blind_id='R001', region=n, component=c, status='acceptable', batch='1',
                 reviewer_identity='gpt-6-astra', reviewer_qualification='AI model; no human clinical credential',
                 reviewed_at='2026-10-06T00:00:00+00:00', review_version='synthetic', lung_coverage='unknown',
                 reason='synthetic', anatomy_reasonable='yes', annotated_or_suspicious_abnormality='no',
                 nonlung_artifact_or_text='no', fit_for_proposed_role='yes', insufficient_information='no')
            for n, c in sorted(module.ALL_COMPONENTS)]


class Qualification(unittest.TestCase):
    def test_all_eight_required_and_probe_separate(self):
        r = rows()
        r[-1]['status'] = 'rejected'  # shifted probe is a separate diagnostic
        summary, decisions = module.aggregate(r, {'R001'})
        self.assertTrue(decisions[0]['unchanged_interface_eligible'])
        r[0]['status'] = 'uncertain'
        summary, decisions = module.aggregate(r, {'R001'})
        self.assertFalse(decisions[0]['unchanged_interface_eligible'])
        self.assertEqual(summary['by_batch']['1']['excluded'], 1)

    def test_missing_duplicate_pending_and_fake_credential_rejected(self):
        for bad in (rows()[:-1], rows() + rows()[:1]):
            with self.assertRaises(ValueError): module.aggregate(bad, {'R001'})
        for key, value in [('status', 'pending'), ('reviewer_qualification', 'clinical expert'), ('lung_coverage', '95%')]:
            r = rows(); r[0][key] = value
            with self.assertRaises(ValueError): module.aggregate(r, {'R001'})


if __name__ == '__main__':
    unittest.main()
