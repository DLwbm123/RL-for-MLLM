"""Validate and aggregate the separately authorized, blinded Astra screening.

No image judgement is made here. Decisions come from the recorded AI reviewers;
human review remains absent. Original packets and frozen R1 files stay unchanged.
"""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path


REQUIRED = {(name, component)
            for name in ('evidence', 'control_1', 'control_2', 'control_3')
            for component in ('region', 'replacement_source')}
ALL_COMPONENTS = REQUIRED | {('wrong', 'region'), ('wrong', 'replacement_source')}


def aggregate(rows, expected_ids):
    cases = defaultdict(dict)
    for row in rows:
        key = (row['region'], row['component'])
        if row['blind_id'] not in expected_ids or key not in ALL_COMPONENTS:
            raise ValueError('Unexpected blind case or component')
        if key in cases[row['blind_id']]:
            raise ValueError('Duplicate review row')
        if row['status'] not in ('acceptable', 'uncertain', 'rejected'):
            raise ValueError('Incomplete AI review')
        if row['reviewer_identity'] != 'gpt-6-astra' or row['lung_coverage'] != 'unknown':
            raise ValueError('Reviewer identity or unverified lung coverage')
        if row['reviewer_qualification'] != 'AI model; no human clinical credential':
            raise ValueError('AI review must not claim human qualification')
        if not all(row.get(k, '').strip() for k in (
                'reason', 'reviewed_at', 'review_version', 'anatomy_reasonable',
                'annotated_or_suspicious_abnormality', 'nonlung_artifact_or_text',
                'fit_for_proposed_role', 'insufficient_information')):
            raise ValueError('Missing visual assessment, provenance or reason')
        stamp = datetime.fromisoformat(row['reviewed_at'].replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('Review timestamp needs timezone')
        cases[row['blind_id']][key] = row
    if set(cases) != set(expected_ids):
        raise ValueError('Incomplete patient coverage')
    decisions = []
    for blind_id in sorted(cases):
        parts = cases[blind_id]
        if set(parts) != ALL_COMPONENTS:
            raise ValueError('Missing region or replacement-source assessment')
        if len({r['batch'] for r in parts.values()}) != 1:
            raise ValueError('Inconsistent review batch')
        exclusions = [{'region': k[0], 'component': k[1], 'status': parts[k]['status'],
                       'reason': parts[k]['reason']} for k in sorted(REQUIRED)
                      if parts[k]['status'] != 'acceptable']
        decisions.append({'blind_id': blind_id, 'batch': int(next(iter(parts.values()))['batch']),
                          'unchanged_interface_eligible': not exclusions,
                          'exclusions': exclusions,
                          'shifted_probe_statuses': {
                              k[1]: parts[k]['status'] for k in sorted(ALL_COMPONENTS - REQUIRED)}})
    counts = Counter(r['status'] for r in rows)
    summary = {'review_kind': 'AI_semantic_screening', 'reviewer_model': 'gpt-6-astra',
               'human_expert_reviews': 0, 'patients': len(cases), 'rows': len(rows),
               'row_status_counts': {k: counts[k] for k in ('acceptable', 'uncertain', 'rejected')},
               'by_component': {}, 'by_batch': {}, 'regions_changed': False,
               'lung_coverage': 'unknown', 'test_images_read': 0,
               'new_GPU_seconds': 0,
               'qualification_rule': 'All eight evidence/control region and source rows acceptable; shifted-overlap probe separate',
               'claim_limit': 'AI-screened geometry; not human-validated medical evidence'}
    for name, component in sorted(ALL_COMPONENTS):
        c = Counter(r['status'] for r in rows if (r['region'], r['component']) == (name, component))
        summary['by_component'][name + ':' + component] = dict(c)
    for batch in (1, 2):
        selected = [d for d in decisions if d['batch'] == batch]
        summary['by_batch'][str(batch)] = {
            'patients': len(selected),
            'unchanged_interface_eligible': sum(d['unchanged_interface_eligible'] for d in selected),
            'excluded': sum(not d['unchanged_interface_eligible'] for d in selected)}
    return summary, decisions


def main():
    base = Path('private/rsna_v3')
    with (base / 'review/blinded/review.csv').open() as stream:
        original = list(csv.DictReader(stream))
    rows = []
    sources = sorted((base / 'ai_review').glob('review_R*.csv'))
    for path in sources:
        with path.open() as stream:
            rows.extend(csv.DictReader(stream))
    expected = {r['blind_id'] for r in original}
    summary, decisions = aggregate(rows, expected)
    if {(r['blind_id'], r['region'], r['component'], r['batch'], r['use']) for r in rows} != {
            (r['blind_id'], r['region'], r['component'], r['batch'], r['use']) for r in original}:
        raise ValueError('Review scope differs from unchanged original packet')
    summary['completed_at'] = datetime.now(timezone.utc).isoformat()
    summary['source_review_files'] = [p.name for p in sources]
    out = base / 'ai_review'
    with (out / 'review_combined.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['blind_id'], r['region'], r['component'])))
    (out / 'case_decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + '\n')
    Path('reports/rsna_v3_ai_review.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
