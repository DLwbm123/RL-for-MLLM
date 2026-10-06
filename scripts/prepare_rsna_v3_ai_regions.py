"""One deterministic proposal pass within visually supplied AI support polygons.

The support mask is not a verified lung segmentation. Final proposals require
separate blinded Astra review; this script never confers semantic acceptance.
"""
import csv
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw
from scripts.prepare_rsna_v3_review import draw_panel, token_rectangles


def box_tokens(box, width, height, grid):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Invalid or out-of-image box')
    return [i for i, (a, b, c, d) in enumerate(token_rectangles(
        list(range(grid[0] * grid[1])), width, height, grid))
        if a < x2 and c > x1 and b < y2 and d > y1]


def support_tokens(polygons, width, height, grid):
    mask = Image.new('1', (width, height))
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError('Support polygon needs at least three vertices')
        for point in polygon:
            if len(point) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in point):
                raise ValueError('Invalid support vertex')
            if not 0 <= point[0] <= width or not 0 <= point[1] <= height:
                raise ValueError('Support vertex outside image')
        draw.polygon([tuple(point) for point in polygon], fill=1)
    valid = set()
    for token, rect in enumerate(token_rectangles(list(range(grid[0] * grid[1])), width, height, grid)):
        # Require the complete original-pixel cell inside the AI proposal, not just its center.
        cell = mask.crop((math.floor(rect[0]), math.floor(rect[1]), math.ceil(rect[2]), math.ceil(rect[3])))
        if cell.getextrema() == (1, 1):
            valid.add(token)
    return valid


def local_source(target, valid, forbidden, grid):
    gh, gw = grid
    surrounding = set()
    for token in target:
        y, x = divmod(token, gw)
        surrounding.update(yy * gw + xx
                           for yy in range(max(0, y - 3), min(gh, y + 4))
                           for xx in range(max(0, x - 3), min(gw, x + 4)))
    return sorted((surrounding & valid) - set(target) - forbidden)


def build_proposal(geometry, polygons):
    width, height = geometry['image_size']; grid = geometry['grid']; gh, gw = grid
    old = geometry['regions']; evidence = old['evidence_region']
    box = evidence['box_xyxy']; ids = evidence['token_ids']
    if set(box_tokens(box, width, height, grid)) != set(ids):
        raise ValueError('Original evidence box/token identity differs')
    x1, y1, x2, y2 = box
    forbidden_box = [max(0, x1 - .1 * (x2 - x1)), max(0, y1 - .1 * (y2 - y1)),
                     min(width, x2 + .1 * (x2 - x1)), min(height, y2 + .1 * (y2 - y1))]
    forbidden = set(box_tokens(forbidden_box, width, height, grid))
    valid = support_tokens(polygons, width, height, grid)
    valid -= {y * gw + x for y in range(gh) for x in range(gw) if y in (0, gh - 1) or x in (0, gw - 1)}
    source = local_source(ids, valid, forbidden, grid)
    ys, xs = [i // gw for i in ids], [i % gw for i in ids]
    eh, ew = max(ys) - min(ys) + 1, max(xs) - min(xs) + 1
    positions = [(y, x) for y in range(1, gh - eh) for x in range(1, gw - ew)]
    random.Random(42).shuffle(positions)
    controls = []
    for y, x in positions:
        selected = {yy * gw + xx for yy in range(y, y + eh) for xx in range(x, x + ew)}
        if not selected <= valid or selected & forbidden:
            continue
        if abs((y + eh / 2) / gh - (y1 + y2) / 2 / height) > .10:
            continue
        if any(len(selected & set(c['tokens'])) / len(selected) > .75 for c in controls):
            continue
        donors = local_source(selected, valid, forbidden, grid)
        if len(donors) < 4:
            continue
        dx, dy = (x - min(xs)) * width / gw, (y - min(ys)) * height / gh
        new_box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        if set(box_tokens(new_box, width, height, grid)) != selected:
            continue
        controls.append({'box': new_box, 'tokens': sorted(selected), 'source': donors})
        if len(controls) == 3:
            break
    reasons = []
    if len(source) < 4:
        reasons.append('insufficient_AI_supported_local_evidence_source')
    if len(controls) < 3:
        reasons.append('fewer_than_three_matching_controls_inside_AI_support')
    return {'eligible_geometry': not reasons, 'reasons': reasons, 'grid': grid,
            'evidence': {'box': box, 'tokens': ids, 'source': source}, 'controls': controls,
            'AI_support_tokens': sorted(valid), 'lung_coverage': 'unknown',
            'semantic_review_status': 'pending', 'proposal_round': 1}


def main():
    root = Path('private/rsna_v3')
    with (root / 'ai_review/review_combined.csv').open() as stream:
        reviewed = list(csv.DictReader(stream))
    if len(reviewed) != 930 or any(r['status'] == 'pending' for r in reviewed):
        raise ValueError('Finish all 93 original-region reviews first')
    ids = {r['blind_id'] for r in reviewed if r['batch'] == '1'}
    if len(ids) != 32:
        raise ValueError('Preserve all 32 original P4 geometry cases')
    supports = {}
    for path in sorted((root / 'ai_review').glob('support_R*.json')):
        for item in json.loads(path.read_text()):
            if item['blind_id'] in supports:
                raise ValueError('Duplicate AI support record')
            if not item.get('reason') or item.get('reviewer_identity') != 'gpt-6-astra':
                raise ValueError('Missing visual support justification')
            supports[item['blind_id']] = item
    if set(supports) != ids:
        raise ValueError('AI support records must cover all original 32 cases, including exclusions')
    out = root / 'ai_revision'
    out.mkdir(exist_ok=False)  # One proposal pass, never overwrite a review return.
    proposals = {}
    for blind_id in sorted(ids):
        support = supports[blind_id]
        evidence_status = next(r['status'] for r in reviewed
                               if (r['blind_id'], r['region'], r['component']) == (blind_id, 'evidence', 'region'))
        if evidence_status != 'acceptable':
            proposals[blind_id] = {'eligible_geometry': False, 'reasons': ['original_evidence_not_AI_acceptable'],
                                   'proposal_round': 0, 'semantic_review_status': 'excluded'}
            continue
        source = root / 'review/blinded' / blind_id
        geometry = json.loads((source / 'geometry.json').read_text())
        proposal = build_proposal(geometry, support['support_polygons'])
        proposals[blind_id] = proposal
        if not proposal['eligible_geometry']:
            continue
        dest = out / blind_id; dest.mkdir()
        width, height = geometry['image_size']
        # Remove only the pre-existing title strip for new overlays; source pixels stay unchanged.
        with Image.open(source / 'original.png') as original:
            image = original.crop((0, 34, width, height + 34)).convert('RGB')
        image.save(dest / 'original.png')
        draw_panel(image, blind_id + ' | all annotations', geometry['annotations_xyxy']).save(dest / 'annotations.png')
        for name, item in [('evidence', proposal['evidence']), *[(f'control_{i}', c) for i, c in enumerate(proposal['controls'], 1)]]:
            for component, tokens in [('region', item['tokens']), ('replacement_source', item['source'])]:
                draw_panel(image, f'{blind_id} | revised {name} | {component}', geometry['annotations_xyxy'],
                           tokens, proposal['grid'], item['box'] if component == 'region' else None).save(dest / f'{name}_{component}.png')
        (dest / 'proposal.json').write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + '\n')
    (out / 'proposals.json').write_text(json.dumps(proposals, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'original_patients': len(proposals),
                      'geometric_proposals_for_AI_review': sum(p['eligible_geometry'] for p in proposals.values()),
                      'GPU_used': False, 'semantic_approval_automatic': False}))


if __name__ == '__main__':
    main()
