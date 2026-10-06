"""CPU-only, private blind packet for the unchanged v2 geometric regions."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import random
from datetime import datetime, timezone

from PIL import Image, ImageDraw

VERSION = 'v3-old-regions-review-1'


def token_rectangles(tokens, width, height, grid):
    """Map row-major merged visual tokens to original-image pixel cells."""
    gh, gw = grid
    if gh <= 0 or gw <= 0 or width <= 0 or height <= 0:
        raise ValueError('Invalid image/grid dimensions')
    if len(set(tokens)) != len(tokens):
        raise ValueError('Duplicate visual tokens')
    result = []
    for token in tokens:
        if not isinstance(token, int) or not 0 <= token < gh * gw:
            raise ValueError('Visual token outside image grid')
        y, x = divmod(token, gw)
        result.append([x * width / gw, y * height / gh,
                       (x + 1) * width / gw, (y + 1) * height / gh])
    return result


def token_stats(tokens, region, annotations, width, height):
    from src.regions import region_tokens
    cells = token_rectangles(tokens, width, height, region['grid'])
    selected = set(tokens)
    annotation_tokens = set()
    for box in annotations:
        annotation_tokens.update(region_tokens(box, width, height, *region['grid']))
    evidence = set(region['evidence']['tokens'])
    return {'token_ids': tokens, 'token_cell_xyxy_original_pixels': cells,
            'token_count': len(tokens),
            'image_token_fraction': len(tokens) / (region['grid'][0] * region['grid'][1]),
            'annotated_token_intersection_count': len(selected & annotation_tokens),
            'evidence_token_intersection_count': len(selected & evidence),
            'evidence_token_iou': len(selected & evidence) / len(selected | evidence),
            'lung_coverage': 'unknown'}


def draw_panel(image, title, annotations=(), tokens=(), grid=None, box=None, color='cyan'):
    """Draw the actual source cells; a source ring must never become a box."""
    out = image.copy().convert('RGB')
    draw = ImageDraw.Draw(out)
    if tokens:
        overlay = Image.new('RGBA', out.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for rect in token_rectangles(tokens, *image.size, grid):
            od.rectangle(rect, fill=(0, 210, 255, 45), outline=color, width=1)
        out = Image.alpha_composite(out.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(out)
    for annotation in annotations:
        draw.rectangle(annotation, outline='yellow', width=3)
    if box is not None:
        draw.rectangle(box, outline='red', width=3)
    canvas = Image.new('RGB', (out.width, out.height + 34), 'white')
    canvas.paste(out, (0, 34))
    ImageDraw.Draw(canvas).text((8, 8), title, fill='black')
    return canvas


def blind_order(eligible_ids, p4_ids, seed):
    if len(set(eligible_ids)) != len(eligible_ids) or len(set(p4_ids)) != len(p4_ids):
        raise ValueError('Duplicate patient/image identity')
    if not set(p4_ids) <= set(eligible_ids):
        raise ValueError('P4 geometric patient missing from eligible regions')
    rng = random.Random(seed)
    first = sorted(p4_ids)
    second = sorted(set(eligible_ids) - set(p4_ids))
    rng.shuffle(first)
    rng.shuffle(second)
    return [(1, key) for key in first] + [(2, key) for key in second]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    import pandas as pd
    from src.v2 import DevelopmentData
    from src.regions import region_tokens
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('Review preparation requires CUDA_VISIBLE_DEVICES empty')
    old = Path(os.environ['V2_OUTPUT_ROOT']).resolve()
    output = Path(os.environ['OUTPUT_ROOT']).resolve()
    if output == old or output.parent.name != 'rsna_diagnostic_v3':
        raise PermissionError('Use independent private v3 outputs directory')
    protocol = old / 'protocol'
    lock = json.loads((protocol / 'protocol_lock.json').read_text())
    for name in ['manifest.csv', 'regions.json', 'subsets.json']:
        if hashlib.sha256((protocol / name).read_bytes()).hexdigest() != lock['hashes'][name]:
            raise ValueError('Frozen v2 identity mismatch: ' + name)
    frame = pd.read_csv(protocol / 'manifest.csv', keep_default_na=False,
                        dtype={'case_id': str, 'image_id': str})
    if not set(frame.split) <= {'train', 'validation'}:
        raise PermissionError('Nondevelopment rows in v2 review manifest')
    regions = json.loads((protocol / 'regions.json').read_text())
    eligible = frame[frame.image_id.map(lambda key: regions.get(key, {}).get('eligible', False))]
    if eligible.case_id.duplicated().any() or eligible.image_id.duplicated().any():
        raise ValueError('Review requires one fixed image per patient')
    if eligible.split.value_counts().to_dict() != {'train': 69, 'validation': 24}:
        raise ValueError('Frozen eligible counts differ; do not silently change review scope')
    p4 = json.loads((protocol / 'subsets.json').read_text())['P4']
    p4_ids = [row['image_id'] for row in p4 if row['stratum'] == 'geometry_only_positive']
    if len(p4_ids) != 32:
        raise ValueError('P4 batch must contain all 32 geometric patients')
    seed = int(os.environ.get('REVIEW_SEED', '42'))
    if seed != 42:
        raise ValueError('Frozen review seed is 42')
    order = blind_order(eligible.image_id.tolist(), p4_ids, seed)
    root = output / 'review'
    root.mkdir(exist_ok=False)  # Never overwrite returned human review or prior materials.
    write_json(root / 'source_provenance.json', {
        'v2_source_hashes': {name: lock['hashes'][name] for name in ['manifest.csv', 'regions.json', 'subsets.json']},
        'review_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'unchanged v2 regions; only train/development images', 'blind_order_seed': seed})
    viewer = root / 'blinded'
    viewer.mkdir()
    data = DevelopmentData(os.environ['DATA_ROOT'], eligible, root / 'private_image_access.jsonl')
    data.install_guard()
    mapping, reviews, links = [], [], []
    for number, (batch, key) in enumerate(order, 1):
        blind_id = f'R{number:03d}'
        row = data.rows[key]
        usage = 'train' if row['split'] == 'train' else 'development'
        label = f'{blind_id} | {usage}'
        case = viewer / blind_id
        case.mkdir()
        image = data.image(key)
        width, height = image.size
        if (width, height) != (int(row['width']), int(row['height'])):
            raise ValueError('Image dimensions differ from frozen manifest')
        annotations = json.loads(row['boxes'])
        region = regions[key]
        if len(region['controls']) != 3:
            raise ValueError('Expected exactly three frozen controls')
        mapping.append({'blind_id': blind_id, 'image_id': key, 'case_id': row['case_id'],
                        'batch': batch, 'split': row['split']})
        draw_panel(image, label + ' | original').save(case / 'original.png')
        draw_panel(image, label + ' | all annotations', annotations).save(case / 'annotations.png')
        page = [f'<h1>{label}</h1>', '<p>Yellow: all existing annotations. Red: region box. Cyan: exact visual token cells.</p>',
                '<p>Lung coverage: unknown. Unannotated does not mean normal lung.</p>',
                '<h2>Original</h2><img src="original.png" alt="Unmodified image">',
                '<h2>All existing annotations</h2><img src="annotations.png" alt="All annotation boxes">']
        geometry = {'blind_id': blind_id, 'use': usage, 'image_size': [width, height],
                    'grid': region['grid'], 'annotations_xyxy': annotations, 'regions': {}}
        items = [('evidence', region['evidence']),
                 *[(f'control_{i}', control) for i, control in enumerate(region['controls'], 1)],
                 ('wrong', region['wrong'])]
        for name, item in items:
            display = 'shifted-overlap probe (historical wrong)' if name == 'wrong' else name
            if set(region_tokens(item['box'], width, height, *region['grid'])) != set(item['tokens']):
                raise ValueError('Frozen box/token mapping mismatch')
            if set(item['source']) & set(item['tokens']):
                raise ValueError('Replacement source intersects its target')
            for component, tokens in [('region', item['tokens']), ('replacement_source', item['source'])]:
                identifier = f'{name}_{component}'
                stats = token_stats(tokens, region, annotations, width, height)
                stats['box_xyxy'] = item['box'] if component == 'region' else None
                stats['source_target_intersection_count'] = len(set(item['source']) & set(item['tokens']))
                geometry['regions'][identifier] = stats
                title = f'{label} | {display} | {component}'
                draw_panel(image, title, annotations, tokens, region['grid'],
                           item['box'] if component == 'region' else None).save(case / (identifier + '.png'))
                page += [f'<h2>{html.escape(display)}: {component}</h2>',
                         f'<img src="{identifier}.png" alt="{html.escape(display)} {component}">',
                         '<pre>' + html.escape(json.dumps(stats, indent=2)) + '</pre>']
                reviews.append({'blind_id': blind_id, 'batch': batch, 'use': usage, 'region': name,
                                'display_name': display, 'component': component, 'status': 'pending',
                                'anatomy_reasonable': '', 'annotated_or_suspicious_abnormality': '',
                                'nonlung_artifact_or_text': '', 'fit_for_proposed_role': '',
                                'insufficient_information': '', 'reason': '', 'reviewer_identity': '',
                                'reviewer_qualification': '', 'reviewed_at': '',
                                'review_version': VERSION, 'lung_coverage': 'unknown'})
        write_json(case / 'geometry.json', geometry)
        (case / 'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1100px;margin:2em auto}img{max-width:100%}pre{white-space:pre-wrap}</style>' + '\n'.join(page))
        links.append(f'<li>Batch {batch}: <a href="{blind_id}/index.html">{label}</a></li>')
    with (viewer / 'review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    write_json(root / 'private_identity_mapping.json', mapping)
    instructions = '''<h1>Blinded regional review</h1><p>Review old, unchanged regions before any revision. Read each image and all annotated boxes, then assess each region and each replacement source independently. Source cells are the actual feature-mean replacement inputs, not their bounding rectangle. Pixel coordinates and row-major token IDs appear below every panel.</p>
<p>Use review.csv. Allowed status: pending / acceptable / uncertain / rejected. Record anatomy, annotated or suspicious abnormality, nonlung tissue/artifact/text, suitability for the proposed role, insufficient information, reason, reviewer identity/qualification, time, and version. Evidence is judged as the intended lesion region; controls and sources as non-evidence candidates; historical wrong is only a shifted-overlap probe, not a normal-lung control.</p>
<p>Do not infer normal lung from absent annotations. Lung coverage stays unknown without verifiable lung labels. Automated/model judgments are not expert review. Pending is not rejected. Acceptability does not establish reward validity.</p>
<p>Keep this entire folder private. Custodian identity mapping is outside this blinded folder. No scores, model correctness, reward ordering, or intervention values are included. First batch is all 32 predetermined cases; second batch is the remaining 61. Both are randomized with frozen seed 42.</p>
<p>After genuine review, preserve the original CSV and create a new dated version. Revise only for documented semantic reasons; freeze matching, exclusion, source rules and final manifest before any rescoring. Do not choose controls by model score. Retain all exclusions. Failure to find three acceptable controls and reliable sources makes the case ineligible. No automated eligibility adjudication is performed by this packet.</p>'''
    (viewer / 'index.html').write_text('<!doctype html><meta charset="utf-8">' + instructions + '<ol>' + '\n'.join(links) + '</ol>')
    status = {'stage': 'V3-R2', 'status': 'human_review_pending', 'packet_ready': True,
              'review_version': VERSION, 'seed': seed, 'patients': 93,
              'train': 69, 'development': 24, 'batch_1': 32, 'batch_2': 61,
              'human_approved_patients': 0, 'human_rejected_patients': 0,
              'pending_patients': 93, 'pending_region_source_rows': len(reviews),
              'images': len(order) * 12, 'image_access_counts': dict(data.access_counts),
              'test_images_read': 0, 'lung_coverage': 'unknown',
              'regions_changed': False, 'gpu_used': False,
              'created_at': datetime.now(timezone.utc).isoformat()}
    write_json(root / 'status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
