"""BUS-BRA release mapping and immutable, case-grouped development protocol."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.model_selection import train_test_split

QUESTIONS = {
    'A': 'Based on this breast ultrasound image, is the lesion more likely benign or malignant? Answer with exactly one label: benign or malignant.',
    'B': 'Locate the breast lesion. Return its bounding box as [x1, y1, x2, y2], with coordinates normalized to 0–1000.',
    'C': 'Predict the BI-RADS assessment category for the lesion in this image. Answer with exactly one category: 2, 3, 4, or 5.',
    'A_equal_tokens': 'Based on this breast ultrasound image, is the lesion more likely benign or malignant? Choose A for benign or B for malignant. Answer with exactly one letter: A or B.',
}
LABELS = ['benign', 'malignant']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def bbox(mask):
    y, x = np.where(mask)
    if len(x) == 0:
        return None
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def transform_box(box, width, height, inverse=False):
    scale = np.array([width, height, width, height], dtype=float)
    a = np.asarray(box, dtype=float)
    if inverse:
        return (np.clip(a, 0, 1000) * scale / 1000).tolist()
    return np.rint(np.clip(a / scale, 0, 1) * 1000).astype(int).tolist()


def smart_size(height, width, min_tokens=576, max_tokens=1024):
    # Qwen2.5-VL processor: patch 14, spatial merge 2; no crop or padding.
    h, w = max(28, round(height / 28) * 28), max(28, round(width / 28) * 28)
    if h * w > max_tokens * 784:
        ratio = np.sqrt(height * width / (max_tokens * 784))
        h, w = int(np.floor(height / ratio / 28) * 28), int(np.floor(width / ratio / 28) * 28)
    elif h * w < min_tokens * 784:
        ratio = np.sqrt(min_tokens * 784 / (height * width))
        h, w = int(np.ceil(height * ratio / 28) * 28), int(np.ceil(width * ratio / 28) * 28)
    return h, w


def group_rows(frame):
    """Union cases connected by either exact byte or decoded pixel duplicates."""
    parent = {str(c): str(c) for c in frame.case_id.unique()}
    def root(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for column in ['image_sha256', 'pixel_sha256']:
        for _, rows in frame.groupby(column):
            cases = [str(c) for c in rows.case_id.unique()]
            for c in cases[1:]:
                parent[root(c)] = root(cases[0])
    return frame.case_id.astype(str).map(lambda c: root(c))


def split_frame(frame, seed=42):
    groups = frame.groupby('duplicate_group').pathology.agg(lambda x: '|'.join(sorted(set(x))))
    # Conflicting target groups are kept together, not majority-voted.
    if any('|' in label for label in groups):
        raise ValueError('Conflicting exact-duplicate group labels require review before splitting')
    train, held = train_test_split(groups.index, test_size=.30, stratify=groups, random_state=seed)
    val, test = train_test_split(held, test_size=.50, stratify=groups.loc[held], random_state=seed)
    mapping = {g: s for s, rows in [('train', train), ('validation', val), ('test', test)] for g in rows}
    result = frame.duplicate_group.map(mapping)
    for column in ['case_id', 'duplicate_group', 'image_sha256', 'pixel_sha256']:
        assert frame.assign(split=result).groupby(column).split.nunique().max() == 1
    return result


def prepare(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = root / 'raw' / 'BUSBRA'
    meta = pd.read_csv(raw / 'bus_data.csv')
    expected = ['ID', 'Case', 'Histology', 'Pathology', 'BIRADS', 'Device', 'Width', 'Height', 'Side', 'BBOX']
    if list(meta.columns) != expected or meta.ID.duplicated().any():
        raise ValueError('Release schema changed or duplicate IDs')
    mapping = {'image_id': 'ID', 'case_id': 'Case', 'pathology': 'Pathology', 'birads': 'BIRADS',
               'histology': 'Histology', 'device': 'Device', 'side_raw': 'Side',
               'bbox_xyxy': 'computed from Masks, half-open xyxy; not CSV BBOX',
               'case_key_basis': 'Official CSV Case; 1064 distinct keys align with official 1064 patients. This is supporting consistency, not an independently documented patient dictionary.',
               'patient_identity_confirmed': False,
               'side_semantics': 'Preserve left/right/single verbatim; do not interpret as breast laterality or scan direction.',
               'label_mapping': {'benign': 'benign', 'malignant': 'malignant'}}
    write_json(output / 'schema_mapping.json', mapping)
    rows, perceptual = [], []
    for record in meta.to_dict('records'):
        image_id = record['ID']
        ip = raw / 'Images' / (image_id + '.png')
        mp = raw / 'Masks' / (image_id.replace('bus_', 'mask_', 1) + '.png')
        if record['Pathology'] not in LABELS:
            raise ValueError('Unexpected pathology: ' + str(record['Pathology']))
        image = Image.open(ip); image.load()
        mask_image = Image.open(mp); mask_image.load()
        arr, mask_arr = np.asarray(image), np.asarray(mask_image)
        if mask_arr.ndim != 2 or mask_arr.shape != arr.shape[:2]:
            raise ValueError('Mask geometry mismatch: ' + image_id)
        metadata_geometry_mismatch = image.size != (record['Width'], record['Height'])
        mask = mask_arr > 0
        components = ndimage.label(mask)[1]
        box = bbox(mask)
        reasons = []
        if box is None: reasons.append('empty_mask')
        if components != 1: reasons.append('multiple_components_need_review')
        if not set(np.unique(mask_arr)).issubset({0, 255, 1}): reasons.append('nonbinary_mask')
        row = {'image_id': image_id, 'case_id': str(record['Case']),
               'image_path': str(ip.relative_to(root)), 'mask_path': str(mp.relative_to(root)),
               'pathology': record['Pathology'], 'birads': int(record['BIRADS']),
               'width': image.width, 'height': image.height, 'mode': image.mode,
               'metadata_width':record['Width'],'metadata_height':record['Height'],
               'metadata_geometry_mismatch':metadata_geometry_mismatch,
               'image_sha256': sha(ip.read_bytes()),
               'pixel_sha256': sha(str((arr.shape, str(arr.dtype))).encode() + arr.tobytes()),
               'mask_area_ratio': float(mask.mean()), 'bbox_xyxy': json.dumps(box),
               'mask_values': json.dumps(np.unique(mask_arr).tolist()), 'mask_components': components,
               'evidence_eligible': not reasons, 'exclusion_reason': ';'.join(reasons),
               'histology': record['Histology'], 'device': record['Device'], 'side_raw': record['Side']}
        small = np.asarray(image.convert('L').resize((9, 8)), dtype=np.int16)
        bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
        perceptual.append(int.from_bytes(np.packbits(bits).tobytes(), 'big'))
        rows.append(row)
    frame = pd.DataFrame(rows)
    conflicts = frame.groupby('case_id').pathology.nunique()
    if (conflicts > 1).any():
        raise ValueError('Within-case label conflict; retain metadata, resolve lesion identity before training')
    frame['duplicate_group'] = group_rows(frame)
    # Near-duplicate candidates are only screened; they are not silently deleted.
    near = []
    for i, h in enumerate(perceptual):
        for j in range(i):
            if rows[i]['case_id'] != rows[j]['case_id'] and (h ^ perceptual[j]).bit_count() <= 4:
                near.append({'image_a': rows[j]['image_id'], 'image_b': rows[i]['image_id'], 'distance': (h ^ perceptual[j]).bit_count(),
                             'exact': rows[i]['pixel_sha256'] == rows[j]['pixel_sha256']})
    write_json(output / 'near_duplicate_candidates.json', near)
    # Unconfirmed cross-case near-duplicates are quarantined before split, not removed from classification silently.
    quarantine = {r[k] for r in near if not r['exact'] for k in ['image_a', 'image_b']}
    q_cases = set(frame.loc[frame.image_id.isin(quarantine), 'case_id'])
    frame['quarantine'] = frame.case_id.isin(q_cases)
    frame['split'] = 'quarantine'
    active = frame.loc[~frame.quarantine].copy()
    frame.loc[active.index, 'split'] = split_frame(active)
    frame['case_sample_weight'] = 1 / frame.groupby('case_id').image_id.transform('count')
    frame.to_csv(output / 'manifest.csv', index=False)
    frame.to_parquet(output / 'manifest.parquet', index=False)
    frame[['image_id', 'case_id', 'duplicate_group', 'split']].to_csv(output / 'splits.csv', index=False)
    folds = {}
    for k in [5, 10]:
        f = pd.read_csv(raw / f'{k}-fold-cv.csv').merge(frame[['image_id', 'case_id']], left_on='ID', right_on='image_id', validate='one_to_one')
        folds[str(k)] = {'test_fold_case_crossings': int((f.groupby('case_id').kFold.nunique() > 1).sum()),
                         'train_validation_case_crossings': {str(j): int((f.loc[f.kFold != j].groupby('case_id')[f'valid_{j}'].nunique() > 1).sum()) for j in range(1, k + 1)}}
    report = {'images': len(frame), 'cases': int(frame.case_id.nunique()), 'pathology_images': frame.pathology.value_counts().to_dict(),
              'pathology_cases': frame.drop_duplicates('case_id').pathology.value_counts().to_dict(),
              'case_label_conflicts': int((conflicts > 1).sum()), 'duplicate_file_rows': int(frame.image_sha256.duplicated().sum()),
              'duplicate_pixel_rows': int(frame.pixel_sha256.duplicated().sum()), 'near_duplicate_pairs': len(near),
              'quarantined_cases': len(q_cases), 'mask_components': frame.mask_components.value_counts().to_dict(),
              'image_modes': frame['mode'].value_counts().to_dict(), 'metadata_geometry_mismatches':frame.loc[frame.metadata_geometry_mismatch,['image_id','width','height','metadata_width','metadata_height']].to_dict('records'), 'fold_audit': folds,
              'split_images': frame.split.value_counts().to_dict(), 'split_cases': frame.drop_duplicates('case_id').split.value_counts().to_dict(),
              'split_labels': frame.groupby(['split','pathology']).size().to_dict()}
    report['split_labels'] = {str(k): int(v) for k, v in report['split_labels'].items()}
    write_json(output / 'data_audit.json', report)
    qa = []
    for row in frame.to_dict('records'):
        if row['split'] == 'quarantine': continue
        targets = {'A': row['pathology'], 'B': json.dumps(transform_box(json.loads(row['bbox_xyxy']), row['width'], row['height'])) if json.loads(row['bbox_xyxy']) and row['mask_components']==1 else None,
                   'C': str(row['birads']) if row['birads'] in [2,3,4,5] else None}
        for task, answer in targets.items():
            if answer is not None:
                qa.append({'image_id': row['image_id'], 'case_id': row['case_id'], 'split': row['split'], 'task': task,
                           'question': QUESTIONS[task], 'answer': answer, 'image_path': row['image_path']})
    with (output / 'qa.jsonl').open('w') as f:
        for row in qa: f.write(json.dumps(row) + '\n')
    write_json(output / 'templates.json', QUESTIONS)
    lock = {'version': '0.1', 'split_seed': 42, 'training_seeds_formal': [17,42,123], 'test_access': 'integrity_only',
            'case_aggregation': 'disabled_until_same_target_identity_confirmed',
            'hashes': {n: sha((output / n).read_bytes()) for n in ['manifest.csv','splits.csv','templates.json','schema_mapping.json']}}
    write_json(output / 'protocol_lock.json', lock)
    return report


def main():
    print(json.dumps(prepare(os.environ['DATA_ROOT'], Path(os.environ['OUTPUT_ROOT']) / 'protocol'), indent=2))


if __name__ == '__main__': main()
