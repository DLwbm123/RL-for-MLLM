"""Synthetic checks only: token geometry and outcome-independent blind ordering."""
import pytest
from PIL import Image
from scripts.prepare_rsna_v3_review import blind_order, draw_panel, token_rectangles, token_stats


def test_synthetic_source_cells_and_order():
    assert token_rectangles([0, 3, 7], 80, 40, [2, 4]) == [[0, 0, 20, 20], [60, 0, 80, 20], [60, 20, 80, 40]]
    with pytest.raises(ValueError):
        token_rectangles([8], 80, 40, [2, 4])
    with pytest.raises(ValueError):
        token_rectangles([0, 0], 80, 40, [2, 4])
    region = {'grid': [2, 4], 'evidence': {'tokens': [0, 1]}}
    stats = token_stats([1, 2], region, [[0, 0, 20, 20]], 80, 40)
    assert stats['evidence_token_iou'] == 1 / 3
    assert stats['annotated_token_intersection_count'] == 0
    panel = draw_panel(Image.new('RGB', (80, 40), 'black'), 'synthetic', tokens=[0, 3], grid=[2, 4])
    assert panel.getpixel((10, 44)) != (0, 0, 0)
    assert panel.getpixel((40, 44)) == (0, 0, 0)  # Never fill source bounding box.
    order = blind_order(list('abcdef'), list('bad'), 42)
    assert order == blind_order(list('fedcba'), list('dba'), 42)
    assert {key for batch, key in order if batch == 1} == set('bad')
    assert {key for batch, key in order if batch == 2} == set('cef')
    with pytest.raises(ValueError):
        blind_order(['a'], ['b'], 42)
