"""Unit tests for well_setup_score: flat=0, deep clean well>0, holes=0."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from two_ply import well_setup_score


def _rows_from_heights(h):
    rows = np.zeros(21, dtype=np.uint32)
    for c, hh in enumerate(h):
        for r in range(hh):
            rows[r] |= np.uint32(1 << c)
    return rows


def test_flat_board_scores_zero():
    assert well_setup_score(_rows_from_heights([5] * 10)) == 0.0


def test_deep_clean_well_scores_positive():
    # Column 9 dug 9 deep, rest at 10: single clean well.
    h = [10] * 9 + [1]
    s = well_setup_score(_rows_from_heights(h))
    assert s > 0.8, s


def test_shallow_well_scores_zero():
    h = [10] * 9 + [6]  # depth 4 < 6
    assert well_setup_score(_rows_from_heights(h)) == 0.0


def test_hole_disqualifies():
    # Well shape but with a hole: col 9 has cells at rows 0..3 and 5 (gap at 4).
    rows = np.zeros(21, dtype=np.uint32)
    for c in range(9):
        for r in range(10):
            rows[r] |= np.uint32(1 << c)
    for r in [0, 1, 2, 3, 5]:
        rows[r] |= np.uint32(1 << 9)
    assert well_setup_score(rows) == 0.0


def test_two_deep_wells_scores_zero():
    h = [1, 10, 10, 10, 10, 10, 10, 10, 10, 1]  # two depth-9 wells
    assert well_setup_score(_rows_from_heights(h)) == 0.0


def test_tall_stack_scales_down():
    h = [18] * 9 + [9]  # depth 9 well but stack top at 18
    s_tall = well_setup_score(_rows_from_heights(h))
    h2 = [10] * 9 + [1]
    s_low = well_setup_score(_rows_from_heights(h2))
    assert 0.0 < s_tall < s_low
