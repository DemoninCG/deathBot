"""Designated-well dense setup potential (§15.1.3).

Replaces the thresholded all-or-nothing ``well_setup_score`` (>=6 deep,
availability 1.8%) with a DENSE potential that rewards partial progress
from the states the flat-stacking policy actually visits.

Key design points from the independent review:
- FIXED well column (default 9, rightmost — TAP standard). Never
  argmax-depth: switching the target well collects fake progress.
- Credit starts BELOW 4 (depth 1-3 give partial credit); saturates at a
  usable 4-line setup (deeper is not better — hoarding is dangerous).
- Explicit Tetris completion reward lives OUTSIDE Phi (in R_clear /
  line_bonus): a setup-only objective punishes cashing out, because
  clearing destroys the setup features.
- All features are exact numpy on packed [21] rows (no learning needed).

Feature vector (all from the post-clear afterstate):
  rows_full_except_well : rows where all 9 non-well cols are filled
  outside_fill          : total filled cells outside well in the top-8
                          band above the well bottom (partial-progress signal)
  well_depth_sat        : min(depth,4)/4 with depth = wall - well height
  well_open             : 1.0 if no hole/overhang obstructs the well channel
  holes_outside         : enclosed holes outside the well column
  roughness_outside     : sum |h[i]-h[i+1]| over non-well adjacencies
  tetris_ready          : 1.0 if depth>=4, well open, top<=14 (reachable I)
  height_penalty        : max(0, top-12) (risk term, subtracted)

Phi = w_full*rows_full + w_fill*outside_fill/8 + w_depth*well_depth_sat
    + w_open*well_open + w_ready*tetris_ready
    - w_holes*holes_outside - w_rough*roughness_outside/10
    - w_height*height_penalty/4
Default weights are a starting heuristic for inspection, NOT a tuned
teacher — trajectories come first, then CEM/random search (§15.1.7).
"""

from __future__ import annotations

import numpy as np

WELL_COL = 9
TOP_SAFE = 14
SETUP_ROWS = 4

DEFAULT_WEIGHTS = {
    "rows_full": 1.0,
    "fill": 0.5,
    "depth": 1.0,
    "open": 0.5,
    "ready": 2.0,
    "holes": 1.0,
    "rough": 0.3,
    "height": 0.5,
}


def _occ(rows21) -> np.ndarray:
    rows = np.asarray(rows21, dtype=np.uint32).reshape(-1)
    occ = np.zeros((21, 10), dtype=bool)
    for r in range(21):
        bits = int(rows[r])
        for c in range(10):
            if bits & (1 << c):
                occ[r, c] = True
    return occ


def _heights(occ: np.ndarray) -> np.ndarray:
    h = np.zeros(10, dtype=int)
    for c in range(10):
        col = np.where(occ[:, c])[0]
        h[c] = int(col.max()) + 1 if len(col) else 0
    return h


def dense_well_features(rows21, well_col: int = WELL_COL) -> dict:
    """Exact dense precursor features for one afterstate."""
    occ = _occ(rows21)
    h = _heights(occ)
    w = int(well_col)
    others = [c for c in range(10) if c != w]
    top = min(int(h.max()), 21)
    hw = min(int(h[w]), 21)
    # Well depth: higher-neighbor wall definition (matches well_setup_score).
    if w == 0:
        depth = int(h[1] - h[0])
    elif w == 9:
        depth = int(h[8] - h[9])
    else:
        depth = int(max(h[w - 1], h[w + 1]) - h[w])
    depth = max(0, depth)
    # Rows filled except the well: only rows AT/ABOVE the well floor count.
    # Buried full rows below the well bottom are just a flat stack, not setup.
    rows_full = 0
    for r in range(hw, top + 1):
        if r >= 21:
            break
        if all(occ[r, c] for c in others):
            rows_full += 1
    # Partial fill: filled outside-well cells in an 8-row band above well bottom.
    band_lo = hw
    band_hi = min(21, band_lo + 8)
    outside_fill = sum(int(occ[r, c]) for r in range(band_lo, band_hi)
                       for c in others)
    # Holes outside the well (enclosed = no side path on their row).
    holes_outside = 0
    for row in range(min(top, 21)):
        row_occ = occ[row]
        left = np.cumprod(~row_occ)
        right = np.cumprod(~row_occ[::-1])[::-1]
        for c in others:
            if row >= h[c] or row_occ[c]:
                continue
            if not (left[c] or right[c]):
                holes_outside += 1
    # Well channel open? A hole strictly inside the well column, or any
    # filled cell above the well bottom that juts over the channel, blocks it.
    # Approximation: well column must have no enclosed hole and no overhang
    # cell from a neighbor above the well floor.
    well_hole = 0
    for row in range(hw):
        if occ[row, w]:
            continue
        # empty cell below well top with a filled cell above in same col
        if occ[row + 1:, w].any():
            well_hole += 1
    well_open = 1.0 if well_hole == 0 else 0.0
    # Roughness outside the well.
    rough = 0
    prev = None
    for c in others:
        if prev is not None:
            # only count adjacent non-well pairs to avoid well-edge jumps
            rough += abs(int(h[c]) - int(prev))
        prev = h[c]
    tetris_ready = 1.0 if (depth >= 4 and well_open and top <= TOP_SAFE) else 0.0
    return {
        "rows_full_except_well": float(rows_full),
        "outside_fill": float(outside_fill),
        "well_depth_sat": min(depth, SETUP_ROWS) / float(SETUP_ROWS),
        "well_depth_raw": float(depth),
        "well_open": float(well_open),
        "holes_outside": float(holes_outside),
        "roughness_outside": float(rough),
        "tetris_ready": float(tetris_ready),
        "height_penalty": float(max(0, top - 12)),
        "top": float(top),
    }


def well_potential(rows21, well_col: int = WELL_COL,
                   weights: dict | None = None) -> float:
    """Scalar dense potential Phi(s) for one afterstate."""
    wth = dict(DEFAULT_WEIGHTS)
    if weights:
        wth.update(weights)
    f = dense_well_features(rows21, well_col)
    return (wth["rows_full"] * min(f["rows_full_except_well"], 4) / 4.0
            + wth["fill"] * min(f["outside_fill"], 36) / 36.0
            + wth["depth"] * f["well_depth_sat"]
            + wth["open"] * f["well_open"]
            + wth["ready"] * f["tetris_ready"]
            - wth["holes"] * min(f["holes_outside"], 4) / 4.0
            - wth["rough"] * min(f["roughness_outside"], 20) / 20.0
            - wth["height"] * min(f["height_penalty"], 8) / 8.0)


def style_fn_factory(well_col: int = WELL_COL, weights: dict | None = None,
                     scale: float = 1.0):
    """Build a ``style_fn(packed_rows_list, lines_list)`` for TwoPlyScorer.

    ``scale`` converts Phi units to z-units (like line_bonus scale).
    The Tetris-completion reward stays in line_bonus/R_clear, NOT here.
    """
    def _fn(packed_rows_list, lines_list):
        return [scale * well_potential(rows, well_col, weights)
                for rows in packed_rows_list]
    return _fn
