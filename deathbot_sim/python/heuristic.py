"""Heuristic reward function for TAP Death mode (phase 2.2).

Encodes the survival-oriented board-shape intuitions:

  1. Aggregate height: lower stacks are better (more room before topping out).
  2. Pyramid shape: peak near column 5, with a slight LEFT-side preference.
     Penalized via a smooth function of distance from column 5, asymmetric
     (left side cheaper than right). A pyramid lets the piece slide either
     direction under 20G.
  3. Holes: empty cells with a filled cell above them in the same column.
     Penalized, weighted higher for middle columns (hard to open up in 20G).
  4. Overhangs: empty cells with filled cells above AND below but not fully
     enclosed (i.e. reachable from the side). Punished less than holes since
     they're recoverable by sliding a piece under.
  5. Deep wells: the deepest column gap below the stack. A 1-row well is fine
     (useful for I); deeper wells create I-piece dependencies. Punished.
  6. No line-clear bonus: survival is the metric, not tetrises. Clearing lines
     already reduces height, which feature (1) rewards implicitly.

All features are computed on a `deathbot_sim.Board`. The main entry point is
`HeuristicScorer`, which implements the `Scorer` protocol by simulating a
candidate lock, clearing lines (so the network sees the post-clear board), and
returning the reward. Higher = better board.

The feature weights are exposed as a module constant so they can be tuned
without code changes. The `extract_features` function returns the raw feature
vector (useful for the visualizer + dataset generation).
"""

from __future__ import annotations

from typing import Protocol

import deathbot_sim as ds

# Number of columns (must match the Rust board).
WIDTH = 10
HEIGHT = 20  # visible rows

# ---- Feature weights ------------------------------------------------------
# These are NEGATIVE for bad features (height, holes, overhangs, wells, pyramid
# deviation). A higher total reward = a better board. Tunable.
#
# Rationale for the magnitudes:
#   - holes_mid: the single most punishing feature; a mid-column hole in 20G
#     is nearly unrecoverable.
#   - holes_edge: edge holes are easier to open up.
#   - aggregate_height: per-cell-of-height penalty; summed over 10 columns so
#     a flat board at height 5 contributes 50 * this weight.
#   - pyramid: per-column penalty weighted by distance from col 5.
#   - overhang: roughly half the hole penalty (recoverable).
#   - well_depth: only the excess beyond 1 row is punished (a 1-row well is
#     fine; deeper creates I dependency).
WEIGHTS = {
    "aggregate_height": -1.0,
    "pyramid": -1.5,        # per unit of weighted distance from col 5
    "holes_mid": -4.0,      # cols 3-6 (0-indexed)
    "holes_edge": -2.0,     # cols 0-2, 7-9
    "overhang": -1.5,       # per overhang cell
    "well_depth": -2.0,     # per row of well depth beyond 1
    "row_transitions": -0.3,  # per row transition (surface roughness)
    "lines_cleared": +7.6,  # per line cleared (encourages clears; tetrises = +30.4)
}


# ---- Pyramid column weights -----------------------------------------------
# A smooth function of distance from column 5 (0-indexed: col index 4 is the
# center of a 10-wide board, but TAP spawn is col 3 so we bias slightly right
# to col 5 = index 4... actually let's use index 4 as the peak, with the LEFT
# side (cols 0-3) using a gentler slope than the RIGHT side (cols 5-9).
#
# w(c) = 1 + k * dist, where:
#   - dist = |c - 4|  (distance from the peak column, index 4)
#   - k_left  = 0.5  (gentler: left side preference)
#   - k_right = 0.8  (steeper: right side punished more)
# So col 4 (peak) has weight 1.0; col 0 has 1 + 0.5*4 = 3.0; col 9 has
# 1 + 0.8*5 = 5.0. These multiply the column height in the pyramid penalty.

def pyramid_weight(col: int) -> float:
    """Smooth distance-from-peak weight, asymmetric (left gentler than right)."""
    peak = 4  # column index 4 (the 5th column, 1-indexed)
    dist = abs(col - peak)
    if col < peak:
        k = 0.5  # left side: gentler slope (slight left preference)
    else:
        k = 0.8  # right side: steeper slope
    return 1.0 + k * dist


PYRAMID_WEIGHTS = [pyramid_weight(c) for c in range(WIDTH)]


# ---- Feature extraction ---------------------------------------------------

def column_heights(board: "ds.Board") -> list[int]:
    """Height of each column (0..WIDTH), counting from the bottom."""
    return [board.column_height(c) for c in range(WIDTH)]


def count_holes(board: "ds.Board", heights: list[int]) -> tuple[int, int]:
    """Count hole cells (empty, with a filled cell above in the same column).

    Returns (holes_mid, holes_edge) where _mid is holes in columns 3-6
    (0-indexed) and _edge is holes in columns 0-2, 7-9. Middle holes are
    harder to recover in 20G.
    """
    mid_cols = {3, 4, 5, 6}
    holes_mid = 0
    holes_edge = 0
    for c in range(WIDTH):
        h = heights[c]
        for r in range(h):
            if not board.cell(c, r):
                if c in mid_cols:
                    holes_mid += 1
                else:
                    holes_edge += 1
    return holes_mid, holes_edge


def count_overhangs(board: "ds.Board", heights: list[int]) -> int:
    """Count overhang cells: empty cells with a filled cell somewhere above
    AND a filled cell directly below (i.e. reachable from the side, not a
    true hole). These are recoverable by sliding a piece under, so they're
    punished less than holes.

    Formally: a cell (c, r) is an overhang if:
      - it's empty,
      - there's a filled cell at (c, r-1) (directly below is filled),
      - there's a filled cell at some (c, r') with r' > r (something above),
      - AND it's NOT a hole (a hole has filled cells both above AND the column
        is blocked below — but we defined holes as empty-below-height, so an
        overhang is specifically an empty cell with filled-below but where the
        cell is above the column's contiguous stack).

    Simplification: an overhang is an empty cell (c, r) where (c, r-1) is
    filled and (c, r) is above the column's solid-stack-top. We compute the
    solid stack top as the highest contiguous-from-bottom filled cell.
    """
    overhangs = 0
    for c in range(WIDTH):
        # Find the solid stack top: highest row where the column is filled
        # contiguously from the bottom.
        solid_top = -1
        for r in range(HEIGHT):
            if board.cell(c, r):
                solid_top = r
            else:
                break  # first gap ends the solid stack
        # Any empty cell above solid_top with a filled cell below it is an
        # overhang (not a hole, since holes are empty cells within the
        # column's height range — but we count overhangs as the cells above
        # the solid stack that have filled cells above them).
        h = heights[c]
        for r in range(solid_top + 1, h):
            # (c, r) is empty (it's above solid_top) and below the column top.
            # It has filled cells above (within the column height). Check if
            # it has a filled cell directly below — if so it's a true overhang
            # (a "pocket"); otherwise it's part of a well.
            if r > 0 and board.cell(c, r - 1):
                overhangs += 1
    return overhangs


def max_well_depth(board: "ds.Board", heights: list[int]) -> int:
    """The deepest well: the largest gap between a column's height and its
    neighbors' heights. A 1-row well is fine (useful for I); we return the
    raw max well depth and let the scorer subtract 1 (only excess depth is
    punished).
    """
    max_well = 0
    for c in range(WIDTH):
        left = heights[c - 1] if c > 0 else HEIGHT
        right = heights[c + 1] if c < WIDTH - 1 else HEIGHT
        neighbor_min = min(left, right)
        well = max(0, neighbor_min - heights[c])
        if well > max_well:
            max_well = well
    return max_well


def count_row_transitions(board: "ds.Board", heights: list[int]) -> int:
    """Count horizontal filled/empty transitions across each row. More
    transitions = rougher surface = harder to place pieces cleanly. This is
    a surface-roughness penalty that complements the pyramid shape."""
    transitions = 0
    for r in range(HEIGHT):
        prev = True  # treat the left wall as filled
        for c in range(WIDTH):
            cur = board.cell(c, r)
            if cur != prev:
                transitions += 1
            prev = cur
        if not prev:  # right wall transition (empty -> wall)
            transitions += 1
    return transitions


def extract_features(board: "ds.Board") -> dict[str, float]:
    """Compute all heuristic features for a board. Returns a dict mapping
    feature name to raw value (before weighting).

    NOTE: `lines_cleared` is always 0 here since this function operates on a
    static board (post-clear). The HeuristicScorer computes lines separately
    during lock simulation and applies the weight in its `_reward` method.
    """
    heights = column_heights(board)
    agg = sum(heights)
    holes_mid, holes_edge = count_holes(board, heights)
    overhangs = count_overhangs(board, heights)
    well = max_well_depth(board, heights)
    row_trans = count_row_transitions(board, heights)
    # Pyramid penalty: sum of (column_height * pyramid_weight) — taller
    # columns far from the peak contribute more.
    pyramid = sum(heights[c] * PYRAMID_WEIGHTS[c] for c in range(WIDTH))
    return {
        "aggregate_height": float(agg),
        "pyramid": float(pyramid),
        "holes_mid": float(holes_mid),
        "holes_edge": float(holes_edge),
        "overhang": float(overhangs),
        "well_depth": float(well),
        "row_transitions": float(row_trans),
        "lines_cleared": 0.0,  # populated by HeuristicScorer; 0 for static boards
    }


def reward(board: "ds.Board", lines_cleared: int = 0) -> float:
    """Compute the total reward for a board state. Higher = better.

    This is the heuristic teacher's target value. The network learns to
    approximate this function. `lines_cleared` is 0 for a static board;
    pass the count explicitly if you're scoring a post-lock state.
    """
    feats = extract_features(board)
    w = WEIGHTS
    total = (
        w["aggregate_height"] * feats["aggregate_height"]
        + w["pyramid"] * feats["pyramid"]
        + w["holes_mid"] * feats["holes_mid"]
        + w["holes_edge"] * feats["holes_edge"]
        + w["overhang"] * feats["overhang"]
        + w["well_depth"] * max(0.0, feats["well_depth"] - 1.0)  # 1-row well is free
        + w["row_transitions"] * feats["row_transitions"]
        + w.get("lines_cleared", 0.0) * lines_cleared
    )
    return total


# ---- Scorer adapter -------------------------------------------------------

class HeuristicScorer:
    """Wraps the heuristic reward to implement the Scorer protocol.

    Simulates a candidate lock, clears lines (so the reward reflects the
    post-clear board), and returns the reward. Higher = better placement.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        if weights is not None:
            global WEIGHTS
            # We don't mutate the module constant; we store a local copy.
            self.weights = {**WEIGHTS, **weights}
        else:
            self.weights = WEIGHTS.copy()

    def _reward(self, board: "ds.Board", lines_cleared: int = 0) -> float:
        feats = extract_features(board)
        w = self.weights
        return (
            w["aggregate_height"] * feats["aggregate_height"]
            + w["pyramid"] * feats["pyramid"]
            + w["holes_mid"] * feats["holes_mid"]
            + w["holes_edge"] * feats["holes_edge"]
            + w["overhang"] * feats["overhang"]
            + w["well_depth"] * max(0.0, feats["well_depth"] - 1.0)
            + w["row_transitions"] * feats["row_transitions"]
            + w.get("lines_cleared", 0.0) * lines_cleared
        )

    def score(self, board: "ds.Board", candidate: "ds.LockCandidate") -> float:
        """Simulate the lock, clear lines, and score the resulting board.
        The line-clear bonus is applied explicitly (the post-clear board
        doesn't encode how many lines were cleared)."""
        sim = ds.Board()
        for r in range(board.height):
            for c in range(board.width):
                if board.cell(c, r):
                    sim.set_cell(c, r)
        sim.lock(candidate.piece)
        lines = sim.clear_lines()  # post-clear board + line count
        return self._reward(sim, lines_cleared=lines)

    def score_batch(
        self, board: "ds.Board", candidates: list["ds.LockCandidate"]
    ) -> list[float]:
        """Score many candidates. No batching speedup here (pure Python), but
        matches the NeuralScorer interface so the visualizer can swap them."""
        return [self.score(board, c) for c in candidates]

    def __repr__(self) -> str:
        return f"HeuristicScorer(weights={self.weights})"
