"""Compact board heuristic for TAP Death's 20G movement rules.

The features deliberately describe different failure modes rather than
repeatedly penalizing ordinary stack height:

* ``weighted_height`` is the sole general height term. Its center-biased
  column weights favor a low pyramid with a peak near the spawn column.
* ``holes`` are roofed cells with no horizontal path to either wall. They are
  genuinely enclosed on their row and exclude side-accessible overhang space.
* ``accessible_cavities`` are roofed cells that do have a horizontal path to
  a wall. They are recoverable under 20G and receive a much smaller penalty.
* ``well_cost`` sums the triangular excess depth of every well, so several
  dependencies cost more than one while one-row wells remain free.
* ``spawn_danger`` is a nonlinear penalty for high central columns. It is
  intentionally dormant on healthy stacks and becomes severe near the spawn.
"""

from __future__ import annotations

import deathbot_sim as ds

WIDTH = 10
HEIGHT = 20
SPAWN_COLUMNS = range(3, 7)
SPAWN_DANGER_HEIGHT = 14

# This is the only ordinary height term. Central cells are cheaper, naturally
# favoring a low pyramid without an additional correlated aggregate penalty.
HEIGHT_WEIGHTS = [1.55, 1.40, 1.25, 1.05, 1.00,
                  1.10, 1.20, 1.35, 1.50, 1.65]

WEIGHTS = {
    "weighted_height": -1.0,
    "holes": -8.0,
    "hole_depth": -1.5,
    "accessible_cavities": -0.75,
    "well_cost": -1.5,
    "spawn_danger": -4.0,
    "lines_cleared": +6.0,
}


def column_heights(board: "ds.Board") -> list[int]:
    """Return all ten column heights, including the one-row buffer."""
    return [board.column_height(col) for col in range(WIDTH)]


def classify_roofed_cells(
    board: "ds.Board", heights: list[int] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Partition roofed empty cells into enclosed holes and open cavities.

    A roofed cell is empty with at least one block above it in its column.
    It is side-accessible when an uninterrupted horizontal run of empty cells
    reaches either wall on that row. This simple 20G-oriented definition keeps
    holes and overhang space strictly disjoint.
    """
    heights = heights or column_heights(board)
    holes: list[tuple[int, int]] = []
    accessible: list[tuple[int, int]] = []
    for row in range(max(heights, default=0)):
        empty_to_left = True
        left_accessible = [False] * WIDTH
        for col in range(WIDTH):
            empty_to_left = empty_to_left and not board.cell(col, row)
            left_accessible[col] = empty_to_left

        empty_to_right = True
        right_accessible = [False] * WIDTH
        for col in range(WIDTH - 1, -1, -1):
            empty_to_right = empty_to_right and not board.cell(col, row)
            right_accessible[col] = empty_to_right

        for col in range(WIDTH):
            if row >= heights[col] or board.cell(col, row):
                continue
            cell = (col, row)
            if left_accessible[col] or right_accessible[col]:
                accessible.append(cell)
            else:
                holes.append(cell)
    return holes, accessible


def count_holes(board: "ds.Board", heights: list[int] | None = None) -> int:
    """Count only horizontally enclosed roofed cells."""
    holes, _ = classify_roofed_cells(board, heights)
    return len(holes)


def count_accessible_cavities(
    board: "ds.Board", heights: list[int] | None = None,
) -> int:
    """Count roofed cells connected horizontally to a side wall."""
    _, accessible = classify_roofed_cells(board, heights)
    return len(accessible)


def hole_depth(board: "ds.Board", holes: list[tuple[int, int]]) -> int:
    """Sum occupied cells above holes; deeply buried holes cost more."""
    return sum(
        sum(board.cell(col, upper) for upper in range(row + 1, board.height))
        for col, row in holes
    )


def well_depths(heights: list[int]) -> list[int]:
    """Return every narrow local well depth.

    A well is one column strictly below both neighbors (or below its one
    neighbor at an edge). Wide flat basins accept many pieces and therefore
    are left to the general height term rather than treated as dependencies.
    """
    depths: list[int] = []
    for col, level in enumerate(heights):
        if col == 0:
            depth = heights[1] - level
        elif col == WIDTH - 1:
            depth = heights[WIDTH - 2] - level
        else:
            depth = min(heights[col - 1], heights[col + 1]) - level
        if depth > 0:
            depths.append(depth)
    return depths


def well_cost(heights: list[int]) -> int:
    """Sum triangular excess depths; one-row wells remain free."""
    return sum((depth - 1) * depth // 2 for depth in well_depths(heights))


def spawn_danger(heights: list[int]) -> int:
    """Return nonlinear risk from central columns entering the spawn region."""
    return sum(
        max(0, heights[col] - SPAWN_DANGER_HEIGHT) ** 2
        for col in SPAWN_COLUMNS
    )


def extract_features(board: "ds.Board") -> dict[str, float]:
    """Extract the compact, mostly non-overlapping heuristic feature vector."""
    heights = column_heights(board)
    holes, accessible = classify_roofed_cells(board, heights)
    return {
        "weighted_height": sum(
            heights[col] * HEIGHT_WEIGHTS[col] for col in range(WIDTH)),
        "holes": float(len(holes)),
        "hole_depth": float(hole_depth(board, holes)),
        "accessible_cavities": float(len(accessible)),
        "well_cost": float(well_cost(heights)),
        "spawn_danger": float(spawn_danger(heights)),
        "lines_cleared": 0.0,
    }


def _weighted_reward(
    board: "ds.Board", weights: dict[str, float], lines_cleared: int = 0,
) -> float:
    features = extract_features(board)
    return (
        weights["weighted_height"] * features["weighted_height"]
        + weights["holes"] * features["holes"]
        + weights["hole_depth"] * features["hole_depth"]
        + weights["accessible_cavities"] * features["accessible_cavities"]
        + weights["well_cost"] * features["well_cost"]
        + weights["spawn_danger"] * features["spawn_danger"]
        + weights.get("lines_cleared", 0.0) * lines_cleared
    )


def reward(board: "ds.Board", lines_cleared: int = 0) -> float:
    """Score a post-clear board; higher values indicate safer afterstates."""
    return _weighted_reward(board, WEIGHTS, lines_cleared)


class HeuristicScorer:
    """Score lock candidates using their post-clear board states."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = {**WEIGHTS, **(weights or {})}

    def _reward(self, board: "ds.Board", lines_cleared: int = 0) -> float:
        return _weighted_reward(board, self.weights, lines_cleared)

    def score(self, board: "ds.Board", candidate: "ds.LockCandidate") -> float:
        sim = board.clone_board()
        sim.lock(candidate.piece)
        lines = sim.clear_lines()
        return self._reward(sim, lines)

    def score_afterstate(self, board: "ds.Board", lines_cleared: int = 0) -> float:
        return self._reward(board, lines_cleared)

    def score_batch(
        self, board: "ds.Board", candidates: list["ds.LockCandidate"],
        current_piece_idx: int | None = None, next_piece_idx: int | None = None,
    ) -> list[float]:
        return [self.score(board, candidate) for candidate in candidates]

    def __repr__(self) -> str:
        return f"HeuristicScorer(weights={self.weights})"
