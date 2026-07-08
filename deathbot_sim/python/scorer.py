"""Scorer protocol + mock implementations for the deathbot visualizer / trainer.

The phase-2 neural network will implement `Scorer` (or be wrapped by an
adapter that does). For phase 1.4 we ship:
  - `MockScorer`: a classic hand-crafted Tetris heuristic (aggregate height,
    holes, bumpiness, lines cleared). Good enough to produce visibly-different
    scores across lock candidates so the visualizer has something sensible to
    show.
  - `MockNetwork`: a stand-in for the real NN with a `layer_sizes` attribute
    and a `score(board, candidate) -> float` method. The visualizer reads
    `layer_sizes` to draw the neuron layout.

Phase 2 will replace `MockNetwork` with a real PyTorch / tinygrad model that
implements the same interface; no visualizer changes required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import deathbot_sim as ds


@runtime_checkable
class Scorer(Protocol):
    """Anything that can score a lock candidate on a board.

    Higher score = better placement. The visualizer's testing panel ranks
    candidates by this score and highlights the argmax.
    """

    def score(self, board: "ds.Board", candidate: "ds.LockCandidate") -> float: ...


class MockScorer:
    """Hand-crafted Tetris heuristic.

    Features (computed by simulating the lock):
      - aggregate_height: sum of column heights (lower is better)
      - holes: total empty cells below a filled cell in the same column
        (lower is better)
      - bumpiness: sum of |height[i] - height[i+1]| (lower is better)
      - lines_cleared: number of full rows completed by this lock
        (higher is better)

    Weights are the classic Dellacherie-style values; tuned to produce
    visibly-different scores so the visualizer is interesting to watch.
    """

    # Feature weights: (aggregate_height, holes, bumpiness, lines_cleared)
    WEIGHTS = (-0.51, -0.36, -0.18, +0.76)

    def __init__(self, weights: tuple[float, ...] | None = None) -> None:
        self.weights = weights or self.WEIGHTS

    def score(self, board: "ds.Board", candidate: "ds.LockCandidate") -> float:
        # Simulate the lock on a copy of the board.
        sim = ds.Board()
        for r in range(board.height):
            for c in range(board.width):
                if board.cell(c, r):
                    sim.set_cell(c, r)
        sim.lock(candidate.piece)
        lines = sim.clear_lines()

        heights = [sim.column_height(c) for c in range(sim.width)]
        agg = sum(heights)
        holes = sum(
            heights[c] - sum(1 for r in range(heights[c]) if sim.cell(c, r))
            for c in range(sim.width)
        )
        bump = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

        w = self.weights
        return w[0] * agg + w[1] * holes + w[2] * bump + w[3] * lines

    def __repr__(self) -> str:
        return f"MockScorer(weights={self.weights})"


class MockNetwork:
    """A stand-in for the phase-2 neural network.

    Has a `layer_sizes` attribute (list of ints) the visualizer reads to draw
    the neuron layout, and a `score` method that delegates to a `MockScorer`.
    Phase 2 will replace this with a real model that implements the same
    interface; the visualizer will work unchanged.
    """

    def __init__(self, layer_sizes: list[int] | None = None) -> None:
        # A small default layout: 4 input features -> 8 -> 8 -> 1 output.
        self.layer_sizes = layer_sizes or [4, 8, 8, 1]
        self._scorer = MockScorer()

    def score(self, board: "ds.Board", candidate: "ds.LockCandidate") -> float:
        return self._scorer.score(board, candidate)

    def __repr__(self) -> str:
        return f"MockNetwork(layers={self.layer_sizes})"


def extract_features(board: "ds.Board", candidate: "ds.LockCandidate") -> list[float]:
    """Extract the 4 heuristic features for a candidate. Used by the
    visualizer to show the NN's input on the training panel."""
    sim = ds.Board()
    for r in range(board.height):
        for c in range(board.width):
            if board.cell(c, r):
                sim.set_cell(c, r)
    sim.lock(candidate.piece)
    lines = sim.clear_lines()
    heights = [sim.column_height(c) for c in range(sim.width)]
    agg = sum(heights)
    holes = sum(
        heights[c] - sum(1 for r in range(heights[c]) if sim.cell(c, r))
        for c in range(sim.width)
    )
    bump = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
    return [float(agg), float(holes), float(bump), float(lines)]
