"""NeuralScorer: adapter wrapping TAPNet to implement the Scorer protocol.

This lets the phase-1.4 visualizer and the future training loop treat the
neural network interchangeably with MockScorer — both just need a
`score(board, candidate) -> float` method.

The scorer simulates locking the candidate onto the board, converts the
resulting board to a tensor, and asks the network for a score. The "next
piece" input is configurable (set at construction or per-call); during live
play it'll be the actual upcoming piece from the piece queue.
"""

from __future__ import annotations

from typing import Optional

import torch

import deathbot_sim as ds
from model import TAPNet, board_to_tensor, piece_to_index
from scorer import Scorer


class NeuralScorer:
    """Wraps a TAPNet to implement the Scorer protocol.

    Args:
        net: the TAPNet to use. If None, a fresh untrained network is created
            (useful for the visualizer before training).
        next_piece: the PieceType to feed as the "next piece" input, or None
            to default to I (index 0). Set per-call via `score_with_next`.
        device: 'cpu' or 'cuda' — defaults to CPU (the model is lightweight).
    """

    def __init__(
        self,
        net: Optional[TAPNet] = None,
        next_piece: Optional["ds.PieceType"] = None,
        device: str = "cpu",
    ) -> None:
        self.net = net or TAPNet()
        self.net.to(device)
        self.net.eval()
        self.device = device
        # Default next-piece index (I = 0). Override per-call if needed.
        self.default_next_idx = (
            piece_to_index(next_piece) if next_piece is not None else 0
        )
        # Expose layer_sizes for the visualizer.
        self.layer_sizes = self.net.layer_sizes

    def score(self, board: "ds.Board", candidate: "ds.LockCandidate") -> float:
        """Score a candidate by simulating its lock and evaluating the board."""
        return self.score_with_next(board, candidate, self.default_next_idx)

    def score_with_next(
        self,
        board: "ds.Board",
        candidate: "ds.LockCandidate",
        next_piece_idx: int,
    ) -> float:
        """Score a candidate with a specific next-piece index."""
        # Simulate the lock on a copy of the board.
        sim = ds.Board()
        for r in range(board.height):
            for c in range(board.width):
                if board.cell(c, r):
                    sim.set_cell(c, r)
        sim.lock(candidate.piece)
        # NOTE: we deliberately do NOT clear lines here — the network sees the
        # raw locked state. The training target (phase 2.2) will encode whether
        # line clears are good. Adjust if the heuristic wants pre-clear boards.
        tensor = board_to_tensor(sim).to(self.device)
        return self.net.score_one(tensor, next_piece_idx)

    def score_batch(
        self,
        board: "ds.Board",
        candidates: list["ds.LockCandidate"],
        next_piece_idx: Optional[int] = None,
    ) -> list[float]:
        """Score many candidates efficiently as a single batch.

        Builds a [B, 1, 20, 10] tensor of simulated boards and a [B] index
        tensor, runs one forward pass, and returns a list of Python floats.
        """
        if not candidates:
            return []
        idx = next_piece_idx if next_piece_idx is not None else self.default_next_idx
        boards = []
        for c in candidates:
            sim = ds.Board()
            for r in range(board.height):
                for col in range(board.width):
                    if board.cell(col, r):
                        sim.set_cell(col, r)
            sim.lock(c.piece)
            boards.append(board_to_tensor(sim))
        batch = torch.stack(boards, dim=0).to(self.device)  # [B, 1, 20, 10]
        idx_t = torch.tensor([idx] * len(candidates), dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.net(batch, idx_t)  # [B, 1]
        return [float(x) for x in out.squeeze(1).tolist()]

    def __repr__(self) -> str:
        return f"NeuralScorer(net={self.net!r}, next_idx={self.default_next_idx})"
