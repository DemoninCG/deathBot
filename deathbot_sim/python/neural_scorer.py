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

import numpy as np
import torch

import deathbot_sim as ds
from model import TAPNet, piece_to_index
from scorer import Scorer
from afterstate import make_afterstate, flat_rows_to_tensor


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
        # default_current_idx is set per-call (the scorer needs to know what
        # piece is being placed); defaults to I (0) if not specified.
        self.default_current_idx = 0
        # Expose layer_sizes for the visualizer.
        self.layer_sizes = self.net.layer_sizes

    def score(self, board: "ds.Board", candidate: "ds.LockCandidate") -> float:
        """Score a candidate by simulating its lock and evaluating the board.
        Uses the candidate's piece type as the current piece."""
        cur_idx = piece_to_index(candidate.piece.kind)
        return self.score_with_next(board, candidate, cur_idx, self.default_next_idx)

    def score_with_next(
        self,
        board: "ds.Board",
        candidate: "ds.LockCandidate",
        current_piece_idx: int,
        next_piece_idx: int,
    ) -> float:
        """Score a candidate with explicit current + next piece indices."""
        # Score the same post-lock, post-clear afterstate used for training.
        tensor = make_afterstate(board, candidate).tensor.to(self.device)
        return self.net.score_one(tensor, current_piece_idx, next_piece_idx)

    def score_batch(
        self,
        board: "ds.Board",
        candidates: list["ds.LockCandidate"],
        current_piece_idx: Optional[int] = None,
        next_piece_idx: Optional[int] = None,
    ) -> list[float]:
        """Score many candidates efficiently as a single batch.

        Args:
            board: the current board.
            candidates: lock candidates to score.
            current_piece_idx: the piece being placed. If None, inferred from
                each candidate's piece type (they should all be the same type).
            next_piece_idx: the upcoming piece index. If None, uses
                default_next_idx.

        Returns a list of Python floats.
        """
        if not candidates:
            return []
        nxt = next_piece_idx if next_piece_idx is not None else self.default_next_idx
        # If current_piece_idx is not given, infer from the first candidate
        # (all candidates for a given piece type share the same type).
        cur = current_piece_idx if current_piece_idx is not None else piece_to_index(candidates[0].piece.kind)

        # Fast path (phase D5): if the candidates came straight from
        # Board.search_locks, we can re-run the search + lock + clear for
        # every candidate in ONE Rust call and decode the packed rows with
        # numpy. Candidates constructed by other means fall back to the
        # per-candidate path (verified numerically identical: the packed
        # rows are produced by the same lock + clear semantics).
        # Item 5: prefer the path-free `lock_candidates_fast` (no input-path
        # alloc); fall back to lock_candidates_full.
        bulk = getattr(board, "lock_candidates_fast", None) or getattr(
            board, "lock_candidates_full", None)
        if bulk is not None and candidates and getattr(candidates[0], "piece", None) is not None:
            # Re-derive the piece kind from the first candidate so the Rust
            # search returns exactly the same candidate list (BFS order).
            try:
                kind = candidates[0].piece.kind
                (states, xs, ys, lockouts, flat_rows, lines) = bulk(kind)
                n = len(states)
                matches = (
                    n == len(candidates)
                    and all(
                        s == c.piece.state and x == c.piece.x and y == c.piece.y
                        for s, x, y, c in zip(states, xs, ys, candidates)
                    )
                )
            except Exception:
                matches = False
            if matches:
                boards = flat_rows_to_tensor(flat_rows, n).to(self.device)
                cur_t = torch.tensor([cur] * n, dtype=torch.long, device=self.device)
                nxt_t = torch.tensor([nxt] * n, dtype=torch.long, device=self.device)
                fast = getattr(self.net, "score_batch_fast", None)
                with torch.inference_mode():
                    if fast is not None and cur == nxt:
                        out = fast(boards, cur_t)  # [B, 1]
                    else:
                        out = self.net(boards, cur_t, nxt_t)  # [B, 1]
                return [float(x) for x in out.squeeze(1).tolist()]

        boards = []
        for c in candidates:
            boards.append(make_afterstate(board, c).tensor)
        batch = torch.stack(boards, dim=0).to(self.device)  # [B, 1, 20, 10]
        cur_t = torch.tensor([cur] * len(candidates), dtype=torch.long, device=self.device)
        nxt_t = torch.tensor([nxt] * len(candidates), dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.net(batch, cur_t, nxt_t)  # [B, 1]
        return [float(x) for x in out.squeeze(1).tolist()]

    def __repr__(self) -> str:
        return f"NeuralScorer(net={self.net!r}, next_idx={self.default_next_idx})"
