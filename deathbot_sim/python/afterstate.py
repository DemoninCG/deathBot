"""Shared construction of post-lock, post-line-clear neural afterstates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import deathbot_sim as ds
from model import board_to_tensor


@dataclass(frozen=True)
class Afterstate:
    """The actual board state reached after a candidate is locked and cleared."""

    board: "ds.Board"
    tensor: torch.Tensor
    lines_cleared: int


def make_afterstate(board: "ds.Board", candidate: "ds.LockCandidate") -> Afterstate:
    """Clone ``board``, lock ``candidate``, clear lines, and encode the result."""
    sim = board.clone_board()
    sim.lock(candidate.piece)
    lines = sim.clear_lines()
    return Afterstate(sim, board_to_tensor(sim), lines)
