"""Shared construction of post-lock, post-line-clear neural afterstates.

2026-09 (phase A2/D5 of the training plan):

* ``make_afterstate`` keeps its historical semantics (clone, lock, clear,
  encode) for supervised dataset generation and the legacy scorer path.
* Packed-row <-> tensor converters let the rollout engine talk to the Rust
  ``lock_candidates_full`` binding (search + lock + clear for EVERY candidate
  in ONE FFI crossing) and let replay buffers store 84 bytes per board
  instead of ~800-byte float tensors plus Python tensor overhead.
* Lockout (``Board.locks_out``) is enforced by rollout code: a candidate
  that locks a cell above the field is a losing placement, not a free
  partial piece. Historical code silently dropped those cells and kept
  playing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import deathbot_sim as ds
from model import board_to_tensor


# ---- Packed row-bitmask helpers (A2/D2/D5) ---------------------------------

BOARD_ROWS = 21  # 20 visible + 1 invisible buffer row
COL_MASKS = np.array([1 << c for c in range(10)], dtype=np.uint32)


def rows_to_tensor(rows: np.ndarray) -> torch.Tensor:
    """Decode packed row bitmasks [B, 21] uint32 -> board tensors [B, 1, 20, 10].

    Bitmask rows are bottom-first (matching ``Board.row_bits()``); the tensor
    is top-first, matching ``model.board_to_tensor``. The invisible row is
    excluded (visible rows only), exactly like the historical encoding.
    """
    rows = np.asarray(rows, dtype=np.uint32)
    single = rows.ndim == 1
    if single:
        rows = rows[None, :]
    visible = rows[:, :20]  # [B, 20] bottom-first
    col_masks = np.array([1 << c for c in range(10)], dtype=np.uint32)
    grid = ((visible[:, :, None] & col_masks[None, None, :]) != 0).astype(np.float32)
    grid = np.ascontiguousarray(grid[:, ::-1, :])  # top-first [B, 20, 10]
    t = torch.from_numpy(grid).unsqueeze(1)  # [B, 1, 20, 10]
    if single:
        t = t.squeeze(0).unsqueeze(0)
    return t


def flat_rows_to_tensor(flat: np.ndarray, n_candidates: int) -> torch.Tensor:
    """Decode flat packed rows [n*21] uint32 into [n, 1, 20, 10] tensors."""
    rows = np.asarray(flat, dtype=np.uint32).reshape(n_candidates, 21)
    return rows_to_tensor(rows)


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
