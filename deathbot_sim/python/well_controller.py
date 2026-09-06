"""Explicit designated-well controller (§15.1.3).

Small finite-state controller — a diagnostic and future teacher, not the
final policy:

  BUILD → CASH-OUT → BUILD, with RECOVER available throughout.

- Fixed well column (default 9). Never switches target wells.
- BUILD: maximize dense Phi (well_potential) with survival guardrails
  (holes, spawn danger, stack height). Keeps the channel open.
- CASH-OUT: if tetris_ready and current piece can fill the well for 4
  lines, take it (largest lines_cleared filling the well column).
- RECOVER: if top > RECOVER_HEIGHT or holes_outside > 0 or spawn danger,
  play the survival move (max heuristic reward / min danger) until safe.
- Uses ARS/20G reachability implicitly: candidates come from
  lock_candidates_full (all reachable locks), never from heights alone.
  Sides are not assumed interchangeable (well col is a parameter).

The controller is intentionally NN-free: if IT produces sustained
Tetrises, representation is proven not to be the prerequisite (§15).
"""
from __future__ import annotations

import numpy as np

import deathbot_sim as ds
from well_potential import (dense_well_features, well_potential, WELL_COL,
                            DEFAULT_WEIGHTS)

RECOVER_HEIGHT = 14
SPAWN_COLS = (3, 4, 5, 6)


def _rows_array(flat_rows, n):
    return np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)


def controller_choose(board: "ds.Board", current, next_piece=None,
                      well_col: int = WELL_COL,
                      weights: dict | None = None,
                      recover_height: int = RECOVER_HEIGHT) -> int:
    """Pick a candidate index (search_locks order) with the explicit policy."""
    bulk = board.lock_candidates_full(current)
    (states, xs, ys, lockouts, flat_rows, lines) = bulk
    n = len(states)
    if n == 0:
        raise ValueError("no candidates")
    viable = [i for i, lo in enumerate(lockouts) if not lo]
    if not viable:
        return 0
    rows = _rows_array(flat_rows, n)
    feats = [dense_well_features(rows[i], well_col) for i in viable]
    lines_v = [int(lines[i]) for i in viable]

    # RECOVER: high stack / holes on the tallest option → survival move.
    # Proxy for 20G survival: lowest resulting top first (stack must come
    # DOWN), then holes, then lines. Tetris reachable → take it regardless.
    tops = [f["top"] for f in feats]
    if (max(tops) > recover_height
            or any(f["holes_outside"] > 0 for f in feats
                   if f["top"] == max(tops))):
        for j in range(len(viable)):
            if lines_v[j] >= 4:
                return viable[j]
        best = min(range(len(viable)),
                   key=lambda j: (feats[j]["top"],
                                  feats[j]["holes_outside"],
                                  -lines_v[j], -j))
        return viable[best]

    # CASH-OUT: Tetris-ready + current can clear 4 → take the Tetris.
    cash = [j for j in range(len(viable)) if lines_v[j] >= 4]
    if cash:
        # Prefer the one that keeps the stack lowest after clearing.
        best = min(cash, key=lambda j: (feats[j]["top"], -j))
        return viable[best]

    # BUILD: max Phi, tie-break toward lines and lower tops.
    # NOTE: pure max-Phi drifts upward (Phi rewards depth/fill without a
    # survival anchor) and dies ~100 pieces in 20G. Gate: only consider
    # candidates that keep the stack at/below today's top; among those,
    # prefer lower tops. If every candidate climbs, take the lowest top.
    cur_top = int(max(board.column_height(c) for c in range(10)))
    safe = [j for j in range(len(viable)) if feats[j]["top"] <= cur_top]
    if not safe:
        best = min(range(len(viable)),
                   key=lambda j: (feats[j]["top"], -lines_v[j], -j))
        return viable[best]
    phis = {j: well_potential(rows[viable[j]], well_col, weights)
            for j in safe}
    best = max(safe,
               key=lambda j: (phis[j], lines_v[j], -feats[j]["top"], -j))
    return viable[best]


def play_game_controller(rng, current_fn=None, *, max_pieces: int = 300,
                         well_col: int = WELL_COL,
                         weights: dict | None = None):
    """Play a game with the explicit controller. Returns (pieces, lines,
    clears, reason). Uses the TAP PieceQueue for the piece stream."""
    import random
    from piece_history import PieceQueue
    rng = rng or random.Random(0)
    board = ds.Board()
    pq = PieceQueue(rng)
    pieces = 0
    lines_total = 0
    clears: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    while pieces < max_pieces:
        bulk = board.lock_candidates_full(pq.current)
        n = len(bulk[0])
        if n == 0:
            return pieces, lines_total, clears, "no_candidates"
        if all(bulk[3]):
            return pieces, lines_total, clears, "lockout"
        pick = controller_choose(board, pq.current, pq.next_piece,
                                 well_col=well_col, weights=weights)
        nl = int(bulk[5][pick])
        lines_total += nl
        clears[nl] = clears.get(nl, 0) + 1
        rows = _rows_array(bulk[4], n)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    return pieces, lines_total, clears, "max_pieces"
