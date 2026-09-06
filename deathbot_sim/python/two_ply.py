"""Multi-ply candidate scoring (Experiment 1 of the standard-levers campaign).

The deployed bot is 1-ply greedy: score each candidate's afterstate, take the
max. This module adds 2-ply lookahead:

    P2(s, a) = value(afterstate(s, a)) + gamma_ply * max_b value(afterstate(s', b))

where s' is the afterstate board and b ranges over the NEXT piece's
placements (the preview piece the network already sees).

IMPORTANT terminal semantics (v2 — v1 had a fatal bug):
  * If the next piece has NO legal placement after choosing `a`, branch `a`
    DIES next turn. v1 left dying branches at their 1-ply value while
    surviving branches received gamma * max(next values) — and since the
    net's outputs are mostly NEGATIVE (anchor mean ~ -85), the "bonus" was
    negative, so v1 systematically PREFERRED moves that killed the game next
    turn (screen: 32-50 pieces vs snapshot9's ~750).
    v2 assigns dying branches -1e6 (always lose unless every branch dies).
  * Lockout placements are masked in BOTH plies (A2 semantics: a lockout is
    a loss, not a free partial piece).
  * beam=0 reproduces pure 1-ply greedy over legal placements (baseline).

Cost: (1 + beam) forward passes per decision.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deathbot_sim as ds
from model import piece_to_index
from afterstate import flat_rows_to_tensor
from piece_history import PieceQueue

DEAD_PENALTY = -1e6


class TwoPlyScorer:
    """Score candidates with 1-ply value + gamma * best next-ply value.

    Optional style bias: ``line_bonus={1:.., 2:.., 3:.., 4:..}`` adds a
    per-clear bonus (z-units) to SURVIVING branches only — dying/lockout
    branches keep their penalties, so survival ordering is preserved and
    the bonus only re-ranks among safe moves. Guideline-style weights
    (e.g. {1:1, 2:3, 3:5, 4:8} scaled into z-units) reward Tetrises most.
    """

    def __init__(self, net, *, beam: int = 5, gamma_ply: float = 0.999,
                 device: str = "cpu",
                 line_bonus: dict | None = None) -> None:
        self.net = net
        self.beam = beam
        self.gamma_ply = gamma_ply
        self.device = device
        self.line_bonus = dict(line_bonus) if line_bonus else {}

    def _placements(self, board: "ds.Board", kind):
        """All legal (non-lockout) placements of `kind` on `board`.

        Returns (successor_boards, values, lines_list, any_candidate_existed).
        Lockout placements are excluded from scoring (they are losses).
        `lines_list` parallels the viable entries (bulk lines_cleared)."""
        cur_idx = piece_to_index(kind)
        cands = board.search_locks(kind)
        if not cands:
            return [], [], [], False
        (states, xs, ys, lockouts, flat_rows, lines) = board.lock_candidates_full(kind)
        rows_per = np.asarray(flat_rows, dtype=np.uint32).reshape(len(states), 21)
        viable = [i for i in range(len(states)) if not lockouts[i]]
        if not viable:
            return [], [], [], True
        boards = flat_rows_to_tensor(
            np.concatenate([rows_per[i] for i in viable]), len(viable)).to(self.device)
        cur = torch.full((len(viable),), cur_idx, dtype=torch.long,
                         device=self.device)
        # The piece after `next` is unknown; reuse `next` as the 2nd-ply
        # preview (the same approximation the 1-ply inputs make; consistent
        # units).
        nxt = cur
        with torch.no_grad():
            v = self.net(boards, cur, nxt).flatten().tolist()
        succs = [ds.Board.from_row_bits([int(t) for t in rows_per[i]])
                 for i in viable]
        lines_list = [int(lines[i]) for i in viable]
        return succs, v, lines_list, True

    def choose(self, board: "ds.Board", current, next_piece) -> int:
        """Return the candidate index (into search_locks order) to play."""
        cands = board.search_locks(current)
        if not cands:
            raise ValueError("no candidates")
        viable = [i for i, c in enumerate(cands) if not board.locks_out(c.piece)]
        if not viable:
            return 0  # every placement loses; all equivalent
        res = self._placements(board, current)
        succs1, vals1 = res[0], res[1]
        lines1 = res[2] if len(res) > 3 else [0] * len(vals1)
        if not vals1:
            return viable[0]
        if self.beam <= 0:
            best_local = max(range(len(vals1)), key=lambda j: (vals1[j], -j))
            return viable[best_local]
        order = sorted(range(len(vals1)), key=lambda j: -vals1[j])
        beam = order[: self.beam]
        final = {j: vals1[j] for j in range(len(vals1))}
        for j in beam:
            placed = self._placements(succs1[j], next_piece)
            succs2, vals2 = placed[0], placed[1]
            if not vals2:
                # This branch dies next turn (no legal placement for the
                # next piece): worst continuation, never preferred while any
                # surviving branch exists.
                final[j] = DEAD_PENALTY
            else:
                bonus = float(self.line_bonus.get(int(lines1[j]), 0.0))
                final[j] = vals1[j] + bonus + self.gamma_ply * max(vals2)
        best_local = max(range(len(final)), key=lambda j: (final[j], -j))
        return viable[best_local]


def play_game_2ply(net, rng: random.Random, *, beam: int = 5,
                   gamma_ply: float = 0.999, epsilon: float = 0.0,
                   max_pieces: int = 1500, explore_rng: random.Random | None = None):
    """Greedy (or epsilon) game with 2-ply lookahead. Returns
    (pieces_placed, lines_cleared, reason). Evaluation mode: no trajectory."""
    explore_rng = explore_rng or rng
    board = ds.Board()
    pq = PieceQueue(rng)
    scorer = TwoPlyScorer(net, beam=beam, gamma_ply=gamma_ply)
    pieces = 0
    lines_total = 0
    while pieces < max_pieces:
        cands = board.search_locks(pq.current)
        if not cands:
            return pieces, lines_total, "no_candidates"
        if all(board.locks_out(c.piece) for c in cands):
            return pieces, lines_total, "lockout"
        pick = scorer.choose(board, pq.current, pq.next_piece)
        if epsilon > 0 and explore_rng.random() < epsilon:
            viable = [i for i, c in enumerate(cands)
                      if not board.locks_out(c.piece)]
            pick = explore_rng.choice(viable)
        res = board.lock_candidates_full(pq.current)
        # lock_candidates_full already returns post-lock POST-CLEAR rows
        # (bulk.rs mirrors Board::lock + Board::clear_lines), so lines must
        # be read from its lines_cleared output (index 5) — calling
        # clear_lines() again on the reconstructed board double-clears and
        # always reports 0.
        lines_total += int(res[5][pick])
        rows = np.asarray(
            res[4], dtype=np.uint32).reshape(len(cands), 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    return pieces, lines_total, "max_pieces"


def play_game_2ply_style(net, rng: random.Random, *, beam: int = 100,
                         gamma_ply: float = 0.999, epsilon: float = 0.0,
                         max_pieces: int = 300,
                         line_bonus: dict | None = None,
                         explore_rng: random.Random | None = None):
    """Style-metric game: same rules as play_game_2ply, but also returns a
    per-clear breakdown {0,1,2,3,4: counts} for Tetris-rate analysis.

    Returns (pieces_placed, lines_cleared, clears_dict, reason).
    With survival solved, callers use a LOW cap (e.g. 300) so every game
    caps and style metrics (lines/piece, tetris share) carry the signal.
    `line_bonus` (e.g. {1:.., 4:..} in z-units) rewards clears on surviving
    branches only — survival ordering is preserved.
    """
    explore_rng = explore_rng or rng
    board = ds.Board()
    pq = PieceQueue(rng)
    scorer = TwoPlyScorer(net, beam=beam, gamma_ply=gamma_ply,
                          line_bonus=line_bonus)
    pieces = 0
    lines_total = 0
    clears: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    while pieces < max_pieces:
        cands = board.search_locks(pq.current)
        if not cands:
            return pieces, lines_total, clears, "no_candidates"
        if all(board.locks_out(c.piece) for c in cands):
            return pieces, lines_total, clears, "lockout"
        pick = scorer.choose(board, pq.current, pq.next_piece)
        if epsilon > 0 and explore_rng.random() < epsilon:
            viable = [i for i, c in enumerate(cands)
                      if not board.locks_out(c.piece)]
            pick = explore_rng.choice(viable)
        res = board.lock_candidates_full(pq.current)
        n = int(res[5][pick])
        lines_total += n
        clears[n] = clears.get(n, 0) + 1
        rows = np.asarray(
            res[4], dtype=np.uint32).reshape(len(cands), 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    return pieces, lines_total, clears, "max_pieces"


def style_summary(pieces: int, lines: int, clears: dict) -> dict:
    """Aggregate style metrics from one game's clears breakdown."""
    singles = int(clears.get(1, 0))
    doubles = int(clears.get(2, 0))
    triples = int(clears.get(3, 0))
    tetrises = int(clears.get(4, 0))
    scoring = singles + doubles + triples + tetrises
    return {
        "pieces": pieces,
        "lines": lines,
        "lines_per_piece": (lines / pieces) if pieces else 0.0,
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "tetrises": tetrises,
        "scoring_moves": scoring,
        "tetris_share": (tetrises / scoring) if scoring else 0.0,
        "tetris_per_100": (100.0 * tetrises / pieces) if pieces else 0.0,
    }
