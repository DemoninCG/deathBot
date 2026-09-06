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


def well_setup_score(rows21, lines_cleared: int = 0) -> float:
    """Tetris-well setup quality of a post-clear afterstate, in [0, 1].

    Rewards exactly one deep (depth>=6), clean well: a single column at
    least 6 below its HIGHER neighbor-wall, with no holes anywhere and no
    second well deeper than 3. Fresh clears don't disqualify (the well that
    just paid off scores via the afterstate that remains), but a board that
    just cleared 4 with no well standing scores 0 — the bonus is for the
    SETUP, not the payoff. Flat boards score ~0 (no well yet); holey or
    multi-well chaos scores 0.
    """
    import numpy as _np
    rows = _np.asarray(rows21, dtype=_np.uint32).reshape(-1)
    occ = _np.zeros((21, 10), dtype=bool)
    for r in range(21):
        bits = int(rows[r])
        for c in range(10):
            if bits & (1 << c):
                occ[r, c] = True
    heights = _np.zeros(10, dtype=int)
    for c in range(10):
        col = _np.where(occ[:, c])[0]
        heights[c] = int(col.max()) + 1 if len(col) else 0
    # Any hole anywhere disqualifies (a Tetris well must be clean).
    max_h = int(heights.max()) if heights.max() > 0 else 0
    for row in range(max_h):
        row_occ = occ[row]
        left = _np.cumprod(~row_occ)
        right = _np.cumprod(~row_occ[::-1])[::-1]
        for col in range(10):
            if row >= heights[col] or row_occ[col]:
                continue
            if not (left[col] or right[col]):
                return 0.0
    # Well depths per column (higher-neighbor wall definition).
    depths = []
    for col in range(10):
        if col == 0:
            d = heights[1] - heights[0]
        elif col == 9:
            d = heights[8] - heights[9]
        else:
            d = max(heights[col - 1], heights[col + 1]) - heights[col]
        depths.append(int(d))
    order = sorted(range(10), key=lambda c: -depths[c])
    best, second = depths[order[0]], depths[order[1]]
    if best < 6 or second > 3:
        return 0.0
    if best >= 9:
        depth_score = 1.0
    elif best == 8:
        depth_score = 0.8
    elif best == 7:
        depth_score = 0.6
    else:  # best == 6
        depth_score = 0.4
    # Tall stacks make any well dangerous; scale down above height 14.
    top = int(heights.max())
    height_factor = 1.0 if top <= 14 else max(0.0, 1.0 - (top - 14) / 6.0)
    _ = lines_cleared  # payoff counted by line_bonus; setup is the state
    return depth_score * height_factor


class TwoPlyScorer:
    """Score candidates with 1-ply value + gamma * best next-ply value.

    Optional style bias: ``line_bonus={1:.., 2:.., 3:.., 4:..}`` adds a
    per-clear bonus (z-units) to SURVIVING branches only — dying/lockout
    branches keep their penalties, so survival ordering is preserved and
    the bonus only re-ranks among safe moves. Guideline-style weights
    (e.g. {1:1, 2:3, 3:5, 4:8} scaled into z-units) reward Tetrises most.

    Optional well shaping: ``well_bonus`` adds ``well_weight * well_score``
    (z-units) for the RESULTING afterstate on surviving branches only, where
    ``well_score`` in [0, 1] measures a deep, clean, single Tetris well
    (see :func:`well_setup_score`). Unlike the clear bonus (which rewards
    the payoff move), this rewards the multi-move SETUP state — holding an
    open well while the stack stays low and hole-free. ``well_weight`` is
    scaled like line_bonus (within-state v1 spread ~10.5).
    """

    def __init__(self, net, *, beam: int = 5, gamma_ply: float = 0.999,
                 device: str = "cpu",
                 line_bonus: dict | None = None,
                 well_bonus: float = 0.0,
                 well_weight: float = 0.0) -> None:
        self.net = net
        self.beam = beam
        self.gamma_ply = gamma_ply
        self.device = device
        self.line_bonus = dict(line_bonus) if line_bonus else {}
        # well_bonus is the legacy scalar name; well_weight is preferred.
        self.well_weight = float(well_weight or well_bonus or 0.0)

    def _placements_from_bulk(self, bulk, kind):
        """Ply-1 scoring from an already-fetched `lock_candidates_full` tuple."""
        (states, xs, ys, lockouts, flat_rows, lines) = bulk
        cur_idx = piece_to_index(kind)
        if not states:
            return [], [], [], False
        n = len(states)
        rows_per = np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)
        viable = [i for i in range(n) if not lockouts[i]]
        if not viable:
            return [], [], [], True
        boards = flat_rows_to_tensor(rows_per[viable], len(viable)).to(self.device)
        cur = torch.full((len(viable),), cur_idx, dtype=torch.long,
                         device=self.device)
        fast = getattr(self.net, "score_batch_fast", None)
        with torch.inference_mode():
            if fast is not None:
                v = fast(boards, cur).flatten().tolist()
            else:
                v = self.net(boards, cur, cur).flatten().tolist()
        packed = [rows_per[i] for i in viable]
        lines_list = [int(lines[i]) for i in viable]
        if self.well_weight:
            well_list = [well_setup_score(rows_per[i], int(lines[i]))
                         for i in viable]
            return packed, v, lines_list, True, well_list
        return packed, v, lines_list, True

    def _bulk(self, board, kind):
        """Scoring fast path when available (no input-path alloc)."""
        fast = getattr(board, "lock_candidates_fast", None)
        if fast is not None:
            return fast(kind)
        return board.lock_candidates_full(kind)

    def _placements(self, board: "ds.Board", kind):
        """All legal (non-lockout) placements of `kind` on `board`.

        Returns (successor_boards, values, lines_list, any_candidate_existed).
        Lockout placements are excluded from scoring (they are losses).
        `lines_list` parallels the viable entries (bulk lines_cleared).

        Compatibility wrapper: reconstructs Board objects (used by viz).
        Hot paths use `_placements_from_bulk` (packed rows, no Boards).
        """
        packed, v, lines_list, existed = self._placements_from_bulk(
            self._bulk(board, kind), kind)
        succs = [ds.Board.from_row_bits([int(t) for t in rows])
                 for rows in packed]
        return succs, v, lines_list, existed

    def choose(self, board: "ds.Board", current, next_piece) -> int:
        """Return the candidate index (into search_locks order) to play."""
        bulk = self._bulk(board, current)
        (states, xs, ys, lockouts, _rows, _lines) = bulk
        if not states:
            raise ValueError("no candidates")
        viable = [i for i, lo in enumerate(lockouts) if not lo]
        if not viable:
            return 0  # every placement loses; all equivalent
        res = self._placements_from_bulk(bulk, current)
        packed1, vals1 = res[0], res[1]
        lines1 = res[2] if len(res) > 3 else [0] * len(vals1)
        wells1 = res[4] if len(res) > 4 else [0.0] * len(vals1)
        if not vals1:
            return viable[0]
        if self.beam <= 0:
            best_local = max(range(len(vals1)), key=lambda j: (vals1[j], -j))
            return viable[best_local]
        order = sorted(range(len(vals1)), key=lambda j: -vals1[j])
        beam = order[: self.beam]
        # Item 1: batch ALL ply-2 forwards into one net call. Per branch we
        # need its own successor boards; gather packed rows + decode once.
        nxt_idx = piece_to_index(next_piece)
        branch_rows: list = []       # flat list of rows arrays per cand
        branch_counts: list[int] = []  # per beam-branch candidate count
        branch_viable_flags: list[bool] = []  # False => dies next turn
        succ_cache: dict[int, "ds.Board"] = {}
        _mk_bulk = ds.Board.from_row_bits_bulk if hasattr(
            ds.Board, "from_row_bits_bulk") else None
        # Item 4: build successor Boards for the beam in ONE FFI call.
        need = [j for j in beam if j not in succ_cache]
        if need and _mk_bulk is not None:
            flat = np.concatenate([packed1[j] for j in need]).astype(
                np.uint32).tolist()
            made = _mk_bulk([int(v) for v in flat])
            for j, b in zip(need, made):
                succ_cache[j] = b
        for j in beam:
            if j not in succ_cache:
                succ_cache[j] = ds.Board.from_row_bits(
                    [int(t) for t in packed1[j]])
            succ = succ_cache[j]
            (s2, _x2, _y2, lo2, flat2, _ln2) = self._bulk(succ, next_piece)
            n2 = len(s2)
            if n2 == 0:
                branch_viable_flags.append(False)
                branch_counts.append(0)
                continue
            rows2 = np.asarray(flat2, dtype=np.uint32).reshape(n2, 21)
            keep = [i for i in range(n2) if not lo2[i]]
            if not keep:
                branch_viable_flags.append(True)  # existed but all lockout
                branch_counts.append(0)
                continue
            branch_viable_flags.append(True)
            branch_counts.append(len(keep))
            for i in keep:
                branch_rows.append(rows2[i])
        final = {j: vals1[j] for j in range(len(vals1))}
        if branch_rows:
            big = flat_rows_to_tensor(np.stack(branch_rows), len(branch_rows)
                                      ).to(self.device)
            nxt_t = torch.full((len(branch_rows),), nxt_idx, dtype=torch.long,
                               device=self.device)
            fast = getattr(self.net, "score_batch_fast", None)
            with torch.inference_mode():
                if fast is not None:
                    all_v = fast(big, nxt_t).flatten().tolist()
                else:
                    all_v = self.net(big, nxt_t, nxt_t).flatten().tolist()
            # Reduce per-branch max over the concatenated values.
            off = 0
            for bpos, j in enumerate(beam):
                if branch_counts[bpos] == 0:
                    final[j] = DEAD_PENALTY
                else:
                    seg = all_v[off:off + branch_counts[bpos]]
                    off += branch_counts[bpos]
                    bonus = float(self.line_bonus.get(int(lines1[j]), 0.0))
                    if self.well_weight:
                        bonus += self.well_weight * float(wells1[j])
                    final[j] = vals1[j] + bonus + self.gamma_ply * max(seg)
        else:
            for j in beam:
                final[j] = DEAD_PENALTY
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
        # NOTE: the fast bulk path returns 5-tuples (no lines list); the
        # full path returns 6-tuples with lines_cleared at index 5. Always
        # use the FULL path here: this function's contract reports lines,
        # and the per-decision cost difference is negligible next to choose().
        res = board.lock_candidates_full(pq.current)
        n_cands = len(res[0])
        if n_cands == 0:
            return pieces, lines_total, "no_candidates"
        if all(res[3]):
            return pieces, lines_total, "lockout"
        pick = scorer.choose(board, pq.current, pq.next_piece)
        if epsilon > 0 and explore_rng.random() < epsilon:
            viable = [i for i, lo in enumerate(res[3]) if not lo]
            pick = explore_rng.choice(viable)
        # lock_candidates_full already returns post-lock POST-CLEAR rows
        # (bulk.rs mirrors Board::lock + Board::clear_lines), so lines must
        # be read from its lines_cleared output (index 5) — calling
        # clear_lines() again on the reconstructed board double-clears and
        # always reports 0.
        lines_total += int(res[5][pick])
        rows = np.asarray(
            res[4], dtype=np.uint32).reshape(n_cands, 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    return pieces, lines_total, "max_pieces"


def play_game_2ply_style(net, rng: random.Random, *, beam: int = 100,
                         gamma_ply: float = 0.999, epsilon: float = 0.0,
                         max_pieces: int = 300,
                         line_bonus: dict | None = None,
                         well_weight: float = 0.0,
                         well_stats: dict | None = None,
                         explore_rng: random.Random | None = None):
    """Style-metric game: same rules as play_game_2ply, but also returns a
    per-clear breakdown {0,1,2,3,4: counts} for Tetris-rate analysis.

    Returns (pieces_placed, lines_cleared, clears_dict, reason).
    With survival solved, callers use a LOW cap (e.g. 300) so every game
    caps and style metrics (lines/piece, tetris share) carry the signal.
    `line_bonus` (e.g. {1:.., 4:..} in z-units) rewards clears on surviving
    branches only — survival ordering is preserved. `well_weight` adds the
    well-setup bonus (see TwoPlyScorer); pass `well_stats` dict to also
    accumulate mean well_setup_score per game (diagnostic for §14).
    """
    explore_rng = explore_rng or rng
    board = ds.Board()
    pq = PieceQueue(rng)
    scorer = TwoPlyScorer(net, beam=beam, gamma_ply=gamma_ply,
                          line_bonus=line_bonus, well_weight=well_weight)
    pieces = 0
    lines_total = 0
    clears: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    while pieces < max_pieces:
        # Always the FULL bulk path (6-tuple with lines_cleared); the fast
        # path has no lines list (perf-pass regression, fixed here).
        bulk = board.lock_candidates_full(pq.current)
        n_cands = len(bulk[0])
        if n_cands == 0:
            return pieces, lines_total, clears, "no_candidates"
        if all(bulk[3]):
            return pieces, lines_total, clears, "lockout"
        pick = scorer.choose(board, pq.current, pq.next_piece)
        if epsilon > 0 and explore_rng.random() < epsilon:
            viable = [i for i, lo in enumerate(bulk[3]) if not lo]
            pick = explore_rng.choice(viable)
        n = int(bulk[5][pick])
        lines_total += n
        clears[n] = clears.get(n, 0) + 1
        rows = np.asarray(
            bulk[4], dtype=np.uint32).reshape(n_cands, 21)[pick]
        if well_stats is not None:
            well_stats["sum"] = well_stats.get("sum", 0.0) + well_setup_score(
                rows, n)
            well_stats["n"] = well_stats.get("n", 0) + 1
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
