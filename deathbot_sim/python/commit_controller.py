"""Rung-1 commitment controller (§38.1): force + rescue at inference.

Design (review §4 + §38 discussion): reversible well commitment with
survival fallback — NOT another scalar bonus, NOT a retrain. The NN/search
still ranks placements; a constraint only removes plan-abandoning reversals:

  COMMITTED (preserve col-9 channel):
    candidate set = placements that preserve a USABLE well
    (well_open AND no new col-9 cover above the well floor AND
    outside-well holes not increased beyond --max-holes).
    Ranked by the existing survival scorer (full 2-ply + completion bonus,
    NO Phi — the constraint replaces the bonus). Falls back to the
    unconstrained set when the preserved set is empty or all-preserved
    branches die.
  CASH-OUT: a reachable Tetris (lines==4 with tetris_ready on the
    afterstate) is always taken when present, then the plan reassesses.
  RECOVER: drop the commitment when top >= --abort-top OR occ >=
    --abort-occ OR no preserving move exists; stay uncommitted until
    occ <= --reenter-occ (hysteresis, E4 pattern). While uncommitted the
    policy is the plain survival champion.
  Re-enter: a fresh commitment starts when uncommitted and a usable
  channel exists (well_open on some candidate) — the well is always col 9
  (fixed; no switching).

Why constrain-first, not shortlist-first (§38 discussion): the v1-top-k
shortlist discards setup moves before Phi ever sees them (E2). The filter
runs on the FULL viable set before expansion; expansion covers the
preserved set fully (bounded: <= n roots, same cost class as full 2-ply).

This module implements the POLICY (choose + game loop). The screen driver
is diag_commit_eval.py (rung-1 panel 20812+, cap 300, paired vs λ10/λ40).

Usage ( screening ):
    python diag_commit_eval.py --weights ../../model_weights_snapshot9.pt \\
        --games 30 --seed0 20812 --workers 4 \\
        --out ../_diag_out/commit_eval.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from well_potential import dense_well_features  # noqa: E402

WELL_COL = 9
COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}


def _col_bits(rows21, col: int = WELL_COL) -> int:
    h = 0
    for r in range(21):
        if (int(rows21[r]) >> col) & 1:
            h = r + 1
    return h


def preserves_well(pre_rows21, post_rows21, post_feats: dict,
                   max_new_holes: int = 0) -> bool:
    """True if the post-afterstate keeps a usable col-9 channel.

    Conditions (all must hold):
      - well_open (no hole/overhang in the channel),
      - col-9 height did not RISE above the pre height (no covering the
        channel; digging deeper is fine),
      - outside-well holes did not increase by more than max_new_holes.
    """
    if post_feats.get("well_open", 0.0) < 0.5:
        return False
    if _col_bits(post_rows21) > _col_bits(pre_rows21):
        return False
    pre_h = dense_well_features(pre_rows21)["holes_outside"]
    if post_feats.get("holes_outside", 0.0) - pre_h > max_new_holes:
        return False
    return True


class CommitScorer:
    """TwoPlyScorer wrapper with BUILD/PRESERVE + CASH-OUT + RECOVER.

    Params (<=4, E4 discipline):
      abort_top / abort_occ : drop commitment at/above (RECOVER).
      reenter_occ           : re-commit at/below (hysteresis).
      max_new_holes         : preserved-set hole tolerance (default 0).
    State: committed (bool), plus counters for the Pareto report
    (decisions_preserved, fallbacks, cashouts, recovers).
    """

    def __init__(self, base_scorer, *, abort_top: int = 14,
                 abort_occ: int = 90, reenter_occ: int = 50,
                 max_new_holes: int = 0, reenter_top: int | None = None,
                 cooldown: int = 0, lam: float = 0.0) -> None:
        self.base = base_scorer
        self.abort_top = int(abort_top)
        self.abort_occ = int(abort_occ)
        self.reenter_occ = int(reenter_occ)
        self.max_new_holes = int(max_new_holes)
        # Usability-gated re-entry (§41 fix): re-commit only if occ <=
        # reenter_occ AND top <= reenter_top AND >= cooldown pieces have
        # passed since the last RECOVER exit. None/0 = legacy behavior.
        # lam: dense-Phi scale applied on COMMITTED branches only
        # (controller-λ combo: constraint + gradient; 0.0 = pure commit).
        self.reenter_top = None if reenter_top is None else int(reenter_top)
        self.cooldown = int(cooldown)
        self.lam = float(lam)
        if self.lam:
            from well_potential import style_fn_factory
            self._phi = style_fn_factory(scale=self.lam)
        else:
            self._phi = None
        self.committed = True  # start building from the opening
        self._cooldown_left = 0
        self.stats = {"preserved_decisions": 0, "fallbacks": 0,
                      "cashouts": 0, "recovers": 0, "style_off": 0,
                      "decisions": 0, "reentries": 0,
                      "reentry_denied_top": 0, "reentry_denied_cool": 0}

    def _occ_top(self, pre_rows21):
        occ = sum(bin(int(v)).count("1") for v in pre_rows21)
        top = 0
        for r in range(20, -1, -1):
            if int(pre_rows21[r]) & 0b1111111111:
                top = r + 1
                break
        return occ, top

    def _usable_channel(self, rows, viable) -> bool:
        """A committable channel exists: some viable afterstate has
        well_open AND top <= reenter_top (when set). Checked on the live
        candidate set — re-entry is earned by geometry, not occupancy."""
        for i in viable:
            try:
                f = dense_well_features(rows[i])
            except Exception:
                continue
            if f.get("well_open", 0.0) < 0.5:
                continue
            if self.reenter_top is not None and f.get("top", 99) > self.reenter_top:
                continue
            return True
        return False

    def choose(self, board, current, next_piece) -> int:
        import deathbot_sim as ds
        bulk = board.lock_candidates_full(current)
        (states, xs, ys, lockouts, flat_rows, lines) = bulk
        n = len(states)
        if n == 0:
            raise ValueError("no candidates")
        viable = [i for i, lo in enumerate(lockouts) if not lo]
        if not viable:
            return 0
        pre_rows = np.asarray(board.row_bits(), dtype=np.uint32)
        occ, top = self._occ_top(pre_rows)
        rows = np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)
        self.stats["decisions"] += 1

        # RECOVER / usability-gated re-entry (§41: occ-only re-entry is
        # the defect — occ-50/60 reads "safe" on unserviceable stacks).
        if self.committed and (top >= self.abort_top or occ >= self.abort_occ):
            self.committed = False
            self.stats["recovers"] += 1
            self._cooldown_left = self.cooldown
        elif not self.committed:
            if self._cooldown_left > 0:
                self._cooldown_left -= 1
            if occ <= self.reenter_occ:
                if self.cooldown and self._cooldown_left > 0:
                    self.stats["reentry_denied_cool"] += 1
                elif (self.reenter_top is not None
                        and top > self.reenter_top):
                    self.stats["reentry_denied_top"] += 1
                else:
                    # Usability gate: a channel must actually exist.
                    usable = self._usable_channel(rows, viable)
                    if usable:
                        self.committed = True
                        self.stats["reentries"] += 1
                    else:
                        self.stats["reentry_denied_top"] += 1
        if not self.committed:
            self.stats["style_off"] += 1
            return self.base.choose(board, current, next_piece)

        # CASH-OUT: reachable Tetris among viable (tetris_ready afterstate).
        for i in viable:
            if int(lines[i]) == 4:
                try:
                    if dense_well_features(rows[i])["tetris_ready"] > 0.5:
                        self.stats["cashouts"] += 1
                        return i
                except Exception:
                    pass

        # PRESERVE: filter viable to channel-preserving moves (constrain
        # FIRST on the full viable set — never after a v1 shortlist).
        kept = []
        for i in viable:
            try:
                f = dense_well_features(rows[i])
            except Exception:
                continue
            if preserves_well(pre_rows, rows[i], f, self.max_new_holes):
                kept.append(i)
        if not kept:
            # No preserving move exists: survival fallback, keep commitment
            # (a single fallback placement rarely destroys the plan).
            self.stats["fallbacks"] += 1
            return self.base.choose(board, current, next_piece)

        # Rank the preserved set with the survival scorer. Full 2-ply over
        # the preserved roots: temporarily mask by choosing among kept via
        # the base scorer's full-expansion values. Cheapest correct route:
        # ask base for its pick; if it preserves, take it; else take the
        # best-v1 preserved root (v1 ranking inside the preserved set keeps
        # survival sense without re-expansion cost).
        base_pick = self.base.choose(board, current, next_piece)
        if base_pick in kept:
            self.stats["preserved_decisions"] += 1
            return base_pick
        # Base wants to abandon: rank preserved roots by 1-ply v1 (the
        # survival net's own valuation) plus optional Phi gradient
        # (controller-λ combo: constraint sets the admissible set, λ
        # biases within it — mechanically distinct from pre-constraint
        # scalar pressure, which also re-ranks abandoning moves).
        v1 = self._v1(board, current, kept, rows)
        if self._phi is not None:
            try:
                phi = self._phi([rows[j] for j in kept],
                                [0] * len(kept))
                v1 = [a + b for a, b in zip(v1, phi)]
            except Exception:
                pass
        best = max(range(len(kept)), key=lambda j: (v1[j], -j))
        self.stats["preserved_decisions"] += 1
        return kept[best]

    def _v1(self, board, kind, kept, rows):
        import torch
        from afterstate import flat_rows_to_tensor
        from model import piece_to_index
        net = self.base.net
        sub = rows[kept]
        boards = flat_rows_to_tensor(sub.reshape(-1), len(kept))
        cur = torch.full((len(kept),), piece_to_index(kind), dtype=torch.long)
        fast = getattr(net, "score_batch_fast", None)
        with torch.inference_mode():
            if fast is not None:
                return fast(boards.to(self.base.device), cur.to(
                    self.base.device)).flatten().tolist()
            nxt = cur
            return net(boards.to(self.base.device), cur.to(
                self.base.device), nxt.to(self.base.device)
                ).flatten().tolist()


def play_game_commit(net, rng, *, max_pieces: int = 300,
                     abort_top: int = 14, abort_occ: int = 90,
                     reenter_occ: int = 50, max_new_holes: int = 0,
                     reenter_top: int | None = None, cooldown: int = 0,
                     lam: float = 0.0, device: str = "cpu"):
    """Closed-loop game with the commitment controller. Returns
    (pieces, lines, clears, reason, stats)."""
    import random  # noqa: F401 (rng is passed in)
    from two_ply import TwoPlyScorer
    from piece_history import PieceQueue
    import deathbot_sim as ds
    base = TwoPlyScorer(net, beam=10 ** 9, device=device,
                        line_bonus=dict(COMPLETION_BONUS))
    ctrl = CommitScorer(base, abort_top=abort_top, abort_occ=abort_occ,
                        reenter_occ=reenter_occ, max_new_holes=max_new_holes,
                        reenter_top=reenter_top, cooldown=cooldown, lam=lam)
    board = ds.Board()
    pq = PieceQueue(rng)
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    lines_total = pieces = 0
    reason = "max_pieces"
    while pieces < max_pieces:
        bulk = board.lock_candidates_full(pq.current)
        n = len(bulk[0])
        if n == 0:
            reason = "no_candidates"
            break
        if all(bulk[3]):
            reason = "lockout"
            break
        pick = ctrl.choose(board, pq.current, pq.next_piece)
        nl = int(bulk[5][pick])
        lines_total += nl
        clears[nl] = clears.get(nl, 0) + 1
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(n, 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    return pieces, lines_total, clears, reason, ctrl.stats
