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


def count_overhangs_outside(rows21, well_col: int = WELL_COL) -> int:
    """Count outside-well covered empties (overhangs, broad sense).

    For each non-well column, every empty cell strictly below the column
    top with ANY filled cell above it in the same column counts. This is
    a superset of `holes_outside` (enclosed holes): it also counts
    covered cells that still have a side escape (non-enclosing
    overhangs), which the current `preserves_well` filter lets through.
    Computed on packed [21] rows; pure numpy-free integer loop.
    """
    rows = np.asarray(rows21, dtype=np.uint32).reshape(-1)
    n = 0
    for c in range(10):
        if c == int(well_col):
            continue
        seen_fill_above = False
        for r in range(20, -1, -1):
            filled = (int(rows[r]) >> c) & 1
            if filled:
                seen_fill_above = True
            elif seen_fill_above:
                n += 1
    return n


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
      tiered_preserve       : tiered strictness (§53.7): Tier 0 blocks new
        overhangs too (strictest), Tier 1 is legacy `preserves_well`
        (blocks new holes, allows overhangs), Tier 2 keeps the channel
        only (well_open + no col-9 cover; allows new holes). Tiers fall
        through in order 0 -> 1 -> 2 -> unconstrained fallback; the used
        tier is recorded in stats (tier0/1/2/unconstrained kept counts).
    State: committed (bool), plus counters for the Pareto report
    (decisions_preserved, fallbacks, cashouts, recovers).
    """

    def __init__(self, base_scorer, *, abort_top: int = 14,
                 abort_occ: int = 90, reenter_occ: int = 50,
                 max_new_holes: int = 0, reenter_top: int | None = None,
                 cooldown: int = 0, lam: float = 0.0,
                 debt_limit: float | None = None,
                 progress_patience: int = 0,
                 uniform_commit_scoring: bool = False,
                 rank_cashouts: bool = False,
                 reenter_holes: int | None = None,
                 reentry_min_progress: float = 0.0,
                 ready_patience: int = 0,
                 lam_lo: float | None = None,
                 lam_hi: float | None = None,
                 tiered_preserve: bool = False) -> None:
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
        self.debt_limit = None if debt_limit is None else float(debt_limit)
        self.progress_patience = int(progress_patience)
        self.uniform_commit_scoring = bool(uniform_commit_scoring)
        self.rank_cashouts = bool(rank_cashouts)
        self.reenter_holes = None if reenter_holes is None else int(reenter_holes)
        self.reentry_min_progress = float(reentry_min_progress)
        self.ready_patience = int(ready_patience)
        self.tiered_preserve = bool(tiered_preserve)
        # Usability-based adjustable λ (watch80-style): when lam_lo is set,
        # the per-decision Phi scale interpolates between lam_lo (risky
        # stacks) and lam_hi (clean stacks) via _usability() of the
        # pre-state. None = fixed-lam legacy behavior (exact default).
        self.lam_lo = None if lam_lo is None else float(lam_lo)
        self.lam_hi = None if lam_hi is None else float(lam_hi)
        if self.lam_lo is not None and self.lam_hi is None:
            self.lam_hi = float(lam)
        if self.lam_lo is not None:
            from well_potential import style_fn_factory
            self._phi_unit = style_fn_factory(scale=1.0)
            self._phi = None  # scaled per decision via lam_eff
        elif self.lam:
            from well_potential import style_fn_factory
            self._phi = style_fn_factory(scale=self.lam)
            self._phi_unit = None
        else:
            self._phi = None
            self._phi_unit = None
        self.committed = True  # start building from the opening
        self._cooldown_left = 0
        self._commit_debt = 0.0
        self._max_commit_debt = 0.0
        self._best_progress = 0.0
        self._stalled_pieces = 0
        self._ready_wait = 0
        self.stats = {"preserved_decisions": 0, "fallbacks": 0,
                      "cashouts": 0, "recovers": 0, "style_off": 0,
                      "decisions": 0, "reentries": 0,
                  "reentry_denied_top": 0, "reentry_denied_cool": 0,
                      "reentry_denied_holes": 0, "reentry_denied_geometry": 0,
                  "debt_exits": 0, "progress_exits": 0,
                      "uniform_expansions": 0, "ranked_cashouts": 0,
                      "ready_exits": 0,
                      "tier0_kept": 0, "tier1_kept": 0, "tier2_kept": 0,
                      "tier_fallback": 0,
                      "max_commit_debt": 0.0}

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
            if self._progress(rows[i]) < self.reentry_min_progress:
                continue
            return True
        return False

    @staticmethod
    def _progress(rows21) -> float:
        """Bounded designated-well progress used only by the watchdog."""
        f = dense_well_features(rows21)
        return (min(f["rows_full_except_well"], 4.0)
                + f["well_depth_sat"]
                + 0.5 * f["tetris_ready"])

    @staticmethod
    def _usability(pre_rows21) -> float:
        """watch80-style stack usability in [0, 1] from the PRE-state.

        1.0 = clean low flat stack (full style pressure safe);
        0.0 = tall / holey / rough stack (style pressure should relax).
        Uses only bounded, already-standardized signals: top (12 safe,
        16+ fully risky), holes_outside (0 safe, 4+ fully risky), and
        well_open (closed halves usability — no usable channel exists).
        """
        f = dense_well_features(pre_rows21)
        top_u = min(1.0, max(0.0, (float(f.get("top", 0)) - 12.0) / 4.0))
        holes_u = min(1.0, float(f.get("holes_outside", 0.0)) / 4.0)
        risk = max(top_u, holes_u)
        if f.get("well_open", 0.0) < 0.5:
            risk = max(risk, 0.5)
        return 1.0 - risk

    def _lam_eff(self, pre_rows21) -> float:
        """Effective Phi scale for this decision (fixed or usability-based)."""
        if self.lam_lo is None:
            return float(self.lam)
        u = self._usability(pre_rows21)
        eff = self.lam_lo + (self.lam_hi - self.lam_lo) * u
        self._lam_last_u = u
        self._lam_last_eff = eff
        return eff

    def _begin_recovery(self, *, debt: bool = False,
                        stalled: bool = False, ready: bool = False) -> None:
        self.committed = False
        self.stats["recovers"] += 1
        self._cooldown_left = self.cooldown
        self._commit_debt = 0.0
        self._best_progress = 0.0
        self._stalled_pieces = 0
        self._ready_wait = 0
        if debt:
            self.stats["debt_exits"] += 1
        if stalled:
            self.stats["progress_exits"] += 1
        if ready:
            self.stats["ready_exits"] += 1

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
            self._begin_recovery()
        elif not self.committed:
            if self._cooldown_left > 0:
                self._cooldown_left -= 1
            if occ <= self.reenter_occ:
                if self.cooldown and self._cooldown_left > 0:
                    self.stats["reentry_denied_cool"] += 1
                elif (self.reenter_top is not None
                        and top > self.reenter_top):
                    self.stats["reentry_denied_top"] += 1
                elif (self.reenter_holes is not None
                      and dense_well_features(pre_rows)["holes_outside"]
                      > self.reenter_holes):
                    self.stats["reentry_denied_holes"] += 1
                else:
                    # Usability gate: a channel must actually exist.
                    usable = self._usable_channel(rows, viable)
                    if usable:
                        self.committed = True
                        self.stats["reentries"] += 1
                        self._commit_debt = 0.0
                        self._best_progress = 0.0
                        self._stalled_pieces = 0
                    else:
                        self.stats["reentry_denied_geometry"] += 1
        if not self.committed:
            self.stats["style_off"] += 1
            return self.base.choose(board, current, next_piece)

        if self.progress_patience > 0:
            progress = self._progress(pre_rows)
            if progress > self._best_progress + 1e-9:
                self._best_progress = progress
                self._stalled_pieces = 0
            else:
                self._stalled_pieces += 1
            if self._stalled_pieces >= self.progress_patience:
                self._begin_recovery(stalled=True)
                self.stats["style_off"] += 1
                return self.base.choose(board, current, next_piece)

        if self.ready_patience > 0:
            ready = dense_well_features(pre_rows)["tetris_ready"] > 0.5
            has_i = current == ds.PieceType.I or next_piece == ds.PieceType.I
            self._ready_wait = self._ready_wait + 1 if ready and not has_i else 0
            if self._ready_wait >= self.ready_patience:
                self._begin_recovery(ready=True)
                self.stats["style_off"] += 1
                return self.base.choose(board, current, next_piece)

        # CASH-OUT: reachable Tetris among viable (tetris_ready afterstate).
        cashouts = []
        for i in viable:
            if int(lines[i]) == 4:
                try:
                    if dense_well_features(rows[i])["tetris_ready"] > 0.5:
                        cashouts.append(i)
                except Exception:
                    pass
        if cashouts:
            pick = cashouts[0]
            if self.rank_cashouts and len(cashouts) > 1:
                v1 = self._v1(board, current, cashouts, rows)
                phi = [0.0] * len(cashouts)
                lam_eff = self._lam_eff(pre_rows)
                if lam_eff:
                    if self._phi_unit is not None:
                        unit = list(self._phi_unit(
                            [rows[j] for j in cashouts],
                            [0] * len(cashouts)))
                        phi = [lam_eff * u for u in unit]
                    elif self._phi is not None:
                        phi = list(self._phi([rows[j] for j in cashouts],
                                             [0] * len(cashouts)))
                exact = self._exact_q2ply(cashouts, rows, lines, next_piece,
                                          v1, phi)
                pick = cashouts[max(range(len(cashouts)),
                                    key=lambda j: (exact[j], -j))]
                self.stats["ranked_cashouts"] += 1
            self.stats["cashouts"] += 1
            self._commit_debt = 0.0
            self._best_progress = 0.0
            self._stalled_pieces = 0
            self._ready_wait = 0
            return pick

        # PRESERVE: filter viable to channel-preserving moves (constrain
        # FIRST on the full viable set — never after a v1 shortlist).
        # Tiered mode (§53.7): strictest non-empty tier wins —
        #   Tier 0: legacy preserves_well + no NEW overhangs outside col 9
        #   Tier 1: legacy preserves_well (no new holes; overhangs allowed)
        #   Tier 2: channel only (well_open + no col-9 cover; holes allowed)
        #   else : unconstrained survival fallback (existing behavior).
        tier = "tier_fallback"
        if not self.tiered_preserve:
            kept = []
            for i in viable:
                try:
                    f = dense_well_features(rows[i])
                except Exception:
                    continue
                if preserves_well(pre_rows, rows[i], f, self.max_new_holes):
                    kept.append(i)
        else:
            pre_over = count_overhangs_outside(pre_rows)
            pre_col = _col_bits(pre_rows)
            tier0, tier1, tier2 = [], [], []
            for i in viable:
                try:
                    f = dense_well_features(rows[i])
                except Exception:
                    continue
                if not (f.get("well_open", 0.0) >= 0.5
                        and _col_bits(rows[i]) <= pre_col):
                    continue
                tier2.append(i)
                pre_h = dense_well_features(pre_rows)["holes_outside"]
                if f.get("holes_outside", 0.0) - pre_h > self.max_new_holes:
                    continue
                tier1.append(i)
                try:
                    if count_overhangs_outside(rows[i]) <= pre_over:
                        tier0.append(i)
                except Exception:
                    pass
            if tier0:
                kept, tier = tier0, "tier0_kept"
            elif tier1:
                kept, tier = tier1, "tier1_kept"
            elif tier2:
                kept, tier = tier2, "tier2_kept"
            else:
                kept = []
        if not kept:
            # No preserving move exists: survival fallback, keep commitment
            # (a single fallback placement rarely destroys the plan).
            self.stats["fallbacks"] += 1
            if self.tiered_preserve:
                self.stats["tier_fallback"] += 1
            return self.base.choose(board, current, next_piece)
        if self.tiered_preserve:
            self.stats[tier] += 1

        # Rank the preserved set with EXACT restricted Q2ply (§44.2 fix:
        # audit A showed the v1-only fallback flips 21.5% of conflict
        # decisions at median Q-gap ~6 z-units. The base pick shortcut
        # above is still valid (unconstrained argmax inside kept ==
        # constrained argmax); only the conflict path needed correction.
        # No second unconstrained base.choose call happens here.
        base_pick = self.base.choose(board, current, next_piece)
        if base_pick in kept and not self.uniform_commit_scoring:
            self.stats["preserved_decisions"] += 1
            return base_pick
        # Conflict: score kept roots with exact Q2ply = v1 + completion
        # bonus + gamma*max(v2) + Phi (masked-root full expansion over the
        # kept set only — bounded: <= n roots, ~0.5x a full 2-ply).
        v1 = self._v1(board, current, kept, rows)
        phi = [0.0] * len(kept)
        lam_eff = self._lam_eff(pre_rows)
        if lam_eff:
            try:
                if self._phi_unit is not None:
                    unit = list(self._phi_unit([rows[j] for j in kept],
                                               [0] * len(kept)))
                    phi = [lam_eff * u for u in unit]
                    self.stats["lam_eff_sum"] = (
                        self.stats.get("lam_eff_sum", 0.0) + lam_eff)
                    self.stats["lam_eff_n"] = (
                        self.stats.get("lam_eff_n", 0) + 1)
                elif self._phi is not None:
                    phi = list(self._phi([rows[j] for j in kept],
                                         [0] * len(kept)))
            except Exception:
                pass
        exact = self._exact_q2ply(kept, rows, lines, next_piece, v1, phi)
        best = max(range(len(kept)), key=lambda j: (exact[j], -j))
        pick = kept[best]
        if self.uniform_commit_scoring and base_pick in kept:
            self.stats["uniform_expansions"] += 1
        if base_pick not in kept and self.debt_limit is not None:
            base_v1 = self._v1(board, current, [base_pick], rows)
            base_q = self._exact_q2ply(
                [base_pick], rows, lines, next_piece, base_v1, [0.0])[0]
            self._commit_debt += max(0.0, base_q - exact[best])
            self._max_commit_debt = max(self._max_commit_debt,
                                        self._commit_debt)
            self.stats["max_commit_debt"] = self._max_commit_debt
            if self._commit_debt >= self.debt_limit:
                self._begin_recovery(debt=True)
                self.stats["style_off"] += 1
                return base_pick
        self.stats["preserved_decisions"] += 1
        self.stats["masked_expansions"] = (
            self.stats.get("masked_expansions", 0) + 1)
        return pick

    def _exact_q2ply(self, kept, rows, lines, next_piece, v1, phi):
        """Exact restricted Q2ply over kept roots (bulk-indexed kept).

        Replicates TwoPlyScorer.choose ply-2 semantics exactly:
        final = v1 + completion_bonus + gamma*max(v2) + phi,
        dead-next-turn roots -> DEAD_PENALTY. Ply-2 scored with the
        versioned fast path (cur=nxt duplication, same as the base).
        One batched forward for all kept roots' ply-2 leaves.
        """
        import torch
        from afterstate import flat_rows_to_tensor
        from model import piece_to_index
        import deathbot_sim as ds
        gamma = float(getattr(self.base, "gamma_ply", 0.999))
        bonus_map = getattr(self.base, "line_bonus", None) or {}
        nxt_idx = piece_to_index(next_piece)
        net = self.base.net
        device = self.base.device
        # Successor boards for the kept roots (one FFI call when available).
        mk = getattr(ds.Board, "from_row_bits_bulk", None)
        succs: list = [None] * len(kept)
        if mk is not None:
            try:
                flat = np.concatenate(
                    [np.asarray(rows[j], dtype=np.uint32)
                     for j in kept]).astype(np.uint32).tolist()
                made = mk([int(v) for v in flat])
                for pos, b in enumerate(made):
                    succs[pos] = b
            except Exception:
                succs = [None] * len(kept)
        for pos, j in enumerate(kept):
            if succs[pos] is None:
                succs[pos] = ds.Board.from_row_bits(
                    [int(t) for t in rows[j]])
        branch_rows: list = []
        branch_counts: list[int] = []
        for b in succs:
            bulk2 = b.lock_candidates_full(next_piece)
            (s2, _x2, _y2, lo2, flat2, _ln2) = bulk2
            n2 = len(s2)
            if n2 == 0:
                branch_counts.append(0)
                continue
            r2 = np.asarray(flat2, dtype=np.uint32).reshape(n2, 21)
            keep2 = [i for i in range(n2) if not lo2[i]]
            if not keep2:
                branch_counts.append(0)
                continue
            branch_counts.append(len(keep2))
            for i in keep2:
                branch_rows.append(r2[i])
        maxv2: list = [None] * len(kept)
        if branch_rows:
            big = flat_rows_to_tensor(
                np.stack(branch_rows), len(branch_rows)).to(device)
            nxt_t = torch.full((len(branch_rows),), nxt_idx,
                               dtype=torch.long, device=device)
            fast = getattr(net, "score_batch_fast", None)
            with torch.inference_mode():
                if fast is not None:
                    all_v = fast(big, nxt_t).flatten().tolist()
                else:
                    all_v = net(big, nxt_t, nxt_t).flatten().tolist()
            off = 0
            for pos in range(len(kept)):
                if branch_counts[pos] == 0:
                    maxv2[pos] = None
                else:
                    seg = all_v[off:off + branch_counts[pos]]
                    off += branch_counts[pos]
                    maxv2[pos] = max(seg)
        exact: list = []
        for pos, j in enumerate(kept):
            if maxv2[pos] is None:
                exact.append(-1e6)
            else:
                try:
                    b = float(bonus_map.get(int(lines[j]), 0.0))
                except Exception:
                    b = 0.0
                exact.append(float(v1[pos]) + b + gamma * float(maxv2[pos])
                             + float(phi[pos]))
        return exact

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
                     lam: float = 0.0, debt_limit: float | None = None,
                     progress_patience: int = 0,
                     uniform_commit_scoring: bool = False,
                     rank_cashouts: bool = False,
                     reenter_holes: int | None = None,
                     reentry_min_progress: float = 0.0,
                     ready_patience: int = 0,
                     lam_lo: float | None = None,
                     lam_hi: float | None = None,
                     tiered_preserve: bool = False,
                     device: str = "cpu"):
    """Closed-loop game with the commitment controller. Returns
    (pieces, lines, clears, reason, stats)."""
    import random  # noqa: F401 (rng is passed in)
    from two_ply import TwoPlyScorer
    from piece_history import PieceQueue
    import deathbot_sim as ds
    base = TwoPlyScorer(net, beam=10 ** 9, device=device,
                        line_bonus=dict(COMPLETION_BONUS))
    ctrl = CommitScorer(
        base, abort_top=abort_top, abort_occ=abort_occ,
        reenter_occ=reenter_occ, max_new_holes=max_new_holes,
        reenter_top=reenter_top, cooldown=cooldown, lam=lam,
        debt_limit=debt_limit, progress_patience=progress_patience,
        uniform_commit_scoring=uniform_commit_scoring,
        rank_cashouts=rank_cashouts, reenter_holes=reenter_holes,
        reentry_min_progress=reentry_min_progress,
        ready_patience=ready_patience,
        lam_lo=lam_lo, lam_hi=lam_hi,
        tiered_preserve=tiered_preserve)
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
