"""Masked-root full-expansion regression tests (§44 step 2, 2026-09-10).

Pins the §44.2 fix in CommitScorer.choose: when the unconstrained full-2-ply
base pick violates the well-preservation constraint, the conflict path must
score the kept roots with the EXACT restricted objective
Q2ply = v1 + completion + gamma*max(v2) + Phi (masked-root full expansion
over the kept set only) — never the old v1-only fallback, and never a
second unconstrained base.choose on the conflict path.

Tests:
  1. conflict-pick parity: on frozen-bank preserved states where the base
     pick is excluded, CommitScorer.choose == exact restricted argmax
     (computed independently here, same versioned semantics: fast-path
     ply-2, completion bonus, DEAD_PENALTY for dead roots, Phi when lam).
  2. masked expansions stay masked: the conflict pick is always a member of
     the kept set (canonical bulk index), and equals the base pick whenever
     the base pick preserves.
  3. determinism: repeated choose on the same state returns the same pick
     and the masked_expansions counter only increments on conflicts.
  4. lam parity: with lam=10 the exact objective includes Phi and parity
     still holds (guards the Phi plumbing on the masked path).
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def _load_net():
    from model import TAPNet
    w = HERE.parent.parent / "model_weights_snapshot9.pt"
    if not w.is_file():
        import pytest
        pytest.skip("snapshot9 missing")
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(w, map_location="cpu"))
    net.eval()
    return net


def _bank(n=60):
    from diag_phase2 import Bank
    p = HERE.parent / "_diag_out" / "state_bank_snapshot9_32games.npz"
    if not p.is_file():
        import pytest
        pytest.skip("bank missing")
    bank = Bank(str(p))
    placed, nxt = bank._replay_piece_ids()
    return bank, placed, nxt, min(n, bank.n_states)


def _exact_restricted(net, base, kept_rows, kept_lines, cur, nxt, lam=0.0):
    """Independent exact Q2ply over kept roots (mirrors the versioned base
    semantics: fast-path ply-2 with cur=nxt duplication, completion bonus,
    DEAD_PENALTY for dead roots, plus Phi when lam)."""
    import deathbot_sim as ds
    from afterstate import flat_rows_to_tensor
    from model import piece_to_index
    from well_potential import style_fn_factory
    gamma = float(getattr(base, "gamma_ply", 0.999))
    bonus_map = getattr(base, "line_bonus", None) or {}
    cur_idx = piece_to_index(cur)
    nxt_idx = piece_to_index(nxt)
    sub = np.stack(kept_rows)
    boards = flat_rows_to_tensor(sub.reshape(-1), len(kept_rows))
    cur_t = torch.full((len(kept_rows),), cur_idx, dtype=torch.long)
    fast = getattr(net, "score_batch_fast", None)
    with torch.inference_mode():
        if fast is not None:
            v1 = fast(boards, cur_t).flatten().tolist()
        else:
            v1 = net(boards, cur_t, cur_t).flatten().tolist()
    phi = [0.0] * len(kept_rows)
    if lam:
        phi = list(style_fn_factory(scale=lam)(
            [np.asarray(r, dtype=np.uint32) for r in kept_rows],
            [0] * len(kept_rows)))
    # ply-2 leaves, one batched forward
    mk = getattr(ds.Board, "from_row_bits_bulk", None)
    flat = np.concatenate(
        [np.asarray(r, dtype=np.uint32) for r in kept_rows]).astype(
            np.uint32).tolist()
    succs = mk([int(v) for v in flat]) if mk is not None else [
        ds.Board.from_row_bits([int(t) for t in r]) for r in kept_rows]
    branch_rows, counts = [], []
    for b in succs:
        bulk2 = b.lock_candidates_full(nxt)
        (s2, _x, _y, lo2, flat2, _ln) = bulk2
        n2 = len(s2)
        if n2 == 0:
            counts.append(0)
            continue
        r2 = np.asarray(flat2, dtype=np.uint32).reshape(n2, 21)
        keep2 = [i for i in range(n2) if not lo2[i]]
        if not keep2:
            counts.append(0)
            continue
        counts.append(len(keep2))
        for i in keep2:
            branch_rows.append(r2[i])
    maxv2 = [None] * len(kept_rows)
    if branch_rows:
        big = flat_rows_to_tensor(
            np.stack(branch_rows), len(branch_rows))
        nxt_t = torch.full((len(branch_rows),), nxt_idx, dtype=torch.long)
        with torch.inference_mode():
            if fast is not None:
                all_v = fast(big, nxt_t).flatten().tolist()
            else:
                all_v = net(big, nxt_t, nxt_t).flatten().tolist()
        off = 0
        for pos in range(len(kept_rows)):
            if counts[pos] == 0:
                maxv2[pos] = None
            else:
                maxv2[pos] = max(all_v[off:off + counts[pos]])
                off += counts[pos]
    exact = []
    for pos in range(len(kept_rows)):
        if maxv2[pos] is None:
            exact.append(-1e6)
        else:
            exact.append(float(v1[pos])
                         + float(bonus_map.get(int(kept_lines[pos]), 0.0))
                         + gamma * float(maxv2[pos]) + float(phi[pos]))
    return exact


def _conflict_states(ctrl_kwargs, n=60, need=8, lam=0.0):
    """Frozen-bank (board, cur, nxt, kept, base_pick) tuples where the base
    pick violates the preserve constraint (the conflict path)."""
    import deathbot_sim as ds
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer, preserves_well
    from well_potential import dense_well_features
    net = _load_net()
    bank, placed, nxt, m = _bank(n)
    base = TwoPlyScorer(net, beam=10 ** 9,
                        line_bonus={1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0})
    ctrl = CommitScorer(base, **ctrl_kwargs)
    out = []
    for s in range(m):
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        bulk = pre.lock_candidates_full(placed[s])
        (states, _xs, _ys, lockouts, flat_rows, lines) = bulk
        n_c = len(states)
        if n_c == 0:
            continue
        viable = [i for i, lo in enumerate(lockouts) if not lo]
        if not viable:
            continue
        rows = np.asarray(flat_rows, dtype=np.uint32).reshape(n_c, 21)
        pre_rows = np.asarray(pre.row_bits(), dtype=np.uint32)
        kept = []
        for i in viable:
            try:
                f = dense_well_features(rows[i])
            except Exception:
                continue
            if preserves_well(pre_rows, rows[i], f, ctrl.max_new_holes):
                kept.append(i)
        if not kept:
            continue
        base_pick = base.choose(pre, placed[s], nxt[s])
        if base_pick in kept:
            continue
        out.append((pre, placed[s], nxt[s], kept))
        if len(out) >= need:
            break
    if not out:
        import pytest
        pytest.skip("no conflict states found in bank slice")
    return net, base, ctrl_kwargs, out


def test_conflict_pick_is_exact_restricted_argmax():
    """On conflict states, choose == independent exact Q2ply argmax."""
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer
    net, base, kw, states = _conflict_states(
        {"abort_top": 99, "abort_occ": 10 ** 9, "reenter_occ": 10 ** 9})
    for pre, cur, nxt_piece, kept in states:
        ctrl = CommitScorer(
            TwoPlyScorer(net, beam=10 ** 9,
                         line_bonus={1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}),
            **kw)
        pick = ctrl.choose(pre, cur, nxt_piece)
        assert pick in kept, (pick, kept)
        bulk = pre.lock_candidates_full(cur)
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(len(bulk[0]), 21)
        exact = _exact_restricted(
            net, base, [rows[j] for j in kept],
            [int(bulk[5][j]) for j in kept], cur, nxt_piece)
        want = kept[max(range(len(kept)),
                        key=lambda j: (exact[j], -j))]
        assert pick == want, (pick, want)


def test_lam_conflict_pick_includes_phi():
    """With lam=10, choose == exact Q2ply+Phi argmax (Phi plumbing)."""
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer
    kw = {"abort_top": 99, "abort_occ": 10 ** 9,
          "reenter_occ": 10 ** 9, "lam": 10.0}
    net, base, _, states = _conflict_states(kw, need=6, lam=10.0)
    for pre, cur, nxt_piece, kept in states:
        ctrl = CommitScorer(
            TwoPlyScorer(net, beam=10 ** 9,
                         line_bonus={1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}),
            **kw)
        pick = ctrl.choose(pre, cur, nxt_piece)
        assert pick in kept, (pick, kept)
        bulk = pre.lock_candidates_full(cur)
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(len(bulk[0]), 21)
        exact = _exact_restricted(
            net, base, [rows[j] for j in kept],
            [int(bulk[5][j]) for j in kept], cur, nxt_piece, lam=10.0)
        want = kept[max(range(len(kept)),
                        key=lambda j: (exact[j], -j))]
        assert pick == want, (pick, want)


def test_nonconflict_shortcut_and_determinism():
    """Base pick inside kept is taken directly; choose is deterministic and
    masked_expansions only counts conflicts."""
    import deathbot_sim as ds
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer
    net = _load_net()
    bank, placed, nxt, m = _bank(40)
    kw = {"abort_top": 99, "abort_occ": 10 ** 9, "reenter_occ": 10 ** 9}
    checked_shortcut = checked_conflict = 0
    for s in range(m):
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        base = TwoPlyScorer(net, beam=10 ** 9,
                            line_bonus={1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0})
        ctrl = CommitScorer(base, **kw)
        before = ctrl.stats.get("masked_expansions", 0)
        a = ctrl.choose(pre, placed[s], nxt[s])
        b = ctrl.choose(pre, placed[s], nxt[s])
        assert a == b, (s, a, b)
        base_pick = base.choose(pre, placed[s], nxt[s])
        bulk = pre.lock_candidates_full(placed[s])
        n_c = len(bulk[0])
        viable = [i for i, lo in enumerate(bulk[3]) if not lo]
        if not viable:
            continue
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(n_c, 21)
        from well_potential import dense_well_features
        from commit_controller import preserves_well
        pre_rows = np.asarray(pre.row_bits(), dtype=np.uint32)
        kept = [i for i in viable
                if preserves_well(pre_rows, rows[i],
                                  dense_well_features(rows[i]), 0)]
        if not kept:
            continue
        if base_pick in kept:
            assert a == base_pick, (s, a, base_pick)
            assert ctrl.stats.get("masked_expansions", 0) == before, s
            checked_shortcut += 1
        else:
            assert a in kept, (s, a, kept)
            checked_conflict += 1
        if checked_shortcut and checked_conflict:
            break
    assert checked_shortcut > 0, "no non-conflict state found"
