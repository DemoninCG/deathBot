"""Top-k shortlist regression tests (§27.3 follow-up, 2026-09-08).

Pins the finite-beam fix in TwoPlyScorer.choose:
  * full expansion (beam >= n) stays BYTE-IDENTICAL to the pre-fix champion
    (fingerprint: _diag_out/champ_picks_bank120.json);
  * with 0 < beam < n, the decision is taken over the EXPANDED shortlist
    only (the historical code argmaxed over ALL candidates while only the
    top-k carried the almost-always-negative gamma*max(v2) term, so any
    finite beam systematically played the rank-(k+1) move — the
    "beam5/beam8 die at 30-39 pieces" §27.3 result was this bug, not a
    coverage problem);
  * diverse/v1_safe shortlists never crash and always return a viable pick.
"""
import json
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


def _bank_states(n=40):
    from diag_phase2 import Bank
    p = HERE.parent / "_diag_out" / "state_bank_snapshot9_32games.npz"
    if not p.is_file():
        import pytest
        pytest.skip("bank missing")
    bank = Bank(str(p))
    placed, nxt = bank._replay_piece_ids()
    return bank, placed, nxt, min(n, bank.n_states)


def test_full_expansion_byte_identity():
    """beam >= n must reproduce the champion picks exactly (pre-edit fp)."""
    import deathbot_sim as ds
    from two_ply import TwoPlyScorer
    fp = HERE.parent / "_diag_out" / "champ_picks_bank120.json"
    if not fp.is_file():
        import pytest
        pytest.skip("champion fingerprint missing")
    net = _load_net()
    bank, placed, nxt, n = _bank_states(120)
    sc = TwoPlyScorer(net, beam=10 ** 9)
    old = json.loads(fp.read_text())
    for s in range(min(n, len(old))):
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        assert sc.choose(pre, placed[s], nxt[s]) == old[s], s


def test_finite_beam_picks_within_expanded_shortlist():
    """mode='v1': the pick must be one of the top-k by 1-ply value.

    Under the historical bug the pick was usually the rank-(k+1) candidate
    (best non-expanded, unpenalized by gamma*max(v2)); the fix makes the
    pick always belong to the expanded set.
    """
    import deathbot_sim as ds
    from model import piece_to_index
    from afterstate import flat_rows_to_tensor
    from two_ply import TwoPlyScorer
    net = _load_net()
    bank, placed, nxt, n = _bank_states(40)
    sc = TwoPlyScorer(net, beam=5)
    fast = getattr(net, "score_batch_fast", None)
    with torch.inference_mode():
        for s in range(n):
            lo, hi = bank.candidate_slice(s)
            rows = np.asarray(bank.cand_rows, dtype=np.uint32)[
                lo * 21:hi * 21].reshape(hi - lo, 21)
            lockouts = np.asarray(bank.cand_lockout)[lo:hi]
            viable = [i for i in range(hi - lo) if not lockouts[i]]
            boards = flat_rows_to_tensor(rows[viable], len(viable))
            cur = torch.full((len(viable),), piece_to_index(placed[s]),
                             dtype=torch.long)
            if fast is not None:
                v1 = fast(boards, cur).flatten().tolist()
            else:
                v1 = net(boards, cur, cur).flatten().tolist()
            order = sorted(range(len(viable)), key=lambda j: (-v1[j], j))
            top5 = {viable[j] for j in order[:5]}
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            pick = sc.choose(pre, placed[s], nxt[s])
            assert pick in top5, (s, pick, sorted(top5))


def test_shortlist_modes_run_and_stay_viable():
    """diverse/v1_safe: no crash, pick is a valid (non-lockout) index."""
    import deathbot_sim as ds
    from two_ply import TwoPlyScorer
    net = _load_net()
    bank, placed, nxt, n = _bank_states(20)
    lockout_sets = []
    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        lockout_sets.append(
            {lo + i for i, lk in enumerate(
                np.asarray(bank.cand_lockout)[lo:hi]) if lk})
    for mode in ("diverse", "v1_safe"):
        sc = TwoPlyScorer(net, beam=5, shortlist_mode=mode)
        for s in range(n):
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            pick = sc.choose(pre, placed[s], nxt[s])
            assert pick not in lockout_sets[s], (mode, s, pick)
            assert isinstance(pick, int) and pick >= 0
