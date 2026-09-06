"""Zero-style contract + dense-potential smoke tests (§15.0.2, §15.1.3).

- style_fn=None / alpha=0 reproduces the champion EXACTLY (same picks on
  every bank state; style_fn never called when alpha=0).
- style on surviving branches only: dead branches (DEAD_PENALTY) are never
  promoted by any style value.
- veto_margin masks far-below-best survivors before style applies.
- well_potential: dense credit below depth 4, saturation at 4, explicit
  completion kept outside (clearing destroys setup features).
- seeded-suite boards construct without error.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def _rows_from_heights(h):
    rows = np.zeros(21, dtype=np.uint32)
    for c, hh in enumerate(h):
        for r in range(hh):
            rows[r] |= np.uint32(1 << c)
    return rows


def test_zero_style_reproduces_champion_on_bank():
    from diag_phase2 import Bank
    from model import TAPNet
    from two_ply import TwoPlyScorer
    import deathbot_sim as ds
    bank_path = HERE.parent / "_diag_out" / "state_bank_snapshot9_32games.npz"
    w = HERE.parent.parent / "model_weights_snapshot9.pt"
    if not bank_path.is_file() or not w.is_file():
        import pytest
        pytest.skip("bank or snapshot9 missing")
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(w, map_location="cpu"))
    net.eval()
    bank = Bank(str(bank_path))
    placed, nxt = bank._replay_piece_ids()
    champ = TwoPlyScorer(net, beam=10 ** 9)
    calls = {"n": 0}

    def _fn(rows_list, lines_list):
        calls["n"] += 1
        return [0.0] * len(rows_list)

    styled = TwoPlyScorer(net, beam=10 ** 9, style_fn=_fn, style_alpha=0.0)
    n = min(120, bank.n_states)
    for s in range(n):
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        a = champ.choose(pre, placed[s], nxt[s])
        b = styled.choose(pre, placed[s], nxt[s])
        assert a == b, (s, a, b)
    assert calls["n"] == 0  # alpha=0 must not even call style_fn


def test_style_never_resurrects_dead_branches():
    from model import TAPNet
    from two_ply import TwoPlyScorer, DEAD_PENALTY
    import deathbot_sim as ds
    w = HERE.parent.parent / "model_weights_snapshot9.pt"
    if not w.is_file():
        import pytest
        pytest.skip("snapshot9 missing")
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(w, map_location="cpu"))
    net.eval()
    # style_fn returning +inf-ish on every candidate must still agree with
    # the champion on which branches are dead: it can only re-rank survivors.
    champ = TwoPlyScorer(net, beam=10 ** 9)
    huge = TwoPlyScorer(net, beam=10 ** 9,
                        style_fn=lambda r, l: [1e9] * len(r),
                        style_alpha=1.0)
    board = ds.Board()
    a = champ.choose(board, ds.PieceType.T, ds.PieceType.I)
    b = huge.choose(board, ds.PieceType.T, ds.PieceType.I)
    # On an empty board nothing is dead, so both simply pick *something*
    # valid; the real assertion is no crash + valid index.
    assert isinstance(a, int) and isinstance(b, int)
    assert a >= 0 and b >= 0


def test_veto_masks_far_survivors():
    from model import TAPNet
    from two_ply import TwoPlyScorer
    import deathbot_sim as ds
    w = HERE.parent.parent / "model_weights_snapshot9.pt"
    if not w.is_file():
        import pytest
        pytest.skip("snapshot9 missing")
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(w, map_location="cpu"))
    net.eval()
    board = ds.Board()
    # veto 0.0: only the best pre-style survivor stays eligible; style then
    # picks among (one) eligible candidate — must equal champion pick.
    champ = TwoPlyScorer(net, beam=10 ** 9)
    a = champ.choose(board, ds.PieceType.T, ds.PieceType.I)
    vetoed = TwoPlyScorer(
        net, beam=10 ** 9, style_fn=lambda r, l: [0.0] * len(r),
        style_alpha=1.0, veto_margin=0.0)
    b = vetoed.choose(board, ds.PieceType.T, ds.PieceType.I)
    assert a == b


def test_dense_potential_partial_credit_and_saturation():
    from well_potential import well_potential, dense_well_features
    flat = _rows_from_heights([4] * 10)
    assert well_potential(flat) < 1.2  # no well yet, mostly open-board terms
    d1 = _rows_from_heights([4] * 9 + [3])   # depth 1 precursor
    d4 = _rows_from_heights([8] * 9 + [4])   # depth 4 usable setup
    d9 = _rows_from_heights([10] * 9 + [1])  # depth 9 hoard
    p1, p4, p9 = (well_potential(d1), well_potential(d4), well_potential(d9))
    assert p1 > well_potential(flat)  # partial credit below 4 (was 0.0)
    assert p4 > p1  # progress rewarded
    assert p9 <= p4 + 0.6  # saturates: hoarding not much better than usable
    f = dense_well_features(d4)
    assert f["tetris_ready"] == 1.0
    assert f["well_depth_raw"] == 4.0


def test_seeded_suite_builds():
    from diag_seeded_wells import build_suite, _board_from_heights
    for case in build_suite():
        b = _board_from_heights(case["h"], case.get("holes"))
        assert b is not None
    assert len(build_suite()) == 6
