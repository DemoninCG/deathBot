"""Parity test: viz stylish scorer must match the sweep's TwoPlyScorer path.

Compares TwoPlyVizScorer (phi_lambda=20 + completion bonus) against a
direct TwoPlyScorer(style_fn) on bank states: identical picks everywhere.
Also pins phi_lambda=0 == pure champion (style_fn never engaged).
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

W = HERE.parent.parent / "model_weights_snapshot9.pt"
COMPLETION = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}


def _net():
    from model import TAPNet
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(W, map_location="cpu"))
    net.eval()
    return net


def test_viz_stylish_matches_sweep_path():
    from diag_phase2 import Bank
    from two_ply import TwoPlyScorer
    from well_potential import style_fn_factory
    from viz_live import TwoPlyVizScorer
    import deathbot_sim as ds
    bank_path = HERE.parent / "_diag_out" / "state_bank_snapshot9_32games.npz"
    if not bank_path.is_file() or not W.is_file():
        import pytest
        pytest.skip("bank or snapshot9 missing")
    net = _net()
    bank = Bank(str(bank_path))
    placed, nxt = bank._replay_piece_ids()
    ref = TwoPlyScorer(net, beam=10 ** 9, line_bonus=dict(COMPLETION),
                       style_fn=style_fn_factory(scale=20.0),
                       style_alpha=1.0)
    viz = TwoPlyVizScorer(net, line_bonus=dict(COMPLETION),
                          phi_lambda=20.0, well_col=9)
    n = min(60, bank.n_states)
    for s in range(n):
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        a = ref.choose(pre, placed[s], nxt[s])
        cands = pre.search_locks(placed[s])
        scores = viz.score_batch(
            pre, cands,
            current_piece_idx=int(__import__("model").piece_to_index(placed[s])),
            next_piece_idx=int(__import__("model").piece_to_index(nxt[s])))
        b = int(np.argmax(scores))
        assert a == b, (s, a, b)


def test_viz_phi_zero_is_champion():
    from two_ply import TwoPlyScorer
    from viz_live import TwoPlyVizScorer
    import deathbot_sim as ds
    if not W.is_file():
        import pytest
        pytest.skip("snapshot9 missing")
    net = _net()
    champ = TwoPlyScorer(net, beam=10 ** 9)
    viz = TwoPlyVizScorer(net, phi_lambda=0.0)
    assert viz._style_fn is None
    board = ds.Board()
    a = champ.choose(board, ds.PieceType.T, ds.PieceType.I)
    cands = board.search_locks(ds.PieceType.T)
    scores = viz.score_batch(board, cands, current_piece_idx=2,
                             next_piece_idx=0)
    assert int(np.argmax(scores)) == a
