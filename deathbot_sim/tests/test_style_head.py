"""Style-head tests: feature parity, zero-init, frozen survival, combination."""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def _bank_rows():
    from diag_phase2 import Bank
    bank = Bank(str(HERE.parent / "_diag_out" / "state_bank_snapshot9_32games.npz"))
    return bank


def test_style_features_match_heuristic_batch():
    """style_features_from_rows agrees with HeuristicScorer batch path."""
    from style_head import style_features_from_rows
    from heuristic import HeuristicScorer, HEIGHT_WEIGHTS
    bank = _bank_rows()
    h = HeuristicScorer()
    rng = np.random.RandomState(0)
    idx = rng.choice(bank.n_states, 25, replace=False)
    for s in idx:
        lo, hi = bank.candidate_slice(int(s))
        k = hi - lo
        rows = bank.cand_rows[lo * 21:hi * 21].reshape(k, 21)
        lines = [int(x) for x in bank.cand_lines[lo:hi]]
        got = h._weighted_reward_batch(rows, lines)["rewards"]
        for j in range(k):
            f = style_features_from_rows(rows[j], lines[j])
            w = h.weights
            manual = (w["weighted_height"] * f[0] + w["holes"] * f[1]
                      + w["hole_depth"] * f[2] + w["accessible_cavities"] * f[3]
                      + w["well_cost"] * f[4] + w["spawn_danger"] * f[5]
                      + w.get("lines_cleared", 0.0) * lines[j])
            assert abs(manual - got[j]) < 1e-4, (s, j, manual, got[j])


def test_style_head_zero_init_is_survival_policy():
    """Fresh head outputs ~0 => combined == survival for any alpha."""
    from style_head import StyleHead, combined_scores
    head = StyleHead()
    feats = torch.randn(17, 8)
    assert torch.allclose(head(feats), torch.zeros(17, 1), atol=1e-6)
    surv = torch.randn(17)
    assert torch.allclose(combined_scores(surv, head(feats).flatten(), 40.0),
                          surv, atol=1e-5)


def test_feature_stats_standardize():
    """Standardized features are zero-mean unit-variance per dim."""
    from style_head import fit_feature_stats, apply_stats
    feats = torch.randn(200, 8) * torch.tensor([100., 5., 8., 3., 4., 2., 1., 6.]) + 50.
    mean, std = fit_feature_stats(feats)
    z = apply_stats(feats, mean, std)
    assert torch.allclose(z.mean(0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(z.std(0, unbiased=False), torch.ones(8), atol=1e-5)


def test_calibrated_output_is_zero_mean():
    """calibrated() removes the global bias, keeps within-state spread."""
    from style_head import StyleHead
    torch.manual_seed(3)
    head = StyleHead()
    with torch.no_grad():
        head.fc2.bias.fill_(86.0)  # simulate Phase-A bias
        head.fc2.weight.normal_()  # trained head: nonzero readout row
        head.fc1.bias.fill_(1.0)  # keep ReLUs alive so spread survives
    feats = torch.randn(17, 8)
    assert float(head(feats).std()) > 0  # guard: test setup has spread
    cal = head.calibrated(feats, feats)
    assert abs(float(cal.mean().detach())) < 1e-4
    assert float(cal.std()) > 0


def test_veto_gate_blocks_far_below_survival():
    """Candidates > veto_margin below best v are -inf regardless of style."""
    from style_head import combined_scores
    surv = torch.tensor([0.0, -5.0, -30.0])
    style = torch.tensor([0.0, 10.0, 100.0])
    out = combined_scores(surv, style, alpha=1.0, veto_margin=25.0)
    assert torch.isneginf(out[2])
    assert int(out[:2].argmax()) == 1  # style re-ranks veto survivors
    out0 = combined_scores(surv, style, alpha=0.0, veto_margin=25.0)
    assert int(out0[:2].argmax()) == 0  # alpha=0 reproduces survival order


def test_snapshot9_frozen_and_head_trainable():
    """Survival params frozen; only head params carry gradients."""
    from style_head import snapshot9_frozen_copy, StyleHead, state_dict_to_vector
    w = HERE.parent.parent / "model_weights_snapshot9.pt"
    if not w.is_file():
        import pytest
        pytest.skip("snapshot9 not found")
    surv = snapshot9_frozen_copy()
    assert all(not p.requires_grad for p in surv.parameters())
    before = state_dict_to_vector(surv.state_dict()).clone()
    head = StyleHead()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss = head(torch.randn(8, 8)).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    after = state_dict_to_vector(surv.state_dict())
    assert torch.equal(before, after)
    assert any(p.grad is not None for p in head.parameters())


def test_combined_ranking_only_moves_with_alpha():
    """Higher alpha moves combined ranking toward the style pick."""
    from style_head import combined_scores
    surv = torch.tensor([0.0, -1.0, -2.0])
    style = torch.tensor([-5.0, 0.0, 5.0])
    c0 = combined_scores(surv, style, 0.0)
    assert int(c0.argmax()) == 0
    c_big = combined_scores(surv, style, 10.0)
    assert int(c_big.argmax()) == 2
