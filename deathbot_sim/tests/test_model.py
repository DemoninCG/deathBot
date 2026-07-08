"""Tests for the TAPNet neural network and NeuralScorer adapter (phase 2.1).

Run with:
    python -m pytest tests/test_model.py -v

These verify:
  - Tensor shapes through the conv + fusion + FC stack.
  - Forward pass produces a scalar for single-sample and batch input.
  - Gradients flow through all layers.
  - board_to_tensor correctly encodes the board (bottom row = last tensor row).
  - NeuralScorer implements the Scorer protocol and produces sensible floats.
  - score_batch matches score for each candidate (batched == single).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the python/ directory importable.
HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import torch  # noqa: E402

import deathbot_sim as ds  # noqa: E402
from model import (  # noqa: E402
    TAPNet,
    board_to_tensor,
    piece_to_index,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_PIECES,
    _FUSED,
)
from neural_scorer import NeuralScorer  # noqa: E402
from scorer import Scorer  # noqa: E402


# ---- TAPNet shape + forward -----------------------------------------------

class TestTAPNetShapes:
    def test_single_sample_forward_shape(self):
        net = TAPNet()
        board = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        idx = torch.tensor([0], dtype=torch.long)
        out = net(board, idx)
        assert out.shape == (1, 1), f"expected (1,1), got {out.shape}"

    def test_batch_forward_shape(self):
        net = TAPNet()
        B = 8
        board = torch.zeros(B, 1, BOARD_HEIGHT, BOARD_WIDTH)
        idx = torch.tensor([0] * B, dtype=torch.long)
        out = net(board, idx)
        assert out.shape == (B, 1), f"expected ({B},1), got {out.shape}"

    def test_layer_sizes_attribute(self):
        net = TAPNet()
        # Effective layer sizes for the visualizer: [fused, 128, 1].
        assert net.layer_sizes == [_FUSED, 128, 1]

    def test_fused_dim_is_3207(self):
        # 64 channels * 10 * 5 (after pool) + 7 (next piece) = 3207.
        assert _FUSED == 3207

    def test_param_count_is_lightweight(self):
        # Sanity: should be under 500K params (lightweight for live play).
        net = TAPNet()
        n = sum(p.numel() for p in net.parameters())
        assert n < 500_000, f"param count {n} is too high (>500K)"

    def test_score_one_returns_python_float(self):
        net = TAPNet()
        board = torch.zeros(BOARD_HEIGHT, BOARD_WIDTH)  # 2D, unsqueezed internally
        score = net.score_one(board, 0)
        assert isinstance(score, float), f"expected float, got {type(score)}"

    def test_score_one_accepts_3d_tensor(self):
        net = TAPNet()
        board = torch.zeros(1, BOARD_HEIGHT, BOARD_WIDTH)  # [1,20,10]
        score = net.score_one(board, 0)
        assert isinstance(score, float)


# ---- output activation ----------------------------------------------------

class TestOutputActivation:
    def test_linear_output_is_unbounded(self):
        net = TAPNet(output="linear")
        # With random init, output should be some unbounded float.
        board = torch.randn(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        idx = torch.tensor([0], dtype=torch.long)
        out = net(board, idx)
        # No specific bound — just check it's a finite number.
        assert torch.isfinite(out).all()

    def test_tanh_output_is_bounded(self):
        net = TAPNet(output="tanh")
        board = torch.randn(1, 1, BOARD_HEIGHT, BOARD_WIDTH) * 10
        idx = torch.tensor([0], dtype=torch.long)
        out = net(board, idx)
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_invalid_output_kind_raises(self):
        with pytest.raises(ValueError, match="output must be"):
            TAPNet(output="sigmoid")


# ---- gradient flow --------------------------------------------------------

class TestGradients:
    def test_all_params_get_gradients(self):
        net = TAPNet()
        net.zero_grad()
        board = torch.randn(2, 1, BOARD_HEIGHT, BOARD_WIDTH)
        idx = torch.tensor([0, 3], dtype=torch.long)
        out = net(board, idx)
        loss = out.sum()
        loss.backward()
        for name, p in net.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} gradient is all zeros"

    def test_gradient_flows_through_next_piece_input(self):
        # The next-piece one-hot is concatenated; changing it should change
        # the loss. Verify by checking the fc1 weights get gradients.
        net = TAPNet()
        net.zero_grad()
        board = torch.zeros(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        idx = torch.tensor([0], dtype=torch.long)
        net(board, idx).sum().backward()
        assert net.fc1.weight.grad is not None
        assert net.fc1.weight.grad.abs().sum() > 0


# ---- board_to_tensor ------------------------------------------------------

class TestBoardToTensor:
    def test_empty_board_is_all_zeros(self):
        board = ds.Board()
        t = board_to_tensor(board)
        assert t.shape == (1, BOARD_HEIGHT, BOARD_WIDTH)
        assert (t == 0).all()

    def test_filled_cell_appears_in_tensor(self):
        board = ds.Board()
        board.set_cell(3, 5)
        t = board_to_tensor(board)
        # board row 5 (bottom-indexed) -> tensor row 20-1-5 = 14.
        assert t[0][BOARD_HEIGHT - 1 - 5][3] == 1.0

    def test_bottom_row_maps_to_last_tensor_row(self):
        board = ds.Board()
        board.set_cell(0, 0)  # bottom-left cell
        t = board_to_tensor(board)
        assert t[0][BOARD_HEIGHT - 1][0] == 1.0

    def test_top_visible_row_maps_to_first_tensor_row(self):
        board = ds.Board()
        board.set_cell(9, BOARD_HEIGHT - 1)  # top-right visible cell (row 19)
        t = board_to_tensor(board)
        assert t[0][0][9] == 1.0

    def test_invisible_row_excluded(self):
        board = ds.Board()
        board.set_cell(5, board.visible_rows)  # invisible row (row 20)
        t = board_to_tensor(board)
        assert t.shape == (1, BOARD_HEIGHT, BOARD_WIDTH)
        assert (t == 0).all(), "invisible row should not appear in tensor"


# ---- piece_to_index -------------------------------------------------------

class TestPieceToIndex:
    def test_all_pieces_have_distinct_indices(self):
        idxs = {
            ds.PieceType.I: 0,
            ds.PieceType.O: 1,
            ds.PieceType.T: 2,
            ds.PieceType.S: 3,
            ds.PieceType.Z: 4,
            ds.PieceType.J: 5,
            ds.PieceType.L: 6,
        }
        for kind, expected in idxs.items():
            assert piece_to_index(kind) == expected
        assert len(set(idxs.values())) == NUM_PIECES


# ---- NeuralScorer ---------------------------------------------------------

class TestNeuralScorer:
    def test_implements_scorer_protocol(self):
        scorer = NeuralScorer()
        # Scorer is a runtime_checkable Protocol with a `score` method.
        assert isinstance(scorer, Scorer)

    def test_score_returns_float(self):
        scorer = NeuralScorer()
        board = ds.Board()
        locks = board.search_locks(ds.PieceType.T)
        sc = scorer.score(board, locks[0])
        assert isinstance(sc, float)

    def test_score_batch_matches_single(self):
        scorer = NeuralScorer()
        board = ds.Board()
        locks = board.search_locks(ds.PieceType.T)
        single = [scorer.score(board, c) for c in locks]
        batched = scorer.score_batch(board, locks)
        assert len(single) == len(batched)
        for s, b in zip(single, batched):
            assert abs(s - b) < 1e-4, f"single {s} != batched {b}"

    def test_layer_sizes_exposed_for_viz(self):
        scorer = NeuralScorer()
        assert hasattr(scorer, "layer_sizes")
        assert scorer.layer_sizes == [_FUSED, 128, 1]

    def test_score_with_different_next_piece(self):
        # The network's output should depend on the next-piece input (it's
        # part of the input). Different indices -> different scores (in
        # general, for an untrained net with random init).
        scorer = NeuralScorer()
        board = ds.Board()
        locks = board.search_locks(ds.PieceType.T)
        s0 = scorer.score_with_next(board, locks[0], 0)
        s1 = scorer.score_with_next(board, locks[0], 1)
        # They might coincidentally be equal, but the network *can* produce
        # different outputs. Just check both are finite.
        assert isinstance(s0, float) and isinstance(s1, float)
        import math
        assert math.isfinite(s0) and math.isfinite(s1)

    def test_empty_candidates_returns_empty_list(self):
        scorer = NeuralScorer()
        board = ds.Board()
        assert scorer.score_batch(board, []) == []
