"""TAPNet: the neural network for scoring board states in TAP Death mode.

Architecture (per phase-2.1 spec):
  - Board branch (convolutional):
      Input:  [1, 20, 10]  (board; single channel, visible rows only)
      Conv2d(1→32, 3×3, pad=1) -> ReLU      -> [32, 20, 10]
      Conv2d(32→64, 3×3, pad=1) -> ReLU     -> [64, 20, 10]
      MaxPool2d(2×2)                        -> [64, 10, 5]
      Flatten                               -> 3200
  - Feature fusion:
      Concat(flattened board, current_piece_one_hot[7], next_piece_one_hot[7]) -> 3214
  - Head (fully connected):
      Linear(3214→128) -> ReLU
      Linear(128→1)    -> scalar goodness score

Output is a single float: higher = better placement. The output activation
defaults to Linear (unbounded) for value-function-style training; pass
`output="tanh"` to the constructor for a bounded [-1, 1] output if you prefer.

Param count: ~411K — lightweight enough for sub-millisecond CPU inference
during live gameplay.

The model exposes `layer_sizes` so the phase-1.4 visualizer can draw the
neuron layout without changes. (We report the *effective* layer sizes after
conv fusion: [3207, 128, 1] — the conv stack is summarized as the 3207-dim
flattened feature vector.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import deathbot_sim as ds


# Board dimensions (must match the Rust side).
BOARD_HEIGHT = 20  # visible rows
BOARD_WIDTH = 10
NUM_PIECES = 7

# Flattened conv output size: 64 channels * 10 * 5 after two 3x3 convs + 2x2 pool.
_CONV_FLAT = 64 * (BOARD_HEIGHT // 2) * (BOARD_WIDTH // 2)  # = 3200
# Fused = conv flat + current_piece_one_hot[7] + next_piece_one_hot[7]
_FUSED = _CONV_FLAT + NUM_PIECES * 2  # = 3214


class TAPNet(nn.Module):
    """The board-state scoring network.

    forward(board_tensor, current_piece_idx, next_piece_idx) -> scalar score
      - board_tensor: float tensor of shape [B, 1, 20, 10]
      - current_piece_idx: LongTensor [B] — the piece being placed (0=I..6=L)
      - next_piece_idx: LongTensor [B] — the upcoming piece (0=I..6=L)
        Both are one-hot encoded internally and concatenated with the conv
        features. This lets the network know what it's placing AND what's
        coming next, so it can place the current piece to accommodate.

    For batched training, pass [B, 1, 20, 10] and [B] LongTensors; output is
    [B, 1]. For single-sample inference (live play), pass [1, 20, 10] and
    ints; output is a Python float.
    """

    def __init__(self, output: str = "linear") -> None:
        super().__init__()
        # Board branch (convolutional).
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Head (fully connected, after fusion with next-piece one-hot).
        self.fc1 = nn.Linear(_FUSED, 128)
        self.fc2 = nn.Linear(128, 1)

        if output == "tanh":
            self.out_act: nn.Module = nn.Tanh()
        elif output == "linear":
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"output must be 'linear' or 'tanh', got {output!r}")
        self.output_kind = output

        # Effective layer sizes for the visualizer: [fused, 128, 1].
        # (The conv stack is summarized as the fused feature dimension.)
        self.layer_sizes = [_FUSED, 128, 1]

    def forward(
        self,
        board: torch.Tensor,
        current_piece_idx: torch.Tensor,
        next_piece_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of (board, current_piece, next_piece) triples.

        Args:
            board: float tensor [B, 1, 20, 10].
            current_piece_idx: LongTensor [B] in [0, 6].
            next_piece_idx: LongTensor [B] in [0, 6].

        Returns:
            [B, 1] tensor of scores.
        """
        # Board branch.
        x = self.relu(self.conv1(board))   # [B, 32, 20, 10]
        x = self.relu(self.conv2(x))       # [B, 64, 20, 10]
        x = self.pool(x)                   # [B, 64, 10, 5]
        x = torch.flatten(x, 1)            # [B, 3200]

        # Current + next piece one-hots: [B, 7] each.
        cur_oh = torch.nn.functional.one_hot(
            current_piece_idx, num_classes=NUM_PIECES
        ).float()
        next_oh = torch.nn.functional.one_hot(
            next_piece_idx, num_classes=NUM_PIECES
        ).float()

        # Fusion: conv features + current piece + next piece.
        x = torch.cat([x, cur_oh, next_oh], dim=1)  # [B, 3214]

        # Head.
        x = self.relu(self.fc1(x))           # [B, 128]
        x = self.fc2(x)                      # [B, 1]
        x = self.out_act(x)                  # [B, 1]
        return x

    @torch.no_grad()
    def score_one(
        self,
        board_tensor: torch.Tensor,
        current_piece_idx: int,
        next_piece_idx: int,
    ) -> float:
        """Convenience for single-sample inference (live play).

        Args:
            board_tensor: [1, 20, 10] or [20, 10] float tensor.
            current_piece_idx: int in [0, 6].
            next_piece_idx: int in [0, 6].

        Returns:
            Python float score.
        """
        if board_tensor.dim() == 2:
            board_tensor = board_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,20,10]
        elif board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)  # [1,1,20,10] if [1,20,10]
        device = board_tensor.device
        cur = torch.tensor([current_piece_idx], dtype=torch.long, device=device)
        nxt = torch.tensor([next_piece_idx], dtype=torch.long, device=device)
        out = self.forward(board_tensor, cur, nxt)
        return float(out.item())


# ---- Board <-> tensor conversion -----------------------------------------

# Piece index mapping (must match PieceType repr in Rust).
PIECE_INDEX = {
    ds.PieceType.I: 0,
    ds.PieceType.O: 1,
    ds.PieceType.T: 2,
    ds.PieceType.S: 3,
    ds.PieceType.Z: 4,
    ds.PieceType.J: 5,
    ds.PieceType.L: 6,
}


def board_to_tensor(board: "ds.Board") -> torch.Tensor:
    """Convert a deathbot_sim Board to a [1, 20, 10] float tensor.

    Row 0 (bottom) is the last row in the tensor's dim 0 (top-first), matching
    the conv stack's expectation that the top of the field is row 0 of the
    input. Filled cells = 1.0, empty = 0.0. The invisible row is excluded.

    Uses `row_bits()` (one FFI call returning all 21 row bitmasks) + numpy
    vectorized bit decoding instead of 200 individual `cell()` calls.
    """
    bits = board.row_bits()  # list of 21 u32 row bitmasks, row 0 = bottom
    # Take only visible rows (0..19), reverse to top-first.
    visible = bits[:BOARD_HEIGHT]  # rows 0..19, bottom-first
    # Build column-bit mask: [1, 2, 4, ..., 512] for cols 0..9.
    col_masks = np.array([1 << c for c in range(BOARD_WIDTH)], dtype=np.uint32)
    # Decode: row_arr is [20] uint32; bitwise AND with col_masks gives [20, 10]
    # boolean. Flip rows so tensor row 0 = top (row 19).
    row_arr = np.array(visible, dtype=np.uint32).reshape(-1, 1)  # [20, 1]
    grid = (row_arr & col_masks).astype(bool).astype(np.float32)  # [20, 10]
    # Flip so row 0 of tensor = top of field (row 19 of board).
    grid = grid[::-1, :].copy()  # [20, 10], top-first, contiguous
    return torch.from_numpy(grid).unsqueeze(0)  # [1, 20, 10]


def piece_to_index(kind: "ds.PieceType") -> int:
    """Map a PieceType to its one-hot index."""
    return PIECE_INDEX[kind]
