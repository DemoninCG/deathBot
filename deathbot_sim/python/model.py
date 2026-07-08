"""TAPNet: the neural network for scoring board states in TAP Death mode.

Architecture (per phase-2.1 spec):
  - Board branch (convolutional):
      Input:  [1, 20, 10]  (board; single channel, visible rows only)
      Conv2d(1→32, 3×3, pad=1) -> ReLU      -> [32, 20, 10]
      Conv2d(32→64, 3×3, pad=1) -> ReLU     -> [64, 20, 10]
      MaxPool2d(2×2)                        -> [64, 10, 5]
      Flatten                               -> 3200
  - Feature fusion:
      Concat(flattened board, next_piece_one_hot[7]) -> 3207
  - Head (fully connected):
      Linear(3207→128) -> ReLU
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

import torch
import torch.nn as nn

import deathbot_sim as ds


# Board dimensions (must match the Rust side).
BOARD_HEIGHT = 20  # visible rows
BOARD_WIDTH = 10
NUM_PIECES = 7

# Flattened conv output size: 64 channels * 10 * 5 after two 3x3 convs + 2x2 pool.
_CONV_FLAT = 64 * (BOARD_HEIGHT // 2) * (BOARD_WIDTH // 2)  # = 3200
_FUSED = _CONV_FLAT + NUM_PIECES  # = 3207


class TAPNet(nn.Module):
    """The board-state scoring network.

    forward(board_tensor, next_piece_idx) -> scalar score
      - board_tensor: float tensor of shape [B, 1, 20, 10] (or [1, 20, 10]
        for a single sample; we unsqueeze batch dim if needed)
      - next_piece_idx: int (or LongTensor) — the index of the next piece
        (0=I, 1=O, 2=T, 3=S, 4=Z, 5=J, 6=L). We one-hot encode it internally.

    For batched training, pass [B, 1, 20, 10] and a [B] LongTensor of indices;
    output is [B, 1]. For single-sample inference (live play), pass
    [1, 20, 10] and an int; output is a Python float.
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
        next_piece_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of (board, next_piece) pairs.

        Args:
            board: float tensor [B, 1, 20, 10].
            next_piece_idx: LongTensor [B] in [0, 6].

        Returns:
            [B, 1] tensor of scores.
        """
        # Board branch.
        x = self.relu(self.conv1(board))   # [B, 32, 20, 10]
        x = self.relu(self.conv2(x))       # [B, 64, 20, 10]
        x = self.pool(x)                   # [B, 64, 10, 5]
        x = torch.flatten(x, 1)            # [B, 3200]

        # Next-piece one-hot: [B, 7].
        piece_oh = torch.nn.functional.one_hot(
            next_piece_idx, num_classes=NUM_PIECES
        ).float()

        # Fusion.
        x = torch.cat([x, piece_oh], dim=1)  # [B, 3207]

        # Head.
        x = self.relu(self.fc1(x))           # [B, 128]
        x = self.fc2(x)                      # [B, 1]
        x = self.out_act(x)                  # [B, 1]
        return x

    @torch.no_grad()
    def score_one(
        self,
        board_tensor: torch.Tensor,
        next_piece_idx: int,
    ) -> float:
        """Convenience for single-sample inference (live play).

        Args:
            board_tensor: [1, 20, 10] or [20, 10] float tensor.
            next_piece_idx: int in [0, 6].

        Returns:
            Python float score.
        """
        if board_tensor.dim() == 2:
            board_tensor = board_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,20,10]
        elif board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)  # [1,1,20,10] if [1,20,10]
        idx = torch.tensor([next_piece_idx], dtype=torch.long)
        out = self.forward(board_tensor, idx)
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
    """
    t = torch.zeros(BOARD_HEIGHT, BOARD_WIDTH, dtype=torch.float32)
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            # board.cell(c, r): row 0 = bottom. Tensor row 0 = top, so flip.
            if board.cell(c, r):
                t[BOARD_HEIGHT - 1 - r][c] = 1.0
    return t.unsqueeze(0)  # [1, 20, 10]


def piece_to_index(kind: "ds.PieceType") -> int:
    """Map a PieceType to its one-hot index."""
    return PIECE_INDEX[kind]
