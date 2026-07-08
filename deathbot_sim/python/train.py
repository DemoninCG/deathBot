"""Dataset generator + training loop for the heuristic teacher (phase 2.2).

The generator creates random board states by simulating random piece
placements (using the pathfinder + a random valid lock). For each board, it:
  1. Picks a random "current piece" and "next piece".
  2. Runs the pathfinder to get all lock candidates for the current piece.
  3. Computes the heuristic reward for each candidate's resulting board.
  4. The TARGET is the reward of the BEST candidate (the network learns to
     predict the value of the optimal placement, not the value of any random
     placement). This is "imitation of the heuristic teacher."
  5. Stores (board_tensor, next_piece_idx, target_reward) as a training sample.

The training loop is standard supervised regression: MSE loss between the
network's output and the target reward, Adam optimizer, mini-batches.

Run:
    python -m train --epochs 10 --batch-size 64 --samples 10000
    python -m train --help

The trained model is saved to `model_weights.pt` and can be loaded by the
visualizer (`viz_live.py --nn --weights model_weights.pt`).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the python/ directory importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deathbot_sim as ds
from model import TAPNet, board_to_tensor, piece_to_index, PIECE_INDEX
from heuristic import HeuristicScorer, reward
from scorer import Scorer


# ---- Random board generation ---------------------------------------------

ALL_PIECES = list(PIECE_INDEX.keys())  # [I, O, T, S, Z, J, L]


def random_board(max_pieces: int = 40, rng: random.Random | None = None) -> "ds.Board":
    """Generate a random board state by simulating random piece placements.

    Plays `max_pieces` random pieces using the pathfinder (pick a random valid
    lock candidate at each step). Stops early if the board tops out. This
    produces realistic board states with varied stack shapes, holes, and wells
    — the kind of states the network will see during real play.
    """
    rng = rng or random.Random()
    board = ds.Board()
    for _ in range(max_pieces):
        kind = rng.choice(ALL_PIECES)
        candidates = board.search_locks(kind)
        if not candidates:
            break  # topped out
        # Pick a random candidate (uniform; we want diverse boards, not
        # heuristic-guided boards, for training data).
        cand = rng.choice(candidates)
        board.lock(cand.piece)
        board.clear_lines()
    return board


def generate_sample(
    scorer: Scorer,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, int, float] | None:
    """Generate one training sample: (board_tensor, next_piece_idx, target).

    The target is the reward of the BEST lock candidate for a random current
    piece on a random board. Returns None if the board has no valid placements
    for the chosen piece (shouldn't happen on non-full boards).
    """
    rng = rng or random.Random()
    board = random_board(max_pieces=rng.randint(5, 30), rng=rng)
    current = rng.choice(ALL_PIECES)
    next_piece = rng.choice(ALL_PIECES)
    candidates = board.search_locks(current)
    if not candidates:
        return None
    # Score all candidates with the heuristic; target = best reward.
    scores = [scorer.score(board, c) for c in candidates]
    best_score = max(scores)
    # The input board is the PRE-lock board (the network sees the current
    # state + next piece, and must predict the value of the best placement).
    board_tensor = board_to_tensor(board)
    next_idx = piece_to_index(next_piece)
    return board_tensor, next_idx, best_score


def generate_dataset(
    n_samples: int,
    scorer: Scorer,
    rng: random.Random | None = None,
) -> Iterator[tuple[torch.Tensor, int, float]]:
    """Generate n_samples training samples, skipping any that fail."""
    rng = rng or random.Random()
    produced = 0
    while produced < n_samples:
        sample = generate_sample(scorer, rng)
        if sample is not None:
            yield sample
            produced += 1


# ---- Training loop --------------------------------------------------------

def train(
    n_samples: int = 5000,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    weights_out: str = "model_weights.pt",
    device: str = "cpu",
) -> TAPNet:
    """Train TAPNet on the heuristic teacher dataset.

    Args:
        n_samples: number of training samples to generate.
        epochs: number of passes over the dataset.
        batch_size: mini-batch size.
        lr: learning rate (Adam).
        seed: RNG seed for reproducibility.
        weights_out: path to save the trained weights.
        device: 'cpu' or 'cuda'.

    Returns:
        The trained TAPNet.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Build the heuristic scorer (teacher).
    scorer = HeuristicScorer()

    print(f"Generating {n_samples} training samples...")
    samples = list(generate_dataset(n_samples, scorer, rng))
    boards = torch.stack([s[0] for s in samples])  # [N, 1, 20, 10]
    next_idxs = torch.tensor([s[1] for s in samples], dtype=torch.long)  # [N]
    targets = torch.tensor([s[2] for s in samples], dtype=torch.float32)  # [N]
    print(f"Generated {len(samples)} samples. Target stats: "
          f"min={targets.min():.2f} max={targets.max():.2f} "
          f"mean={targets.mean():.2f}")

    # Build the network + optimizer.
    net = TAPNet(output="linear")
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(samples)
    print(f"\nTraining for {epochs} epochs (batch_size={batch_size}, lr={lr})...")
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b = boards[idx].to(device)
            ni = next_idxs[idx].to(device)
            t = targets[idx].to(device).unsqueeze(1)  # [B, 1]
            optimizer.zero_grad()
            pred = net(b, ni)  # [B, 1]
            loss = loss_fn(pred, t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        print(f"  epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    # Save weights.
    out_path = HERE.parent / weights_out
    torch.save(net.state_dict(), out_path)
    print(f"\nSaved trained weights to {out_path}")
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TAPNet on the heuristic teacher.")
    parser.add_argument("--samples", type=int, default=5000, help="number of training samples")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--weights-out", type=str, default="model_weights.pt", help="output weights path")
    parser.add_argument("--device", type=str, default="cpu", help="device (cpu/cuda)")
    args = parser.parse_args()

    train(
        n_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        weights_out=args.weights_out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
