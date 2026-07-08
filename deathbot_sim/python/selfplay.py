"""Monte Carlo self-play training (phase 2.3).

The network plays full Death-mode games greedily (argmax over lock candidates
at each step). We record the (board_state, next_piece) trajectory and label
each state with the number of pieces it survived before game-over. This gives
the network a *temporal* credit assignment signal: states that lead to long
survival get high targets, states that lead to imminent death get low targets.

This is the "keep the board low and clean over time" learning the user asked
about — the target is literally survival length, not a one-step heuristic.

Two key design choices:
  - **Discounted survival**: we use `target = sum_t (gamma^t * 1)` for future
    steps, which equals `(1 - gamma^n) / (1 - gamma)` for n surviving steps.
    With gamma=1.0 this is just `n` (raw survival count). With gamma<1 it
    weights near-term survival more, reducing variance from late-game noise.
    Default gamma=0.99 — close to raw count but slightly front-loaded.
  - **Epsilon-greedy exploration**: with probability epsilon, pick a random
    candidate instead of the argmax. This is critical — pure greedy play
    collapses to one strategy and the network never sees diverse states.
    Default epsilon=0.1, decaying over rounds.

The training loop alternates:
  1. Play N games with the current network → collect (state, target) samples.
  2. Retrain the network on those samples for K epochs.
  3. Repeat.

Run:
    cd python; python selfplay.py --games 200 --rounds 5 --epochs 3
    cd python; python selfplay.py --help

The trained model is saved to `model_weights.pt` (same path as phase 2.2, so
the visualizer can load it unchanged).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Make the python/ directory importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deathbot_sim as ds
from model import TAPNet, board_to_tensor, piece_to_index, PIECE_INDEX
from neural_scorer import NeuralScorer
from heuristic import HeuristicScorer
from scorer import Scorer


ALL_PIECES = list(PIECE_INDEX.keys())  # [I, O, T, S, Z, J, L]


# ---- Game rollout engine ---------------------------------------------------

def play_game(
    scorer: Scorer,
    rng: random.Random | None = None,
    *,
    epsilon: float = 0.1,
    max_pieces: int = 500,
) -> list[tuple[torch.Tensor, int, int]]:
    """Play one full Death-mode game greedily with the scorer.

    At each step:
      1. Spawn a random piece (the "current" piece).
      2. Run the pathfinder to get all lock candidates.
      3. If no candidates (topped out), the game ends.
      4. Score all candidates; pick argmax (or random with prob epsilon).
      5. Lock the piece, clear lines, record the (board, next_piece) state
         BEFORE the lock.

    Returns a list of (board_tensor, next_piece_idx, steps_survived_after)
    tuples. `steps_survived_after` is the number of pieces placed AFTER this
    state (including this one) — i.e. the survival length from this state.
    """
    rng = rng or random.Random()
    board = ds.Board()

    # Pre-roll the piece queue so we know the "next piece" at each step.
    # In TAP the next piece is visible; we model that by peeking ahead.
    trajectory: list[tuple[torch.Tensor, int]] = []  # (board_tensor, next_idx)

    pieces_placed = 0
    while pieces_placed < max_pieces:
        current = rng.choice(ALL_PIECES)
        next_piece = rng.choice(ALL_PIECES)
        candidates = board.search_locks(current)
        if not candidates:
            break  # topped out — game over

        # Record the state BEFORE the lock.
        trajectory.append((board_to_tensor(board), piece_to_index(next_piece)))

        # Pick a candidate: epsilon-greedy.
        if rng.random() < epsilon:
            cand = rng.choice(candidates)
        else:
            # Score all candidates; pick argmax.
            if hasattr(scorer, "score_batch"):
                scores = scorer.score_batch(board, candidates)
            else:
                scores = [scorer.score(board, c) for c in candidates]
            best_i = max(range(len(scores)), key=lambda i: scores[i])
            cand = candidates[best_i]

        board.lock(cand.piece)
        board.clear_lines()
        pieces_placed += 1

    # Now label each state with its survival length (how many pieces were
    # placed AFTER that state, including the one at that state).
    # State at index i survived (len(trajectory) - i) more pieces.
    n = len(trajectory)
    samples: list[tuple[torch.Tensor, int, int]] = []
    for i, (bt, ni) in enumerate(trajectory):
        steps_after = n - i  # pieces placed from this state onward
        samples.append((bt, ni, steps_after))
    return samples


def discounted_survival(steps: int, gamma: float = 0.99) -> float:
    """Convert a raw step count into a discounted survival value.

    target = sum_{t=1}^{steps} gamma^t = gamma * (1 - gamma^steps) / (1 - gamma)

    With gamma=1.0, this is just `steps` (raw count). With gamma<1, near-term
    survival is weighted more, reducing variance from late-game noise.
    """
    if gamma >= 1.0:
        return float(steps)
    if steps == 0:
        return 0.0
    return gamma * (1.0 - gamma ** steps) / (1.0 - gamma)


# ---- MC dataset generator -------------------------------------------------

def generate_selfplay_dataset(
    scorer: Scorer,
    n_games: int,
    rng: random.Random | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    max_pieces: int = 500,
) -> list[tuple[torch.Tensor, int, float]]:
    """Play n_games and return (board_tensor, next_idx, discounted_survival)
    samples for training."""
    rng = rng or random.Random()
    samples: list[tuple[torch.Tensor, int, float]] = []
    for g in range(n_games):
        game = play_game(scorer, rng, epsilon=epsilon, max_pieces=max_pieces)
        for bt, ni, steps in game:
            target = discounted_survival(steps, gamma)
            samples.append((bt, ni, target))
    return samples


# ---- MC training loop -----------------------------------------------------

def train_selfplay(
    n_rounds: int = 5,
    games_per_round: int = 100,
    epochs_per_round: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    seed: int = 42,
    weights_in: Optional[str] = None,
    weights_out: str = "model_weights.pt",
    device: str = "cpu",
) -> TAPNet:
    """Train TAPNet via Monte Carlo self-play.

    Each round:
      1. Play `games_per_round` games with the current network (epsilon-greedy).
      2. Label each state with discounted survival.
      3. Retrain for `epochs_per_round` epochs on the collected samples.

    If `weights_in` is given, load those weights as the starting point
    (typically the phase-2.2 heuristic-trained weights). Otherwise start from
    a fresh network (will play randomly at first — slower convergence).
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Build the network.
    net = TAPNet(output="linear")
    net.to(device)
    if weights_in is not None:
        p = HERE.parent / weights_in if not Path(weights_in).is_absolute() else Path(weights_in)
        if p.exists():
            net.load_state_dict(torch.load(p, map_location=device))
            print(f"Loaded initial weights from {p}")
        else:
            print(f"Weights file {p} not found; starting from scratch.")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for round_idx in range(n_rounds):
        # --- Self-play phase: collect games with the current network. ---
        net.eval()
        scorer = NeuralScorer(net=net, device=device)
        # Decay epsilon slightly each round for less exploration over time.
        cur_epsilon = epsilon * (0.9 ** round_idx)
        print(f"\n=== Round {round_idx + 1}/{n_rounds} (epsilon={cur_epsilon:.3f}) ===")
        print(f"  Playing {games_per_round} games...")
        samples = generate_selfplay_dataset(
            scorer, games_per_round, rng,
            epsilon=cur_epsilon, gamma=gamma,
        )
        if not samples:
            print("  No samples collected (all games topped out immediately?)")
            continue

        boards = torch.stack([s[0] for s in samples]).to(device)
        next_idxs = torch.tensor([s[1] for s in samples], dtype=torch.long).to(device)
        targets = torch.tensor([s[2] for s in samples], dtype=torch.float32).to(device)

        # Stats.
        survivals = [s[2] for s in samples]
        print(f"  Collected {len(samples)} samples from {games_per_round} games.")
        print(f"  Survival stats: min={min(survivals):.1f} "
              f"max={max(survivals):.1f} mean={sum(survivals)/len(survivals):.1f}")

        # --- Training phase: retrain on the collected samples. ---
        n = len(samples)
        net.train()
        for epoch in range(epochs_per_round):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                b = boards[idx]
                ni = next_idxs[idx]
                t = targets[idx].unsqueeze(1)
                optimizer.zero_grad()
                pred = net(b, ni)
                loss = loss_fn(pred, t)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg = epoch_loss / max(1, n_batches)
            print(f"    epoch {epoch + 1}/{epochs_per_round}: loss={avg:.4f}")

    # Save final weights.
    out_path = HERE.parent / weights_out
    torch.save(net.state_dict(), out_path)
    print(f"\nSaved final weights to {out_path}")
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TAPNet via Monte Carlo self-play.")
    parser.add_argument("--rounds", type=int, default=5, help="number of self-play rounds")
    parser.add_argument("--games", type=int, default=100, help="games per round")
    parser.add_argument("--epochs", type=int, default=3, help="epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="exploration rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="survival discount factor")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--weights-in", type=str, default=None, help="initial weights to load")
    parser.add_argument("--weights-out", type=str, default="model_weights.pt", help="output weights path")
    parser.add_argument("--device", type=str, default="cpu", help="device (cpu/cuda)")
    args = parser.parse_args()

    train_selfplay(
        n_rounds=args.rounds,
        games_per_round=args.games,
        epochs_per_round=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon=args.epsilon,
        gamma=args.gamma,
        seed=args.seed,
        weights_in=args.weights_in,
        weights_out=args.weights_out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
