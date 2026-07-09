"""Monte Carlo self-play training with replay buffer + reward shaping (phase 2.3+).

Three improvements over the original phase-2.3 self-play:

  1. **Reward shaping** (#1): the target is a blend of discounted survival
     and the heuristic reward of the resulting board:
         target = survival_weight * discounted_survival(steps, gamma)
                + heuristic_weight * heuristic_reward(board_after_lock)
     The heuristic gives a dense per-step signal (survival alone is sparse);
     survival gives the true long-term outcome. Default 0.7 survival + 0.3
     heuristic.

  2. **Experience replay buffer** (#2): a rolling buffer stores samples across
     *all* rounds, not just the current one. Each round's new samples are
     added, and training minibatches are sampled uniformly from the buffer.
     This (a) reuses each game ~10× (sample efficiency) and (b) prevents
     catastrophic forgetting of early-game states once the network mostly
     sees late-game ones.

  3. **Richer state input** (#3): the network now receives the current piece
     (what it's placing) in addition to the board + next piece. This closes
     a major information gap — the network was previously blind to what it
     was placing.

Run:
    cd python; python selfplay.py --rounds 10 --games 200 --epochs 3 --weights-in model_weights_snapshot1.pt
    cd python; python selfplay.py --help
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
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
from heuristic import HeuristicScorer, reward as heuristic_reward
from scorer import Scorer
from piece_history import PieceHistory, PieceQueue


ALL_PIECES = list(PIECE_INDEX.keys())  # [I, O, T, S, Z, J, L]


# ---- Replay buffer (#2) ---------------------------------------------------

class ReplayBuffer:
    """A FIFO replay buffer that stores (board, cur_idx, next_idx, target)
    samples across all rounds. Training minibatches are sampled uniformly.

    The buffer has a max capacity; old samples are evicted FIFO when full.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, board: torch.Tensor, cur_idx: int, next_idx: int,
             target: float) -> None:
        """Add a sample to the buffer."""
        self.buffer.append((board, cur_idx, next_idx, target))

    def extend(self, samples: list[tuple[torch.Tensor, int, int, float]]) -> None:
        """Add many samples at once."""
        for s in samples:
            self.buffer.append(s)

    def sample(self, batch_size: int, rng: random.Random | None = None
               ) -> list[tuple[torch.Tensor, int, int, float]]:
        """Sample `batch_size` items uniformly at random.

        Samples random indices directly (O(batch_size)) instead of copying
        the entire buffer to a list (which was O(buffer_size) per batch and
        caused severe slowdowns as the buffer grew toward capacity).
        """
        if not self.buffer:
            return []
        rng = rng or random.Random()
        n = min(batch_size, len(self.buffer))
        buf_len = len(self.buffer)
        # Sample distinct indices in [0, buf_len) without materializing the
        # whole buffer as a list.
        indices = rng.sample(range(buf_len), n)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


# ---- Game rollout engine (#3: current piece tracking) ---------------------

def play_game(
    scorer: Scorer,
    rng: random.Random | None = None,
    *,
    epsilon: float = 0.1,
    max_pieces: int = 1500,
    heuristic_scorer: Optional[HeuristicScorer] = None,
) -> list[tuple[torch.Tensor, int, int, float, float]]:
    """Play one full Death-mode game greedily with the scorer.

    Returns a list of (board_tensor, current_idx, next_idx, heuristic_reward,
    steps_survived) tuples. The caller blends heuristic_reward +
    discounted_survival into the final target (reward shaping, #1).

    `heuristic_scorer` (if given) is used to compute the per-step heuristic
    reward of the board AFTER the lock. If None, the heuristic component is
    0 (pure survival targets — the old behavior).
    """
    rng = rng or random.Random()
    board = ds.Board()
    piece_queue = PieceQueue(rng)

    # Each trajectory entry: (board_tensor, cur_idx, next_idx, heuristic_val)
    trajectory: list[tuple[torch.Tensor, int, int, float]] = []

    pieces_placed = 0
    while pieces_placed < max_pieces:
        current = piece_queue.current
        next_piece = piece_queue.next_piece
        candidates = board.search_locks(current)
        if not candidates:
            break  # topped out

        # Pick a candidate: epsilon-greedy.
        if rng.random() < epsilon:
            cand = rng.choice(candidates)
        else:
            if hasattr(scorer, "score_batch"):
                # Pass the current piece index so the network knows what it's
                # placing (#3).
                cur_idx = piece_to_index(current)
                nxt_idx = piece_to_index(next_piece)
                scores = scorer.score_batch(board, candidates,
                                            current_piece_idx=cur_idx,
                                            next_piece_idx=nxt_idx)
            else:
                scores = [scorer.score(board, c) for c in candidates]
            best_i = max(range(len(scores)), key=lambda i: scores[i])
            cand = candidates[best_i]

        # Record the state BEFORE the lock + the heuristic reward AFTER.
        board_tensor = board_to_tensor(board)
        cur_idx = piece_to_index(current)
        nxt_idx = piece_to_index(next_piece)
        h_val = 0.0
        if heuristic_scorer is not None:
            h_val = heuristic_scorer.score(board, cand)
        trajectory.append((board_tensor, cur_idx, nxt_idx, h_val))

        board.lock(cand.piece)
        board.clear_lines()
        piece_queue.advance()  # next-piece becomes current
        pieces_placed += 1

    # Label each state with its survival length.
    n = len(trajectory)
    samples: list[tuple[torch.Tensor, int, int, float, float]] = []
    for i, (bt, ci, ni, hv) in enumerate(trajectory):
        steps_after = n - i
        samples.append((bt, ci, ni, hv, float(steps_after)))
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


# ---- MC dataset generator (#1: reward shaping) ---------------------------

def generate_selfplay_dataset(
    scorer: Scorer,
    n_games: int,
    rng: random.Random | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    max_pieces: int = 1500,
    survival_weight: float = 0.7,
    heuristic_weight: float = 0.3,
    heuristic_scorer: Optional[HeuristicScorer] = None,
) -> list[tuple[torch.Tensor, int, int, float]]:
    """Play n_games and return (board_tensor, cur_idx, next_idx, target)
    samples for training.

    The target is a blend of discounted survival and the heuristic reward
    (#1 reward shaping):
        target = survival_weight * discounted_survival(steps, gamma)
                + heuristic_weight * heuristic_reward
    """
    rng = rng or random.Random()
    samples: list[tuple[torch.Tensor, int, int, float]] = []
    for g in range(n_games):
        game = play_game(
            scorer, rng,
            epsilon=epsilon, max_pieces=max_pieces,
            heuristic_scorer=heuristic_scorer,
        )
        for bt, ci, ni, hv, steps in game:
            surv = discounted_survival(int(steps), gamma)
            target = survival_weight * surv + heuristic_weight * hv
            samples.append((bt, ci, ni, target))
    return samples


# ---- MC training loop (#2: replay buffer) ---------------------------------

def train_selfplay(
    n_rounds: int = 5,
    games_per_round: int = 100,
    epochs_per_round: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    survival_weight: float = 0.7,
    heuristic_weight: float = 0.3,
    replay_capacity: int = 50_000,
    seed: int = 42,
    weights_in: Optional[str] = None,
    weights_out: str = "model_weights.pt",
    device: str = "cpu",
) -> TAPNet:
    """Train TAPNet via MC self-play with replay buffer + reward shaping.

    Each round:
      1. Play `games_per_round` games with the current network (epsilon-greedy).
         Collect (state, cur_idx, next_idx, blended_target) samples.
      2. Add samples to the replay buffer.
      3. Train for `epochs_per_round` epochs, sampling minibatches from the
         *entire buffer* (not just this round's samples).

    If `weights_in` is given, load those weights as the starting point.
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

    # Replay buffer (#2) + heuristic scorer for reward shaping (#1).
    replay = ReplayBuffer(capacity=replay_capacity)
    heuristic_scorer = HeuristicScorer() if heuristic_weight > 0 else None

    for round_idx in range(n_rounds):
        # --- Self-play phase ---
        net.eval()
        scorer = NeuralScorer(net=net, device=device)
        cur_epsilon = epsilon * (0.9 ** round_idx)
        print(f"\n=== Round {round_idx + 1}/{n_rounds} (epsilon={cur_epsilon:.3f}) ===")
        print(f"  Playing {games_per_round} games...")
        new_samples = generate_selfplay_dataset(
            scorer, games_per_round, rng,
            epsilon=cur_epsilon, gamma=gamma,
            survival_weight=survival_weight,
            heuristic_weight=heuristic_weight,
            heuristic_scorer=heuristic_scorer,
        )
        if not new_samples:
            print("  No samples collected (all games topped out immediately?)")
            continue

        # Add to replay buffer (#2).
        replay.extend(new_samples)

        # Stats.
        targets_list = [s[3] for s in new_samples]
        print(f"  Collected {len(new_samples)} new samples. "
              f"Replay buffer: {len(replay)} samples.")
        print(f"  Target stats: min={min(targets_list):.1f} "
              f"max={max(targets_list):.1f} "
              f"mean={sum(targets_list)/len(targets_list):.1f}")

        # --- Training phase: sample from the ENTIRE replay buffer (#2) ---
        n = len(replay)
        net.train()
        steps_per_epoch = max(1, n // batch_size)
        for epoch in range(epochs_per_round):
            epoch_loss = 0.0
            n_batches = 0
            for _ in range(steps_per_epoch):
                batch = replay.sample(batch_size, rng)
                if not batch:
                    break
                boards = torch.stack([s[0] for s in batch]).to(device)
                cur_idxs = torch.tensor([s[1] for s in batch], dtype=torch.long).to(device)
                next_idxs = torch.tensor([s[2] for s in batch], dtype=torch.long).to(device)
                t = torch.tensor([s[3] for s in batch], dtype=torch.float32).to(device).unsqueeze(1)
                optimizer.zero_grad()
                pred = net(boards, cur_idxs, next_idxs)  # [B, 1]
                loss = loss_fn(pred, t)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg = epoch_loss / max(1, n_batches)
            print(f"    epoch {epoch + 1}/{epochs_per_round}: loss={avg:.4f} "
                  f"({n_batches} batches from buffer of {len(replay)})")

        # Checkpoint: save weights after each round so an interrupted run
        # preserves progress. Overwrites the output file each round.
        out_path = HERE.parent / weights_out
        torch.save(net.state_dict(), out_path)
        print(f"  Checkpointed weights to {out_path} (after round {round_idx + 1})")

    print(f"\nSaved final weights to {HERE.parent / weights_out}")
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TAPNet via MC self-play with replay buffer + reward shaping."
    )
    parser.add_argument("--rounds", type=int, default=5, help="number of self-play rounds")
    parser.add_argument("--games", type=int, default=100, help="games per round")
    parser.add_argument("--epochs", type=int, default=3, help="epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="exploration rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="survival discount factor")
    parser.add_argument("--survival-weight", type=float, default=0.7,
                        help="weight for survival in blended target (#1)")
    parser.add_argument("--heuristic-weight", type=float, default=0.3,
                        help="weight for heuristic reward in blended target (#1)")
    parser.add_argument("--replay-capacity", type=int, default=50_000,
                        help="max replay buffer size (#2)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--weights-in", type=str, default=None, help="initial weights to load")
    parser.add_argument("--weights-out", type=str, default="model_weights.pt",
                        help="output weights path")
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
        survival_weight=args.survival_weight,
        heuristic_weight=args.heuristic_weight,
        replay_capacity=args.replay_capacity,
        seed=args.seed,
        weights_in=args.weights_in,
        weights_out=args.weights_out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
