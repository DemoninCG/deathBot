"""Monte Carlo self-play training with replay buffer + reward shaping (phase 2.3+).

Three improvements over the original phase-2.3 self-play:

  1. **Reward shaping** (#1): frozen z-score normalization puts discounted
      survival and heuristic afterstate quality on comparable scales before
      blending them.
     The heuristic gives a dense per-step signal (survival alone is sparse);
     survival gives the true long-term outcome. Default 0.7 survival + 0.3
     heuristic.

  2. **Recent experience replay** (#2): raw reward components and policy
      generation are retained. Stale Monte Carlo labels are expired, and each
      batch deliberately includes samples from the newest policy.

  3. **Consistent afterstates** (#3): inference and training both encode the
      post-lock, post-line-clear board with the placed and preview pieces.

Run:
    cd python; python selfplay.py --rounds 10 --games 200 --epochs 3 --weights-in model_weights_snapshot1.pt
    cd python; python selfplay.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

# Make the python/ directory importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deathbot_sim as ds
from model import TAPNet, piece_to_index, PIECE_INDEX
from neural_scorer import NeuralScorer
from heuristic import HeuristicScorer
from scorer import Scorer
from piece_history import PieceQueue
from afterstate import make_afterstate


ALL_PIECES = list(PIECE_INDEX.keys())  # [I, O, T, S, Z, J, L]


# ---- Replay buffer (#2) ---------------------------------------------------

@dataclass(frozen=True)
class ReplaySample:
    board: torch.Tensor
    current_idx: int
    next_idx: int
    survival: float
    heuristic: float
    generation: int = 0

    def __iter__(self):
        """Preserve the historical four-field sample API for callers/tests."""
        yield self.board
        yield self.current_idx
        yield self.next_idx
        yield self.survival

    def __getitem__(self, index: int):
        return tuple(self)[index]


@dataclass(frozen=True)
class RewardNormalizer:
    """Frozen z-score statistics used to put reward components on equal scales."""

    survival_mean: float
    survival_std: float
    heuristic_mean: float
    heuristic_std: float

    @classmethod
    def fit(cls, samples: Sequence[ReplaySample]) -> "RewardNormalizer":
        if not samples:
            raise ValueError("cannot fit reward normalization without samples")

        def stats(values: list[float]) -> tuple[float, float]:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            return mean, max(math.sqrt(variance), 1e-6)

        sm, ss = stats([s.survival for s in samples])
        hm, hs = stats([s.heuristic for s in samples])
        return cls(sm, ss, hm, hs)

    def blend(self, sample: ReplaySample, survival_weight: float,
              heuristic_weight: float) -> float:
        total_weight = survival_weight + heuristic_weight
        if total_weight <= 0:
            raise ValueError("survival and heuristic weights cannot both be zero")
        sw = survival_weight / total_weight
        hw = heuristic_weight / total_weight
        survival = (sample.survival - self.survival_mean) / self.survival_std
        heuristic = (sample.heuristic - self.heuristic_mean) / self.heuristic_std
        return sw * survival + hw * heuristic


NORMALIZER_FORMAT_VERSION = 1


def normalizer_sidecar_path(weights_path: Path) -> Path:
    """Return the metadata path paired with a model-weight checkpoint."""
    return weights_path.with_name(f"{weights_path.name}.normalizer.json")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_config(gamma: float, survival_weight: float,
                   heuristic_weight: float) -> dict[str, float]:
    return {
        "gamma": gamma,
        "survival_weight": survival_weight,
        "heuristic_weight": heuristic_weight,
    }


def save_normalizer_checkpoint(path: Path, normalizer: RewardNormalizer, *,
                               weights_path: Path, gamma: float,
                               survival_weight: float,
                               heuristic_weight: float) -> None:
    """Save target-scale metadata that must accompany a resumable checkpoint."""
    payload = {
        "format_version": NORMALIZER_FORMAT_VERSION,
        "weights_sha256": _file_sha256(weights_path),
        "target_config": _target_config(
            gamma, survival_weight, heuristic_weight),
        "normalizer": {
            "survival_mean": normalizer.survival_mean,
            "survival_std": normalizer.survival_std,
            "heuristic_mean": normalizer.heuristic_mean,
            "heuristic_std": normalizer.heuristic_std,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_normalizer_checkpoint(path: Path, *, weights_path: Path, gamma: float,
                               survival_weight: float,
                               heuristic_weight: float) -> RewardNormalizer:
    """Load and validate the normalizer paired with ``weights_path``.

    The target configuration is deliberately immutable when resuming. Changing
    it changes the meaning of a trained network's output and can destabilize
    ranking just as surely as replacing the normalizer itself.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read normalizer checkpoint {path}: {exc}") from exc
    if payload.get("format_version") != NORMALIZER_FORMAT_VERSION:
        raise ValueError(f"unsupported normalizer checkpoint format: {path}")
    if payload.get("weights_sha256") != _file_sha256(weights_path):
        raise ValueError(
            f"normalizer checkpoint {path} does not match weights {weights_path}")
    expected = _target_config(gamma, survival_weight, heuristic_weight)
    if payload.get("target_config") != expected:
        raise ValueError(
            "target configuration differs from the checkpoint metadata; "
            "keep gamma and reward weights unchanged when resuming")
    try:
        data = payload["normalizer"]
        return RewardNormalizer(
            survival_mean=float(data["survival_mean"]),
            survival_std=float(data["survival_std"]),
            heuristic_mean=float(data["heuristic_mean"]),
            heuristic_std=float(data["heuristic_std"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid normalizer checkpoint: {path}") from exc


class ReplayBuffer:
    """A FIFO replay buffer that stores (board, cur_idx, next_idx, target)
    samples across all rounds. Training minibatches are sampled uniformly.

    The buffer has a max capacity; old samples are evicted FIFO when full.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, board: torch.Tensor, cur_idx: int, next_idx: int,
             target: float, generation: int = 0) -> None:
        """Add a sample to the buffer."""
        self.buffer.append(ReplaySample(board, cur_idx, next_idx, target, 0.0,
                                        generation))

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        """Add many samples at once."""
        for s in samples:
            if isinstance(s, ReplaySample):
                self.buffer.append(s)
            else:
                board, cur_idx, next_idx, target = s
                self.push(board, cur_idx, next_idx, target)

    def sample(self, batch_size: int, rng: random.Random | None = None,
               *, generation: int | None = None, recent_fraction: float = 0.5
               ) -> list[ReplaySample]:
        """Sample `batch_size` items uniformly at random.

        Samples random indices directly (O(batch_size)) instead of copying
        the entire buffer to a list (which was O(buffer_size) per batch and
        caused severe slowdowns as the buffer grew toward capacity).
        """
        if not self.buffer:
            return []
        rng = rng or random.Random()
        n = min(batch_size, len(self.buffer))
        if generation is None:
            return rng.sample(list(self.buffer), n)
        recent = [s for s in self.buffer if s.generation == generation]
        older = [s for s in self.buffer if s.generation != generation]
        recent_n = min(len(recent), round(n * recent_fraction))
        older_n = min(len(older), n - recent_n)
        recent_n = min(len(recent), n - older_n)
        return rng.sample(recent, recent_n) + rng.sample(older, older_n)

    def discard_before(self, generation: int) -> None:
        """Discard stale Monte Carlo labels generated before ``generation``."""
        self.buffer = deque(
            (s for s in self.buffer if s.generation >= generation),
            maxlen=self.capacity,
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---- Game rollout engine (#3: current piece tracking) ---------------------

def select_candidate_index(scores: Sequence[float], rng: random.Random, *,
                           epsilon: float, top_k: int = 5) -> int:
    """Select greedily, or explore uniformly among the strongest ``top_k``."""
    if not scores:
        raise ValueError("cannot select from an empty score sequence")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    finite = [i for i, score in enumerate(scores) if math.isfinite(score)]
    if not finite:
        raise ValueError("all candidate scores are non-finite")
    ranked = sorted(finite, key=lambda i: (-scores[i], i))
    if rng.random() < epsilon:
        return rng.choice(ranked[:top_k])
    return ranked[0]

def play_game(
    scorer: Scorer,
    rng: random.Random | None = None,
    *,
    epsilon: float = 0.1,
    exploration_top_k: int = 5,
    max_pieces: int = 1500,
    heuristic_scorer: Optional[HeuristicScorer] = None,
) -> tuple[list[tuple[torch.Tensor, int, int, float, float]], int]:
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
    total_lines_cleared = 0
    while pieces_placed < max_pieces:
        current = piece_queue.current
        next_piece = piece_queue.next_piece
        candidates = board.search_locks(current)
        if not candidates:
            break  # topped out

        cur_idx = piece_to_index(current)
        nxt_idx = piece_to_index(next_piece)
        if hasattr(scorer, "score_batch"):
            scores = scorer.score_batch(board, candidates,
                                        current_piece_idx=cur_idx,
                                        next_piece_idx=nxt_idx)
        else:
            scores = [scorer.score(board, c) for c in candidates]
        cand = candidates[select_candidate_index(
            scores, rng, epsilon=epsilon, top_k=exploration_top_k)]

        # Construct the actual next state once and use it for both training
        # and game progression.
        afterstate = make_afterstate(board, cand)
        h_val = 0.0
        if heuristic_scorer is not None:
            h_val = heuristic_scorer.score_afterstate(
                afterstate.board, afterstate.lines_cleared)
        trajectory.append((afterstate.tensor, cur_idx, nxt_idx, h_val))

        board = afterstate.board
        total_lines_cleared += afterstate.lines_cleared
        piece_queue.advance()  # next-piece becomes current
        pieces_placed += 1

    # Label each state with its survival length.
    n = len(trajectory)
    samples: list[tuple[torch.Tensor, int, int, float, float]] = []
    for i, (bt, ci, ni, hv) in enumerate(trajectory):
        steps_after = n - i
        samples.append((bt, ci, ni, hv, float(steps_after)))
    return samples, total_lines_cleared


def discounted_survival(steps: int, gamma: float = 0.99) -> float:
    """Convert a raw step count into a discounted survival value.

    target = sum_{t=0}^{steps-1} gamma^t = (1 - gamma^steps) / (1 - gamma)

    With gamma=1.0, this is just `steps` (raw count). With gamma<1, near-term
    survival is weighted more, reducing variance from late-game noise.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if gamma == 1.0:
        return float(steps)
    if steps == 0:
        return 0.0
    return (1.0 - gamma ** steps) / (1.0 - gamma)


# ---- MC dataset generator (#1: reward shaping) ---------------------------

def generate_selfplay_dataset(
    scorer: Scorer,
    n_games: int,
    rng: random.Random | None = None,
    *,
    epsilon: float = 0.1,
    exploration_top_k: int = 5,
    gamma: float = 0.99,
    max_pieces: int = 1500,
    survival_weight: float = 0.7,
    heuristic_weight: float = 0.3,
    heuristic_scorer: Optional[HeuristicScorer] = None,
    game_stats_out: Optional[list] = None,
    generation: int = 0,
) -> list[ReplaySample]:
    """Play games and retain separate raw reward components for normalization."""
    rng = rng or random.Random()
    samples: list[ReplaySample] = []
    for g in range(n_games):
        game, lines = play_game(
            scorer, rng,
            epsilon=epsilon, exploration_top_k=exploration_top_k,
            max_pieces=max_pieces,
            heuristic_scorer=heuristic_scorer,
        )
        if game_stats_out is not None:
            game_stats_out.append((len(game), lines))
        for bt, ci, ni, hv, steps in game:
            surv = discounted_survival(int(steps), gamma)
            samples.append(ReplaySample(bt, ci, ni, surv, hv, generation))
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
    replay_max_age: int = 2,
    replay_recent_fraction: float = 0.5,
    exploration_top_k: int = 5,
    seed: int = 42,
    weights_in: Optional[str] = None,
    weights_out: str = "model_weights.pt",
    device: str = "cpu",
    allow_legacy_weights: bool = False,
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
    normalizer: RewardNormalizer | None = None
    if weights_in is not None:
        p = HERE.parent / weights_in if not Path(weights_in).is_absolute() else Path(weights_in)
        if p.exists():
            net.load_state_dict(torch.load(p, map_location=device))
            print(f"Loaded initial weights from {p}")
            metadata_path = normalizer_sidecar_path(p)
            if not metadata_path.is_file():
                if not allow_legacy_weights:
                    raise ValueError(
                        f"cannot safely resume {p}: missing normalizer metadata "
                        f"{metadata_path}. To deliberately bootstrap a new target scale "
                        "from legacy weights, pass --allow-legacy-weights.")
                print("No normalizer metadata found; bootstrapping a new target "
                      "scale from legacy weights.")
            else:
                normalizer = load_normalizer_checkpoint(
                    metadata_path, weights_path=p, gamma=gamma,
                    survival_weight=survival_weight,
                    heuristic_weight=heuristic_weight)
                print(f"Loaded reward normalization from {metadata_path}")
        else:
            print(f"Weights file {p} not found; starting from scratch.")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

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
        game_stats: list[tuple[int, int]] = []
        new_samples = generate_selfplay_dataset(
            scorer, games_per_round, rng,
            epsilon=cur_epsilon, exploration_top_k=exploration_top_k,
            gamma=gamma,
            survival_weight=survival_weight,
            heuristic_weight=heuristic_weight,
            heuristic_scorer=heuristic_scorer,
            game_stats_out=game_stats,
            generation=round_idx,
        )
        if not new_samples:
            print("  No samples collected (all games topped out immediately?)")
            continue

        # Freeze normalization from the first rollout batch so every replay
        # generation retains the same target scale.
        if normalizer is None:
            normalizer = RewardNormalizer.fit(new_samples)
            print("  Reward normalization — "
                  f"survival mean={normalizer.survival_mean:.2f} "
                  f"std={normalizer.survival_std:.2f}; "
                  f"heuristic mean={normalizer.heuristic_mean:.2f} "
                  f"std={normalizer.heuristic_std:.2f}")

        # Keep only recent behavior-policy generations; old MC outcomes are
        # stale labels for the improving policy.
        replay.extend(new_samples)
        replay.discard_before(max(0, round_idx - replay_max_age + 1))

        # Survival stats (primary performance indicator).
        pieces_list = [p for p, _ in game_stats]
        lines_list  = [l for _, l in game_stats]
        print(f"  Collected {len(new_samples)} samples from {len(game_stats)} games. "
              f"Replay buffer: {len(replay)}.")
        print(f"  Survival  — pieces: min={min(pieces_list)} "
              f"avg={sum(pieces_list)/len(pieces_list):.1f} "
              f"max={max(pieces_list)} | "
              f"lines: avg={sum(lines_list)/len(lines_list):.1f} "
              f"max={max(lines_list)}")
        # Target stats (training signal quality).
        targets_list = [normalizer.blend(
            s, survival_weight, heuristic_weight) for s in new_samples]
        print(f"  Targets   — min={min(targets_list):.2f} "
              f"avg={sum(targets_list)/len(targets_list):.2f} "
              f"max={max(targets_list):.2f} "
              f"(survival*{survival_weight}+heuristic*{heuristic_weight})")

        # --- Training phase: sample from the ENTIRE replay buffer (#2) ---
        n = len(replay)
        net.train()
        steps_per_epoch = max(1, n // batch_size)
        for epoch in range(epochs_per_round):
            epoch_loss = 0.0
            n_batches = 0
            for _ in range(steps_per_epoch):
                batch = replay.sample(
                    batch_size, rng, generation=round_idx,
                    recent_fraction=replay_recent_fraction)
                if not batch:
                    break
                boards = torch.stack([s.board for s in batch]).to(device)
                cur_idxs = torch.tensor(
                    [s.current_idx for s in batch], dtype=torch.long,
                    device=device)
                next_idxs = torch.tensor(
                    [s.next_idx for s in batch], dtype=torch.long,
                    device=device)
                targets = [normalizer.blend(
                    s, survival_weight, heuristic_weight) for s in batch]
                t = torch.tensor(
                    targets, dtype=torch.float32, device=device).unsqueeze(1)
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
        assert normalizer is not None
        metadata_path = normalizer_sidecar_path(out_path)
        save_normalizer_checkpoint(
            metadata_path, normalizer, weights_path=out_path, gamma=gamma,
            survival_weight=survival_weight,
            heuristic_weight=heuristic_weight)
        print(f"  Checkpointed weights to {out_path} (after round {round_idx + 1})")
        print(f"  Checkpointed normalization to {metadata_path}")

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
    parser.add_argument("--exploration-top-k", type=int, default=5,
                        help="explore only among this many strongest candidates")
    parser.add_argument("--gamma", type=float, default=0.99, help="survival discount factor")
    parser.add_argument("--survival-weight", type=float, default=0.7,
                        help="weight for survival in blended target (#1)")
    parser.add_argument("--heuristic-weight", type=float, default=0.3,
                        help="weight for heuristic reward in blended target (#1)")
    parser.add_argument("--replay-capacity", type=int, default=50_000,
                        help="max replay buffer size (#2)")
    parser.add_argument("--replay-max-age", type=int, default=2,
                        help="number of recent self-play rounds retained")
    parser.add_argument("--replay-recent-fraction", type=float, default=0.5,
                        help="fraction of each batch drawn from the newest round")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--weights-in", type=str, default=None, help="initial weights to load")
    parser.add_argument("--weights-out", type=str, default="model_weights.pt",
                        help="output weights path")
    parser.add_argument("--allow-legacy-weights", action="store_true",
                        help="bootstrap new normalization from weights without metadata; "
                             "only use for pre-sidecar legacy checkpoints")
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
        replay_max_age=args.replay_max_age,
        replay_recent_fraction=args.replay_recent_fraction,
        exploration_top_k=args.exploration_top_k,
        seed=args.seed,
        weights_in=args.weights_in,
        weights_out=args.weights_out,
        device=args.device,
        allow_legacy_weights=args.allow_legacy_weights,
    )


if __name__ == "__main__":
    main()
