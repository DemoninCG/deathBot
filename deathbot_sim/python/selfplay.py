"""Monte Carlo self-play training with scale-matched reward shaping.

2026-09 fix for the catastrophic-resume regression (snapshot7: 503 pieces ->
~30 after one round):

  * **Scale-matched targets**: the old frozen z-score normalizer emitted
    targets in roughly [-3, +1.5], while a warm-started network's value
    function lived on a [-300, +35] scale. Regression toward the tiny target
    range crushed the value function in a single round. Targets are still
    z-scored for component balance (fitted once from the first rollout), but
    are then re-anchored onto the network's own output distribution:
        target = anchor_mean + anchor_std * z_blend
    The anchor is measured once from the loaded network on real afterstate
    boards, so a warm-started network keeps its output scale and training
    only reshapes the *ranking*. From-scratch networks get an anchor std
    floor of 1.0. The anchor is re-measured at the start of each run, which
    keeps it matched to whatever checkpoint is being resumed.
  * **Replay amnesia off by default**: the old `discard_before` expiry could
    shrink the buffer to only the newest (near-death) boards once games got
    short, starving training of opening states and accelerating collapse.
    `--replay-max-age 0` (now the default) keeps all generations; FIFO
    capacity eviction still bounds staleness.
  * **Per-round guardrails**: a quick greedy evaluation after every round
    (plus a baseline eval before round 1) makes collapse immediately visible
    in the log. The best-so-far weights are kept in
    `<weights_out>.best.pt`, and `--rollback-factor` restores the best
    weights into the live network when a round's eval collapses.
  * **Optional conv freezing**: `--freeze-conv N` freezes the convolutional
    feature extractor for the first N rounds, limiting forgetting when
    adapting a pre-afterstate checkpoint (snapshot1-7 era) to the afterstate
    representation.

Run:
    cd python; python selfplay.py --rounds 10 --games 200 --epochs 3 --weights-in model_weights.pt
    cd python; python selfplay.py --help
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
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


# ---- Scale-matched targets (2026-09 collapse fix) -------------------------

def measure_output_scale(net, samples: Sequence[ReplaySample],
                         device: str = "cpu", max_samples: int = 512,
                         ) -> tuple[float, float]:
    """Mean/std of the network's outputs on a subsample of replay samples.

    Used to anchor z-scored targets onto the value-function scale the network
    already uses, so resuming a trained checkpoint does not crush its outputs
    toward a tiny target range (the snapshot7 -> snapshot8 failure mode).
    """
    if not samples:
        raise ValueError("cannot measure output scale without samples")
    stride = max(1, len(samples) // max_samples)
    chosen = list(samples[::stride])[:max_samples]
    boards = torch.stack([s.board for s in chosen]).to(device)
    cur = torch.tensor([s.current_idx for s in chosen], dtype=torch.long,
                       device=device)
    nxt = torch.tensor([s.next_idx for s in chosen], dtype=torch.long,
                       device=device)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        out = net(boards, cur, nxt).flatten()
    if was_training:
        net.train()
    return float(out.mean()), float(out.std(unbiased=False))


def anchored_target(sample: ReplaySample, normalizer: RewardNormalizer,
                    anchor_mean: float, anchor_std: float,
                    survival_weight: float, heuristic_weight: float) -> float:
    """Blend z-scored reward components, then re-anchor to the net's scale."""
    z = normalizer.blend(sample, survival_weight, heuristic_weight)
    return anchor_mean + anchor_std * z


def quick_eval(net: TAPNet, n_games: int, seed: int, device: str = "cpu",
               max_pieces: int = 1500) -> list[int]:
    """Greedy self-play evaluation on independent fixed seeds.

    Returns pieces survived per game. Uses its own RNG stream so training
    rollouts are unaffected. Used as the per-round collapse guardrail.
    """
    was_training = net.training
    net.eval()
    scorer = NeuralScorer(net=net, device=device)
    pieces: list[int] = []
    for i in range(n_games):
        trajectory, _ = play_game(scorer, random.Random(seed + i),
                                  epsilon=0.0, max_pieces=max_pieces)
        pieces.append(len(trajectory))
    if was_training:
        net.train()
    return pieces


class ReplayBuffer:
    """A FIFO replay buffer that stores (board, cur_idx, next_idx, target)
    samples across all rounds.

    Sampling is O(batch): per-generation index lists are cached when the
    buffer changes and minibatches are drawn by index. (The 2026-09-04 code
    rebuilt the recent/older index lists by scanning the entire deque for
    EVERY minibatch, which dominated wall-clock once the buffer grew.)
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self._gen_index: dict[int, list[int]] = {}
        self._index_dirty = True

    def _rebuild_index(self) -> None:
        self._gen_index = {}
        for i, s in enumerate(self.buffer):
            self._gen_index.setdefault(s.generation, []).append(i)
        self._index_dirty = False

    def push(self, board: torch.Tensor, cur_idx: int, next_idx: int,
             target: float, generation: int = 0) -> None:
        """Add a sample to the buffer."""
        self.buffer.append(ReplaySample(board, cur_idx, next_idx, target, 0.0,
                                        generation))
        self._index_dirty = True

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        """Add many samples at once."""
        for s in samples:
            if isinstance(s, ReplaySample):
                self.buffer.append(s)
            else:
                board, cur_idx, next_idx, target = s
                self.push(board, cur_idx, next_idx, target)
        self._index_dirty = True

    def _all_indices(self) -> list[int]:
        if self._index_dirty:
            self._rebuild_index()
        return [i for idxs in self._gen_index.values() for i in idxs]

    def sample(self, batch_size: int, rng: random.Random | None = None,
               *, generation: int | None = None, recent_fraction: float = 0.5
               ) -> list[ReplaySample]:
        """Sample `batch_size` items, half from the newest generation and
        half from the rest (when ``generation`` is given), by cached index.

        When ``generation`` is None the batch is drawn uniformly from the
        whole buffer (O(batch), no full-buffer scan).
        """
        if not self.buffer:
            return []
        rng = rng or random.Random()
        n = min(batch_size, len(self.buffer))
        if generation is None:
            return [self.buffer[i] for i in rng.sample(self._all_indices(), n)]
        if self._index_dirty:
            self._rebuild_index()
        recent = self._gen_index.get(generation, [])
        older = [i for g, idxs in self._gen_index.items()
                 if g != generation for i in idxs]
        recent_n = min(len(recent), round(n * recent_fraction))
        older_n = min(len(older), n - recent_n)
        recent_n = min(len(recent), n - older_n)
        picked = rng.sample(recent, recent_n) + rng.sample(older, older_n)
        return [self.buffer[i] for i in picked]

    def discard_before(self, generation: int) -> None:
        """Discard stale Monte Carlo labels generated before ``generation``."""
        self.buffer = deque(
            (s for s in self.buffer if s.generation >= generation),
            maxlen=self.capacity,
        )
        self._index_dirty = True

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


@dataclass(frozen=True)
class GameResult:
    """One rollout with explicit terminal semantics (A2, 2026-09).

    ``trajectory`` rows are (board_tensor, cur_idx, next_idx, heuristic_val).
    The final stored state is the last committed afterstate; ``steps_after``
    labels (see ``play_game``) count placements including that one.

    ``topout`` distinguishes real game-over (no reachable lock, or the chosen
    placement would lock a cell above the field) from artificial truncation
    at ``max_pieces``. Historical code labeled both identically.
    """

    trajectory: list[tuple[torch.Tensor, int, int, float]]
    lines_cleared: int
    pieces_placed: int
    topout: bool
    capped: bool
    reason: str  # "no_candidates" | "lockout" | "max_pieces"

    @property
    def survived_pieces(self) -> int:
        """Successful placements (excludes a losing final lockout attempt)."""
        return self.pieces_placed


def play_game_detailed(
    scorer: Scorer,
    rng: random.Random | None = None,
    *,
    epsilon: float = 0.1,
    exploration_top_k: int = 5,
    max_pieces: int = 1500,
    heuristic_scorer: Optional[HeuristicScorer] = None,
    legacy_lockout: bool = False,
    explore_rng: random.Random | None = None,
) -> GameResult:
    """Play one Death-mode game greedily, with explicit terminal semantics.

    ``trajectory`` rows are (board_tensor, cur_idx, next_idx, heuristic_val)
    for each committed placement; the last row is the final committed
    afterstate.

    Transition semantics (2026-09 phase A2):
      * ``legacy_lockout=False`` (corrected default): candidates whose lock
        would put a cell above the field are LOSING placements. They are
        masked out of selection; if every candidate loses, the game ends
        with ``topout=True`` and the piece is NOT committed. Reaching
        ``max_pieces`` is an artificial truncation (``topout=False``).
      * ``legacy_lockout=True``: historical behavior — all reachable
        candidates are selectable, cells above the field are silently
        discarded on lock, and the game continues. Evaluation of legacy
        checkpoints uses this to reproduce the historical benchmark.

    ``reason`` is one of "no_candidates", "lockout", "max_pieces".
    """
    rng = rng or random.Random()
    explore_rng = explore_rng or rng  # same stream unless caller separates
    board = ds.Board()
    piece_queue = PieceQueue(rng)

    # Each trajectory entry: (board_tensor, cur_idx, next_idx, heuristic_val)
    trajectory: list[tuple[torch.Tensor, int, int, float]] = []

    pieces_placed = 0
    total_lines_cleared = 0
    topout = False
    reason = "max_pieces"
    while pieces_placed < max_pieces:
        current = piece_queue.current
        next_piece = piece_queue.next_piece
        cur_idx = piece_to_index(current)
        nxt_idx = piece_to_index(next_piece)
        candidates = board.search_locks(current)
        if not candidates:
            topout = True
            reason = "no_candidates"
            break  # topped out

        if hasattr(scorer, "score_batch"):
            scores = scorer.score_batch(board, candidates,
                                        current_piece_idx=cur_idx,
                                        next_piece_idx=nxt_idx)
        else:
            scores = [scorer.score(board, c) for c in candidates]

        if legacy_lockout:
            # Historical behavior: choose over ALL reachable candidates.
            cand_idx = select_candidate_index(
                scores, explore_rng, epsilon=epsilon, top_k=exploration_top_k)
            cand = candidates[cand_idx]
        else:
            # Phase A2: a lockout placement ends the game, so mask those
            # candidates out of selection instead of silently dropping
            # their above-field cells and continuing.
            viable_idx = [i for i, c in enumerate(candidates)
                          if not board.locks_out(c.piece)]
            if not viable_idx:
                topout = True
                reason = "lockout"
                break
            if len(viable_idx) < len(candidates):
                scores = [scores[i] for i in viable_idx]
            cand_idx = select_candidate_index(
                scores, explore_rng, epsilon=epsilon, top_k=exploration_top_k)
            cand = candidates[viable_idx[cand_idx]]

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

    return GameResult(
        trajectory=trajectory,
        lines_cleared=total_lines_cleared,
        pieces_placed=pieces_placed,
        topout=topout,
        capped=pieces_placed >= max_pieces,
        reason=reason,
    )


def play_game(
    scorer: Scorer,
    rng: random.Random | None = None,
    *,
    epsilon: float = 0.1,
    exploration_top_k: int = 5,
    max_pieces: int = 1500,
    heuristic_scorer: Optional[HeuristicScorer] = None,
    legacy_lockout: bool = False,
    explore_rng: random.Random | None = None,
) -> tuple[list[tuple[torch.Tensor, int, int, float, float]], int]:
    """Play one full Death-mode game greedily with the scorer.

    Compatibility wrapper around :func:`play_game_detailed` returning the
    historical (samples, total_lines) tuple:

        samples: (board_tensor, current_idx, next_idx, heuristic_reward,
        steps_survived) per committed placement. The caller blends
        heuristic_reward + discounted_survival into the final target.

    With the corrected (default) lockout semantics the piece count reported
    by evaluators excludes any losing final lockout attempt, and reaching
    ``max_pieces`` keeps ``capped`` semantics identical to before.
    """
    result = play_game_detailed(
        scorer, rng,
        epsilon=epsilon, exploration_top_k=exploration_top_k,
        max_pieces=max_pieces, heuristic_scorer=heuristic_scorer,
        legacy_lockout=legacy_lockout, explore_rng=explore_rng,
    )

    # Label each state with its survival length.
    n = len(result.trajectory)
    samples: list[tuple[torch.Tensor, int, int, float, float]] = []
    for i, (bt, ci, ni, hv) in enumerate(result.trajectory):
        steps_after = n - i
        samples.append((bt, ci, ni, hv, float(steps_after)))
    return samples, result.lines_cleared


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
    elite_fraction: float = 1.0,
    explore_rng: random.Random | None = None,
) -> list[ReplaySample]:
    """Play games and retain separate raw reward components for normalization.

    With ``elite_fraction < 1``, only samples from the strongest games (top
    fraction by pieces survived) are returned. A trained network's own MC
    labels regress toward its *behavior* policy's outcomes — which include
    fatal exploration accidents — so at a high skill level only elite games
    carry improvement signal (self-imitation). Default keeps everything.
    """
    rng = rng or random.Random()
    per_game: list[tuple[float, list[ReplaySample]]] = []
    for g in range(n_games):
        game, lines = play_game(
            scorer, rng,
            epsilon=epsilon, exploration_top_k=exploration_top_k,
            max_pieces=max_pieces,
            heuristic_scorer=heuristic_scorer,
            explore_rng=explore_rng,
        )
        if game_stats_out is not None:
            game_stats_out.append((len(game), lines))
        game_samples: list[ReplaySample] = []
        for bt, ci, ni, hv, steps in game:
            surv = discounted_survival(int(steps), gamma)
            game_samples.append(ReplaySample(bt, ci, ni, surv, hv, generation))
        per_game.append((float(len(game)), game_samples))

    if elite_fraction >= 1.0 or not per_game:
        return [s for _, samples in per_game for s in samples]
    ranked = sorted(per_game, key=lambda entry: -entry[0])
    keep = max(1, round(len(ranked) * elite_fraction))
    kept = [s for _, samples in ranked[:keep] for s in samples]
    print(f"  Elite filter: kept {keep}/{len(ranked)} games "
          f"(survival >= {ranked[keep - 1][0]:.0f} pieces), "
          f"{len(kept)} samples")
    return kept


# ---- Parallel rollout workers (phase D4, 2026-09) --------------------------

def _rollout_worker(payload) -> tuple[int, list, int]:
    """Play a chunk of games with a frozen behavior snapshot. Process-pool
    entry point: each worker loads the weights ONCE, then plays its shard.

    Payload: (weights_path, game_seeds, epsilon, max_pieces, top_k,
              surv_w, heur_w, gamma, generation, legacy_lockout).
    Returns (shard_id, per_game samples list, total lines).
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    (weights_path, seeds, epsilon, max_pieces, top_k, surv_w, heur_w,
     gamma, generation, legacy_lockout) = payload
    sys.path.insert(0, str(HERE))
    from model import TAPNet  # noqa: F401  (re-imported for spawn safety)
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()
    scorer = NeuralScorer(net=net, device="cpu")
    heur = HeuristicScorer() if heur_w > 0 else None

    out: list[list[ReplaySample]] = []
    total_lines = 0
    for seed in seeds:
        env_rng = random.Random(seed)
        explore_rng = random.Random(seed + 7_000_000_001)
        game, lines = play_game(
            scorer, env_rng,
            epsilon=epsilon, exploration_top_k=top_k,
            max_pieces=max_pieces,
            heuristic_scorer=heur,
            legacy_lockout=legacy_lockout,
            explore_rng=explore_rng,
        )
        total_lines += lines
        samples = [ReplaySample(bt, ci, ni, discounted_survival(int(st), gamma),
                                hv, generation)
                   for bt, ci, ni, hv, st in game]
        out.append(samples)
    return (seeds[0], out, total_lines)


def generate_selfplay_dataset_parallel(
    weights_path: str,
    n_games: int,
    *,
    epsilon: float,
    max_pieces: int,
    top_k: int,
    survival_weight: float,
    heuristic_weight: float,
    gamma: float,
    generation: int,
    legacy_lockout: bool = False,
    workers: int = 1,
    base_seed: int = 0,
    game_stats_out: Optional[list] = None,
    elite_fraction: float = 1.0,
) -> list[ReplaySample]:
    """Collect `n_games` rollouts across `workers` processes.

    Determinism contract (phase A4/D4): game `base_seed + i` gets env RNG
    seed `base_seed + i` and exploration seed `base_seed + i + 7_000_000_001`
    regardless of scheduling, so results are identical for any worker count
    (>=1). Uses serial in-process generation when workers == 1.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    seeds = [base_seed + i for i in range(n_games)]
    per_game: list[tuple[float, list[ReplaySample]]] = []
    total_lines = 0

    def record(shard_games: list[list[ReplaySample]], shard_lines: int) -> None:
        nonlocal total_lines
        total_lines += shard_lines
        for samples in shard_games:
            per_game.append((float(len(samples)), samples))

    if workers <= 1:
        payload_shards = [(seeds, epsilon, max_pieces, top_k, survival_weight,
                           heuristic_weight, gamma, generation, legacy_lockout)]
        (shard_games, shard_lines) = _rollout_worker(
            (weights_path, seeds, epsilon, max_pieces, top_k, survival_weight,
             heuristic_weight, gamma, generation, legacy_lockout))[1:]
        record(shard_games, shard_lines)
    else:
        chunk = max(1, (n_games + workers - 1) // workers)
        shards = [seeds[i:i + chunk] for i in range(0, len(seeds), chunk)]
        payloads = [(weights_path, shard, epsilon, max_pieces, top_k,
                     survival_weight, heuristic_weight, gamma, generation,
                     legacy_lockout) for shard in shards]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_rollout_worker, p): p for p in payloads}
            for fut in as_completed(futures):
                _sid, shard_games, shard_lines = fut.result()
                record(shard_games, shard_lines)

    if game_stats_out is not None:
        game_stats_out.extend((int(n), 0) for n, _ in per_game)  # lines unknown per game
        if game_stats_out and game_stats_out[-1][1] == 0:
            pass  # per-game line counts are folded into total_lines only
    _ = total_lines  # reported by caller via game_stats aggregate

    if elite_fraction >= 1.0 or not per_game:
        return [s for _, samples in per_game for s in samples]
    ranked = sorted(per_game, key=lambda entry: -entry[0])
    keep = max(1, round(len(ranked) * elite_fraction))
    print(f"  Elite filter: kept {keep}/{len(ranked)} games "
          f"(survival >= {ranked[keep - 1][0]:.0f} pieces)")
    return [s for _, samples in ranked[:keep] for s in samples]


# ---- MC training loop (#2: replay buffer) ---------------------------------

# Full training-state checkpointing (phase A3 remainder, 2026-09).
#
# Weight-only checkpoints cannot resume training faithfully: Adam moments,
# replay contents, RNG streams, the reward normalizer, anchor, epsilon
# schedule, and round counter all change what the NEXT update computes.
# ``save_train_state`` persists all of it; ``load_train_state`` restores it.

STATE_VERSION = 1


def save_train_state(path, *, net, optimizer, replay: ReplayBuffer,
                     normalizer: Optional[RewardNormalizer],
                     anchor_mean: float, anchor_std: float,
                     round_idx: int, epsilon_schedule: str, epsilon0: float,
                     env_rng: random.Random, explore_rng: random.Random,
                     replay_rng: random.Random, best_mean: float,
                     best_round: int, config: dict) -> None:
    """Atomically persist the complete resumable training state.

    Writes to ``<path>.tmp`` then os.replace()s, so an interrupted save never
    corrupts the previous checkpoint.
    """
    import os
    import tempfile
    env_state = env_rng.getstate()
    explore_state = explore_rng.getstate()
    replay_state = replay_rng.getstate()
    payload = {
        "format_version": STATE_VERSION,
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay_samples": list(replay.buffer),
        "replay_capacity": replay.capacity,
        "normalizer": (None if normalizer is None else {
            "survival_mean": normalizer.survival_mean,
            "survival_std": normalizer.survival_std,
            "heuristic_mean": normalizer.heuristic_mean,
            "heuristic_std": normalizer.heuristic_std,
        }),
        "anchor_mean": anchor_mean,
        "anchor_std": anchor_std,
        "round_idx": round_idx,
        "epsilon_schedule": epsilon_schedule,
        "epsilon0": epsilon0,
        "rng": {"env": env_state, "explore": explore_state,
                "replay": replay_state},
        "best_mean": best_mean,
        "best_round": best_round,
        "config": config,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(path.parent),
                               prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_train_state(path, *, net, optimizer, replay: ReplayBuffer,
                     device: str = "cpu") -> dict:
    """Restore a training state saved by :func:`save_train_state`.

    Mutates net/optimizer/replay in place and returns the remaining fields
    (normalizer, anchor, round index, RNG states, best tracking, config).
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    version = payload.get("format_version")
    if version != STATE_VERSION:
        raise ValueError(f"train-state format {version} unsupported "
                         f"(expected {STATE_VERSION})")
    net.load_state_dict(payload["net"])
    optimizer.load_state_dict(payload["optimizer"])
    replay.buffer = deque(payload["replay_samples"],
                          maxlen=payload["replay_capacity"])
    replay._index_dirty = True
    norm = payload["normalizer"]
    normalizer = None if norm is None else RewardNormalizer(**norm)
    rng = payload["rng"]
    env_rng = random.Random()
    env_rng.setstate(tuple(rng["env"]))
    explore_rng = random.Random()
    explore_rng.setstate(tuple(rng["explore"]))
    replay_rng = random.Random()
    replay_rng.setstate(tuple(rng["replay"]))
    return {
        "normalizer": normalizer,
        "anchor_mean": payload["anchor_mean"],
        "anchor_std": payload["anchor_std"],
        "round_idx": payload["round_idx"],
        "epsilon_schedule": payload["epsilon_schedule"],
        "epsilon0": payload["epsilon0"],
        "env_rng": env_rng,
        "explore_rng": explore_rng,
        "replay_rng": replay_rng,
        "best_mean": payload["best_mean"],
        "best_round": payload["best_round"],
        "config": payload["config"],
    }


def train_selfplay(
    n_rounds: int = 5,
    games_per_round: int = 100,
    epochs_per_round: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    survival_weight: float = 0.7,
    heuristic_weight: float = 0.3,
    replay_capacity: int = 50_000,
    replay_max_age: int = 0,
    replay_recent_fraction: float = 0.5,
    exploration_top_k: int = 5,
    eval_games: int = 8,
    eval_seed: int = 9001,
    eval_max_pieces: int = 1500,
    eval_every: int = 1,
    rollback_factor: float = 0.3,
    freeze_conv_rounds: int = 0,
    elite_fraction: float = 1.0,
    seed: int = 42,
    env_seed: int | None = None,
    explore_seed: int | None = None,
    replay_seed: int | None = None,
    rollout_workers: int = 1,
    legacy_lockout: bool = False,
    resume_state: Optional[str] = None,
    weights_in: Optional[str] = None,
    weights_out: str = "model_weights.pt",
    device: str = "cpu",
) -> TAPNet:
    """Train TAPNet via MC self-play with scale-matched reward shaping.

    Each round:
      1. Play `games_per_round` games with the current network (epsilon-greedy).
         Collect (state, cur_idx, next_idx, raw_reward) samples.
      2. Add samples to the replay buffer (all generations kept by default).
      3. Train for `epochs_per_round` epochs on scale-matched targets:
         z-scored reward blend re-anchored onto the network's own output
         distribution (mean/std measured from the loaded weights).
      4. Greedy-eval `eval_games` seeds; keep the best weights in
         `<weights_out>.best.pt`; optionally roll back if eval collapses
         (see `rollback_factor`).

    Warm-starting (`weights_in`) measures the value-function scale BEFORE
    training, so a trained checkpoint keeps its output scale — this is the
    2026-09 fix for the snapshot7->snapshot8 collapse.

    RNG separation (phase A4): `seed` seeds torch; piece generation, epsilon
    exploration, and replay minibatch sampling each get their own Random
    instance (`env_seed`/`explore_seed`/`replay_seed`, defaulting to
    seed+1/seed+2/seed+3). Changing exploration or replay settings can no
    longer perturb the training piece streams.
    """
    torch.manual_seed(seed)
    env_rng = random.Random(env_seed if env_seed is not None else seed + 1)
    explore_rng = random.Random(explore_seed if explore_seed is not None else seed + 2)
    replay_rng = random.Random(replay_seed if replay_seed is not None else seed + 3)

    # Build the network.
    net = TAPNet(output="linear")
    net.to(device)
    if weights_in is not None:
        # Resolve against several plausible bases and NEVER silently fall
        # back to from-scratch: that failure mode produced mislabeled
        # checkpoints during the 2026-07 debugging session.
        supplied = Path(weights_in).expanduser()
        candidates = ([supplied] if supplied.is_absolute()
                      else [supplied, HERE / supplied, HERE.parent / supplied,
                            HERE.parent.parent / supplied])
        p = next((c for c in candidates if c.is_file()), None)
        if p is not None:
            net.load_state_dict(torch.load(p, map_location=device))
            warm_start = True
            print(f"Loaded initial weights from {p.resolve()}")
        else:
            tried = ", ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"--weights-in not found; tried: {tried}. "
                "Refusing to silently train from scratch.")
    else:
        warm_start = False
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    # Replay buffer (#2) + heuristic scorer for reward shaping (#1).
    replay = ReplayBuffer(capacity=replay_capacity)
    heuristic_scorer = HeuristicScorer() if heuristic_weight > 0 else None
    normalizer: RewardNormalizer | None = None
    anchor_mean = 0.0
    anchor_std = 1.0

    # Resume (phase A3): restore weights + optimizer + replay + RNGs +
    # normalizer/anchor + schedules + best tracking from a saved state.
    # Runs AFTER replay/normalizer creation so they can be populated.
    start_round = 0
    if resume_state is not None:
        supplied = Path(resume_state).expanduser()
        rp = next((c for c in ([supplied, HERE / supplied,
                                HERE.parent / supplied]) if c.is_file()), None)
        if rp is None:
            raise FileNotFoundError(f"--resume state not found: {resume_state}")
        restored = load_train_state(rp, net=net, optimizer=optimizer,
                                    replay=replay, device=device)
        normalizer = restored["normalizer"]
        anchor_mean = restored["anchor_mean"]
        anchor_std = restored["anchor_std"]
        start_round = restored["round_idx"]
        env_rng = restored["env_rng"]
        explore_rng = restored["explore_rng"]
        replay_rng = restored["replay_rng"]
        best_mean = restored["best_mean"]
        best_round = restored["best_round"]
        warm_start = True  # weights are trained weights now
        # Guard against config drift that would silently change semantics.
        cfg = restored["config"]
        mismatched = {k: (cfg.get(k), v) for k, v in
                      [("gamma", gamma), ("lr", lr),
                       ("replay_capacity", replay_capacity),
                       ("survival_weight", survival_weight),
                       ("heuristic_weight", heuristic_weight)]
                      if cfg.get(k) != v}
        if mismatched:
            print(f"  WARNING: resume config differs from CLI args: "
                  f"{mismatched} (using CLI values)")
        print(f"  Resumed training state from {rp} "
              f"(next round {start_round + 1}, replay {len(replay)}, "
              f"best_mean {best_mean:.1f} from round {best_round})")

    # Round-checkpoint destination (also used for the .best.pt guardrail file).
    out_path = HERE.parent / weights_out

    # Baseline eval of the loaded weights (or fresh init) so per-round evals
    # have something to compare against. Also the source of anchor samples.
    # SKIPPED when resuming: the anchor and best tracking were restored from
    # the train state, and re-measuring would silently overwrite them.
    baseline_pieces: list[int] = []
    anchor_probe_samples: list[ReplaySample] = []
    if eval_games > 0 and resume_state is None:
        net.eval()
        probe_scorer = NeuralScorer(net=net, device=device)
        # Phase A3 fix: the baseline now uses the SAME eval_games seeds as
        # every later round. The old min(eval_games, 3) subset made 3-seed
        # baselines incomparable with 8-seed round evals (564.0 vs 502.9 on
        # the same weights), corrupting best-checkpoint selection.
        for i in range(eval_games):
            trajectory, _ = play_game(probe_scorer, random.Random(eval_seed + i),
                                      epsilon=0.0, max_pieces=eval_max_pieces)
            baseline_pieces.append(len(trajectory))
            for bt, ci, ni, _hv, _steps in trajectory:
                anchor_probe_samples.append(
                    ReplaySample(bt, ci, ni, 0.0, 0.0, -1))
        if baseline_pieces:
            print(f"  Baseline eval ({len(baseline_pieces)} games, "
                  f"seeds {eval_seed}..): "
                  f"mean={statistics.fmean(baseline_pieces):.1f} "
                  f"max={max(baseline_pieces)}")

    if warm_start and resume_state is None:
        # Measure the value-function scale from the loaded network on real
        # afterstate boards (from the probe games). This is what keeps a
        # resumed checkpoint from being crushed toward the target scale.
        anchor_probe = anchor_probe_samples or None
        if anchor_probe:
            anchor_mean, anchor_std = measure_output_scale(
                net, anchor_probe, device=device)
            # Phase A2 fix: enforce the documented 1.0 std floor (a constant
            # or near-constant value function would otherwise receive
            # near-zero targets and train toward a constant).
            if anchor_std < 1.0:
                print(f"  Anchor std {anchor_std:.3f} < floor 1.0 — using 1.0")
                anchor_std = 1.0
        else:
            anchor_mean, anchor_std = 0.0, 1.0
        print(f"  Value-function anchor from warm start: "
              f"mean={anchor_mean:.2f} std={anchor_std:.2f}")

    # Best-tracking initialization: restored from train state when resuming,
    # otherwise seeded from the baseline eval. best_state must hold the
    # RESTORED champion weights on resume, not the live (resumed) net — but
    # on resume the live net IS the last state's weights, so they coincide.
    if resume_state is None:
        best_mean = statistics.fmean(baseline_pieces) if baseline_pieces else 0.0
        best_round = 0
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if resume_state is None:
        # Persist the starting weights immediately so a run that never beats
        # its baseline still leaves the best-known checkpoint on disk.
        torch.save(best_state, str(Path(str(out_path) + ".best.pt")))

    # Optional conv freezing for representation adaptation.
    conv_params = list(net.conv1.parameters()) + list(net.conv2.parameters())

    def set_conv_frozen(frozen: bool) -> None:
        for p in conv_params:
            p.requires_grad_(not frozen)

    if freeze_conv_rounds > 0:
        set_conv_frozen(True)
        print(f"  Conv layers frozen for the first {freeze_conv_rounds} "
              "round(s) (representation adaptation).")

    for round_idx in range(start_round, n_rounds):
        # --- Self-play phase ---
        net.eval()
        scorer = NeuralScorer(net=net, device=device)
        cur_epsilon = epsilon * (0.9 ** round_idx)
        print(f"\n=== Round {round_idx + 1}/{n_rounds} (epsilon={cur_epsilon:.3f}) ===")
        print(f"  Playing {games_per_round} games...")
        game_stats: list[tuple[int, int]] = []

        # Phase D4: optional parallel rollout collection. The parallel path
        # snapshots the behavior weights once and plays deterministic
        # per-game seeds in worker processes (identical results for any
        # worker count). workers=1 stays on the in-process path.
        behavior_path = Path(str(out_path) + ".behavior.pt")
        use_parallel = rollout_workers > 1 and games_per_round >= rollout_workers
        if use_parallel:
            torch.save(net.state_dict(), str(behavior_path))
            new_samples = generate_selfplay_dataset_parallel(
                str(behavior_path), games_per_round,
                epsilon=cur_epsilon, max_pieces=1500,
                top_k=exploration_top_k,
                survival_weight=survival_weight,
                heuristic_weight=heuristic_weight,
                gamma=gamma, generation=round_idx,
                legacy_lockout=legacy_lockout,
                workers=rollout_workers,
                base_seed=env_seed if env_seed is not None else seed + 1,
                game_stats_out=game_stats,
                elite_fraction=elite_fraction,
            )
        else:
            new_samples = generate_selfplay_dataset(
                scorer, games_per_round, env_rng,
                epsilon=cur_epsilon, exploration_top_k=exploration_top_k,
                gamma=gamma,
                survival_weight=survival_weight,
                heuristic_weight=heuristic_weight,
                heuristic_scorer=heuristic_scorer,
                game_stats_out=game_stats,
                generation=round_idx,
                elite_fraction=elite_fraction,
                explore_rng=explore_rng,
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

        # Keep only recent behavior-policy generations when explicitly asked.
        # Default (replay_max_age=0) keeps ALL generations: the old expiry
        # starved the buffer of opening states once games got short.
        replay.extend(new_samples)
        if replay_max_age > 0:
            replay.discard_before(max(0, round_idx - replay_max_age + 1))

        # Keep the anchor fresh: once real samples exist, re-measure the
        # network's output scale against the actual replay distribution.
        if round_idx == 0:
            anchor_mean, anchor_std = measure_output_scale(
                net, new_samples, device=device)
            print(f"  Value-function anchor (on round-1 rollouts): "
                  f"mean={anchor_mean:.2f} std={anchor_std:.2f}")

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
        targets_list = [anchored_target(
            s, normalizer, anchor_mean, anchor_std,
            survival_weight, heuristic_weight) for s in new_samples]
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
                    batch_size, replay_rng, generation=round_idx,
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
                targets = [anchored_target(
                    s, normalizer, anchor_mean, anchor_std,
                    survival_weight, heuristic_weight) for s in batch]
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

        # --- Per-round greedy eval guardrail -------------------------------
        eval_mean: float | None = None
        if eval_games > 0 and (round_idx + 1) % eval_every == 0:
            pieces = quick_eval(net, eval_games, eval_seed,
                                device=device, max_pieces=eval_max_pieces)
            eval_mean = statistics.fmean(pieces)
            print(f"  Eval ({eval_games} games, seeds {eval_seed}..): "
                  f"mean={eval_mean:.1f} med={statistics.median(pieces):.1f} "
                  f"max={max(pieces)} min={min(pieces)}")
            if eval_mean > best_mean:
                best_mean = eval_mean
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
                best_round = round_idx + 1
                torch.save(best_state,
                           str(Path(str(out_path) + ".best.pt")))
                print(f"  New best — saved {Path(str(out_path) + '.best.pt')} "
                      f"(mean {best_mean:.1f})")
            elif (rollback_factor > 0
                  and eval_mean < best_mean * rollback_factor):
                # Phase A3 fix: persist the rejected candidate before
                # restoring, and allow round-zero restoration. The old code
                # required best_round > 0, so a run that only deteriorated
                # NEVER restored the starting champion.
                rejected_path = Path(str(out_path) + f".rejected_r{round_idx + 1:02d}.pt")
                torch.save(net.state_dict(), str(rejected_path))
                print(f"  Rejected candidate saved to {rejected_path.name}")
                print(f"  COLLAPSE GUARDRAIL: eval {eval_mean:.1f} < "
                      f"{rollback_factor:.0%} of best {best_mean:.1f} — "
                      f"restoring best weights from round {best_round} "
                      f"({Path(str(out_path) + '.best.pt').name}).")
                net.load_state_dict(best_state)

        # Unfreeze conv layers after the adaptation window.
        if freeze_conv_rounds > 0 and round_idx + 1 == freeze_conv_rounds:
            set_conv_frozen(False)
            print("  Conv layers unfrozen.")

        # Checkpoint: save weights + FULL training state after each round so
        # an interrupted run resumes faithfully. Weights go to the usual
        # output file; the resumable state goes to <out>.state.pt.
        torch.save(net.state_dict(), out_path)
        try:
            save_train_state(
                Path(str(out_path) + ".state.pt"),
                net=net, optimizer=optimizer, replay=replay,
                normalizer=normalizer,
                anchor_mean=anchor_mean, anchor_std=anchor_std,
                round_idx=round_idx + 1,
                epsilon_schedule="0.9^round", epsilon0=epsilon,
                env_rng=env_rng, explore_rng=explore_rng,
                replay_rng=replay_rng,
                best_mean=best_mean, best_round=best_round,
                config={"n_rounds": n_rounds,
                        "games_per_round": games_per_round,
                        "epochs_per_round": epochs_per_round,
                        "batch_size": batch_size, "lr": lr,
                        "gamma": gamma,
                        "survival_weight": survival_weight,
                        "heuristic_weight": heuristic_weight,
                        "replay_capacity": replay_capacity,
                        "replay_max_age": replay_max_age,
                        "replay_recent_fraction": replay_recent_fraction,
                        "exploration_top_k": exploration_top_k,
                        "eval_games": eval_games, "eval_seed": eval_seed,
                        "eval_max_pieces": eval_max_pieces,
                        "eval_every": eval_every,
                        "rollback_factor": rollback_factor,
                        "freeze_conv_rounds": freeze_conv_rounds,
                        "elite_fraction": elite_fraction,
                        "seed": seed,
                        "env_seed": env_seed, "explore_seed": explore_seed,
                        "replay_seed": replay_seed,
                        "rollout_workers": rollout_workers,
                        "legacy_lockout": legacy_lockout,
                        "weights_in": weights_in},
            )
        except Exception as exc:  # noqa: BLE001 — state save must not kill training
            print(f"  WARNING: train-state save failed: {exc}")
        capture_hook = globals().get("PER_ROUND_CAPTURE")
        if capture_hook is not None:
            capture_hook(round_idx, net)
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
    parser.add_argument("--exploration-top-k", type=int, default=5,
                        help="explore only among this many strongest candidates")
    parser.add_argument("--gamma", type=float, default=0.99, help="survival discount factor")
    parser.add_argument("--survival-weight", type=float, default=0.7,
                        help="weight for survival in blended target (#1)")
    parser.add_argument("--heuristic-weight", type=float, default=0.3,
                        help="weight for heuristic reward in blended target (#1)")
    parser.add_argument("--replay-capacity", type=int, default=50_000,
                        help="max replay buffer size (#2)")
    parser.add_argument("--replay-max-age", type=int, default=0,
                        help="generations to retain (0 = keep all; old default 2 "
                             "starved the buffer of opening states)")
    parser.add_argument("--replay-recent-fraction", type=float, default=0.5,
                        help="fraction of each batch drawn from the newest round")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--weights-in", type=str, default=None, help="initial weights to load")
    parser.add_argument("--weights-out", type=str, default="model_weights.pt",
                        help="output weights path")
    parser.add_argument("--eval-games", type=int, default=8,
                        help="greedy eval games per round (0 disables eval guardrail)")
    parser.add_argument("--eval-seed", type=int, default=9001,
                        help="first eval game seed")
    parser.add_argument("--eval-max-pieces", type=int, default=1500,
                        help="piece cap for eval games")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="evaluate every N rounds")
    parser.add_argument("--rollback-factor", type=float, default=0.3,
                        help="restore best weights if eval drops below this "
                             "fraction of best (0 disables)")
    parser.add_argument("--freeze-conv", type=int, default=0,
                        help="freeze conv layers for the first N rounds "
                             "(representation adaptation)")
    parser.add_argument("--elite-fraction", type=float, default=1.0,
                        help="train only on the strongest games (top fraction "
                             "by survival); 1.0 keeps all games")
    parser.add_argument("--legacy-lockout", action="store_true",
                        help="restore historical lockout semantics (above-field "
                             "cells silently dropped, game continues). Default is "
                             "corrected: lockout placements lose the game.")
    parser.add_argument("--env-seed", type=int, default=None,
                        help="piece-stream RNG seed (default: seed+1)")
    parser.add_argument("--explore-seed", type=int, default=None,
                        help="exploration RNG seed (default: seed+2)")
    parser.add_argument("--replay-seed", type=int, default=None,
                        help="replay-sampling RNG seed (default: seed+3)")
    parser.add_argument("--rollout-workers", type=int, default=1,
                        help="parallel rollout workers per round (1 = serial, "
                             "historical behavior)")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from a <weights>.state.pt saved by a "
                             "previous interrupted run")
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
        eval_games=args.eval_games,
        eval_seed=args.eval_seed,
        eval_max_pieces=args.eval_max_pieces,
        eval_every=args.eval_every,
        rollback_factor=args.rollback_factor,
        freeze_conv_rounds=args.freeze_conv,
        elite_fraction=args.elite_fraction,
        seed=args.seed,
        env_seed=args.env_seed,
        explore_seed=args.explore_seed,
        replay_seed=args.replay_seed,
        rollout_workers=args.rollout_workers,
        legacy_lockout=args.legacy_lockout,
        resume_state=args.resume,
        weights_in=args.weights_in,
        weights_out=args.weights_out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
