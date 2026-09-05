"""Deterministic, optimizer-free evaluation of TAPNet checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet
from neural_scorer import NeuralScorer
from selfplay import play_game


@dataclass(frozen=True)
class GameResult:
    seed: int
    pieces: int
    lines: int
    capped: bool


@dataclass(frozen=True)
class EvaluationResult:
    weights: str
    sha256: str
    games: list[GameResult]
    mean_pieces: float
    median_pieces: float
    stdev_pieces: float
    min_pieces: int
    max_pieces: int
    mean_lines: float
    capped_games: int


def resolve_weights(path: str) -> Path:
    """Resolve explicit paths first, then paths relative to the crate root."""
    supplied = Path(path).expanduser()
    candidates = [supplied, HERE.parent / supplied]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"weights file not found: {path}")


def evaluate_weights(path: str, *, games: int = 100, seed: int = 42,
                     max_pieces: int = 1500, device: str = "cpu",
                     verbose: bool = False,
                     legacy_lockout: bool = False) -> EvaluationResult:
    """Evaluate one checkpoint greedily on independent seeds ``seed..seed+games``.

    ``legacy_lockout=True`` restores the historical game-over semantics
    (above-field cells silently dropped, game continues) so the 2026-09-04
    benchmark numbers stay reproducible. Default (False) uses the corrected
    semantics where a lockout placement ends the game.
    """
    if games < 1:
        raise ValueError("games must be at least 1")
    weights_path = resolve_weights(path)
    net = TAPNet(output="linear").to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    scorer = NeuralScorer(net=net, device=device)

    results: list[GameResult] = []
    for game_seed in range(seed, seed + games):
        trajectory, lines = play_game(
            scorer, random.Random(game_seed), epsilon=0.0,
            max_pieces=max_pieces, legacy_lockout=legacy_lockout)
        result = GameResult(game_seed, len(trajectory), lines,
                            len(trajectory) >= max_pieces)
        results.append(result)
        if verbose:
            suffix = " (capped)" if result.capped else ""
            print(f"  seed={game_seed}: pieces={result.pieces} "
                  f"lines={result.lines}{suffix}")

    pieces = [result.pieces for result in results]
    lines = [result.lines for result in results]
    return EvaluationResult(
        weights=str(weights_path),
        sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        games=results,
        mean_pieces=statistics.fmean(pieces),
        median_pieces=statistics.median(pieces),
        stdev_pieces=statistics.stdev(pieces) if len(pieces) > 1 else 0.0,
        min_pieces=min(pieces),
        max_pieces=max(pieces),
        mean_lines=statistics.fmean(lines),
        capped_games=sum(result.capped for result in results),
    )


def print_result(result: EvaluationResult) -> None:
    print(f"\n{result.weights}")
    print(f"  SHA-256: {result.sha256}")
    print(f"  Pieces — mean={result.mean_pieces:.1f} "
          f"median={result.median_pieces:.1f} "
          f"std={result.stdev_pieces:.1f} min={result.min_pieces} "
          f"max={result.max_pieces}")
    print(f"  Lines  — mean={result.mean_lines:.1f}")
    print(f"  Capped — {result.capped_games}/{len(result.games)}")


# ---- Parallel game-level evaluation (phase D3, 2026-09) --------------------

def _eval_one_game(args: tuple) -> dict:
    """Worker: play ONE greedy game. Spawn-safe; re-imports lazily."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    (here, weights_path, seed, max_pieces, legacy_lockout) = args
    import sys
    sys.path.insert(0, here)
    from model import TAPNet
    from neural_scorer import NeuralScorer
    from selfplay import play_game
    import random as _random

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()
    scorer = NeuralScorer(net=net, device="cpu")
    import time
    t0 = time.perf_counter()
    trajectory, lines = play_game(scorer, _random.Random(seed), epsilon=0.0,
                                  max_pieces=max_pieces,
                                  legacy_lockout=legacy_lockout)
    return {
        "seed": seed,
        "pieces": len(trajectory),
        "lines": lines,
        "capped": len(trajectory) >= max_pieces,
        "seconds": time.perf_counter() - t0,
    }


def evaluate_weights_parallel(
    path: str, *, games: int = 100, seed: int = 42,
    max_pieces: int = 1500, workers: int | None = None,
    legacy_lockout: bool = False,
) -> EvaluationResult:
    """Greedy evaluation with games sharded across processes.

    Phase D3: the historical evaluator parallelized only across checkpoints,
    so a single-checkpoint run occupied one core. This shards GAMES across
    workers; each game keeps its fixed seed, so results are identical to the
    serial evaluator for any worker count.
    """
    if games < 1:
        raise ValueError("games must be at least 1")
    weights_path = resolve_weights(path)
    if workers is None:
        workers = max(1, min(games, (os.cpu_count() or 2)))
    if workers <= 1:
        return evaluate_weights(path, games=games, seed=seed,
                                max_pieces=max_pieces,
                                legacy_lockout=legacy_lockout)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    here = str(HERE)
    jobs = [(here, str(weights_path), s, max_pieces, legacy_lockout)
            for s in range(seed, seed + games)]
    results: list[GameResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_eval_one_game, jobs):
            results.append(GameResult(res["seed"], res["pieces"],
                                      res["lines"], res["capped"]))
    results.sort(key=lambda r: r.seed)
    pieces = [r.pieces for r in results]
    lines = [r.lines for r in results]
    return EvaluationResult(
        weights=str(weights_path),
        sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        games=results,
        mean_pieces=statistics.fmean(pieces),
        median_pieces=statistics.median(pieces),
        stdev_pieces=statistics.stdev(pieces) if len(pieces) > 1 else 0.0,
        min_pieces=min(pieces),
        max_pieces=max(pieces),
        mean_lines=statistics.fmean(lines),
        capped_games=sum(r.capped for r in results),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy fixed-seed evaluation with no exploration or training.")
    parser.add_argument("--weights", nargs="+", required=True,
                        help="one or more checkpoint paths")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42,
                        help="first independent game seed")
    parser.add_argument("--max-pieces", type=int, default=1500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel game workers (default: cpu_count; "
                             "1 = serial)")
    parser.add_argument("--legacy-lockout", action="store_true",
                        help="use historical lockout semantics (reproduces "
                             "the 2026-09-04 benchmark; default is corrected "
                             "semantics where lockout placements lose)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", type=str, default=None,
                        help="optional output path for machine-readable results")
    args = parser.parse_args()

    results = []
    for path in args.weights:
        if args.workers is not None and args.workers > 1:
            results.append(evaluate_weights_parallel(
                path, games=args.games, seed=args.seed,
                max_pieces=args.max_pieces, workers=args.workers,
                legacy_lockout=args.legacy_lockout))
        else:
            results.append(evaluate_weights(
                path, games=args.games, seed=args.seed,
                max_pieces=args.max_pieces, device=args.device,
                verbose=args.verbose, legacy_lockout=args.legacy_lockout))
    for result in results:
        print_result(result)

    if len(results) > 1:
        baseline = results[0]
        for candidate in results[1:]:
            deltas = [b.pieces - a.pieces
                      for a, b in zip(baseline.games, candidate.games)]
            wins = sum(delta > 0 for delta in deltas)
            ties = sum(delta == 0 for delta in deltas)
            losses = sum(delta < 0 for delta in deltas)
            print(f"  Paired vs {Path(baseline.weights).name}: "
                  f"mean delta={statistics.fmean(deltas):+.1f}; "
                  f"W/T/L={wins}/{ties}/{losses}")

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps([asdict(result) for result in results],
                                     indent=2), encoding="utf-8")
        print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
