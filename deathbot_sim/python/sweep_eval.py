"""Sweep all TAPNet checkpoints and rank them by greedy self-play survival.

For each checkpoint:
  - Greedy (epsilon=0) games on independent fixed seeds (seed..seed+games-1).
  - Reports pieces (survival), lines cleared, capped games, score stability.

Also fingerprints each checkpoint:
  - SHA-256, L2 norm per parameter tensor, and overall param L2 norm
  - Pairwise cosine similarity between checkpoints (flattened params)

All per-game raw results are cached to a JSON file so that later sweeps can
be compared against earlier runs without re-evaluating.

Usage (from deathbot_sim/python, with the workspace venv):
    python sweep_eval.py                 # all checkpoints, default settings
    python sweep_eval.py --games 40
    python sweep_eval.py --max-pieces 3000
    python sweep_eval.py --compare old.json   # paired deltas vs cached run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Fingerprinting needs torch but evaluation workers import their own modules,
# so torch is imported lazily in the worker to avoid fork/spawn overhead here.
import torch


# ---- Checkpoint discovery --------------------------------------------------

def discover_checkpoints(extra: list[str] | None = None) -> list[Path]:
    """Find candidate weight files.

    With ``extra`` (the --weights list), resolve ONLY those names/paths
    (as-given, then relative to the crate root, then the workspace root).
    Without it, discover every .pt in the crate and workspace roots.
    """
    if extra:
        resolved: list[Path] = []
        for name in extra:
            supplied = Path(name).expanduser()
            for candidate in (supplied, HERE.parent / supplied,
                              HERE.parent.parent / supplied):
                if candidate.is_file():
                    resolved.append(candidate.resolve())
                    break
            else:
                raise FileNotFoundError(f"weights file not found: {name}")
        # Deduplicate by resolved path, preserving order.
        seen: set[Path] = set()
        unique = [p for p in resolved
                  if not (p in seen or seen.add(p))]
        return unique
    roots = [HERE.parent, HERE.parent.parent]
    names: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.pt")):
            names.setdefault(p.name, p)  # crate-root copy wins
    return [names[k] for k in sorted(names)]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- Fingerprinting --------------------------------------------------------

def fingerprint(path: Path) -> dict:
    """Load a checkpoint and compute per-tensor norms + overall stats."""
    state = torch.load(path, map_location="cpu")
    tensors = {k: v.detach().float() for k, v in state.items()
               if isinstance(v, torch.Tensor) and v.numel() > 0}
    total_sq = sum(float(t.pow(2).sum()) for t in tensors.values())
    per_tensor = {k: float(t.pow(2).sum().sqrt()) for k, t in tensors.items()}
    return {
        "path": str(path),
        "sha256": sha256_of(path),
        "param_count": sum(t.numel() for t in tensors.values()),
        "l2": math.sqrt(total_sq),
        "per_tensor_l2": per_tensor,
        "tensors": sorted(tensors),
    }


def flatten_state(state: dict) -> torch.Tensor:
    keys = sorted(state.keys())
    return torch.cat([state[k].detach().float().flatten() for k in keys])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / denom) if denom > 0 else float("nan")


# ---- Per-game evaluation (worker) ------------------------------------------

def _evaluate_one(args: tuple) -> dict:
    """Worker: greedy games for a single checkpoint. Returns per-game rows."""
    (weights_path, games, seed, max_pieces, torch_threads, verbose) = args
    torch.set_num_threads(torch_threads)
    from model import TAPNet
    from neural_scorer import NeuralScorer
    from selfplay import play_game

    path = Path(weights_path)
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    scorer = NeuralScorer(net=net, device="cpu")

    rows = []
    for game_seed in range(seed, seed + games):
        t0 = time.perf_counter()
        trajectory, lines = play_game(
            scorer, random.Random(game_seed), epsilon=0.0,
            max_pieces=max_pieces)
        pieces = len(trajectory)
        rows.append({
            "seed": game_seed,
            "pieces": pieces,
            "lines": lines,
            "capped": pieces >= max_pieces,
            "seconds": time.perf_counter() - t0,
        })
        if verbose:
            print(f"  {path.name} seed={game_seed}: pieces={pieces} "
                  f"lines={lines}{' (capped)' if rows[-1]['capped'] else ''}",
                  flush=True)
    return {"weights": str(path), "name": path.name, "games": rows}


# ---- Aggregation -----------------------------------------------------------

def summarize(rows: list[dict]) -> dict:
    pieces = [g["pieces"] for g in rows]
    lines = [g["lines"] for g in rows]
    return {
        "mean_pieces": statistics.fmean(pieces),
        "median_pieces": statistics.median(pieces),
        "stdev_pieces": statistics.stdev(pieces) if len(pieces) > 1 else 0.0,
        "min_pieces": min(pieces),
        "max_pieces": max(pieces),
        "mean_lines": statistics.fmean(lines),
        "capped_games": sum(bool(g["capped"]) for g in rows),
        "n_games": len(rows),
    }


# ---- Main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", nargs="*", default=None,
                        help="restrict to these checkpoint names/paths")
    parser.add_argument("--games", type=int, default=30)
    seeds = None
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pieces", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--compare", type=str, default=None,
                        help="JSON from a previous sweep: paired per-seed deltas")
    args = parser.parse_args()

    checkpoints = discover_checkpoints(args.weights)
    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"Found {len(checkpoints)} checkpoints; "
          f"games={args.games} seed0={args.seed} "
          f"max_pieces={args.max_pieces} workers={args.workers}")
    for p in checkpoints:
        print(f"  - {p}")

    # Fingerprints: detect duplicates/renames and L2 drift between files.
    fingerprints = {}
    for p in checkpoints:
        fp = fingerprint(p)
        fingerprints[p.name] = fp
        print(f"  {p.name}: sha256={fp['sha256'][:12]}… l2={fp['l2']:.1f} "
              f"params={fp['param_count']}")
    # Pairwise cosine similarity of flattened params (duplicate detection).
    flat = {p.name: flatten_state(torch.load(p, map_location="cpu"))
            for p in checkpoints}
    names = sorted(flat)
    print("\n=== Checkpoint similarity (cosine of flattened params) ===")
    header = " " * 28 + "".join(f"{n[-11:]:>12s}" for n in names)
    print(header)
    for a in names:
        row = f"  {a[-26:]:26s}"
        for b in names:
            c = cosine(flat[a], flat[b]) if flat[a].numel() == flat[b].numel() else float("nan")
            row += f"{c:12.4f}"
        print(row)

    jobs = [(str(p), args.games, args.seed, args.max_pieces,
             args.threads_per_worker, args.verbose) for p in checkpoints]
    t0 = time.perf_counter()
    per_ckpt = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_evaluate_one, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            per_ckpt[res["name"]] = res
            s = summarize(res["games"])
            print(f"[{res['name']}] mean={s['mean_pieces']:.1f} "
                  f"med={s['median_pieces']:.1f} sd={s['stdev_pieces']:.1f} "
                  f"min={s['min_pieces']} max={s['max_pieces']} "
                  f"lines={s['mean_lines']:.1f} "
                  f"capped={s['capped_games']}/{s['n_games']} "
                  f"({sum(g['seconds'] for g in res['games']):.0f}s)",
                  flush=True)
    elapsed = time.perf_counter() - t0
    print(f"\nAll evaluations finished in {elapsed:.0f}s "
          f"(wall clock, {args.workers} workers)")

    # Rank and print a leaderboard.
    entries = []
    for name, res in per_ckpt.items():
        s = summarize(res["games"])
        s["name"] = name
        s["weights"] = res["weights"]
        entries.append(s)
    entries.sort(key=lambda e: (-e["mean_pieces"], e["name"]))
    print("\n=== Leaderboard (by mean pieces, greedy, fixed seeds) ===")
    print(f"{'checkpoint':40s} {'mean':>8s} {'med':>8s} {'sd':>7s} "
          f"{'min':>5s} {'max':>5s} {'lines':>7s} {'capped':>7s}")
    for e in entries:
        print(f"{e['name']:40s} {e['mean_pieces']:8.1f} {e['median_pieces']:8.1f} "
              f"{e['stdev_pieces']:7.1f} {e['min_pieces']:5d} {e['max_pieces']:5d} "
              f"{e['mean_lines']:7.1f} {e['capped_games']:3d}/{e['n_games']}")

    # Paired per-seed comparison of the top checkpoint vs the rest.
    if len(entries) > 1:
        by_seed = {name: {g["seed"]: g for g in res["games"]}
                   for name, res in per_ckpt.items()}
        best = entries[0]["name"]
        print(f"\n=== Paired per-seed deltas vs {best} (positive = opponent better) ===")
        for e in entries[1:]:
            name = e["name"]
            deltas = [by_seed[name][g["seed"]]["pieces"]
                      - by_seed[best][g["seed"]]["pieces"]
                      for g in per_ckpt[best]["games"]]
            wins = sum(d > 0 for d in deltas)
            ties = sum(d == 0 for d in deltas)
            losses = sum(d < 0 for d in deltas)
            print(f"  {name:40s} mean_delta={statistics.fmean(deltas):+8.1f} "
                  f"W/T/L={wins}/{ties}/{losses}")

    # Optional comparison against a cached previous sweep.
    if args.compare:
        prev = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        prev_by_seed = {}
        for entry in prev if isinstance(prev, list) else prev.get("results", []):
            prev_by_seed[entry["name"]] = {g["seed"]: g["pieces"]
                                           for g in entry["games"]}
        print(f"\n=== Paired deltas vs cached sweep {Path(args.compare).name} ===")
        for e in entries:
            if e["name"] not in prev_by_seed:
                continue
            deltas = [prev_by_seed[e["name"]][g["seed"]]
                      - g["pieces"] for g in per_ckpt[e["name"]]["games"]]
            wins = sum(d > 0 for d in deltas)
            ties = sum(d == 0 for d in deltas)
            losses = sum(d < 0 for d in deltas)
            print(f"  {e['name']:40s} "
                  f"prev_mean={statistics.fmean(prev_by_seed[e['name']].values()):8.1f} "
                  f"now_mean={e['mean_pieces']:8.1f} "
                  f"delta={statistics.fmean(deltas):+8.1f} "
                  f"W/T/L={wins}/{ties}/{losses}")

    if args.json:
        payload = {
            "args": {k: v for k, v in vars(args).items() if k != "weights"},
            "games_per_ckpt": args.games,
            "seed0": args.seed,
            "max_pieces": args.max_pieces,
            "results": [
                {
                    "name": name,
                    "weights": res["weights"],
                    "summary": summarize(res["games"]),
                    "games": res["games"],
                }
                for name, res in sorted(per_ckpt.items())
            ],
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()