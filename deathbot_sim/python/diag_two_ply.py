"""Style baseline: 1-ply vs full 2-ply Tetris-rate comparison (2026-09-06).

Survival is solved (2-ply caps every game at 1500), so games run at a LOW
cap (default 300 — every game caps) and the signal is STYLE: lines/100,
Tetris counts and Tetris share of scoring moves, from per-game clears
breakdowns. beam=0 is 1-ply greedy; beam=100 is full-expansion 2-ply.

Usage:
    python diag_two_ply.py --weights ../../model_weights_snapshot9.pt \
        --beams 0 100 --games 30 --seed0 20104 --max-pieces 300 --workers 8 \
        --out ../_diag_out/two_ply_style30.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet  # noqa: E402
from two_ply import play_game_2ply_style, style_summary  # noqa: E402


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def _worker_eval(payload) -> dict:
    """Process worker: evaluate one (weights, beam) config on one shard of seeds."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, beam, gamma_ply, seeds, max_pieces) = payload
    sys.path.insert(0, here)
    import random
    from model import TAPNet
    from two_ply import play_game_2ply_style
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    results = []
    for seed in seeds:
        t0 = time.perf_counter()
        pieces, lines, clears, reason = play_game_2ply_style(
            net, random.Random(seed), beam=beam, gamma_ply=gamma_ply,
            max_pieces=max_pieces)
        results.append({"seed": seed, "pieces": pieces, "lines": lines,
                        "clears": {str(k): int(v) for k, v in clears.items()},
                        "reason": reason, "s": time.perf_counter() - t0})
    return {"beam": beam, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--beams", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=17000)
    ap.add_argument("--max-pieces", type=int, default=300,
                    help="cap per game; low (e.g. 300) now survival is solved")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve_weights(args.weights))
    from model import TAPNet
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    del net

    seeds = [args.seed0 + i for i in range(args.games)]
    shards = 8
    per_shard = (len(seeds) + shards - 1) // shards
    seed_shards = [seeds[i:i + per_shard]
                   for i in range(0, len(seeds), per_shard)]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_results: dict[int, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, beam, args.gamma_ply, shard,
                 args.max_pieces)
                for beam in args.beams for shard in seed_shards]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_worker_eval, p) for p in payloads]):
            r = fut.result()
            all_results.setdefault(r["beam"], []).extend(r["results"])
    elapsed = time.perf_counter() - t0

    out = {"weights": weights, "games": args.games, "seed0": args.seed0,
           "max_pieces": args.max_pieces,
           "gamma_ply": args.gamma_ply, "configs": {}, "wall_s": round(elapsed, 1)}
    print(f"{'beam':>5s} {'pieces':>7s} {'lines':>7s} {'l/100':>6s} "
          f"{'tetris':>6s} {'tshr':>5s} {'cap':>4s} {'s/g':>6s}")
    for beam in args.beams:
        rows = sorted(all_results.get(beam, []), key=lambda r: r["seed"])
        pieces = [r["pieces"] for r in rows]
        lines = [r["lines"] for r in rows]
        secs = [r["s"] for r in rows]
        capped = sum(1 for r in rows if r["pieces"] >= args.max_pieces)
        tet = sum(int(r.get("clears", {}).get("4", 0)) for r in rows)
        scoring = sum(int(r.get("clears", {}).get(k, 0))
                      for r in rows for k in ("1", "2", "3", "4"))
        tshr = (tet / scoring) if scoring else 0.0
        l100 = 100.0 * sum(lines) / max(1, sum(pieces))
        print(f"{beam:5d} {statistics.fmean(pieces):7.1f} "
              f"{statistics.fmean(lines):7.1f} {l100:6.1f} "
              f"{tet:6d} {tshr:5.2f} {capped:4d} "
              f"{statistics.fmean(secs):6.1f}")
        out["configs"][str(beam)] = {
            "mean_pieces": statistics.fmean(pieces),
            "median_pieces": statistics.median(pieces),
            "mean_lines": statistics.fmean(lines),
            "lines_per_100": l100,
            "tetrises": tet,
            "tetris_share": tshr,
            "capped": capped,
            "per_seed": {str(r["seed"]): r["pieces"] for r in rows},
            "per_seed_lines": {str(r["seed"]): r["lines"] for r in rows},
            "per_seed_clears": {str(r["seed"]): r.get("clears", {})
                                for r in rows},
            "sec_per_game": round(statistics.fmean(secs), 2),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} (wall {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
