"""Well-seeking teacher screen (§14 step 1): well_weight sweep at cap 300.

Design: full-expansion 2-ply (beam=100) + survivors-only well_setup_score
bonus (w in {0, 5, 15, 40} z-units — sub-margin → dominant vs ~10.5
within-state spread). Same 30-game cap-300 style protocol as the k-sweep:
survival gate (30/30 cap) + Tetris-share signal, paired per-seed lines vs
the bonus-free arm on FRESH panel 20224+.

well_setup_score in [0,1]: one deep (>=6) clean well, no holes, no second
well >3, scaled down above height 14. Rewards the SETUP state, not the
payoff (cf. line_bonus, which failed in §13.2).

Usage:
    python diag_well_bonus.py --weights ../../model_weights_snapshot9.pt \
        --weights-list 0 5 15 40 --games 30 --seed0 20224 \
        --out ../_diag_out/well_bonus.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet  # noqa: E402
from two_ply import play_game_2ply_style, well_setup_score  # noqa: E402


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, well_w, seed, cap) = payload
    sys.path.insert(0, here)
    import random
    from model import TAPNet as _N
    from two_ply import play_game_2ply_style as _pg
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    t0 = time.perf_counter()
    stats: dict = {}
    pieces, lines, clears, reason = _pg(
        net, random.Random(seed), beam=100, well_weight=well_w,
        max_pieces=cap, well_stats=stats)
    mean_well = stats.get("sum", 0.0) / max(1, stats.get("n", 0))
    return {"well_w": well_w, "seed": seed, "pieces": pieces, "lines": lines,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "mean_well": mean_well, "reason": reason,
            "s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--well-weights", nargs="+", type=float,
                    default=[0.0, 5.0, 15.0, 40.0])
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=20224)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve_weights(args.weights))
    seeds = [args.seed0 + i for i in range(args.games)]
    shards = 8
    seed_shards = [seeds[i:i + shards]
                   for i in range(0, len(seeds), shards)]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_results: dict[float, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, w, s, args.max_pieces)
                for w in args.well_weights for shard in seed_shards
                for s in shard]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_worker, p) for p in payloads]):
            r = fut.result()
            all_results.setdefault(r["well_w"], []).append(r)
    elapsed = time.perf_counter() - t0

    base_lines = {r["seed"]: r["lines"]
                  for r in all_results[args.well_weights[0]]}
    out = {"weights": weights, "games": args.games, "seed0": args.seed0,
           "max_pieces": args.max_pieces, "arms": {},
           "wall_s": round(elapsed, 1)}
    print(f"{'w':>6s} {'pieces':>7s} {'lines':>7s} {'l/100':>6s} "
          f"{'tetris':>6s} {'tshr':>6s} {'cap':>4s} {'well':>6s} {'s/g':>6s} | paired dLines t")
    for w in args.well_weights:
        rows = sorted(all_results[w], key=lambda r: r["seed"])
        pieces = [r["pieces"] for r in rows]
        lines = [r["lines"] for r in rows]
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        scoring = sum(int(r["clears"].get(k, 0)) for r in rows
                      for k in ("1", "2", "3", "4"))
        tshr = tet / scoring if scoring else 0.0
        l100 = 100.0 * sum(lines) / max(1, sum(pieces))
        cap = sum(1 for p in pieces if p >= args.max_pieces)
        mw = statistics.fmean([r["mean_well"] for r in rows])
        if w == args.well_weights[0]:
            extra = "—"
        else:
            dl = [r["lines"] - base_lines[r["seed"]] for r in rows]
            md = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            se = sd / (len(dl) ** 0.5)
            extra = f"{md:+.1f} t{md / se:+.2f}" if se else f"{md:+.1f} t—"
        print(f"{w:6.1f} {statistics.fmean(pieces):7.1f} "
              f"{statistics.fmean(lines):7.1f} {l100:6.1f} {tet:6d} "
              f"{tshr:6.3f} {cap:4d} {mw:6.3f} "
              f"{statistics.fmean([r['s'] for r in rows]):6.1f} | {extra}")
        out["arms"][str(w)] = {
            "well_weight": w,
            "mean_pieces": statistics.fmean(pieces),
            "mean_lines": statistics.fmean(lines),
            "lines_per_100": l100, "tetrises": tet, "tetris_share": tshr,
            "capped": cap, "mean_well": mw,
            "per_seed": {str(r["seed"]): r["pieces"] for r in rows},
            "per_seed_lines": {str(r["seed"]): r["lines"] for r in rows},
            "per_seed_clears": {str(r["seed"]): r["clears"] for r in rows},
        }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} (wall {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
