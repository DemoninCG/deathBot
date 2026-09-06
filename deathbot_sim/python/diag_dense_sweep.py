"""Dense-potential lambda sweep (§15.1.3 + §16.3): first screen that can show
Tetrises from empty boards.

Design (§15.1.3): full-expansion 2-ply (beam=100) + designated-well dense
potential Phi (well_potential.py, fixed col 9) as style_fn + guideline
line_bonus completion reward {1:1,2:3,3:5,4:8} on surviving branches only.
Phi rewards the SETUP path (partial credit from depth 1, saturates at 4);
line_bonus rewards cashing out (setup-only objectives punish cash-out, so
the completion reward stays OUTSIDE Phi).

Arms: lambda in {0, 2, 5, 10, 20} (Phi->z scale). lambda=0 arm carries NO
bonus at all (pure champion reference = zero-style contract check; must be
30/30 cap with action agreement). All lambda>0 arms carry the same
completion line_bonus.

Protocol: 30 games, cap 300, FRESH panel 20254+ (CONSUMED after). Metrics
(§15.2): cap rate (survival gate), Tetrises/100, Tetris share, fraction of
cleared lines in Tetrises, mean Phi (setup holding), paired per-seed lines
vs the bonus-free arm.

Usage:
    python diag_dense_sweep.py --weights ../../model_weights_snapshot9.pt \
        --lambdas 0 2 5 10 20 --games 30 --seed0 20254 \
        --out ../_diag_out/dense_sweep.json
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

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}


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
    (here, weights, lam, seed, cap) = payload
    sys.path.insert(0, here)
    import random
    from model import TAPNet as _N
    from two_ply import play_game_2ply_style as _pg
    from well_potential import style_fn_factory as _ff, well_potential as _wp
    import numpy as _np
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    style_fn = None if lam == 0 else _ff(scale=float(lam))
    bonus = None if lam == 0 else dict(COMPLETION_BONUS)
    t0 = time.perf_counter()
    phi_stats: dict = {}
    orig_wp = _wp

    def _tracked(rows, n):
        s = orig_wp(rows, 9)
        phi_stats["sum"] = phi_stats.get("sum", 0.0) + s
        phi_stats["n"] = phi_stats.get("n", 0) + 1
        return s

    # Track mean Phi by wrapping: evaluate post-hoc on the played trajectory
    # is expensive; instead record Phi of each played afterstate via well_stats
    # hook is for well_setup_score — so recompute cheaply: wrap style_fn.
    if style_fn is not None:
        base = style_fn

        def style_fn_tracked(packed, lines):
            vals = base(packed, lines)
            # mean Phi in Phi units (divide by lam)
            phi_stats["sum"] = phi_stats.get("sum", 0.0) + sum(
                v / float(lam) for v in vals) / max(1, len(vals))
            phi_stats["n"] = phi_stats.get("n", 0) + 1
            return vals

        style_fn = style_fn_tracked
    pieces, lines, clears, reason = _pg(
        net, random.Random(seed), beam=100, line_bonus=bonus,
        max_pieces=cap, style_fn=style_fn, style_alpha=1.0)
    mean_phi = phi_stats.get("sum", 0.0) / max(1, phi_stats.get("n", 0))
    return {"lam": lam, "seed": seed, "pieces": pieces, "lines": lines,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "mean_phi": mean_phi, "reason": reason,
            "s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.0, 2.0, 5.0, 10.0, 20.0])
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=20254)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve_weights(args.weights))
    seeds = [args.seed0 + i for i in range(args.games)]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_results: dict[float, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, lam, s, args.max_pieces)
                for lam in args.lambdas for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_worker, p) for p in payloads]):
            r = fut.result()
            all_results.setdefault(r["lam"], []).append(r)
    elapsed = time.perf_counter() - t0

    base_lines = {r["seed"]: r["lines"]
                  for r in all_results[args.lambdas[0]]}
    out = {"weights": weights, "games": args.games, "seed0": args.seed0,
           "max_pieces": args.max_pieces, "completion_bonus": COMPLETION_BONUS,
           "arms": {}, "wall_s": round(elapsed, 1)}
    print(f"{'lam':>6s} {'pieces':>7s} {'lines':>7s} {'tetris':>6s} "
          f"{'t/100':>6s} {'tshr':>6s} {'linTet%':>7s} {'cap':>4s} "
          f"{'mPhi':>6s} {'s/g':>6s} | paired dLines t")
    for lam in args.lambdas:
        rows = sorted(all_results[lam], key=lambda r: r["seed"])
        pieces = [r["pieces"] for r in rows]
        lines = [r["lines"] for r in rows]
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        scoring = sum(int(r["clears"].get(k, 0)) for r in rows
                      for k in ("1", "2", "3", "4"))
        tshr = tet / scoring if scoring else 0.0
        t100 = 100.0 * tet / max(1, sum(pieces))
        lin_tet_frac = (4 * tet) / max(1, sum(lines))
        cap = sum(1 for p in pieces if p >= args.max_pieces)
        mph = statistics.fmean([r["mean_phi"] for r in rows])
        if lam == args.lambdas[0]:
            extra = "—"
        else:
            dl = [r["lines"] - base_lines[r["seed"]] for r in rows]
            md = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            se = sd / (len(dl) ** 0.5)
            extra = f"{md:+.1f} t{md / se:+.2f}" if se else f"{md:+.1f} t—"
        print(f"{lam:6.1f} {statistics.fmean(pieces):7.1f} "
              f"{statistics.fmean(lines):7.1f} {tet:6d} "
              f"{t100:6.2f} {tshr:6.3f} {lin_tet_frac:7.3f} {cap:4d} "
              f"{mph:6.3f} {statistics.fmean([r['s'] for r in rows]):6.1f} "
              f"| {extra}")
        out["arms"][str(lam)] = {
            "lambda": lam,
            "mean_pieces": statistics.fmean(pieces),
            "mean_lines": statistics.fmean(lines),
            "tetrises": tet, "tetris_per_100": t100,
            "tetris_share": tshr,
            "lines_in_tetris_frac": lin_tet_frac,
            "capped": cap, "mean_phi": mph,
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
