"""Forced-well style ladder, rung 0: λ30/λ40 cap-300 rule-out (§37.4/§38).

Design: full-expansion 2-ply (beam=100) + designated-well dense Phi
(well_potential.py, fixed col 9) as style_fn + guideline completion bonus
{1:1,2:3,3:5,4:8} on surviving branches only — the EXACT §17.1 dense-sweep
recipe, extended to λ ∈ {0, 10, 30, 40}. λ=0 arm is bonus-free (zero-style
contract check: must reproduce the champion 30/30 cap). All λ>0 arms carry
the same completion bonus.

Protocol: 30 games, cap 300, FRESH panel 20752+ (CONSUMED after — the rung
consumes its panel whatever the outcome, per registry rules). Metrics
(§15.2): cap rate (survival gate), Tetrises/100, Tetris share + conversion
proxies (tetris_ready fraction), mean Phi, paired per-seed lines/Tetrises
vs the λ=10 arm.

Verdict bins (pre-registered):
  "saturates" — λ30/40 T/100 ≈ λ10/λ20 with worse cap rate: pure-λ forcing
    is exhausted; proceed to rung 1 (commitment controller), not λ80.
  "climbs"    — T/100 materially above λ20 (≥0.3) with cap rate intact:
    λ is NOT saturated; consider λ60 before building the controller.
  "collapses" — cap rate craters (λ20 pattern): hard forcing needs rescue
    from the start; skip to rung 1 with rescue on.

Usage:
    python diag_lambda_high.py --weights ../../model_weights_snapshot9.pt \\
        --lambdas 0 10 30 40 --games 30 --seed0 20752 --workers 8 \\
        --out ../_diag_out/lambda_high.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
RUNG0_SEED0 = 20752  # next free dev (CONSUMED by this driver)
REPORTING_SEED0 = 31200


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for cand in (sup, HERE.parent / sup, HERE.parent.parent / sup,
                 HERE.parent.parent.parent / sup):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {p}")


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, lam, seed, cap) = payload
    sys.path.insert(0, here)
    import random
    import numpy as _np
    from model import TAPNet as _N
    from two_ply import play_game_2ply_style as _pg
    from well_potential import style_fn_factory as _ff
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    style_fn = None if lam == 0 else _ff(scale=float(lam))
    bonus = None if lam == 0 else dict(COMPLETION_BONUS)
    t0 = time.perf_counter()
    ready_n = decided_n = 0
    base = style_fn

    def tracked(packed, lines):
        nonlocal ready_n, decided_n
        from well_potential import dense_well_features as _dwf
        vals = base(packed, lines)
        # tetris_ready fraction over the viable set (conversion proxy)
        for rows in packed:
            try:
                if _dwf(rows)["tetris_ready"] > 0.5:
                    ready_n += 1
            except Exception:
                pass
            decided_n += 1
        return vals

    pieces, lines, clears, reason = _pg(
        net, random.Random(seed), beam=100, line_bonus=bonus,
        max_pieces=cap, style_fn=tracked if style_fn else None,
        style_alpha=1.0 if style_fn else 0.0)
    return {"lam": lam, "seed": seed, "pieces": pieces, "lines": lines,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "ready_frac": (ready_n / max(1, decided_n)),
            "reason": reason, "s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.0, 10.0, 30.0, 40.0])
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=RUNG0_SEED0)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve(args.weights))
    seeds = [args.seed0 + i for i in range(args.games)]
    if args.seed0 < RUNG0_SEED0:
        print(f"[rung0] seed0 {args.seed0} < {RUNG0_SEED0} — refusing "
              f"(registry rules).")
        sys.exit(2)
    tag = "reporting-only" if args.seed0 >= REPORTING_SEED0 else \
        f"SELECTION panel {seeds[0]}-{seeds[-1]} (CONSUMED)"
    print(f"[rung0] {tag}; lambdas {args.lambdas}.")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: dict[tuple, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, lam, s, args.max_pieces)
                for lam in args.lambdas for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in payloads]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.setdefault(r["lam"], []).append(r)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(payloads)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    ref = {r["seed"]: r for r in results.get(10.0, [])}
    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "seed0": args.seed0, "games": args.games,
           "max_pieces": args.max_pieces,
           "selection_panel": args.seed0 < REPORTING_SEED0,
           "arms": {}, "wall_seconds": 0.0}
    print(f"\n{'lam':>6s} {'cap':>7s} {'pcs':>7s} {'lines':>7s} "
          f"{'tet':>5s} {'T/100':>6s} {'tshr':>6s} {'ready':>6s} "
          f"{'dTet(vs10)':>10s}")
    for lam in args.lambdas:
        rows = sorted(results[lam], key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        capn = sum(1 for p in pcs if p >= args.max_pieces)
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        scoring = sum(int(r["clears"].get(k, 0))
                      for r in rows for k in ("1", "2", "3", "4"))
        t100 = 100 * tet / max(1, sum(pcs))
        tshr = tet / max(1, scoring)
        ready = statistics.fmean(r["ready_frac"] for r in rows)
        dtxt = "—"
        if lam != 10.0 and ref:
            d = [int(r["clears"].get("4", 0))
                 - int(ref[r["seed"]]["clears"].get("4", 0)) for r in rows]
            m = statistics.fmean(d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
            dtxt = f"{m:+.2f} t{m / se:+.2f}" if se else f"{m:+.2f} t—"
        print(f"{lam:6.0f} {capn:3d}/{len(rows):<3d} "
              f"{statistics.fmean(pcs):7.1f} "
              f"{statistics.fmean(r['lines'] for r in rows):7.1f} "
              f"{tet:5d} {t100:6.3f} {tshr:6.3f} {ready:6.3f} {dtxt:>10s}")
        out["arms"][str(lam)] = {
            "capped": capn, "mean_pieces": round(statistics.fmean(pcs), 1),
            "mean_lines": round(statistics.fmean(
                r["lines"] for r in rows), 1),
            "tetrises": tet, "t_per_100": round(t100, 4),
            "tetris_share": round(tshr, 4),
            "mean_ready_frac": round(ready, 4),
            "paired_dTet_vs_lam10": dtxt,
            "per_seed": {str(r["seed"]): {
                "pieces": r["pieces"], "lines": r["lines"],
                "clears": r["clears"], "reason": r["reason"]} for r in rows}}
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
