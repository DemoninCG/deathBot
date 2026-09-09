"""Rung-1 screen: commitment controller vs λ10/λ40 (§38.1).

Arms (all snapshot9, full-root expansion, completion bonus everywhere):
  commit_above : CommitScorer(abort_top=14, abort_occ=90, reenter=50)
  commit_loose : CommitScorer(abort_top=16, abort_occ=110, reenter=60)
  lam10        : dense Phi λ10 + bonus (standing stylish reference)
  lam40        : dense Phi λ40 + bonus (rung-0 style king)

Protocol: 30 games, cap 300, FRESH panel 20812+ (CONSUMED). Metrics (§15.2):
cap rate, T/100, Tetris share, ready-frac proxy, style-off frac (commit
arms), paired dTet vs λ10. Pre-registered verdict:
  "converts" — commit arm holds ≥0.4 T/100 with cap rate ≥ λ40 + 8
    (≈23+/30): rung 1 WORKS, proceed to rescue-tuning/long-horizon.
  "neutral"  — style/survival inside the λ10–λ40 segment: controller adds
    nothing over the λ dial; stop, keep λ-ladder Pareto.
  "regresses"— worse than λ-matched baselines on both axes: constraint
    formulation wrong; revisit preserve-predicate/recover triggers.

Usage:
    python diag_commit_eval.py --weights ../../model_weights_snapshot9.pt \\
        --games 30 --seed0 20812 --workers 4 \\
        --out ../_diag_out/commit_eval.json
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

RUNG1_SEED0 = 20812  # next free dev (CONSUMED by this driver)
REPORTING_SEED0 = 31200
COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}


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
    (here, weights, arm, seed, cap) = payload
    sys.path.insert(0, here)
    import random
    from model import TAPNet as _N
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    t0 = time.perf_counter()
    if arm.startswith("commit"):
        from commit_controller import play_game_commit as _pg
        kw = {"commit_above": {"abort_top": 14, "abort_occ": 90,
                               "reenter_occ": 50},
              "commit_loose": {"abort_top": 16, "abort_occ": 110,
                               "reenter_occ": 60}}[arm]
        pieces, lines, clears, reason, stats = _pg(
            net, random.Random(seed), max_pieces=cap, **kw)
        return {"arm": arm, "seed": seed, "pieces": pieces, "lines": lines,
                "clears": {str(k): int(v) for k, v in clears.items()},
                "reason": reason, "s": time.perf_counter() - t0,
                "stats": stats}
    from two_ply import play_game_2ply_style as _pg2
    from well_potential import style_fn_factory as _ff
    lam = {"lam10": 10.0, "lam40": 40.0}[arm]
    pieces, lines, clears, reason = _pg2(
        net, random.Random(seed), beam=100,
        line_bonus=dict(COMPLETION_BONUS), max_pieces=cap,
        style_fn=_ff(scale=float(lam)), style_alpha=1.0)
    return {"arm": arm, "seed": seed, "pieces": pieces, "lines": lines,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=RUNG1_SEED0)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve(args.weights))
    arms = ["commit_above", "commit_loose", "lam10", "lam40"]
    seeds = [args.seed0 + i for i in range(args.games)]
    if args.seed0 < RUNG1_SEED0:
        print(f"[rung1] seed0 {args.seed0} < {RUNG1_SEED0} — refusing.")
        sys.exit(2)
    tag = "reporting-only" if args.seed0 >= REPORTING_SEED0 else \
        f"SELECTION panel {seeds[0]}-{seeds[-1]} (CONSUMED)"
    print(f"[rung1] {tag}; arms {arms}.")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: dict[tuple, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, a, s, args.max_pieces)
                for a in arms for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in payloads]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.setdefault(r["arm"], []).append(r)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(payloads)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    ref = {r["seed"]: r for r in results["lam10"]}
    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "seed0": args.seed0, "games": args.games,
           "max_pieces": args.max_pieces,
           "selection_panel": args.seed0 < REPORTING_SEED0,
           "arms": {}, "wall_seconds": 0.0}
    print(f"\n{'arm':>13s} {'cap':>7s} {'pcs':>7s} {'lines':>7s} "
          f"{'tet':>5s} {'T/100':>6s} {'tshr':>6s} {'off':>6s} "
          f"{'dTet(vs10)':>10s}")
    for arm in arms:
        rows = sorted(results[arm], key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        capn = sum(1 for p in pcs if p >= args.max_pieces)
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        scoring = sum(int(r["clears"].get(k, 0))
                      for r in rows for k in ("1", "2", "3", "4"))
        t100 = 100 * tet / max(1, sum(pcs))
        tshr = tet / max(1, scoring)
        off = ""
        extra = {}
        if rows and "stats" in rows[0]:
            offs = [r["stats"]["style_off"] / max(1, r["stats"]["decisions"])
                    for r in rows]
            rec = [r["stats"]["recovers"] for r in rows]
            fb = [r["stats"]["fallbacks"] for r in rows]
            co = [r["stats"]["cashouts"] for r in rows]
            off = f"{statistics.fmean(offs):.3f}"
            extra = {"style_off_frac": round(statistics.fmean(offs), 4),
                     "mean_recovers": round(statistics.fmean(rec), 2),
                     "mean_fallbacks": round(statistics.fmean(fb), 2),
                     "mean_cashouts": round(statistics.fmean(co), 2)}
        d = [int(r["clears"].get("4", 0))
             - int(ref[r["seed"]]["clears"].get("4", 0)) for r in rows]
        m = statistics.fmean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
        dtxt = f"{m:+.2f} t{m / se:+.2f}" if se and arm != "lam10" else "—"
        print(f"{arm:>13s} {capn:3d}/{len(rows):<3d} "
              f"{statistics.fmean(pcs):7.1f} "
              f"{statistics.fmean(r['lines'] for r in rows):7.1f} "
              f"{tet:5d} {t100:6.3f} {tshr:6.3f} {off:>6s} {dtxt:>10s}")
        out["arms"][arm] = {
            "capped": capn, "mean_pieces": round(statistics.fmean(pcs), 1),
            "mean_lines": round(statistics.fmean(
                r["lines"] for r in rows), 1),
            "tetrises": tet, "t_per_100": round(t100, 4),
            "tetris_share": round(tshr, 4),
            "paired_dTet_vs_lam10": dtxt,
            "per_seed": {str(r["seed"]): {
                "pieces": r["pieces"], "lines": r["lines"],
                "clears": r["clears"], "reason": r["reason"]} for r in rows},
            **extra}
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
