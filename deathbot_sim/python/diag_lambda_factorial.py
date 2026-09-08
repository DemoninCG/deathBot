"""Lambda10 factorial with per-game traces (§20 item 3, §20.7; FRESH 20550+).

Separates the §17.1/§18.1 confound (lam>0 arms carried BOTH dense Phi and
the completion bonus; lam=0 carried neither): does the completion reward
release inventory, or merely increase clears? Four matched-tape arms at
cap 1500 with per-window AND per-game occupancy traces (pooled windows can
hide individual upward trends at n=10 — §20.1 qualification folded in
here, not a separate pass):

  phi0_b0  : lam=0,  no bonus (pure champion)
  phi0_b8  : lam=0,  completion bonus {1:1,2:3,3:5,4:8}
  phi10_b0 : lam=10, no bonus (Phi only)
  phi10_b8 : lam=10, completion bonus (viable stylish candidate)

Metrics: cap rate, pieces, clears/Tetrises, T/100, paired dLines vs phi0_b0,
window means (occ/top) + PER-GAME occ trajectories (slope of occ over the
last 4 windows per game: stable vs accumulating), Tetris continuation
(windows with Tetrises), final occ/top/holes.

Decision: Phi-only Tetrises without backlog growth -> Phi is the ingredient;
bonus-only change -> completion matters; phi10_b8 ~= phi10_b0 -> bonus is
dead weight on top of Phi (drop it from the teacher config).

Panel: FRESH 20550+ (CONSUMED after — selection). 20520-20549 belongs to
the death ablation (§20 item 2).

Usage:
    python diag_lambda_factorial.py --weights ../../model_weights_snapshot9.pt \\
        --games 10 --seed0 20550 --cap 1500 --window 200 --workers 4 \\
        --out ../_diag_out/lambda_factorial.json
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
PANEL_START = 20550
ARMS: list[tuple[str, float, bool]] = [
    ("phi0_b0", 0.0, False),
    ("phi0_b8", 0.0, True),
    ("phi10_b0", 10.0, False),
    ("phi10_b8", 10.0, True),
]


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def _popcount(rows) -> int:
    return sum(bin(int(v)).count("1") for v in rows)


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, arm, lam, bonus, seed, cap, window) = payload
    sys.path.insert(0, here)
    import random
    import numpy as _np
    from model import TAPNet as _N
    from two_ply import TwoPlyScorer as _S
    from well_potential import (style_fn_factory as _ff,
                                well_potential as _wp,
                                dense_well_features as _dwf)
    from piece_history import PieceQueue as _PQ
    import deathbot_sim as _ds
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    style_fn = _ff(scale=float(lam)) if lam else None
    scorer = _S(net, beam=100,
                line_bonus=dict(COMPLETION_BONUS) if bonus else None,
                style_fn=style_fn, style_alpha=1.0 if lam else 0.0)
    rng = random.Random(seed)
    pq = _PQ(rng)
    board = _ds.Board()
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    lines_total = pieces = 0
    reason = "max_pieces"
    windows: list[dict] = []
    acc = {"n": 0, "lines": 0, "tet": 0, "occ": 0.0, "top": 0.0}

    def _flush(a):
        n = max(1, a["n"])
        return {"pieces": a["n"], "lines": a["lines"], "tetrises": a["tet"],
                "mean_occ": a["occ"] / n, "mean_top": a["top"] / n}
    t0 = time.perf_counter()
    while pieces < cap:
        bulk = board.lock_candidates_full(pq.current)
        n_cands = len(bulk[0])
        if n_cands == 0:
            reason = "no_candidates"
            break
        if all(bulk[3]):
            reason = "lockout"
            break
        pick = scorer.choose(board, pq.current, pq.next_piece)
        rows = _np.asarray(bulk[4], dtype=_np.uint32).reshape(n_cands, 21)[pick]
        nl = int(bulk[5][pick])
        lines_total += nl
        clears[nl] = clears.get(nl, 0) + 1
        feats = _dwf(rows)
        acc["n"] += 1
        acc["lines"] += nl
        if nl == 4:
            acc["tet"] += 1
        acc["occ"] += _popcount(rows)
        acc["top"] += float(feats["top"])
        board = _ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
        if pieces % window == 0:
            windows.append(_flush(acc))
            acc = {"n": 0, "lines": 0, "tet": 0, "occ": 0.0, "top": 0.0}
    if acc["n"]:
        windows.append(_flush(acc))
    fin_rows = _np.asarray(board.row_bits(), dtype=_np.uint32)
    fin_feats = _dwf(fin_rows)
    return {"arm": arm, "seed": seed, "pieces": pieces, "lines": lines_total,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "windows": windows,
            "final": {"occ": _popcount(fin_rows),
                      "top": float(fin_feats["top"]),
                      "holes": float(fin_feats["holes_outside"])},
            "s": round(time.perf_counter() - t0, 1)}


def _occ_slope(windows: list[dict]) -> float:
    """Slope of mean_occ over the last 4 windows (per-game accumulation)."""
    tail = [w["mean_occ"] for w in windows[-4:] if "mean_occ" in w]
    if len(tail) < 2:
        return 0.0
    xs = list(range(len(tail)))
    mx = statistics.fmean(xs)
    my = statistics.fmean(tail)
    denom = sum((x - mx) ** 2 for x in xs)
    if not denom:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, tail)) / denom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=PANEL_START)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve_weights(args.weights))
    if args.seed0 < PANEL_START:
        raise SystemExit(f"refusing consumed panel: seed0 {args.seed0} "
                         f"< next-free {PANEL_START}")
    seeds = [args.seed0 + i for i in range(args.games)]
    print(f"lambda-factorial: arms={[a[0] for a in ARMS]} games={args.games} "
          f"seeds {seeds[0]}-{seeds[-1]} cap={args.cap} "
          f"window={args.window} workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: dict[str, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, name, lam, bon, s, args.cap,
                 args.window)
                for (name, lam, bon) in ARMS for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in payloads]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.setdefault(r["arm"], []).append(r)
            done += 1
            if done % args.workers == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} games "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0

    out: dict = {"weights": weights,
                 "weights_sha256": hashlib.sha256(
                     Path(weights).read_bytes()).hexdigest(),
                 "games": args.games, "seed0": args.seed0, "cap": args.cap,
                 "window": args.window,
                 "completion_bonus": COMPLETION_BONUS,
                 "arms": {}, "wall_s": round(elapsed, 1)}
    base_lines = {r["seed"]: r["lines"] for r in results["phi0_b0"]}
    print(f"\n{'arm':>10s} {'cap':>7s} {'pcs':>7s} {'lines':>7s} "
          f"{'tet':>5s} {'T/100':>6s} {'Tshr':>6s} {'occ_end':>7s} "
          f"{'slope':>7s} {'dLines':>7s}")
    for (name, _lam, _bon) in ARMS:
        rows = sorted(results[name], key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        lns = [r["lines"] for r in rows]
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        scoring = sum(int(r["clears"].get(k, 0)) for r in rows
                      for k in ("1", "2", "3", "4"))
        tshr = tet / scoring if scoring else 0.0
        t100 = 100.0 * tet / max(1, sum(pcs))
        cap_n = sum(1 for p in pcs if p >= args.cap)
        slopes = [_occ_slope(r["windows"]) for r in rows]
        occ_end = statistics.fmean([r["windows"][-1]["mean_occ"]
                                    for r in rows if r["windows"]])
        if name == "phi0_b0":
            extra = "—"
        else:
            dl = [r["lines"] - base_lines[r["seed"]] for r in rows]
            md = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            se = sd / (len(dl) ** 0.5) if len(dl) > 1 else 0.0
            extra = (f"{md:+.1f} t{md / se:+.2f}"
                     if se else f"{md:+.1f} t—")
        print(f"{name:>10s} {cap_n:3d}/{len(rows):<3d} "
              f"{statistics.fmean(pcs):7.1f} {statistics.fmean(lns):7.1f} "
              f"{tet:5d} {t100:6.2f} {tshr:6.3f} {occ_end:7.1f} "
              f"{statistics.fmean(slopes):+7.2f} {extra:>7s}")
        out["arms"][name] = {
            "mean_pieces": statistics.fmean(pcs),
            "mean_lines": statistics.fmean(lns),
            "capped": cap_n, "tetrises": tet,
            "tetris_per_100": t100, "tetris_share": tshr,
            "mean_occ_slope": statistics.fmean(slopes),
            "per_seed": {str(r["seed"]): {
                "pieces": r["pieces"], "lines": r["lines"],
                "clears": r["clears"], "reason": r["reason"],
                "occ_traj": [round(w["mean_occ"], 1)
                             for w in r["windows"]],
                "top_traj": [round(w["mean_top"], 1)
                             for w in r["windows"]],
                "tet_traj": [w["tetrises"] for w in r["windows"]],
                "occ_slope": round(_occ_slope(r["windows"]), 2),
                "final": r["final"], "s": r["s"]} for r in rows},
        }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} (wall {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
