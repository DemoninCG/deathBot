"""Long-horizon sustainability probe for the stylish teacher (§18.1).

Compares full-2-ply baseline (lam=0, bonus-free champion) vs lam=10/20
(dense Phi + completion bonus) on MATCHED tapes at cap 1500, recording
per-window occupancy / height / holes / well / Tetris metrics.

Question: does the stylish policy reach a stable build–cash-out cycle,
or does it accumulate backlog that cap-300 screens never bill?
(B_end = 4P - 10L: lam=20 retains ~92 vs ~19 blocks at cap 300.)
Decision rule: occupancy/risk stabilize across windows while Tetrises
continue → teacher viable; steady backlog growth or late deaths →
teacher rejected, back to §18 order item 3 (factorial Phi×bonus,
death-rejection ablation).

NOTE (confound, deliberate): lam>0 arms carry BOTH dense Phi and the
completion bonus (same config as the §17.1 sweep); lam=0 carries
neither (pure champion). Factorial separation is priority 3, separate run.

Usage:
    python diag_long_sustain.py --weights ../../model_weights_snapshot9.pt \
        --lambdas 0 10 20 --games 10 --seed0 20510 --cap 1500 \
        --window 200 --workers 4 --out ../_diag_out/long_sustain.json
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


def _popcount(rows) -> int:
    return sum(bin(int(v)).count("1") for v in rows)


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, lam, seed, cap, window) = payload
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
    style_fn = None if lam == 0 else _ff(scale=float(lam))
    bonus = None if lam == 0 else dict(COMPLETION_BONUS)
    scorer = _S(net, beam=100, line_bonus=bonus, style_fn=style_fn,
                style_alpha=1.0 if lam != 0 else 0.0)
    rng = random.Random(seed)
    pq = _PQ(rng)
    board = _ds.Board()
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    lines_total = pieces = 0
    reason = "max_pieces"
    windows: list[dict] = []
    acc = None  # current-window accumulator

    def _new_acc():
        return {"n": 0, "lines": 0, "tet": 0, "occ": 0.0, "top": 0.0,
                "holes": 0.0, "depth": 0.0, "open": 0.0, "ready": 0.0,
                "phi": 0.0,
                "clears": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}

    def _flush(a):
        n = max(1, a["n"])
        return {"pieces": a["n"], "lines": a["lines"], "tetrises": a["tet"],
                "clears": {str(k): int(v) for k, v in a["clears"].items()},
                "mean_occ": a["occ"] / n, "mean_top": a["top"] / n,
                "mean_holes": a["holes"] / n,
                "mean_well_depth": a["depth"] / n,
                "frac_well_open": a["open"] / n,
                "frac_tetris_ready": a["ready"] / n,
                "mean_phi": a["phi"] / n}

    tetris_at: list[int] = []
    t0 = time.perf_counter()
    acc = _new_acc()
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
        acc["clears"][nl] += 1
        if nl == 4:
            acc["tet"] += 1
            tetris_at.append(pieces + 1)
        acc["occ"] += _popcount(rows)
        acc["top"] += float(feats["top"])
        acc["holes"] += float(feats["holes_outside"])
        acc["depth"] += float(feats["well_depth_raw"])
        acc["open"] += float(feats["well_open"])
        acc["ready"] += float(feats["tetris_ready"])
        acc["phi"] += float(_wp(rows))
        board = _ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
        if pieces % window == 0:
            windows.append(_flush(acc))
            acc = _new_acc()
    if acc["n"]:
        windows.append(_flush(acc))
    fin_rows = _np.asarray(board.row_bits(), dtype=_np.uint32)
    fin_feats = _dwf(fin_rows)
    return {"lam": lam, "seed": seed, "pieces": pieces, "lines": lines_total,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "windows": windows, "tetris_at": tetris_at,
            "final": {"occ": _popcount(fin_rows),
                      "top": float(fin_feats["top"]),
                      "holes": float(fin_feats["holes_outside"])},
            "s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 10.0, 20.0])
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=20510)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve_weights(args.weights))
    seeds = [args.seed0 + i for i in range(args.games)]
    print(f"long-sustain: lambdas={args.lambdas} games={args.games} "
          f"seeds {seeds[0]}-{seeds[-1]} cap={args.cap} "
          f"window={args.window} workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: dict[float, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, lam, s, args.cap, args.window)
                for lam in args.lambdas for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in payloads]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.setdefault(r["lam"], []).append(r)
            done += 1
            if done % args.workers == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} games "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0

    out = {"weights": weights, "lambdas": args.lambdas, "games": args.games,
           "seed0": args.seed0, "cap": args.cap, "window": args.window,
           "completion_bonus": COMPLETION_BONUS, "arms": {},
           "wall_s": round(elapsed, 1)}
    print(f"\n{'lam':>6s} {'cap':>7s} {'pcs':>7s} {'deaths':>14s} "
          f"{'tet':>5s} {'T/100':>6s} {'linT%':>6s} "
          f"{'occ_w0':>7s} {'occ_wL':>7s} {'top_w0':>6s} {'top_wL':>6s}")
    for lam in args.lambdas:
        rows = sorted(results[lam], key=lambda r: r["seed"])
        pieces = [r["pieces"] for r in rows]
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        lines = sum(r["lines"] for r in rows)
        t100 = 100.0 * tet / max(1, sum(pieces))
        lin_tet = (4 * tet) / max(1, lines)
        cap_n = sum(1 for p in pieces if p >= args.cap)
        deaths = {str(r["seed"]): f"{r['reason']}@{r['pieces']}"
                  for r in rows if r["reason"] != "max_pieces"}
        # window-aligned means across contributing games
        maxw = max(len(r["windows"]) for r in rows)
        wocc, wtop, wtet, wn = [], [], [], []
        for w in range(maxw):
            contrib = [r["windows"][w] for r in rows
                       if len(r["windows"]) > w]
            wocc.append(statistics.fmean([c["mean_occ"] for c in contrib]))
            wtop.append(statistics.fmean([c["mean_top"] for c in contrib]))
            wtet.append(sum(c["tetrises"] for c in contrib))
            wn.append(len(contrib))
        print(f"{lam:6.1f} {cap_n:3d}/{len(rows):<3d} "
              f"{statistics.fmean(pieces):7.1f} {len(deaths):>14d} "
              f"{tet:5d} {t100:6.2f} {lin_tet:6.3f} "
              f"{wocc[0]:7.1f} {wocc[-1]:7.1f} {wtop[0]:6.1f} {wtop[-1]:6.1f}")
        out["arms"][str(lam)] = {
            "lambda": lam, "capped": cap_n,
            "mean_pieces": statistics.fmean(pieces),
            "deaths": deaths, "tetrises": tet,
            "tetris_per_100": t100, "lines_in_tetris_frac": lin_tet,
            "window_mean_occ": wocc, "window_mean_top": wtop,
            "window_tetrises": wtet, "window_n": wn,
            "per_seed": {str(r["seed"]): {
                "pieces": r["pieces"], "lines": r["lines"],
                "clears": r["clears"], "reason": r["reason"],
                "windows": r["windows"], "tetris_at": r["tetris_at"],
                "final": r["final"], "s": r["s"]} for r in rows},
        }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} (wall {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
