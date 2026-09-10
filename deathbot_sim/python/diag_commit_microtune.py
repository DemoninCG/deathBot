"""Step-4 micro-tune (§44.4.1): abort/re-entry near the Stage-C winner (NO training).

Winner: corrected cool30lam10 (abort_top 16, abort_occ 110, reenter 60,
cooldown 30, lam 10). Four cells only:
  abort_occ {110, 120} x reenter_occ {50, 60} (top 16, cooldown 30, lam 10 fixed).
Protocol: 30 FRESH tapes at cap 300 (cheap style screen), MATCHED across
cells. Metrics: cap rate, T/100, T/cap, off-frac, recovers/reentries.
Pre-registered: promote a cell iff cap>=28/30 AND T/100 beats the winner
estimate; else carry the winner forward with no revisit.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
CELLS = {
    "a110_r50": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 50, "cooldown": 30, "lam": 10.0},
    "a110_r60": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0},
    "a120_r50": {"abort_top": 16, "abort_occ": 120, "reenter_occ": 50, "cooldown": 30, "lam": 10.0},
    "a120_r60": {"abort_top": 16, "abort_occ": 120, "reenter_occ": 60, "cooldown": 30, "lam": 10.0},
}
STAGEC_MINI_SEED0 = 20877
REPORTING_SEED0 = 31200


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(p)


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, cell, seed, cap) = payload
    sys.path.insert(0, here)
    import random
    import numpy as _np
    from model import TAPNet as _N
    from two_ply import TwoPlyScorer as _S
    from commit_controller import CommitScorer as _C, COMPLETION_BONUS as _CB
    from piece_history import PieceQueue as _PQ
    import deathbot_sim as _ds
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    base = _S(net, beam=10 ** 9, line_bonus=dict(_CB))
    ctrl = _C(base, **CELLS[cell])
    pq = _PQ(random.Random(seed))
    board = _ds.Board()
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    pieces = 0
    reason = "max_pieces"
    t0 = time.perf_counter()
    while pieces < cap:
        bulk = board.lock_candidates_full(pq.current)
        n = len(bulk[0])
        if n == 0:
            reason = "no_candidates"
            break
        if all(bulk[3]):
            reason = "lockout"
            break
        pick = ctrl.choose(board, pq.current, pq.next_piece)
        nl = int(bulk[5][pick])
        clears[nl] = clears.get(nl, 0) + 1
        rows = _np.asarray(bulk[4], dtype=_np.uint32).reshape(n, 21)[pick]
        board = _ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    return {"cell": cell, "seed": seed, "pieces": pieces,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "s": round(time.perf_counter() - t0, 1),
            "style_off_frac": ctrl.stats["style_off"] / max(1, ctrl.stats["decisions"]),
            "stats": dict(ctrl.stats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = str(resolve(args.weights))
    seeds = [args.seed0 + i for i in range(args.games)]
    tag = "reporting-only" if args.seed0 >= REPORTING_SEED0 else f"SELECTION {seeds[0]}-{seeds[-1]}"
    print(f"[micro] {tag}; cells {list(CELLS)} x {args.games} tapes, cap {args.cap}.")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    res: dict[str, list] = {}
    pays = [(str(HERE), weights, c, s, args.cap) for c in CELLS for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in pays]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            res.setdefault(r["cell"], []).append(r)
            done += 1
            if done % 30 == 0:
                print(f"  {done}/{len(pays)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
           "seed0": args.seed0, "games": args.games, "cap": args.cap,
           "cells": CELLS, "summary": {}, "wall_seconds": 0.0}
    for cell, rows0 in res.items():
        rows = sorted(rows0, key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        print(f"{cell:>10s}: cap {sum(1 for p in pcs if p >= args.cap)}/{len(rows)} "
              f"pcs {statistics.fmean(pcs):.1f} tet {tet} "
              f"T/100 {100*tet/max(1, sum(pcs)):.3f} "
              f"off {statistics.fmean(r['style_off_frac'] for r in rows):.3f}")
        out["summary"][cell] = {
            "capped": sum(1 for p in pcs if p >= args.cap),
            "mean_pieces": round(statistics.fmean(pcs), 1),
            "tetrises": tet,
            "t_per_100": round(100 * tet / max(1, sum(pcs)), 4),
            "style_off_frac": round(statistics.fmean(r["style_off_frac"] for r in rows), 3),
            "per_seed": {str(r["seed"]): {"pieces": r["pieces"], "clears": r["clears"],
                                          "reason": r["reason"]} for r in rows}}
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[micro] wrote {args.out}")


if __name__ == "__main__":
    main()
