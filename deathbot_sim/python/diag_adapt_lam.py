"""Usability-λ probe (reporting-only): fixed cool30lam10 vs adaptive λ.

Adaptive λ ("watch80-style"): per-decision Phi scale interpolates
lam_lo (risky stack) -> lam_hi (clean stack) via CommitScorer._usability
(top<=12 & holes==0 & well open => 1.0; top>=16 or holes>=4 => 0.0).
Two narrow cells only:
  adapt_lo0_hi20 : lam 0 (risky) -> 20 (clean)
  adapt_lo5_hi15 : lam 5 (risky) -> 15 (clean)
Baseline: corrected cool30lam10 (fixed lam 10), abort16/110, reenter60,
cooldown30. Cap 300, 10 REPORTING tapes 32010-32019, matched.
Reporting-only: never a selection panel, never a promotion claim.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPORTING_SEED0 = 32010
ARMS = {
    "baseline": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                 "cooldown": 30, "lam": 10.0},
    "adapt_lo0_hi20": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                       "cooldown": 30, "lam": 10.0,
                       "lam_lo": 0.0, "lam_hi": 20.0},
    "adapt_lo5_hi15": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                       "cooldown": 30, "lam": 10.0,
                       "lam_lo": 5.0, "lam_hi": 15.0},
    "adapt_lo0_hi30": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                       "cooldown": 30, "lam": 10.0,
                       "lam_lo": 0.0, "lam_hi": 30.0},
    "adapt_lo0_hi40": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                       "cooldown": 30, "lam": 10.0,
                       "lam_lo": 0.0, "lam_hi": 40.0},
}


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
    (here, weights, arm, seed, cap) = payload
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
    ctrl = _C(base, **ARMS[arm])
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
    lam_n = ctrl.stats.get("lam_eff_n", 0)
    out = {"arm": arm, "seed": seed, "pieces": pieces,
           "clears": {str(k): int(v) for k, v in clears.items()},
           "reason": reason, "s": round(time.perf_counter() - t0, 1),
           "style_off_frac": ctrl.stats["style_off"] / max(1, ctrl.stats["decisions"]),
           "mean_lam_eff": (round(ctrl.stats.get("lam_eff_sum", 0.0) / lam_n, 2)
                            if lam_n else None),
           "stats": dict(ctrl.stats)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seed0", type=int, default=REPORTING_SEED0)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.seed0 < REPORTING_SEED0:
        print(f"[adapt] refusing: reporting-only, seed0 >= {REPORTING_SEED0}")
        sys.exit(2)
    weights = str(resolve(args.weights))
    seeds = [args.seed0 + i for i in range(args.games)]
    print(f"[adapt] reporting-only {seeds[0]}-{seeds[-1]}; "
          f"arms {list(ARMS)} cap {args.cap}.")
    t0 = time.perf_counter()
    res = {}
    pays = [(str(HERE), weights, a, s, args.cap)
            for a in ARMS for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in pays]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            res.setdefault(r["arm"], []).append(r)
            done += 1
            if done % len(ARMS) == 0:
                print(f"  {done}/{len(pays)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    base = {r["seed"]: int(r["clears"].get("4", 0)) for r in res["baseline"]}
    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "seed0": args.seed0, "games": args.games, "cap": args.cap,
           "reporting_only": True, "arms": ARMS, "summary": {},
           "wall_seconds": 0.0}
    for arm, rows0 in res.items():
        rows = sorted(rows0, key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        if arm == "baseline":
            dt = "—"
        else:
            d = [int(r["clears"].get("4", 0)) - base[r["seed"]] for r in rows]
            m = statistics.fmean(d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            t = m / (sd / len(d) ** 0.5) if sd else 0.0
            dt = f"{m:+.2f} t{t:+.2f}"
        lam = [r["mean_lam_eff"] for r in rows if r["mean_lam_eff"] is not None]
        print(f"{arm:>14s}: cap {sum(1 for p in pcs if p >= args.cap)}/{len(rows)} "
              f"tet {tet} T/100 {100*tet/max(1, sum(pcs)):.3f} dTet {dt} "
              f"mean_lam {[round(statistics.fmean(lam), 2)] if lam else ['fixed']}")
        for r in rows:
            b = base[r["seed"]]
            a = int(r["clears"].get("4", 0))
            if a != b or r["pieces"] < args.cap:
                print(f"    seed {r['seed']}: base {b} vs {arm} {a} "
                      f"(d{a-b:+d}) {r['pieces']} {r['reason']}")
        out["summary"][arm] = {"capped": sum(1 for p in pcs if p >= args.cap),
                               "tetrises": tet,
                               "t_per_100": round(100 * tet / max(1, sum(pcs)), 4),
                               "per_seed": {str(r["seed"]): r for r in rows}}
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[adapt] wrote {args.out}")


if __name__ == "__main__":
    main()
