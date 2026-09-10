"""a120_r50 follow-up (§44.6 extension): paired vs style default a110_r60.

Two legs on the SAME 30 fresh tapes 20990-21019 (SELECTION, CONSUMED):
  leg1 cap 300  (style significance: paired dTet, t)
  leg2 cap 1500 (long-horizon survival + sustained style: cap, pieces,
                 paired dTet/dPc, discordant pairs + McNemar exact)

Arms (corrected masked-root controller, snapshot9 weights):
  a110_r60 = abort16/110, reenter60, cooldown30, lam10 (current default)
  a120_r50 = abort16/120, reenter50, cooldown30, lam10 (micro-tune lean)

Gate: promote a120_r50 iff leg2 cap separates (paired pieces t>=2 or
discordant exact p<=0.05) or leg1+leg2 style separates (paired dTet t>=2
on either leg) with cap tied-or-better on leg2. Else hold a110_r60.
"""
from __future__ import annotations
import argparse, hashlib, json, math, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
ARMS = {
    "a110_r60": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                 "cooldown": 30, "lam": 10.0},
    "a120_r50": {"abort_top": 16, "abort_occ": 120, "reenter_occ": 50,
                 "cooldown": 30, "lam": 10.0},
}
FOLLOWUP_SEED0 = 20990


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
    return {"arm": arm, "seed": seed, "pieces": pieces,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "s": round(time.perf_counter() - t0, 1),
            "style_off_frac": ctrl.stats["style_off"] / max(1, ctrl.stats["decisions"]),
            "stats": dict(ctrl.stats)}


def _paired_t(xs):
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, 0.0
    sd = statistics.stdev(xs)
    return m, (m / (sd / len(xs) ** 0.5) if sd else 0.0)


def _mcnemar_exact(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seed0", type=int, default=FOLLOWUP_SEED0)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--cap1", type=int, default=300)
    ap.add_argument("--cap2", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.seed0 != FOLLOWUP_SEED0:
        print(f"[a120] refusing: seed0 must be {FOLLOWUP_SEED0}")
        sys.exit(2)
    weights = str(resolve(args.weights))
    seeds = [args.seed0 + i for i in range(args.games)]
    print(f"[a120] SELECTION {seeds[0]}-{seeds[-1]}; "
          f"a110_r60 vs a120_r50; cap1 {args.cap1}, cap2 {args.cap2}.")
    t0 = time.perf_counter()

    def run_leg(cap):
        res = {}
        pays = [(str(HERE), weights, a, s, cap)
                for a in ARMS for s in seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_worker, p) for p in pays]
            done = 0
            for f in as_completed(futs):
                r = f.result()
                res.setdefault(r["arm"], []).append(r)
                done += 1
                if done % 20 == 0:
                    print(f"  cap{cap}: {done}/{len(pays)} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
        return res

    leg1 = run_leg(args.cap1)
    leg2 = run_leg(args.cap2)

    def summarize(res, cap):
        summ = {}
        ref = {r["seed"]: r for r in res["a110_r60"]}
        for arm in ARMS:
            rows = sorted(res[arm], key=lambda r: r["seed"])
            pcs = [r["pieces"] for r in rows]
            capn = sum(1 for p in pcs if p >= cap)
            tet = sum(int(r["clears"].get("4", 0)) for r in rows)
            t100 = 100 * tet / max(1, sum(pcs))
            if arm == "a110_r60":
                mt = tt = mp = tp = 0.0
            else:
                d = [int(r["clears"].get("4", 0))
                     - int(ref[r["seed"]]["clears"].get("4", 0)) for r in rows]
                dp = [r["pieces"] - ref[r["seed"]]["pieces"] for r in rows]
                mt, tt = _paired_t([float(x) for x in d])
                mp, tp = _paired_t([float(x) for x in dp])
            a_cap = {r["seed"]: (r["pieces"] >= cap) for r in rows}
            b_cap = {s: (ref[s]["pieces"] >= cap) for s in ref}
            n10 = sum(1 for s in ref if a_cap.get(s) and not b_cap.get(s))
            n01 = sum(1 for s in ref if (not a_cap.get(s)) and b_cap.get(s))
            p = _mcnemar_exact(n10, n01)
            print(f"{arm:>10s} cap{cap}: {capn}/{len(rows)} "
                  f"pcs{statistics.fmean(pcs):.1f} tet{tet} T/100 {t100:.3f} "
                  f"dTet {mt:+.2f} t{tt:+.2f} dPc {mp:+.1f} t{tp:+.2f} "
                  f"disc {n10}/{n01} p={p:.3f}")
            summ[arm] = {"capped": capn,
                         "mean_pieces": round(statistics.fmean(pcs), 1),
                         "tetrises": tet, "t_per_100": round(t100, 4),
                         "paired_dTet": round(mt, 3),
                         "paired_dTet_t": round(tt, 2),
                         "paired_dPieces": round(mp, 1),
                         "paired_dPieces_t": round(tp, 2),
                         "discordant": f"{n10}/{n01}",
                         "mcnemar_p": round(p, 4),
                         "style_off_frac": round(statistics.fmean(
                             r["style_off_frac"] for r in rows), 3),
                         "per_seed": {str(r["seed"]): r for r in rows}}
        return summ

    print("--- leg1 cap300 ---")
    s1 = summarize(leg1, args.cap1)
    print("--- leg2 cap1500 ---")
    s2 = summarize(leg2, args.cap2)
    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "seed0": args.seed0, "games": args.games,
           "cap1": args.cap1, "cap2": args.cap2,
           "selection_panel": True, "arms": ARMS,
           "legs": {"cap300": s1, "cap1500": s2},
           "wall_seconds": round(time.perf_counter() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[a120] wrote {args.out}")


if __name__ == "__main__":
    main()
