"""Round C — decisive fresh-panel test of Round A/B controller controls (§45/§46).

Selection-grade (NOT reporting-only): 30 FRESH matched tapes 20960-20989.
Two legs on the SAME seeds:
  leg1 cap 300  (style triage, all arms — cheap, confirms/rejects large harms)
  leg2 cap 1500 (survival + sustained-style decision, 4 arms only)

Arms (all on the corrected masked-root controller; baseline = corrected
cool30lam10 exactly):
  baseline    : abort16/110, reenter60, cooldown30, lam10
  watch80     : + progress_patience 80 (designated-well progress watchdog)
  reentry_cert: + reenter_top10 / reenter_holes2 / reentry_min_progress1.0
                (earned re-entry certificate)
  ready8      : + ready_patience 8 (READY/I-drought timeout)
  Cap-300-only reject-confirm / inactivity-check arms (leg1 only):
  reenter_top10, cashout_q, debt80, uniform

Endpoints per §44.1-C: paired cap (discordant pairs + McNemar exact),
paired pieces, paired Tetris deltas (mean + t), T/100, T per capped game,
longest no-Tetris gap, mode transitions (recovers/reentries), constrained
picks/fallbacks, per-control firing counts (debt_exits/progress_exits/
ready_exits/ranked_cashouts/denied_top/cool/holes/geometry), off-frac.

Pre-registered read (Round C gate):
  - A control PROMOTES as the new stylish default iff on leg2 (cap 1500)
    cap-rate separates (paired pieces t>=2 or discordant exact p<=0.05)
    or style separates (paired dTet t>=2) with cap tied or better.
  - debt80 / uniform CONFIRM-REJECT iff leg1 shows large style harm
    (paired dTet t<=-2) or any cap loss.
  - reenter_top10 / cashout_q judged INACTIVE (not negative) iff firing
    counts are ~zero at both horizons; no promotion, no rejection.
Otherwise the corrected cool30lam10 baseline stands and no rerun follows.

Usage:
    python diag_commit_roundC.py --weights ../../model_weights_snapshot9.pt \
        --seed0 20960 --games 30 --cap1 300 --cap2 1500 --workers 8 \
        --out ../_diag_out/commit_roundC.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROUND_C_SEED0 = 20960
COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}

ARMS: dict[str, dict] = {
    "baseline": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                 "cooldown": 30, "lam": 10.0},
    "watch80": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                "cooldown": 30, "lam": 10.0, "progress_patience": 80},
    "reentry_cert": {"abort_top": 16, "abort_occ": 110,
                     "reenter_occ": 60, "reenter_top": 10,
                     "reenter_holes": 2, "reentry_min_progress": 1.0,
                     "cooldown": 30, "lam": 10.0},
    "ready8": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "cooldown": 30, "lam": 10.0, "ready_patience": 8},
    # leg-1-only arms:
    "reenter_top10": {"abort_top": 16, "abort_occ": 110,
                      "reenter_occ": 60, "reenter_top": 10,
                      "cooldown": 30, "lam": 10.0},
    "cashout_q": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                  "cooldown": 30, "lam": 10.0, "rank_cashouts": True},
    "debt80": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "cooldown": 30, "lam": 10.0, "debt_limit": 80.0},
    "uniform": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                "cooldown": 30, "lam": 10.0,
                "uniform_commit_scoring": True},
}
LEG2_ARMS = ["baseline", "watch80", "reentry_cert", "ready8"]


def resolve(path: str) -> Path:
    supplied = Path(path).expanduser()
    for candidate in (supplied, HERE.parent / supplied,
                      HERE.parent.parent / supplied,
                      HERE.parent.parent.parent / supplied):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _fnv(rows) -> int:
    h = 0x811C9DC5
    for v in rows:
        h ^= int(v) & 0xFFFFFFFF
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _worker(payload) -> dict:
    import os
    import random

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    here, weights, arm, seed, cap, window = payload
    sys.path.insert(0, here)
    import numpy as _np
    from model import TAPNet as _N
    from two_ply import TwoPlyScorer as _S
    from commit_controller import CommitScorer as _C
    from piece_history import PieceQueue as _PQ
    from well_potential import dense_well_features as _dwf
    import deathbot_sim as _ds

    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    base = _S(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS))
    scorer = _C(base, **ARMS[arm])
    pq = _PQ(random.Random(seed))
    board = _ds.Board()
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    lines_total = pieces = 0
    reason = "max_pieces"
    tetris_at: list[int] = []
    windows: list[dict] = []

    def _new_acc():
        return {"n": 0, "lines": 0, "tet": 0, "occ": 0,
                "top": 0.0, "holes": 0.0}

    def _flush(a):
        n = max(1, a["n"])
        return {"pieces": a["n"], "lines": a["lines"],
                "tetrises": a["tet"], "mean_occ": a["occ"] / n,
                "mean_top": a["top"] / n, "mean_holes": a["holes"] / n}

    acc = _new_acc()
    off_n = off_d = 0
    chain = 0x811C9DC5
    first_hash = None
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
        rows = _np.asarray(bulk[4], dtype=_np.uint32).reshape(
            n_cands, 21)[pick]
        if first_hash is None:
            first_hash = _fnv(rows)
        chain ^= _fnv(rows)
        chain = (chain * 0x01000193) & 0xFFFFFFFF
        nl = int(bulk[5][pick])
        lines_total += nl
        clears[nl] = clears.get(nl, 0) + 1
        if nl == 4:
            tetris_at.append(pieces + 1)
        occ = sum(bin(int(v)).count("1") for v in rows)
        feats = _dwf(rows)
        acc["n"] += 1
        acc["lines"] += nl
        if nl == 4:
            acc["tet"] += 1
        acc["occ"] += occ
        acc["top"] += float(feats["top"])
        acc["holes"] += float(feats["holes_outside"])
        off_d += 1
        off_n += int(not scorer.committed)
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
    marks = [0] + tetris_at + [pieces]
    gap = max([marks[i + 1] - marks[i]
               for i in range(len(marks) - 1)] or [pieces])
    return {"arm": arm, "seed": seed, "pieces": pieces,
            "lines": lines_total,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "windows": windows, "tetris_at": tetris_at,
            "longest_no_tetris_gap": gap,
            "final": {"occ": sum(bin(int(v)).count("1")
                                 for v in fin_rows),
                      "top": float(fin_feats["top"]),
                      "holes": float(fin_feats["holes_outside"])},
            "board_chain_head": first_hash, "board_chain": chain,
            "style_off_frac": off_n / max(1, off_d),
            "ctrl_stats": scorer.stats,
            "s": round(time.perf_counter() - t0, 1)}


def _paired_t(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, 0.0
    sd = statistics.stdev(xs)
    return m, (m / (sd / len(xs) ** 0.5) if sd else 0.0)


def _mcnemar_exact(n10: int, n01: int) -> float:
    # Two-sided exact binomial p for discordant cap pairs.
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1))
               / 2.0 ** n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seed0", type=int, default=ROUND_C_SEED0)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--cap1", type=int, default=300)
    ap.add_argument("--cap2", type=int, default=1500)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.seed0 != ROUND_C_SEED0:
        print(f"[roundC] refusing: seed0 must be {ROUND_C_SEED0} "
              f"(next free); got {args.seed0}")
        sys.exit(2)
    weights = str(resolve(args.weights))
    seeds = [args.seed0 + i for i in range(args.games)]
    leg1_arms = list(ARMS)
    leg2_arms = list(LEG2_ARMS)
    print(f"[roundC] SELECTION {seeds[0]}-{seeds[-1]}; "
          f"leg1 cap{args.cap1} arms={leg1_arms}; "
          f"leg2 cap{args.cap2} arms={leg2_arms}")
    t0 = time.perf_counter()

    def run_leg(arms: list[str], cap: int) -> dict:
        res: dict[str, list[dict]] = {}
        pays = [(str(HERE), weights, a, s, cap, args.window)
                for a in arms for s in seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_worker, p) for p in pays]
            done = 0
            for f in as_completed(futs):
                r = f.result()
                res.setdefault(r["arm"], []).append(r)
                done += 1
                if done % 20 == 0:
                    print(f"  cap{cap}: {done}/{len(pays)} "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
        return res

    leg1 = run_leg(leg1_arms, args.cap1)
    leg2 = run_leg(leg2_arms, args.cap2)

    def summarize(res: dict, cap: int) -> dict:
        summ: dict = {}
        base_tet = {r["seed"]: int(r["clears"].get("4", 0))
                    for r in res.get("baseline", [])}
        base_pcs = {r["seed"]: r["pieces"]
                    for r in res.get("baseline", [])}
        base_cap = {r["seed"]: (r["pieces"] >= cap)
                    for r in res.get("baseline", [])}
        for arm, rows0 in res.items():
            rows = sorted(rows0, key=lambda r: r["seed"])
            pcs = [r["pieces"] for r in rows]
            capn = sum(1 for p in pcs if p >= cap)
            tet = sum(int(r["clears"].get("4", 0)) for r in rows)
            t100 = 100 * tet / max(1, sum(pcs))
            capped_t = [int(r["clears"].get("4", 0)) for r in rows
                        if r["pieces"] >= cap]
            t_per_capped = (sum(capped_t) / len(capped_t)
                            if capped_t else 0.0)
            gaps = [r["longest_no_tetris_gap"] for r in rows]
            if arm == "baseline":
                mt = tt = mp = tp = 0.0
                d = []
                dp = []
            else:
                d = [int(r["clears"].get("4", 0)) - base_tet[r["seed"]]
                     for r in rows]
                dp = [r["pieces"] - base_pcs[r["seed"]] for r in rows]
                mt, tt = _paired_t([float(x) for x in d])
                mp, tp = _paired_t([float(x) for x in dp])
            a_cap = {r["seed"]: (r["pieces"] >= cap) for r in rows}
            n10 = sum(1 for s in base_cap
                      if a_cap.get(s) and not base_cap.get(s))
            n01 = sum(1 for s in base_cap
                      if (not a_cap.get(s)) and base_cap.get(s))
            p = _mcnemar_exact(n10, n01)
            cs = [r["ctrl_stats"] for r in rows if r.get("ctrl_stats")]
            extra = {}
            if cs:
                extra = {
                    "style_off_frac": round(statistics.fmean(
                        r["style_off_frac"] for r in rows), 3),
                    "mean_recovers": round(statistics.fmean(
                        c["recovers"] for c in cs), 2),
                    "mean_reentries": round(statistics.fmean(
                        c["reentries"] for c in cs), 2),
                    "mean_fallbacks": round(statistics.fmean(
                        c["fallbacks"] for c in cs), 2),
                    "mean_cashouts": round(statistics.fmean(
                        c["cashouts"] for c in cs), 3),
                    "sum_debt_exits": sum(
                        c.get("debt_exits", 0) for c in cs),
                    "sum_progress_exits": sum(
                        c.get("progress_exits", 0) for c in cs),
                    "sum_ready_exits": sum(
                        c.get("ready_exits", 0) for c in cs),
                    "sum_ranked_cashouts": sum(
                        c.get("ranked_cashouts", 0) for c in cs),
                    "sum_denied_top": sum(
                        c.get("reentry_denied_top", 0) for c in cs),
                    "sum_denied_cool": sum(
                        c.get("reentry_denied_cool", 0) for c in cs),
                    "sum_denied_holes": sum(
                        c.get("reentry_denied_holes", 0) for c in cs),
                    "sum_denied_geometry": sum(
                        c.get("reentry_denied_geometry", 0) for c in cs),
                }
            print(f"{arm:>14s} cap{cap}: {capn:3d}/{len(rows)} "
                  f"pcs{statistics.fmean(pcs):7.1f} "
                  f"tet{tet:4d} T/100 {t100:.3f} T/cap {t_per_capped:.2f} "
                  f"gap{statistics.fmean(gaps):.0f} dTet {mt:+.2f} "
                  f"t{tt:+.2f} dPc {mp:+.1f} t{tp:+.2f} "
                  f"disc {n10}/{n01} p={p:.3f}")
            summ[arm] = {
                "capped": capn, "mean_pieces": round(
                    statistics.fmean(pcs), 1),
                "mean_lines": round(statistics.fmean(
                    r["lines"] for r in rows), 1),
                "tetrises": tet, "t_per_100": round(t100, 4),
                "t_per_capped": round(t_per_capped, 3),
                "mean_longest_gap": round(statistics.fmean(gaps), 1),
                "paired_dTet": round(mt, 3),
                "paired_dTet_t": round(tt, 2),
                "paired_dPieces": round(mp, 1),
                "paired_dPieces_t": round(tp, 2),
                "discordant_cap_vs_baseline": f"{n10}/{n01}",
                "mcnemar_exact_p": round(p, 4),
                "per_seed": {
                    str(r["seed"]): {
                        "pieces": r["pieces"], "lines": r["lines"],
                        "clears": r["clears"], "reason": r["reason"],
                        "tetris_at": r["tetris_at"],
                        "longest_no_tetris_gap":
                            r["longest_no_tetris_gap"],
                        "final": r["final"],
                        "board_chain_head": r["board_chain_head"],
                        "board_chain": r["board_chain"],
                        "style_off_frac": r.get("style_off_frac"),
                        "ctrl_stats": r.get("ctrl_stats"),
                        "windows": r["windows"],
                    } for r in rows},
                **extra}
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
           "selection_panel": True, "arms_config": ARMS,
           "legs": {"cap300": s1, "cap1500": s2},
           "wall_seconds": round(time.perf_counter() - t0, 1)}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[roundC] wrote {args.out}")


if __name__ == "__main__":
    main()
