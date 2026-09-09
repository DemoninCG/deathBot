"""Stage B: triage winners at cap 1500, 15 tapes (§42).

Arms (all snapshot9, completion bonus everywhere):
  base_loose : commit_loose (16/110/60) — rung-1 CONVERTS config.
  cool30     : + 30-piece post-exit cooldown (triage mover: 0.54 T/100).
  cool30lam10: cool30 + lam 10 in-commit gradient (untested cross —
               triage says combo is suspect; included to bound it, not
               to promote it).
  lam10      : dense Phi λ10 + bonus (standing reference).

Protocol: 15 games, cap 1500, FRESH panel 20862+ (CONSUMED). Windows +
per-game trajectories + FAIL flags (same shape as diag_commit_long).
Pre-registered:
  "promote" — an arm holds ≥0.45 T/100 at ≥14/15 cap with paired dTet
    vs lam10 significant (t≥2): conclusive stage C (best two, 30 tapes
    + cap-3000 leg) on a fresh panel.
  "hold"    — ordering matches triage but nothing separates significantly:
    stage C is base_loose vs cool30 only.
  "regress" — cool30 at/below base_loose on T/100 or cap: cooldown does
    not survive the horizon; keep base_loose, stop tuning.

Usage:
    python diag_commit_stageB.py --weights ../../model_weights_snapshot9.pt \\
        --games 15 --seed0 20862 --cap 1500 --workers 8 \\
        --out ../_diag_out/commit_stageB.json
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

STAGEB_SEED0 = 20862  # next free dev (CONSUMED by this driver)
REPORTING_SEED0 = 31200
COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}

ARMS: dict[str, dict] = {
    "base_loose": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60},
    "cool30": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "cooldown": 30},
    "cool30lam10": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                    "cooldown": 30, "lam": 10.0},
}


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
    (here, weights, arm, seed, cap, window) = payload
    sys.path.insert(0, here)
    import random
    import numpy as _np
    from model import TAPNet as _N
    from two_ply import TwoPlyScorer as _S
    from well_potential import style_fn_factory as _ff
    from well_potential import dense_well_features as _dwf
    from commit_controller import CommitScorer as _C
    from piece_history import PieceQueue as _PQ
    import deathbot_sim as _ds
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    if arm == "lam10":
        scorer = _S(net, beam=100, line_bonus=dict(COMPLETION_BONUS),
                    style_fn=_ff(scale=10.0), style_alpha=1.0)
        lam_style = True
    else:
        base = _S(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS))
        scorer = _C(base, **ARMS[arm])
        lam_style = False
    pq = _PQ(random.Random(seed))
    board = _ds.Board()
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    lines_total = pieces = 0
    reason = "max_pieces"
    tetris_at: list[int] = []
    windows: list[dict] = []

    def _new_acc():
        return {"n": 0, "lines": 0, "tet": 0, "occ": 0, "top": 0.0,
                "holes": 0.0}

    def _flush(a):
        n = max(1, a["n"])
        return {"pieces": a["n"], "lines": a["lines"], "tetrises": a["tet"],
                "mean_occ": a["occ"] / n, "mean_top": a["top"] / n,
                "mean_holes": a["holes"] / n}

    acc = _new_acc()
    off_n = off_d = 0
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
        if not lam_style:
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
    out = {"arm": arm, "seed": seed, "pieces": pieces, "lines": lines_total,
           "clears": {str(k): int(v) for k, v in clears.items()},
           "reason": reason, "windows": windows, "tetris_at": tetris_at,
           "final": {"occ": sum(bin(int(v)).count("1") for v in fin_rows),
                     "top": float(fin_feats["top"]),
                     "holes": float(fin_feats["holes_outside"])},
           "s": round(time.perf_counter() - t0, 1)}
    if not lam_style:
        out["style_off_frac"] = off_n / max(1, off_d)
        out["ctrl_stats"] = scorer.stats
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=STAGEB_SEED0)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve(args.weights))
    arms = ["base_loose", "cool30", "cool30lam10", "lam10"]
    seeds = [args.seed0 + i for i in range(args.games)]
    if args.seed0 < STAGEB_SEED0:
        print(f"[stageB] seed0 {args.seed0} < {STAGEB_SEED0} — refusing.")
        sys.exit(2)
    tag = "reporting-only" if args.seed0 >= REPORTING_SEED0 else \
        f"SELECTION panel {seeds[0]}-{seeds[-1]} (CONSUMED)"
    print(f"[stageB] {tag}; arms {arms}, cap {args.cap}.")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: dict[str, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, a, s, args.cap, args.window)
                for a in arms for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in payloads]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.setdefault(r["arm"], []).append(r)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(payloads)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "seed0": args.seed0, "games": args.games, "cap": args.cap,
           "selection_panel": args.seed0 < REPORTING_SEED0,
           "arms": {}, "wall_seconds": 0.0}
    print(f"\n{'arm':>12s} {'cap':>7s} {'pcs':>7s} {'lines':>7s} "
          f"{'tet':>5s} {'T/100':>6s} {'off':>6s} {'dTet':>12s}")
    ref = {r["seed"]: int(r["clears"].get("4", 0))
           for r in results["lam10"]}
    for arm in arms:
        rows = sorted(results[arm], key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        capn = sum(1 for p in pcs if p >= args.cap)
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        t100 = 100 * tet / max(1, sum(pcs))
        off = ""
        extra = {}
        if rows and "style_off_frac" in rows[0]:
            off = f"{statistics.fmean(r['style_off_frac'] for r in rows):.3f}"
            extra = {"style_off_frac": off}
            if rows[0].get("ctrl_stats"):
                rec = statistics.fmean(
                    r["ctrl_stats"]["recovers"] for r in rows)
                extra["mean_recovers"] = round(rec, 2)
        d = [int(r["clears"].get("4", 0)) - ref[r["seed"]] for r in rows]
        m = statistics.fmean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
        dtxt = f"{m:+.2f} t{m / se:+.2f}" if se and arm != "lam10" else "—"
        print(f"{arm:>12s} {capn:3d}/{len(rows):<3d} "
              f"{statistics.fmean(pcs):7.1f} "
              f"{statistics.fmean(r['lines'] for r in rows):7.1f} "
              f"{tet:5d} {t100:6.3f} {off:>6s} {dtxt:>12s}")
        out["arms"][arm] = {
            "capped": capn, "mean_pieces": round(statistics.fmean(pcs), 1),
            "mean_lines": round(statistics.fmean(
                r["lines"] for r in rows), 1),
            "tetrises": tet, "t_per_100": round(t100, 4),
            "paired_dTet_vs_lam10": dtxt,
            "per_seed": {str(r["seed"]): {
                "pieces": r["pieces"], "lines": r["lines"],
                "clears": r["clears"], "reason": r["reason"],
                "windows": r["windows"], "tetris_at": r["tetris_at"],
                "final": r["final"],
                **({"style_off_frac": r.get("style_off_frac"),
                    "ctrl_stats": r.get("ctrl_stats")}
                   if "style_off_frac" in r else {})} for r in rows},
            **extra}
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
