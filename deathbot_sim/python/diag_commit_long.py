"""Rung-1 long-horizon screen: commit_loose vs λ10 at cap 1500 (§39).

Question (§39 gating): does the abandon/rebuild cycle SUSTAIN over long
games, or does backlog accumulate across recoveries? E4's 7/7 rescue +
flat λ10 bands predict sustain; λ20's monotone drift warns.

Arms (all snapshot9, completion bonus everywhere):
  commit_loose : CommitScorer(abort_top=16, abort_occ=110, reenter=60)
                 — the rung-1 CONVERTS config (0.52 T/100 at 30/30).
  lam10        : dense Phi λ10 + bonus (standing stylish reference).

Protocol: 20 games, cap 1500, FRESH panel 20842+ (CONSUMED), 200-piece
windows (occ/top/holes/lines/tet per window — same instrumented shape as
diag_lam10_gate.py): pooled bands + PER-GAME trajectories (the §20.1
qualification — pooled means hid λ10's +20 slopes in §23). Pre-registered:
  "sustains"  — commit cap rate ≥ lam10 − 1 with T/100 ≥ 2× lam10 and no
    game with sustained positive occ slope + no release (the §31 FAIL
    flag, applied to BOTH arms): proceed to cap-3000 confirmation.
  "accumulates" — commit shows sustained-slope games lam10 lacks:
    backlog builds across recoveries; retune abort/reenter before any
    longer horizon.
  "neutral"   — both arms 20/20 with T/100 inside noise: style does not
    survive the horizon; rung-1 stays a cap-300 result.

Usage:
    python diag_commit_long.py --weights ../../model_weights_snapshot9.pt \\
        --games 20 --seed0 20842 --cap 1500 --workers 4 \\
        --out ../_diag_out/commit_long.json
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

LONG_SEED0 = 20842  # next free dev (CONSUMED by this driver)
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
    if arm == "commit":
        base = _S(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS))
        scorer = _C(base, abort_top=16, abort_occ=110, reenter_occ=60)
        lam_style = False
    else:
        scorer = _S(net, beam=100, line_bonus=dict(COMPLETION_BONUS),
                    style_fn=_ff(scale=10.0), style_alpha=1.0)
        lam_style = True
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
    off_frac_n = off_frac_d = 0
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
            off_frac_d += 1
            off_frac_n += int(not scorer.committed)
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
        out["style_off_frac"] = off_frac_n / max(1, off_frac_d)
        out["ctrl_stats"] = scorer.stats
    return out


def _fail_flag(windows) -> bool:
    occ = [w["mean_occ"] for w in windows]
    d = [occ[i + 1] - occ[i] for i in range(len(occ) - 1)]
    run = cur = 0
    for x in d:
        cur = cur + 1 if x >= 2.0 else 0
        run = max(run, cur)
    rel = sum(1 for x in d if x <= -20.0)
    return run >= 2 and rel == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=LONG_SEED0)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve(args.weights))
    arms = ["commit", "lam10"]
    seeds = [args.seed0 + i for i in range(args.games)]
    if args.seed0 < LONG_SEED0:
        print(f"[commit-long] seed0 {args.seed0} < {LONG_SEED0} — refusing.")
        sys.exit(2)
    tag = "reporting-only" if args.seed0 >= REPORTING_SEED0 else \
        f"SELECTION panel {seeds[0]}-{seeds[-1]} (CONSUMED)"
    print(f"[commit-long] {tag}; arms {arms}, cap {args.cap}.")

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
            print(f"  {done}/{len(payloads)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "seed0": args.seed0, "games": args.games, "cap": args.cap,
           "selection_panel": args.seed0 < REPORTING_SEED0,
           "arms": {}, "wall_seconds": 0.0}
    print(f"\n{'arm':>8s} {'cap':>7s} {'pcs':>7s} {'lines':>7s} "
          f"{'tet':>5s} {'T/100':>6s} {'off':>6s} {'FAIL':>4s} "
          f"{'dTet':>12s}")
    for arm in arms:
        rows = sorted(results[arm], key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        capn = sum(1 for p in pcs if p >= args.cap)
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        t100 = 100 * tet / max(1, sum(pcs))
        fails = sum(1 for r in rows if _fail_flag(r["windows"]))
        off = ""
        if rows and "style_off_frac" in rows[0]:
            off = f"{statistics.fmean(r['style_off_frac'] for r in rows):.3f}"
        out["arms"][arm] = {
            "capped": capn, "mean_pieces": round(statistics.fmean(pcs), 1),
            "mean_lines": round(statistics.fmean(
                r["lines"] for r in rows), 1),
            "tetrises": tet, "t_per_100": round(t100, 4),
            "fail_flagged": fails, "style_off_frac": off or None,
            "per_seed": {str(r["seed"]): {
                "pieces": r["pieces"], "lines": r["lines"],
                "clears": r["clears"], "reason": r["reason"],
                "windows": r["windows"], "tetris_at": r["tetris_at"],
                "final": r["final"],
                **({"style_off_frac": r.get("style_off_frac"),
                    "ctrl_stats": r.get("ctrl_stats")} if "style_off_frac"
                   in r else {})} for r in rows}}
        print(f"{arm:>8s} {capn:3d}/{len(rows):<3d} "
              f"{statistics.fmean(pcs):7.1f} "
              f"{statistics.fmean(r['lines'] for r in rows):7.1f} "
              f"{tet:5d} {t100:6.3f} {off:>6s} {fails:4d}")
    # paired dTet
    ref = {r["seed"]: int(r["clears"].get("4", 0))
           for r in results["lam10"]}
    rows = sorted(results["commit"], key=lambda r: r["seed"])
    d = [int(r["clears"].get("4", 0)) - ref[r["seed"]] for r in rows]
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
    print(f"commit vs lam10 dTet: {m:+.2f} t{m / se:+.2f}" if se
          else f"commit vs lam10 dTet: {m:+.2f} t—")
    out["paired_dTet_commit_vs_lam10"] = {
        "mean": round(m, 2), "t": round(m / se, 2) if se else None,
        "deltas": d}
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
