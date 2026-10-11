"""Stylish-champion lineage re-eval (2026-09-15, post-§54 physics).

Re-evaluates each previous T/100 champion (or last-close challenger) on the
SAME 20 fresh tapes @cap 1500, head-to-head. Pure style read: Tetris rate
at horizon under corrected TAP rotation.

Cells (weights x policy — the lineage winners):
  snap9_lam10      snap9  + full 2-ply beam100 + Phi(λ10) + completion bonus
  snap9_cross      snap9  + corrected cool30lam10 (abort16/110/re60/cd30/lam10)
  snap12_lam10     snap12 + same pure-λ λ10 policy (stable fallback)
  snap12_cross     snap12 + corrected cool30lam10 (pre-tier default)
  snap12_tiered    snap12 + cool30lam10tiered (tiered PROMOTION, §53.8)
  snap12_burn6     snap12 + cool30lam10tiered_burn6 (current champ, §58)

Why these six: lam10 is the pure-λ champion (§18/§23, last pure-λ point
with λ10-level survival); cool30lam10 is the Stage-B/C stylish default
(§42 PROMOTE, §44.4 corrected point estimate, §47 Round-C 27/30 @0.60);
tiered is the interaction-only promotion (§53.7 +13.25 t+6.69, §53.8
+22.7 t+6.89 PROMOTED); burn6 is the current default (§58 +18.3 t+3.47
PROMOTED). snap9 rows isolate the physics hit (§54) on the old chassis;
snap12 rows rank the current-chassis frontier. commit_loose approx and
pure-λ λ20/30/40/50/60 are NOT re-run (superseded/closed, §§39–44/§53).

Harness: identical game loop for every cell (lock_candidates_full +
scorer.choose + clears/tetris_at/windows/board-hash chain, same as
diag_commit_stageC._worker). commit cells use the CORRECTED masked-root
controller (§44.3); lam10 cells use TwoPlyScorer beam=100 + Phi + bonus
(the §17.1/§38 recipe). Post-§54 physics throughout.

Panel: 21930-21949 (20 fresh tapes, CONSUMED on run). 5500+ untouched.

Usage (from workspace root, venv):
    .venv\\Scripts\\python.exe deathbot_sim\\python\\diag_style_lineage.py ^
        --seed0 21930 --games 20 --cap 1500 --workers 8 ^
        --out deathbot_sim\\_diag_out\\style_lineage_21930.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEED0 = 21930  # next free dev range (21730-21929 CONSUMED by §59.3)
COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}

CELLS = {
    "snap9_lam10": {"weights": "model_weights_snapshot9.pt", "kind": "lam10"},
    "snap9_cross": {"weights": "model_weights_snapshot9.pt", "kind": "commit",
                    "commit": {"abort_top": 16, "abort_occ": 110,
                               "reenter_occ": 60, "cooldown": 30,
                               "lam": 10.0}},
    "snap12_lam10": {"weights": "model_weights_snapshot12.pt", "kind": "lam10"},
    "snap12_cross": {"weights": "model_weights_snapshot12.pt", "kind": "commit",
                     "commit": {"abort_top": 16, "abort_occ": 110,
                                "reenter_occ": 60, "cooldown": 30,
                                "lam": 10.0}},
    "snap12_tiered": {"weights": "model_weights_snapshot12.pt", "kind": "commit",
                      "commit": {"abort_top": 16, "abort_occ": 110,
                                 "reenter_occ": 60, "cooldown": 30,
                                 "lam": 10.0, "tiered_preserve": True}},
    "snap12_burn6": {"weights": "model_weights_snapshot12.pt", "kind": "commit",
                     "commit": {"abort_top": 16, "abort_occ": 110,
                                "reenter_occ": 60, "cooldown": 30,
                                "lam": 10.0, "tiered_preserve": True,
                                "burn_penalty": 6.0}},
}


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"weights not found: {p}")


def _fnv(rows) -> int:
    h = 0x811C9DC5
    for v in rows:
        h ^= int(v) & 0xFFFFFFFF
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    (here, weights, cell, seed, cap, window) = payload
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
    cfg = CELLS[cell]
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    lam_style = False
    if cfg["kind"] == "lam10":
        scorer = _S(net, beam=100, line_bonus=dict(COMPLETION_BONUS),
                    style_fn=_ff(scale=10.0), style_alpha=1.0)
        lam_style = True
    else:
        base = _S(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS))
        scorer = _C(base, **cfg["commit"])
    pq = _PQ(random.Random(seed))
    board = _ds.Board()
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    lines_total = pieces = 0
    reason = "max_pieces"
    tetris_at = []
    windows = []

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
        rows = _np.asarray(bulk[4], dtype=_np.uint32).reshape(n_cands, 21)[pick]
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
    marks = [0] + tetris_at + [pieces]
    gap = max([marks[i + 1] - marks[i] for i in range(len(marks) - 1)]
              or [pieces])
    out = {"cell": cell, "seed": seed, "pieces": pieces,
           "lines": lines_total,
           "clears": {str(k): int(v) for k, v in clears.items()},
           "reason": reason, "windows": windows, "tetris_at": tetris_at,
           "longest_no_tetris_gap": gap,
           "final": {"occ": sum(bin(int(v)).count("1") for v in fin_rows),
                     "top": float(fin_feats["top"]),
                     "holes": float(fin_feats["holes_outside"])},
           "board_chain_head": first_hash, "board_chain": chain,
           "s": round(time.perf_counter() - t0, 1)}
    if not lam_style:
        out["style_off_frac"] = off_n / max(1, off_d)
        out["ctrl_stats"] = scorer.stats
    return out


def _paired_t(xs):
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    se = sd / len(xs) ** 0.5 if len(xs) > 1 else 0.0
    return m, (m / se if se else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cells", type=str, default="all",
                    help="comma-separated subset of " + ",".join(CELLS)
                    + " (default all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.seed0 < SEED0:
        print(f"[lineage] seed0 {args.seed0} < {SEED0} — refusing "
              f"(registry rules).")
        sys.exit(2)
    cells = list(CELLS) if args.cells.strip().lower() == "all" else \
        [c.strip() for c in args.cells.split(",") if c.strip()]
    for c in cells:
        if c not in CELLS:
            print(f"[lineage] unknown cell {c!r}")
            sys.exit(2)
    weights = {c: str(resolve(CELLS[c]["weights"])) for c in cells}
    for c in cells:
        print(f"  - {c}: {CELLS[c]['weights']} "
              f"sha={hashlib.sha256(Path(weights[c]).read_bytes()).hexdigest()[:12]} "
              f"{CELLS[c]['kind']}")
    seeds = [args.seed0 + i for i in range(args.games)]
    print(f"[lineage] SELECTION {seeds[0]}-{seeds[-1]} (CONSUMED); "
          f"{len(cells)} cells x {args.games} @cap {args.cap}.")

    t0 = time.perf_counter()
    res: dict[str, list] = {}
    pays = [(str(HERE), weights[c], c, s, args.cap, args.window)
            for c in cells for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in pays]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            res.setdefault(r["cell"], []).append(r)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(pays)} ({time.perf_counter() - t0:.0f}s)",
                      flush=True)

    champ = "snap12_burn6"
    ref = {r["seed"]: int(r["clears"].get("4", 0))
           for r in res.get(champ, [])}
    refp = {r["seed"]: r["pieces"] for r in res.get(champ, [])}
    summ = {}
    print(f"\n{'cell':>14s} {'cap':>7s} {'pcs':>7s} {'tet':>5s} "
          f"{'T/100':>6s} {'T/cap':>6s} {'gap':>4s} {'dTet':>12s} "
          f"{'dPc':>12s} {'disc':>7s}")
    for cell in cells:
        rows = sorted(res[cell], key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        capn = sum(1 for p in pcs if p >= args.cap)
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        t100 = 100 * tet / max(1, sum(pcs))
        capped_t = [int(r["clears"].get("4", 0)) for r in rows
                    if r["pieces"] >= args.cap]
        tpc = (sum(capped_t) / len(capped_t)) if capped_t else 0.0
        gaps = [r["longest_no_tetris_gap"] for r in rows]
        d = [int(r["clears"].get("4", 0)) - ref[r["seed"]] for r in rows] \
            if cell != champ and ref else []
        dp = [r["pieces"] - refp[r["seed"]] for r in rows] \
            if cell != champ and refp else []
        mt, tt = _paired_t(d) if d else (0.0, 0.0)
        mp, tp = _paired_t(dp) if dp else (0.0, 0.0)
        disc = ""
        if cell != champ and refp:
            a_cap = {r["seed"]: (r["pieces"] >= args.cap) for r in rows}
            n10 = sum(1 for s in refp if a_cap.get(s) and not (refp[s] >= args.cap))
            n01 = sum(1 for s in refp if (not a_cap.get(s)) and (refp[s] >= args.cap))
            disc = f"{n10}/{n01}"
        extra = {}
        if rows and "style_off_frac" in rows[0]:
            extra["style_off_frac"] = round(
                statistics.fmean(r["style_off_frac"] for r in rows), 3)
            cs = [r["ctrl_stats"] for r in rows if r.get("ctrl_stats")]
            if cs:
                extra["mean_recovers"] = round(
                    statistics.fmean(c["recovers"] for c in cs), 2)
                extra["mean_reentries"] = round(
                    statistics.fmean(c["reentries"] for c in cs), 2)
                extra["mean_fallbacks"] = round(
                    statistics.fmean(c["fallbacks"] for c in cs), 2)
                extra["mean_cashouts"] = round(
                    statistics.fmean(c["cashouts"] for c in cs), 3)
        print(f"{cell:>14s} {capn:3d}/{len(rows)} "
              f"{statistics.fmean(pcs):7.1f} {tet:5d} {t100:6.3f} "
              f"{tpc:6.2f} {statistics.fmean(gaps):4.0f} "
              f"{mt:+6.2f} t{tt:+5.2f} {mp:+7.1f} t{tp:+5.2f} {disc:>7s}")
        summ[cell] = {"weights": weights[cell],
                      "sha256": hashlib.sha256(
                          Path(weights[cell]).read_bytes()).hexdigest(),
                      "kind": CELLS[cell]["kind"],
                      "commit": CELLS[cell].get("commit"),
                      "capped": capn,
                      "mean_pieces": round(statistics.fmean(pcs), 1),
                      "mean_lines": round(statistics.fmean(
                          r["lines"] for r in rows), 1),
                      "tetrises": tet, "t_per_100": round(t100, 4),
                      "t_per_capped": round(tpc, 3),
                      "mean_longest_gap": round(statistics.fmean(gaps), 1),
                      "paired_dTet_vs_burn6": round(mt, 3),
                      "paired_dTet_t": round(tt, 2),
                      "paired_dPieces_vs_burn6": round(mp, 1),
                      "paired_dPieces_t": round(tp, 2),
                      "discordant_cap_vs_burn6": disc,
                      "per_seed": {
                          str(r["seed"]): {
                              "pieces": r["pieces"], "lines": r["lines"],
                              "clears": r["clears"], "reason": r["reason"],
                              "windows": r["windows"],
                              "tetris_at": r["tetris_at"],
                              "longest_no_tetris_gap":
                                  r["longest_no_tetris_gap"],
                              "final": r["final"],
                              "board_chain_head": r["board_chain_head"],
                              "board_chain": r["board_chain"],
                              **({"style_off_frac": r.get("style_off_frac"),
                                  "ctrl_stats": r.get("ctrl_stats")}
                                 if "style_off_frac" in r else {})}
                          for r in rows},
                      **extra}
    out = {"seed0": args.seed0, "games": args.games, "cap": args.cap,
           "selection_panel": True, "cells": summ,
           "wall_seconds": round(time.perf_counter() - t0, 1)}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[lineage] wrote {args.out}")


if __name__ == "__main__":
    main()
