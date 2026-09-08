"""λ10 stylish-teacher promotion gate (§23/§27.1 follow-up, 2026-09-08).

§27.1 blocked promotion on two phi10_b8 tail slopes (+20.3, +13.4 out of 10
games) — but the §20.1/§27.1 caution also noted phi0 games show single-game
slopes up to +16 that RELEASE (20551/20559), so tail slope alone is noisy.
This driver replaces the slope eyeball with a decisive paired design:

  Arm A (champ)    : bonus-free full 2-ply (survival champion).
  Arm B (lam10)    : dense Phi (scale 10, col 9) + completion bonus
                     {1:1,2:3,3:5,4:8} — the standing stylish candidate.

N tapes per arm at cap 1500 (+ optional stress cap --cap2), MATCHED seeds.
Per-game record: pieces, lines, clears, reason, per-window occupancy
trajectory (final 4 windows), final occ/top/holes, time-between-Tetrises.

Gate statistics (pre-registered):
  S1 survival : paired cap rate arm B >= arm A (and no unexplained
                deaths); per-seed pieces paired delta >= 0.
  S2 Tetris   : Tetrises/100 paired delta B vs A (direction test, t).
  S3 backlog  : three tiered checks instead of a raw tail-slope threshold —
                (a) per-game max sustained run: max consecutive windows
                (incl. partial tail) with positive dOcc >= 200 blocks*windows;
                (b) release events: windows where occ drops >= 20;
                (c) endgame occ at cap (last full window) vs champion.
                A game is flagged FAIL only if (a) >= 2 AND (b) == 0
                (sustained accumulation with no release) — matching the
                §23 observation that real failures (phi10_b0 slopes +20/+23,
                occ 85-121, no release) look exactly like that.

Panels: --seed0 must be >= 20572 (20572-20591 reserved for the top-k gate
2026-09-08); default seed0 31200 is REPORTING (demo range §28); pass an
explicit fresh range for a selection decision.

Usage:
    python diag_lam10_gate.py --weights ../../model_weights_snapshot9.pt \
        --games 20 --seed0 31200 --cap 1500 --cap2 0 --workers 4 \
        --out ../_diag_out/lam10_gate.json
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
MIN_SELECTION_SEED0 = 20592  # 20572-20591 = top-k gate (2026-09-08)
REPORTING_SEED0 = 31200      # demo/reporting range (§28)


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
    (here, weights, arm, lam, seed, cap, window) = payload
    sys.path.insert(0, here)
    import random
    import numpy as _np
    from model import TAPNet as _N
    from two_ply import TwoPlyScorer as _S
    from well_potential import style_fn_factory as _ff
    from well_potential import dense_well_features as _dwf
    from piece_history import PieceQueue as _PQ
    import deathbot_sim as _ds
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    style_fn = None if lam == 0 else _ff(scale=float(lam))
    bonus = None if lam == 0 else dict(COMPLETION_BONUS)
    scorer = _S(net, beam=100, line_bonus=bonus, style_fn=style_fn,
                style_alpha=1.0 if lam else 0.0)
    rng = random.Random(seed)
    pq = _PQ(rng)
    board = _ds.Board()
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    lines_total = pieces = 0
    reason = "max_pieces"
    windows: list[dict] = []
    tetris_at: list[int] = []

    def _new_acc():
        return {"n": 0, "lines": 0, "tet": 0, "occ": 0, "top": 0.0,
                "holes": 0.0}

    def _flush(a):
        n = max(1, a["n"])
        return {"pieces": a["n"], "lines": a["lines"], "tetrises": a["tet"],
                "mean_occ": a["occ"] / n, "mean_top": a["top"] / n,
                "mean_holes": a["holes"] / n}

    acc = _new_acc()
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
        occ = _popcount(rows)
        feats = _dwf(rows)
        acc["n"] += 1
        acc["lines"] += nl
        if nl == 4:
            acc["tet"] += 1
        acc["occ"] += occ
        acc["top"] += float(feats["top"])
        acc["holes"] += float(feats["holes_outside"])
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
    return {"arm": arm, "seed": seed, "pieces": pieces, "lines": lines_total,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "windows": windows, "tetris_at": tetris_at,
            "final": {"occ": _popcount(fin_rows),
                      "top": float(fin_feats["top"]),
                      "holes": float(fin_feats["holes_outside"])},
            "s": round(time.perf_counter() - t0, 1)}


def max_sustained_gain(occ_traj: list[float], thresh: float = 2.0) -> tuple:
    """Longest run of consecutive windows with positive dOcc >= thresh,
    plus the cumulative occupancy gained over that run.

    The final window may be partial (< 200 pieces); its dOcc still counts
    (conservative — a dying game's last-window spike flags). The trailing
    partial window never EXTENDS the reported run length by itself beyond
    real consecutive window pairs.
    Returns (longest_run, max_cumulative_gain).
    """
    d = [occ_traj[i + 1] - occ_traj[i] for i in range(len(occ_traj) - 1)]
    best_run = cur = 0
    best_gain = cur_gain = 0.0
    for x in d:
        if x >= thresh:
            cur += 1
            cur_gain += x
            if cur > best_run or (cur == best_run and cur_gain > best_gain):
                best_run, best_gain = cur, cur_gain
        else:
            cur, cur_gain = 0, 0.0
    return best_run, best_gain


def releases(occ_traj: list[float], drop: float = 20.0) -> int:
    return sum(1 for i in range(len(occ_traj) - 1)
               if occ_traj[i + 1] - occ_traj[i] <= -drop)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=REPORTING_SEED0)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--cap2", type=int, default=0,
                    help="optional second stress cap (e.g. 3000) on the "
                         "same seeds; 0 disables")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.seed0 < MIN_SELECTION_SEED0 and args.seed0 != REPORTING_SEED0:
        raise SystemExit(
            f"refusing: seed0 {args.seed0} is inside a consumed/reserved "
            f"range (top-k gate 20572-20591). Use >= {MIN_SELECTION_SEED0} "
            f"for selection or {REPORTING_SEED0} for reporting.")
    selection = args.seed0 != REPORTING_SEED0
    weights = str(resolve_weights(args.weights))
    caps = [args.cap] + ([args.cap2] if args.cap2 else [])
    seeds = [args.seed0 + i for i in range(args.games)]
    print(f"lam10-gate: games={args.games} seeds {seeds[0]}-{seeds[-1]} "
          f"caps={caps} lam={args.lam} workers={args.workers} "
          f"({'SELECTION' if selection else 'REPORTING'} range)", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    arms = [("champ", 0.0), ("lam10", args.lam)]
    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "games": args.games, "seed0": args.seed0,
           "selection": selection, "lam": args.lam,
           "completion_bonus": COMPLETION_BONUS, "results": {}}
    for cap in caps:
        payloads = [(str(HERE), weights, nm, lam, s, cap, args.window)
                    for (nm, lam) in arms for s in seeds]
        results: dict[str, list] = {}
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_worker, p) for p in payloads]
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                results.setdefault(r["arm"], []).append(r)
                done += 1
                print(f"  cap{cap} {done}/{len(payloads)} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        rows_c = sorted(results["champ"], key=lambda r: r["seed"])
        rows_b = sorted(results["lam10"], key=lambda r: r["seed"])
        by_c = {r["seed"]: r for r in rows_c}
        by_b = {r["seed"]: r for r in rows_b}
        cap_c = sum(1 for r in rows_c if r["pieces"] >= cap)
        cap_b = sum(1 for r in rows_b if r["pieces"] >= cap)
        dpc = [by_b[s]["pieces"] - by_c[s]["pieces"] for s in seeds]
        md = statistics.fmean(dpc)
        sd = statistics.stdev(dpc) if len(dpc) > 1 else 0.0
        se = sd / len(dpc) ** 0.5 if len(dpc) > 1 else 0.0
        t = md / se if se else 0.0
        wl = (sum(1 for x in dpc if x > 0), sum(1 for x in dpc if x < 0))
        # Tetris rate paired (T/100 per game)
        t100_c = [100.0 * int(r["clears"].get("4", 0)) / max(1, r["pieces"])
                  for r in rows_c]
        t100_b = [100.0 * int(r["clears"].get("4", 0)) / max(1, r["pieces"])
                  for r in rows_b]
        dt = [b - c for b, c in zip(t100_b, t100_c)]
        mdt = statistics.fmean(dt)
        sdt = statistics.stdev(dt) if len(dt) > 1 else 0.0
        tt = mdt / (sdt / len(dt) ** 0.5) if sdt else 0.0
        # S3 backlog: per-game FAIL flag = sustained run >=2 windows AND
        # zero releases; also report endgame occ
        fail_b, runs_b, rel_b = 0, [], []
        for r in rows_b:
            traj = [w["mean_occ"] for w in r["windows"]]
            run, gain = max_sustained_gain(traj)
            rel = releases(traj)
            runs_b.append((run, round(gain, 1)))
            rel_b.append(rel)
            if run >= 2 and rel == 0:
                fail_b += 1
        runs_c, rel_c = [], []
        for r in rows_c:
            traj = [w["mean_occ"] for w in r["windows"]]
            run, gain = max_sustained_gain(traj)
            rel_c.append(releases(traj))
            runs_c.append((run, round(gain, 1)))
        captag_b = [r for r in rows_b if r["pieces"] >= cap]
        end_occ_b = statistics.fmean(
            [r["windows"][-1]["mean_occ"] for r in captag_b
             if r["windows"]]) if captag_b else 0.0
        captag_c = [r for r in rows_c if r["pieces"] >= cap]
        end_occ_c = statistics.fmean(
            [r["windows"][-1]["mean_occ"] for r in captag_c
             if r["windows"]]) if captag_c else 0.0
        tet_b = sum(int(r["clears"].get("4", 0)) for r in rows_b)
        tet_c = sum(int(r["clears"].get("4", 0)) for r in rows_c)
        pc_b = statistics.fmean([r["pieces"] for r in rows_b])
        pc_c = statistics.fmean([r["pieces"] for r in rows_c])
        t100_all_b = 100.0 * tet_b / max(1, sum(r["pieces"] for r in rows_b))
        t100_all_c = 100.0 * tet_c / max(1, sum(r["pieces"] for r in rows_c))
        print(f"\n  cap {cap}: "
              f"champ {cap_c}/{len(rows_c)} cap, {pc_c:.0f} pcs, "
              f"{tet_c} tet (T/100 {t100_all_c:.3f}), "
              f"end-occ {end_occ_c:.1f} | "
              f"lam10 {cap_b}/{len(rows_b)} cap, {pc_b:.0f} pcs, "
              f"{tet_b} tet (T/100 {t100_all_b:.3f}), "
              f"end-occ {end_occ_b:.1f}")
        print(f"  paired pieces delta {md:+.1f} t{t:+.2f} W/L {wl[0]}/{wl[1]}"
              f" | paired dT/100 {mdt:+.3f} t{tt:+.2f}")
        print(f"  backlog: lam10 FAIL games (run>=2 & no release): {fail_b}"
              f" | runs {runs_b} | releases {rel_b}")
        print(f"           champ runs {runs_c} | releases {rel_c}")
        deaths_b = {str(r["seed"]): f"{r['reason']}@{r['pieces']}"
                    for r in rows_b if r["reason"] != "max_pieces"}
        deaths_c = {str(r["seed"]): f"{r['reason']}@{r['pieces']}"
                    for r in rows_c if r["reason"] != "max_pieces"}
        if deaths_b:
            print(f"  lam10 deaths: {deaths_b}")
        if deaths_c:
            print(f"  champ deaths: {deaths_c}")
        out["results"][str(cap)] = {
            "champ": {"capped": cap_c, "mean_pieces": pc_c, "tetrises": tet_c,
                      "t100": t100_all_c, "end_occ": end_occ_c,
                      "deaths": deaths_c,
                      "runs": runs_c, "releases": rel_c,
                      "per_seed": {str(r["seed"]): r for r in rows_c}},
            "lam10": {"capped": cap_b, "mean_pieces": pc_b, "tetrises": tet_b,
                      "t100": t100_all_b, "end_occ": end_occ_b,
                      "deaths": deaths_b,
                      "runs": runs_b, "releases": rel_b, "fail_games": fail_b,
                      "per_seed": {str(r["seed"]): r for r in rows_b}},
            "paired": {"d_pieces": md, "t": t, "wins": wl[0], "losses": wl[1],
                       "d_t100": mdt, "t_t100": tt},
        }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op}")


if __name__ == "__main__":
    main()
