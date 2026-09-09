"""E1 weights x search crossover (§32.3 E1, SELECTION dev panel 20612+).

Question: which (weights, search) configuration should the campaign use —
does the promoted student (fullsplit_seed42) also improve 2-ply leaves,
and does it permit narrower (cheaper) search?

Matrix (2 weights x 5 searches, MATCHED tapes, cap 1500, paired):
  weights : snapshot9 (229bbe2a) vs fullsplit_seed42 (9989ce86)
  searches: 1-ply (beam 0) / b5_diverse / b8_diverse / b8_v1 / full 2-ply
Inter-arm comparison is paired per-seed (shared env RNG; 1-ply and 2-ply
paths both draw explore_rng ONLY when epsilon>0, and this driver uses
epsilon=0 throughout — cf. E0 §33 — so tapes are identical across ALL
cells by construction).

PRIMARY decision (cap 1500, directly comparable to all prior panels):
  cap rate per cell, paired pieces deltas, lines/Tetris health.
HORIZON columns (cap 3000 on the SAME tapes, same driver run): cap rate
only — a legibility check, not a second decision.

Panel: --seed0 default 20612, 20 tapes (dev panel 20612-20631 per §32.3).
CONSUMES 20612-20631 on first run (registry rules). 5300+ reserves and
cap-3000 confirmation panels stay untouched.

Usage:
    python diag_weights_search_xover.py --seed0 20612 --games 20 \\
        --cap 1500 --cap2 3000 --workers 4 \\
        --out ../_diag_out/weights_search_xover.json
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

E1_SEED0 = 20612  # dev panel start (§32.3); CONSUMED by this driver
REPORTING_SEED0 = 31200  # demo/reporting range (§28)


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"not found: {p}")


CELLS = [
    ("ply1", 0, "v1"),
    ("b5_diverse", 5, "diverse"),
    ("b8_diverse", 8, "diverse"),
    ("b8_v1", 8, "v1"),
    ("full", 10 ** 9, "v1"),
]


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, cell, beam, mode, seed, cap) = payload
    sys.path.insert(0, here)
    import random
    from model import TAPNet
    from two_ply import play_game_2ply_style
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    t0 = time.perf_counter()
    pieces, lines, clears, reason = play_game_2ply_style(
        net, random.Random(seed), beam=beam, shortlist_mode=mode,
        max_pieces=cap)
    return {"weights": Path(weights).name, "cell": cell, "seed": seed,
            "pieces": pieces, "lines": lines,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snap9", default="../../model_weights_snapshot9.pt")
    ap.add_argument("--student", default="../_diag_out/fullsplit_seed42.pt")
    ap.add_argument("--seed0", type=int, default=E1_SEED0)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--cap2", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap9 = str(resolve(args.snap9))
    student = str(resolve(args.student))
    arms = [("snap9", snap9), ("student", student)]
    seeds = [args.seed0 + i for i in range(args.games)]
    if args.seed0 < REPORTING_SEED0 and args.seed0 < E1_SEED0:
        print(f"[E1] seed0 {args.seed0}: below E1 dev range — refusing "
              f"(registry rules). Use --seed0 {E1_SEED0}+ or {REPORTING_SEED0}+.")
        sys.exit(2)
    if args.seed0 >= REPORTING_SEED0:
        print(f"[E1] reporting-only seeds {seeds[0]}-{seeds[-1]}.")
    else:
        print(f"[E1] SELECTION dev panel {seeds[0]}-{seeds[-1]} (CONSUMED).")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: dict[tuple, list] = {}
    t0 = time.perf_counter()
    caps = [args.cap] + ([args.cap2] if args.cap2 and args.cap2 > 0 else [])
    for cap in caps:
        payloads = [(str(HERE), w, cell, beam, mode, s, cap)
                    for (_nm, w) in arms for (cell, beam, mode) in CELLS
                    for s in seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_worker, p) for p in payloads]
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                results.setdefault((cap, r["weights"], r["cell"]), []).append(r)
                done += 1
                if done % 20 == 0:
                    print(f"  cap {cap}: {done}/{len(payloads)} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    def paired(a, b):
        da = {r["seed"]: r["pieces"] for r in a}
        db = {r["seed"]: r["pieces"] for r in b}
        d = [db[s] - da[s] for s in da if s in db]
        m = statistics.fmean(d) if d else 0.0
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
        return m, (m / se if se else 0.0), d

    out = {"snap9": snap9,
           "snap9_sha256": hashlib.sha256(
               Path(snap9).read_bytes()).hexdigest()[:8],
           "student": student,
           "student_sha256": hashlib.sha256(
               Path(student).read_bytes()).hexdigest()[:8],
           "seed0": args.seed0, "games": args.games,
           "cap": args.cap, "cap2": args.cap2,
           "selection_panel": args.seed0 < REPORTING_SEED0,
           "caps": {},
           "note": ("per-cell s/g is MEAN-dominated by full-2-ply cap-games "
                    "(~375 s/game at cap 1500; progress as_completed order "
                    "makes early batches look fast). Partial cap-1500-only "
                    "runs (--cap2 0) are complete and decision-grade.")}
    for cap in caps:
        out["caps"][str(cap)] = {"cells": {}}
        print(f"\n=== cap {cap} ===")
        print(f"{'cell':>12s} {'snap9 cap':>9s} {'stud cap':>9s} "
              f"{'dPieces(stu-snp)':>15s} {'tet snp/stu':>11s} "
              f"{'s/g snp':>7s} {'s/g stu':>7s}")
        for (cell, _beam, _mode) in CELLS:
            a = sorted(results[(cap, "model_weights_snapshot9.pt", cell)],
                       key=lambda r: r["seed"])
            b = sorted(results[(cap, "fullsplit_seed42.pt", cell)],
                       key=lambda r: r["seed"])
            m, t, _d = paired(a, b)
            cap_a = sum(1 for r in a if r["pieces"] >= cap)
            cap_b = sum(1 for r in b if r["pieces"] >= cap)
            tet_a = sum(int(r["clears"].get("4", 0)) for r in a)
            tet_b = sum(int(r["clears"].get("4", 0)) for r in b)
            sa = statistics.fmean(r["s"] for r in a)
            sb = statistics.fmean(r["s"] for r in b)
            print(f"{cell:>12s} {cap_a:3d}/{len(a):<3d}     "
                  f"{cap_b:3d}/{len(b):<3d}     {m:+7.1f} t{t:+5.2f}   "
                  f"{tet_a:3d}/{tet_b:<3d}     {sa:6.1f}   {sb:6.1f}")
            out["caps"][str(cap)]["cells"][cell] = {
                "snap9": {"capped": cap_a,
                          "mean_pieces": round(statistics.fmean(
                              r["pieces"] for r in a), 1),
                          "mean_lines": round(statistics.fmean(
                              r["lines"] for r in a), 1),
                          "tetrises": tet_a,
                          "sec_per_game": round(sa, 1),
                          "per_seed": {str(r["seed"]): {
                              "pieces": r["pieces"], "lines": r["lines"],
                              "clears": r["clears"], "reason": r["reason"]}
                              for r in a}},
                "student": {"capped": cap_b,
                            "mean_pieces": round(statistics.fmean(
                                r["pieces"] for r in b), 1),
                            "mean_lines": round(statistics.fmean(
                                r["lines"] for r in b), 1),
                            "tetrises": tet_b,
                            "sec_per_game": round(sb, 1),
                            "per_seed": {str(r["seed"]): {
                                "pieces": r["pieces"], "lines": r["lines"],
                                "clears": r["clears"], "reason": r["reason"]}
                                for r in b}},
                "paired_dPieces": round(m, 2), "paired_t": round(t, 2),
            }
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
