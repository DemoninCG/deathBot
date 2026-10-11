"""Snapshot showdown: pure 1-ply greedy eval of all 12 snapshots (2026-09-15).

Single head-to-head evaluation — NO 2-ply, NO controller, NO style bonuses.
Each of model_weights_snapshot{1..12}.pt plays the SAME fresh tapes at
cap 3000, epsilon=0, corrected lockout, one harness.

Harness (matches sweep_eval.py / play_game_detailed exactly):
  - per-game rng = random.Random(seed), ALIASED explore convention
    (select_candidate_index draws rng.random() per placement even at
    epsilon=0 — see §33 E0; both arms share one harness so the comparison
    is valid; tape labels are convention-relative).
  - NeuralScorer.score_batch with TRUE preview (cur + nxt), NOT the
    2-ply cur-duplicated fast path (two_ply._placements_from_bulk scores
    with cur in both slots — do NOT use play_game_2ply_style(beam=0) here).
  - lockout masking: losing placements excluded; all-lose -> "lockout".
  - per-step lines from make_afterstate -> clears{0..4} breakdown for the
    Tetris-rate read (sweep_eval.py stores lines only, no clears).

Panel: 21680-21729 (50 fresh tapes, CONSUMED on run). 5500+ reserves untouched.
Physics: post-§54 corrected TAP rotation (rebuild extension before running).

Read-out per snapshot: mean/med/sd/min/max pieces, cap rate, mean lines,
clears breakdown, T/100 = 100*Tetris/pieces, T/capped-game, longest
no-Tetris gap (from tetris_at), reason histogram, plus the censored-data
MTTD point estimate under an exponential model:
    T_tot = sum(pieces) ; k = deaths ; MTTD = T_tot / k  (k > 0),
    95% CI from 2*lam*T_tot ~ chi2(2k) via Wilson-Hilferty (no scipy).
k = 0 -> report lower bound MTTD > T_tot / 3 (95% rule-of-three) + cap rate.
The exponential is an EXTRAPOLATION (Death backlog deaths are not proven
memoryless) — always report Kaplan-Meier cap rate + median alongside it.

Usage (from deathbot_sim/python, workspace venv):
    python diag_snapshot_showdown.py --seed0 21680 --games 50 --cap 3000 \
        --workers 8 --out ../_diag_out/snapshot_showdown_21680.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEED0 = 21680  # next free dev range per registry (21650-21679 CONSUMED by §58)
N_SNAPS = 12


def resolve_weights(want: list[int] | None = None) -> list[Path]:
    ids = list(want) if want else list(range(1, N_SNAPS + 1))
    out = []
    for i in ids:
        name = f"model_weights_snapshot{i}.pt"
        for cand in (HERE.parent.parent / name, HERE.parent / name):
            if cand.is_file():
                out.append(cand.resolve())
                break
        else:
            raise FileNotFoundError(f"weights not found: {name}")
    return out


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _chi2_quantile(df: int, p: float) -> float:
    """Wilson-Hilferty approximation to chi2(df) quantile (no scipy)."""
    # z for p=0.025/0.975
    z = -1.9599639861201952 if p < 0.5 else 1.9599639861201952
    nu = float(df)
    return nu * (1.0 - 2.0 / (9.0 * nu) + z * math.sqrt(2.0 / (9.0 * nu))) ** 3


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    (here, weights, seeds, cap) = payload
    sys.path.insert(0, here)
    import random
    from model import TAPNet, piece_to_index
    from neural_scorer import NeuralScorer
    from selfplay import select_candidate_index
    from piece_history import PieceQueue
    from afterstate import make_afterstate
    import deathbot_sim as ds

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    scorer = NeuralScorer(net=net, device="cpu")
    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        rng = random.Random(seed)  # aliased explore convention (sweep harness)
        board = ds.Board()
        pq = PieceQueue(rng)
        clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        tetris_at: list[int] = []
        lines_total = 0
        pieces = 0
        reason = "max_pieces"
        while pieces < cap:
            cur = pq.current
            nxt = pq.next_piece
            cur_idx = piece_to_index(cur)
            nxt_idx = piece_to_index(nxt)
            cands = board.search_locks(cur)
            if not cands:
                reason = "no_candidates"
                break
            viable = [i for i, c in enumerate(cands)
                      if not board.locks_out(c.piece)]
            if not viable:
                reason = "lockout"
                break
            scores = scorer.score_batch(
                board, cands,
                current_piece_idx=cur_idx, next_piece_idx=nxt_idx)
            vscores = [scores[i] for i in viable]
            pick = select_candidate_index(vscores, rng, epsilon=0.0)
            cand = cands[viable[pick]]
            aft = make_afterstate(board, cand)
            n = int(aft.lines_cleared)
            clears[n] = clears.get(n, 0) + 1
            if n == 4:
                tetris_at.append(pieces)
            board = aft.board
            lines_total += n
            pq.advance()
            pieces += 1
        rows.append({"seed": seed, "pieces": pieces, "lines": lines_total,
                     "clears": {str(k): int(v) for k, v in clears.items()},
                     "tetris_at": tetris_at, "reason": reason,
                     "s": round(time.perf_counter() - t0, 1)})
    return {"weights": weights, "rows": rows}


def summarize(rows: list[dict], cap: int) -> dict:
    pcs = [r["pieces"] for r in rows]
    lns = [r["lines"] for r in rows]
    tet = sum(int(r["clears"].get("4", 0)) for r in rows)
    capped = sum(1 for p in pcs if p >= cap)
    deaths = len(pcs) - capped
    t_tot = sum(pcs)
    reasons: dict[str, int] = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    # longest no-Tetris gap per game, then max over games
    max_gap = 0
    for r in rows:
        marks = [0] + sorted(r["tetris_at"]) + [r["pieces"]]
        max_gap = max(max_gap, max(b - a for a, b in zip(marks, marks[1:])))
    out = {
        "n": len(pcs), "cap": cap,
        "mean_pieces": round(statistics.fmean(pcs), 1),
        "median_pieces": statistics.median(pcs),
        "stdev_pieces": round(statistics.stdev(pcs), 1) if len(pcs) > 1 else 0.0,
        "min_pieces": min(pcs), "max_pieces": max(pcs),
        "capped": capped, "deaths": deaths,
        "mean_lines": round(statistics.fmean(lns), 1),
        "tetrises": tet,
        "t_per_100": round(100.0 * tet / t_tot, 3) if t_tot else 0.0,
        "t_per_capped": round(tet / capped, 2) if capped else 0.0,
        "longest_gap": max_gap,
        "reasons": reasons,
        "t_total": t_tot,
    }
    if deaths > 0:
        mttd = t_tot / deaths
        lo_q = _chi2_quantile(2 * deaths, 0.025)
        hi_q = _chi2_quantile(2 * deaths, 0.975)
        # lam in [lo/(2T), hi/(2T)] -> MTTD in [2T/hi, 2T/lo]
        out["mttd_exp"] = round(mttd, 1)
        out["mttd_exp_ci95"] = [round(2 * t_tot / hi_q, 1),
                                round(2 * t_tot / lo_q, 1)]
    else:
        out["mttd_exp"] = None
        out["mttd_exp_lower95"] = round(t_tot / 3.0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards-per-snap", type=int, default=5)
    ap.add_argument("--snaps", nargs="*", type=int, default=None,
                    help="subset of snapshot ids to play (default: all 12)")
    args = ap.parse_args()

    if args.seed0 < SEED0:
        print(f"[showdown] REFUSING seed0 {args.seed0} < {SEED0} (registry).")
        sys.exit(2)
    standard = (args.games == 50 and args.cap == 3000 and args.snaps is None)
    if not standard:
        print(f"[showdown] NOTE non-standard games={args.games} cap={args.cap} "
              f"snaps={args.snaps} (standard: 12 x 50 x cap-3000).")

    ckpts = resolve_weights(args.snaps)
    print(f"[showdown] {len(ckpts)} snapshots x {args.games} tapes "
          f"{args.seed0}-{args.seed0 + args.games - 1} @cap {args.cap} "
          f"(CONSUMED on completion), workers={args.workers}")
    for p in ckpts:
        print(f"  - {p.name} sha={sha16(p)}")

    seeds = [args.seed0 + i for i in range(args.games)]
    per = (len(seeds) + args.shards_per_snap - 1) // args.shards_per_snap
    payloads = [(str(HERE), str(p), seeds[i:i + per], args.cap)
                for p in ckpts for i in range(0, len(seeds), per)]

    per_snap: dict[str, list] = {p.name: [] for p in ckpts}
    name_of = {str(p.resolve()): p.name for p in ckpts}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, pl): pl for pl in payloads}
        for fut in as_completed(futs):
            r = fut.result()
            nm = name_of[r["weights"]]
            per_snap[nm].extend(r["rows"])
            done = sum(len(v) for v in per_snap.values())
            print(f"  [{nm}] +{len(r['rows'])} ({done}/{len(ckpts) * args.games})",
                  flush=True)
    wall = time.perf_counter() - t0

    table = {}
    for p in ckpts:
        rows = sorted(per_snap[p.name], key=lambda r: r["seed"])
        assert len(rows) == args.games, (p.name, len(rows))
        table[p.name] = {"sha16": sha16(p), "summary": summarize(rows, args.cap),
                         "per_seed": {str(r["seed"]): r for r in rows}}

    # paired pieces deltas vs snapshot12 (champion reference, when present)
    ref_rows = per_snap.get("model_weights_snapshot12.pt") or None
    ref = ({r["seed"]: r["pieces"] for r in ref_rows} if ref_rows else None)
    print(f"\n{'snap':>28s} {'mean':>7s} {'med':>6s} {'cap':>7s} "
          f"{'T/100':>6s} {'MTTDexp':>9s} {'dPiecesv12':>10s}")
    for p in ckpts:
        s = table[p.name]["summary"]
        n = len(per_snap[p.name])
        vs = None
        vs_str = "        --"
        if ref is not None and p.name != "model_weights_snapshot12.pt":
            common = [r for r in per_snap[p.name] if r["seed"] in ref]
            if common:
                d = [r["pieces"] - ref[r["seed"]] for r in common]
                m = statistics.fmean(d)
                sd = statistics.stdev(d) if len(d) > 1 else 0.0
                t = m / (sd / math.sqrt(len(d))) if sd else 0.0
                w = sum(1 for x in d if x > 0)
                l = sum(1 for x in d if x < 0)
                vs = {"mean": round(m, 1), "t": round(t, 2), "W": w, "L": l,
                      "n": len(d)}
                table[p.name]["vs_snap12"] = vs
                vs_str = f"{m:+8.1f} t{t:+5.2f} W/L {w}/{l}"
        mttd = (str(s["mttd_exp"]) if s["mttd_exp"] is not None
                else f">{s['mttd_exp_lower95']}")
        print(f"{p.name:>28s} {s['mean_pieces']:7.1f} {s['median_pieces']:6.0f} "
              f"{s['capped']:3d}/{n:<3d} {s['t_per_100']:6.3f} {mttd:>9s} "
              f"{vs_str}")

    out = {"seed0": args.seed0, "games": args.games, "cap": args.cap,
           "harness": "1-ply NeuralScorer true-preview, eps=0, corrected "
                      "lockout, aliased RNG (sweep convention), post-§54 physics",
           "arms": table, "wall_seconds": round(wall, 1)}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} in {wall:.0f}s")


if __name__ == "__main__":
    main()
