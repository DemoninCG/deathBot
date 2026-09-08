"""Death-rejection ablation with intervention logging (§20 item 2, §18.3b).

HIGH priority: 0.33% bank frequency cannot dismiss preview-death rejection
(0.0033/placement ~= 300-placement expected wait; bank states are correlated
incumbent-trajectory states, events concentrate near death). Only the
closed-loop ablation separates late-symptom (already doomed, rejection just
delays) from recoverable-trap (rejection restores a lower-risk board).

Arms (matched tapes, fresh panel, cap 1500):
  A. 1-ply greedy (beam=0, bonus-free) — baseline.
  B. 1-ply + death-rejection filter: mask roots whose preview has no legal
     (non-lockout) placement, i.e. 1-ply argmax over SURVIVING roots only.
     Falls back to plain 1-ply argmax when every root is dead (logged as
     all_dead). This is the cheap rule the ablation tests.
  C. full 2-ply (beam=100, bonus-free) — champion reference (NOT a selection
     candidate here; bounds the survival gain distillation could chase).

Per-placement intervention log (arm B only): filter-change events (1-ply pick
!= surviving-roots pick), post-intervention survival (pieces after event),
risk restoration (top/occ/holes before vs +50 pieces later — restored vs
delayed), all-roots-dead cases.

Decision: report filter-change rate, survival after each intervention,
restoration-vs-delay, all-dead count, and paired cap-rate/pieces vs arm A.
Death-rejection-win -> use the rule; continuation-valuation-win (C >> B ~= A)
-> that IS the case for distillation (§20.6).

Usage:
    python diag_death_ablation.py --weights ../../model_weights_snapshot9.pt \\
        --games 30 --seed0 20520 --cap 1500 --workers 4 \\
        --out ../_diag_out/death_ablation.json
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

PANEL_START = 20520


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
    (here, weights, arm, seed, cap) = payload
    sys.path.insert(0, here)
    import random
    import numpy as _np
    from model import TAPNet as _N, piece_to_index as _pi
    from two_ply import TwoPlyScorer as _S
    from afterstate import flat_rows_to_tensor as _fr
    from piece_history import PieceQueue as _PQ
    import deathbot_sim as _ds
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    rng = random.Random(seed)
    pq = _PQ(rng)
    board = _ds.Board()
    pieces = lines = 0
    reason = "max_pieces"
    changes: list[dict] = []
    all_dead = 0
    risks: list[dict] = []
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
        if arm in ("ply1", "ply1_dr"):
            rows = _np.asarray(bulk[4], dtype=_np.uint32).reshape(n_cands, 21)
            viable = [i for i, lo in enumerate(bulk[3]) if not lo]
            boards = _fr(rows[viable].reshape(-1), len(viable))
            cur = _pi(pq.current)
            # Arm A: semantic true-preview 1-ply (cur + preview known).
            # (Champion ply-1 duplicates cur; the audit shows 0.6% flips, so
            # the ablation uses the CORRECTED 1-ply as its baseline and logs
            # the filter effect separately from the slot effect.)
            nxt = _pi(pq.next_piece)
            cur_t = torch.full((len(viable),), cur, dtype=torch.long)
            nxt_t = torch.full((len(viable),), nxt, dtype=torch.long)
            with torch.inference_mode():
                v1 = net(boards, cur_t, nxt_t).flatten().tolist()
            order = sorted(range(len(viable)), key=lambda j: (-v1[j], j))
            pick_viable_pos = order[0]
            if arm == "ply1_dr":
                # Death-rejection filter: preview-dead roots are ineligible.
                succs = _ds.Board.from_row_bits_bulk(
                    [int(v) for vv in rows[viable] for v in vv]) \
                    if hasattr(_ds.Board, "from_row_bits_bulk") else \
                    [_ds.Board.from_row_bits([int(v) for v in rows[i]])
                     for i in viable]
                alive_pos = []
                for j, b in enumerate(succs):
                    r2 = b.lock_candidates_full(pq.next_piece)
                    n2 = len(r2[0])
                    if n2 and not all(r2[3]):
                        alive_pos.append(j)
                if not alive_pos:
                    all_dead += 1
                else:
                    best_alive = max(alive_pos,
                                     key=lambda j: (v1[j], -j))
                    if best_alive != pick_viable_pos:
                        changes.append({
                            "piece": pieces,
                            "v1_pick_gap": float(
                                v1[best_alive] - v1[pick_viable_pos]),
                            "top": int(_np.asarray(
                                board.row_bits(),
                                dtype=_np.uint32).max() and 0) or 0,
                        })
                        # risk snapshot: occ of current board
                        changes[-1]["occ"] = _popcount(
                            _np.asarray(board.row_bits(), dtype=_np.uint32))
                    pick_viable_pos = best_alive
            pick = viable[pick_viable_pos]
        else:  # full 2-ply champion reference
            scorer = _S(net, beam=100)
            pick = scorer.choose(board, pq.current, pq.next_piece)
        rows = _np.asarray(bulk[4], dtype=_np.uint32).reshape(n_cands, 21)[pick]
        lines += int(bulk[5][pick])
        if pieces % 50 == 0 or pieces == cap - 1:
            risks.append({"piece": pieces,
                          "occ": _popcount(rows)})
        board = _ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    return {"arm": arm, "seed": seed, "pieces": pieces, "lines": lines,
            "reason": reason, "n_changes": len(changes),
            "all_dead": all_dead, "changes": changes[:50],
            "risks": risks, "s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=PANEL_START)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve_weights(args.weights))
    if args.seed0 < PANEL_START:
        raise SystemExit(f"refusing consumed panel: seed0 {args.seed0} "
                         f"< next-free {PANEL_START}")
    seeds = [args.seed0 + i for i in range(args.games)]
    arms = ["ply1", "ply1_dr", "ply2"]
    print(f"death-ablation: arms={arms} games={args.games} "
          f"seeds {seeds[0]}-{seeds[-1]} cap={args.cap} "
          f"workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: dict[str, list] = {}
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, a, s, args.cap)
                for a in arms for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, p) for p in payloads]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.setdefault(r["arm"], []).append(r)
            done += 1
            if done % args.workers == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} games "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0

    out: dict = {"weights": weights,
                 "weights_sha256": hashlib.sha256(
                     Path(weights).read_bytes()).hexdigest(),
                 "games": args.games, "seed0": args.seed0, "cap": args.cap,
                 "arms": {}, "wall_s": round(elapsed, 1)}
    base_pieces = {r["seed"]: r["pieces"] for r in results["ply1"]}
    for arm in arms:
        rows = sorted(results[arm], key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        cap_n = sum(1 for p in pcs if p >= args.cap)
        chg = sum(r["n_changes"] for r in rows)
        ad = sum(r["all_dead"] for r in rows)
        if arm == "ply1":
            extra = "—"
        else:
            dl = [r["pieces"] - base_pieces[r["seed"]] for r in rows]
            md = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            se = sd / (len(dl) ** 0.5) if len(dl) > 1 else 0.0
            w = sum(1 for d in dl if d > 0)
            l = sum(1 for d in dl if d < 0)
            extra = (f"{md:+.1f} t{md / se:+.2f} W/L {w}/{l}"
                     if se else f"{md:+.1f} t— W/L {w}/{l}")
        print(f"{arm:8s} pcs {statistics.fmean(pcs):7.1f} "
              f"cap {cap_n:3d}/{len(rows)} changes {chg:5d} "
              f"all_dead {ad:4d} | vs ply1: {extra}")
        out["arms"][arm] = {
            "mean_pieces": statistics.fmean(pcs),
            "capped": cap_n,
            "total_changes": chg,
            "total_all_dead": ad,
            "per_seed": {str(r["seed"]): {
                "pieces": r["pieces"], "lines": r["lines"],
                "reason": r["reason"], "n_changes": r["n_changes"],
                "all_dead": r["all_dead"], "s": r["s"]} for r in rows},
            "change_log_sample": [c for r in rows for c in r["changes"]][:50],
        }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op} (wall {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
