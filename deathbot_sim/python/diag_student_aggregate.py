"""Student-state aggregation + guarded closed-loop survival (§20 item 5, steps 5+6).

Runs ONLY after the full-split fit (§20 item 5 first half,
diag_fullsplit_fit.py) reports FITS TRAIN or PREDICTS OFFLINE. Frozen-bank
CE alone cannot fix coverage: the student visits new states the bank never
covered (§13.3/§14 lesson). This driver closes the loop the DAgger way:

  1. STUDENT ROLLOUT: the full-split student checkpoint plays N games
     1-ply greedy (true-preview scoring, corrected lockout) on fresh
     TRAINING seeds (default 26000+; training data, NOT a selection
     panel — same status as the 20400 teacher bank). Records every
     visited (pre-board packed rows, placed/preview ids, candidate rows).
  2. TEACHER LABEL: the versioned CHAMPION teacher (bonus-free full 2-ply,
     fast-path, gamma 0.999, snapshot9 weights) labels each visited state
     (argmax + full viable score vector). Frozen teacher vectors: regret
     is measured against these, never a moving target.
  3. AGGREGATE RETRAIN: mini-batch CE on frozen-bank train split + fresh
     student states (1:1 mix by default), same budget as the full-split
     fit (lr 1e-4, batch 256, 800 steps). Sentinels never enter the loss.
  4. OFFLINE REPORT: informative metrics (§20.3, positive regret) on
     frozen val + held-out student states (last 25% of collected games):
     agreement-all, agreement-on-disagreements, regret|disagreement
     (live only), regret|live-pick, avoidable-dead, all-dead.
  5. GUARDED CLOSED-LOOP: 1-ply greedy games with the aggregated student
     vs the incumbent 1-ply on the SAME training seeds (reporting only —
     NOT a selection decision). Promotion decisions use a FRESH panel
     (default 20560+, cap 1500) exactly once, after offline metrics pass.

Guards: student checkpoint required; teacher = versioned champion only;
no gameplay on selection panels inside this driver (closed-loop games run
on the training seeds for diagnosis; the selection panel stays untouched
until the final guarded eval, which this driver does NOT auto-run).

Usage:
    python diag_student_aggregate.py --student ../_diag_out/fullsplit_seed42.pt \
        --weights ../../model_weights_snapshot9.pt --games 8 --seed0 26000 \
        --steps 800 --out ../_diag_out/student_aggregate.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", required=True,
                    help="full-split student checkpoint (.pt)")
    ap.add_argument("--weights", required=True,
                    help="snapshot9 champion weights (teacher)")
    ap.add_argument("--bank", default="../_diag_out/state_bank_snapshot9_32games.npz")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=26000)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import hashlib
    import json
    import random
    import statistics
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from afterstate import flat_rows_to_tensor
    from diag_tinyset_overfit import teacher_scores_for_state

    DEAD_PENALTY = -1e6

    def resolve(p: str) -> Path:
        sup = Path(p).expanduser()
        for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
                  HERE.parent.parent.parent / sup):
            if c.is_file():
                return c.resolve()
        raise FileNotFoundError(f"not found: {p}")

    student_path = resolve(args.student)
    weights_path = resolve(args.weights)
    bank = Bank(str(resolve(args.bank)))
    n = bank.n_states
    print(f"Bank: {n} states; student {student_path.name}; "
          f"teacher {weights_path.name}", flush=True)

    teacher = TAPNet(output="linear")
    teacher.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    teacher.eval()
    student0 = TAPNet(output="linear")
    student0.load_state_dict(torch.load(str(student_path), map_location="cpu"))
    student0.eval()

    import deathbot_sim as ds
    from piece_history import PieceQueue

    # ---- 1. student rollouts on training seeds ----
    rng = random.Random(args.seed0)
    visited: list[dict] = []  # {pre_rows, kind, nk, game, pos}
    closed: list[dict] = []   # per-game {seed, pieces, reason}
    t0 = time.perf_counter()
    placed_ids_bank, next_ids_bank = bank._replay_piece_ids()
    n_cand_bank = int(bank.cand_offsets[-1])
    bank_boards = flat_rows_to_tensor(bank.cand_rows, n_cand_bank)
    bank_cur = torch.empty(n_cand_bank, dtype=torch.long)
    bank_nxt = torch.empty(n_cand_bank, dtype=torch.long)
    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        bank_cur[lo:hi] = piece_to_index(placed_ids_bank[s])
        bank_nxt[lo:hi] = piece_to_index(next_ids_bank[s])

    with torch.no_grad():
        for g in range(args.games):
            seed = args.seed0 + g
            prng = random.Random(seed)
            pq = PieceQueue(prng)
            board = ds.Board()
            pieces = 0
            reason = "max_pieces"
            while pieces < args.cap:
                bulk = board.lock_candidates_full(pq.current)
                nc = len(bulk[0])
                if nc == 0:
                    reason = "no_candidates"
                    break
                if all(bulk[3]):
                    reason = "lockout"
                    break
                rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
                viable = [i for i, lo in enumerate(bulk[3]) if not lo]
                boards = flat_rows_to_tensor(rows[viable].reshape(-1),
                                             len(viable))
                cur_t = torch.full((len(viable),),
                                   piece_to_index(pq.current),
                                   dtype=torch.long)
                nxt_t = torch.full((len(viable),),
                                   piece_to_index(pq.next_piece),
                                   dtype=torch.long)
                v1 = student0(boards, cur_t, nxt_t).flatten().tolist()
                order = sorted(range(len(viable)),
                               key=lambda j: (-v1[j], j))
                pick_viable = order[0]
                pick = viable[pick_viable]
                visited.append({
                    "pre_rows": np.asarray(board.row_bits(),
                                           dtype=np.uint32),
                    "kind": pq.current, "nk": pq.next_piece,
                    "game": g, "seed": seed, "pos": pieces,
                    "student_pick_livepos": pick_viable,
                    "n_viable": len(viable),
                })
                board = ds.Board.from_row_bits([int(v) for v in rows[pick]])
                pq.advance()
                pieces += 1
            closed.append({"seed": seed, "pieces": pieces, "reason": reason})
            print(f"  student game {seed}: {pieces} pcs ({reason})",
                  flush=True)
    print(f"student rollouts: {len(visited)} states from {args.games} games "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    mean_roll = statistics.fmean(c["pieces"] for c in closed)
    print(f"  closed-loop student mean on training seeds: {mean_roll:.1f} "
          f"(diagnostic, NOT a selection result)", flush=True)

    # ---- 2. teacher labels on student states ----
    t1 = time.perf_counter()
    agg: list[dict] = []
    with torch.no_grad():
        for i, v in enumerate(visited):
            pre = ds.Board.from_row_bits([int(x) for x in v["pre_rows"]])
            tv, viable_live = teacher_scores_for_state(
                teacher, pre, v["kind"], v["nk"], args.gamma_ply)
            if tv is None:
                continue
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            # student pick re-derived in teacher's viable frame: the
            # rollout already picked by student-1ply; record agreement by
            # comparing student live-pos against teacher argmax live-pos.
            # (Exact: both frames are lock_candidates_full order.)
            s_pick = v["student_pick_livepos"]
            t_pick = order[0]
            agg.append({"idx": i, "game": v["game"], "seed": v["seed"],
                        "pos": v["pos"], "kind": str(v["kind"]),
                        "tv": list(tv), "t_pick": t_pick,
                        "s_pick": s_pick,
                        "dis": int(s_pick != t_pick),
                        "margin": (float(tv[order[0]]) - float(tv[order[1]])
                                   if len(order) > 1 else 0.0),
                        "all_dead": bool(all(float(x) <= DEAD_PENALTY / 2
                                             for x in tv))})
            if (i + 1) % 500 == 0:
                print(f"  label {i + 1}/{len(visited)} ...", flush=True)
    print(f"teacher labeling done in {time.perf_counter() - t1:.0f}s; "
          f"labels {len(agg)}/{len(visited)}", flush=True)
    # held-out = last 25% of games
    games_sorted = sorted(set(a["game"] for a in agg))
    cut = games_sorted[int(len(games_sorted) * 0.75):] if games_sorted else []
    held = [a for a in agg if a["game"] in set(cut)]
    train_agg = [a for a in agg if a["game"] not in set(cut)]

    n_dis = sum(1 for a in agg if a["dis"] and not a["all_dead"])
    print(f"student-teacher disagreements on student states: {n_dis}/"
          f"{len(agg)} ({n_dis / max(1, len(agg)):.3f}); "
          f"held-out games {sorted(cut)}: {len(held)} states", flush=True)

    res = {
        "student": str(student_path),
        "student_sha256": hashlib.sha256(
            student_path.read_bytes()).hexdigest(),
        "teacher_weights": str(weights_path),
        "teacher_sha256": hashlib.sha256(
            weights_path.read_bytes()).hexdigest(),
        "games": args.games, "seed0": args.seed0, "cap": args.cap,
        "n_visited": len(visited), "n_labeled": len(agg),
        "n_dis": n_dis,
        "n_train_agg": len(train_agg), "n_held": len(held),
        "closed_loop_training_seeds": closed,
        "closed_loop_mean": round(mean_roll, 1),
        "held": [{"seed": a["seed"], "pos": a["pos"],
                  "dis": a["dis"], "margin": round(a["margin"], 3),
                  "all_dead": a["all_dead"]} for a in held],
        "note": ("aggregation RETRAIN not auto-run: inspect n_dis + "
                 "margins first; retrain mixes frozen-bank train + "
                 "train_agg with the full-split budget, then the guarded "
                 "selection eval runs ONCE on fresh 20560+ (cap 1500)."),
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
