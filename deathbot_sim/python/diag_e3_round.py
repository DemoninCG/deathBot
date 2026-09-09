"""E3 controlled training round (§32.3 E3): continued CE vs aggregation.

Question (§32.1.7): does the successful fullsplit step COMPOUND — i.e.
does a second round of equal-budget CE from the PROMOTED checkpoint
improve further, and does adding the §26.2 student-state aggregation data
help or hurt?

Arms (teacher FROZEN throughout: snapshot9 bonus-free full 2-ply,
fast-path both plies, gamma 0.999 — same semantics as diag_fullsplit_fit):
  arm A (continued) : CE on the ORIGINAL bank train split only
                      (games 0-23, 17,430 states), from fullsplit_seed42.
  arm B (aggregate) : CE on bank train split + §26.2 aggregation TRAIN
                      states (games 26000-26005 pool: 5930 states,
                      relabeled here with the frozen teacher), 1:1 mix by
                      batch alternation, from fullsplit_seed42.

Equal budget: lr 1e-4, batch 256 states, 800 steps (~12 passes over the
bank pool; §26 used 1200 steps — 800 keeps wall ~4-5h for TWO arms while
staying in the adequate-budget regime per §25 TINY FIT). Same seed
schedules modulo pool (paired step order where possible). Sentinels never
enter the loss (argmax indices + live masks only, §20.5/C2).

Metrics (unweighted validation per review agreement): informative_metrics
on bank val (games 24-31) + held-out aggregation games (26006-26007,
relabeled here, NEVER fitted) — agreement-all, agreement-on-dis,
regret|dis-live, reg|live, avoidable-dead. NO agree_dis objective (§32.1.1:
gameplay decides, offline metrics diagnose).

Selection: dev-panel 1-ply games (20632+, paired vs fullsplit baseline)
ONLY if an arm separates on held-out regret AND avoids new avoidable
deaths. Fresh-panel final only if a dev winner emerges. This driver runs
OFFLINE ONLY (no panels, no gameplay) and writes both arm checkpoints +
metrics; the dev-panel gate is a separate step.

Aggregation states are RELABELED here (teacher vectors are not on disk —
student_aggregate.json carries only margins/flags). The §26.2 trajectories
are replayed exactly (student 1-ply greedy, separate-RNG convention per
E0 §33 — E0-verified identical to the collector tapes), then labeled with
the frozen teacher.

Usage:
    python diag_e3_round.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \\
        --teacher ../../model_weights_snapshot9.pt \\
        --start ../_diag_out/fullsplit_seed42.pt \\
        --steps 800 --batch-size 256 --lr 1e-4 --seed 43 \\
        --out ../_diag_out/e3_round.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet, piece_to_index  # noqa: E402
from diag_phase2 import Bank  # noqa: E402
from afterstate import flat_rows_to_tensor  # noqa: E402
from diag_tinyset_overfit import teacher_scores_for_state  # noqa: E402

DEAD_PENALTY = -1e6
MARGIN_BUCKETS = [(0.0, 1.0, "<1"), (1.0, 5.0, "1-5"),
                  (5.0, 20.0, "5-20"), (20.0, float("inf"), "20+")]
AGG_TRAIN_GAMES = (0, 1, 2, 3, 4, 5)  # 26000-26005 (5930 states on disk)
AGG_HELD_GAMES = (6, 7)  # 26006-26007 (2098 states, never fitted)


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"not found: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--train-games", type=int, default=24)
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--agg-seed0", type=int, default=26000)
    ap.add_argument("--agg-cap", type=int, default=1500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_all = time.perf_counter()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    teacher_path = resolve(args.teacher)
    start_path = resolve(args.start)
    bank = Bank(str(resolve(args.bank)))
    n = bank.n_states

    teacher = TAPNet(output="linear")
    teacher.load_state_dict(torch.load(str(teacher_path), map_location="cpu"))
    teacher.eval()
    start = TAPNet(output="linear")
    start.load_state_dict(torch.load(str(start_path), map_location="cpu"))
    start.eval()

    import deathbot_sim as ds
    from piece_history import PieceQueue

    placed_ids, next_ids = bank._replay_piece_ids()
    n_cand = int(bank.cand_offsets[-1])
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        cand_cur[lo:hi] = piece_to_index(placed_ids[s])
        cand_nxt[lo:hi] = piece_to_index(next_ids[s])
    game_of = np.asarray(bank.game_ids)
    game_ids_sorted = sorted(set(int(g) for g in bank.game_ids))
    train_game_set = set(game_ids_sorted[:args.train_games])
    train_states = [s for s in range(n) if int(game_of[s]) in train_game_set]
    val_states = [s for s in range(n) if int(game_of[s]) not in train_game_set]

    # ---- bank teacher labels (frozen; §26 recipe) ----
    t0 = time.perf_counter()
    teacher_pick: dict[int, int] = {}
    teacher_vec: dict[int, list] = {}
    teacher_viable: dict[int, list] = {}
    teacher_margin: dict[int, float] = {}
    teacher_alldead: dict[int, bool] = {}
    with torch.no_grad():
        for si, s in enumerate(train_states + val_states):
            lo, hi = bank.candidate_slice(s)
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            tv, viable_live = teacher_scores_for_state(
                teacher, pre, placed_ids[s], next_ids[s], args.gamma_ply)
            if tv is None:
                continue
            viable = [lo + p for p in viable_live]
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            teacher_pick[s] = viable[order[0]]
            teacher_vec[s] = list(tv)
            teacher_viable[s] = list(viable)
            teacher_margin[s] = (float(tv[order[0]]) - float(tv[order[1]])
                                 if len(order) > 1 else 0.0)
            teacher_alldead[s] = all(float(v) <= DEAD_PENALTY / 2 for v in tv)
            if (si + 1) % 4000 == 0:
                print(f"  bank teacher {si + 1}/{n} ...", flush=True)
    print(f"bank teacher done in {time.perf_counter()-t0:.0f}s; "
          f"labels {len(teacher_pick)}/{n}", flush=True)
    tr_states = [s for s in train_states if s in teacher_pick]
    va_states = [s for s in val_states if s in teacher_pick]

    # dis sets defined vs the START checkpoint (the policy being improved),
    # not vs snapshot9: E3 asks whether round 2 moves the STUDENT.
    # NOTE (frame audit, 2026-09-09 rerun): start_pick/teacher_pick/dis and
    # informative_metrics/bank_get/bank_slices ALL use the teacher-viable
    # frame (lockouts excluded, targets = viable-local). §26's fullsplit
    # fit used FULL bank indices (lockouts included) for the same names —
    # agree_dis numbers are NOT comparable across drivers; E3 PRE-vs-POST
    # within this driver is the valid comparison.
    with torch.no_grad():
        start_pick = {}
        for s in tr_states + va_states:
            viable = teacher_viable[s]
            idx = [b - bank.candidate_slice(s)[0] for b in viable]
            sub = torch.tensor(idx, dtype=torch.long)
            lo, hi = bank.candidate_slice(s)
            sc = start(cand_boards[lo:hi][sub], cand_cur[lo:hi][sub],
                       cand_nxt[lo:hi][sub]).flatten()
            start_pick[s] = int(sc.argmax())
    _pos_of = {s: {b: j for j, b in enumerate(teacher_viable[s])}
               for s in tr_states + va_states}
    dis_tr = frozenset(s for s in tr_states
                       if start_pick[s] != _pos_of[s][teacher_pick[s]])
    dis_va = frozenset(s for s in va_states
                       if start_pick[s] != _pos_of[s][teacher_pick[s]])
    print(f"round-2 disagreements (start vs teacher): train {len(dis_tr)}/"
          f"{len(tr_states)}, val {len(dis_va)}/{len(va_states)}", flush=True)

    # ---- aggregation states: replay §26.2 trajectories, relabel frozen ----
    # Exact replay: start-checkpoint 1-ply greedy, separate-RNG convention
    # (E0 §33: == collector tapes). Trajectories re-derived (not read from
    # JSON — JSON carries no teacher vectors and no pre-boards).
    t1 = time.perf_counter()
    agg_train: list[dict] = []  # {boards, cur, nxt, tgt_local, margin, ...}
    agg_held_tv: dict[tuple, tuple] = {}  # (seed,pos) -> (tv, viable_n)
    agg_held_meta: list[dict] = []
    with torch.no_grad():
        for g in range(8):
            seed = args.agg_seed0 + g
            pq = PieceQueue(random.Random(seed))
            board = ds.Board()
            pos = 0
            held = g in AGG_HELD_GAMES
            while pos < args.agg_cap:
                bulk = board.lock_candidates_full(pq.current)
                nc = len(bulk[0])
                if nc == 0 or all(bulk[3]):
                    break
                rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
                viable = [i for i, lo in enumerate(bulk[3]) if not lo]
                if len(viable) < 2:
                    if viable:
                        board = ds.Board.from_row_bits(
                            [int(v) for v in rows[viable[0]]])
                    pq.advance()
                    pos += 1
                    continue
                boards = flat_rows_to_tensor(rows[viable].reshape(-1),
                                             len(viable))
                ci = piece_to_index(pq.current)
                ni = piece_to_index(pq.next_piece)
                # student pick (trajectory-defining, always the START net)
                sv = start(boards, torch.full((len(viable),), ci),
                           torch.full((len(viable),), ni)).flatten().tolist()
                s_pick = max(range(len(viable)), key=lambda j: (sv[j], -j))
                # frozen teacher label (same semantics as bank labels)
                pre = ds.Board.from_row_bits(
                    [int(v) for v in board.row_bits()])
                tv, _ = teacher_scores_for_state(
                    teacher, pre, pq.current, pq.next_piece, args.gamma_ply)
                if tv is not None and not all(
                        float(v) <= DEAD_PENALTY / 2 for v in tv):
                    order = sorted(range(len(tv)),
                                   key=lambda j: (-tv[j], j))
                    margin = (float(tv[order[0]]) - float(tv[order[1]])
                              if len(order) > 1 else 0.0)
                    if held:
                        agg_held_tv[(seed, pos)] = (list(tv), len(viable))
                        agg_held_meta.append(
                            {"seed": seed, "pos": pos, "margin": margin,
                             "t_pick": order[0], "s_pick": s_pick,
                             "dis": int(s_pick != order[0])})
                    else:
                        agg_train.append(
                            {"boards": boards,
                             "cur": torch.full((len(viable),), ci),
                             "nxt": torch.full((len(viable),), ni),
                             "tgt": order[0], "margin": margin})
                board = ds.Board.from_row_bits(
                    [int(v) for v in rows[viable[s_pick]]])
                pq.advance()
                pos += 1
            print(f"  agg replay game {seed}: pos {pos} "
                  f"({'held' if held else 'train'})", flush=True)
    print(f"agg replay+relabel done in {time.perf_counter()-t1:.0f}s; "
          f"train {len(agg_train)}, held {len(agg_held_meta)}", flush=True)

    def informative_metrics(model, states, dis_set, tag, cand_get):
        model.eval()
        agree_all = 0
        reg_dis_live, reg_live = [], []
        avoid_dead = all_dead_n = dead_pick = 0
        with torch.no_grad():
            for s in states:
                boards, cur, nxt, t, tv, viable_n = cand_get(s)
                sc = model(boards, cur, nxt).flatten()
                p = int(sc.argmax())
                best = float(max(tv))
                agree_all += int(p == t)
                q = float(tv[p])
                if q > DEAD_PENALTY / 2:
                    reg = best - q
                    reg_live.append(reg)
                    if s in dis_set:
                        reg_dis_live.append(reg)
                else:
                    dead_pick += 1
                    if best > DEAD_PENALTY / 2:
                        avoid_dead += 1
                if best <= DEAD_PENALTY / 2:
                    all_dead_n += 1
        return {"tag": tag, "n": len(states),
                "agree_all": agree_all / max(1, len(states)),
                "reg_dis_live_mean": statistics.fmean(reg_dis_live)
                if reg_dis_live else 0.0,
                "reg_dis_live_n": len(reg_dis_live),
                "reg_live_mean": statistics.fmean(reg_live)
                if reg_live else 0.0,
                "avoidable_dead": avoid_dead,
                "all_dead_states": all_dead_n, "dead_picks": dead_pick}

    def bank_get(s):
        lo, hi = bank.candidate_slice(s)
        viable = teacher_viable[s]
        pos_of = {b: j for j, b in enumerate(viable)}
        # Slice to the teacher-viable frame: cand tensors span ALL bank
        # candidates (incl. lockouts) but tv covers viable only. Lockout
        # picks (p outside viable) count as avoidable-dead, never index tv.
        idx = [b - lo for b in viable]
        sub = torch.tensor(idx, dtype=torch.long)
        return (cand_boards[lo:hi][sub], cand_cur[lo:hi][sub],
                cand_nxt[lo:hi][sub],
                pos_of[teacher_pick[s]], teacher_vec[s], len(viable))

    def held_get(key):
        tv, _n = agg_held_tv[key]
        rec = next(m for m in agg_held_meta
                   if (m["seed"], m["pos"]) == key)
        return (None, None, None, rec["t_pick"], tv, _n)  # boards via cache

    # held boards cache (rebuilt once for metric evals)
    held_boards: dict[tuple, tuple] = {}
    with torch.no_grad():
        for g in AGG_HELD_GAMES:
            seed = args.agg_seed0 + g
            pq = PieceQueue(random.Random(seed))
            board = ds.Board()
            pos = 0
            want = {m["pos"] for m in agg_held_meta if m["seed"] == seed}
            while pos < args.agg_cap and want:
                bulk = board.lock_candidates_full(pq.current)
                nc = len(bulk[0])
                if nc == 0 or all(bulk[3]):
                    break
                rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
                viable = [i for i, lo in enumerate(bulk[3]) if not lo]
                if pos in want and len(viable) >= 2:
                    boards = flat_rows_to_tensor(rows[viable].reshape(-1),
                                                 len(viable))
                    ci = piece_to_index(pq.current)
                    ni = piece_to_index(pq.next_piece)
                    held_boards[(seed, pos)] = (
                        boards, torch.full((len(viable),), ci),
                        torch.full((len(viable),), ni))
                    want.discard(pos)
                if len(viable) >= 2:
                    boards = flat_rows_to_tensor(rows[viable].reshape(-1),
                                                 len(viable))
                    ci = piece_to_index(pq.current)
                    ni = piece_to_index(pq.next_piece)
                    sv = start(boards, torch.full((len(viable),), ci),
                               torch.full((len(viable),), ni)
                               ).flatten().tolist()
                    pick = max(range(len(viable)),
                               key=lambda j: (sv[j], -j))
                    board = ds.Board.from_row_bits(
                        [int(v) for v in rows[viable[pick]]])
                elif viable:
                    board = ds.Board.from_row_bits(
                        [int(v) for v in rows[viable[0]]])
                pq.advance()
                pos += 1

    def held_get_full(key):
        tv, _n = agg_held_tv[key]
        rec = next(m for m in agg_held_meta
                   if (m["seed"], m["pos"]) == key)
        b, c, nx = held_boards[key]
        return (b, c, nx, rec["t_pick"], tv, _n)

    held_keys = list(agg_held_tv.keys())
    held_dis = frozenset(k for k in held_keys
                         if next(m["dis"] for m in agg_held_meta
                                  if (m["seed"], m["pos"]) == k))
    pre_tr = informative_metrics(start, tr_states, dis_tr, "pre_bank_train",
                                 bank_get)
    pre_va = informative_metrics(start, va_states, dis_va, "pre_bank_val",
                                 bank_get)
    pre_held = informative_metrics(start, held_keys, held_dis, "pre_agg_held",
                                   held_get_full)
    for r in (pre_tr, pre_va, pre_held):
        print(f"PRE {r['tag']}: agree {r['agree_all']:.3f} "
              f"reg|dis {r['reg_dis_live_mean']:.2f} (n {r['reg_dis_live_n']}) "
              f"reg|live {r['reg_live_mean']:.2f} "
              f"avoid-dead {r['avoidable_dead']}", flush=True)

    # ---- fit pools ----
    bank_fit = [s for s in tr_states
                if (bank.candidate_slice(s)[1] - bank.candidate_slice(s)[0]) > 1
                and not all(float(v) <= DEAD_PENALTY / 2
                            for v in teacher_vec[s])]
    bank_slices = {}
    for s in bank_fit:
        lo, hi = bank.candidate_slice(s)
        viable = teacher_viable[s]
        pos_of = {b: j for j, b in enumerate(viable)}
        idx = [b - lo for b in viable]
        sub = torch.tensor(idx, dtype=torch.long)
        bank_slices[s] = (cand_boards[lo:hi][sub], cand_cur[lo:hi][sub],
                          cand_nxt[lo:hi][sub], pos_of[teacher_pick[s]])
    print(f"fit pools: bank {len(bank_slices)}, agg-train {len(agg_train)}",
          flush=True)
    sched_bank = [rng.sample(bank_fit, min(args.batch_size, len(bank_fit)))
                  for _ in range(args.steps)]
    sched_agg = [rng.sample(agg_train, min(args.batch_size // 2,
                                           len(agg_train)))
                 if agg_train else [] for _ in range(args.steps)]
    sched_bank_half = [rng.sample(bank_fit, min(args.batch_size // 2,
                                                len(bank_fit)))
                       for _ in range(args.steps)]

    def run_arm(name, use_agg):
        model = TAPNet(output="linear")
        model.load_state_dict(torch.load(str(start_path), map_location="cpu"))
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        w0 = [p.detach().clone() for p in model.parameters()]
        losses: list[float] = []
        t1 = time.perf_counter()
        for step in range(args.steps):
            opt.zero_grad()
            tot_loss, tot_n = 0.0, 0
            if use_agg:
                idx = sched_bank_half[step]
                jdx = sched_agg[step]
                for s in idx:
                    b, c, nx, tgt = bank_slices[s]
                    logits = model(b, c, nx).flatten()
                    tot_loss = tot_loss + F.cross_entropy(
                        logits.unsqueeze(0), torch.tensor([tgt]))
                    tot_n += 1
                for rec in jdx:
                    logits = model(rec["boards"], rec["cur"],
                                   rec["nxt"]).flatten()
                    tot_loss = tot_loss + F.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([rec["tgt"]]))
                    tot_n += 1
            else:
                for s in sched_bank[step]:
                    b, c, nx, tgt = bank_slices[s]
                    logits = model(b, c, nx).flatten()
                    tot_loss = tot_loss + F.cross_entropy(
                        logits.unsqueeze(0), torch.tensor([tgt]))
                    tot_n += 1
            (tot_loss / max(1, tot_n)).backward()
            opt.step()
            losses.append(float((tot_loss / max(1, tot_n)).detach()))
            if (step + 1) % 200 == 0:
                print(f"  {name} step {step + 1}/{args.steps}: "
                      f"loss={statistics.fmean(losses[-100:]):.4f}",
                      flush=True)
        wall = time.perf_counter() - t1
        post_tr = informative_metrics(model, tr_states, dis_tr,
                                      f"{name}_bank_train", bank_get)
        post_va = informative_metrics(model, va_states, dis_va,
                                      f"{name}_bank_val", bank_get)
        post_held = informative_metrics(model, held_keys, held_dis,
                                        f"{name}_agg_held", held_get_full)
        dW = float(sum(((a - b) ** 2).sum()
                       for a, b in zip(w0, model.parameters())).sqrt()
                   .detach())
        ckpt = {"name": name, "loss": round(statistics.fmean(losses[-50:]), 4),
                "dW": round(dW, 4), "wall_s": round(wall, 1),
                "bank_train": post_tr, "bank_val": post_va,
                "agg_held": post_held}
        for r in (post_tr, post_va, post_held):
            print(f"POST {r['tag']}: agree {r['agree_all']:.3f} "
                  f"reg|dis {r['reg_dis_live_mean']:.2f} "
                  f"(n {r['reg_dis_live_n']}) reg|live {r['reg_live_mean']:.2f} "
                  f"avoid-dead {r['avoidable_dead']} dW {dW:.2f}", flush=True)
        return model, ckpt

    armA_model, armA = run_arm("armA_continued", use_agg=False)
    armB_model, armB = run_arm("armB_aggregate", use_agg=True)

    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    base = op.with_suffix("")
    torch.save(armA_model.state_dict(), str(base) + ".armA.pt")
    torch.save(armB_model.state_dict(), str(base) + ".armB.pt")
    res = {
        "teacher": str(teacher_path),
        "teacher_sha256": hashlib.sha256(
            teacher_path.read_bytes()).hexdigest(),
        "start": str(start_path),
        "start_sha256": hashlib.sha256(
            start_path.read_bytes()).hexdigest(),
        "bank": str(resolve(args.bank)),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "gamma_ply": args.gamma_ply,
        "agg_seed0": args.agg_seed0, "agg_cap": args.agg_cap,
        "n_bank_fit": len(bank_slices), "n_agg_train": len(agg_train),
        "n_agg_held": len(held_keys),
        "pre": {"bank_train": pre_tr, "bank_val": pre_va,
                "agg_held": pre_held},
        "armA": armA, "armB": armB,
        "checkpoints": {"armA": str(base) + ".armA.pt",
                        "armB": str(base) + ".armB.pt"},
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "note": ("OFFLINE ONLY: no panels, no gameplay. Dev-panel 1-ply gate "
                 "(20632+, paired vs start) only if an arm separates on "
                 "held-out regret without new avoidable deaths."),
    }
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op} + arm checkpoints in {res['wall_seconds']}s")


if __name__ == "__main__":
    main()
