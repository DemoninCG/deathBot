"""Offline 1-ply distillability probe (§19 steps 1+4, read-only).

Question: can a snapshot9 COPY fit the survival 2-ply teacher's choices
on states it can see, and does it generalize to held-out games?
Teacher = full-expansion 2-ply (beam=100, bonus-free, gamma_ply=0.999)
computed from the SAME snapshot9 weights — pure lookahead signal, no style.
Student inputs are exactly TAPNet's (board + current + preview ids), so a
fit here answers the input-sufficiency question for SURVIVAL distillation
(contrast the 8-feature style head, which is likely bottlenecked).

Protocol (no panels consumed — frozen bank, no play):
- game-disjoint split: first 24 games train, last 8 games validation
  (bank games are seed-ordered; game_ids 0..23 train, 24..31 held out).
- per-state teacher label: full 2-ply argmax + full within-state teacher
  score vector (final = v1 + gamma*max(v2), dead branches DEAD_PENALTY).
- baseline (untrained copy): train/val top-1 agreement overall + on
  disagreement states, and teacher-score REGRET of the 1-ply pick
  (teacher_score[1ply] - teacher_score[teacher], <= 0; 0 = no loss).
- short preference fit (default 400 steps, batch 64, lr 1e-5, Adam) on
  train states only: per-state segmented CE toward teacher argmax over
  the FULL viable candidate set (lockouts masked, singles skipped).
- post-fit: same agreement + regret metrics on train AND held-out games.
- stratification: regret/agreement by top height bucket, holes bucket,
  preview piece, and teacher-vs-1ply margin bucket.

Outcome bins (user's three-way split):
  "cannot fit train"   — train regret stays large after 400 steps
  "fits train, not val"— train regret -> ~0 but val regret stays large
  "predicts offline"   — both regrets -> ~0 (then steps 2/5/6 justified)

Usage:
    python diag_p1_fit.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \
        --weights ../../model_weights_snapshot9.pt --steps 400 --seed 42 \
        --out ../_diag_out/p1_fit.json
"""
from __future__ import annotations

import argparse
import copy
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

DEAD_PENALTY = -1e6


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def teacher_scores_for_state(net, pre_board, kind, nk, gamma_ply=0.999):
    """Full 2-ply teacher score vector in bank-candidate order.

    Mirrors TwoPlyScorer.choose EXACTLY (bonus-free, no style), including
    the champion's fast-path scoring (score_batch_fast: cur used for both
    piece slots). Returns (scores in viable order, viable bank indices).
    """
    import deathbot_sim as ds
    from afterstate import flat_rows_to_tensor
    bulk = pre_board.lock_candidates_full(kind)
    n = len(bulk[0])
    if n == 0:
        return None, None
    lockouts = list(bulk[3])
    viable = [i for i, lo in enumerate(lockouts) if not lo]
    if not viable:
        return None, None
    rows = np.asarray(bulk[4], dtype=np.uint32).reshape(n, 21)
    cur_idx = piece_to_index(kind)
    nxt_idx = piece_to_index(nk)
    boards = flat_rows_to_tensor(rows[viable].reshape(-1), len(viable))
    cur = torch.full((len(viable),), cur_idx, dtype=torch.long)
    fast = getattr(net, "score_batch_fast", None)
    with torch.inference_mode():
        if fast is not None:
            v1 = fast(boards, cur).flatten().tolist()
        else:
            v1 = net(boards, cur, cur).flatten().tolist()
    # ply-2: successor boards per viable root, batched into one forward.
    succ_rows, counts, flags = [], [], []
    mk = ds.Board.from_row_bits_bulk if hasattr(ds.Board,
                                                "from_row_bits_bulk") else None
    succs = (mk([int(v) for vv in rows[viable] for v in vv])
             if mk is not None else
             [ds.Board.from_row_bits([int(v) for v in rows[i]])
              for i in viable])
    for b in succs:
        r2 = b.lock_candidates_full(nk)
        n2 = len(r2[0])
        if n2 == 0:
            flags.append(False)
            counts.append(0)
            continue
        rr = np.asarray(r2[4], dtype=np.uint32).reshape(n2, 21)
        keep = [i for i in range(n2) if not r2[3][i]]
        if not keep:
            flags.append(True)
            counts.append(0)
            continue
        flags.append(True)
        counts.append(len(keep))
        for i in keep:
            succ_rows.append(rr[i])
    final = list(v1)
    if succ_rows:
        big = flat_rows_to_tensor(np.stack(succ_rows), len(succ_rows))
        nxt_t = torch.full((len(succ_rows),), nxt_idx, dtype=torch.long)
        with torch.inference_mode():
            if fast is not None:
                all_v = fast(big, nxt_t).flatten().tolist()
            else:
                all_v = net(big, nxt_t, nxt_t).flatten().tolist()
        off = 0
        for j in range(len(viable)):
            if counts[j] == 0:
                final[j] = DEAD_PENALTY
            else:
                seg = all_v[off:off + counts[j]]
                off += counts[j]
                final[j] = v1[j] + gamma_ply * max(seg)
    else:
        final = [DEAD_PENALTY] * len(viable)
    return final, viable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-games", type=int, default=24)
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--save-student", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    bank = Bank(args.bank)
    n = bank.n_states
    weights_path = resolve_weights(args.weights)
    print(f"Bank: {n} states / {int(bank.cand_offsets[-1])} candidates; "
          f"games {sorted(set(int(g) for g in bank.game_ids))}")

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    net.eval()

    import deathbot_sim as ds
    from afterstate import flat_rows_to_tensor
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
    print(f"split: {len(train_states)} train / {len(val_states)} val states "
          f"(game-disjoint, train games {sorted(train_game_set)})")

    # Teacher labeling (both splits; needed for train fit + val metrics).
    # Cache the frozen teacher score vectors: regret is always measured
    # against these, so the fit cannot move its own target.
    # INDEX CONVENTION (verified: bank order == live bulk order, and
    # lockout states match): live position p <-> bank index lo+p. The
    # teacher fn returns viable as LIVE positions; convert to bank indices
    # once here (lo+pos) so every downstream dict is in bank space.
    t0 = time.perf_counter()
    teacher_pick: dict[int, int] = {}   # state -> bank-order teacher idx
    teacher_vec: dict[int, list] = {}   # state -> viable-order scores
    teacher_viable: dict[int, list] = {}  # state -> BANK-order viable idx
    teacher_margin: dict[int, float] = {}   # best - second-best teacher score
    with torch.no_grad():
        for si, s in enumerate(train_states + val_states):
            lo, hi = bank.candidate_slice(s)
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            kind, nk = placed_ids[s], next_ids[s]
            tv, viable_live = teacher_scores_for_state(net, pre, kind, nk,
                                                       args.gamma_ply)
            if tv is None:
                continue
            viable = [lo + p for p in viable_live]
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            best_local = order[0]
            teacher_pick[s] = viable[best_local]
            teacher_vec[s] = list(tv)
            teacher_viable[s] = list(viable)
            teacher_margin[s] = (float(tv[order[0]]) - float(tv[order[1]])
                                 if len(order) > 1 else 0.0)
            if (si + 1) % 2000 == 0:
                print(f"  teacher {si + 1}/{n} ...", flush=True)
    print(f"teacher done in {time.perf_counter() - t0:.0f}s; "
          f"labels {len(teacher_pick)}/{n}")
    tr_states = [s for s in train_states if s in teacher_pick]
    va_states = [s for s in val_states if s in teacher_pick]

    # Simpler + exact: single pass computing agreement AND regret live.
    def full_metrics(model, states):
        # Regret is measured against the FROZEN teacher scores (computed
        # from the incumbent weights): fit quality of the student, not a
        # moving target. All indices in BANK order (bank.cand_* layout);
        # p1 is a bank index, teacher_pick too, so compare directly.
        model.eval()
        agree = tot = 0
        regrets, dis_regrets = [], []
        with torch.no_grad():
            for s in states:
                lo, hi = bank.candidate_slice(s)
                sc = model(cand_boards[lo:hi], cand_cur[lo:hi],
                           cand_nxt[lo:hi]).flatten()
                # bank-order pick (argmax over the bank-ordered slice)
                p1 = lo + int(sc.argmax())
                t = teacher_pick[s]
                tot += 1
                agree += int(p1 == t)
                tv = teacher_vec[s]
                viable = teacher_viable[s]
                pos_of = {b: j for j, b in enumerate(viable)}
                r = float(tv[pos_of[p1]]) - float(max(tv)) \
                    if p1 in pos_of else DEAD_PENALTY
                regrets.append(r)
                if t != int(bank.chosen_idx[s]):
                    dis_regrets.append(r)
        return {"agree": agree, "n": tot,
                "agree_rate": agree / max(1, tot),
                "mean_regret": statistics.fmean(regrets),
                "mean_dis_regret": statistics.fmean(dis_regrets)
                if dis_regrets else 0.0,
                "n_dis": len(dis_regrets)}

    incumbent_ref = TAPNet(output="linear")
    incumbent_ref.load_state_dict(torch.load(str(weights_path),
                                             map_location="cpu"))
    incumbent_ref.eval()
    pre_tr = full_metrics(incumbent_ref, tr_states)
    pre_va = full_metrics(incumbent_ref, va_states)
    print(f"PRE train: agree {pre_tr['agree']}/{pre_tr['n']} "
          f"({pre_tr['agree_rate']:.3f}) regret {pre_tr['mean_regret']:.2f} "
          f"dis-regret {pre_tr['mean_dis_regret']:.2f} (n_dis {pre_tr['n_dis']})")
    print(f"PRE val:   agree {pre_va['agree']}/{pre_va['n']} "
          f"({pre_va['agree_rate']:.3f}) regret {pre_va['mean_regret']:.2f} "
          f"dis-regret {pre_va['mean_dis_regret']:.2f} (n_dis {pre_va['n_dis']})")

    # Preference fit on train states: segmented CE toward teacher argmax.
    # NOTE: teacher_pick is in bank order; candidate tensors here are in
    # bank order too (bank.cand_* layout), so the bank index IS the local
    # index into the per-state slice. Do NOT subtract lo (that mixes the
    # live lock_candidates_full order with the bank order).
    student = TAPNet(output="linear")
    student.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    fit_states = [s for s in tr_states
                  if bank.candidate_slice(s)[1] - bank.candidate_slice(s)[0] > 1]
    sched = [rng.sample(fit_states, min(args.batch_size, len(fit_states)))
             for _ in range(args.steps)]
    # Precompute slices once (local target = bank idx - lo).
    pre_slices = {}
    for s in fit_states:
        lo, hi = bank.candidate_slice(s)
        pre_slices[s] = (cand_boards[lo:hi], cand_cur[lo:hi],
                         cand_nxt[lo:hi], int(teacher_pick[s]) - lo)
    losses = []
    t1 = time.perf_counter()
    for step in range(args.steps):
        idx = sched[step]
        opt.zero_grad()
        tot_loss, tot_n = 0.0, 0
        for s in idx:
            b, c, nx, tgt_local = pre_slices[s]
            logits = student(b, c, nx).flatten()
            loss = F.cross_entropy(logits.unsqueeze(0),
                                   torch.tensor([tgt_local]))
            tot_loss = tot_loss + loss
            tot_n += 1
        (tot_loss / max(1, tot_n)).backward()
        opt.step()
        losses.append(float((tot_loss / max(1, tot_n)).detach()))
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{args.steps}: "
                  f"loss={statistics.fmean(losses[-100:]):.4f}", flush=True)
    print(f"fit done in {time.perf_counter() - t1:.0f}s; "
          f"final loss {statistics.fmean(losses[-50:]):.4f}")
    post_tr = full_metrics(student, tr_states)
    post_va = full_metrics(student, va_states)
    print(f"POST train: agree {post_tr['agree']}/{post_tr['n']} "
          f"({post_tr['agree_rate']:.3f}) regret {post_tr['mean_regret']:.2f}")
    print(f"POST val:   agree {post_va['agree']}/{post_va['n']} "
          f"({post_va['agree_rate']:.3f}) regret {post_va['mean_regret']:.2f}")

    res = {
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(
            weights_path.read_bytes()).hexdigest(),
        "bank": str(Path(args.bank).resolve()),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "train_games": args.train_games,
        "gamma_ply": args.gamma_ply,
        "n_train": len(tr_states), "n_val": len(va_states),
        "final_loss": round(statistics.fmean(losses[-50:]), 4),
        "pre_train": pre_tr, "pre_val": pre_va,
        "post_train": post_tr, "post_val": post_va,
    }
    if args.save_student:
        sp = Path(args.out).resolve().parent / f"p1fit_seed{args.seed}.pt"
        torch.save(student.state_dict(), str(sp))
        res["student_checkpoint"] = str(sp)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
