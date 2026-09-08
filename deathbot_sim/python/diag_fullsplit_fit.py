"""Full-split survival-distillation fit (§20 item 5; NO panels, NO gameplay).

Unlocked by §25 TINY FIT (256-state full-batch CE lr 1e-4 / 800 steps ->
agree_dis 1.000, dW 8.7): the update learns these decisions, so §19's
400-step probe was under-optimization, not a capacity/input failure.

Protocol (frozen snapshot9 bank, game-disjoint split, no gameplay eval):
- Teacher = versioned CHAMPION: bonus-free full 2-ply (beam>=n, fast-path
  cur-in-both-slots both plies, gamma_ply 0.999) from snapshot9
  sha 229bbe2a. Labels the ACTUAL champion (§20.4a: do NOT "fix" slots).
- Split: games 0-23 train (17,430 states) / games 24-31 val (5,586 states),
  same as diag_p1_fit.py.
- Training: mini-batch CE toward teacher argmax over the FULL viable
  candidate set (lockouts masked, singles + all-dead skipped, sentinels
  NEVER enter the loss — argmax indices only, §20.5/C2).
  Budget: lr 1e-4, batch 256 states, 1200 steps (~17 passes over train).
- Metrics use POSITIVE regret (§20.3):
    regret(s) = max_a Q_T - Q_T(student pick)   (>= 0; lockout/dead picks
    counted separately, NEVER averaged with ranking gaps).
  Reported on train AND val: agreement-all, agreement-on-frozen-
  disagreements (dis = states where the PRE incumbent 1-ply != teacher,
  so PRE agree_dis = 0 by construction), regret|disagreement (live picks
  only), regret|live-pick, avoidable-dead count, all-dead count, plus
  margin-bucket stratification.

Outcome bins:
  "predicts offline"    — val agree_dis high  -> steps 2/5/6 (aggregation
                           + guarded closed-loop survival) justified.
  "fits train, not val" — train fits, val lags -> aggregation may still
                           help (coverage gap, not capacity gap).
  "cannot fit train"    — even train fails   -> reconsider student/setup.

Usage:
    python diag_fullsplit_fit.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \
        --weights ../../model_weights_snapshot9.pt --steps 1200 --batch-size 256 \
        --lr 1e-4 --seed 42 --out ../_diag_out/fullsplit_fit.json
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


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-games", type=int, default=24)
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--save-student", action="store_true", default=True)
    ap.add_argument("--no-save-student", dest="save_student",
                    action="store_false")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    bank = Bank(args.bank)
    n = bank.n_states
    weights_path = resolve_weights(args.weights)
    print(f"Bank: {n} states / {int(bank.cand_offsets[-1])} candidates; "
          f"games {sorted(set(int(g) for g in bank.game_ids))}", flush=True)

    net0 = TAPNet(output="linear")
    net0.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    net0.eval()

    import deathbot_sim as ds
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
          f"(game-disjoint, train games {sorted(train_game_set)})", flush=True)

    # ---- teacher labeling (frozen; regret always measured vs these) ----
    # INDEX CONVENTION: teacher fn returns viable as LIVE positions
    # (bank order == live bulk order, verified); convert once to bank
    # indices (lo+pos). All downstream dicts are in bank space.
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
            kind, nk = placed_ids[s], next_ids[s]
            tv, viable_live = teacher_scores_for_state(net0, pre, kind, nk,
                                                       args.gamma_ply)
            if tv is None:
                continue
            viable = [lo + p for p in viable_live]
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            teacher_pick[s] = viable[order[0]]
            teacher_vec[s] = list(tv)
            teacher_viable[s] = list(viable)
            teacher_margin[s] = (float(tv[order[0]]) - float(tv[order[1]])
                                 if len(order) > 1 else 0.0)
            teacher_alldead[s] = all(float(v) <= DEAD_PENALTY / 2
                                     for v in tv)
            if (si + 1) % 2000 == 0:
                print(f"  teacher {si + 1}/{n} ...", flush=True)
    print(f"teacher done in {time.perf_counter() - t0:.0f}s; "
          f"labels {len(teacher_pick)}/{n}", flush=True)
    tr_states = [s for s in train_states if s in teacher_pick]
    va_states = [s for s in val_states if s in teacher_pick]

    # ---- incumbent 1-ply picks (frozen baseline; defines dis sets) ----
    inc_pick: dict[int, int] = {}
    with torch.no_grad():
        for s in tr_states + va_states:
            lo, hi = bank.candidate_slice(s)
            sc = net0(cand_boards[lo:hi], cand_cur[lo:hi],
                      cand_nxt[lo:hi]).flatten()
            inc_pick[s] = lo + int(sc.argmax())
    dis_tr = frozenset(s for s in tr_states if inc_pick[s] != teacher_pick[s])
    dis_va = frozenset(s for s in va_states if inc_pick[s] != teacher_pick[s])
    print(f"disagreements (incumbent vs teacher): train {len(dis_tr)}/"
          f"{len(tr_states)} ({len(dis_tr) / max(1, len(tr_states)):.3f}), "
          f"val {len(dis_va)}/{len(va_states)} "
          f"({len(dis_va) / max(1, len(va_states)):.3f})", flush=True)

    def informative_metrics(model, states, dis_set, tag):
        """Positive-regret metrics (§20.3). Sentinels never averaged."""
        model.eval()
        agree_all = 0
        n_dis_l = 0
        reg_dis_live, reg_live = [], []
        avoid_dead = all_dead_n = lockout_pick = dead_pick = 0
        bucket_stats: dict[str, list] = {lbl: [] for _, _, lbl in
                                         MARGIN_BUCKETS}
        with torch.no_grad():
            for s in states:
                lo, hi = bank.candidate_slice(s)
                sc = model(cand_boards[lo:hi], cand_cur[lo:hi],
                           cand_nxt[lo:hi]).flatten()
                p = lo + int(sc.argmax())
                t = teacher_pick[s]
                tv = teacher_vec[s]
                viable = teacher_viable[s]
                pos_of = {b: j for j, b in enumerate(viable)}
                best = float(max(tv))
                agree_all += int(p == t)
                if s in dis_set:
                    n_dis_l += 1
                if p in pos_of:
                    q = float(tv[pos_of[p]])
                    if q > DEAD_PENALTY / 2:
                        reg = best - q  # >= 0 positive regret
                        reg_live.append(reg)
                        if s in dis_set:
                            reg_dis_live.append(reg)
                            for lo_b, hi_b, lbl in MARGIN_BUCKETS:
                                if lo_b <= teacher_margin[s] < hi_b:
                                    bucket_stats[lbl].append(reg)
                                    break
                    else:
                        dead_pick += 1
                        # Student picked a preview-dead root while the
                        # teacher's pick is live: an avoidable death.
                        if (not teacher_alldead[s] and
                                float(tv[pos_of[t]]) > DEAD_PENALTY / 2):
                            avoid_dead += 1
                else:
                    lockout_pick += 1
                    if not teacher_alldead[s]:
                        avoid_dead += 1
                if teacher_alldead[s]:
                    all_dead_n += 1
        res = {
            "tag": tag,
            "n": len(states),
            "agree_all": agree_all / max(1, len(states)),
            "n_dis": n_dis_l,
            "agree_dis": None,  # filled by agreement_on_dis() after call
            "reg_dis_live_mean": statistics.fmean(reg_dis_live)
            if reg_dis_live else 0.0,
            "reg_dis_live_max": max(reg_dis_live) if reg_dis_live else 0.0,
            "reg_dis_live_n": len(reg_dis_live),
            "reg_live_mean": statistics.fmean(reg_live) if reg_live else 0.0,
            "avoidable_dead": avoid_dead,
            "all_dead_states": all_dead_n,
            "lockout_picks": lockout_pick,
            "dead_picks": dead_pick,
            "margin_buckets": {
                lbl: {"n": len(v),
                      "reg_mean": statistics.fmean(v) if v else 0.0}
                for lbl, v in bucket_stats.items()
                for _, _, _lbl in MARGIN_BUCKETS if _lbl == lbl
            },
        }
        # agree_dis needs the student-vs-teacher count; recompute cleanly:
        return res

    def agreement_on_dis(model, states, dis_set):
        model.eval()
        a = 0
        with torch.no_grad():
            for s in states:
                if s not in dis_set:
                    continue
                lo, hi = bank.candidate_slice(s)
                sc = model(cand_boards[lo:hi], cand_cur[lo:hi],
                           cand_nxt[lo:hi]).flatten()
                a += int(lo + int(sc.argmax()) == teacher_pick[s])
        return a / max(1, len([s for s in states if s in dis_set]))

    pre_tr = informative_metrics(net0, tr_states, dis_tr, "pre_train")
    pre_va = informative_metrics(net0, va_states, dis_va, "pre_val")
    pre_tr["agree_dis"] = agreement_on_dis(net0, tr_states, dis_tr)
    pre_va["agree_dis"] = agreement_on_dis(net0, va_states, dis_va)
    print(f"PRE train: agree {pre_tr['agree_all']:.3f} "
          f"agree_dis {pre_tr['agree_dis']:.3f} "
          f"reg|dis-live {pre_tr['reg_dis_live_mean']:.2f} "
          f"(n {pre_tr['reg_dis_live_n']}) "
          f"reg|live {pre_tr['reg_live_mean']:.2f} "
          f"avoid-dead {pre_tr['avoidable_dead']} "
          f"all-dead {pre_tr['all_dead_states']}", flush=True)
    print(f"PRE val:   agree {pre_va['agree_all']:.3f} "
          f"agree_dis {pre_va['agree_dis']:.3f} "
          f"reg|dis-live {pre_va['reg_dis_live_mean']:.2f} "
          f"(n {pre_va['reg_dis_live_n']}) "
          f"reg|live {pre_va['reg_live_mean']:.2f} "
          f"avoid-dead {pre_va['avoidable_dead']} "
          f"all-dead {pre_va['all_dead_states']}", flush=True)

    # ---- fit: mini-batch CE, adequate budget ----
    # Train states only: >1 candidate, not all-dead (imitating a dead
    # teacher pick is wrong; those states are counted, not fitted).
    fit_states = [s for s in tr_states
                  if (bank.candidate_slice(s)[1] -
                      bank.candidate_slice(s)[0]) > 1
                  and not teacher_alldead[s]]
    print(f"fit pool: {len(fit_states)} train states "
          f"({len(tr_states) - len(fit_states)} skipped: "
          f"singles/all-dead)", flush=True)
    pre_slices = {}
    for s in fit_states:
        lo, hi = bank.candidate_slice(s)
        pre_slices[s] = (cand_boards[lo:hi], cand_cur[lo:hi],
                         cand_nxt[lo:hi], int(teacher_pick[s]) - lo)

    student = TAPNet(output="linear")
    student.load_state_dict(torch.load(str(weights_path),
                                       map_location="cpu"))
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    w0 = [p.detach().clone() for p in student.parameters()]
    sched = [rng.sample(fit_states, min(args.batch_size, len(fit_states)))
             for _ in range(args.steps)]
    losses: list[float] = []
    history: list[dict] = []
    t1 = time.perf_counter()

    def snapshot(step):
        post_tr = informative_metrics(student, tr_states, dis_tr,
                                      f"step{step}_train")
        post_va = informative_metrics(student, va_states, dis_va,
                                      f"step{step}_val")
        post_tr["agree_dis"] = agreement_on_dis(student, tr_states, dis_tr)
        post_va["agree_dis"] = agreement_on_dis(student, va_states, dis_va)
        dW = float(sum(((a - b) ** 2).sum()
                       for a, b in zip(w0, student.parameters())).sqrt()
                   .detach())
        rec = {"step": step,
               "loss": round(statistics.fmean(losses[-100:]), 4),
               "dW": round(dW, 4), "train": post_tr, "val": post_va}
        print(f"  step {step}/{args.steps}: "
              f"loss={statistics.fmean(losses[-100:]):.4f} "
              f"train agree {post_tr['agree_all']:.3f}/"
              f"{post_tr['agree_dis']:.3f} "
              f"val agree {post_va['agree_all']:.3f}/"
              f"{post_va['agree_dis']:.3f} "
              f"val reg|dis {post_va['reg_dis_live_mean']:.2f} "
              f"avoid-dead {post_va['avoidable_dead']} dW {dW:.2f}",
              flush=True)
        return rec

    for step in range(args.steps):
        idx = sched[step]
        opt.zero_grad()
        tot_loss, tot_n = 0.0, 0
        for s in idx:
            b, c, nx, tgt_local = pre_slices[s]
            logits = student(b, c, nx).flatten()
            tot_loss = tot_loss + F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([tgt_local]))
            tot_n += 1
        (tot_loss / max(1, tot_n)).backward()
        opt.step()
        losses.append(float((tot_loss / max(1, tot_n)).detach()))
        if (step + 1) % 200 == 0:
            history.append(snapshot(step + 1))
    fit_wall = time.perf_counter() - t1
    print(f"fit done in {fit_wall:.0f}s; "
          f"final loss {statistics.fmean(losses[-50:]):.4f}", flush=True)

    post_tr = informative_metrics(student, tr_states, dis_tr, "post_train")
    post_va = informative_metrics(student, va_states, dis_va, "post_val")
    post_tr["agree_dis"] = agreement_on_dis(student, tr_states, dis_tr)
    post_va["agree_dis"] = agreement_on_dis(student, va_states, dis_va)
    dW = float(sum(((a - b) ** 2).sum()
                   for a, b in zip(w0, student.parameters())).sqrt()
               .detach())

    if post_va["agree_dis"] >= 0.70:
        verdict = ("PREDICTS OFFLINE — proceed to student-state "
                   "aggregation + guarded closed-loop survival")
    elif post_tr["agree_dis"] >= 0.70:
        verdict = ("FITS TRAIN, NOT VAL — coverage gap, not capacity gap; "
                   "aggregation (student-state labels) is the next test, "
                   "not more frozen-bank steps")
    else:
        verdict = ("CANNOT FIT TRAIN at this budget — reconsider "
                   "student/setup; do NOT collect more labels")
    print(f"POST train: agree {post_tr['agree_all']:.3f}/"
          f"{post_tr['agree_dis']:.3f} "
          f"reg|dis {post_tr['reg_dis_live_mean']:.2f}", flush=True)
    print(f"POST val:   agree {post_va['agree_all']:.3f}/"
          f"{post_va['agree_dis']:.3f} "
          f"reg|dis {post_va['reg_dis_live_mean']:.2f}", flush=True)
    print(f"verdict: {verdict}", flush=True)

    res = {
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(
            weights_path.read_bytes()).hexdigest(),
        "bank": str(Path(args.bank).resolve()),
        "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "seed": args.seed,
        "train_games": args.train_games, "gamma_ply": args.gamma_ply,
        "n_train": len(tr_states), "n_val": len(va_states),
        "n_dis_train": len(dis_tr), "n_dis_val": len(dis_va),
        "fit_pool": len(fit_states),
        "final_loss": round(statistics.fmean(losses[-50:]), 4),
        "dW": round(dW, 4),
        "C2_sentinel": ("CE uses teacher-argmax indices + live masks only; "
                        "-1e6 appears in EVAL regret vectors, never in loss"),
        "pre_train": pre_tr, "pre_val": pre_va,
        "history": history,
        "post_train": post_tr, "post_val": post_va,
        "verdict": verdict,
        "wall_fit_s": round(fit_wall, 1),
    }
    if args.save_student:
        sp = Path(args.out).resolve().parent / f"fullsplit_seed{args.seed}.pt"
        torch.save(student.state_dict(), str(sp))
        res["student_checkpoint"] = str(sp)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
