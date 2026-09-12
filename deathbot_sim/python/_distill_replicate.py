"""Seed replication of the fullsplit survival distillation (2026-09-11).

The §30 promotion (+468.4, t+14.1, n=300) was produced by ONE training seed
(42). This driver reruns the EXACT recipe at new schedule seeds (43, 44) to
test whether the gain is a property of the recipe or a one-off draw:

  - teacher  = versioned champion (snapshot9 sha 229bbe2a, bonus-free full
    2-ply, fast-path both plies, gamma_ply 0.999) — identical labeling
    procedure to diag_fullsplit_fit.py (same helper, same bank order).
  - fit pool = train games 0-23, multi-candidate, not-all-dead (17,265;
    asserted against the original run).
  - fit      = CE toward teacher argmax, Adam lr 1e-4, batch 256, 1200
    steps. The ONLY stochastic element is the minibatch schedule seed.
    Batched segmented CE (mathematically identical to the original's
    per-state loop on this pure conv/linear net; not bit-identical).

Eval: REPORTING ONLY, paired vs snapshot9 in the same sweep harness:
  (a) training seeds 26000-26007 (8; consumed as training data in §26.2),
  (b) demo-reporting seeds 31000-31029 (30; §28 demo range).
The original fullsplit_seed42.pt is evaluated in the same harness so all
arms share one tape set. No dev/promotion panels consumed; no champion
change implied either way (the promoted champion is fullsplit_seed42
regardless; this tests recipe robustness).

Usage:
    python _distill_replicate.py --seeds 43 44 \
        --out ../_diag_out/_fullsplit_replicate.json
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
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"

from model import TAPNet, piece_to_index  # noqa: E402
from diag_phase2 import Bank  # noqa: E402
from afterstate import flat_rows_to_tensor  # noqa: E402
from diag_tinyset_overfit import teacher_scores_for_state  # noqa: E402

DEAD = -1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(D / "state_bank_snapshot9_32games.npz"))
    ap.add_argument("--weights", default=str(ROOT / "model_weights_snapshot9.pt"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    ap.add_argument("--train-games", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(42)
    bank = Bank(args.bank)
    n = bank.n_states
    wpath = Path(args.weights)
    print(f"bank {n} states; seeds {args.seeds}; steps {args.steps} "
          f"lr {args.lr} batch {args.batch_size}", flush=True)

    net0 = TAPNet(output="linear")
    net0.load_state_dict(torch.load(str(wpath), map_location="cpu"))
    net0.eval()

    placed, nxt = bank._replay_piece_ids()
    n_cand = int(bank.cand_offsets[-1])
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        cand_cur[lo:hi] = piece_to_index(placed[s])
        cand_nxt[lo:hi] = piece_to_index(nxt[s])

    game_of = np.asarray(bank.game_ids)
    game_ids_sorted = sorted(set(int(g) for g in game_of))
    train_set = set(game_ids_sorted[:args.train_games])
    train_states = [s for s in range(n) if int(game_of[s]) in train_set]
    val_states = [s for s in range(n) if int(game_of[s]) not in train_set]

    # ---- teacher labeling (once; identical to the original run) ----
    t0 = time.perf_counter()
    teacher_pick: dict[int, int] = {}
    alldead: dict[int, bool] = {}
    import deathbot_sim as ds
    with torch.inference_mode():
        for si, s in enumerate(train_states + val_states):
            lo, hi = bank.candidate_slice(s)
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            tv, viable_live = teacher_scores_for_state(
                net0, pre, placed[s], nxt[s], args.gamma_ply)
            if tv is None:
                continue
            viable = [lo + p for p in viable_live]
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            teacher_pick[s] = viable[order[0]]
            alldead[s] = all(float(v) <= DEAD / 2 for v in tv)
            if (si + 1) % 4000 == 0:
                print(f"  teacher {si + 1}/{n} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"teacher done {time.perf_counter() - t0:.0f}s "
          f"({len(teacher_pick)} states)", flush=True)

    tr = [s for s in train_states if s in teacher_pick]
    va = [s for s in val_states if s in teacher_pick]
    fit_states = [s for s in tr
                  if (bank.candidate_slice(s)[1]
                      - bank.candidate_slice(s)[0]) > 1
                  and not alldead[s]]
    print(f"split {len(tr)}/{len(va)}; fit pool {len(fit_states)} "
          f"(original run: 17265)", flush=True)

    def agree(model, states, limit=None):
        a = t = 0
        with torch.no_grad():
            for s in (states if limit is None else states[:limit]):
                lo, hi = bank.candidate_slice(s)
                sc = model(cand_boards[lo:hi], cand_cur[lo:hi],
                           cand_nxt[lo:hi]).flatten()
                a += int(lo + int(sc.argmax()) == teacher_pick[s])
                t += 1
        return a / max(1, t)

    pre_tr_agree = agree(net0, tr)
    print(f"PRE train agree_all {pre_tr_agree:.4f} "
          f"(original: 0.7867)", flush=True)

    results = {
        "weights_sha": hashlib.sha256(wpath.read_bytes()).hexdigest()[:16],
        "bank": str(Path(args.bank).resolve()),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "n_train": len(tr), "n_val": len(va), "fit_pool": len(fit_states),
        "pre_train_agree_all": round(pre_tr_agree, 4),
        "arms": {},
    }

    ckpts = {"snap9": wpath,
             "fullsplit_seed42": D / "fullsplit_seed42.pt"}

    for seed in args.seeds:
        rng = random.Random(seed)
        sched = [rng.sample(fit_states, min(args.batch_size, len(fit_states)))
                 for _ in range(args.steps)]
        # pre-batch (concatenated candidates per step)
        batches = []
        for idx in sched:
            bl, cl, nl, tl, lens = [], [], [], [], []
            for s in idx:
                lo, hi = bank.candidate_slice(s)
                bl.append(cand_boards[lo:hi])
                cl.append(cand_cur[lo:hi])
                nl.append(cand_nxt[lo:hi])
                tl.append(int(teacher_pick[s]) - lo)
                lens.append(hi - lo)
            batches.append((torch.cat(bl), torch.cat(cl), torch.cat(nl),
                            torch.tensor(tl, dtype=torch.long), lens))
        student = TAPNet(output="linear")
        student.load_state_dict(torch.load(str(wpath), map_location="cpu"))
        student.train()
        opt = torch.optim.Adam(student.parameters(), lr=args.lr)
        w0 = [p.detach().clone() for p in student.parameters()]
        losses: list[float] = []
        t1 = time.perf_counter()
        hist = []
        for step in range(args.steps):
            b, c, nx, tl, lens = batches[step]
            opt.zero_grad()
            scores = student(b, c, nx).flatten()
            total = scores.new_zeros(())
            off = 0
            for li, t in zip(lens, tl):
                seg = scores[off:off + li]
                off += li
                total = total + F.cross_entropy(seg.unsqueeze(0),
                                                torch.tensor([t]))
            loss = total / len(lens)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 200 == 0:
                dW = float(sum(((a - bb) ** 2).sum()
                               for a, bb in zip(w0, student.parameters())
                               ).sqrt().detach())
                ag = agree(student, tr, limit=3000)
                hist.append({"step": step + 1, "loss":
                             round(statistics.fmean(losses[-100:]), 4),
                             "dW": round(dW, 3),
                             "agree3k": round(ag, 4)})
                print(f"  seed{seed} step {step + 1}/{args.steps}: "
                      f"loss {hist[-1]['loss']:.4f} dW {dW:.2f} "
                      f"agree3k {ag:.4f} ({time.perf_counter() - t1:.0f}s)",
                      flush=True)
        wall = time.perf_counter() - t1
        dW = float(sum(((a - bb) ** 2).sum()
                       for a, bb in zip(w0, student.parameters())
                       ).sqrt().detach())
        post_agree = agree(student, tr)
        post_agree_val = agree(student, va)
        ckpt = D / f"_fullsplit_repl_seed{seed}.pt"
        torch.save(student.state_dict(), str(ckpt))
        ckpts[f"repl_seed{seed}"] = ckpt
        results["arms"][f"repl_seed{seed}"] = {
            "seed": seed, "wall_fit_s": round(wall, 1),
            "final_loss": round(statistics.fmean(losses[-50:]), 4),
            "dW": round(dW, 4),
            "post_train_agree_all": round(post_agree, 4),
            "post_val_agree_all": round(post_agree_val, 4),
            "history": hist,
            "checkpoint": str(ckpt),
        }
        print(f"seed {seed} DONE: dW {dW:.2f} "
              f"agree {post_agree:.4f}/{post_agree_val:.4f} "
              f"({wall:.0f}s)", flush=True)

    # ---- paired eval, REPORTING ONLY ----
    import tempfile
    from evaluate import evaluate_weights_parallel

    def run_eval(weights: Path, games: int, seed0: int) -> list[int]:
        if weights == wpath:
            src = str(weights)
            cleanup = False
        else:
            src = str(weights)
            cleanup = False
        ev = evaluate_weights_parallel(src, games=games, seed=seed0,
                                       max_pieces=1500, workers=8)
        return [g.pieces for g in ev.games]

    for tag, count, seed0 in (("train26000", 8, 26000),
                              ("report31000", 30, 31000)):
        print(f"eval {tag} ({count} games x {len(ckpts)} arms)...", flush=True)
        per = {}
        for name, p in ckpts.items():
            per[name] = run_eval(p, count, seed0)
        base = per["snap9"]
        row = {}
        for name, pieces in per.items():
            dl = [x - y for x, y in zip(pieces, base)]
            m = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
            row[name] = {"mean": round(statistics.fmean(pieces), 1),
                         "capped": sum(p >= 1500 for p in pieces),
                         "delta_vs_snap9": round(m, 1), "t": round(t, 2),
                         "wins": sum(d > 0 for d in dl),
                         "losses": sum(d < 0 for d in dl),
                         "per_seed": pieces}
        results[f"eval_{tag}"] = row
        for name, r in row.items():
            print(f"  {name:20s} mean {r['mean']:7.1f} cap {r['capped']:3d} "
                  f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                  f"W/L {r['wins']}/{r['losses']}", flush=True)

    op = Path(args.out)
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", op, flush=True)


if __name__ == "__main__":
    main()
