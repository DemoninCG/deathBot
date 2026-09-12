"""Gen-2 distillation: champion start, fullsplit-2ply teacher (2026-09-11).

Probe (_teacher2_probe.json) PASSED all pre-registered gates:
  teacher-vs-teacher agreement 0.890 (fs2ply differs from snap9-2ply on
  11% of states); student-vs-fs2ply dis 17.8% (4102 states); NEW targets
  782 (student == snap9-teacher-pick but != fs2ply-pick — information the
  E3 negative could not contain); affordability median -1.1 (all) / -1.1
  (new targets), 0/4102 picks <= -20 on the fullsplit net's own v1 scale.

This is the clean complement to E3: E3 = same teacher (snap9-2ply) into
the distilled start (info-exhausted -> nothing). Here: same start (the
champion fullsplit_seed42) with a NEW teacher (its own 2-ply expansion —
a function the student has never been fit toward). This is a gen-2
self-improvement loop: distill the champion's own 2-ply behavior back
into its 1-ply weights.

Dose: the validated recipe (lr 1e-4, batch 256, 1200 steps, full-batch
CE toward teacher argmax) — confirmed at the play-optimal dose by the
2026-09-11 dose sweep (3e-4 n.s. +44; 1e-3 trending worse).

Stage 0: label the bank with the fs2ply teacher ONCE, cache to NPZ
         (~24 min; asserted against probe counts).
Stage 1: fit at schedule seed 42 (~35 min).
Stage 2: screen REPORTING-ONLY: 31000-31029 (30) + 26000-26007 (8),
         paired vs snap9 AND vs the champion (s42). Per the 2026-09-11
         soup lesson: NO promotion from this screen; a win must replicate
         across schedule seeds 43/44 before any fresh-panel gate.

Usage:
    python _distill_gen2.py --out ../_diag_out/_distill_gen2.json
Incremental: writes after every stage; interrupt-safe.
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

CHAMP_SHA = "9989ce86390bcba6"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_distill_gen2.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}", flush=True)
    results.setdefault("note",
                       "Gen-2: champion start + fullsplit-2ply teacher; "
                       "reporting-only seeds; no panels consumed.")
    results.setdefault("cells", {})
    results.setdefault("eval_report31000", {})
    results.setdefault("eval_train26000", {})

    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from diag_tinyset_overfit import teacher_scores_for_state
    from afterstate import flat_rows_to_tensor
    import deathbot_sim as ds

    bank = Bank(str(D / "state_bank_snapshot9_32games.npz"))
    n = bank.n_states
    start_path = D / "fullsplit_seed42.pt"
    assert hashlib.sha256(start_path.read_bytes()).hexdigest()[:16] == CHAMP_SHA

    start = TAPNet(output="linear")
    start.load_state_dict(torch.load(str(start_path), map_location="cpu"))
    start.eval()

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
    train_set = set(game_ids_sorted[:24])

    # ---- stage 0: teacher labels (cache) ----
    lab_path = D / "_teacher_labels_fs2ply_bank.npz"
    if lab_path.exists():
        z = np.load(str(lab_path))
        teacher_pick = {int(s): int(p) for s, p in
                        zip(z["state_index"], z["teacher_pick"])}
        alldead = {int(s): bool(a) for s, a in zip(z["state_index"],
                                                   z["alldead"])}
        print(f"labels cached: {len(teacher_pick)}", flush=True)
    else:
        teacher_pick: dict[int, int] = {}
        alldead: dict[int, bool] = {}
        t0 = time.perf_counter()
        with torch.inference_mode():
            for s in range(n):
                pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
                tv, viable_live = teacher_scores_for_state(
                    start, pre, placed[s], nxt[s], args.gamma_ply)
                if tv is None:
                    continue
                lo, _hi = bank.candidate_slice(s)
                viable = [lo + p for p in viable_live]
                order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
                teacher_pick[s] = viable[order[0]]
                alldead[s] = all(float(v) <= -5e5 for v in tv)
                if len(teacher_pick) % 4000 == 0:
                    print(f"  fs2ply label {len(teacher_pick)} "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
        print(f"labeling done ({time.perf_counter() - t0:.0f}s)", flush=True)
        fit_pool_probe = [s for s in range(n)
                          if s in teacher_pick and int(game_of[s]) in train_set
                          and (bank.candidate_slice(s)[1]
                               - bank.candidate_slice(s)[0]) > 1
                          and not alldead[s]]
        np.savez(str(lab_path),
                 state_index=np.array(sorted(teacher_pick), dtype=np.int32),
                 teacher_pick=np.array([teacher_pick[s]
                                        for s in sorted(teacher_pick)],
                                       dtype=np.int64),
                 alldead=np.array([alldead[s] for s in sorted(teacher_pick)],
                                  dtype=bool),
                 fit_pool=np.array(fit_pool_probe, dtype=np.int32))
        print(f"cached {lab_path.name}", flush=True)

    fit_pool = [s for s in range(n)
                if s in teacher_pick and int(game_of[s]) in train_set
                and (bank.candidate_slice(s)[1]
                     - bank.candidate_slice(s)[0]) > 1
                and not alldead[s]]
    tr = [s for s in range(n) if s in teacher_pick
          and int(game_of[s]) in train_set]
    va = [s for s in range(n) if s in teacher_pick
          and int(game_of[s]) not in train_set]
    print(f"fit pool {len(fit_pool)} (gen-1 was 17265); "
          f"tr {len(tr)} va {len(va)}", flush=True)

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

    # PRE: how often does the START already follow the new teacher?
    pre_tr = agree(start, tr)
    pre_va = agree(start, va)
    print(f"PRE agree vs fs2ply teacher: train {pre_tr:.4f} "
          f"val {pre_va:.4f}", flush=True)
    results["pre_agree"] = {"train": round(pre_tr, 4),
                            "val": round(pre_va, 4)}
    results["fit_pool"] = len(fit_pool)
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for seed in args.seeds:
        tag = f"gen2_seed{seed}"
        if tag in results["cells"] and (D / f"_{tag}.pt").exists():
            print(f"{tag}: already fit", flush=True)
            continue
        rng = random.Random(seed)
        student = TAPNet(output="linear")
        student.load_state_dict(torch.load(str(start_path),
                                           map_location="cpu"))
        student.train()
        opt = torch.optim.Adam(student.parameters(), lr=1e-4)
        w0 = [p.detach().clone() for p in student.parameters()]
        losses: list[float] = []
        hist = []
        t1 = time.perf_counter()
        for step in range(args.steps):
            idx = rng.sample(fit_pool, 256)
            bl, cl, nl, tl, lens = [], [], [], [], []
            for s in idx:
                lo, hi = bank.candidate_slice(s)
                bl.append(cand_boards[lo:hi])
                cl.append(cand_cur[lo:hi])
                nl.append(cand_nxt[lo:hi])
                tl.append(teacher_pick[s] - lo)
                lens.append(hi - lo)
            b, c, nx = torch.cat(bl), torch.cat(cl), torch.cat(nl)
            tl = torch.tensor(tl, dtype=torch.long)
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
                hist.append({"step": step + 1,
                             "loss": round(statistics.fmean(losses[-100:]), 4),
                             "dW": round(dW, 3), "agree3k": round(ag, 4)})
                print(f"  {tag} step {step + 1}/{args.steps}: "
                      f"loss {hist[-1]['loss']:.4f} dW {dW:.2f} "
                      f"agree3k {ag:.4f} ({time.perf_counter() - t1:.0f}s)",
                      flush=True)
        wall = time.perf_counter() - t1
        dW = float(sum(((a - bb) ** 2).sum()
                       for a, bb in zip(w0, student.parameters())
                       ).sqrt().detach())
        ckpt = D / f"_{tag}.pt"
        # SAVE FIRST — nothing else may run before the checkpoint is on
        # disk (2026-09-11 lesson: the original run lost its completed
        # fit because results assembly crashed before torch.save).
        torch.save(student.state_dict(), str(ckpt))
        assert ckpt.exists(), "checkpoint save failed"
        post_tr = agree(student, tr)
        post_va = agree(student, va)
        results["cells"][tag] = {
            "start": "fullsplit_seed42 (champion)",
            "teacher": "fullsplit_seed42 + full 2-ply (fs2ply)",
            "steps": args.steps, "lr": args.lr, "schedule_seed": seed,
            "final_loss": round(statistics.fmean(losses[-50:]), 4),
            "dW": round(dW, 4),
            "post_train_agree": round(post_tr, 4),
            "post_val_agree": round(post_va, 4),
            "history": hist,
            "checkpoint": str(ckpt),
            "sha": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16],
        }
        op.write_text(json.dumps(results, indent=2), encoding="utf-8")
        c = results["cells"][tag]
        print(f"{tag} DONE: dW {c['dW']} agree "
              f"{c['post_train_agree']}/{c['post_val_agree']} "
              f"({wall:.0f}s) — SAVED", flush=True)

    # ---- stage 2: reporting-only screen ----
    from evaluate import evaluate_weights_parallel
    ckpts = {"snap9": ROOT / "model_weights_snapshot9.pt",
             "s42_champ": start_path}
    for seed in args.seeds:
        p = D / f"_gen2_seed{seed}.pt"
        if p.exists():
            ckpts[f"gen2_seed{seed}"] = p
    for tag_leg, count, seed0 in (("eval_train26000", 8, 26000),
                                  ("eval_report31000", 30, 31000)):
        leg = results.setdefault(tag_leg, {})
        if "snap9" not in leg:
            ev = evaluate_weights_parallel(str(ckpts["snap9"]),
                                           games=count, seed=seed0,
                                           max_pieces=1500, workers=8)
            leg["snap9"] = {"mean": round(statistics.fmean(
                [g.pieces for g in ev.games]), 1),
                "per_seed": [g.pieces for g in ev.games]}
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
        base = leg["snap9"]["per_seed"]
        for name in sorted(ckpts):
            if name == "snap9" or name in leg:
                continue
            ev = evaluate_weights_parallel(str(ckpts[name]), games=count,
                                           seed=seed0, max_pieces=1500,
                                           workers=8)
            pieces = [g.pieces for g in ev.games]
            dl = [a - b for a, b in zip(pieces, base)]
            m = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
            leg[name] = {"mean": round(statistics.fmean(pieces), 1),
                         "capped": sum(p >= 1500 for p in pieces),
                         "delta_vs_snap9": round(m, 1), "t": round(t, 2),
                         "wins": sum(x > 0 for x in dl),
                         "losses": sum(x < 0 for x in dl),
                         "per_seed": pieces}
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            r = leg[name]
            print(f"{tag_leg} {name:14s} mean {r['mean']:7.1f} "
                  f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                  f"— SAVED", flush=True)

    # paired vs champion within the same leg
    for tag_leg in ("eval_train26000", "eval_report31000"):
        leg = results.get(tag_leg, {})
        if "s42_champ" in leg:
            for seed in args.seeds:
                a = leg.get(f"gen2_seed{seed}", {}).get("per_seed")
                b = leg["s42_champ"]["per_seed"]
                if not a:
                    continue
                dl = [x - y for x, y in zip(a, b)]
                m = statistics.fmean(dl)
                sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
                t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
                leg[f"gen2_vs_champ_seed{seed}"] = {
                    "delta": round(m, 1), "t": round(t, 2),
                    "wins": sum(x > 0 for x in dl),
                    "losses": sum(x < 0 for x in dl)}
                op.write_text(json.dumps(results, indent=2),
                              encoding="utf-8")
                print(f"{tag_leg} gen2_seed{seed}-vs-champ: {m:+.1f} "
                      f"t {t:+.2f}", flush=True)


if __name__ == "__main__":
    main()