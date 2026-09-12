"""Dose-response screen for the validated distillation recipe (2026-09-11).

§49 established the fullsplit recipe (CE toward the frozen 2-ply teacher's
argmax, from snapshot9) at ONE dose point: lr 1e-4, batch 256, 1200 steps.
The §26 trajectory shows the fit had NOT converged at cutoff: dW grew
linearly (2.11@200 -> 4.69@1200), loss still falling, val agree flat since
~800. Question (user): is the successful recipe itself under-dosed — would
more optimization rate or budget extract more of the teacher?

Cells (all from snapshot9, schedule seed 42 = the same minibatch schedule
as the champion product, cached teacher labels, batched CE identical to
_distill_replicate.py's):
  cell B: lr 3e-4, 1200 steps   (rate x3; the §24 untested axis)
  cell C: lr 1e-4, 2400 steps   (budget x2; continues the unconverged fit)
  (cell D lr 3e-4 x 2400 deferred — run only if B or C screens positive)

Screen: paired vs snap9 AND vs s42 (the standard-dose same-schedule
product) on 26000s (8) + 31000s (30), sweep harness, cap 1500.
Decision rule (pre-registered): a cell advances to 2-seed replication
(seeds 43/44) only if it beats s42 by >= +100 mean on 31000s; after
replication, any claim goes through a fresh-panel gate (21110+) like §50.
This ordering exists because §50 just demonstrated the n=30 mirage.

Notes:
  - agree3k probes are BATCHED here (the replicate's unbatched probe cost
    ~5 min/cell); dose cells are ~15-20 min each.
  - Teacher labels come from _diag_out/_teacher_labels_snap9_bank.npz
    (fit pool asserted == 17265). If missing, run _cache_teacher_labels.py.
  - All eval seeds are reporting-only (26000s training, 31000s demo-report);
    the gate panel 21110+ is consumed only by the §50-style confirm step.

Usage:
    python _dose_screen.py --cells B C --out ../_diag_out/_dose_screen.json
Incremental: writes after each cell (fit metrics AND eval legs separately).
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

CELLS = {  # id -> (lr, steps)
    "B": (3e-4, 1200),
    "C": (1e-4, 2400),
    "D": (3e-4, 2400),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["B", "C"])
    ap.add_argument("--out", default=str(D / "_dose_screen.json"))
    ap.add_argument("--schedule-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming from {op.name}: {sorted(results.get('cells', {}))}",
              flush=True)
    results.setdefault("note",
                       "Dose-response screen over the validated fullsplit "
                       "recipe (schedule seed 42, cached labels). Gate: "
                       "replicate at seeds 43/44 before any fresh panel.")
    results.setdefault("cells", {})

    # ---- labels (cached) ----
    z = np.load(str(D / "_teacher_labels_snap9_bank.npz"))
    state_index = z["state_index"].tolist()
    pick_of = dict(zip(state_index, z["teacher_pick"].tolist()))
    alldead_of = dict(zip(state_index, z["alldead"].tolist()))
    fit_pool = z["fit_pool"].tolist()
    assert len(fit_pool) == 17265, f"fit pool {len(fit_pool)} != 17265"
    print(f"labels: {len(pick_of)} states, fit pool {len(fit_pool)}",
          flush=True)

    # ---- bank tensors ----
    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from afterstate import flat_rows_to_tensor

    bank = Bank(str(D / "state_bank_snapshot9_32games.npz"))
    n = bank.n_states
    snap9_path = ROOT / "model_weights_snapshot9.pt"
    net0 = TAPNet(output="linear")
    net0.load_state_dict(torch.load(str(snap9_path), map_location="cpu"))
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
    train_set = set(game_ids_sorted[:24])
    tr = [s for s in state_index if int(game_of[s]) in train_set]
    va = [s for s in state_index if int(game_of[s]) not in train_set]
    fit_states = [s for s in fit_pool]

    # ---- batched agree probe (fast; replaces the replicate's unbatched) --
    def agree(model, states, chunk=600):
        a = t = 0
        with torch.no_grad():
            for i in range(0, len(states), chunk):
                idx = states[i:i + chunk]
                bl, cl, nl, tl = [], [], [], []
                for s in idx:
                    lo, hi = bank.candidate_slice(s)
                    bl.append(cand_boards[lo:hi])
                    cl.append(cand_cur[lo:hi])
                    nl.append(cand_nxt[lo:hi])
                    tl.append(int(pick_of[s]) - lo)
                boards = torch.cat(bl)
                cur = torch.cat(cl)
                nx = torch.cat(nl)
                sc = model(boards, cur, nx).flatten()
                off = 0
                for s, li, tloc in zip(idx, [len(b) for b in bl], tl):
                    seg = sc[off:off + li]
                    off += li
                    a += int(int(seg.argmax()) == tloc)
                    t += 1
        return a / max(1, t)

    # ---- schedule (fixed at args.schedule_seed; same convention as the
    #      original run: rng.sample per step, then concatenated) ----
    rng = random.Random(args.schedule_seed)
    batches = []
    for _ in range(max(CELLS[c][1] for c in args.cells)):
        idx = rng.sample(fit_states, min(args.batch_size, len(fit_states)))
        bl, cl, nl, tl, lens = [], [], [], [], []
        for s in idx:
            lo, hi = bank.candidate_slice(s)
            bl.append(cand_boards[lo:hi])
            cl.append(cand_cur[lo:hi])
            nl.append(cand_nxt[lo:hi])
            tl.append(int(pick_of[s]) - lo)
            lens.append(hi - lo)
        batches.append((torch.cat(bl), torch.cat(cl), torch.cat(nl),
                        torch.tensor(tl, dtype=torch.long), lens))
    print(f"schedule built: {len(batches)} steps", flush=True)

    pre_tr = agree(net0, tr[:3000])
    pre_va = agree(net0, va)
    print(f"PRE agree tr(3000) {pre_tr:.4f} va {pre_va:.4f}", flush=True)

    # ---- spread probe (§49 signature: informative distill halves spread) --
    def spread(model, states):
        vals = []
        with torch.no_grad():
            for s in states[:300]:
                lo, hi = bank.candidate_slice(s)
                sc = model(cand_boards[lo:hi], cand_cur[lo:hi],
                           cand_nxt[lo:hi]).flatten()
                vals.append(float(sc.max() - sc.min()))
        return statistics.fmean(vals)

    spread0 = spread(net0, fit_states)
    print(f"PRE within-state spread {spread0:.1f} (§49: success halves to "
          f"~17; sharpen doubles to 77)", flush=True)

    for cell in args.cells:
        lr, steps = CELLS[cell]
        if cell in results["cells"] and \
                results["cells"][cell].get("eval_report31000"):
            print(f"cell {cell}: done", flush=True)
            continue
        rec = results["cells"].setdefault(cell, {"lr": lr, "steps": steps})
        student = TAPNet(output="linear")
        student.load_state_dict(torch.load(str(snap9_path), map_location="cpu"))
        student.train()
        opt = torch.optim.Adam(student.parameters(), lr=lr)
        w0 = [p.detach().clone() for p in student.parameters()]
        losses: list[float] = []
        hist = []
        t1 = time.perf_counter()
        for step in range(steps):
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
                rec = {"step": step + 1,
                       "loss": round(statistics.fmean(losses[-100:]), 4),
                       "dW": round(dW, 3)}
                hist.append(rec)
                print(f"  {cell} step {step + 1}/{steps}: loss "
                      f"{rec['loss']:.4f} dW {dW:.2f} "
                      f"({time.perf_counter() - t1:.0f}s)", flush=True)
        dW = float(sum(((a - bb) ** 2).sum()
                       for a, bb in zip(w0, student.parameters())
                       ).sqrt().detach())
        post_tr = agree(student, tr[:3000])
        post_va = agree(student, va)
        post_spread = spread(student, fit_states)
        rec.update({
            "dW": round(dW, 3),
            "final_loss": round(statistics.fmean(losses[-50:]), 4),
            "post_agree_tr3000": round(post_tr, 4),
            "post_agree_va": round(post_va, 4),
            "spread": round(post_spread, 1),
            "wall_fit_s": round(time.perf_counter() - t1, 1),
            "history": hist,
        })
        ck = D / f"_dose_{cell}_seed{args.schedule_seed}.pt"
        torch.save(student.state_dict(), str(ck))
        rec["checkpoint"] = str(ck)
        op.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"cell {cell} FIT DONE: dW {dW:.2f} agree "
              f"{post_tr:.4f}/{post_va:.4f} spread {post_spread:.1f} "
              f"({rec['wall_fit_s']:.0f}s) — SAVED", flush=True)

        # ---- eval legs (reporting only) ----
        from evaluate import evaluate_weights_parallel
        for tag, count, seed0 in (("eval_train26000", 8, 26000),
                                  ("eval_report31000", 30, 31000)):
            ev = evaluate_weights_parallel(str(ck), games=count, seed=seed0,
                                           max_pieces=1500, workers=8)
            rec[tag] = {"mean": round(statistics.fmean(
                [g.pieces for g in ev.games]), 1),
                "capped": sum(g.capped for g in ev.games),
                "per_seed": [g.pieces for g in ev.games]}
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"cell {cell} {tag}: mean {rec[tag]['mean']} cap "
                  f"{rec[tag]['capped']}/{count} — SAVED", flush=True)

    # ---- paired comparisons vs snap9 and s42 (from §49.3/§50 records) ----
    ref = {}
    rep = json.loads((D / "_fullsplit_replicate.json").read_text(
        encoding="utf-8"))
    for tag in ("eval_train26000", "eval_report31000"):
        ref.setdefault(tag, {})["snap9"] = \
            rep[tag]["snap9"]["per_seed"]
        ref[tag]["s42"] = rep[tag]["fullsplit_seed42"]["per_seed"]
    for cell, rec in results["cells"].items():
        comp = {}
        for tag, legs in ref.items():
            for name, base in legs.items():
                dl = [a - b for a, b in zip(rec[tag]["per_seed"], base)]
                m = statistics.fmean(dl)
                sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
                t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
                comp[f"{tag}_vs_{name}"] = {
                    "delta": round(m, 1), "t": round(t, 2),
                    "wins": sum(x > 0 for x in dl),
                    "losses": sum(x < 0 for x in dl)}
        rec["comparisons"] = comp
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== DOSE SCREEN SUMMARY ===", flush=True)
    for cell, rec in results["cells"].items():
        c = rec.get("comparisons", {})
        b31 = c.get("eval_report31000_vs_s42", {})
        print(f"cell {cell} (lr {rec['lr']:.0e} x {rec['steps']}): dW "
              f"{rec['dW']} spread {rec['spread']} | 31000s mean "
              f"{rec['eval_report31000']['mean']} vs s42 "
              f"{b31.get('delta', '+?')} (t {b31.get('t')})", flush=True)


if __name__ == "__main__":
    main()
