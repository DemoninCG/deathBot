"""LR/dose sweep on the validated survival-distillation recipe (2026-09-11).

Question (user): the validated recipe was tested at exactly ONE dose point
(lr 1e-4 x 1200 steps x batch 256 full-batch CE toward the frozen 2-ply
teacher's argmax, from snapshot9). Is the +468 past the dose knee, or was
the successful run itself still under-dosed?

Evidence motivating this (§49/§26):
  - §26 trajectory: dW grew LINEARLY 2.1@200 -> 4.7@1200 with loss still
    falling and val flat since ~800 — the fit never converged on train.
  - §49 failure class 1: every under-dosed run (lr 1e-5) was an
    un-executed fit; the dose-response curve above lr 1e-4 is UNMEASURED.
  - E3 (negative) ran from the ALREADY-DISTILLED start — it bounds
    re-extraction, not the first fit's dose optimum.

Cells (all from snapshot9, identical CE recipe, schedule seed 42; label
cache _teacher_labels_snap9_bank.npz asserted: fit pool 17265):
  base     lr 1e-4 / 1200 steps  (reproduces the validated recipe;
                                   sanity anchor vs §49.3 numbers)
  lr3e-4   lr 3e-4 / 1200 steps  (3x per-step dose)
  lr1e-3   lr 1e-3 / 1200 steps  (10x; does Adam destabilize?)
  steps2x  lr 1e-4 / 2400 steps  (2x horizon, same per-step dose)
  lr3e-5   lr 3e-5 / 2400 steps  (1/3 dose, extended horizon — bridges
                                   toward the under-dose regime)

Stages (incremental JSON after every stage; interrupt-safe):
  1. fit each cell (~35 min each at 1200 steps, ~70 for steps2x;
     batches built ONCE and shared across cells)
  2. offline metrics (agree all/dis on train+val vs frozen teacher picks,
     dW, value-spread ratio on 500 val states)
  3. gameplay screen on REPORTING seeds 31000-31029 (30 games, paired vs
     snap9 AND vs s42) + training seeds 26000-26007 (8 games)

Gate discipline (§50 lesson): any cell that beats the base recipe's
gameplay on the reporting panel MUST then be replicated across schedule
seeds 43/44 BEFORE any fresh-panel promotion gate. This driver does stage
1-3 only; replication is a follow-up call.

Usage:
    python _dose_sweep.py --out ../_diag_out/_dose_sweep.json \
        --cells base lr3e-4 lr1e-3 steps2x lr3e-5
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

CELLS = {  # name -> (lr, steps)
    "base": (1e-4, 1200),
    "lr3e-4": (3e-4, 1200),
    "lr1e-3": (1e-3, 1200),
    "steps2x": (1e-4, 2400),
    "lr3e-5": (3e-5, 2400),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_dose_sweep.json"))
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--schedule-seed", type=int, default=42)
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}: cells done = "
              f"{sorted(results.get('cells', {}))}", flush=True)
    results.setdefault("note",
                       "Dose sweep on the validated recipe from snapshot9; "
                       "reporting-only seeds; no panels.")
    results.setdefault("cells", {})
    results.setdefault("eval_report31000", {})
    results.setdefault("eval_train26000", {})

    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from afterstate import flat_rows_to_tensor

    bank = Bank(str(D / "state_bank_snapshot9_32games.npz"))
    n = bank.n_states
    wpath = ROOT / "model_weights_snapshot9.pt"
    labels_path = D / "_teacher_labels_snap9_bank.npz"
    z = np.load(str(labels_path))
    meta = json.loads(str(z["meta"]))
    assert meta["weights_sha"] == "229bbe2a91bab6b1", meta
    assert len(z["fit_pool"]) == 17265, len(z["fit_pool"])
    print(f"labels {len(z['state_index'])}, fit pool {len(z['fit_pool'])} "
          f"(asserted == original run)", flush=True)

    teacher_pick = {int(s): int(p) for s, p in
                    zip(z["state_index"], z["teacher_pick"])}
    fit_states = [int(s) for s in z["fit_pool"]]

    net0 = TAPNet(output="linear")
    net0.load_state_dict(torch.load(str(wpath), map_location="cpu"))
    net0.eval()

    # candidate tensors (shared across cells)
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
    tr = [s for s in range(n) if s in teacher_pick
          and int(game_of[s]) in train_set]
    va = [s for s in range(n) if s in teacher_pick
          and int(game_of[s]) not in train_set]

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

    pre_tr = agree(net0, tr)
    pre_va = agree(net0, va)
    print(f"PRE agree train {pre_tr:.4f} val {pre_va:.4f} "
          f"(original: 0.7867/0.7911)", flush=True)
    results["pre_agree"] = {"train": round(pre_tr, 4),
                            "val": round(pre_va, 4)}
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # build shared batches ONCE (longest horizon; cells index into it)
    max_steps = max(CELLS[c][1] for c in args.cells)
    rng = random.Random(args.schedule_seed)
    print(f"building {max_steps} shared batches...", flush=True)
    t0 = time.perf_counter()
    batches = []
    for _ in range(max_steps):
        idx = rng.sample(fit_states, 256)
        bl, cl, nl, tl, lens = [], [], [], [], []
        for s in idx:
            lo, hi = bank.candidate_slice(s)
            bl.append(cand_boards[lo:hi])
            cl.append(cand_cur[lo:hi])
            nl.append(cand_nxt[lo:hi])
            tl.append(teacher_pick[s] - lo)
            lens.append(hi - lo)
        batches.append((torch.cat(bl), torch.cat(cl), torch.cat(nl),
                        torch.tensor(tl, dtype=torch.long), lens))
    print(f"batches ready ({time.perf_counter() - t0:.0f}s)", flush=True)

    for cell in args.cells:
        if cell not in CELLS:
            raise SystemExit(f"unknown cell {cell}")
        if cell in results["cells"] and (D / f"_dose_{cell}.pt").exists():
            print(f"cell {cell}: already fit", flush=True)
            continue
        lr, steps = CELLS[cell]
        student = TAPNet(output="linear")
        student.load_state_dict(torch.load(str(wpath), map_location="cpu"))
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
                ag = agree(student, tr, limit=3000)
                hist.append({"step": step + 1,
                             "loss": round(statistics.fmean(losses[-100:]), 4),
                             "dW": round(dW, 3), "agree3k": round(ag, 4)})
                print(f"  {cell} step {step + 1}/{steps}: "
                      f"loss {hist[-1]['loss']:.4f} dW {dW:.2f} "
                      f"agree3k {ag:.4f} ({time.perf_counter() - t1:.0f}s)",
                      flush=True)
        wall = time.perf_counter() - t1
        dW = float(sum(((a - bb) ** 2).sum()
                       for a, bb in zip(w0, student.parameters())
                       ).sqrt().detach())
        ckpt = D / f"_dose_{cell}.pt"
        torch.save(student.state_dict(), str(ckpt))
        results["cells"][cell] = {
            "lr": lr, "steps": steps,
            "wall_fit_s": round(wall, 1),
            "final_loss": round(statistics.fmean(losses[-50:]), 4),
            "dW": round(dW, 4),
            "post_train_agree": round(agree(student, tr), 4),
            "post_val_agree": round(agree(student, va), 4),
            "history": hist,
            "checkpoint": str(ckpt),
            "sha": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16],
        }
        op.write_text(json.dumps(results, indent=2), encoding="utf-8")
        c = results["cells"][cell]
        print(f"cell {cell} DONE: dW {c['dW']} agree "
              f"{c['post_train_agree']}/{c['post_val_agree']} "
              f"({c['wall_fit_s']:.0f}s)", flush=True)

    # ---- gameplay screen (reporting + training seeds; paired) ----
    from evaluate import evaluate_weights_parallel
    ckpts = {"snap9": wpath, **{c: D / f"_dose_{c}.pt" for c in args.cells
                                if (D / f"_dose_{c}.pt").exists()}}
    for tag, count, seed0 in (("eval_train26000", 8, 26000),
                              ("eval_report31000", 30, 31000)):
        leg = results.setdefault(tag, {})
        if "snap9" not in leg:
            ev = evaluate_weights_parallel(str(wpath), games=count,
                                           seed=seed0, max_pieces=1500,
                                           workers=8)
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
            print(f"{tag} {name:8s} mean {r['mean']:7.1f} "
                  f"cap {r['capped']:3d}/{count} "
                  f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                  f"W/L {r['wins']}/{r['losses']} — SAVED", flush=True)

    # paired vs the validated-dose cell (base) as well
    rep = results.get("eval_report31000", {})
    if "base" in rep:
        base42 = rep["base"]["per_seed"]
        for name in rep:
            if name in ("snap9", "base"):
                continue
            dl = [a - b for a, b in zip(rep[name]["per_seed"], base42)]
            m = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
            rep[name]["delta_vs_base"] = round(m, 1)
            rep[name]["t_vs_base"] = round(t, 2)
        op.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== DOSE SWEEP SUMMARY ===", flush=True)
    for cell, c in results["cells"].items():
        print(f"  {cell:8s} lr {c['lr']:.0e} steps {c['steps']:5d} "
              f"dW {c['dW']:6.2f} agree {c['post_train_agree']:.4f}/"
              f"{c['post_val_agree']:.4f}")
    for tag in ("eval_train26000", "eval_report31000"):
        leg = results.get(tag, {})
        if not leg:
            continue
        print(tag)
        for name, r in leg.items():
            extra = (f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f}"
                     if "delta_vs_snap9" in r else "(baseline)")
            print(f"  {name:8s} mean {r['mean']:7.1f} {extra}")


if __name__ == "__main__":
    main()
