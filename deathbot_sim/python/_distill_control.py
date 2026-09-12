"""Control experiment (2026-09-11): adequate-dose CE with ZERO-INFORMATION
targets, to pin the mechanism behind the fullsplit (+468) distillation gain.

Arm SHARPEN: CE toward the incumbent's OWN argmax on every bank state.
  Same optimizer/budget/schedule as the successful fullsplit fit
  (lr 1e-4, batch 256, 800 steps, snap9 start, snapshot9 bank) but the
  target carries no information the function doesn't already act on.
  Expected effect under the "re-projection smoothing" hypothesis: margins
  inflate (spread UP), and play does NOT gain (+468 is not reproducible
  from zero-information labels). If it DOES gain, the mechanism is "any
  large-dose re-projection on visited states", not flattening per se.

Batched fit (one concatenated forward per step, segmented CE) — the
fullsplit driver's per-state forwards were ~10x slower with no effect on
the math. NO panels: eval uses reporting-only seeds (26000-26007 training
seeds + 31000-31029 demo-reporting seeds), paired vs snap9 in the same
sweep harness.

Usage:
    python _distill_control.py --arm sharpen --steps 800 --lr 1e-4 \
        --out ../_diag_out/_control_sharpen.json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sharpen", "noise"], required=True)
    ap.add_argument("--bank", default=str(D / "state_bank_snapshot9_32games.npz"))
    ap.add_argument("--weights", default=str(ROOT / "model_weights_snapshot9.pt"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-fast", type=int, default=8)
    ap.add_argument("--eval-report", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    bank = Bank(args.bank)
    n = bank.n_states
    wpath = Path(args.weights)
    print(f"bank {n} states; arm {args.arm}; steps {args.steps} "
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

    # Fit pool: multi-candidate states (single-candidate states have no CE
    # gradient); matches the fullsplit driver's pool rule.
    fit_states = [s for s in range(n)
                  if (bank.candidate_slice(s)[1]
                      - bank.candidate_slice(s)[0]) > 1]
    print(f"fit pool {len(fit_states)} states", flush=True)

    # ---- zero-information targets ----
    t0 = time.perf_counter()
    targets: dict[int, int] = {}
    with torch.no_grad():
        for s in fit_states:
            lo, hi = bank.candidate_slice(s)
            sc = net0(cand_boards[lo:hi], cand_cur[lo:hi],
                      cand_nxt[lo:hi]).flatten()
            targets[s] = lo + int(sc.argmax())
    print(f"own-argmax targets in {time.perf_counter() - t0:.0f}s", flush=True)

    if args.arm == "noise":
        # Fixed random viable index per state, independent of the function.
        rng2 = random.Random(args.seed + 1)
        for s in fit_states:
            lo, hi = bank.candidate_slice(s)
            targets[s] = lo + rng2.randrange(hi - lo)

    # ---- precompute per-step batches (concatenated candidates) ----
    schedule = [rng.sample(fit_states, min(args.batch_size, len(fit_states)))
                for _ in range(args.steps)]
    step_batches = []
    for idx in schedule:
        bl, cl, nl, tl, lens = [], [], [], [], []
        for s in idx:
            lo, hi = bank.candidate_slice(s)
            bl.append(cand_boards[lo:hi])
            cl.append(cand_cur[lo:hi])
            nl.append(cand_nxt[lo:hi])
            tl.append(int(targets[s]) - lo)
            lens.append(hi - lo)
        step_batches.append((torch.cat(bl), torch.cat(cl), torch.cat(nl),
                             torch.tensor(tl, dtype=torch.long), lens))
    print(f"batches ready (avg rows/step "
          f"{statistics.fmean(sum(l) for _, _, _, _, l in step_batches[:50]) / 50:.0f})",
          flush=True)

    # ---- fit ----
    student = TAPNet(output="linear")
    student.load_state_dict(torch.load(str(wpath), map_location="cpu"))
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    w0 = [p.detach().clone() for p in student.parameters()]
    losses: list[float] = []
    t1 = time.perf_counter()
    for step in range(args.steps):
        b, c, nx, tl, lens = step_batches[step]
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
        if (step + 1) % 100 == 0:
            dW = float(sum(((a - b) ** 2).sum()
                           for a, b in zip(w0, student.parameters())).sqrt()
                       .detach())
            print(f"  step {step + 1}/{args.steps}: "
                  f"loss {statistics.fmean(losses[-100:]):.4f} dW {dW:.2f}",
                  flush=True)
    wall_fit = time.perf_counter() - t1
    dW = float(sum(((a - b) ** 2).sum()
                   for a, b in zip(w0, student.parameters())).sqrt().detach())
    print(f"fit done in {wall_fit:.0f}s; final dW {dW:.2f}", flush=True)

    # ---- offline movement probe (same 795-state probe as the forensics) --
    probe = [r["s"] for r in json.loads(
        (HERE / "_distill_forensics_out.json").read_text(encoding="utf-8")
    )] if False else None  # (probe states are re-derived below)

    # derive probe states exactly like the forensics script did
    import random as _r
    _rng = _r.Random(42)
    label_states = _rng.sample(range(n), min(4000, n))[:800]
    stu = student
    stu.eval()
    spread_stu, spread_ref, agree9 = [], [], []
    with torch.no_grad():
        for s in label_states:
            lo, hi = bank.candidate_slice(s)
            if hi - lo < 2:
                continue
            sc_s = stu(cand_boards[lo:hi], cand_cur[lo:hi],
                       cand_nxt[lo:hi]).flatten()
            sc_9 = net0(cand_boards[lo:hi], cand_cur[lo:hi],
                        cand_nxt[lo:hi]).flatten()
            spread_stu.append(float(sc_s.max() - sc_s.min()))
            spread_ref.append(float(sc_9.max() - sc_9.min()))
            agree9.append(int(int(sc_s.argmax()) == int(sc_9.argmax())))
    res = {
        "arm": args.arm, "steps": args.steps, "lr": args.lr,
        "batch_size": args.batch_size, "seed": args.seed,
        "weights_sha": hashlib.sha256(wpath.read_bytes()).hexdigest()[:16],
        "fit_pool": len(fit_states),
        "final_loss": round(statistics.fmean(losses[-50:]), 4),
        "dW": round(dW, 4), "wall_fit_s": round(wall_fit, 1),
        "probe_n": len(agree9),
        "agree_snap9": round(statistics.fmean(agree9), 4),
        "spread_mean": round(statistics.fmean(spread_stu), 2),
        "spread_ref_snap9": round(statistics.fmean(spread_ref), 2),
    }
    print(json.dumps({k: v for k, v in res.items()}, indent=2), flush=True)

    # ---- paired gameplay eval (REPORTING ONLY) ----
    import tempfile
    from evaluate import evaluate_weights_parallel
    tmp = Path(tempfile.gettempdir()) / f"_ctl_{args.arm}_{args.seed}.pt"
    torch.save(stu.state_dict(), str(tmp))

    def run_eval(weights, games, seed0):
        ev = evaluate_weights_parallel(str(weights), games=games, seed=seed0,
                                       max_pieces=1500, workers=8)
        return [g.pieces for g in ev.games]

    base_fast = run_eval(wpath, args.eval_fast, 26000)
    stu_fast = run_eval(tmp, args.eval_fast, 26000)
    base_rep = run_eval(wpath, args.eval_report, 31000)
    stu_rep = run_eval(tmp, args.eval_report, 31000)
    tmp.unlink(missing_ok=True)

    def paired(a, b):
        dl = [x - y for x, y in zip(a, b)]
        m = statistics.fmean(dl)
        sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
        t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
        return {"mean_delta": round(m, 1), "t": round(t, 2),
                "wins": sum(d > 0 for d in dl),
                "losses": sum(d < 0 for d in dl)}

    res["eval_fast_26000"] = {"base": base_fast, "arm": stu_fast,
                              "paired": paired(stu_fast, base_fast)}
    res["eval_report_31000"] = {"base": base_rep, "arm": stu_rep,
                                "paired": paired(stu_rep, base_rep)}
    print("fast(26000s, training seeds, reporting):", res["eval_fast_26000"]["paired"])
    print("report(31000s, demo seeds):", res["eval_report_31000"]["paired"])

    ckpt = D / f"_control_{args.arm}_seed{args.seed}.pt"
    torch.save(stu.state_dict(), str(ckpt))
    res["checkpoint"] = str(ckpt)
    (Path(args.out)).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
