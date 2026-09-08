"""Decisive tiny-set probe (§24 follow-up): ONE full-batch CE cell, fast.

Same tiny set as diag_tinyset_overfit.py (seed 11, 256 states, identical
pool/label/selection logic) but the fit loop is ONE forward/backward over
all ~4k candidates per step instead of 256 unbatched per-state forwards.
Expected: minutes for 800 steps instead of hours.

Cell: --lr 1e-4, --steps 800, full-batch CE toward teacher argmax.
Reports every 100 steps: loss, agree_all/agree_dis, avoidable-dead count,
param delta norm. Writes JSON at end.

Usage:
    python diag_tinyset_fast.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \\
        --weights ../../model_weights_snapshot9.pt --states 256 --seed 11 \\
        --lr 1e-4 --steps 800 --out ../_diag_out/tinyset_fast.json
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
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    bank = Bank(args.bank)
    n = bank.n_states
    weights_path = resolve_weights(args.weights)
    print(f"Bank: {n} states; tiny-set target {args.states} (seed {args.seed})",
          flush=True)

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

    # ---- identical pool/label/selection as diag_tinyset_overfit.py ----
    pool = rng.sample(range(n), min(n, max(args.states * 4, 1024)))
    recs: list[dict] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for si, s in enumerate(pool):
            lo, hi = bank.candidate_slice(s)
            if hi - lo < 2:
                continue
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            kind, nk = placed_ids[s], next_ids[s]
            tv, viable_live = teacher_scores_for_state(net0, pre, kind, nk,
                                                       args.gamma_ply)
            if tv is None:
                continue
            viable = [lo + p for p in viable_live]
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            t_pick = viable[order[0]]
            sc = net0(cand_boards[lo:hi], cand_cur[lo:hi],
                      cand_nxt[lo:hi]).flatten()
            p1 = lo + int(sc.argmax())
            margin = (float(tv[order[0]]) - float(tv[order[1]])
                      if len(order) > 1 else 0.0)
            p1_dead = float(tv[[b for b in viable].index(p1)]) <= DEAD_PENALTY / 2 \
                if p1 in viable else True
            t_dead = float(tv[order[0]]) <= DEAD_PENALTY / 2
            all_dead = all(float(v) <= DEAD_PENALTY / 2 for v in tv)
            recs.append({"s": s, "lo": lo, "hi": hi, "t_pick": t_pick,
                         "p1": p1, "dis": int(t_pick != p1),
                         "margin": margin,
                         "p1_dead": bool(p1_dead), "t_dead": bool(t_dead),
                         "all_dead": bool(all_dead),
                         "tv": list(tv), "viable": list(viable)})
            if (si + 1) % 500 == 0:
                print(f"  label {si + 1}/{len(pool)} ...", flush=True)
    print(f"labeled {len(recs)}/{len(pool)} in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    dis = [r for r in recs if r["dis"] and not r["all_dead"]]
    avoid = [r for r in dis if r["p1_dead"] and not r["t_dead"]]
    agree = [r for r in recs if not r["dis"]]
    nT = args.states
    n_dis = min(len(dis), int(0.40 * nT))
    n_av = min(len(avoid), int(0.15 * nT))
    n_ag = min(len(agree), nT - n_dis - n_av)
    tiny = (sorted(dis, key=lambda r: -r["margin"])[:n_dis]
            + (rng.sample(avoid, min(n_av, len(avoid))) if avoid else [])
            + (rng.sample(agree, min(n_ag, len(agree))) if agree else []))
    tiny = tiny[:nT]
    print(f"tiny set: {len(tiny)} states ({len([r for r in tiny if r['dis']])} "
          f"dis / {len([r for r in tiny if r['p1_dead'] and not r['t_dead']])} "
          f"avoidable-dead / "
          f"{len([r for r in tiny if not r['dis']])} agree)", flush=True)

    # ---- flat batch: one row per candidate, state boundaries ----
    flat_b: list[torch.Tensor] = []
    flat_c: list[torch.Tensor] = []
    flat_n: list[torch.Tensor] = []
    bounds: list[tuple[int, int]] = []  # (start, end) into flat per state
    tgts: list[int] = []
    off = 0
    for r in tiny:
        s, lo, hi = r["s"], r["lo"], r["hi"]
        m = hi - lo
        flat_b.append(cand_boards[lo:hi])
        flat_c.append(cand_cur[lo:hi])
        flat_n.append(cand_nxt[lo:hi])
        bounds.append((off, off + m))
        tgts.append(int(r["t_pick"]) - lo)
        off += m
    big_b = torch.cat(flat_b, dim=0)
    big_c = torch.cat(flat_c, dim=0)
    big_n = torch.cat(flat_n, dim=0)
    print(f"flat batch: {off} candidates / {len(tiny)} states", flush=True)

    def metrics_from_logits(all_logits: torch.Tensor):
        agree_all = agree_dis = 0
        n_dis_l = 0
        avoid_dead = 0
        with torch.no_grad():
            for i, r in enumerate(tiny):
                a, b = bounds[i]
                lg = all_logits[a:b]
                p_local = int(lg.argmax())
                p = r["lo"] + p_local
                agree_all += int(p == r["t_pick"])
                if r["dis"]:
                    n_dis_l += 1
                    agree_dis += int(p == r["t_pick"])
                tv = r["tv"]
                viable = r["viable"]
                pos_of = {bb: j for j, bb in enumerate(viable)}
                q = float(tv[pos_of[p]]) if p in pos_of else DEAD_PENALTY
                if r["p1_dead"] and not r["t_dead"]:
                    if q <= DEAD_PENALTY / 2:
                        avoid_dead += 1
        return {"agree_all": agree_all / max(1, len(tiny)),
                "agree_dis": agree_dis / max(1, n_dis_l),
                "avoidable_dead": avoid_dead, "n_dis": n_dis_l}

    stu = TAPNet(output="linear")
    stu.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    stu.train()
    opt = torch.optim.Adam(stu.parameters(), lr=args.lr)
    w0 = [p.detach().clone() for p in stu.parameters()]

    t1 = time.perf_counter()
    hist = []
    for step in range(args.steps):
        opt.zero_grad()
        logits = stu(big_b, big_c, big_n).flatten()
        tot = 0.0
        for i in range(len(tiny)):
            a, b = bounds[i]
            tot = tot + F.cross_entropy(logits[a:b].unsqueeze(0),
                                        torch.tensor([tgts[i]]))
        loss = tot / len(tiny)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            with torch.no_grad():
                lg = stu(big_b, big_c, big_n).flatten().detach()
            m = metrics_from_logits(lg)
            dW = float(sum(((a - b) ** 2).sum()
                           for a, b in zip(w0, stu.parameters())).sqrt()
                       .detach())
            print(f"  step {step + 1}/{args.steps}: loss={float(loss.detach()):.3f} "
                  f"agree {m['agree_all']:.3f}/{m['agree_dis']:.3f} "
                  f"avoidable-dead {m['avoidable_dead']} dW {dW:.3f}",
                  flush=True)
            hist.append({"step": step + 1, "loss": round(float(loss.detach()), 4),
                         **{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in m.items()},
                         "dW": round(dW, 4)})
    with torch.no_grad():
        lg = stu(big_b, big_c, big_n).flatten().detach()
    final = metrics_from_logits(lg)
    dW = float(sum(((a - b) ** 2).sum()
                   for a, b in zip(w0, stu.parameters())).sqrt().detach())
    res = {
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(
            weights_path.read_bytes()).hexdigest(),
        "tiny_states": len(tiny), "n_candidates": off,
        "seed": args.seed, "lr": args.lr, "steps": args.steps,
        "loss": "ce-full-batch", "history": hist, "final": final,
        "dW": round(dW, 4),
        "verdict": ("TINY FIT — proceed to §20 item 5 full-split fit"
                    if final["agree_dis"] >= 0.95 else
                    "fits-in-progress — extend budget before item 5"),
        "wall_fit_s": round(time.perf_counter() - t1, 1),
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op}: agree_dis {final['agree_dis']:.3f} dW {dW:.3f}")


if __name__ == "__main__":
    main()
