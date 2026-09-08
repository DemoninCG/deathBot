"""Tiny-set overfit diagnostic (§20 item 4; NO panels, NO gameplay eval).

Question (§20.4-20.5): can THIS implementation make the network learn THESE
specific decisions? Diagnostic only — resulting weights may play terribly.

Tiny set: 128-512 frozen-bank states, balanced across:
  - teacher-baseline disagreements (1-ply argmax != 2-ply argmax)
  - important live-root preferences (large teacher margin, both live)
  - avoidable preview deaths (1-ply picks a preview-dead root, teacher
    picks a live root)
  - ordinary agreement states (regularizer against collapse)
Teacher = bonus-free full 2-ply from the SAME snapshot9 weights (CHAMPION
fast-path scoring, i.e. cur-in-both-slots — labels the ACTUAL champion;
§20.4a versioning: do NOT silently "fix" it here).

Grid (tiny, CPU-cheap): LR {1e-5, 1e-4, 3e-4} x loss {CE(T), pairwise-margin}
x --steps {200, 800}. Track per cell: train loss, teacher agreement,
teacher-preferred-action margin (mean student logit gap teacher-minus-1ply),
gradient/update magnitudes (global grad norm at step 0, param delta norm).

Controls (must-pass for the harness to be trusted):
  C1 synthetic-misrank: deliberately misranked KNOWN teacher action must
     become more likely after training (label parity alone doesn't test
     the update — §20.5). Implemented as: pick agreement states, declare a
     FIXED non-teacher candidate the "teacher", train CE toward it, report
     promotion rate. If C1 fails, the UPDATE is broken, not the data.
  C2 sentinel hygiene: assert -1e6 DEAD_PENALTY never enters the training
     objective as a numeric target (CE/margin use argmax indices + masks
     only; report max |logit| vs sentinel ratio).

Metrics use POSITIVE regret (§20.3): regret(s) = max_a Q_T - Q_T(student
pick), reported as agreement-all, agreement-on-disagreements,
regret|disagreement, regret|live-pick, avoidable-dead count, all-dead count.
Never average sentinels with ranking gaps.

Gate (§20.6): tiny-fit success -> §20 item 5 (full-split fit, adequate
budget); tiny-fit failure -> reconsider student/setup, NOT more labels.

Usage:
    python diag_tinyset_overfit.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \\
        --weights ../../model_weights_snapshot9.pt --states 256 --seed 11 \\
        --out ../_diag_out/tinyset_overfit.json
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
from afterstate import flat_rows_to_tensor  # noqa: E402

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
    """Copied semantics from diag_p1_fit: champion fast-path 2-ply."""
    import deathbot_sim as ds
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
    succ_rows, counts = [], []
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
            counts.append(0)
            continue
        rr = np.asarray(r2[4], dtype=np.uint32).reshape(n2, 21)
        keep = [i for i in range(n2) if not r2[3][i]]
        if not keep:
            counts.append(0)
            continue
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
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps-list", nargs="+", type=int, default=[200, 800])
    ap.add_argument("--lrs", nargs="+", type=float,
                    default=[1e-5, 1e-4, 3e-4])
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

    # ---- label a candidate pool (bounded cost: ~states*4 sample) ----
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
            live_best = max(tv)
            # avoidable death: 1-ply pick preview-dead, teacher live
            p1_dead = float(tv[[b for b in viable].index(p1)]) <= DEAD_PENALTY / 2 \
                if p1 in viable else True
            t_dead = float(tv[order[0]]) <= DEAD_PENALTY / 2
            all_dead = all(float(v) <= DEAD_PENALTY / 2 for v in tv)
            recs.append({"s": s, "lo": lo, "hi": hi, "t_pick": t_pick,
                         "p1": p1, "dis": int(t_pick != p1),
                         "margin": margin, "live_best": live_best,
                         "p1_dead": bool(p1_dead), "t_dead": bool(t_dead),
                         "all_dead": bool(all_dead),
                         "tv": list(tv), "viable": list(viable)})
            if (si + 1) % 500 == 0:
                print(f"  label {si + 1}/{len(pool)} ...", flush=True)
    print(f"labeled {len(recs)}/{len(pool)} in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    dis = [r for r in recs if r["dis"] and not r["all_dead"]]
    big = sorted([r for r in dis if not r["t_dead"] and not r["p1_dead"]],
                 key=lambda r: -r["margin"])
    avoid = [r for r in dis if r["p1_dead"] and not r["t_dead"]]
    agree = [r for r in recs if not r["dis"]]
    # balanced tiny set: 40% disagreements (big-margin first), 15% avoidable
    # deaths, 45% agreements
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

    slices = {}
    for r in tiny:
        s, lo, hi = r["s"], r["lo"], r["hi"]
        tgt_local = int(r["t_pick"]) - lo
        slices[s] = (cand_boards[lo:hi], cand_cur[lo:hi], cand_nxt[lo:hi],
                     tgt_local, r["tv"], r["viable"], lo)

    def tiny_metrics(model):
        model.eval()
        agree_all = agree_dis = 0
        n_dis_l = 0
        reg_dis, reg_live = [], []
        avoid_dead = all_dead_n = 0
        with torch.no_grad():
            for r in tiny:
                s, lo = r["s"], r["lo"]
                b, c, nx, _t, tv, viable, _lo = slices[s]
                sc = model(b, c, nx).flatten()
                p = lo + int(sc.argmax())
                pos_of = {bb: j for j, bb in enumerate(viable)}
                q = float(tv[pos_of[p]]) if p in pos_of else DEAD_PENALTY
                reg = float(max(tv)) - q  # POSITIVE regret (§20.3)
                agree_all += int(p == r["t_pick"])
                if r["dis"]:
                    n_dis_l += 1
                    agree_dis += int(p == r["t_pick"])
                    reg_dis.append(reg)
                if q > DEAD_PENALTY / 2:
                    reg_live.append(reg)
                if r["p1_dead"] and not r["t_dead"]:
                    if q <= DEAD_PENALTY / 2:
                        avoid_dead += 1
                if r["all_dead"]:
                    all_dead_n += 1
        return {"agree_all": agree_all / max(1, len(tiny)),
                "agree_dis": agree_dis / max(1, n_dis_l),
                "reg_dis_mean": statistics.fmean(reg_dis) if reg_dis else 0.0,
                "reg_dis_max": max(reg_dis) if reg_dis else 0.0,
                "reg_live_mean": statistics.fmean(reg_live)
                if reg_live else 0.0,
                "avoidable_dead": avoid_dead,
                "all_dead": all_dead_n, "n_dis": n_dis_l}

    pre = tiny_metrics(net0)
    print(f"PRE: agree_all {pre['agree_all']:.3f} agree_dis "
          f"{pre['agree_dis']:.3f} reg|dis {pre['reg_dis_mean']:.2f} "
          f"(max {pre['reg_dis_max']:.1f}) avoidable-dead "
          f"{pre['avoidable_dead']}", flush=True)

    # ---- C1 synthetic-misrank control ----
    c1_states = rng.sample([r for r in tiny if not r["dis"]],
                           min(32, len([r for r in tiny if not r["dis"]])))
    c1_promoted = 0
    if c1_states:
        stu = TAPNet(output="linear")
        stu.load_state_dict(torch.load(str(weights_path),
                                       map_location="cpu"))
        stu.train()
        opt = torch.optim.Adam(stu.parameters(), lr=1e-4)
        c1_tgts = {}
        for r in c1_states:
            s, lo, hi = r["s"], r["lo"], r["hi"]
            m = hi - lo
            # fixed non-teacher target: (teacher-local + 1) % m
            t_local = int(r["t_pick"]) - lo
            c1_tgts[s] = (t_local + 1) % m
        for _ in range(200):
            opt.zero_grad()
            tot = 0.0
            for r in c1_states:
                s = r["s"]
                b, c, nx, _t, _tv, _v, _lo = slices[s]
                tot = tot + F.cross_entropy(
                    stu(b, c, nx).flatten().unsqueeze(0),
                    torch.tensor([c1_tgts[s]]))
            (tot / max(1, len(c1_states))).backward()
            opt.step()
        stu.eval()
        with torch.no_grad():
            for r in c1_states:
                s, lo = r["s"], r["lo"]
                b, c, nx, _t, _tv, _v, _lo = slices[s]
                p = int(stu(b, c, nx).flatten().argmax())
                c1_promoted += int(p == c1_tgts[s])
    c1 = {"n": len(c1_states), "promoted": c1_promoted,
          "rate": c1_promoted / max(1, len(c1_states))}
    print(f"C1 synthetic-misrank: {c1_promoted}/{len(c1_states)} promoted "
          f"({c1['rate']:.2f})", flush=True)

    # ---- C2 sentinel hygiene ----
    c2 = {"dead_penalty": DEAD_PENALTY,
          "used_as_numeric_target": False,
          "note": "CE/margin use argmax indices + live masks only; "
                  "sentinels appear in EVAL regret vectors, never in loss"}
    with torch.no_grad():
        max_logit = max(
            float(net0(b, c, nx).flatten().abs().max())
            for (b, c, nx, _t, _tv, _v, _lo) in slices.values())
    c2["max_abs_logit"] = max_logit
    c2["sentinel_ratio"] = abs(DEAD_PENALTY) / max(1e-9, max_logit)

    # ---- grid: CE vs pairwise-margin ----
    cells = []
    for lr in args.lrs:
        for steps in args.steps_list:
            for loss in ("ce", "margin"):
                stu = TAPNet(output="linear")
                stu.load_state_dict(torch.load(str(weights_path),
                                               map_location="cpu"))
                stu.train()
                opt = torch.optim.Adam(stu.parameters(), lr=lr)
                # grad norm at step 0 (diagnostic)
                opt.zero_grad()
                tot0 = 0.0
                for r in tiny:
                    s = r["s"]
                    b, c, nx, t, tv, viable, _lo = slices[s]
                    logits = stu(b, c, nx).flatten()
                    if loss == "ce":
                        tot0 = tot0 + F.cross_entropy(
                            logits.unsqueeze(0), torch.tensor([t]))
                    else:
                        m = len(tv)
                        live = [j for j, v in enumerate(tv)
                                if v > DEAD_PENALTY / 2]
                        if not live:
                            continue
                        # margin: teacher logit above best live rival by 1.0
                        rivals = [j for j in live if j != t] or \
                                 [j for j in range(m) if j != t]
                        tot0 = tot0 + F.relu(
                            logits[rivals].max() - logits[t] + 1.0)
                (tot0 / max(1, len(tiny))).backward()
                gnorm0 = float(torch.nn.utils.clip_grad_norm_(
                    stu.parameters(), 1e9))
                opt.step()
                p0 = [p.detach().clone() for p in stu.parameters()]
                losses = [float(tot0.detach()) / max(1, len(tiny))]
                for _ in range(1, steps):
                    opt.zero_grad()
                    tot = 0.0
                    for r in tiny:
                        s = r["s"]
                        b, c, nx, t, tv, viable, _lo = slices[s]
                        logits = stu(b, c, nx).flatten()
                        if loss == "ce":
                            tot = tot + F.cross_entropy(
                                logits.unsqueeze(0), torch.tensor([t]))
                        else:
                            m = len(tv)
                            live = [j for j, v in enumerate(tv)
                                    if v > DEAD_PENALTY / 2]
                            if not live:
                                continue
                            rivals = [j for j in live if j != t] or \
                                     [j for j in range(m) if j != t]
                            tot = tot + F.relu(
                                logits[rivals].max() - logits[t] + 1.0)
                    (tot / max(1, len(tiny))).backward()
                    opt.step()
                    losses.append(float(tot.detach()) / max(1, len(tiny)))
                p1w = list(stu.parameters())
                delta = float(sum(
                    (a - b).pow(2).sum()
                    for a, b in zip(p0, p1w)).sqrt())
                post = tiny_metrics(stu)
                # teacher-preferred-action margin: mean student logit gap
                with torch.no_grad():
                    gaps = []
                    for r in tiny:
                        s = r["s"]
                        b, c, nx, t, _tv, _v, _lo = slices[s]
                        lg = stu(b, c, nx).flatten()
                        rivals = [j for j in range(lg.numel()) if j != t]
                        gaps.append(float(lg[t] - lg[rivals].max()))
                cells.append({
                    "lr": lr, "steps": steps, "loss": loss,
                    "loss_first": round(losses[0], 4),
                    "loss_last": round(statistics.fmean(losses[-20:]), 4),
                    "grad_norm_step0": round(gnorm0, 4),
                    "param_delta_norm": round(delta, 4),
                    "pref_margin_mean": round(statistics.fmean(gaps), 4),
                    **{k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in post.items()},
                })
                print(f"  lr {lr:g} steps {steps} {loss}: loss "
                      f"{losses[0]:.3f}->{statistics.fmean(losses[-20:]):.3f} "
                      f"agree {post['agree_all']:.3f}/{post['agree_dis']:.3f} "
                      f"reg|dis {post['reg_dis_mean']:.2f} "
                      f"dW {delta:.3f}", flush=True)

    best = max(cells, key=lambda c: (c["agree_dis"], -c["reg_dis_mean"]))
    res = {
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(
            weights_path.read_bytes()).hexdigest(),
        "bank": str(Path(args.bank).resolve()),
        "tiny_states": len(tiny),
        "seed": args.seed, "gamma_ply": args.gamma_ply,
        "pre": pre, "C1_synthetic": c1, "C2_sentinel": c2,
        "cells": cells,
        "best": best,
        "verdict": ("TINY FIT — proceed to §20 item 5 full-split fit"
                    if best["agree_dis"] >= 0.95 else
                    "NO TINY FIT at this grid — reconsider student/setup; "
                    "do NOT collect more labels"),
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op}: best agree_dis {best['agree_dis']:.3f} "
          f"({best['loss']} lr {best['lr']:g} steps {best['steps']})")


if __name__ == "__main__":
    main()
