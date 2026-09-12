"""Forensic check (2026-09-11): why did distillation almost never work?

Three read-only measurements on the frozen snapshot9 bank + surviving
checkpoints. NO panels, NO gameplay, NO training.

1. MOVEMENT: did the P2T arms (2-ply-argmax CE at lr 1e-5, 400x64 — the
   §13.3 "distillation is neutral" negative) move toward their own teacher
   at all, vs the adequate-dose fullsplit fit (lr 1e-4, 1200x256)?
2. AFFORDABILITY: on teacher-vs-incumbent disagreement states, how much
   immediate survival value (snap9's own v1) does the SURVIVAL teacher's
   pick give up, compared to the style teachers' recorded v1 gaps
   (teacher-head r1: -10.55; commit-head: -23.5, both mean, 0/N >= 0)?
3. SPREAD: within-state v1 spread per checkpoint (crude smoothing signature).

Teacher here = `teacher_scores_for_state` (fast-path 2-ply, the fullsplit
teacher; P2T's own full-path teacher agrees on ~99.4% of actions per §21).
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"

from model import TAPNet, piece_to_index  # noqa: E402
from diag_phase2 import Bank  # noqa: E402
from afterstate import flat_rows_to_tensor  # noqa: E402
from diag_tinyset_overfit import teacher_scores_for_state  # noqa: E402

DEAD = -1e6
N_STATES_PROBE = 800

NETS = {
    "snap9": ROOT / "model_weights_snapshot9.pt",
    "snap7": ROOT / "model_weights_snapshot7.pt",
    "C3": D / "phase2_arm_C3_affine_fit_seed42.pt",
    "fullsplit": D / "fullsplit_seed42.pt",
    "p2t_T1": D / "p2t_T1_seed42.pt",
    "p2t_T3": D / "p2t_T3_seed42.pt",
    "D1_r2": D / "distill_D1_iter2_scale30_steps400_seed42.pt",
    "e3_armA": D / "e3_round.armA.pt",
    "e3_armB": D / "e3_round.armB.pt",
}


def load_net(p: Path) -> TAPNet:
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(str(p), map_location="cpu"))
    net.eval()
    return net


def main() -> None:
    t0 = time.perf_counter()
    bank = Bank(D / "state_bank_snapshot9_32games.npz")
    n = bank.n_states
    print(f"bank {n} states", flush=True)

    nets = {k: load_net(p) for k, p in NETS.items() if p.exists()}
    print("nets:", list(nets), flush=True)

    placed, nxt = bank._replay_piece_ids()
    n_cand = int(bank.cand_offsets[-1])
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        cand_cur[lo:hi] = piece_to_index(placed[s])
        cand_nxt[lo:hi] = piece_to_index(next_ := nxt[s])

    # Same label-sample construction as diag_p2t_distill.py (seed 42),
    # first N_STATES_PROBE of it -> these ARE states the P2T arms trained on.
    rng = random.Random(42)
    label_states = rng.sample(range(n), min(4000, n))[:N_STATES_PROBE]

    import deathbot_sim as ds

    recs = []
    teacher9 = nets["snap9"]
    for i, s in enumerate(label_states):
        lo, hi = bank.candidate_slice(s)
        if hi - lo < 2:
            continue
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        tv, viable_live = teacher_scores_for_state(
            teacher9, pre, placed[s], nxt[s], 0.999)
        if tv is None:
            continue
        viable = [lo + p for p in viable_live]
        order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
        t_pick = viable[order[0]]
        t_margin = (float(tv[order[0]]) - float(tv[order[1]])
                    if len(order) > 1 else 0.0)
        all_dead = all(float(v) <= DEAD / 2 for v in tv)

        with torch.no_grad():
            sc = {k: net(cand_boards[lo:hi], cand_cur[lo:hi],
                         cand_nxt[lo:hi]).flatten()
                  for k, net in nets.items()}
        inc9 = lo + int(sc["snap9"].argmax())
        pos_of = {b: j for j, b in enumerate(viable)}
        rec = {"s": s, "lo": lo, "t_pick": t_pick, "t_margin": t_margin,
               "all_dead": bool(all_dead),
               "dis": int(inc9 != t_pick)}
        for k, v in sc.items():
            rec[f"pick_{k}"] = lo + int(v.argmax())
            vv = [float(v[b - lo]) for b in viable]
            rec[f"spread_{k}"] = max(vv) - min(vv)
        if rec["dis"] and not all_dead and t_pick in pos_of:
            rec["v1_gap"] = (float(sc["snap9"][t_pick - lo])
                             - float(sc["snap9"][inc9 - lo]))
            q_inc = (float(tv[pos_of[inc9]]) if inc9 in pos_of else DEAD)
            rec["tq_regret_inc"] = float(tv[pos_of[t_pick]]) - q_inc \
                if q_inc > DEAD / 2 else None
        recs.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(label_states)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    live = [r for r in recs if not r["all_dead"]]
    dis = [r for r in live if r["dis"]]
    print(f"\nprobe states {len(recs)} (live {len(live)}, "
          f"dis {len(dis)}, all-dead {len(recs) - len(live)})", flush=True)

    res = {"n_probe": len(recs), "n_live": len(live), "n_dis": len(dis)}
    for k in nets:
        agree = statistics.fmean(1.0 if r[f"pick_{k}"] == r["t_pick"] else 0.0
                                 for r in recs)
        res[f"agree_teacher_{k}"] = round(agree, 4)
        agree9 = statistics.fmean(
            1.0 if r[f"pick_{k}"] == r["pick_snap9"] else 0.0 for r in recs)
        res[f"agree_snap9_{k}"] = round(agree9, 4)
    res["mean_t_margin"] = round(statistics.fmean(
        r["t_margin"] for r in recs), 2)
    gaps = [r["v1_gap"] for r in dis if "v1_gap" in r]
    if gaps:
        res["v1_gap_mean"] = round(statistics.fmean(gaps), 2)
        res["v1_gap_median"] = round(statistics.median(gaps), 2)
        res["v1_gap_frac_below_0"] = round(
            statistics.fmean(1.0 if g < 0 else 0.0 for g in gaps), 3)
        res["v1_gap_frac_below_-5"] = round(
            statistics.fmean(1.0 if g < -5 else 0.0 for g in gaps), 3)
        res["v1_gap_frac_below_-20"] = round(
            statistics.fmean(1.0 if g < -20 else 0.0 for g in gaps), 3)
        res["v1_gap_min"] = round(min(gaps), 2)
    regs = [r["tq_regret_inc"] for r in dis if r.get("tq_regret_inc") is not None]
    if regs:
        res["tq_regret_inc_mean"] = round(statistics.fmean(regs), 2)
    for k in nets:
        sp = [r[f"spread_{k}"] for r in live]
        res[f"spread_mean_{k}"] = round(statistics.fmean(sp), 2)
        res[f"spread_median_{k}"] = round(statistics.median(sp), 2)

    # P2T movement in v1-space distance to snap9 (did the arms move at all?)
    # dW already measured: p2t_T1 0.407 / p2t_T3 0.356 vs fullsplit 4.689.
    print(json.dumps(res, indent=2), flush=True)
    (HERE / "_distill_forensics_out.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote _distill_forensics_out.json "
          f"({time.perf_counter() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
