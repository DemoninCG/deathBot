"""Probe: fullsplit_seed42 + full 2-ply as a NEW distillation teacher.

User question (2026-09-11): "why did we never test fullsplit_seed42 +
full 2-ply as the teacher?" It survives endlessly like snapshot9's 2-ply
(E1: 20/20 both, paired pieces delta exactly 0 at cap 1500) but may be a
DIFFERENT decision function. Never tested as a teacher: E3 re-ran the
SAME teacher (snap9 2-ply) into the distilled start (info-exhausted
negative); E1 compared OUTCOMES (both cap) but never DECISIONS.

Stage 1 (offline, read-only; no panels, no fits, no gameplay):
  a. teacher-vs-teacher agreement: snap9-2ply argmax (cached labels from
     _cache_teacher_labels.py) vs fullsplit-2ply argmax (computed here,
     champion fast-path convention via teacher_scores_for_state:
     score_batch_fast, cur==nxt both plies, gamma_ply 0.999)
  b. round-2 headroom: student(fullsplit 1-ply) vs fullsplit-2ply argmax
     disagreement rate — compare E3's 19.9% train / 22.4% val vs the
     snap9 teacher
  c. NEW targets: states where student disagrees with fs2ply but AGREED
     with snap9-2ply (the old teacher never pointed there) — the only
     states carrying information beyond the E3 negative
  d. affordability (§49 metric): fullsplit-v1 shortfall of the new
     teacher's pick vs the net's own best viable candidate, on
     disagreement states (snap9-teacher reference: median -2.8 / mean
     -5.3; style teachers -10.5 / -23.5)
  e. teacher margin (fs2ply top1-top2) distribution on disagreement set
  f. sanity: fs2ply-vs-snap9-2ply agreement (are the two teachers even
     different?)

Gate (pre-registered) for a fit: dis >= 5% of states AND new-targets
>= 300 AND affordability median > -8 (vs style teachers' -10.5). If the
gate passes, the fit is "same start, NEW teacher" — the clean complement
to E3 (same teacher, already-distilled start).

Usage:
    python _teacher2_probe.py --out ../_diag_out/_teacher2_probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_teacher2_probe.json"))
    ap.add_argument("--limit", type=int, default=0,
                    help="probe only the first N labelled states")
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}", flush=True)
    results.setdefault("note",
                       "Offline probe: fullsplit-2ply as new teacher; "
                       "read-only; no panels, no fits, no gameplay.")
    results.setdefault("metrics", {})

    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from diag_tinyset_overfit import teacher_scores_for_state
    from afterstate import flat_rows_to_tensor
    import deathbot_sim as ds

    bank = Bank(str(D / "state_bank_snapshot9_32games.npz"))
    n = bank.n_states
    fs_path = D / "fullsplit_seed42.pt"
    assert (hashlib.sha256(fs_path.read_bytes()).hexdigest()[:16]
            == "9989ce86390bcba6")

    fs = TAPNet(output="linear")
    fs.load_state_dict(torch.load(str(fs_path), map_location="cpu"))
    fs.eval()

    # cached snap9-teacher labels
    z = np.load(str(D / "_teacher_labels_snap9_bank.npz"))
    snap9_pick = {int(s): int(p) for s, p in
                  zip(z["state_index"], z["teacher_pick"])}

    placed, nxt = bank._replay_piece_ids()
    n_cand = int(bank.cand_offsets[-1])
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        cand_cur[lo:hi] = piece_to_index(placed[s])
        cand_nxt[lo:hi] = piece_to_index(nxt[s])

    states = sorted(snap9_pick)[: args.limit] if args.limit \
        else sorted(snap9_pick)
    print(f"probing {len(states)} bank states", flush=True)

    # ---- student (fullsplit 1-ply) argmax, full-bank frame ----
    t0 = time.perf_counter()
    student_pick: dict[int, int] = {}
    with torch.inference_mode():
        for i, s in enumerate(states):
            lo, hi = bank.candidate_slice(s)
            sc = fs(cand_boards[lo:hi], cand_cur[lo:hi],
                    cand_nxt[lo:hi]).flatten()
            student_pick[s] = lo + int(sc.argmax())
            if (i + 1) % 8000 == 0:
                print(f"  student {i + 1}/{len(states)}", flush=True)
    print(f"student picks done ({time.perf_counter() - t0:.0f}s)",
          flush=True)

    # ---- fullsplit-2ply teacher labels (viable frame) ----
    t1 = time.perf_counter()
    fs2_pick: dict[int, int] = {}
    fs2_margin: dict[int, float] = {}
    gamma = 0.999
    with torch.inference_mode():
        for i, s in enumerate(states):
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            tv, viable_live = teacher_scores_for_state(
                fs, pre, placed[s], nxt[s], gamma)
            if tv is None:
                continue
            lo, _hi = bank.candidate_slice(s)
            viable = [lo + p for p in viable_live]
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            fs2_pick[s] = viable[order[0]]
            fs2_margin[s] = (float(tv[order[0]]) - float(tv[order[1]])
                             if len(order) > 1 else 0.0)
            if (i + 1) % 2000 == 0:
                print(f"  fs2ply {i + 1}/{len(states)} "
                      f"({time.perf_counter() - t1:.0f}s)", flush=True)
    print(f"fs2ply teacher done ({time.perf_counter() - t1:.0f}s)",
          flush=True)

    both = [s for s in states if s in fs2_pick]

    # ---- headline sets ----
    dis_student_fs2 = [s for s in both if student_pick[s] != fs2_pick[s]]
    new_targets = [s for s in dis_student_fs2
                   if student_pick[s] == snap9_pick[s]]
    retargeted = [s for s in dis_student_fs2
                  if student_pick[s] != snap9_pick[s]
                  and fs2_pick[s] == snap9_pick[s]]
    both_dis = [s for s in dis_student_fs2
                if fs2_pick[s] != snap9_pick[s]]

    # ---- affordability: fullsplit-v1 shortfall of the new pick ----
    def v1_full(net: TAPNet, s: int) -> tuple[list[int], list[float]]:
        lo, hi = bank.candidate_slice(s)
        with torch.inference_mode():
            fast = getattr(net, "score_batch_fast", None)
            if fast is not None:
                v = fast(cand_boards[lo:hi],
                         cand_cur[lo:hi]).flatten().tolist()
            else:
                v = net(cand_boards[lo:hi], cand_cur[lo:hi],
                        cand_nxt[lo:hi]).flatten().tolist()
        return list(range(lo, hi)), v

    def v1_gap(net: TAPNet, s: int, pick: int) -> float:
        idx, v = v1_full(net, s)
        return v[idx.index(pick)] - max(v)

    t2 = time.perf_counter()
    new_set = set(new_targets)
    gaps: list[float] = []
    gap_new: list[float] = []
    for k, s in enumerate(dis_student_fs2):
        g = v1_gap(fs, s, fs2_pick[s])
        gaps.append(g)
        if s in new_set:
            gap_new.append(g)
        if (k + 1) % 500 == 0:
            print(f"  afford {k + 1}/{len(dis_student_fs2)}", flush=True)
    print(f"affordability done ({time.perf_counter() - t2:.0f}s)",
          flush=True)

    def summarise(vals: list[float]) -> dict:
        if not vals:
            return {}
        sv = sorted(vals)
        return {"n": len(vals),
                "mean": round(statistics.fmean(vals), 2),
                "median": round(statistics.median(vals), 2),
                "p90": round(sv[min(len(sv) - 1, int(0.9 * len(sv)))], 2),
                "frac_le_-20": round(
                    sum(v <= -20 for v in vals) / len(vals), 3)}

    metrics = {
        "n_states": len(both),
        "teacher_teacher_agree": round(
            sum(fs2_pick[s] == snap9_pick[s] for s in both) / len(both), 4),
        "dis_student_vs_fs2ply": len(dis_student_fs2),
        "dis_rate": round(len(dis_student_fs2) / len(both), 4),
        "new_targets": len(new_targets),
        "new_target_rate_of_dis": round(
            len(new_targets) / max(1, len(dis_student_fs2)), 3),
        "retargeted_same_as_old": len(retargeted),
        "both_teachers_disagree_with_student": len(both_dis),
        "affordability_all": summarise(gaps),
        "affordability_new_targets": summarise(gap_new),
        "fs2_margin_on_dis": summarise(
            [fs2_margin[s] for s in dis_student_fs2]),
        "fs2_margin_all": summarise(list(fs2_margin.values())),
    }
    results["metrics"] = metrics
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=1), flush=True)

    gate = (metrics["dis_rate"] >= 0.05
            and metrics["new_targets"] >= 300
            and metrics["affordability_all"].get("median", -99) > -8)
    results["gate_pass"] = bool(gate)
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("GATE:", "PASS — fit justified (same start, new teacher)"
          if gate else "FAIL — do not fit", flush=True)


if __name__ == "__main__":
    main()
