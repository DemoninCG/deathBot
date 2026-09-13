"""Gen-N teacher probe (generalized from _teacher2_probe.py; 2026-09-12).

Measures whether a candidate teacher (a checkpoint + full 2-ply) carries
extractable information for a student checkpoint, given the labels the
student was LAST distilled toward (old teacher). Read-only; no fits, no
panels, no gameplay.

Default configuration = GEN-3 PROBE:
  student = snapshot11 (gen2_seed44, the champion)
  teacher = snapshot11 + full 2-ply (its own expansion)
  old     = _teacher_labels_fs2ply_bank.npz (gen-2's fs2ply labels)

Metrics (identical semantics to §51.2):
  a. teacher-teacher agreement (new 2-ply teacher vs old labels)
  b. student-vs-new-teacher disagreement (round headroom)
  c. NEW targets: student == old-teacher pick but != new-teacher pick
     (information the previous distillation could not contain)
  d. affordability: student-v1 shortfall of the new teacher's pick
     (references: gen-2 survival teacher median -2.8; fs2ply -1.1;
     style teachers -10.5/-23.5)
  e. teacher margin distribution
Gate (pre-registered, same thresholds as gen-2):
  PASS iff dis >= 5% AND new_targets >= 300 AND afford median > -8.
  PASS justifies ONE distillation burst from this start toward this
  teacher. For the continuous loop, the same gate is re-evaluated per
  burst against the previous burst's labels.

Usage:
    python _teacherN_probe.py --out ../_diag_out/_teacher3_probe.json \
        [--student PATH] [--teacher PATH] [--old-labels NPZ] \
        [--new-labels-out NPZ] [--limit N]
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
    ap.add_argument("--out", default=str(D / "_teacher3_probe.json"))
    ap.add_argument("--student", default=str(ROOT / "model_weights_snapshot11.pt"))
    ap.add_argument("--teacher", default=None,
                    help="default: same file as student (own 2-ply)")
    ap.add_argument("--old-labels", default=str(D / "_teacher_labels_fs2ply_bank.npz"))
    ap.add_argument("--new-labels-out", default=str(D / "_teacher_labels_snap11_bank.npz"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}", flush=True)
    results.setdefault("note",
                       "Gen-N teacher probe; read-only; no panels/fits/gameplay.")
    results.setdefault("metrics", {})

    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from diag_tinyset_overfit import teacher_scores_for_state
    from afterstate import flat_rows_to_tensor
    import deathbot_sim as ds

    bank = Bank(str(D / "state_bank_snapshot9_32games.npz"))
    n = bank.n_states
    student_path = Path(args.student)
    teacher_path = Path(args.teacher) if args.teacher else student_path
    print(f"student/teacher file: {student_path.name} "
          f"(sha {hashlib.sha256(student_path.read_bytes()).hexdigest()[:16]})",
          flush=True)

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(str(student_path), map_location="cpu"))
    net.eval()

    zold = np.load(str(args.old_labels))
    old_pick = {int(s): int(p) for s, p in
                zip(zold["state_index"], zold["teacher_pick"])}
    print(f"old labels: {len(old_pick)} from {Path(args.old_labels).name}",
          flush=True)

    placed, nxt = bank._replay_piece_ids()
    n_cand = int(bank.cand_offsets[-1])
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(bank.n_states):
        lo, hi = bank.candidate_slice(s)
        cand_cur[lo:hi] = piece_to_index(placed[s])
        cand_nxt[lo:hi] = piece_to_index(nxt[s])

    states = sorted(old_pick)[: args.limit] if args.limit else sorted(old_pick)
    print(f"probing {len(states)} bank states", flush=True)

    # student (1-ply) argmax
    t0 = time.perf_counter()
    student_pick: dict[int, int] = {}
    with torch.inference_mode():
        for i, s in enumerate(states):
            lo, hi = bank.candidate_slice(s)
            sc = net(cand_boards[lo:hi], cand_cur[lo:hi],
                     cand_nxt[lo:hi]).flatten()
            student_pick[s] = lo + int(sc.argmax())
            if (i + 1) % 8000 == 0:
                print(f"  student {i + 1}/{len(states)}", flush=True)
    print(f"student picks done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # new teacher (2-ply) labels
    t1 = time.perf_counter()
    new_pick: dict[int, int] = {}
    new_margin: dict[int, float] = {}
    with torch.inference_mode():
        for i, s in enumerate(states):
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            tv, viable_live = teacher_scores_for_state(
                net, pre, placed[s], nxt[s], 0.999)
            if tv is None:
                continue
            lo, _hi = bank.candidate_slice(s)
            viable = [lo + p for p in viable_live]
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            new_pick[s] = viable[order[0]]
            new_margin[s] = (float(tv[order[0]]) - float(tv[order[1]])
                             if len(order) > 1 else 0.0)
            if (i + 1) % 2000 == 0:
                print(f"  new-teacher {i + 1}/{len(states)} "
                      f"({time.perf_counter() - t1:.0f}s)", flush=True)
    print(f"new-teacher labels done ({time.perf_counter() - t1:.0f}s)",
          flush=True)

    if args.new_labels_out and not args.limit:
        zpath = Path(args.new_labels_out)
        if not zpath.exists():
            np.savez(str(zpath),
                     state_index=np.array(sorted(new_pick), dtype=np.int32),
                     teacher_pick=np.array([new_pick[s]
                                            for s in sorted(new_pick)],
                                           dtype=np.int64),
                     meta=json.dumps({
                         "teacher_sha": hashlib.sha256(
                             student_path.read_bytes()).hexdigest()[:16],
                         "gamma_ply": 0.999}))
            print(f"cached new labels -> {zpath.name}", flush=True)

    both = [s for s in states if s in new_pick]
    dis = [s for s in both if student_pick[s] != new_pick[s]]
    new_targets = [s for s in dis if student_pick[s] == old_pick[s]]
    retargeted = [s for s in dis
                  if student_pick[s] != old_pick[s]
                  and new_pick[s] == old_pick[s]]
    both_dis = [s for s in dis if new_pick[s] != old_pick[s]]

    # affordability on the student's own v1 scale
    # 2026-09-12 audit fix: true-preview forward (same policy as the
    # survival harness). score_batch_fast duplicates cur into both slots
    # (the versioned 2-ply approximation) and must NOT be used for 1-ply
    # student affordability.
    def v1_full(s: int) -> tuple[list[int], list[float]]:
        lo, hi = bank.candidate_slice(s)
        with torch.inference_mode():
            v = net(cand_boards[lo:hi], cand_cur[lo:hi],
                      cand_nxt[lo:hi]).flatten().tolist()
        return list(range(lo, hi)), v

    t2 = time.perf_counter()
    new_set = set(new_targets)
    gaps, gap_new = [], []
    for k, s in enumerate(dis):
        idx, v = v1_full(s)
        g = v[idx.index(new_pick[s])] - max(v)
        gaps.append(g)
        if s in new_set:
            gap_new.append(g)
        if (k + 1) % 500 == 0:
            print(f"  afford {k + 1}/{len(dis)}", flush=True)
    print(f"affordability done ({time.perf_counter() - t2:.0f}s)", flush=True)

    def summarise(vals):
        if not vals:
            return {}
        sv = sorted(vals)
        return {"n": len(vals), "mean": round(statistics.fmean(vals), 2),
                "median": round(statistics.median(vals), 2),
                "p90": round(sv[min(len(sv) - 1, int(0.9 * len(sv)))], 2),
                "frac_le_-20": round(
                    sum(v <= -20 for v in vals) / len(vals), 3)}

    metrics = {
        "student_sha": hashlib.sha256(
            student_path.read_bytes()).hexdigest()[:16],
        "n_states": len(both),
        "teacher_teacher_agree_vs_old": round(
            sum(new_pick[s] == old_pick[s] for s in both) / len(both), 4),
        "dis_student_vs_new": len(dis),
        "dis_rate": round(len(dis) / len(both), 4),
        "new_targets": len(new_targets),
        "new_target_rate_of_dis": round(
            len(new_targets) / max(1, len(dis)), 3),
        "retargeted_same_as_old": len(retargeted),
        "both_teachers_disagree_with_student": len(both_dis),
        "affordability_all": summarise(gaps),
        "affordability_new_targets": summarise(gap_new),
        "margin_on_dis": summarise([new_margin[s] for s in dis]),
        "margin_all": summarise(list(new_margin.values())),
    }
    results["metrics"] = metrics
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=1), flush=True)

    gate = (metrics["dis_rate"] >= 0.05
            and metrics["new_targets"] >= 300
            and metrics["affordability_all"].get("median", -99) > -8)
    results["gate_pass"] = bool(gate)
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("GATE:", "PASS — one distillation burst justified"
          if gate else "FAIL — loop stops here (information exhausted)",
          flush=True)


if __name__ == "__main__":
    main()