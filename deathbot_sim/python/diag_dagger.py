"""DAgger distillation from a stylish teacher (§15.1.5).

Why DAgger, not frozen-bank CE (§13.3/§14 lesson): fitting the teacher's
picks on states ANOTHER policy visited memorizes picks without the
lookahead that justifies them, and the student then visits new states the
bank never covered. DAgger fixes both: the STUDENT plays, the TEACHER
labels the states the student actually reaches, repeat.

Teacher choices (in order of expected value):
  1. dense-Phi 2-ply winner (style_fn_factory + completion bonus) — only if
     the §17 sweep shows it builds wells from empty boards.
  2. sampled-rollout winner (diag_rollout) — expensive, trigger selectively.
  3. explicit controller action — only on states where the controller is
     competent (seeded-suite build/preserve/cashout states); the controller
     does NOT survive full games (§17.2: dies ~91 pieces), so never use it
     as a full-game teacher.

Student: decoupled head chassis (frozen snapshot9 + zero-init style MLP
over standardized heuristic features, veto gate at search). Training signal
is pairwise preference (teacher pick > student pick) with survival-margin
weighting (Phase-B fix) — no absolute labels, no shared-weight regression.

Loop:
  for round in 1..R:
    student plays N games (cap 300) → visited states
    teacher labels each visited state (2-ply+Phi argmax or rollout winner)
    append (feat_teacher, feat_student) pairs to aggregate dataset
    train head K steps (margin ranking loss, survival-weighted)
    screen on the NEXT fresh panel slice (never reuse)

Usage (after the sweep names a teacher):
    python diag_dagger.py --weights ../../model_weights_snapshot9.pt \\
        --teacher dense_phi --lambda 5 --rounds 3 --games-per-round 10 \\
        --seed0 20400 --out ../_diag_out/dagger.json

Status: SCAFFOLD — teacher gate (§17 sweep) has not passed yet. Do not run
full rounds until a demonstrably stylish teacher exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TEACHERS = ("dense_phi", "rollout", "controller")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--teacher", choices=TEACHERS, default="dense_phi")
    ap.add_argument("--lambda", dest="lam", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games-per-round", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=20400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raise SystemExit(
        "SCAFFOLD: teacher gate has not passed (see §17). "
        f"Requested teacher={args.teacher}. "
        "Run the dense sweep + rollout probe first; this driver runs "
        "only once a teacher builds wells from empty boards.")


if __name__ == "__main__":
    main()
