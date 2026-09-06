# Evaluation panel registry — deathbot training campaign
#
# Panel discipline: each seed range is used for AT MOST ONE selection purpose.
# Once a panel has influenced a decision (which arm/hyperparameter to keep),
# it is dev-contaminated and never used for selection again.
#
# status: reserved | in-use | CONSUMED (contaminated for future selection)

- seeds: 42-71
  protocol: legacy-lockout + corrected
  status: CONSUMED
  history:
    - historical benchmark (2026-09-04 sweep, 502.9)
    - Phase-2 C3 validation (2026-09-05)
  notes: in-distribution for the snapshot7 bank (bank built from seeds 42-73)

- seeds: 9001-9008
  protocol: training-time quick_eval
  status: CONSUMED
  history: default train_selfplay guardrail panel; Phase-2 pilot screening
  notes: training-time eval only; never a selection panel again

- seeds: 1000-1029
  protocol: corrected-lockout
  status: CONSUMED
  history: Phase-2 C3 fresh validation (+231.3, t=2.80)
  notes: first untouched panel after the fix campaign

- seeds: 2000-2011
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: distillation round-2 screening (snapshot9 ref 794.1)

- seeds: 3000-3011
  protocol: corrected-lockout, cap 1500
  status: reserved (unused; screening was small enough to skip replication)
  note 2026-09-06: 3000-3029 used AD-HOC (not a selection panel) for the
    full-expansion 2-ply retest pilot (beam-0 603.1 / beam-100 30/30 capped).
    No decision was made on it; the decision panel is 20000-20029 below.
    Seeds 3000-3029 remain usable for reporting but should not back a
    selection decision without fresh confirmation.

- seeds: 4000-4029
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: distillation finalist validation (D1 795.1 vs snap9 759.0, n.s.)

- seeds: 6000-6011
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: Phase-3 rounds 1+2 screening (snapshot9 ref 773.9)

- seeds: 7000-7029
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: Phase-3 P3_fitd validation (neutral, delta -3.9)

- seeds: 8000-8029
  protocol: corrected-lockout, cap 1500
  status: reserved → next validation panel (online fitted-TD campaign)

- seeds: 11000-11031
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: fitd session-1 dev panel (per-round monitoring)

- seeds: 12000-12099
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: fitd session-1 final screen (100 games; fitd1.best 749.2 vs
           snapshot9 665.9, +83.3, t=1.36, n.s.)

- seeds: 13000-13031
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: fitd session-2 dev panel (per-round monitoring)

- seeds: 14000-14399
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: fitd session-2 final screen (400 games; fitd2.best 777.8 vs
           snapshot9 762.6, +15.1, t=0.51 — neutral)

- seeds: 15000-15031
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: fitd session-2 extra dev monitoring (unused; superseded by
           session-2 final screen)

- seeds: 17000-17011
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: 2-ply beam screen (beam0 852.7; 2-ply collapses)

- seeds: 18000-18011
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: SIL screen (SIL1 866/SIL3 856/SIL10 745 vs snap9 631.5)

- seeds: 19000-19099
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history: SIL finals (SIL1 739.4 vs snap9 707.5, +31.9 n.s.)

- seeds: 20000-20131
  protocol: corrected-lockout, cap 1500
  status: CONSUMED
  history:
    - full-expansion 2-ply validation 2026-09-06 (seeds 20000-20029):
      beam-0 994.6 (8/30 capped) vs beam-100 1500.0 (30/30 capped);
      paired delta +505.4, t +6.16, W/L/T 22/0/8 — DECISION: adopt full
      2-ply as the playing policy (files: two_ply_fresh20000.json).
    - cap probe 2026-09-06 (seeds 20100-20103, cap 5000): beam-100 4/4
      capped at 5000 (~1998 lines); beam-0 1040.5 (files: two_ply_cap5000.json).
  notes: remaining 20104-20131 still fresh for future finals.

- seeds: 20104-20133
  protocol: corrected-lockout, cap 300 (style baseline)
  status: CONSUMED
  history: style baseline 2026-09-06 (snapshot9, cap 300): beam-0 295.2 pcs /
    110.3 lines (28/30 cap) vs beam-100 300.0 / 117.8 (30/30 cap); clears
    beam-0 {0:5958, 1:2546, 2:301, 3:47, 4:5}, beam-100 {0:5848, 1:2798,
    2:328, 3:25, 4:1} — DECISION: both policies are near-zero-Tetris
    flat-stackers; style work (Tetris-bias search term) approved
    (file: two_ply_style30.json).

- seeds: 20134-20163
  protocol: corrected-lockout, cap 300 (Tetris-bias k-sweep)
  status: CONSUMED
  history: k-sweep 2026-09-06 (full 2-ply + guideline-shaped line_bonus,
    survivors only): base 117.7 lines / 0 tetrises; k5 117.1 / 0; k15
    117.4 / 0; k40 117.6 / 2 — paired dLines all ~0 (|t|<2.1), cap 30/30
    everywhere. DECISION: static per-clear bonus does NOT create Tetris
    play; the value function outvotes it on ~every state. Next: setup
    bonus (well-shaping term) rather than larger k
    (file: two_ply_tetris_k.json).

- seeds: 20104-20133
  protocol: corrected-lockout, cap 300 (style baseline)
  status: CONSUMED
  history: style baseline 2026-09-06 (snapshot9, cap 300): beam-0 295.2 pcs /
    110.3 lines (28/30 cap) vs beam-100 300.0 / 117.8 (30/30 cap); clears
    beam-0 {0:5958, 1:2546, 2:301, 3:47, 4:5}, beam-100 {0:5848, 1:2798,
    2:328, 3:25, 4:1} — DECISION: both policies are near-zero-Tetris
    flat-stackers; style work (Tetris-bias search term) approved
    (file: two_ply_style30.json).

- seeds: 5000-5999
  protocol: corrected-lockout, cap 1500
  status: reserved → future promotion panels (allocate 300 at a time)

## Rules
1. A model may be evaluated on any panel for REPORTING; a decision (keep/drop/
   tune) may only be made on a panel whose range is fresh for that decision.
2. After use for a decision, append the usage to history and mark CONSUMED.
3. Never reuse a CONSUMED range for selecting among new candidates.
4. Record protocol (lockout semantics, cap, seed count) beside every result.
