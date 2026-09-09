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

- seeds: 20164-20193
  protocol: corrected-lockout, cap 300 (2-ply distillation screen)
  status: CONSUMED
  history: P2T 2026-09-06 (4000 bank states labeled with full-2-ply argmax,
    21.6% disagreement vs 1-ply; 400 CE steps): P2T1 290.1 pcs / 106.7
    lines (26/30 cap), P2T3 292.9 / 109.9 (28/30); paired vs snap9 1-ply
    (294.4 / 109.2, 27/30): dLines −2.5 (t−0.71) / +0.7 (t+0.22) —
    NEUTRAL-to-negative. DECISION: 1-ply CE distillation of 2-ply choices
    does not transfer; do not pursue without a new mechanism
    (files: p2t_distill.json, p2t_T1/T3_seed42.pt).

- seeds: 20194-20223
  protocol: corrected-lockout, cap 300 (style-head Phase B screen)
  status: CONSUMED
  history: Phase B 2026-09-06 (standardized features, survival-weighted
    pairs, calibrated style, veto 25, ONE head swept over alphas):
    a0 283.9/104.8 (26/30, ref) / a0.5 287.2/106.2 / a1.0 294.6/109.3
    (27/30, best) / a2.0 283.5/103.6 (23/30); paired dLines vs a0 all
    |t|<1.1 — head is SURVIVAL-SAFE but style-neutral. DECISION: keep the
    decoupled head + veto as the chassis; style signal must come from a
    well-seeking teacher, not flat-2-ply disagreements
    (files: style_head_B.json, style_head_seed42.pt).

- seeds: 20224-20253
  protocol: corrected-lockout, cap 300 (well-seeking teacher screen)
  status: CONSUMED
  history: well-bonus sweep 2026-09-06 (full 2-ply + survivors-only
    well_setup_score, w 0/5/15/40): lines 117.0/116.9/116.9/117.3,
    tetrises 2/2/2/4, paired dLines −0.1/−0.1/+0.3 (|t|<1.4), cap 30/30
    everywhere; mean_well 0.0001→0.0036 (w40 still ~never holds a well).
    Availability probe: only 36/2000 bank states (1.8%) HAVE any
    well-setup candidate. DECISION: single-move candidate bonuses cannot
    create wells — the policy never generates the precursor states. Next:
    multi-move well scaffolding or training-time representation change;
    see §14.3 (files: well_bonus.json).

- seeds: 20254-20283
  protocol: corrected-lockout, cap 300 (dense-potential λ sweep, §17)
  status: CONSUMED
  history: dense sweep 2026-09-06 (full 2-ply + dense Phi style_fn +
    {1:1,2:3,3:5,4:8} completion bonus, λ 0/2/5/10/20): Tetrises 4/3/4/15/34,
    cap 30/30 everywhere; λ=0 reference 300.0/118.1 (zero-style contract
    holds — full 2-ply champion reproduced). DECISION: dense setup incentive
    creates Tetris play where thresholded bonuses could not; λ=10 candidate
    for validation, λ=20 strongest style (0.38 Tetris/100, share 0.014)
    pending survival-gate confirmation on long games
    (file: dense_sweep.json).

- seeds: 20350-20359
  protocol: corrected-lockout, cap 300 (rollout probe vs champion baseline)
  status: CONSUMED
  history: rollout probe 2026-09-06 (top-3+3 roots, K=4 tapes, D=6,
    controller continuation): 300.0/112.6, 2 Tetrises (0.07/100), 10/10
    vs champion 300.0/116.5, 0 Tetrises, 10/10 — directionally stylish
    but ~10x weaker per compute than dense Phi; continuation (dies ~100pcs)
    is the ceiling. DECISION: rollouts gated on a surviving continuation
    (files: rollout_probe.json, rollout_baseline.json).

- seeds: 20400-20415
  protocol: corrected-lockout, cap 300 (TEACHER DATA — not a selection panel)
  status: CONSUMED
  history: λ=20 teacher bank 2026-09-06 (16 games → 4683 states, clears
    {0:3370, 1:996, 2:239, 3:64, 4:14}; 15/16 capped, one 183-pc death —
    stylish but not death-proof). 2026-09-07 re-collected with piece ids +
    lockouts as teacher_bank_lam20_r1.npz (same 16 seeds, identical
    trajectories/state counts — training-data fix, not a selection panel).
    Used for DAgger round-0 (lowest-Phi flat side, unweighted) and round-1
    (exact 1-ply argmax + survival weighting) head training
    (files: teacher_bank_lam20.npz, teacher_bank_lam20_r1.npz).

- seeds: 20450-20479
  protocol: corrected-lockout, cap 300 (teacher-head screen)
  status: CONSUMED
  history: DAgger round-0 head 2026-09-06 (4520 pairs, 400 steps, veto 25):
    a0 298.8/112.1 (28/30, 6 tet) / a1 291.2/106.7 (24/30, 9 tet) /
    a2 277.2/99.7 (22/30, 13 tet) / a4 174.6/53.6 (2/30, 19 tet) —
    style TRANSFERS to 1-ply but gate fails above a~1. DECISION: round-1
    with piece ids + tighter veto (file: teacher_head.json,
    teacher_head_seed42.pt).

- seeds: 20480-20509
  protocol: corrected-lockout, cap 300 (DAgger round-1 head screen)
  status: CONSUMED
  history: round-1 head 2026-09-07 (r1 bank, 1331 exact pairs, 400 steps,
    margins mean −10.55, 0/1331 ≥ 0, mean weight 0.513; veto×alpha grid):
    a0 289.1/107.4 (26/30, 6 tet) / a0.5 296.6/109.3 (28/30, 10 tet,
    paired dLines +1.9 t+0.81) / a1.0 285.5/103.7 (25/30, 6 tet,
    dLines −3.7 t−0.74); veto 10 ≡ veto 25 everywhere (no candidate
    within 25 z-units was vetoed differently than within 10 — veto
    never binds on this panel). DECISION: gate holds at a≤0.5 but the
    Tetris signal is noise-scale (+4 Tetrises, |t|<1.1); no promotion.
    Next per §17.5.3: long-game gate for the λ=20 2-ply stylish config,
    or a well-direction feature fix (holedepth/wellcost signs) before
    another head round (file: teacher_head_r1.json,
    teacher_head_seed42.pt overwritten by round-1 bundle).

- seeds: 20510-20519
  protocol: corrected-lockout, cap 1500 (§18.1 long-sustain diagnosis)
  status: CONSUMED
  history: baseline/λ10/λ20 windowed runs 2026-09-07 (10 matched tapes,
    200-piece windows: occupancy/height/holes/well/Tetris traces):
    λ0 10/10 cap, 5 tet / λ10 10/10 cap, 26 tet, occ flat 29.7→39.4 /
    λ20 4/10 cap, 6 no_candidates deaths, occ 61.4→114.2 — DECISION:
    λ20 REJECTED as teacher (accumulating backlog), λ10 viable stylish
    candidate (file: long_sustain.json).

- seeds: 20520-20549
  protocol: corrected-lockout, cap 1500 (§20 item 2 death-rejection ablation)
  status: CONSUMED
  history: 1-ply vs 1-ply+death-rejection vs full-2-ply, 30 matched tapes,
    2026-09-07: ply1 763.9 (4/30 cap) / ply1_dr 791.2 (4/30 cap, +27.3
    t+1.23, W/T/L 22/8/0, 34 filter changes, 26 all-dead placements) /
    ply2 1500.0 (30/30 cap, +736.1 t+9.96) — DECISION: continuation-
    valuation win (§20.6: the rule never hurts but explains ~4% of the
    +736 gain; the case for distillation, not the rule)
    (file: death_ablation.json).

- seeds: 20550-20559
  protocol: corrected-lockout, cap 1500 (§20 item 3 λ10 factorial)
  status: CONSUMED
  history: Phi×bonus factorial, 10 matched tapes, 2026-09-07: phi0_b0
    1500.0/596.1 (10/10, 3 tet) / phi0_b8 1500.0/597.0 (10/10, 4 tet,
    dLines +0.9 t+1.17) / phi10_b0 1500.0/595.0 (10/10, 24 tet,
    dLines −1.1 t−0.66) / phi10_b8 1500.0/596.3 (10/10, 22 tet,
    dLines +0.2 t+0.14) — DECISION: Phi is the Tetris ingredient (3–4
    vs 22–24 tetrises); the completion bonus is dead weight on top of
    Phi (phi10_b8 ≈ phi10_b0 on lines AND Tetrises); bonus-only does
    nothing (phi0_b8 ≈ phi0_b0). Per-game occ slopes: phi10_b0 mean
    +4.0 (3/10 games end occ 85–121 with slopes +20/+23 — backlog
    accumulates on SOME tapes even at λ10); phi10_b8 mean −5.9 with
    mid-game occ spikes 86–101 that release (bonus helps release
    inventory WITHOUT adding Tetrises — keep it in the teacher config
    for sustain, not style). Next free: 20560+ (cap-1500); 5000s
    reserves for promotion (file: lambda_factorial.json).

- seeds: 26000-26007
  protocol: corrected-lockout, cap 1500 (TRAINING DATA — not a selection panel)
  status: CONSUMED
  history: §26 student-state aggregation 2026-09-08 (fullsplit_seed42.pt
    1-ply rollouts: 4x1500 cap, deaths 196/503/731 no_candidates + 598
    lockout, mean 1003.5 diagnostic-only; 8028 states teacher-relabeled,
    1801 disagreements, held-out games 2098 / train-agg pool 5930).
    Same status as 20400s teacher bank — never a selection panel
    (file: student_aggregate.json).

- seeds: 20560-20571
  protocol: corrected-lockout, cap 1500 (fullsplit-student fresh screen)
  status: CONSUMED
  history: fullsplit_seed42.pt vs snapshot9 1-ply, 12 fresh tapes,
    2026-09-08: student 1031.3 (7/12 cap) / incumbent 794.6 (1/12 cap),
    paired +236.8, t+1.08, W/L 7/5 — encouraging, n.s. Next free: 20572+;
    5000s reserves for promotion (files: fullsplit_paired_20560.json,
    fullsplit_paired_26000.json [training-seed report only]).
  notes: 26000-26007 sweep (student 1347.2 vs incumbent 670.6, +676.6,
    t+4.71, 8/0) is REPORTING ONLY — training-data seeds, never selection.
    Open: driver (1003.5) vs sweep (1347.2) closed-loop gap on the same
    26000-26007 seeds — bulk-vs-search order audit required before promotion.

- seeds: 5000-5299
  protocol: corrected-lockout, cap 1500 (fullsplit-student PROMOTION)
  status: CONSUMED
  history: fullsplit_seed42.pt vs snapshot9 1-ply, 300 paired games,
    2026-09-08: student 1187.1 (153/300 cap, med 1500) vs incumbent 718.7
    (25/300 cap); paired +468.4, 95% CI [+403, +534], t +14.1, W/T/L
    231/12/57 — PROMOTED (file: fullsplit_promotion_5000.json).

- seeds: 5300-5999
  protocol: corrected-lockout, cap 1500
  status: reserved → future promotion panels (allocate 300 at a time)

- seeds: 20572-20591
  protocol: corrected-lockout, cap 1500 (top-k shortlist gate, §31)
  status: CONSUMED
  history: champion full-2-ply vs FIXED top-k variants (beam5/8 × v1/diverse;
    the 2026-09-08 two_ply.py decision-restriction fix), 20 matched tapes:
    all arms 20/20 cap, min pieces 1500, dLines all |t|<=1.33 (b5_v1 +0.8
    t+1.33 W/L 8/5; b8_diverse +0.5 t+0.83) — top-k SURVIVAL-NEUTRAL after
    the fix. Tetrises: champ 15 / b8_div 17 / b8_v1 13 / b5_v1 9 /
    b5_div 6 (noise at these n). Frozen-bank stage1 (150 states):
    b8_* and b5_diverse agree 1.000, ~1.9-2.4x decision speedup.
    DECISION: beam8_diverse = champion-equivalent at ~2.3x speed, horizon
    caveat per §32.1.2 (file: topk_gate_1500.json; sanity runs on
    26000-26002 + 31000s are reporting-only).

- seeds: 20592-20611
  protocol: corrected-lockout, cap 3000 (λ10 selection-grade estimate, §31.2)
  status: CONSUMED
  history: champ vs λ10 (Phi + completion bonus), 20 matched tapes × 1 cap,
    2026-09-08: champ 20/20 cap / lam10 20/20 cap, ZERO deaths either arm,
    paired pieces +0.0; dT/100 +0.173 t+8.19 (132 vs 28 tetrises, 31 of
    132 in final 3 windows); window occ flat (champ 23-29, lam10 29-43);
    FAIL-flag games champ 8/20 vs lam10 1/20. Combined with the 31200s
    stress pass: lam10 long-horizon failure 1/40 ≈ 2.5%/game (95% upper
    ~11%), champ 0/40 (file: lam10_gate_20592.json). Next free: 20612+
    (§32.3 E1 dev panel); 5300+ reserves for promotion.

- seeds: 20612-20631
  protocol: corrected-lockout, cap 1500 (§32.3 E1 dev)
  status: CONSUMED
  history: E1 weights×search crossover 2026-09-09 (snap9 vs fullsplit ×
    1-ply/b5_div/b8_div/b8_v1/full, 20 matched tapes, cap 1500, eps=0):
    1-ply stud +180.2 t+1.19 (10/20 vs 3/20 cap, n.s.); every 2-ply cell
    20/20 both weights, paired dPieces EXACTLY 0 — search saturates cap
    1500, student adds nothing under 2-ply. DECISION: operating config
    snap9+b8_diverse (2.1x, Tetris-neutral). Cap-3000 leg deferred to
    item 6 (file: weights_search_xover_1500.json). Next free: 20632+
    (dev); 5300+ reserves for promotion.

- seeds: 20632-20651
  protocol: corrected-lockout, cap 1500 (§32.3 E3 dev gate)
  status: CONSUMED
  history: E3 dev gate 2026-09-09 (start vs armA vs armB 1-ply, 20 fresh
    tapes: start 1093.1 9/20 / armA 1083.2 8/20, −9.8 t−0.07 / armB
    1208.5 11/20, +115.5 t+0.80 — neither arm separates at n=20; E3
    CLOSED offline-negative, no fresh-panel final (files: e3_gate.json,
    e3_round.json). Next free: 20652+ (dev); 5300+ reserves for
    promotion.

- seeds: 20652-20751
  protocol: corrected-lockout, cap 1500 (§36.3 armB extension)
  status: CONSUMED
  history: armB-vs-start extension 2026-09-09, 100 fresh tapes
    (start vs armB only — armA dropped after the 20-tape gate went
    exactly neutral): start 1204.0 (59/100) vs armB 1246.6 (60/100),
    paired +42.5, t+0.69, W/L 36/34 — the +115 lean shrank to +42.5,
    n.s. E3 stays CLOSED (file: e3_gate_ext100.json). Next free: 20752+
    (dev); 5300+ reserves for promotion.

- seeds: 20752-20781
  protocol: corrected-lockout, cap 300 (forced-well rung 0, §38)
  status: CONSUMED
  history: rung 0 λ30/40 2026-09-09 (λ 0/10/30/40 × 30 tapes, cap 300):
    T/100 0.044/0.189/0.375/0.544 (tet 4/17/31/41, paired dTet vs λ10
    +0.47 t+2.14 / +0.80 t+2.93); cap 30/30/21/30/15/30 — CLIMBS bin:
    λ NOT saturated, deaths are early well-digging (97-290 pcs), not
    backlog drift. Next: rung-1 commitment controller, not λ60
- seeds: 20782-20811
  protocol: corrected-lockout, cap 300 (forced-well rung 0b, §38.1)
  status: CONSUMED
  history: rung 0b λ50/60 2026-09-09 (λ 10/50/60 × 30 tapes, cap 300,
    λ10 paired reference): T/100 0.200/0.560/0.739 (tet 18/39/43,
    dTet +0.70 t+2.54 / +0.83 t+2.89); cap 30/12/4 — KEEPS CLIMBING,
    §38 knee prediction wrong; deaths early-dig (8/30 by pc 150, two
    zero-Tetris). λ60 as forcing distribution only (file:
    lambda_high2.json). Next free: 20812+ (dev);
    5300+ reserves for promotion.

- seeds: 20812-20841
  protocol: corrected-lockout, cap 300 (forced-well rung 1, §39)
  status: CONSUMED
  history: rung 1 commitment screen 2026-09-09 (commit_above/loose vs
    lam10/lam40 × 30 tapes, cap 300): T/100 0.333/0.522 vs 0.211/0.512
    (tet 30/47 vs 19/38, dTet +0.37 t+1.94 / +0.93 t+4.36); cap
    30/30/30/15 — CONVERTS bin: loose holds λ40-level style at λ10-level
- seeds: 20862-20876
  protocol: corrected-lockout, cap 1500 (rung-2 stage B, §42)
  status: CONSUMED
  history: stage B 2026-09-09 (base_loose vs cool30 vs cool30lam10 vs
    lam10 × 15 tapes, cap 1500, windows + trajectories): T/100
    0.470/0.512/0.506 vs 0.231 (tet 95/104/108 vs 52, dTet +2.87
    t+2.98 / +3.47 t+4.22 / +3.73 t+4.21); cap 13/11/13 vs 15 —
    PROMOTE bin (all commit arms ≥0.45 T/100, all t≥2.9). cool30lam10
    matches cool30 on style at better cap (13/15 vs 11/15). Stage C:
    cool30 vs cool30lam10, 30 tapes + cap-3000 leg
    (file: commit_stageB.json). Next free: 20877+ (dev);
    5300+ reserves for promotion.

- seeds: 20842-20861
  protocol: corrected-lockout, cap 1500 (rung-1 long horizon, §40)
  status: CONSUMED
  history: commit_loose vs lam10 long screen 2026-09-09 (20 matched
    tapes, cap 1500, 200pc windows + per-game trajectories + FAIL flags):
    commit 18/20 (1428.7, 159 tet, 0.556 T/100) vs lam10 20/20 (1500,
    57 tet, 0.190) — SUSTAINS bin: style 2.9× at 18/20 cap, paired
    dTet +5.10 t+6.80 (every tape ≥ +1). 2 deaths are backlog-shaped
    (occ 65→83+, top 9→12+) on tapes lam10 caps — the accumulate
    mechanism, contained to 2/20. Next: cap-3000 confirmation
    (file: commit_long.json). Next free: 20862+ (dev);
    5300+ reserves for promotion.

- seeds: 31200-31219
  protocol: corrected-lockout, cap 1500 + cap-3000 stress (λ10 gate, §31)
  status: CONSUMED
  history: champ vs λ10 (Phi + completion bonus), 20 matched tapes × 2 caps,
    2026-09-08. cap1500: 20/20 vs 20/20, dT/100 +0.140 t+4.47 (50 vs 8 tet),
    end-occ 23.2 vs 40.6. cap3000 stress + per-game backlog flags: see
    lam10_gate_31200.json (REPORTING range 31200+, per §28 rule 1 — a λ10
    promotion decision still needs a fresh selection panel or user sign-off
    that reporting-range + cap-3000 evidence suffices).

## Rules
1. A model may be evaluated on any panel for REPORTING; a decision (keep/drop/
   tune) may only be made on a panel whose range is fresh for that decision.
2. After use for a decision, append the usage to history and mark CONSUMED.
3. Never reuse a CONSUMED range for selecting among new candidates.
4. Record protocol (lockout semantics, cap, seed count) beside every result.
