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

- seeds: 5000-5999
  protocol: corrected-lockout, cap 1500
  status: reserved → future promotion panels (allocate 300 at a time)

## Rules
1. A model may be evaluated on any panel for REPORTING; a decision (keep/drop/
   tune) may only be made on a panel whose range is fresh for that decision.
2. After use for a decision, append the usage to history and mark CONSUMED.
3. Never reuse a CONSUMED range for selecting among new candidates.
4. Record protocol (lockout semantics, cap, seed count) beside every result.
