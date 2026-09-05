# deathBot — Project State & Training Plan Handoff

*Prepared 2026-09-05 after a full diagnostic + patching campaign. Intended as a
complete briefing for a fresh reviewer (human or AI) so no context is lost.*

---

## 1. What this project is

A Tetris bot for **TAP Death mode (20G gravity)** in the Tetris: The Grand Master
series, built as a Rust simulator + Python neural network:

- **Rust crate `deathbot_sim/`** (PyO3 via maturin, installed into the workspace
  venv): board state, TAP-accurate ARS rotation/kicks, IRS, 20G drop, and a BFS
  pathfinder (`Board.search_locks(kind)`) returning every reachable lock
  candidate with its input path.
- **`python/model.py` — TAPNet** (~430K params): conv stack over the [1,20,10]
  board → fused with current-piece + next-piece one-hots (3214 dims) →
  128→1 linear head. Outputs a single "how good is this board" value.
- **`python/selfplay.py`** — Monte Carlo self-play trainer (heavily patched, see §3).
- **`python/heuristic.py`** — hand-crafted survival heuristic used as dense reward
  shaping in the 0.7 survival + 0.3 heuristic blended target.
- **`python/sweep_eval.py`** — checkpoint sweep tool: greedy games on fixed seeds,
  fingerprinting (SHA-256, per-tensor norms, cosine matrix), leaderboard, paired
  per-seed deltas, `--json` results cache.
- **`python/diag_value_probe.py`** — value-function health probe (output scale,
  spread, top-1 agreement with the heuristic, Kendall tau — binned by game
  position).
- **`python/diag_run_experiments.py` + `_capture_worker.py`** — multi-round
  training A/B experiments in isolated subprocesses with per-round checkpoint
  capture (PER_ROUND_CAPTURE hook).
- **`python/evaluate.py`** — original single-checkpoint deterministic evaluator.

Environment: Windows, workspace `.venv` (Python 3.13.3, torch 2.12.1+cpu),
Rust extension installed editable. From `deathbot_sim/`:
`<venv>/Scripts/python.exe -m maturin develop --release` rebuilds it.
Tests: `cargo test` (Rust) and `python -m pytest tests/` (26 selfplay tests +
sim/model/heuristic suites). All green as of this writing.

---

## 2. Model weight files — what each one is

**All checkpoints are the same architecture (430,465 params) and load fine.**
Greedy eval = mean pieces survived over 30 fixed seeds (42–71), 1500-piece cap.

### Keep — the lineage and the champion

| File | Meaning | Eval (mean pieces) |
|---|---|---|
| `model_weights_snapshot7.pt` | **Champion.** Best of the healthy session-1 climb (Jul 8–9). The reference baseline everything is compared against. | **502.9** |
| `model_weights_snapshot1..6.pt` | Session-1 progression (67.6 → 127.5 → 358.1 → 263.6 → 481.2 → 480.7). Historical value only. | see left |
| `model_weights_snapshot8.pt` | **The catastrophic collapse** (2026-07-17). Proof artifact of the squash bug: identical architecture, plays 16× worse. | 31.6 |
| `model_weights.pt` (crate root) | Byte-identical to snapshot7 (sha `36139a2c…`). The "current weights" slot the visualizer loads. | 502.9 |

### Keep — the collapsed-family evidence (crate root)

| File | Meaning | Eval |
|---|---|---|
| `model_weights_test_2.pt` (+ `.normalizer.json`) | One tiny self-play run from snapshot7 that collapsed *partially* (~388) — evidence the collapse mechanism scales with training amount. | 334.9–387.6 |
| `model_weights_afterstate_bootstrap.pt` (+ sidecar) | Collapsed-family artifact from the July 18 recovery attempt (cosine 0.999 to snapshot8, plays like it). | 30.6 |
| `_diagnostic_one_round.pt` (+ sidecar) | Same family; one diagnostic round. | 42.4 |

### Deleted during cleanup (all documented in `_diag_out/` JSONs + git history of this doc)

All post-patch adaptation experiment outputs (adapt7/adapt7b/greedy7/puresurv/
elite7 and their `.best.pt` variants — every one decayed to ~205–215, all
reproducible), the collapsed-family duplicates `model_weights_git_snapshot8.pt`
(= `test_2`) and the empty E-experiment checkpoint directories, plus pytest
scratch files (`_fixcheck*`, `_test_mc_weights*.best.pt`, orphaned normalizer
JSONs) and superseded scripts (`diag_train_probe.py`, `selfplay_refit.py`,
`selfplay_rawnorm.py` — their findings are recorded in §4).

**Remaining `_diag_out/`**: logs + result JSONs only (the evidence trail;
a few MB total, no weights).

### Known identities (verify with `python sweep_eval.py`)

- `model_weights.pt` ≡ `snapshot7` (byte-identical).
- `test_2` ≡ the old `git_snapshot8` file (deleted as duplicate).
- `snapshot8` ≈ `afterstate_bootstrap` (cos 0.999) ≈ `_diagnostic_one_round`
  (0.997) — one collapsed lineage, three files.
- **Weight-space distance is a poor predictor of play strength here**: cosine
  0.999 can hide a 16× performance gap. Always evaluate behavior, never weights.

---

## 3. What was done (the two investigations)

### Phase A — Root cause of the catastrophic regressions (2026-09-04)

Timeline reconstructed from file mtimes + git: snapshots 1–7 climbed healthily
on 2026-07-08/09 (session 1, legacy trainer: raw blended targets ±250,
MSELoss, pre-lock board representation, uniform replay). Between sessions only
benign changes landed (pathfinder dedup fix, max_pieces 500→1500). The 2026-07-17
session-2 run produced `snapshot8` = collapse to ~32 pieces, and *every*
subsequent attempt collapsed the same way. The previous agent's fix attempt
(Jul 18) added a z-score normalizer sidecar system — which made things worse
and also rewrote the heuristic + state representation.

**Root cause (verified by direct experiment):** the current `selfplay.py`
z-score-normalizes training targets into ≈[-3, +1.5], but a warm-started
network's value function lives on a scale of ≈[-300, +35]. Resuming under
normalized targets crushes the value function ~100× in one round.
Measured: snapshot7 output scale 40.5 → 9.9 after one round → 4.8 after two —
and the user's actual `snapshot8.pt` sits exactly on that trajectory
(cosine to the squashed checkpoints 0.997–0.998 vs 0.992 to snapshot7).
A second, slower poison: the current code scores *afterstates* (post-lock,
post-clear) while snapshot7 was trained on *pre-lock* boards.

Ablations (all resumed from snapshot7, 4 rounds × 25 games, greedy-eval after
each round; baseline 502.9):

| Config | Result |
|---|---|
| current code (normalized targets) | 503 → **38** in one round ☠️ |
| current code + refit normalizer per round | 503 → **38** ☠️ (not a stats problem) |
| current code, raw targets | 503 → 344 → 243 → 175 → 160 ⚠️ |
| legacy code, 100 games/round (session-2 faithful) | 260 → 376 → 287 → 201 → **372** → 274 ✓ never collapses |
| current code from scratch (control) | ~77 (matches snapshot1-2 era) |

### Phase B — Patching `selfplay.py` and warm-start adaptation (2026-09-05)

`selfplay.py` now has (all verified, tests updated):

1. **Scale-anchored targets** — the collapse fix. Z-scored reward blend is
   re-anchored onto the network's own output scale (anchor measured from the
   loaded weights on real boards): `target = anchor_mean + anchor_std·z_blend`.
   Resumed networks keep their scale; only rankings are reshaped.
2. **Replay amnesia off by default** (`--replay-max-age 0`).
3. **Eval guardrails** — baseline eval, per-round greedy eval, best weights
   always persisted to `<out>.best.pt` (from the start, so even a run that
   never improves leaves the best-known file), `--rollback-factor` collapse
   restore. Verified: every experiment's `.best.pt` held the strongest weights;
   snapshot7 was never damaged.
4. **`--freeze-conv N`** — freeze the conv feature extractor during
   representation adaptation.
5. **`--elite-fraction F`** — self-imitation: train only on the top-fraction
   of games by survival.
6. **Loud `--weights-in`** — a missing weights file raises `FileNotFoundError`
   instead of silently training from scratch (this silent fallback produced
   mislabeled checkpoints in July). Contract test updated.

Re-verification: the collapse experiment now yields 547 → 342 (was 503 → 38);
all 26 tests pass.

### Phase B results — adaptation runs from snapshot7 (100 games/round)

| Config | Per-round evals | 30-seed final | Cos to snap7 |
|---|---|---|---:|
| defaults (γ=1.0, lr 1e-4, ε=0.1) | 564 → 298 → 172 → … → 165 | 210.2 | 0.861 |
| γ=0.99, lr 5e-5, ε=0.03, 2 epochs | 385/219/392/230/322/265 | 214.5 | 0.964 |
| pure survival target (no heuristic) | 242/199/161 | — | — |
| fully greedy rollouts (ε=0) | 201/315/114 | — | — |
| **elite self-imitation (top 20%)** | **469.8** → 217 → 205 → 211 → 206 → 211 | 205.7 (round-1 ckpt preserved in `.best.pt`) | 0.970 |

**Interpretation.** The catastrophic mode is fixed (503 → ~210 instead of
503 → 30), and nothing is ever lost. But every configuration converges to the
same ~205–215 equilibrium. Isolating variables one at a time ruled out the
heuristic component, exploration bias, target scale, and discount factor as the
primary cause. The actual mechanism:

> **Monte Carlo labels are weaker than the policy they evaluate.** Rollouts of
> a 500-piece policy average 150–340 pieces (exploration accidents, stochastic
> piece queues, representation shift). Every state along those games is labeled
> "survived N steps" with N *below* what the network already achieves. The
> value function regresses toward behavior-policy outcomes. Session 1 worked
> because the network was climbing toward its own (then-adequate) labels from
> scratch; on resume, labels sit below the policy and drag it down.

The elite run is the strongest evidence: its round 1 (trained on games ≥373
pieces) hit 469.8 — the best single round of any run — but the elite threshold
*ratchets down* with weakening rollouts (280 → 229 → 207 → 201), so decay
resumed. The labels chase the policy downward instead of holding the bar.

---

## 4. Suggested next steps (in priority order)

The safety fixes are done; **improving beyond 503 needs the target semantics to
change**, not just guardrails.

1. **Relative/advantage targets.** Replace absolute survival labels with a
   relative signal, e.g. normalize against the round's best game or label
   states by *within-game ranking* vs the policy's current prediction. Weak
   rollouts then carry ranking information instead of dragging absolute values
   down. Highest-leverage single change.
2. **Absolute elite threshold.** Keep only games ≥ some absolute bar (e.g.
   450 pieces, near snapshot7's level), even if that discards 95% of rollouts.
   The relative top-20% filter ratchets down; an absolute bar does not.
3. **Advantage-weighted updates (AWR-style).** Weight each sample by
   `exp((target − net_prediction)/β)`. Outcomes the net already believes are
   better get ~zero weight — directly implements "don't learn from games worse
   than you play."
4. **N-step / TD(λ) bootstrapped targets.** Pure MC ignores the current value
   function, which is what lets labels drag it around. A short-horizon return
   that bootstraps from the network's own value (e.g. 5-step) anchors training
   and should stop the decay entirely.
5. **Representation decision.** All adapted models score *afterstates*, but
   snapshot7 was trained on *pre-lock* boards — the legacy-code runs recovered
   precisely because they used the representation snapshot7 expects. Either
   (a) commit to afterstates and distill snapshot7's rankings on afterstate
   boards first (supervised warm-up, then self-play), or (b) run the next
   campaign on the legacy pre-lock representation with the new guardrails.
   Do not mix them mid-run.
6. **Ops notes.** Always pass absolute `--weights-out` paths (relative paths
   resolve against the crate root, not the cwd — one run wrote outside the
   project); use `sweep_eval.py --json` after every run and keep the JSONs as
   the evidence trail; long runs should be launched so they survive terminal
   pauses; `model_weights_snapshot7.pt` is the single canonical baseline —
   protect it (a git-tracked copy would be prudent).

### A concrete recipe for the next attempt (combining the above)

- Start: fresh copy of `model_weights_snapshot7.pt`.
- Representation: pre-lock (legacy `NeuralScorer` semantics) OR afterstate +
  distillation warm-up — pick one, stay consistent.
- Targets: 5-step bootstrapped return, advantage-weighted (β ≈ 1.0 in
  anchor-scale units), γ = 0.99.
- Data: elite games only, absolute threshold ≥ 400 pieces; top up the buffer
  from previous elite rounds; capacity 400k.
- Guardrails already in place: per-round eval, `.best.pt`, rollback.
- Judge progress on the 30-seed benchmark (`sweep_eval.py --games 30`), never
  on single-round evals (8-game evals swing ±150 pieces).

---

## 5. Benchmark reference (30 seeds, greedy, cap 1500)

| checkpoint | mean | median | sd | max | lines |
|---|---:|---:|---:|---:|---:|
| snapshot7 ≡ model_weights.pt | 502.9 | 430 | 314 | 1316 | 184.6 |
| elite7 round-1 (.best.pt, preserved) | 502.9* | — | — | — | — |
| snapshot6 | 480.7 | 448 | 267 | 1112 | 176.0 |
| snapshot5 | 481.2 | 440 | 326 | 1500 | 176.0 |
| test_2 (partial collapse) | 334.9 | 298 | 187 | 1000 | 117.3 |
| snapshot3 | 358.1 | 344 | 134 | 609 | 126.8 |
| snapshot4 | 263.6 | 209 | 145 | 769 | 89.0 |
| *any* adapted-from-7 endpoint | ~210 | ~170 | ~120 | ~500 | ~68 |
| snapshot2 | 127.5 | 119 | 41 | 243 | 35.9 |
| snapshot1 | 67.6 | 64 | 20 | 139 | 14.7 |
| snapshot8 (collapsed) | 31.6 | 32 | 5 | 42 | 0.2 |
| afterstate_bootstrap | 30.6 | 29 | 6 | 43 | 0.3 |
| _diagnostic_one_round | 42.4 | 42 | 6 | 56 | 2.4 |

\* the elite round-1 checkpoint evals identically to snapshot7 on the full
benchmark because the guardrail preserved snapshot7's weights there; its
8-game training-time eval was 469.8.

**Reproduce any of this with:**
```
cd deathbot_sim/python
<venv>/Scripts/python.exe sweep_eval.py --games 30 --max-pieces 1500 --json <out>.json
<venv>/Scripts/python.exe diag_value_probe.py --weights ..\model_weights_snapshot7.pt
```
NOTE: the default sweep cap is 3000; pass `--max-pieces 1500` to reproduce the
documented benchmark. `diag_run_experiments.py` still references deleted
experiment modules (selfplay_refit.py / selfplay_rawnorm.py) — its historical
E-series results are archived in `_diag_out/` but the driver needs rework
before reuse (see Phase 1 notes below).

---

## 6. Phase 1 implementation record (2026-09-05, this session)

A full plan review produced a phased campaign (correctness → fixed-data
diagnosis → controlled improvement). Phase 1 + the top CPU improvements are
now **implemented and tested** (149 Python tests + 29 Rust tests green).

### What changed, by plan item

**A2 — transition/lockout correctness (`selfplay.py`, `afterstate.py`)**
- New `play_game_detailed()` returns a `GameResult` with explicit
  `topout` / `capped` / `reason` ("no_candidates" | "lockout" | "max_pieces").
- Corrected semantics (new default): candidates whose lock would put a cell
  above the field are masked out of selection; if ALL candidates lose, the
  game ends and the piece is NOT committed. Historical code silently dropped
  above-field cells and kept playing (`Board.lock` documented this contract
  but callers never checked `locks_out`).
- `legacy_lockout=True` reproduces the historical benchmark exactly.
- Reaching `max_pieces` is now distinguishable from death (was labeled
  identically before). Evaluators still count committed placements, so
  benchmark numbers remain comparable under `--legacy-lockout` semantics.
- Anchor std floor (documented but previously unimplemented) is enforced.

**A3 — evaluation/checkpoint guardrails (`selfplay.py`)**
- Baseline eval now uses the SAME `eval_games` seeds as later rounds (was
  `min(eval_games, 3)` — a 3-seed baseline was compared against 8-seed round
  evals, corrupting best-checkpoint selection).
- Rollback works from round 0: a run that only deteriorates now restores the
  starting champion (was gated on `best_round > 0`).
- Rejected candidates are saved to `<out>.rejected_rNN.pt` before rollback.

**A4 — RNG separation (`selfplay.py`)**
- `train_selfplay` now uses separate Random streams: env (pieces, seed+1),
  exploration (seed+2), replay sampling (seed+3). Changing epsilon, top_k,
  or replay settings can no longer perturb the training piece streams.
- New CLI flags: `--env-seed`, `--explore-seed`, `--replay-seed`,
  `--legacy-lockout`, `--rollout-workers`.

**D2 — replay sampling (`selfplay.py`)**
- `ReplayBuffer` caches per-generation index lists (rebuilt only when the
  buffer changes); minibatches are drawn by index. Historical code rescanned
  the whole deque for every minibatch. **Measured 6.8× faster** at 50k
  buffer (597 ms → 88 ms per 100 minibatches of 64).

**D3 — parallel evaluation (`evaluate.py`)**
- New `evaluate_weights_parallel()` shards GAMES across processes (the
  historical evaluator parallelized only across checkpoints, so a single
  checkpoint run used one core). `--workers N` flag added. **Verified
  bit-identical to serial** for the same seeds.

**D4 — parallel rollout collection (`selfplay.py`)**
- `generate_selfplay_dataset_parallel()` plays games across a process pool,
  snapshotting behavior weights once. Per-game seeds are deterministic
  regardless of worker count (**verified: identical sample digest, 1 vs 4
  workers**). **Measured 1.8× with 4 workers** on 32 champion-strength games
  (50 s → 27 s); short/untrained games don't amortize process spawn.
- Enable with `--rollout-workers N` (default 1 = historical serial path).

**D5 — bulk FFI scoring (`src/bulk.rs`, `lib.rs`, `neural_scorer.py`,
`heuristic.py`, `afterstate.py`)**
- Rust `Board.lock_candidates_full(kind)` runs search + lock + clear for
  EVERY candidate in ONE FFI crossing, returning per-candidate
  (state, x, y, locks_out, packed 21-row bitmasks, lines_cleared).
  `Board.from_row_bits()` reconstructs boards from packed rows.
- `NeuralScorer.score_batch` uses this fast path automatically when passed
  candidates straight from `search_locks` (**verified numerically identical**
  to the per-candidate path; ~1.06× on small candidate sets — the NN forward
  dominates, consistent with the earlier project finding).
- `afterstate.rows_to_tensor` / `flat_rows_to_tensor` decode packed rows with
  numpy (vectorized).
- `HeuristicScorer._weighted_reward_batch` computes teacher rewards from
  packed rows vectorized — **verified exact parity** (4599 boards, 0
  mismatches) with per-board `score_afterstate`, including boards with
  invisible-row-20 content (which the 20-row tensor encoding drops; this
  caused a real parity bug during development that packed rows fix).

**A5 — reference state bank (`python/bank_builder.py`)**
- New tool: plays greedy games with a checkpoint and stores every decision
  state (pre-board packed rows + FULL candidate list with lockout flags +
  teacher scores + chosen candidate) as a compressed `.npz`. This is the
  fixed-input dataset Phase 2 update-damage experiments need.
- 4 champion games → 1694 states / 30 655 candidates in 25 s.

### Tests
- `tests/test_phase1.py` (new): lockout masking, topout-vs-truncation,
  bulk-vs-per-candidate scoring parity, packed-row round trips, batched
  heuristic parity (incl. invisible-row fixture), replay index-cache
  correctness, round-zero rollback (rejected candidate preserved), parallel
  eval parity, per-seed rollout determinism.
- Full suite: **149 passed, 2 skipped** (skips: no exploration death in 50
  heuristic seeds; snapshot7 path guard).

### Known remaining gaps (for the next session)
- Checkpoints still save weights only (no optimizer/replay/normalizer/RNG
  state) — full resume is NOT yet implemented (A3 remainder).
- `quick_eval`/`sweep_eval`/`diag_value_probe` still use the corrected
  lockout semantics by default; historical-benchmark reproduction should
  pass `legacy_lockout`-equivalent paths (sweep/evaluate CLI flag not yet
  added — the corrected semantics can change piece counts on near-death
  boards, so re-baseline snapshot7 before comparing).
- `sweep_eval.py` default cap is 3000 (documented benchmark is 1500).
- `diag_run_experiments.py` still references deleted experiment modules.
- `_capture_worker.py` capture hook still saves weights only.

### Recommended next session
1. Re-baseline snapshot7 (30 seeds, cap 1500) under corrected lockout
   semantics — establishes the corrected-v2 reference number.
2. Implement full training-state checkpoints (A3 remainder).
3. Phase 2: fixed-data update controls on the bank
   (`bank_builder.py` output) — the experiments in plan section 3.

---

## 7. Phase 1 remainder + Phase 2 results (2026-09-05, same day)

### Remaining Phase 1 gaps: closed

**Full training-state checkpointing (A3 remainder) — done.**
`save_train_state()` / `load_train_state()` in `selfplay.py` persist
weights + Adam moments + replay contents + the three RNG streams +
normalizer/anchor + epsilon schedule + best tracking + full config,
atomically (tmp + os.replace). `--resume <out>.state.pt` restores everything;
**verified: a 2-round-then-resume-to-4 run reproduces an uninterrupted
4-round run bit-for-bit** (test `TestTrainStateResume`).

**legacy-lockout flags — done.** `evaluate.py` and `sweep_eval.py` both take
`--legacy-lockout`; `sweep_eval.py` default cap is now 1500 (was 3000).

**diag_run_experiments.py — reworked.** Deleted-module experiment defs
removed (E3_refit/E7_raw had referenced `selfplay_refit.py` /
`selfplay_rawnorm.py`); per-run timestamped output files and round
directories (no more stale-capture or overwrite hazards); `--legacy-lockout`
passthrough.

**bank_builder.py — fixed.** The commit step no longer calls
`lock_candidates_full` twice per decision.

### Baselines re-established (30 seeds, cap 1500, snapshot7
sha 36139a2c…)

| protocol | mean | median | sd | file |
|---|---:|---:|---:|---|
| legacy lockout (historical) | **502.9** | 430 | 314.2 | `_diag_out/baseline_legacy_lockout_30seeds.json` |
| corrected lockout (v2) | **503.4** | 430 | 314.9 | `_diag_out/baseline_corrected_30seeds.json` |

The historical 502.9 **reproduces digit-for-digit** through the new parallel
evaluator (bit-identical per-seed to serial). Corrected semantics barely move
snapshot7 (+0.5): it almost never reaches lockout states. Corrected-v2 is the
going-forward reference.

### Reference state bank

`_diag_out/state_bank_snapshot7_32games.npz`: 32 champion games → **16,979
decision states / 312,865 candidates**, every candidate stored with packed
rows + lockout flag + teacher score; chosen action and placed/preview piece
identities recoverable by deterministic queue replay.

### Phase 2 pilot — the update-damage experiments (fixed data)

`python/diag_phase2.py` runs all arms on the frozen bank with paired
minibatch schedules. 400 updates, batch 64, lr 1e-5, SmoothL1 (unless
stated), eval = 8 matched seeds; replication across 3 schedule seeds.
**Ranking drift = candidate ranking vs the incumbent on identical held-out
candidate sets** (tie-aware Kendall tau + top-1 agreement; the first pilot
run used a buggy unbounded tau — discarded, results below use the fixed one).

| arm | what it tests | eval mean (3 seeds) | tau | top1 agree |
|---|---|---:|---:|---:|
| C2_selfdistill | pipeline no-op (train to own outputs) | **532.5** | 1.000 | 1.000 |
| C1_affine | analytic y→10y+5, no training | **532.5** | 1.000 | 1.000 |
| C3_affine_fit | optimize toward affine-transformed outputs | **835–961** | 0.82 | 0.84 |
| T1_raw_mc | raw blended MC targets | **66–68** | 0.63 | 0.43 |
| T2_norm_mc | z-normalized MC (snapshot8 recipe) | 91 (1 pilot seed) | — | 0.51 |
| T3_anchor_mc | anchored MC (current code) | **397–478** | 0.87 | 0.81 |
| L1_mse_raw_mc | raw MC + MSE instead of SmoothL1 | 109 (1 pilot seed) | — | 0.52 |

Incumbent on the same panels: 532.5.

**Interpretation (strongest evidence yet on the regression mechanism):**

1. **The harness is clean.** Self-distillation is a perfect no-op (tau 1.000,
   identical eval); the analytic affine map changes scores by ~10× with zero
   behavioral change. Scale is irrelevant; the pipeline preserves behavior
   when targets do.
2. **Optimization is not the culprit by itself.** Fitting a *consistent*
   rescaled objective (C3) does not collapse behavior — it actually improved
   eval (835–961 vs 532), meaning 400 updates at 1e-5 toward a strictly
   monotone transform of the incumbent's outputs are benign-to-helpful.
3. **The MC labels themselves are the poison.** 400 gentle updates toward
   raw MC survival labels on the champion's OWN visited states drop play to
   ~67 pieces (tau 0.63, top-1 agreement 0.43). This is on FIXED data with
   no fresh rollouts, no distribution shift, no exploration accidents —
   the labels alone, fit conservatively, destroy the policy. The
   "MC labels are weaker than the policy" hypothesis is now directly
   supported: the champion's rollouts average ~530 pieces but per-state
   remaining-survival labels average far less, and regressing onto them
   rewrites candidate rankings.
4. **Anchoring protects a lot but not enough** (397–478, tau 0.87): reshaping
   only the output scale is insufficient — the ranking damage comes from the
   labels' *ordering* across states, which anchoring does not change.
5. **Loss choice is a minor axis** (MSE 109 vs SmoothL1 66 on the same raw
   MC targets — both collapse).

### What this gates

- Phase 2 conclusion: **do not fit absolute MC survival labels on champion
  states at any of the tested budgets.** Any future target must either
  preserve the incumbent's candidate ordering (teacher-ranking regularizer,
  plan C1 branch) or provide a consistent relative signal (paired-rollout
  comparisons, plan C2 branch), before touching absolute returns.
- T2 (z-normalized MC) reproducing the collapse on fixed data retroactively
  validates the 2026-09-04 snapshot8 root-cause diagnosis at the ranking
  level, not just the output-scale level.
- The affine-fit arm (C3) suggests a practical warm-start recipe: a *scaled*
  value target that keeps the incumbent's ordering is safe. This supports
  the plan's calibrated-TD path (C3/C4): bootstrap in a transformed space
  with the transform derived from the incumbent, never from raw returns.

### FOLLOW-UP (same day): C3 validated on fresh panels — it is a real improvement

The surprising C3 result (fitting the incumbent's own outputs stretched to
`10·f(x)+5` for 400 updates at lr 1e-5) was promoted from curiosity to
candidate and validated with paired statistics:

| panel | C3 mean | incumbent mean | paired delta | t | W/L |
|---|---:|---:|---:|---:|---:|
| fresh seeds 1000–1029, corrected lockout (never used before) | **740.4** | 509.1 | **+231.3** | 2.80 | 20/10 |
| historical seeds 42–71, legacy lockout | **736.4** | 502.9 | **+233.5** | 2.82 | 23/7 |

Bootstrap 95% CI on the fresh-panel mean delta: **[+79, +393]**. Under-100
early deaths: 0 for both. Capped games (1500): C3 hit the cap 3× on the
fresh panel (incumbent: 0) — the cap truncates C3's best games, so its true
mean is if anything understated. Files: `_diag_out/c3_fresh30_seed1000.json`,
`_diag_out/incumbent_fresh30_seed1000.json`,
`_diag_out/c3_legacy30_seed42.json`. Checkpoint:
`_diag_out/phase2_arm_C3_affine_fit_seed42.pt`
(sha 229bbe2a…; deterministic — re-running the arm reproduces 897.6 on the
pilot panel exactly).

Two independent 30-seed panels, both ~+230 with t≈2.8, make "lucky 8-seed
fluctuation" implausible. **Interpretation: this is a self-distillation /
born-again-network effect.** Fitting a stretched copy of the incumbent's own
outputs is a consistency objective that smooths the value function on
states the champion actually visits; tau 0.81 means ~19% of candidate pairs
reordered — a genuinely different policy, not a rescaled copy. Mechanism
remains a hypothesis; what is established is the behavioral gain on held-out
seeds.

**Caveats before treating this as the new champion:**
- The historical panel is *in-distribution* for the bank (built from seeds
  42–73); the fresh 1000–1029 panel is the clean validation. Both agree.
- One training seed only so far; per the plan, recipe replication (≥3
  training seeds) and a larger promotion panel (300 fresh tapes) are still
  required before crowning it.
- The affine parameters (10, 5) and 400-step budget were the first tried —
  no hyperparameter search has happened, and selection bias on the pilot
  panel cannot be fully excluded for the *recipe*, only for *this
  checkpoint's* superiority over the incumbent (which the fresh panel
  establishes).
- Next natural steps: replicate C3 across training seeds; sweep steps
  {100, 400, 1600} and scales {3, 10, 30}; then use the C3-distilled net as
  the behavior policy for Phase 3 branches (teacher-ranking regularizer,
  calibrated TD) since it is both stronger and smoother.

### Pilot artifacts

- `_diag_out/phase2_pilot_lr1e-5.json` (7 arms, 1 seed, buggy tau)
- `_diag_out/phase2_replicate_seed{42,43,44}.json` (4 arms × 3 seeds, fixed tau)

### Test status

156 passed, 2 skipped (`tests/`), including new resume-reproducibility and
Kendall-tau tests. Rust suite unchanged (29 pass).