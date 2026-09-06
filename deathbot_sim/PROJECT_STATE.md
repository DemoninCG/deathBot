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

---

## 8. Distillation push round 2 (2026-09-05, later the same day)

User confirmed snapshot9's play visually (viz_live). Goal: push the
self-distillation effect as far as it goes, then hand the best model to
Phase 3. New tooling: `python/diag_distill.py` (distillation arms with
ensemble/scratch/off-policy variants, chunked teacher-target computation,
`--save-arms`); `_diag_out/PANEL_REGISTRY.md` now records every seed range
and its contamination status (screening 2000s, validation 4000s, promotion
reserves 5000s — each used for at most one decision).

New reference bank: `_diag_out/state_bank_snapshot9_32games.npz`
(23,016 states / 400,733 candidates; mean game 719 pieces vs snapshot7's 531).

### Screening + probe round (12-game panel, seeds 2000–2011, paired vs snapshot9 at 794.1)

| arm / variant | mean | delta | t | W/L |
|---|---:|---:|---:|---:|
| D1 iter2 scale10/400 (the C3 recipe again) | 844.2 | +50 | +0.23 | 4/8 |
| D1 scale30/400 seed42 | 931.3 | +137 | +0.62 | 6/6 |
| D1 scale30/400 seed43 | 914.8 | +121 | +0.51 | 6/6 |
| D1 scale30/400 seed44 | 790.5 | −4 | −0.02 | 5/7 |
| D1 scale100 | 870.8 | +77 | +0.44 | 4/8 |
| D1 scale3 | 644.3 | −150 | −1.19 | 4/8 |
| D1 steps100 | 639.7 | −154 | −1.23 | 4/7 |
| D1 steps800 | 706.9 | −87 | −0.48 | 6/6 |
| D1 steps1600 | 408.8 | −385 | −2.67 | 2/10 |
| D1 lr3e-5 | 498.2 | −296 | −2.23 | 4/8 |
| D2 ensemble(5,6,7)→9 scale10 fixed | 632.4 | −162 | −1.13 | 3/9 |
| D2 ensemble scale30 | 618.1 | −176 | −1.56 | 2/10 |
| D3 scratch init + snap9 targets | **46.9** | −747 | −6.46 | 0/12 |
| D4 off-policy inputs scale10 | 733.4 | −61 | −0.36 | 7/5 |
| D4 off-policy scale30 seed42/43/44 | 875/858/877 | +81/+64/+83 | ≈+0.4–0.5 | 24/12 agg |

### Mechanism conclusions (these ARE well-supported)

1. **D3 is the decisive one**: training from a *scratch init* toward
   snapshot9's stretched outputs yields 46.9 pieces — nothing. The gains
   come from the **trajectory** (SGD re-projection of a trained solution),
   not from the target values. This kills any "the targets contain hidden
   information" story and firmly establishes the born-again/self-distillation
   (implicit-smoothing) interpretation.
2. **Off-policy states work almost as well** (D4 ≈ D1): the smoothing
   generalizes beyond the champion's own visited states.
3. **The effect is fragile to dosage**: 100 or 1600 steps, lr 3e-5, or scale
   3 all *hurt*; scale 30 at 400 steps at lr 1e-5 is a local optimum. This
   mirrors the self-distillation literature's "too many iterations collapse"
   finding.
4. **Ensemble teacher hurts** (−162/−176): mixing snapshot5/6/7's weaker,
   noisier value functions pulls the student off snapshot9's smoother
   manifold. Averaging heterogeneous checkpoints is not free signal here.

### Validation (30 fresh seeds 4000–4029, paired)

| model | mean | median | capped |
|---|---:|---:|---:|
| D1 scale30 seed42 | 795.1 | 701 | 4 |
| D4 scale30 seed42 | 708.5 | 568 | 4 |
| snapshot9 | 759.0 | 722 | 3 |

Paired: **D1 vs snap9 +36.1 (t +0.30, CI [−195,+262], W/L 16/14)** —
*not significant*. D4 vs snap9 −50.5 (n.s.). The 12-game screening advantage
did not survive the bigger panel: **no round-2 variant beats snapshot9 with
statistical confidence.**

### Verdict

- Round-2 self-distillation is **not a reliable improvement lever** beyond
  snapshot9: the first jump (7→9) remains the only validated gain. The
  screening-panel means (~880–930) were inflated by a handful of capped
  games; paired per-seed records show real regressions on other seeds.
- **Snapshot9 stays the champion** and is the handoff base for Phase 3.
  (Round-2 artifacts are kept: `_diag_out/distill_D{1,4}_*_seed42.pt`,
  `distill_round2*.json`, `_probe_*.json` — D1-seed42 is nominally the best
  single checkpoint found, 795.1 vs 759.0, but within noise; swapping
  champions on that is not justified.)
- What the round *does* give Phase 3: mechanism knowledge (trajectory
  smoothing, dosage-fragile, off-policy-capable, ensemble-hostile) and a
  proven-safe class of targets (incumbent-order-preserving transforms) that
  the calibrated-TD branch can build on.

### Test status

156 passed, 2 skipped (`tests/`), including new resume-reproducibility and
Kendall-tau tests. Rust suite unchanged (29 pass).

### Pilot artifacts

- `_diag_out/phase2_pilot_lr1e-5.json` (7 arms, 1 seed, buggy tau)
- `_diag_out/phase2_replicate_seed{42,43,44}.json` (4 arms × 3 seeds, fixed tau)

### Test status

156 passed, 2 skipped (`tests/`), including new resume-reproducibility and
Kendall-tau tests. Rust suite unchanged (29 pass).

---

## 9. Phase 3 round 1 (2026-09-05, session 3) — `python/diag_phase3.py`

Goal per plan: controlled policy improvement from snapshot9. Fixed-data arms
on the snapshot9 bank (23,016 states / 400,733 candidates), paired schedules,
detached targets, screened on fresh panel 6000–6011 (snapshot9 ref: 773.9).

### Round 1 (heuristic ranking source; shrink-to-zero TD target)

| arm | eval | paired W/L |
|---|---:|---:|
| P1_rank (heuristic-softmax CE only) | 401.2 | 1/11 |
| P2_rank_mc (T3 MC + heuristic ranking) | 588.9 | 5/7 |
| P3_fitd (bootstrap γ·z(s′)) | 612.1 | 4/7 |
| P4_rank_fitd | 673.8 | 3/9 |

All regress. Two diagnoses:
1. **Heuristic rankings hurt**: snapshot9 already outperforms the teacher
   that generated them, so imitating the teacher's ordering (even softly)
   pulls the policy *back* toward the teacher. P1 confirms: ordering-only
   learning is powerful and destructive here.
2. **γ·z(s′) with no reward term shrinks the value function** (loss stuck at
   ~1500, wrong fixed point).

### Round 2 (both fixes applied)

- Ranking source → the **incumbent's own** candidate ordering.
- Fitted-TD target → incumbent-anchored Bellman image:
  `z(s) → γ·z(s′) + (1−γ)·z(s)` on states with a recorded successor
  (the `(1−γ)·z(s)` term is the placement reward expressed in z units),
  own-value fallback at endgame edge.

| arm | eval | tau | top1 | W/L vs snap9 |
|---|---:|---:|---:|---:|
| P4_rank_fitd (incumbent ranking) | 594.8 | 0.968 | 0.970 | 4/8 |
| P3_fitd (corrected Bellman image) | **792.2** | 0.984 | 0.970 | 6/6 |

- The ranking regularizer still hurts even against the incumbent's own
  ordering (it duplicates the smoothing effect at higher variance) — dropped.
- **P3_fitd is now behavior-neutral with a converged value loss (12.6)** —
  the first Phase 3 update that touches the value function without damage.
  Paired per-seed: high variance (+575 to −652), mean +18 on the 12-game
  panel.
- Cross-check: D4 off-policy distillation on this panel scored 584.7 — the
  distillation arms do NOT transfer across panels, reinforcing that their
  screening means were inflated.

### Validation (30 fresh seeds 7000–7029, paired)

| model | mean | median | capped | under-100 |
|---|---:|---:|---:|---:|
| P3_fitd | 791.3 | 802 | 4 | 1 |
| snapshot9 | 795.2 | 754 | 4 | 0 |

Paired: **delta −3.9, t −0.04, CI [−205, +200], W/L 14/15** — exactly neutral.

### Phase 3 gate status

**PASSED, with no free lunch.** For the first time the project has a
value-learning update (fitted one-step TD toward the incumbent-anchored
Bellman image) that provably does not damage snapshot9 — that was the gate
Phase 2 required before any long training. It is not, by itself, an
improvement lever on frozen data; its value is that it defines a *stable*
update rule the online loop can now iterate with.

**Recommendation for the first online campaign (next session):**
1. Online fitted-TD self-play from snapshot9 using the corrected semantics:
   collect fresh rollouts (env/explore RNGs separated, D4 parallel), fit
   P3-style targets with a lagged target network at a fixed
   update-to-data ratio, per-round quick-eval guardrails already in place.
2. The value targets now have correct units (incumbent-transformed z-space
   with the Bellman image), addressing plan §C4's calibration requirement —
   no absolute survival labels anywhere.
3. Judge on the 32-tape dev panel; promotion only via the registry's fresh
   panels. Expect slow, not catastrophic, dynamics — the round-2 result
   says this update is safe to leave running overnight.

Artifacts: `phase3_round{1,2}.json`, `phase3_P{1..4}_*_seed42.pt`,
`val_P3_7000.json`, `val_snap9_7000.json`, `_xcheck_d4_panel6000.json`.

---

## 10. Online fitted-TD session 1 (2026-09-05, session 4) — `python/train_fitd.py`

First online campaign, deliberately small: 3 rounds × 8 games × 200 Adam
updates (batch 64, lr 1e-5, γ=0.999, ε=0.05, target-net refresh every 2
rounds) from snapshot9. Transitions carry successor boards, so every batch
uses the Bellman-image target from Phase 3 round 2:
`z(s) → γ·z_tgt(s′) + (1−γ)·z_online(s)` with a lagged target net; terminals
(no successor) fall back to `z_online(s)`. New tooling: `train_fitd.py`
(dev-panel freshness assertion, `.best.pt` tracking, per-round JSON).

Dev panel: `_diag_out/dev_panel_fitd1_32.json` (seeds 11000–11031, fresh;
registry updated). Snapshot9 dev baseline: 703.2.

### Per-round trajectory (32-tape dev panel)

| round | transitions | rollout pieces avg | loss | dev mean | Δ vs start |
|---|---:|---:|---:|---:|---:|
| start | — | — | — | 703.2 | — |
| 1 | 2838 | 355 | 11.7 | 602.4 | −100.8 |
| 2 | 2430 | 304 | 10.9 | **824.8** | **+121.6** |
| 3 | 2923 | 365 | 11.6 | 571.7 | −131.6 |

Round-to-round swings (±120) dwarf the trend: with 8 games/round the
*rollout* variance alone (150–838 pieces) dominates. The round-2 "best"
was kept by the guardrail; note the target net refreshed between rounds 2
and 3.

### Final tight-CI screen (100 paired games, seeds 12000–12099, never used before)

| model | mean | median | capped | under-100 |
|---|---:|---:|---:|---:|
| fitd1.best (round 2) | 749.2 | 680 | 11 | 0 |
| snapshot9 | 665.9 | 536 | 9 | 0 |

Paired: **delta +83.3, t +1.36, 95% CI [−34, +202], W/T/L 54/0/46**.
Positive but **not significant** at n=100 (paired SD 615 — one 1500-cap
game swings the mean by 15). No degradation either: under-100 deaths 0/0,
so the stable-update property held online exactly as the Phase 3 gate
predicted.

### What session 1 establishes

1. **The online loop is safe.** Three rounds of live Bellman-image fitting
   produced no collapse (the historical failure mode) — the Phase 3 gate
   transfers from fixed data to the online loop.
2. **A plausible positive signal (+83)** that needs more data or more
   training to confirm. The CI math: at paired-SD ~615, reaching t≥2
   requires either ~4× the games (400) or a larger true effect (more
   rounds of training before testing).
3. **Session-level measurement noise is the bottleneck.** Per-round dev
   means (±120 on 32 tapes) cannot resolve round-level improvements; only
   the final 100+ game screen can.

### Session 2 plan (bigger before longer)

- 8 rounds × 16 games × 400 updates (≈8× session 1's transitions), same
  hyperparameters, then a **400-game** paired final screen (CI ±~60 at
  SD 615 — resolvable if the +83 effect is real).
- If the effect confirms: longer unattended sessions and the 8000s
  promotion panel. If it decays to zero: the Bellman image alone is only a
  stabilizer, and the next lever is *targeted* exploration data (the plan's
  paired-rollout branch) rather than more of the same updates.

Artifacts: `fitd_session1.{pt,pt.best.pt,json}`,
`fitd1_final_100games.json`, `snap9_100games.json`,
`dev_panel_fitd1_32.json`.

---

## 11. Online fitted-TD session 2 (2026-09-05, session 4 cont.) — the verdict

8 rounds × 16 games × 400 updates (~8× session 1's transition volume; 46,632
transitions in replay), same Bellman-image update, target net refreshed every
3 rounds. Dev panel 13000s: snapshot9 baseline 747.0.

Dev trajectory: 784 → 726 → 686 → 658 → 681 → **844.1 (best, round 6)** →
698 → 837. Same pattern as session 1: round-to-round swings (±100) swamp any
trend; the "best" is again a mid-session round that happened to land high.

### Final screen: 400 paired games (seeds 14000–14399, never used before)

| model | mean | median | capped | under-100 |
|---|---:|---:|---:|---:|
| fitd2.best (round 6) | 777.8 | 673 | 56 | 1 |
| snapshot9 | 762.6 | 692 | 38 | 0 |

Paired: **delta +15.1, t +0.51, 95% CI [−42, +73], W/L 194/204, paired SD
590.** Statistically indistinguishable from zero.

### The verdict on fitted-TD self-play (2 sessions, ~30k updates total)

1. **Session 1's +83 (n=100) did not survive.** At n=400 the effect is +15
   ± 58. Combined with session 1's CI [−34, +202], the honest conclusion is:
   **fitted-TD Bellman-image self-play is a *stabilizer*, not an improver** —
   it holds the policy at snapshot9's level (through 8 rounds and ~47k
   transitions of live training with zero degradation, which the July code
   could never do) but does not climb.
2. This matches the Phase 3 fixed-data prediction exactly: the Bellman image
   of the incumbent's own function has no new information to give. Fitting
   the function toward its own bootstrap image converges to… itself. The
   +83 was panel luck; the 400-game number is the truth.
3. **What the stability is still worth:** the loop is now a safe chassis.
   Whatever improvement signal we plug in (better data, better targets) can
   be run overnight without catastrophic risk — the thing that was missing
   all along.

### Where improvement must come from next (evidence-ranked)

The fixed-data campaign showed the only arm that ever *improved* play was
fitting toward a *strictly monotone transform of the incumbent's outputs*
(C3 → snapshot9, +230 validated) — trajectory smoothing, not new information.
TD on the incumbent's own function is also self-referential but *not*
smoothing (it bootstraps rather than regresses to fixed targets). The
untested levers, in order of expected value:

1. **Combine them**: alternate distillation rounds (C3-style, scale 30 /
   400 steps — the proven smoothing lever) with fitted-TD rounds in the
   same online loop. Distillation provides the compressive step, TD the
   consistency check.
2. **Real evaluation signal into the Bellman image**: the current "reward"
   is implicit (the (1−γ)·z(s) term). A measured per-placement shaping term
   (e.g. heuristic delta on the successor, scaled into z units) would give
   the bootstrap something to be right about. Must be validated with the
   Phase 2 fixed-data controls first — any absolute signal is poison until
   proven otherwise.
3. **Targeted exploration data** (plan's paired-rollout branch): the ε=0.05
   rollouts average ~350 pieces vs ~750 greedy — training states are heavily
   weighted toward weak-game situations. A collector that biases toward
   near-death states (where value estimates matter most) would change what
   the same updates can learn.

Artifacts: `fitd_session2.{pt,pt.best.pt,json}`,
`fitd2_final_400games.json`, `snap9_400games.json`,
`dev_panel_fitd2_32.json`.

---

## 12. Standard-levers campaign (2026-09-06) — search, policy-gradient, counterfactuals

The user challenged whether the unusual distillation/TD path was really more
promising than standard methods. Three standard levers were tested head-to-head.

### Experiment 1: 2-ply lookahead (`two_ply.py`, `diag_two_ply.py`) — **REVERSED 2026-09-06: full expansion is a large validated win**

`P2(s,a) = v(afterstate) + γ·max_b v(next-afterstates)`, full expansion over
ALL viable 1-ply candidates (beam=100 ≥ bank max 35/state; the code's
`beam` is a top-k expansion filter, not a search width).

The original screen (beams {1,3,5,10}, panel 17000s) was a **truncation
artifact, not a 2-ply result**: unexpanded candidates keep `final = v1`
while expanded ones get `v1 + γ·max(v2)` with `max(v2)` almost always
negative, so beam < n_candidates systematically prefers *unexpanded bad
moves* (beam=1 provably picks 1-ply rank 2 on a checked state). The
"uncalibrated values" diagnosis was wrong — a bank probe (150 states)
showed v1 and max(v2) on the SAME scale (means −213.3 vs −213.7, corr
0.99, mean |diff| 13.1 vs across-state std 116).

Corrected results (snapshot9 weights, corrected lockout, cap 1500):

| panel | beam-0 (1-ply) | beam-100 (full 2-ply) | paired delta | t | W/L/T |
|---|---|---|---|---|---|
| 3000–3029 ad-hoc pilot | 603.1 (1/30 cap) | **1500.0 (30/30 cap)** | +896.9 | — | 29/0/1* |
| 20000–20029 fresh validation | 994.6 (8/30 cap) | **1500.0 (30/30 cap)** | **+505.4** | **+6.16** | 22/0/8 |

\* pilot made no decision (see registry note); the 20000s panel is the
decision panel. Cap probe (seeds 20100–20103, cap 5000): beam-100 **4/4
capped at 5000** (~1998 lines/game); beam-0 1040.5 on the same seeds.
**The 1500 cap truncates every 2-ply game — true means are understated.**

Cost: ~400 s/game at cap 1500 (~16× 1-ply; was 1.7–6.8 s/game for the
broken truncated beams). No training involved — pure inference-time search.

Also fixed: `play_game_2ply`'s lines counter double-cleared (bulk rows are
already post-clear; `clear_lines()` re-reported 0). Lines now read from
`lock_candidates_full` index 5 (verified: 76 lines/200 pieces on seeds
3000/3001 vs 0 before).

Artifacts: `two_ply_full30.json` (pilot), `two_ply_fresh20000.json`
(validation), `two_ply_cap5000.json` (cap probe).

### Experiment 2: policy-gradient labels (`diag_sil.py`) — **neutral at n=100**

Self-imitation learning (Oh et al. 2018 style, Bellman-shaped advantage in
z-units): `A = [γ·z(s′) + (1−γ)·z(s)] − z(s)`, policy-softmax CE toward
advantage-weighted taken actions; value head untouched — no value regression
anywhere, so the failure mode of Phase 2 cannot occur by construction.

Screen (12 games, panel 18000s, snapshot9 ref 631.5): SIL1 865.9 (+234),
SIL3 856.4 (+225), SIL10 745.2 (+114) — all positive, 8/4 W/L.

**Final (100 paired games, fresh seeds 19000–19099):**

| model | mean | median |
|---|---:|---:|
| SIL1 (T=1) | 739.4 | 620 |
| SIL3 (T=3) | 709.6 | 592 |
| snapshot9 | 707.5 | 581 |

SIL1 vs snap9: **+31.9, t +0.53, CI [−84, +148], W/L 55/44** — the +234
screen signal shrank to noise at n=100. Same pattern as the fitd sessions:
12-game screens are unreliable; 100 games resolves ~±120.

### Experiment 3: paired-rollout counterfactual labels (`diag_paired.py`) — **weak signal, quantified**

100 ambiguous states (incumbent top-2 margin ≤ 2.0; 6,237 such states in the
bank): chosen move vs top-2 rivals, 3 shared future tapes, 80-piece greedy
continuation. Cost 5.4 s/state.

Findings:
- The incumbent's chosen move beats its best rival **84%** of the time —
  snapshot9's 1-ply choices are already good.
- Within-state survival spread: mean 5.2 pieces, **median 0** — most
  ambiguous ties are genuinely equivalent at 80-piece horizon.
- Mean chosen-minus-best-rival: −1.0 pieces; only 4% of states have a rival
  better by >5 pieces, 10% worse by >5.
- The signal exists but is small and rare: ~4 exploitable states per 100
  ambiguous states, at 5 s/state. Scaled to the whole bank (~6.2k ambiguous
  states), that's ~9 hours of labeling for maybe ~250 informative
  comparisons — a slow but genuinely causal signal.

### Campaign verdict (updated 2026-09-06)

| lever | screening promise | confirmed at scale |
|---|---|---|
| **full-expansion 2-ply** | 30/30 capped | **+505, t +6.16 (30 fresh games); 4/4 × 5000-cap** ✅ |
| SIL policy-gradient | +234 | **+32 ± 116 (n.s.)** |
| paired rollouts | small (5.2 ± 5.2) | signal exists, expensive |

The "snapshot9 is at its 1-ply plateau" framing still holds for *1-ply
greedy play* — but it no longer bounds what the system can do, because
**search multiplies the value of the existing network enormously**, exactly
the AlphaZero lesson §12 anticipated under the wrong premise (no separate
calibrated head was needed). The remaining questions are all about search,
not representation:

1. **Adopt full 2-ply as the playing policy** (snapshot9 weights unchanged).
   Next: measure the true ceiling with uncapped/long-cap games, and try
   cheap variants that keep most of the gain: top-k expansion (e.g. beam 5–8
   over FULLY expanded top-k with a de-biased `v1 + γ·(max(v2)−mean(v2))`
   form so partial expansion is valid), and γ_ply sensitivity.
2. **Better representation is deprioritized** — it addressed a 1-ply
   bottleneck that search just bypassed. Revisit only if 3-ply or
   search-aware training stalls.
3. **Training direction changes**: distill/search-aware targets (e.g. distill
   2-ply choices back into the 1-ply net for a fast policy, or fitd-TD on
   2-ply rollouts) now have a teacher worth imitating — unlike the heuristic
   or the 1-ply incumbent. The fitd loop remains the safe chassis for this.
4. **Baselines**: snapshot9 + full 2-ply is the new champion configuration
   (no new weights file — same snapshot9, new inference). Keep SIL1/fitd
   results as 1-ply references.

Artifacts: `two_ply.py`, `diag_two_ply.py`, `two_ply_screen{,_v2}.json`,
`two_ply_full30.json`, `two_ply_fresh20000.json`, `two_ply_cap5000.json`,
`diag_sil.py`, `sil_screen.json`, `sil_T{1,3,10}_seed42.pt`,
`sil{1,3}_final_100.json`, `snap9_19000_100.json`, `diag_paired.py`,
`paired_labels.npz`.