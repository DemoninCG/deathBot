# deathBot — Project State & Training Plan Handoff

*Prepared 2026-09-05 after a full diagnostic + patching campaign; updated
2026-09-06 (evening) for the survival-solved → style era. Intended as a
complete briefing for a fresh reviewer (human or AI) so no context is lost.*

**Where things stand (2026-09-06 evening):** survival is solved by
*inference*, not weights — **snapshot9 + full-expansion 2-ply caps every
game** (30/30 at cap 1500, 4/4 at cap 5000). The active program is *style*:
the bot survives but plays flat (near-zero Tetrises), and the goal is
Tetris-seeking play that is impressive to watch and useful for human
training. Eval protocol has shifted to **cap 300** (every game caps; style
metrics carry the signal). See §12 (search reversal) and the new §13
(style era).

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
  Survival-era tool; not the current lever (see §13).
- **`python/two_ply.py`** — **the current playing policy.** Full-expansion
  2-ply lookahead (`TwoPlyScorer`, `play_game_2ply`, style-metric
  `play_game_2ply_style` + `style_summary`, optional survivors-only
  `line_bonus`). Validated §12; style-bias extension §13.
- **`python/diag_two_ply.py`** — 2-ply screen driver (default **cap 300**,
  per-game clears breakdowns, lines/100 + Tetris-share reporting).
- **`python/diag_tetris_bias.py`** — Tetris-bias k-sweep driver (§13).
- **`python/diag_p2t_distill.py`** — 2-ply→1-ply preference distillation (§13).
- **`python/viz_live.py`** — live pygame visualizer; supports `--two-ply`
  (full-expansion 2-ply) and `--line-bonus "1,3,5,8"` (z-unit per-clear
  bonuses), per-candidate `+n`/`TETRIS` tags, live `l/100` stat.
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
| `model_weights_snapshot9.pt` (workspace root, sha `229bbe2a…`) | **Champion weights.** C3 self-distillation of snapshot7 (§7 follow-up, +231 × 2 panels). The weights half of the current champion *configuration*. | **~740 fresh-panel** (1-ply); **caps everything** (2-ply, §12) |
| `model_weights_snapshot7.pt` | Best of the healthy session-1 climb (Jul 8–9). Superseded by snapshot9 as weights; kept as the collapse-investigation reference baseline. | **502.9** |
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
a few MB total) **plus a small number of style-era arm checkpoints**
(`p2t_T1/T3_seed42.pt` — kept because they are the only record of the
distillation negative; everything else is JSON).

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

## 4. Suggested next steps (in priority order) — SUPERSEDED 2026-09-06

This section is kept for history. Its program (beat 503 via better 1-ply
*targets*) was overtaken twice: first by snapshot9's self-distillation jump
(§7 follow-up), then decisively by full-expansion 2-ply search (§12), which
solved survival with no training at all. The active program is *style*
(Tetris-seeking play under a survival gate) — see §13 for the current plan
and next steps. Do not launch §4-recipe training runs; the 1-ply target
family it describes (MC/elite/AWR/TD on absolute survival) is documented as
neutral-or-harmful in §§7–11.

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

---

## 13. Style era (2026-09-06 evening) — survival solved, now play *well*

**Trigger:** live visualization (`viz_live.py --two-ply`) confirmed
snapshot9 + full 2-ply "plays practically forever." Goal restated by the
user: reward *good* play (Tetrises most, singles least) so the bot is
impressive to watch and useful for human training. Survival is now a
*gate* (cap rate must not drop), not the objective.

**Protocol change:** evals run at **cap 300** (every survival-capable game
caps; style metrics carry the signal). Core metrics per game: clears
breakdown {0,1,2,3,4}, lines/100, Tetris share of scoring moves, cap rate.
`two_ply.py` gained `play_game_2ply_style()` + `style_summary()`;
`TwoPlyScorer` gained a survivors-only `line_bonus` (dying/lockout
penalties untouched, so survival ordering is preserved by construction);
`diag_two_ply.py` defaults to cap 300 with clears reporting; `viz_live.py`
gained `--line-bonus "1,3,5,8"`, per-candidate `+n`/`TETRIS` tags, live
`l/100`.

### 13.1 Style baseline — both policies are flat-stackers

snapshot9, seeds 20104–20133 (fresh, CONSUMED), cap 300
(`two_ply_style30.json`):

| | beam-0 (1-ply) | beam-100 (full 2-ply) |
|---|---|---|
| pieces / lines | 295.2 / 110.3 (28/30 cap) | 300.0 / 117.8 (30/30 cap) |
| clears {0/1/2/3/4} | 5958/2546/301/47/**5** | 5848/2798/328/25/**1** |
| Tetris share | 0.17% | 0.03% |

6 Tetrises in ~18k placements combined. 2-ply's extra lines come from more
singles/doubles, not bigger clears. There is no latent Tetris preference to
amplify — a style bias must *create* the behavior. (First run of this screen
had empty `per_seed_clears` for beam-0 — that path used
`selfplay.play_game`, lines-only. Fixed so both beams use
`play_game_2ply_style`; numbers above are the fixed re-run.)

### 13.2 Tetris-bias k-sweep — clean negative

Guideline-shaped bonus on survivors only, `{1:k/8, 2:3k/8, 3:5k/8, 4:k}`
for k = 5/15/40 (sub-margin → dominant vs ~10.5 within-state spread).
Seeds 20134–20163 (fresh, CONSUMED), cap 300 (`two_ply_tetris_k.json`,
driver `diag_tetris_bias.py`):

| arm | lines | Tetrises | paired Δlines | cap |
|---|---|---|---|---|
| base | 117.7 | 0 | — | 30/30 |
| k5 | 117.1 | 0 | −0.6, t −2.01 | 30/30 |
| k15 | 117.4 | 0 | −0.2, t −0.88 | 30/30 |
| k40 | 117.6 | 2 | −0.0, t −0.10 | 30/30 |

The bonus flips only 6–14/9000 moves by ±1–6 lines. The value function
outvotes even 4×-dominant k on nearly every state. **Static per-clear
bonuses don't create Tetris play.** Survival gate passes everywhere (the
survivors-only construction works). Next: a *setup* (well-shaping) term,
not larger k — Tetrises need a prepared well several pieces out, which a
one-move clear bonus can't see.

### 13.3 2-ply→1-ply distillation — neutral-to-negative

`diag_p2t_distill.py`: 4000 bank states labeled with full-2-ply argmax
(geometry-verified; 864/4000 = 21.6% disagreement vs 1-ply — real signal),
then 400 preference-only CE steps (T=1/3, no value regression possible).
Screened 1-ply, seeds 20164–20193 (fresh, CONSUMED), cap 300
(`p2t_distill.json`, `p2t_T1/T3_seed42.pt` kept as the record of the
negative):

| arm | pieces | lines | paired Δlines vs snap9 1-ply (294.4/109.2) | cap |
|---|---|---|---|---|
| P2T1 | 290.1 | 106.7 | −2.5, t −0.71 | 26/30 |
| P2T3 | 292.9 | 109.9 | +0.7, t +0.22 | 28/30 |

Fitting the teacher's picks on frozen states doesn't transfer to play —
same signature as SIL/fitd. Likely cause: the disagreements sit exactly
where 1-ply valuation is wrong, and matching argmax there without the
lookahead that justifies it memorizes picks, not reasons. (First run
crashed at the screen stage — closure passed to ProcessPoolExecutor isn't
picklable on Windows; fixed with a module-level worker.)

### 13.4 Viz for human training — done

`--line-bonus "1,3,5,8"` flag (z-unit per-clear bonuses through the viz
scorer), live `l/100`, gold `TETRIS` tags in the testing panel. Smoke-tested;
suite green. Example:
`viz_live.py --nn --weights ../../model_weights_snapshot9.pt --two-ply --line-bonus 1,3,5,8`

### Planned next steps (evidence order)

1. **Well-shaping setup bonus (search-time).** Bonus for holding a deep,
   clean well (depth 8–9, zero holes) on surviving branches — rewards the
   Tetris *setup state* the value function avoids, instead of the clear
   itself. Screen at cap 300 on fresh 20194+ with the survival gate (30/30
   cap) + Tetris-share signal. If search shows Tetrises, *then* distill that
   behavior — distilling flat-stacking 2-ply (§13.3) was distilling the
   wrong teacher.
2. **Training-target change on top of (1).** Decoupled style head (see
   §14): frozen survival net + trainable style module combined at search
   only. Absolute survival labels stay banned (§§7–11).
3. **Search speed.** Full 2-ply is ~16× 1-ply cost — worth paying while it
   buys survival + style, but the bill comes due for long screens: top-k
   with a de-biased `v1 + γ·(max(v2)−mean(v2))` form, γ_ply sensitivity.
   Only after (1) gives something worth speeding up.
4. **Panel discipline continues:** one decision per fresh panel; next free
   range is **20194+** (cap-300 style) / 5000s reserves for promotion.
   Standing champion configuration: **snapshot9 weights + full 2-ply**
   (same `.pt`, new inference); P2T/k-bonus arms are negatives, kept as
   JSONs (+ 2 P2T checkpoints as the distillation record).

---

## 14. Perf pass + decoupled style head (2026-09-06 night)

External perf agent (commit `3599071`): ~7% 2-ply speedup. Profiling result
stands — `conv2d` + `max_pool` ≈ 90% of `choose`; nothing outside the conv
moves the needle. Kept wins: batched ply-2 forward, `lock_candidates_fast`
/ `from_row_bits_bulk` FFI, `score_batch_fast` + one-hot buffers,
vectorized heuristic/train packing, segmented P2T CE, `bench_speed.py`.
Rust suite 26+3 green; Python 156 passed / 2 skipped after merge.

Implication for style: 2-ply cost is FLOPs-bound. Further speedups need a
smaller net or fewer forwards (top-k, §13 step 3) — not more Python
micro-opts.

Perf follow-up (same night): the fast path returns a **5-tuple** (no lines
list) vs the full path's 6-tuple, and `play_game_2ply` /
`play_game_2ply_style` briefly read `res[5]` off whichever path was bound —
`IndexError` on any game played through `lock_candidates_fast`. Fixed by
pinning both line-reporting loops to the FULL `lock_candidates_full` path
(cost negligible next to `choose()`); `TwoPlyScorer.choose` keeps the fast
path (it never reads lines except via `_placements_from_bulk`). New
`tests/test_bulk_contract.py` pins the contract (300-piece games report
lines>0, clears sum to pieces and lines). Full suite now **165 passed, 2
skipped**.

Training-target consequence (user direction): P2T shared-weight
distillation was neutral-to-negative (§13.3), so style training moved to a
**decoupled head** — `python/style_head.py`: frozen snapshot9 survival
value + small trainable style MLP over heuristic features, combined only
at search (`v + α·style`, survivors only, dying/lockout penalties
untouched).

### 14.1 Style-head Phase A — collapse (the concern was justified)

864/4000 disagreement pairs, loss 1.17→0.91, screened at α=15 on fresh
20194–20223 (cap 300): **24.1 pcs / 0.0 lines, 0/30 cap**
(`style_head_A.json`). Diagnosis (measured, not guessed): the head learned
mean +86 / sd 41 over candidates with within-state spread 11.4 vs survival
spread ~10.5 — α·style spanned ~+39/+2877 against survival margins ~11, so
style *dominated* ranking. Worse, the teacher direction isn't well-building
at all: mean(feat₂ₚₗᵧ−feat₁ₚₗᵧ) = wheight +1.94, holedepth +1.02,
lines −0.12, deepwell −0.01 — imitating flat-stacking 2-ply disagreements
as static preference is lethal, exactly as suspected.

### 14.2 Foundational fixes + Phase B — safe but style-neutral

Hardened the chassis (all in `style_head.py` / `diag_style_head.py`, tested):
- **Feature standardization** (`fit_feature_stats`/`apply_stats` — raw spans
  O(100)→O(1), lines std was 0.09 vs wheight 52).
- **Survival-weighted pairs** (`exp(margin/10)`, capped at 1): all 864
  teacher picks sit *below* their 1-ply rival in survival value (mean −5.53,
  0/864 ≥ 0, mean weight 0.66) — training now discounts the harmful ones.
- **Per-state calibration** (`calibrated()`: minus state-mean — only
  within-state ranking reaches the combination) and a **survival veto gate**
  (`combined_scores(..., veto_margin=25)`: >25 below best-v is −inf before
  style applies). α=0 reproduces the survival policy by construction.
- Driver now supports `--alphas` sweeps over one trained head (no
  retraining) and persists feat stats inside the checkpoint
  (`style_head_seed42.pt` is now a dict bundle).

Phase B (same 864 pairs, 400 steps, veto 25, panel 20194–20223 reused for
the sweep — one head, four alphas, no reselection across panels):

| α | pieces | lines | cap | Tetrises | paired Δlines vs α=0 |
|---|---|---|---|---|---|
| 0.0 (ref) | 283.9 | 104.8 | 26/30 | 8 | — |
| 0.5 | 287.2 | 106.2 | 26/30 | 4 | +1.4, t +0.27 |
| 1.0 | 294.6 | 109.3 | 27/30 | 2 | +4.5, t +0.95 |
| 2.0 | 283.5 | 103.6 | 23/30 | 5 | −1.2, t −0.20 |

(`style_head_B.json`.) The head is **survival-safe but style-neutral**
(all |t|<1.1): the veto + calibration fixed the collapse without finding
any Tetris signal — because the teacher still has ~0 Tetrises in it.
Distilling flat-2-ply disagreement, however safely, cannot teach wells.

### What §14 gates (updated plan)

1. **Well-seeking teacher first (§14.3 below — now CLOSED as a negative).**
   The decoupled head + veto is a safe chassis (7 head tests + 2 contract
   tests green), but the well bonus produced no well-building teacher (see
   §14.3). Do not train the head against flat-2-ply disagreement again.
2. **Next: the foundational-representation branch (§14.4).** Single-move
   search bonuses cannot create wells (availability 1.8%); training
   against 2-ply choices cannot transfer (§13.3, §14.1–14.2). The remaining
   lever with a mechanism is changing what the value function can *see*
   (well/depth input channels or a separate value head) — §14.4 is the
   plan.
3. **Search speed after style pays** (top-k de-biased, γ sensitivity).
   Standing champion unchanged: **snapshot9 + full 2-ply**; Phase-A
   checkpoint superseded by the Phase-B bundle (`style_head_seed42.pt`).

### 14.3 Well-seeking teacher — clean negative, mechanism found

`well_setup_score()` (`two_ply.py`: one deep ≥6 clean well, no holes, no
second well >3, height-scaled; unit-tested in `tests/test_well_bonus.py`)
wired as a survivors-only `well_weight` bonus in `TwoPlyScorer.choose`
(`play_game_2ply_style(..., well_weight=w)` accumulates mean well score;
driver `diag_well_bonus.py`). Screened w = 0/5/15/40 on fresh 20224–20253
(cap 300, `well_bonus.json`):

| w | lines | Tetrises | paired Δlines | cap | mean well |
|---|---|---|---|---|---|
| 0 | 117.0 | 2 | — | 30/30 | 0.0001 |
| 5 | 116.9 | 2 | −0.1, t −1.00 | 30/30 | 0.0003 |
| 15 | 116.9 | 2 | −0.1, t −0.63 | 30/30 | 0.0007 |
| 40 | 117.3 | 4 | +0.3, t +1.39 | 30/30 | 0.0036 |

Even w=40 (4×-dominant) leaves mean well at 0.0036 — the bonus is never
*available*, not merely outvoted. Availability probe (2000 bank states,
every candidate scored): only **36/2000 states (1.8%) have ANY well-setup
candidate** (mean max score 0.0089). The flat-stacking policy never
generates precursor states, so no single-move candidate bonus — clear or
well — can select what doesn't exist. This closes the search-bonus branch:
Tetris play needs states the current policy never visits, which requires
either multi-move scaffolding (reward well-depth *trajectories*) or a
training-time change so the value function itself generates them.

### 14.4 Next: foundational representation (SUPERSEDED 2026-09-06 — see §15)

~~All three cheaper branches are now closed with mechanism: clear bonus
(§13.2), pick distillation (§13.3, §14.1–14.2), well bonus (§14.3). The
common cause is representational, not incentive~~ — this diagnosis was
rejected on independent review (see `AI_plan_response.txt` + §15). The
§14.3 result closes only *thresholded* single-move bonuses (a ≥6-deep
all-or-nothing detector rewarded from flat-stack states, availability
1.8%), not incentives in general. Height/well/hole channels are derivable
from occupancy — they add inductive bias, not motive — and distilling a
flat teacher was never a test of Tetris style. §14.4's order (representation
first, scaffolding as fallback) is therefore reversed by §15.

---

## 15. Revised style plan (2026-09-06, independent-review reorder)

Independent review (`AI_plan_response.txt`) accepted: snapshot9 + full
2-ply stays the champion configuration, but **representation is NOT the
next lever**. The missing ingredient is a signal that values the *path*
to a Tetris. Prior work rewarded completed/near-completed setups while
collecting data from a policy that avoids building them. Fix incentives
and scaffolding first; touch representation only when fitting/
generalization diagnostics demand it.

### 15.0 Safety-contract corrections (do first)

1. **"Survivors-only" is not a survival guarantee.** Masking immediate
   deaths / 2-ply deaths does not protect against positions that die 10
   pieces later. Likewise: freezing the survival net protects its weights,
   not the combined policy; the veto gate is a heuristic, not a certified
   constraint; mean-subtraction does not bound style influence within a
   candidate set (pairwise gaps unchanged). Treat all of these as
   engineering guardrails, and validate style policies on long games +
   difficult tapes, not just cap-300 screens.
2. **Zero-style contract (broken — fix before any more style screens).**
   `style_head_B` α=0 scored 26/30 cap while the champion full-2-ply
   scores 30/30 on the same protocol: the style harness runs **1-ply +
   style**, not the champion policy, so α=0 does not reproduce the
   champion. Contract going forward: **style weight zero must reproduce
   the champion's chosen actions, terminal handling, and game results
   exactly** (action agreement on a fixed bank + identical cap rate).
   Includes wiring the style combination through `TwoPlyScorer` (not a
   separate 1-ply worker) and a regression test pinning it.
3. **Survival is demonstrated, not proven.** 30/30 caps at 300 (or even
   4/4 at 5000) still permit ~9.5%/game failure at a one-sided 95% bound
   on the short horizon. Keep cap-300 for cheap screening, but promote
   only on long games + hard sequences.

### 15.1 Order of work (evidence order, revised)

1. **Zero-style harness fix (§15.0.2).** Style combination lives inside
   `TwoPlyScorer.choose` (optional `style_fn` + `style_alpha` +
   `veto_margin`); `None`/α=0 is byte-identical to today's champion path.
   Bank action-agreement test + cap-rate check. No style decisions until
   this gate passes.
2. **Seeded well diagnostics** (`diag_seeded_wells.py`): clean 1-deep
   precursor, 2–3-row partials, Tetris-ready well, ready well with
   delayed I, nearly-blocked well, high-stack must-abandon. Report per
   class: build / preserve / cash-out / recover. If the scorer takes
   seeded Tetrises but never builds from empty → horizon/incentive
   problem. If it rejects a reachable Tetris → inspect scoring/
   transition logic first. These states double as a later curriculum.
3. **Designated-well dense potential + explicit controller.**
   Fixed well column (not argmax-depth — prevents well-switching reward
   hacking). Dense features from actually-visited states: rows filled
   except well, partial outside-well fill, clean well depth with credit
   below 4, well obstructions, holes/covered holes outside well, surface
   roughness outside well, reachable I-clear opportunity, stack-height /
   placement-diversity risk. Form
   `S(a) = Q_base,2ply(s,a) + λ·Φ(s_a) + β·R_clear(s,a)` with Φ a
   *potential* (saturate past a usable 4-line setup; never unbounded
   depth) and an explicit Tetris-completion reward (setup-only objectives
   punish cashing out). Small finite-state controller
   BUILD → CASH-OUT → BUILD with RECOVER available; ARS/20G reachability,
   not heights alone; sides not assumed interchangeable. Success here
   proves representation is not the prerequisite.
4. **Sampled multi-step rollouts with a style-capable continuation.**
   Root candidates × sampled future tapes (preview-correct: only the true
   preview is known; beyond it sample the TAP randomizer), continuation
   = well controller / dense-style policy (NOT the flat champion, which
   dismantles setups and reports zero value). Common random numbers per
   comparison. Trigger expensive planning selectively (well start/abandon,
   competing plans, drought, risk threshold) rather than every placement.
5. **Train/distill only from a demonstrably stylish teacher**
   (controller actions, rollout winners, human demos, Tetris-reward
   returns, trajectory preferences). DAgger (label states the student
   visits), not frozen-bank CE. Decoupled head stays the chassis
   (frozen survival + zero-init style branch); old absolute-survival
   labels stay banned. A genuine style critic target is
   `V_style(s) ← r_style + γ·V_style,tgt(s′)`, optionally with
   potential shaping `r′ = r + η[γΦ(s′)−Φ(s)]` (standard discount/
   terminal conditions for the invariance claim) — the old
   `z ← γz′ + (1−γ)z` update is not a task reward and must not be
   relabeled as one.
6. **Representation last, with a clean gate.** Only if good style targets
   exist that the net cannot fit/generalize. If tried: copy weights,
   zero-init new-channel conv weights (exact function preservation) —
   not C3-affine warm-up (which changes rankings). Same for any residual
   style branch.
7. **Search hygiene.** Drop the `v1 + γ(max−mean)` partial-expansion
   form (best-sibling advantage ≠ absolute continuation quality);
   expand shortlisted roots fully, keep shortlists diverse enough to
   retain setup moves, cache duplicate leaves. Optimize small heuristic
   weight vectors (5–15 dims) with CEM/random search under a survival
   constraint on paired tapes before training 430K params.

### 15.2 Metrics (lines/100 is conserved — stop optimizing it)

During stable survival, `L = (4P + B_start − B_end)/10 → ~40/100`
regardless of singles vs Tetrises. Report instead: Tetrises/100,
fraction of cleared lines in Tetrises, Tetris share of clear events,
setup→Tetris conversion, time-between-Tetrises, height/recovery stats,
long-horizon failure rate. Keep lines/100 as a health check only. For
human training: plan legibility + input-sequence executability.

Panels: next free range is **20254+** (cap-300 style); champion
   unchanged (**snapshot9 + full 2-ply**).

---

## 16. §15 implementation — zero-style contract + dense potential (2026-09-06 night)

### 16.1 Zero-style contract — FIXED in `TwoPlyScorer`

`TwoPlyScorer` takes optional `style_fn(packed_rows, lines) -> [float]`,
`style_alpha`, `veto_margin`. Style applies to SURVIVING 2-ply branches
only (dead branches stay at `DEAD_PENALTY`; lockouts already excluded);
`style_fn=None`/α=0 skips it without calling `style_fn` (byte-identical
champion path). The beam≤0 1-ply path honors the same contract.
`play_game_2ply_style` forwards the three params. New
`tests/test_style_contract.py`: α=0 action-agreement vs the champion on
120 bank states (style_fn call-count 0), dead-branch non-resurrection,
veto-0 == champion pick. **5/5 pass**; full suite **176 passed, 2 skipped**.

### 16.2 Seeded well diagnostics — `diag_seeded_wells.py`, mechanism found

Six hand-built states (fixed well col 9), champion full-2-ply picks:

| case | cur | bulk lines | dPhi | depth bef→aft | verdict |
|---|---|---:|---:|---:|---|
| precursor1 | T | 4 | −0.64 | 1→0 | avoids-setup |
| partial2 | L | 4 | +0.03 | 2→2 | builds-ish |
| ready | I | 8 | −4.50 | 4→0 | CASH-OUT ✓ |
| delayed_I | T | 5 | −5.02 | 4→0 | damaged-setup |
| blocked | I | 8 | −4.50 | 4→0 | (same as ready) |
| highstack | O | 14 | +0.28 | 2→2 | (sweeps stack) |

Two findings: **(a)** the scorer DOES cash out a ready well with I
(depth 4→0), so scoring/transition logic is sound — the problem is
purely incentive/horizon, exactly the review's prediction. **(b)** the
"bulk lines" numbers are inflated: `lock_candidates_full` compacts ALL
full rows, and solid-height suite boards have full rows below the well
floor — a Tetris there also sweeps pre-existing rows (8/14-line reports).
Suite verdicts use depth-collapse + lines≥4, not exact counts; a future
`shelf` (capped-well) case should give exact Tetris counts. Also:
`delayed_I` (T with a ready well) burns the setup instead of preserving
it — expected under zero setup incentive; the dense potential (§16.3) is
the fix to test, not a scorer bug.

### 16.3 Dense designated-well potential — `well_potential.py`

`dense_well_features()` + `well_potential()` (fixed well col 9, default
weights = inspection start, not a teacher): rows-full-except-well counted
at/above the well floor only (buried full rows are flat stack, not setup),
partial outside-well fill, `min(depth,4)/4` credit from depth 1 (the old
≥6 gate is gone), well-open, `tetris_ready`, hole/roughness/height
penalties, saturation past a usable setup. `style_fn_factory()` adapts it
to the `TwoPlyScorer` style interface (scale = Phi→z units; completion
reward stays in `line_bonus`, never in Phi).

### Next (still §15 order) — SUPERSEDED by §17 results below (2026-09-06 night)

1. λ-scale sweep — DONE (§17.1): dense Phi creates Tetris play (λ=20:
   34 Tetrises, 30/30 cap). Teacher gate PASSED.
2. DAgger round-0 — DONE (§17.4): head distills λ=20 teacher's well
   direction into 1-ply student (α=2: 13 Tetrises, 22/30 cap; α=4:
   19 Tetrises but 2/30 cap — veto too loose for 1-ply).
3. Rollouts — DONE as a probe (§17.3): controller continuation is the
   weak link, not the planning idea; deprioritized until a surviving
   continuation exists.
4. Representation stays deferred (§15.1.6 gate holds — no fitting failure
   shown; the style signal now exists without it).

---

## 17. §15 campaign results (2026-09-06 night) — dense incentive works

### 17.1 Dense-potential λ sweep — FIRST TETRIS SIGNAL (`diag_dense_sweep.py`)

Full 2-ply + dense Phi `style_fn` (fixed col 9) + `{1:1,2:3,3:5,4:8}`
completion bonus, 30 games cap 300, FRESH panel 20254–20283 (CONSUMED).
λ=0 arm is bonus-free (zero-style contract: 300.0 pcs / 118.1 lines,
30/30 cap — champion reproduced exactly). File: `dense_sweep.json`
(wall 1632 s).

| λ | pieces | lines | Tetrises | T/100 | T-share | lines-in-Tetris | cap | mean Φ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 300.0 | 118.1 | 4 | 0.04 | 0.001 | 0.005 | 30 | 0.000 |
| 2 | 300.0 | 117.8 | 3 | 0.03 | 0.001 | 0.003 | 30 | 0.281 |
| 5 | 300.0 | 116.7 | 4 | 0.04 | 0.001 | 0.005 | 30 | 0.413 |
| 10 | 300.0 | 115.7 | 15 | 0.17 | 0.005 | 0.017 | 30 | 0.720 |
| 20 | 300.0 | 110.8 | 34 | 0.38 | 0.014 | 0.041 | 30 | 1.142 |

Paired dLines vs λ=0: −0.3 (t−0.90) / −1.4 (t−2.71) / −2.4 (t−4.36) /
−7.3 (t−10.61). Survival gate HOLDS at every λ (30/30 cap): the cost of
style is lines/game (−7 at λ=20), not death — exactly the §15 trade the
review predicted (lines/100 is conserved only under flat survival; wells
hold cells back, so lines dip while Tetris share rises 10×).

Why this succeeded where §13.2/§14.3 failed: availability probe on 500
bank states — maxΦ>0.5 on **123/500 (24.6%)** vs 1.8% for the ≥6-deep
gate; mean within-state Φ spread **1.245**, spread>0.3 on 403/500. Dense
Phi scores precursors the flat policy actually visits. Correlation
check: mean within-state corr(v1, Φ) = **+0.122** (only 6/196 states
< −0.5) — Phi is a genuinely NEW axis, not a survival echo, and mostly
does not oppose survival (hence the gate holds).

Bug fixed en route: `dense_well_features` indexed `occ[21]` on near-death
stacks (heights can exceed the 21-row bitmask via lockout-tracked cells) —
clamped `top`/`hw` to 21 and broke the `rows_full` loop at 21
(`well_potential.py`; contract tests still 5/5).

Panel: 20254–20283 CONSUMED (registry updated).

### 17.2 Explicit controller — NEGATIVE as a player, kept as a tool

`well_controller.py` (BUILD→CASH-OUT→BUILD + RECOVER, NN-free, fixed col 9):
pure max-Phi BUILD drifts upward and dies ~91–115 pcs (lockout top-21);
lowest-top gating extends survival only to ~115–200 pcs with 0 Tetrises.
Even the hand-built heuristic baseline dies at 108 on seed 20254 — 20G
Death punishes any non-NN survival sense. Cash-out override test (NN
survival + forced take-any-≥4-clear): 300/300 cap on 3 seeds but 0
Tetrises — the champion never OPENS a well, so there is nothing to cash
out. Verdict: the controller is NOT a player and NOT a full-game teacher;
its value is as a rollout CONTINUATION and a seeded-suite probe. The
review's "diagnostic and teacher" framing holds for the diagnostic half;
the teacher half needs the λ=20 2-ply policy (§17.1), not the raw
controller.

### 17.3 Sampled multi-step rollouts — probe NEGATIVE, mechanism clear

`diag_rollout.py` (root shortlist top-3+3, K=4 tapes, D=6, common random
numbers, preview-correct, controller continuation, v1 anchor): 10 games
cap 300, seeds 20350–20359: **300.0 pcs / 112.6 lines, 2 Tetrises
(0.07/100, share 0.002), 10/10 cap** (wall 863 s) vs champion baseline on
the SAME seeds **300.0 / 116.5, 0 Tetrises, 10/10** (`rollout_probe.json`,
`rollout_baseline.json`; baseline wall 179 s, ~5× cheaper). Rollout adds
+2 Tetrises at −4 lines — directionally stylish but ~10× weaker per
compute than λ=10/20 single-move Phi. Mechanism: the continuation is the
ceiling — the controller dies at ~100 pcs, so 6-deep simulated futures
penalize exactly the setup paths they should reward (death-penalty −50
fires on well-building tapes). Planning is NOT closed as an idea; it is
gated on a surviving continuation (λ=20 2-ply itself, once distilled
fast). Until then: deprioritized. (Also fixed: default `--workers` 8→4;
co-running with the sweep OOM-killed workers — `BrokenProcessPool`.)

### 17.4 DAgger round-0: λ=20 teacher → 1-ply style head — TRANSFERS

`collect_teacher_bank.py`: 16 teacher games (λ=20, seeds 20400–20415) →
**4683 states, clears {0:3370, 1:996, 2:239, 3:64, 4:14}** — 15/16 capped
(one 183-pc game: λ=20 is stylish but not death-proof; the survival gate
at cap 300 is 30/30 only for the sweep panel, and longer/drought games
can still kill it). `train_teacher_head.py`: 4520/4683 pairs (teacher
pick vs lowest-Phi candidate — piece-id-free; teacher-bank has no piece
ids), 400 steps margin-ranking, feat direction
wheight −9.5 / holedepth −3.8 / cav −2.6 / wellcost +2.6 / spawn −1.7 /
lines +0.35 / deepwell +0.95 — the head learns the WELL direction (lower,
cleaner, deeper-well), opposite sign to the flat-2-ply Phase-A direction
(wheight +1.94, holedepth +1.02). Screened FRESH 20450–20479 (CONSUMED),
1-ply + α·style with veto 25 (`teacher_head.json`):

| α | pieces | lines | cap | Tetrises |
|---|---:|---:|---:|---:|
| 0.0 (ref) | 298.8 | 112.1 | 28/30 | 6 |
| 1.0 | 291.2 | 106.7 | 24/30 | 9 |
| 2.0 | 293.5→277.2 | 99.7 | 22/30 | 13 |
| 4.0 | 174.6 | 53.6 | 2/30 | 19 |

Style TRANSFERS to the fast 1-ply student (9→13 Tetrises vs 6 ref) but the
survival gate FAILS above α≈1 (28→22→2/30 cap): the 1-ply student has no
lookahead to refuse death, and veto-25 on 1-ply values is too loose. Two
caveats: (a) α=0 ref itself is 28/30 (this panel runs harsher tapes than
the sweep panel — λ=0 full-2-ply reference on 20450+ was not run, so the
2-cap deficit vs 30/30 is panel difficulty, not a contract break; the
§16 bank-agreement test still holds); (b) pairs use lowest-Phi as the
flat side (no survival margins — teacher bank stores no piece ids), so
the Phase-B survival weighting is absent; storing piece ids in the bank
is the fix.

### 17.5 What §17 gates (decision)

1. **λ=20 2-ply + completion bonus is the first well-building teacher.**
   Standing stylish configuration (not the survival champion — that stays
   snapshot9 + full 2-ply): same weights, `style_fn` λ=20 + `{1:1,2:3,3:5,
   4:8}` bonus. Cost: −7 lines/game at cap 300, survival 30/30 on the
   sweep panel.
2. **Next: DAgger round-1 with piece ids + tighter veto.** Store
   (placed, preview) ids in the teacher bank → restore survival-margin
   weighting → retrain → screen with veto {10, 25} × α {0.5, 1.0} on fresh
   20480+. Goal: keep α=1-style Tetrises (9–13) at ≥28/30 cap.
3. **Then: long-game survival gate** for the stylish config (cap 1500+,
   drought tapes): §15.0.3 requires promotion beyond cap-300.
4. **Rollouts gated** on a surviving continuation (§17.3); **controller**
   stays a probe, not a teacher (§17.2); **representation** still deferred
   (no fitting failure; §15.1.6 gate holds).
5. Panels: 20254–20283 (sweep), 20350–20359 (rollout probe + baseline),
   20400–20415 (teacher bank — training data, not a selection panel),
   20450–20479 (head screen) all CONSUMED. Next free: **20480+**
   (cap-300 style); 5000s reserves for promotion.

### 17.6 Stylish config in the live visualizer (2026-09-07)

`viz_live.py` now supports the §17.1 stylish config directly
(`TwoPlyVizScorer` takes `phi_lambda`/`well_col`; `--phi-lambda`,
`--well-col` CLI flags; scorer label shows `Stylish(lam=..,well=..)`).
Validated stylish command (same snapshot9 weights, no new `.pt`):

```
viz_live.py --nn --weights ../model_weights_snapshot9.pt --two-ply --line-bonus 1,3,5,8 --phi-lambda 20
```

(`--well-col 9` is the default.) Parity pinned by
`tests/test_viz_stylish.py`: viz stylish picks == sweep-path
`TwoPlyScorer(style_fn)` picks on 60 bank states; `--phi-lambda 0`
(or omitting it) == pure champion. Caveat carried over from the viz
adapter: ply-2 expansion inside `TwoPlyVizScorer.score_batch` still uses
the fast `search_placements` order while style rows are read per-candidate
only when bulk/search orders disagree (fallback lock+clear); the parity
test above guards the combined result.