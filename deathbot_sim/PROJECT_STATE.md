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
<venv>/Scripts/python.exe sweep_eval.py --games 30 --json <out>.json
<venv>/Scripts/python.exe diag_value_probe.py --weights ..\model_weights_snapshot7.pt
<venv>/Scripts/python.exe diag_run_experiments.py --rounds 4 --games 25
```