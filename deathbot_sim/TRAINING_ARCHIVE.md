# deathBot training archive (condensed)

Replaces the 5000-line `PROJECT_STATE.md` (§§1–60, 2026-09-04 → 2026-09-15).
Per-seed tables, superseded plans, and intermediate panels were cut; full detail survives in git history.
Training is CLOSED. Final: 1-ply champion **snapshot12**, stylish default
**`cool30lam10tiered_burn6`** on snapshot12, first live Death GM (level 999, 2026-10-12).

## 1. System (§1)

- TAP Death (20G) bot: Rust `deathbot_sim` (board, ARS rotation/kicks, IRS, 20G drop, BFS `search_locks` with input paths) + Python TAPNet (~430K params: conv over [1,20,10] + current/next one-hots → 128→1 value).
- Playing policy is `two_ply.py` full-expansion 2-ply. `selfplay.py` is the survival-era MC trainer (patched, no longer the lever).
- Style stack: `well_potential.py` (dense designated-well Φ), `commit_controller.py` (constrain-first col-9 + cash-out + hysteresis), `well_controller.py`, `style_head.py`.

## 2. Weight lineage (§§2, 59) — all 430,465 params, same arch

| Weights | Meaning | 1-ply level (post-§54, cap 3000) |
|---|---|---|
| snap1–6 | Session-1 climb (67→481) | 74 → 299 |
| snap7 (`model_weights.pt` identical) | Healthy session-1 peak; collapse-investigation baseline | 329 |
| snap8 | Catastrophic collapse artifact (see §3) | 30.5 |
| snap9 | C3 self-distillation of snap7 (+231 × 2 panels); pre-loop champ | 366 |
| snap10 (`fullsplit_seed42`) | Fullsplit student (+468, §30) | 553 |
| snap11 (`gen2_seed44`) | Champion-start + fs2ply teacher (§51) | 892 |
| snap12 | Loop-gen-2 champ r3 (§52); **final 1-ply champion** | 1769 (9/50 cap) |

Learning: weight-space distance lies — cosine 0.999 can hide a 16× play gap. Evaluate behavior only. Style lives in search/controller: pure 1-ply T/100 is ~0–0.11 on every snapshot.

## 3. Collapse root cause + selfplay patches (§§3–7)

- Session-2 collapse (snap7 503 → snap8 ~32, every resume after) root-caused by experiment: `selfplay.py` z-scored targets to ≈[-3,+1.5] while warm-started values live ≈[-300,+35] → ~100× crush in one round. Second poison: afterstate (post-lock) scoring vs snap7 pre-lock training. Jul-18 normalizer sidecar made it worse.
- Patches landed: scale-anchored targets, replay amnesia off, eval guardrails (same-seed baseline, per-round eval, always `.best.pt`, rollback), `--freeze-conv`, `--elite-fraction`, loud `--weights-in`.
- Result: collapse fixed but all adaptations equilibrate ~205–215. Mechanism: **MC labels are weaker than the policy** (rollouts 150–340 vs 500 policy drag values down). Lesson that repeats all campaign: no training step compounds without a fresh information source.

## 4. Fixed-data distillation + fitted-TD: stabilizers, not improvers (§§7–11)

- C3 affine-fit on fixed bank: +231 fresh (740 vs 509, t+2.8) — born-again smoothing, the only fixed-data win. Round-2 at same recipe: neutral.
- Phase 3 / fitted-TD (`train_fitd.py`, γ0.999): behavior-neutral (dev gate passed, validation −3.9 and +15.1, both n.s.). Verdict: fitted-TD holds snap9 through live training (July code couldn't) but converges to itself.
- Distillation post-mortem (§49) rules, all replicated: (a) dose discipline (early dW~1; P2T was under-dosed dW 0.36 vs 4.69, not a true negative), (b) affordability match (style margins −10…−23 vs survival −2.8 need decoupled heads, never full-network CE), (c) fresh teacher/start required (E3 bank+aggregation shaves ~1 regret, dev n.s.; extension +42.5 t+0.69 is truncation luck), (d) gameplay selection (agree_dis is not a gameplay gate).

## 5. Search reversal: survival solved by inference (§12, E1)

- Initial beams “fail” was a truncation artifact (unexpanded kept v1, expanded got v1+γ·max(v2) with negative max → beam<n picks unexpanded bad moves).
- Full-expansion 2-ply: 30/30 caps at 1500 and 5000 (+505 t+6.16). Program pivots to style.
- E1: under 2-ply every weight cell is 20/20 with identical pieces — search erases the +468 gap. Operating config snap9+b8_diverse (top-k bug fixed, §31: decide over expanded shortlist only; beam8_diverse ≈ champion at ~2.3× speed).

## 6. Style era breakthroughs (§§13–23, 31–37)

- Flat policy never visits precursors (only 1.8% of bank states have any well-setup candidate) — single-move bonuses can't select what doesn't exist. Well bonus: clean negative.
- Dense designated-well potential Φ (`well_potential.py`, col-9, depth/4 from depth 1, no ≥6 gate): FIRST Tetris signal — λ 0→20 gives 4→34 Tetrises, all 30/30 capped. Φ builds, `{1:1,2:3,3:5,4:8}` line bonus releases (factorial §23).
- Long-sustain: λ10 viable (10/10, 26T, flat occupancy); λ20 rejected (4/10, 6 late deaths, 61→114 occupancy).
- λ10 gate (§31): 20/20 vs 20/20 at cap 3000, +0.173 T/100 t+8.19; failure ≈1/40. Rescue-capable (7/7 handoffs recover; λ10-control dies 7/7 — recoverable traps). Hysteresis (90,50) holds style at 7.5% style-off.
- E0: the §29 gap was RNG aliasing (`select_candidate_index` draws explore RNG even at ε=0), scoring bit-identical. E2: student proposals sit inside the v1 size/coverage frontier — no closed loop.

## 7. Commitment controller + Pareto (§§38–48)

- λ30/40/50/60 keep climbing on T/100 but die early (cap-rate knee, not style knee) — controller job: keep openings that work, abandon grave-digging.
- Rung-1 `commit_controller.py` CONVERTS: 0.522 (30/30) vs λ40 0.512 (15/30), dTet +0.93 t+4.36. Sustains at 1500 (0.556, 18/20). Defect was re-entry — fixed with earned re-entry (well_open + top≤10 + cooldown), not occupancy.
- Stage-B PROMOTE: `cool30lam10` cross (0.506, 13/15) ties cool30 style at better caps/lines. Audit-A FAILED (21.5% flips) → masked-root full expansion implemented + validated; Stage-C mini: corrected cross beats cool30 on style (dTet +2.73 t+2.10), cap tied.
- Rounds A/B/C: debt80 CONFIRM-REJECTED, top10/cashout confirmed inactive, watch80 sole candidate then no-promotion on fresh panel (max |t| 0.90), a120 lean reversed. n=30 leans evaporate at n=50 — pooled/fresh confirmation required (winner's curse ×3: soup, burn-§57, challengers).
- Pareto close-out: survival corner snap9+full (30/30, T~0.03); λ10 knee (~0.15–0.20, ~97.5%); stylish default corrected `cool30lam10` (27/30, 0.60 T/cap 8.8). CLOSED: λ20, debt, uniform, READY, all 1-ply style heads, 3-ply (15/120 flips, all doomed, 5.26× cost).

## 8. Self-distillation loop → snapshot11 → snapshot12 (§§50–52)

- Soup over recipe seeds: same basin (cos≥0.9989), safe, no promotion (pooled +25.7 t+0.42).
- Dose sweep: LR axis CLOSED (1e-4 optimal). Teacher-2 probe PASSED: champ's own 2-ply gives 782 new affordable targets (median −1.1) — outcome-identity ≠ decision-identity.
- Gen-2 (champ start + fs2ply, 3 seeds): pooled +184 t+2.12, all positive. Promotion (n=100 cap 3000): gen2_s44 +768 t+5.97 → **snapshot11**.
- Loop-gen-2 (mixed 70/30 bank + 21,959 hard-start states, dual survival+recovery screen, adopt delta≥0): r1 ADOPT, r2 REVERT, r3 ADOPT, r4 genuine exhaustion (242<300). Final gate G1 +294 t+2.20 AND G2 +42.6 t+1.23 → promotion (n=100 cap 3000) +283 t+2.75 → **snapshot12**. Chain snap9→10→11→12 validates fit→expand→refit.

## 9. Snapshot12 style re-exam + burn promotion (§§53, 55–58)

- Knee moved to λ10 on snap12 (29/30, 1.07 T/100, 5.6× snap9 λ10); λ>10 CLOSED. Cross re-gated (+1.87 t+3.34).
- Tiered preserve (block overhangs, T0→T1→T2→fallback, T0 binds 96–98%, fb 0.0): +13.25 t+6.69 at 1500, +22.7 t+6.89 at 3000 → PROMOTED. Factorial: interaction-only (tiering+λ needed together).
- Burn: B=2→4 dose-responsive; burn4 +1.40 t+2.28 (cap 300) → +8.0 t+2.96 (1500); §57 promotion FAILED (t+1.49, HOLD) → §58 rerun (30 tapes) burn4 +14.5 t+2.52 AND burn6 +18.3 t+3.47, both PASS → **`cool30lam10tiered_burn6` PROMOTED** (2.41 T/100). `STYLISH_DEFAULTS` updated.

## 10. TAP rotation fix — all §§12–53 numbers stale (§54)

I never kicks, center-column bottom-center-before-bottom-left (order …6,8,7,9 — wiki 1..9 refuted on symmetry), IRS bare-only. 10 new tests. Champs unchanged, everything re-gated above (§§55–60) under corrected physics.

## 11. Final validation (§§59–60, post-§54)

- 12-snapshot 1-ply showdown (12×50 @3000): snap12 1769 > snap11 892 > snap10 553 > snap9/7/6 ~300 > snap4 dip 191 > snap2/1 > snap8 30; all vs snap12 t≤−5.5. Refinement (1–10 ×200): same order ±, snap4 dip holds.
- Stylish lineage (6×20 @1500): burn6 18/20 @2.27 holds over tiered (−5.7 t−2.15), cross (−19.2 t−9.21), snap12_lam10 5/20 horizon-dead, snap9 rows collapse (9/20, 14/20). Controller is what keeps λ10 alive at horizon.

## 12. Live bridge + GM (`tgm_bizhawk/`, see `BIZHAWK_STATE.md` + `MEMORY_MAP.md`)

Snapshot12 1-ply + true IRS drove real TGM2 Death to GM (level 999). Sim fixes for live: single-root/bare + root-constrained BFS (`search_locks_bare`/`search_locks_from`, `irs_root` on candidates), T_2 lowered one cell, J/L/S/Z mirror-swap remap in bridge (no retraining), direction translation (sim+1 is CCW for T/J/L). Emulator findings: sprite-list board read (x=18+8c, y=41+8r), RAM `0x64C04/05` next/cur, IRS = ARE hold (taps do nothing), Down = instant lock, quiescence stable-pair publishes + turnover backstop, 4f→2f cadence past 500.

## 13. Method learnings (keep)

- Panel discipline: each seed range used for at most one selection; `PANEL_REGISTRY.md` is authoritative (21950+/5500+ next free). Never select on reporting panels.
- Paired stats only (same tapes, per-seed deltas, t + W/L/T + discordant caps); cap-rate and MTTD alongside means; n=30 leans require n=50–100 confirmation.
- Harness parity: aliased explore RNG, true-preview vs cur-duplicated slots, corrected lockout masking — both arms share one harness or the comparison is void.
- Bugs that ate weeks: z-score collapse (§3), top-k unexpanded-argmax (§31), RNG aliasing (E0), multi-root flat bias (bridge §3), center-column order (§54). Each got a regression test.

## 14. What was removed

One-off `diag_*` drivers (kept: `diag_phase2`, `diag_tinyset_overfit`, `diag_seeded_wells`, `diag_commit_stageC`, `diag_style_head`, `diag_snapshot_showdown`, `diag_style_lineage`), scratch `_*` scripts (kept: `_recovery_suite`), `_diag_out/` intermediates (kept: registry + key promotion/showdown JSONs + test fixtures), root pretty-printers, and collapsed-family duplicates. Git history preserves everything.
