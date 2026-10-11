# deathBot — TAP Death (TGM2) bot

First live Death GM achieved (level 999) with `model_weights_snapshot12.pt`.

Rust simulator + Python neural net + BizHawk live-play bridge.

## Layout

- `model_weights_snapshot12.pt` — final weights (TAPNet, 430,465 params). Snapshots 1–11 are the training lineage (same arch).
- `deathbot_sim/` — Rust sim (`src/`: board, ARS rotation/kicks, IRS, 20G, BFS `search_locks`) + Python (`python/`: `model.py`, `two_ply.py`, `selfplay.py`, trainers, visualizer).
  - `deathbot_sim/TRAINING_ARCHIVE.md` — condensed training history and learnings (replaces the old 5000-line `PROJECT_STATE.md`; full detail in git history).
  - `deathbot_sim/_diag_out/PANEL_REGISTRY.md` — seed-panel discipline record.
  - `deathbot_sim/_diag_out/` — key promotion/showdown evidence JSONs only. Intermediate checkpoints, banks (`.pt`/`.npz`), and logs were removed; regenerate with the drivers in `python/`.
- `tgm_bizhawk/` — live-play bridge: `tgm_1ply.py` (snapshot12 1-ply suggester), `tgm_mirror.py` (viewer), `tgm_player.lua` (BizHawk state publisher + input playback). See `BIZHAWK_STATE.md` + `MEMORY_MAP.md`.
- `BizHawk/` — local emulator install (gitignored, never commit).

## Quickstart

From `deathbot_sim/` with the workspace venv:

- Rebuild sim: `<venv>/Scripts/python.exe -m maturin develop --release`
- Tests: `cargo test` and `python -m pytest tests/` (needs `model_weights_snapshot{7,9,11}.pt` at root + `_diag_out/state_bank_snapshot9_32games.npz` fixture)
- Sweep: `python python/sweep_eval.py --help`
- Live viz: `python python/viz_live.py --two-ply`
- Live bridge: start `tgm_player.lua` in EmuHawk (TGM2 Death), then `python tgm_mirror.py` from `tgm_bizhawk/`.

## Final configs

- 1-ply champion: snapshot12 (pure greedy).
- Stylish default: `cool30lam10tiered_burn6` on snapshot12 (commitment controller + λ10 dense well potential, tiered preserve, burn6) — ~2.3–2.4 T/100 at cap 1500/3000 with 18/20 and 24–26/30 caps. Style lives in search/controller, not weights (pure 1-ply T/100 ~0.1 on all snapshots).
- Live GM config: snapshot12 1-ply + true IRS (no style bonuses — survival only).
