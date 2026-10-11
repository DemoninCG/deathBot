# DeathBot: Neural network for 20G Tetris play

DeathBot/TAPNet is a neural network trained to survive and achieve Tetrises in 20G, designed for the Tetris: The Grand Master series of games (specifically TGM2's T.A. Death mode). It is a Pytorch neural net (430k parameters) built on a Rust simulation of TGM game mechanics, with a bridge to the BizHawk emulator for live play.

The 1-ply champion (snapshot12) is pure greedy play with no Tetris-scoring ability. It can survive for ~2,150 pieces on average, meaning it is good enough to achieve GM in most Death games. The model's abilities can be expanded to practically endless survival with 2-ply (search over next piece placements), which can be tuned for "stylish" play (scoring Tetrises). The stylish default setup is `cool30lam10tiered_burn6` on snapshot12 (commitment controller + λ10 dense well potential, tiered preserve, burn6), which achieves a Tetris rate around 22-24% of the theoretical maximum. Style is currently achieved only through search and the controller, not the weights (self-distillation for stylish play was unsuccessful).

## Layout

- `model_weights_snapshot12.pt`- final weights ("TAPNet", 430,465 params). Snapshots 1–11 are the training lineage (same arch)
- `deathbot_sim/` - Rust sim (`src/`: board, ARS rotation/kicks, IRS, 20G, BFS `search_locks`) + Python (`python/`: `model.py`, `two_ply.py`, `selfplay.py`, trainers, visualizer)
  - `deathbot_sim/TRAINING_ARCHIVE.md` - condensed training history and learnings (replaces the old unsorted 5000-line `PROJECT_STATE.md`)
  - `deathbot_sim/_diag_out/PANEL_REGISTRY.md` - seed-panel discipline record
  - `deathbot_sim/_diag_out/` - key promotion/showdown evidence JSONs only. Intermediate checkpoints, banks (`.pt`/`.npz`), and logs were removed; regenerate with the drivers in `python/`
- `tgm_bizhawk/` - live-play bridge: `tgm_1ply.py` (snapshot12 1-ply suggester), `tgm_mirror.py` (viewer), `tgm_player.lua` (BizHawk state publisher + input playback). See `BIZHAWK_STATE.md` + `MEMORY_MAP.md`. You will need your own copy of BizHawk and a TGM2P ROM.

## Quickstart

From `deathbot_sim/` with the workspace venv:

- Rebuild sim: `<venv>/Scripts/python.exe -m maturin develop --release`
- Tests: `cargo test` and `python -m pytest tests/` (needs `model_weights_snapshot{7,9,11}.pt` at root + `_diag_out/state_bank_snapshot9_32games.npz` fixture)
- Sweep: `python python/sweep_eval.py --help`
- Live viz: `python python/viz_live.py --two-ply`
- Live bridge: start `tgm_player.lua` in EmuHawk (T.A. Death), then `python tgm_mirror.py` from `tgm_bizhawk/`.
