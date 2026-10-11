# DeathBot live-play bridge (TGM2 on BizHawk): project state

GM run achieved 2026-10-12 (Death, snapshot12 1-ply + true IRS). What
follows is the running build log, kept as the bridge reference.
Date: 2026-10-12. Prior agent hit a provider-side false-positive tripwire;
this document briefs a successor model. Workspace root:
`C:\My Things\HTML\Demonin's Item Shop\other\deathBot`, bridge code in
`tgm_bizhawk/`, simulator in `deathbot_sim/` (Rust PyO3 + Python),
emulator in `BizHawk/` (2.11, MAME 0.252 core, `tgm2p.zip` = TGM2+ Death).

## Goal and status

Drive real TGM2 Death mode with `model_weights_snapshot12.pt` (1-ply
greedy first, 2-ply later). State READ works fully (board mirror verified
against hand-recorded stacks). Input PLAYBACK mostly works (pieces move,
lock via Down) but is currently degraded by two open issues (see §6).

## 1. Memory map (all empirically verified, see MEMORY_MAP.md)

- Main RAM Lua domain: `sh2 : ram : 0x6000000-0x60FFFFF`
  (lua_addr = bus − 0x06000000, big-endian). P1 buttons:
  `Button 1/2/3, Down, Left, Right, Up` (controller 1).
- `0x64C04` NEXT piece, `0x64C05` CURRENT piece (each the sole RAM mirror).
  IDs: **I=0 Z=1 S=2 J=3 L=4 O=5 T=6**.
- `0x64C16` spawn-ish counter: counts 0→20 then STALLS (never use as trigger).
- `0x64C35` active timer: 0…lockval during fall; freezes through lock-flash;
  counts on through clear delay/ARE (43 observed). Lock edge = first frame
  hitting the lock-delay byte `0x64BE0` (30/26/… by level).
- `0x64BF0–2` gravity (`00 14 00` = 5120 = 20G). `0x64C2B` game-active flag.
- Spawns detected by cur/next turnover (NOT the counter).

## 2. Board reading (sprite list — no playfield array exists in main RAM)

- Video domain `sh2 : ram : 0x4000000-0x400FFFF`: u16-BE draw list at byte
  `0x3800` (≤0x400 entries, bit14 = end); `sprnum=(e&0x3FF)*4` → 4×u32-BE
  descriptor: xpos=w0&0x3FF (sext-10), ypos=(w0>>16)&0x3FF,
  wide=((w1>>8)&0xF)+1, high=((w1>>24)&0xF)+1, tnum=w2&0x7FFFF,
  colr=(w2>>24)&0xFF.
- Field cells: 1×1, `0x18200<=tnum<=0x182FF`, **x=18+8c, y=41+8r**
  (28/28 + 4/4 verified vs hand-recorded Death stacks).
- Locked palettes (palette-RAM hue-verified): I=0x3F red, J=0x5D dark blue,
  L=0x67 orange, O=0x71 yellow, S=0x53 pink, T=0x7B light blue, Z=0x49 green.
- Active pieces use OTHER animated palettes (I=0x3E/0x42, T=0x7E/0x7D…),
  double-drawn; previews sit off-grid (y≈15). Next display excluded by grid
  bounds. Falling pieces must be subtracted (or avoided) when modeling.

## 3. Sim-vs-TGM geometry (all verified, do not regress)

- J/L/S/Z tables are mirror-swapped vs SRS naming: ARS-L-spawn≡sim-J_0,
  ARS-J-spawn≡sim-L_0, ARS-S-spawn≡sim-Z_0, ARS-Z-spawn≡sim-S_0 (ARS-diagram
  pixel reads). Bridge remaps in `tgm_1ply.TGM2SIM` (index↔geometry pairing
  preserved, no retraining). I/O/T symmetric.
- Sim state+1 cycles visually CCW for T/J/L (cell-verified); bridge emits
  TGM-side step directions per piece (`INVERTED={"T","J","L"}`); S/Z/I/O
  direction-free. Sim untouched (self-consistent).
- T_2 ("up") lowered one cell (`[(0,0),(1,0),(2,0),(1,1)]`): old height let
  T escape well mouths (fundoshi) needing TGM3-only floor kicks. Pinned by
  `rotation::tests::t_up_orientation_sits_low_no_fundoshi` (failed pre-fix).
- Single-root BFS `search_locks_bare`/`search_locks_from` (+ tests); the old
  multi-root filter biased picks flat (measured: rotated bests rejected).

## 4. Runtime architecture (files exchange via tgm_bizhawk/*.txt)

- `tgm_player.lua` (EmuHawk): publishes state file (spawn reuses cached lock
  board; lock boards via quiescence: scans every 4f while timer>=lockval-ish
  window, publishes on STABLE PAIR, skips full-row / full-empty-gap boards,
  cap failsafe). Consumes `tgm_moves.txt` (4-line: phase/for_frame/hold/
  steps). Executes: ARE holds, steps at STEP_FRAMES (**currently 1** —
  1f presses demonstrably miss polls, 3f holds; DAS-trip risk on runs),
  Down finale (instant lock, frame-verified), turnover abort, releases.
- `tgm_mirror.py` (PC Pygame): two-phase decisions — lock publishes choose
  the IRS ROOT (full argmax, next≈itself, single-tap roots) → hold move;
  spawn publishes decide placement constrained to the HELD root
  (header.held, authoritative) with true next → steps move. Ghost overlay.
- Probe-verified controls: B1=CCW, B2=CW, B3=CCW-dup; Left/Right = 1 cell;
  Up = sonic no-op grounded; Down = instant lock grounded (NOT natural
  expiry — early readings were wrong: 4f probe windows can't see the ~19f
  ARE-to-turnover); IRS = HOLD through spawn rotates exactly once, taps do
  nothing (decisive probe 2026-10-12).
- Test status: cargo 35+3 green; pytest 195 passed + 2 known-stale
  shortlist fixtures (pin pre-§54 search order) + 1 stochastic phase1 flake
  (fails identically on pristine source).

## 5. Design rationale log (don't re-litigate without new data)

- Spawn-time (not lock-time) decisions: header.cur at lock = dead piece
  (caused systematic one-piece lag). No pre-spawn adoption (ARE-eaten
  steps + DAS pre-charge). No same-frameend press+release (last-set-wins).
  Post-spawn tap ≠ IRS in 20G (piece already grounded; kicks differ) →
  bare-only era, then true ARE holds. cnt never trusted. next≈cur only for
  root choice; spawn uses true next.

## 6. CURRENT OPEN ISSUE (as of last session)

User reports holds still never engage (no `holding … for IRS` lines) even
after the done_hold-shadow fix (which was proven: ROOT lines flowed, Lua
never held). Suspects, ranked:
1. STALE FILE: user may be running an older `tgm_player.lua` (the fix is
   brand-new and unverified — confirm the file contains "no done_hold
   shadow here" comment and the hold press block).
2. Poll still shadowed: re-check the `elseif plan == nil` chain ordering
   and the `not turnover` + `tmr >= lockval` gates against a real ARE
   (timer reads small?? if the ARE timer doesn't read >= lockval, holds
   never start — log tmr in the poll would confirm).
3. Moves mismatch: `for_frame == lastpub` failing every frame (log both
   values on mismatch — currently silent).
4. Path mismatch: mirror writes elsewhere than Lua reads (both must be
   `tgm_bizhawk/tgm_moves.txt`; state observed landing in tgm_bizhawk/).
Recommended next instrumentation (cheap): log every poll outcome
(match/mismatch + tmr/lockval + file presence) for one ARE cycle — it
discriminates all four hypotheses in a single run. Then re-test with
`ORI spawncheck` shapes vs intended roots as the acceptance check.
Also still open: STEP_FRAMES=1 poll-miss rate (user testing), DAS-runaway
watch, high-stack IRS kicks, O-piece edge cases.

## 7. BAD-HEADER ROOT CAUSE (found + fixed 2026-10-12, unverified live)

`STATUS: bad header` was not a read race — it was Lua's own `exec`
values. `tgm_player.lua` publishes `exec=play 3/4` (step progress) and
`exec=hold Button 1/2` (ARE hold), both containing spaces, while
`tgm_mirror.py parse_state` split the header on whitespace and demanded
`k=v` per token (`3/4`, `Button`, `1` fail → ValueError → "bad header").
Reproduced offline against the live `tgm_state.txt`
(`frame=2593 ... exec=play 3/4 held=-` fails the old parser verbatim).
Impact on IRS: any publish landing mid-path (lock publishes during the
post-Down re-adoption loop almost always do) was unparseable, so
`_decide_root` never ran, no hold move was written, and Lua had nothing
to hold — the "holds never engage" starvation. Spawn publishes carry
`exec=idle/done` (no space), which is why bare placement kept working.
Fixes (both sides, either suffices alone):
- `tgm_player.lua publish_rows`: sanitizes `(exec:gsub("%s+","_"))` on
  the wire (`play_3/4`, `hold_Button_1`); internal exec unchanged.
  Same edit makes the state swap Windows-atomic (remove-then-rename;
  previously rename always failed when the dest existed and fell back
  to a torn direct write).
- `tgm_mirror.py parse_state`: regex header grammar, exec captured
  greedily up to trailing `held=` (spaced or sanitized both read;
  legacy 6-field writer still reads; true corruption still errors).
  Verified offline: spaced/sanitized/idle/hold/legacy parse, corrupt
  still reports bad header. `py_compile` clean.
- `tgm_player.lua` adopt block: behavior-preserving diagnostics only
  (deduped ~1 line/publish-cycle, never per-frame): `hold ready but
  tmr<lockval` (gate hypothesis), `play ready but tmr>=lockval`,
  `moves waiting (for_frame/lastpub/phase/tmr/lockval)` (mismatch
  hypotheses), throttled `no moves file` (path hypothesis).
NEXT TEST: reload the edited `tgm_player.lua` in EmuHawk (confirm the
console shows the new "armed" line AND the file contains "holdgate"),
restart `tgm_mirror.py`, play one ARE cycle. Expect: no more
`bad header`; lock publishes yield ROOT lines; then exactly one of
`holding ... for IRS` (fixed) / `hold ready but tmr<N` (gate is the
remaining blocker — Down-lock ARE timer reads small) / `moves waiting`
(stale for_frame or path split). Report which line appears.

## 8. TIMER-GATE DIAGNOSIS (confirmed 2026-10-12, fix applied, unverified live)

The §7 diagnostics discriminated fully: ROOT lines flow every lock
(hold moves written, for_frame matched), but Lua logged
`hold ready but tmr=12..27 < lockval=30` on all 8 hold opportunities —
the `tmr >= lockval` adoption gate never opened. Cause: Down freezes
0x64C35 small and the ~19f ARE ends (spawn turnover) before it can
climb to 30; the "counts to 43 through ARE" observation only holds for
natural locks (timer already at 30). Since every bot piece ends in
Down, the hold window was zero frames wide.
Fix (`tgm_player.lua` adopt block): no timer gate on holds — hold
moves are written exclusively on lock publishes and the for_frame match
pins adoption to that piece's post-lock window (no falling piece
exists to twist; holding rotation with no active piece is a probe-
verified no-op until spawn sampling). The dead `holdgate` log arm was
removed; play/mismatch/nomoves diagnostics stay. Play path untouched.
NEXT TEST: reload Lua, restart mirror. Expect `holding Button 1/2 for
IRS` lines, spawn headers with `held=CW/CCW`, and MOVES lines with
nonzero roots. Acceptance: ORI spawncheck orientations match the
intended roots (e.g. a `hold=CCW` T piece spawns rotated vs bare).
Watch items (non-blocking): one startup `no moves file` transient;
one `moves waiting (phase=play, tmr=1)` during a 118ms Python decision
— both self-recovered. Follow-up candidate (not touched): the stale-
play re-adoption loop replays each path + Down several times per piece
(noisy, self-terminates at the lock publish).

## 9. LOCK-PUBLISH STARVATION (diagnosed 2026-10-12, fix applied, unverified live)

Symptom: mirror board freezes a few pieces into each game while
cur/next keep updating. Logs showed lock publishes flowing for cnt=1-2
then stopping entirely (cnt=3/4/5: spawns only, no locks, no ERRORs, no
`unsettled at cap`). Spawns reuse `board_cache`, so a stale cache =
frozen board + live header — exactly the reported symptom.
Cause: the stale-play re-adoption loop (§8 follow-up). Every Down
finale re-arms quiescence (lock_frame=now, last_qkey=nil), and a
stable-pair publish needs ~8 undisturbed frames (two scans 4f apart).
Long paths (6 steps, cnt=1) leave a wide enough window; 2-step loops
re-arm every ~7f and 0-step loops every ~2f (`frame >= next_qscan`
never true), so no stable pair ever forms, and the spawn turnover then
silently kills the watch (`lock_watch=false`). First short piece in a
game freezes the board from there on.
Fix (`tgm_player.lua`): one-shot adoption via `consumed_for` — a play
plan adopts once per piece (cleared on turnover); re-adoption of the
same for_frame hits a silent arm. Re-adoption was never correct anyway
(replaying an absolute step list from a diverged/no-longer-falling
position), and the quiet lets quiescence publish. Hold adoption
untouched (idempotent refresh). No Python changes.
NEXT TEST: reload Lua, restart mirror, play past the first short/
0-step piece. Expect: exactly one `plan adopted` per piece (the
repeated adopt/Down spam gone), a `wrote ... lock` line per piece, and
the mirror board advancing every piece. Then re-run the §8 IRS
acceptance (ORI spawncheck vs intended roots).

## 10. CENTER-COLUMN RULE CORRECTION (fixed 2026-10-12, verified in sim)

Live-game report (Death ~level 443): flat TGM-J resting its left half
on two stack blocks (`JJJ`/`##J`) refuses rotation in BOTH directions;
wiki-described 1..9 scan sees bottom-left first (off-center) and allows
the +1 kick. Reproduced in sim pre-fix: TGM-J == sim-L geometry, and
sim-L state 0 with stack at box cells 7+8 returned CW-success via +1
(the snag: bot plans a TGM-CCW press that does nothing); CCW already
failed, L-mirror (cells 8+9) already failed.
Fix (`deathbot_sim/src/rotation.rs` CENTER_SCAN): bottom-center (8) is
checked before bottom-left (7) — order 1,2,3,4,5,6,8,7,9. No
piece-specific flag was needed: the wiki rule is refuted on symmetry
alone (it verdicts mirror situations {7,8}/{8,9} differently, the game
blocks both), and the swap changes outcomes ONLY in the
first-hit-7-with-8-blocked family — all 11 pre-existing Rust rotation
tests + mirrors pass unchanged. New pins: Rust
`center_column_bottom_center_checked_before_bottom_left` (J-case both
directions + L-mirror) and its `test_sim.py` mirror.
Verified: cargo 36+3 green, pytest test_sim.py 60 passed, live repro
script shows None/None post-fix. Extension rebuilt via
`maturin develop` — the next mirror run already uses fixed physics, no
retraining needed (fix only removes phantom rotations the game never
allowed; reachable-lock sets shrink toward truth).

## 11. 500+ BOARD FREEZE (diagnosed 2026-10-12, fix applied, unverified live)

Symptom: past 500 the mirror board freezes while cur/next keep going.
Terminal tail: spawn-to-spawn intervals collapse to 12-22f (vs ~34f
early) and ROOTs vanish entirely after f≈27234 while MOVES continue —
quiescence starvation Mk II (§9 was the re-arm loop; this is pure
timing): the 4f-cadence stable pair needs ~8f post-Down, but 500+ ARE
(lockval observed stepping 30->26->22) ends first, and turnover kills
the watch. Spawn reuses the stale cache → frozen board, live header.
Fixes (`tgm_player.lua`, no sim changes):
- Quiescence cadence 4f->2f (all three `next_qscan` sites): pairs fit
  in ~4-6f, preserving lock publishes (and IRS) deeper into 500+.
- Turnover backstop: if lock_watch is still armed at turnover (i.e. no
  publish happened), refresh board_cache with one settled scan.
  Deliberately SILENT (no publish): a post-turnover lock publish would
  trigger a hold move for an already-sampled spawn — useless for IRS
  and adoptable mid-fall (twists the new piece). Graceful degradation:
  board stays fresh, that piece plays root-0.
- (`tgm_mirror.py`): `_write_moves` retries os.replace 4x10ms — kills
  the sporadic WinError 5 drops (2 seen: f=19338, f=23998) where Lua
  holds the file open during its frameend read.
- Comments refreshed (STEP_FRAMES=2 validated through 500+; lockval
  30->26->22 observed; 500+ DAS-shortening watch).
NEXT TEST: reload Lua, restart mirror, push past 500. Expect lock
publishes to resume (ROOTs flowing again if pairs fit; silent-cache
pieces otherwise). Watch: whether IRS holds still engage at 500+ pace,
and any DAS runaways on long same-direction runs (DAS shortens up
there — STEP_FRAMES may need revisiting per section).
