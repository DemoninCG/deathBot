# TGM2 (tgm2p) memory map — BizHawk MAME 0.252, Death mode

Derived empirically 2026-10-11 from RAM dumps + sprite-list decoding.
Lua addresses are offsets into the named memory domain
(`lua_addr = SH2_bus − domain_base`). SH-2 is big-endian.

## Main RAM domain: `sh2 : ram : 0x6000000-0x60FFFFF`

| Lua addr | Bus | Name | Values / notes |
|---|---|---|---|
| 0x64BE0 | 0x06064BE0 | lock delay | 30 in Death 0–99 (known from wiki, confirmed) |
| 0x64BE1 | 0x06064BE1 | DAS counter | counts while held (attract demo confirms) |
| 0x64BF0–0x64BF2 | 0x06064BF0–2 | gravity | `00 14 00`; LE@BF0 = BE@BF1 = 5120 = 20G |
| 0x64C01 | 0x06064C01 | level-ish | 2,4,6,8 for spawns 1–4 (GO phase: 0x15, ignore) |
| **0x64C04** | 0x06064C04 | **NEXT piece** | unique in RAM; see IDs below |
| **0x64C05** | 0x06064C05 | **CURRENT piece** | unique in RAM (= most recent deal) |
| 0x64C06–07 | 0x06064C06–7 | older history | newest→oldest; initial `[Z,S,S]`-ish |
| 0x64C16 | 0x06064C16 | spawn-ish counter | ++ per spawn 0→20, then STALLS (pieces keep spawning; cur/next keep cycling). DO NOT use as a trigger — detect spawns via (cur,next) turnover, locks via timer edge |
| 0x64C2B | 0x06064C2B | game-active flag | 0 in GO, 1 in game |
| 0x64C35 | 0x06064C35 | active-ish timer | 0…lockval during fall, frozen at lock value through lock-flash, counts on through line-clear delay/ARE (43 seen), 0 at next spawn |

Piece IDs: **I=0, Z=1, S=2, J=3, L=4, O=5, T=6**.
(TGM order, NOT the sim's I0 O1 T2 S3 Z4 J5 L6 — the bridge must remap.
Note the J/L geometry swap below: wire-J uses sim-L type+index and vice
versa. S/Z map straight through — spawn shapes, locked shapes, and palette
hues all agree; only sim *rotated* states for S/Z remain unverified SRS
placeholders, see notes.)

## Rotation correctness notes (sim vs ARS)

Pixel reads of the ARS basic-rotation diagram (screen frame) vs sim tables:
ARS-L-spawn ###/#.. == sim-J_0, ARS-J-spawn ###/..# == sim-L_0,
ARS-S-spawn .##/##. == sim-Z_0, ARS-Z-spawn ##./.## == sim-S_0; rotated
states follow the same mirror (spot-checked ARS-L verticals vs sim-J_1/J_3).
I/O/T are mirror-symmetric and unaffected. The bridge
(`tgm_1ply.TGM2SIM`) crosses both pairs (wire-J→sim-L, wire-L→sim-J,
wire-S→sim-Z, wire-Z→sim-S), keeping each net one-hot index paired with
the geometry it learned (no retraining). Palette hues corroborate all
color↔letter links independently of shapes:
0x49 green=Z, 0x53 pink=S, 0x5D dark-blue=J, 0x67 orange=L, 0x3F red=I,
0x71 yellow=O, 0x7B light-blue=T.

- J/L/S/Z geometry is mirror-swapped vs SRS naming (bridge remaps both
  pairs; I/O/T symmetric). Rotation *direction*: sim state+1 cycles
  visually CCW for T/J/L, so the bridge emits TGM-side directions per
  piece; S/Z/I/O are direction-free.
- T_2 ("up") sits one tile lower than a naive bottom-alignment
  (`[(0,0),(1,0),(2,0),(1,1)]`): at the old height T could rotate out of
  well mouths (fundoshi), which needs a floor kick that only TGM3 has.
  Pinned by `rotation::tests::t_up_orientation_sits_low_no_fundoshi`.
- Lock publishes require settled boards: no full rows (clear in progress)
  and no fully-empty row under filled cells (mid-drop gap — cleared lines
  vanish first, rows above fall after the clear delay). Unsettled boards
  are never decided on or cached.

Timing (Death 0–99, no input): spawn → timer 0…30 (lock delay) → ~19f
lock-flash + ARE with timer frozen at 30 → next spawn (cntA++, timer 0,
`64C04/05` shift). Bot cycle: on timer→30, snapshot board; at next spawn,
execute inputs during the 30f lock window.

## Input playback (v1, diagnostic)

## Input playback (true IRS via ARE hold)

- Decisions are two-phase. LOCK publishes (quiescence) refresh the board;
  Python picks the IRS ROOT for the upcoming piece (full argmax with next
  approximated by itself, restricted to single-tap roots) and writes a
  hold move. SPAWN publishes decide the full placement constrained to the
  root Lua actually held (header.held, authoritative — never assumed) with
  the true next.
- Lua HOLDS the root button every frameend through spawn+1, then releases
  (probe: taps do nothing, a hold rotates exactly once — safe). A hold
  that never executes (cap, missed window) reports "-" and the spawn
  decision falls back to bare root 0.
- v1 path search is root-constrained BFS (`search_locks_from`): every path
  starts exactly where the real piece is. An orientation trace (ORI console
  lines: spawncheck after every turnover plus post-rotation shapes, active
  minos minus cached locked cells) audits each hold offline.
- Lua adopts Python's moves file whenever plan==nil, its for_frame
  matches the latest publish, the piece is actually falling (timer below
  lock value), and NOT on the turnover frameend itself (the spawn publish
  below hasn't run yet there, so lastpub still points at the previous
  state — adopting there replayed every plan one piece late); steps play
  at STEP_FRAMES (currently 1f test spacing; 1f presses demonstrably miss
  input polls, 3f holds).
- Lock publishes use quiescence with a stable-pair rule, not the raw edge:
  the locking piece converts palettes over several frames and every flicker
  would otherwise publish a different board and flip-flop the decision
  mid-piece. Publish on two consecutive identical scans (cap +24f
  best-effort); spawn publishes reuse the cached lock board and never
  rescan (the falling piece must not pollute the model). Frame-gated only
  (no timer condition): Down-locked pieces never raise the timer edge, and
  quiescence is also armed by our own Down press so the cache can't go
  stale.
- Python (`tgm_1ply.suggest`) plans from an explicit root
  (`search_locks_from`; bare 0 by default, held root at spawn); turnover
  abort keeps leftovers from corrupting the next piece. Buttons
  force-released between presses, fully released (nil override) on
  abort/done (releases exempt an active hold).
- Rotation direction: sim state+1 cycles visually CCW for T/J/L (cell-
  verified vs ARS + probe), so sim "RotateCw" steps/IRS execute via the
  TGM CCW button and vice versa. `tgm_1ply` translates per piece and emits
  TGM-side names; S/Z/I/O are direction-free. The sim stays untouched
  (internally self-consistent); if its tables are ever renumbered, the
  bridge translation must flip back.
- Button map (probe-verified): Left/Right/Down + Button 1 = CCW,
  Button 2 = CW (Button 3 duplicates CCW). Left/Right move exactly 1 cell
  per tap at 3f+ spacing; 1f spacing misses polls, and long same-direction
  runs at any spacing read as a hold (DAS trip risk). Up = sonic, no-op
  when grounded.
- Down locks a grounded piece on the spot (frame-by-frame verified; the
  frozen timer in probes is the *result* of the lock, and the ~20f delay
  to the next spawn is just ARE). Playback ends every plan with one Down
  press — but never press-then-release in the same frameend (last set
  wins, which silently canceled the old finale).
- joypad.set uses the exact key names from joypad.get(1), controller 1,
  set explicitly every frame while a plan runs, released after.

## Board: NOT a main-RAM grid — read the sprite list instead

Exhaustive template search (byte/cell rowTD/rowBU/col, strides 10–40,
2-byte cells, 16/32-bit row/col bitmasks) finds no playfield array in the
1 MB main RAM. The authoritative visible board is the Psikyo sprite list.

Video domain: `sh2 : ram : 0x4000000-0x400FFFF` (sprite RAM, 64 KB).
Draw list: u16 BE entries at byte `0x3800`, ≤0x400 entries, bit14 = end.
`sprnum = (entry & 0x3FF) * 4` → descriptor of 4×u32 BE at byte `sprnum*4`:
- w0: `ypos = sext((w0>>16)&0x3FF,10)`, `xpos = sext(w0&0x3FF,10)`
- w1: `wide = ((w1>>8)&0xF)+1`, `high = ((w1>>24)&0xF)+1`
- w2: `tnum = w2&0x7FFFF`, `colr = (w2>>24)&0xFF`

Playfield cells: `wide == 1 and high == 1`, `0x18200 <= tnum <= 0x182FF`,
`x = 18 + 8*c`, `y = 41 + 8*r` for column c 0–9, row r 0–19 (r top-first).
Verified 28/28 locked cells against a hand-recorded Death stack, plus the
4-cell active I piece at (col 6, rows 14–17) → x=66, y=153/161/169/177.

`colr` = piece/state (all verified except Z-locked, unseen so far):

| colr | meaning |
|---|---|
| 0x3F | locked I |
| 0x5D | locked J |
| 0x67 | locked L |
| 0x71 | locked O |
| 0x53 | locked S |
| 0x7B | locked T |
| 0x49 | locked Z (from `tgm_dom_test_*_8186.bin`: 4 cells in Z shape) |
| 0x3E, 0x42 | ACTIVE piece minos (double-drawn, 4 cells × 2) |
| 0x43, 0x57 | NEXT-preview minos (up by y≈15–23) |
| others (0x02/03/08/09/9F/…) | UI text, ignore (off-grid anyway) |

Reader rule: on-grid 1×1 `0x182xx` sprites whose colr is in the locked set
(or: not in the active/preview/UI set, to tolerate Z) → filled. Read during
ARE (timer frozen at 30, no active piece) so no exclusion needed. Dedupe by
cell — lock-flash may double-draw.

## Still open

- Z-locked `colr` (no Z locked in any dump yet).
- P1 Button 1/2/3 → TGM A (CCW) / B (CW) / C (180°) mapping; Up = sonic.
- `0x64C2D` (4,4,5 pattern) — minor counter, not needed.
- Levels ≥100 timings (DAS 8–12, ARE 4–14) only matter for input scheduling.
