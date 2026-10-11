"""Pygame mirror of the live TGM2 board (state-read test harness).

Reads snapshots written by tgm_bizhawk/tgm_player.lua (BizHawk) and
draws the locked board + current/next pieces. Redraws only when the state
file changes, so the cost here is ~zero; the point of this test is to
confirm the *emulator side* doesn't lag the game.

Run from the repo root:
    .venv/Scripts/python.exe tgm_bizhawk/tgm_mirror.py [--state PATH]

Default state path: tgm_bizhawk/tgm_state.txt. In practice EmuHawk's
working directory (where the Lua script writes) turned out to be
tgm_bizhawk/; if the Lua log shows the file landing elsewhere, pass --state.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pygame

import tgm_1ply

# TGM piece IDs on the wire: I=0 Z=1 S=2 J=3 L=4 O=5 T=6.
ID2LETTER = ["I", "Z", "S", "J", "L", "O", "T"]

# Arika piece colors as specified by the player (palette-verified hues:
# 0x49 green=Z, 0x53 pink=S, 0x5D dark-blue=J, 0x67 orange=L, plus I red,
# O yellow, T light blue).
COLORS = {
    "I": (235, 55, 55),      # red
    "O": (235, 205, 45),     # yellow
    "T": (105, 185, 240),    # light blue
    "L": (240, 145, 35),     # orange
    "J": (55, 85, 220),      # dark blue
    "S": (240, 130, 195),    # pink
    "Z": (65, 200, 95),      # green
    ".": (16, 16, 24),
}
STACK_GREY = (150, 150, 160)  # locked stack renders grey; ghost is colored

# Spawn-orientation minos for the side panels (4-wide boxes, top-down).
SHAPES = {
    "I": [(0, 1), (1, 1), (2, 1), (3, 1)],
    "O": [(1, 0), (2, 0), (1, 1), (2, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}

CELL = 26
STEP_ABBR = {"Left": "L", "Right": "R", "CW": "CW", "CCW": "CC"}
BOARD_W, BOARD_H = 10 * CELL, 20 * CELL
MARGIN = 16
PANEL_X = MARGIN * 2 + BOARD_W
PANEL_W = 4 * CELL + MARGIN
BRAIN_X = PANEL_X + PANEL_W + MARGIN
BRAIN_W = 300
WIN_W = BRAIN_X + BRAIN_W + MARGIN
WIN_H = MARGIN * 2 + BOARD_H
BG = (8, 8, 12)
GRID = (40, 40, 50)
TEXT = (220, 220, 230)
TEXT_DIM = (140, 140, 150)


# State-header grammar. Fields are whitespace-separated k=v pairs, except
# exec may itself contain spaces (Lua "play i/n", "hold Button N" —
# sanitized to underscores going forward, but old files must still read).
# exec is captured greedily up to the trailing held field; both are absent
# from the legacy 6-field writer (retired tgm_mirror_state.lua).
_HEADER_RE = re.compile(
    r"^frame=(\S+) cnt=(\S+) cur=(\S+) next=(\S+) tmr=(\S+) "
    r"why=(\S+)(?: exec=(.*) held=(\S+))?\s*$")


def parse_state(path: Path):
    """Return (header_dict, rows, error_str). error_str is None on success."""
    try:
        text = path.read_text()
    except OSError as exc:
        return None, None, f"unreadable: {exc}"
    lines = text.splitlines()
    if len(lines) != 21 or not lines[0].startswith("frame="):
        return None, None, f"bad shape: {len(lines)} lines"
    m = _HEADER_RE.match(lines[0])
    if m is None:
        return None, None, f"bad header: {lines[0][:60]!r}"
    header = {}
    for k, v in zip(("frame", "cnt", "cur", "next", "tmr", "why", "exec",
                      "held"), m.groups()):
        if v is None:
            continue
        header[k] = int(v) if v.lstrip("-").isdigit() else v
    rows = lines[1:]
    if any(len(r) != 10 or any(ch not in "IZSJLOT." for ch in r) for r in rows):
        return None, None, "bad grid rows"
    return header, rows, None


def draw_board(surf, rows, ox, oy):
    for r in range(20):
        for c in range(10):
            rect = pygame.Rect(ox + c * CELL + 1, oy + r * CELL + 1,
                               CELL - 2, CELL - 2)
            if rows[r][c] == ".":
                pygame.draw.rect(surf, COLORS["."], rect, border_radius=3)
            else:
                pygame.draw.rect(surf, STACK_GREY, rect, border_radius=3)
    # grid lines
    for r in range(21):
        pygame.draw.line(surf, GRID, (ox, oy + r * CELL),
                         (ox + BOARD_W, oy + r * CELL))
    for c in range(11):
        pygame.draw.line(surf, GRID, (ox + c * CELL, oy),
                         (ox + c * CELL, oy + BOARD_H))


def draw_piece_box(surf, font, small, letter, ox, oy, label):
    surf.blit(font.render(label, True, TEXT_DIM), (ox, oy))
    for (x, y) in SHAPES[letter]:
        rect = pygame.Rect(ox + x * CELL + 1, oy + 24 + y * CELL + 1,
                           CELL - 2, CELL - 2)
        pygame.draw.rect(surf, COLORS[letter], rect, border_radius=3)
    surf.blit(small.render(letter, True, TEXT), (ox + 4 * CELL + 6, oy + 24))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="tgm_bizhawk/tgm_state.txt",
                      help="snapshot file written by tgm_player.lua "
                           "(it lands next to the Lua working dir; the Lua "
                           "console log confirms where)")
    ap.add_argument("--moves", default="tgm_bizhawk/tgm_moves.txt",
                      help="move file this mirror writes for "
                           "tgm_player.lua (same directory as --state)")
    args = ap.parse_args()
    state_path = Path(args.state)
    moves_path = Path(args.moves)

    pygame.init()
    pygame.display.set_caption("TGM2 mirror v3 (1-ply + moves)")
    print("MIRROR: warming up net...", flush=True)
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    font = pygame.font.SysFont("consolas", 20)
    small = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()
    try:
        import time as _t
        _t0 = _t.perf_counter()
        tgm_1ply.ensure_net()
        _r = tgm_1ply.suggest([".........." ] * 20, 0, 0)
        print(f"MIRROR: warmup done in "
              f"{(_t.perf_counter() - _t0) * 1000:.0f}ms "
              f"(ok={_r.get('ok')})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"MIRROR: warmup FAILED ({exc}); first pieces will lag",
              flush=True)

    rows = ["." * 10] * 20
    header = {"frame": 0, "cnt": 0, "cur": 0, "next": 0, "tmr": 0}
    last_mtime = 0.0
    last_update = time.time()
    file_missing = True
    parse_err = "no file yet"
    brain = None       # latest tgm_1ply.suggest() dict
    brain_for = None   # (rows-tuple, cur, next) it was computed for
    ghost_surf = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
    moves_info = "moves: none yet"

    def _write_moves(payload: dict) -> None:
        """Atomic moves-file write; payload has phase/for_frame/hold/steps.

        Retries os.replace: on Windows it sporadically raises WinError 5
        (Lua holds the destination open during its per-frameend read), and
        a dropped write means a lost hold/plan for that piece.
        """
        tmp = moves_path.with_suffix(".tmp")
        data = (f"phase={payload['phase']}\n"
                f"for_frame={payload['for_frame']}\n"
                f"hold={payload['hold']}\n"
                f"steps={','.join(payload['steps'])}\n")
        for attempt in range(4):
            try:
                with open(tmp, "w") as fh:
                    fh.write(data)
                os.replace(tmp, moves_path)
                return
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(0.01)

    def _decide_root(header, rows, mtime) -> None:
        """Lock-time: pick the IRS root for the upcoming piece."""
        nonlocal moves_info
        up = header["next"]
        ans = tgm_1ply.choose_root(rows, up, up)
        if not ans.get("ok"):
            print(f"ROOT skip f={header.get('frame')}: "
                  f"{ans.get('reason', '?')}", flush=True)
            return
        try:
            _write_moves({"phase": "hold", "for_frame": header["frame"],
                          "hold": ans["hold"], "steps": []})
            moves_info = (f"root: f={header['frame']} hold={ans['hold']} "
                          f"(sim {ans['root']})")
            print(f"ROOT {moves_info} upcoming={up} "
                  f"lat={1000 * (time.time() - mtime):.0f}ms", flush=True)
        except OSError as exc:
            moves_info = f"root FAILED: {exc}"
            print(moves_info, flush=True)

    def _decide_play(header, rows, mtime) -> None:
        """Spawn-time: full placement constrained to the held root."""
        nonlocal brain, brain_for, moves_info
        key = (tuple(rows), header["cur"], header["next"], header.get("held"))
        if key == brain_for:
            return
        brain_for = key
        held = header.get("held", "-")
        sim_name, nstates = tgm_1ply.sim_info(header["cur"])
        root = tgm_1ply.root_for_hold(held, sim_name, nstates)
        for attempt in (root, 0) if root != 0 else (0,):
            brain = tgm_1ply.suggest(rows, header["cur"],
                                     header["next"], root=attempt)
            if brain.get("ok"):
                break
        ghost_surf.fill((0, 0, 0, 0))
        if brain.get("ok"):
            cur_letter = ID2LETTER[header["cur"]] \
                if 0 <= header["cur"] < 7 else None
            tint = COLORS.get(cur_letter, (255, 255, 255))
            for c, r in brain["ghost"]:
                rect = pygame.Rect(c * CELL + 1, r * CELL + 1,
                                   CELL - 2, CELL - 2)
                ghost_surf.fill(tint + (130,), rect)
        if brain.get("ok"):
            try:
                _write_moves({"phase": "play", "for_frame": header["frame"],
                              "hold": "-", "steps": brain["steps"]})
                moves_info = (
                    f"moves: f={header['frame']} root={brain.get('root')} "
                    f"{len(brain['steps'])} steps")
                print(f"MOVES wrote {moves_info} "
                      f"cur={header['cur']} next={header['next']} "
                      f"lat={1000 * (time.time() - mtime):.0f}ms", flush=True)
            except OSError as exc:
                moves_info = f"moves FAILED: {exc}"
                print(moves_info, flush=True)
        else:
            print(f"MOVES skip f={header.get('frame')}: "
                  f"brain {brain.get('reason', '?')}", flush=True)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        try:
            mtime = state_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if mtime == 0.0:
            file_missing = True
            parse_err = "file not found"
        elif mtime != last_mtime:
            last_mtime = mtime
            file_missing = False
            parsed_h, parsed_r, err = parse_state(state_path)
            if err is None:
                header, rows = parsed_h, parsed_r
                parse_err = None
                last_update = time.time()
                # Two-phase decisions. LOCK publishes choose the IRS ROOT for
                # the upcoming piece (cur=header.next, next approximated by
                # itself — the true next isn't drawn yet). SPAWN publishes
                # decide the full placement constrained to the root Lua
                # actually held (header.held, authoritative) with the true
                # next. Lock publishes refresh the board visual only.
                why = header.get("why")
                if why == "lock":
                    _decide_root(header, rows, mtime)
                elif why == "spawn":
                    _decide_play(header, rows, mtime)
            else:
                parse_err = err
        screen.fill(BG)
        draw_board(screen, rows, MARGIN, MARGIN)
        screen.blit(ghost_surf, (MARGIN, MARGIN))
        cur = ID2LETTER[header["cur"]] if 0 <= header["cur"] < 7 else "?"
        nxt = ID2LETTER[header["next"]] if 0 <= header["next"] < 7 else "?"
        if cur in SHAPES:
            draw_piece_box(screen, font, small, cur, PANEL_X, MARGIN, "CURRENT")
        if nxt in SHAPES:
            draw_piece_box(screen, font, small, nxt, PANEL_X,
                           MARGIN + 130, "NEXT")
        info = [
            f"pieces: {header['cnt']}",
            f"frame: {header['frame']}",
            f"bot: {header.get('exec', '?')}",
            f"updated {time.time() - last_update:.1f}s ago",
            f"watching: {state_path}",
        ]
        if file_missing or parse_err is not None:
            info.append(f"STATUS: {parse_err or 'missing'}")
            info.append("check --state path!")
        y = MARGIN + 270
        for line in info:
            screen.blit(small.render(line, True, TEXT_DIM), (PANEL_X, y))
            y += 22
        # --- 1-ply panel ---
        screen.blit(font.render("1-PLY snap12", True, TEXT_DIM),
                    (BRAIN_X, MARGIN))
        by = MARGIN + 30
        if brain is None:
            screen.blit(small.render("waiting for state…", True, TEXT_DIM),
                        (BRAIN_X, by))
        elif not brain.get("ok"):
            screen.blit(small.render("no suggestion:", True, TEXT_DIM),
                        (BRAIN_X, by))
            screen.blit(small.render(str(brain.get("reason", "?"))[:38],
                                     True, TEXT), (BRAIN_X, by + 22))
            screen.blit(small.render(str(brain.get("reason", "?"))[38:76],
                                     True, TEXT), (BRAIN_X, by + 44))
        else:
            blines = [
                f"best {brain['best_score']:.2f}",
                f"x={brain['best_x']} rot={brain['best_state']}",
                f"eval {brain['eval_ms']:.1f}ms N={brain['n']}",
                f"root={brain.get('root', '?')} "
                f"{len(brain['steps'])} steps",
                f"{' '.join(STEP_ABBR.get(s, s) for s in brain['steps'][:8])}",
                f"{moves_info}",
                "",
                "top3:",
            ]
            for s, st, x, yy in brain["top3"]:
                blines.append(f"{s:7.2f} x={x} r={st}")
            for line in blines:
                screen.blit(small.render(line, True, TEXT), (BRAIN_X, by))
                by += 22
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
