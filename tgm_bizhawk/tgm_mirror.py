"""Pygame mirror of the live TGM2 board (video overlay).

Reads snapshots written by tgm_bizhawk/tgm_player.lua (BizHawk) and
draws the locked board + current/next pieces plus the bot's current
decision (top-3 moves, movement sequence, IRS). Viewer-facing only:
no diagnostic panels.

Run from the repo root:
    .venv/Scripts/python.exe tgm_bizhawk/tgm_mirror.py [--state PATH]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Perf: same single-thread pin as tgm_1ply. The mirror shares the box with
# the MAME thread; a 4-thread torch burst starves emulation (audio
# cutouts) and is slower per batch anyway. Must run before warmup.
try:
    import torch as _torch

    try:
        _torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        _torch.set_num_threads(1)
    except RuntimeError:
        pass
except ImportError:
    pass

import pygame

import tgm_1ply

# TGM piece IDs on the wire: I=0 Z=1 S=2 J=3 L=4 O=5 T=6.
ID2LETTER = ["I", "Z", "S", "J", "L", "O", "T"]


def _is_piece_id(v) -> bool:
    """Valid wire ID (0-6). At boot/menus the RAM bytes hold non-piece
    values, and Lua forwards whatever it reads — skip those states."""
    return isinstance(v, int) and 0 <= v < 7

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
    pygame.display.set_caption("DeathBot — live view")
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
        if not _is_piece_id(up):
            print(f"ROOT skip f={header.get('frame')}: bad id next={up!r}",
                  flush=True)
            return
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
        cur, nxt = header["cur"], header["next"]
        if not (_is_piece_id(cur) and _is_piece_id(nxt)):
            print(f"MOVES skip f={header.get('frame')}: "
                  f"bad ids cur={cur!r} next={nxt!r}", flush=True)
            return
        key = (tuple(rows), cur, nxt, header.get("held"))
        if key == brain_for:
            return
        brain_for = key
        held = header.get("held", "-")
        sim_name, nstates = tgm_1ply.sim_info(cur)
        root = tgm_1ply.root_for_hold(held, sim_name, nstates)
        for attempt in (root, 0) if root != 0 else (0,):
            brain = tgm_1ply.suggest(rows, cur, nxt, root=attempt)
            if brain.get("ok"):
                break
        ghost_surf.fill((0, 0, 0, 0))
        if brain.get("ok"):
            tint = COLORS.get(ID2LETTER[cur], (255, 255, 255))
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
    needs_draw = True
    last_draw = 0.0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        try:
            mtime = state_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if mtime == 0.0:
            if file_missing is not True or parse_err != "file not found":
                needs_draw = True
            file_missing = True
            parse_err = "file not found"
        elif mtime != last_mtime:
            last_mtime = mtime
            file_missing = False
            parsed_h, parsed_r, err = parse_state(state_path)
            if err is None:
                header, rows = parsed_h, parsed_r
                parse_err = None
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
                needs_draw = True
            else:
                if parse_err != err:
                    needs_draw = True
                parse_err = err
        # Perf: redraw only on new state/decisions (plus a 2Hz refresh so
        # a stale frame can't stick). Polling stays at 60Hz so move
        # latency is unaffected; only the blits are gated.
        now = time.time()
        if needs_draw or (now - last_draw) >= 0.5:
            last_draw = now
            needs_draw = False
            screen.fill(BG)
            draw_board(screen, rows, MARGIN, MARGIN)
            screen.blit(ghost_surf, (MARGIN, MARGIN))
            cur = ID2LETTER[header["cur"]] if _is_piece_id(header["cur"]) else "?"
            nxt = ID2LETTER[header["next"]] if _is_piece_id(header["next"]) else "?"
            if cur in SHAPES:
                draw_piece_box(screen, font, small, cur, PANEL_X, MARGIN, "CURRENT")
            if nxt in SHAPES:
                draw_piece_box(screen, font, small, nxt, PANEL_X,
                               MARGIN + 130, "NEXT")
            # --- viewer panel: top 3, move sequence, IRS only ---
            screen.blit(font.render("TOP MOVES", True, TEXT_DIM),
                        (BRAIN_X, MARGIN))
            by = MARGIN + 30
            if file_missing or parse_err is not None:
                screen.blit(small.render("waiting for emulator…", True, TEXT),
                            (BRAIN_X, by))
            elif brain is None or not brain.get("ok"):
                screen.blit(small.render("no move yet", True, TEXT),
                            (BRAIN_X, by))
                by += 22
                held = header.get("held", "-")
                irs_line = f"IRS: {held}" if held in ("CW", "CCW") else "IRS: none"
                screen.blit(small.render(irs_line, True, TEXT), (BRAIN_X, by + 8))
            else:
                for i, (s, st, x, _yy) in enumerate(brain["top3"][:3]):
                    screen.blit(small.render(f"{i + 1}. {s:.2f}  x={x}  r={st}",
                                             True, TEXT), (BRAIN_X, by))
                    by += 22
                by += 10
                screen.blit(font.render("MOVES", True, TEXT_DIM), (BRAIN_X, by))
                by += 28
                seq = [STEP_ABBR.get(stp, stp) for stp in brain["steps"]]
                if not seq:
                    screen.blit(small.render("—", True, TEXT), (BRAIN_X, by))
                    by += 22
                else:
                    for k in range(0, len(seq), 8):
                        screen.blit(small.render(" ".join(seq[k:k + 8]),
                                                 True, TEXT), (BRAIN_X, by))
                        by += 22
                by += 10
                held = header.get("held", "-")
                irs_line = f"IRS: {held}" if held in ("CW", "CCW") else "IRS: none"
                screen.blit(small.render(irs_line, True, TEXT), (BRAIN_X, by))
            pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
