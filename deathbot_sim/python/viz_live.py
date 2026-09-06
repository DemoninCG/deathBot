"""Live pygame visualizer for the deathbot network playing TAP Death mode.

Two panels in one window:
  - LEFT (live game): the network continuously plays full Death-mode games.
    Shows the current board (white-on-black) with the ghost piece (where the
    network will place the current piece, in dim blue), the next-piece preview,
    and live stats: current level, pieces placed, games played, max level
    reached, max/avg survival. When a game tops out, a new one starts after a
    brief pause.
  - RIGHT (testing): cycles through all reachable lock positions for a
    hard-coded board + next piece, drawing each candidate in light blue on
    a white-on-black board. Below it, a separate board shows the network's
    predicted best placement, with the top-5 scores listed.

Run:
    python viz_live.py --nn --weights model_weights.pt
    python viz_live.py --nn --weights ../../model_weights_snapshot9.pt --two-ply
    python viz_live.py --nn --weights ../../model_weights_snapshot9.pt --two-ply --line-bonus 1,3,5,8
    python viz_live.py --piece T --board stack
    python viz_live.py --help

Controls:
    SPACE  - pause/resume the live game + lock cycling
    LEFT / RIGHT - manually step through candidates (testing panel) when paused
    R      - toggle reduced-motion mode (slower cycling + game speed)
    N      - toggle between MockScorer and NeuralScorer (if torch installed)
    G      - force-start a new game immediately
    ESC    - quit
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import pygame

import deathbot_sim as ds
from scorer import MockNetwork, Scorer, extract_features
from piece_history import PieceQueue

# Lazy-import torch + the NN so the visualizer still works without torch.
_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    from neural_scorer import NeuralScorer
    from model import TAPNet
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Make viz_pathfind importable for the board parser.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from viz_pathfind import _parse_board  # noqa: E402


# ---- Constants ------------------------------------------------------------

WINDOW_W = 1280
WINDOW_H = 720
PANEL_MARGIN = 16
GAME_PANEL_W = 480
TEST_PANEL_W = WINDOW_W - GAME_PANEL_W - PANEL_MARGIN * 3

# Colors (RGB).
BG = (8, 8, 12)
PANEL_BG = (16, 16, 24)
GRID = (40, 40, 50)
TEXT = (220, 220, 230)
TEXT_DIM = (140, 140, 150)
LOCKED = (200, 200, 200)
ACTIVE = (120, 200, 255)
BEST = (120, 255, 180)
GHOST = (60, 120, 160)
STAT_GOOD = (120, 255, 180)
STAT_DIM = (140, 140, 150)
SCORE_BAR = (120, 200, 255)
PIECE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (120, 200, 255),  # I - cyan
    1: (255, 220, 80),   # O - yellow
    2: (200, 120, 255),  # T - purple
    3: (120, 255, 120),  # S - green
    4: (255, 120, 120),  # Z - red
    5: (120, 120, 255),  # J - blue
    6: (255, 180, 80),   # L - orange
}

PIECE_KINDS = {
    "I": ds.PieceType.I, "O": ds.PieceType.O, "T": ds.PieceType.T,
    "S": ds.PieceType.S, "Z": ds.PieceType.Z, "J": ds.PieceType.J,
    "L": ds.PieceType.L,
}
ALL_PIECES = list(PIECE_KINDS.values())

_PIECE_INDEX = {
    ds.PieceType.I: 0, ds.PieceType.O: 1, ds.PieceType.T: 2,
    ds.PieceType.S: 3, ds.PieceType.Z: 4, ds.PieceType.J: 5,
    ds.PieceType.L: 6,
}


# ---- Game runner (continuous play) ---------------------------------------

class TwoPlyVizScorer:
    """Scorer-protocol adapter around full-expansion 2-ply lookahead.

    Implements the same `score`/`score_batch` interface as NeuralScorer so
    GameRunner works unchanged: `score_batch` returns the 2-ply value
    P2(s,a) = v(afterstate) + gamma * max_b v(next-afterstates) for every
    candidate, with dying branches at -1e6 and lockout branches at -inf
    (masked out of selection by GameRunner's argmax only when a survivor
    exists). Full expansion (no top-k filter) — this is the validated
    snapshot9+2-ply configuration, not the truncated beams.
    """

    def __init__(self, net, *, gamma_ply: float = 0.999,
                 line_bonus: dict | None = None) -> None:
        from two_ply import TwoPlyScorer, DEAD_PENALTY
        self._two_ply = TwoPlyScorer(net, beam=10**9, gamma_ply=gamma_ply,
                                     line_bonus=line_bonus)
        self._dead = DEAD_PENALTY
        self.net = net

    def score(self, board, candidate) -> float:
        return self.score_batch(board, [candidate])[0]

    def score_batch(self, board, candidates, current_piece_idx=None,
                    next_piece_idx=None) -> list[float]:
        import numpy as np
        if not candidates:
            return []
        two_ply = self._two_ply
        # 1-ply values + successors for all viable candidates.
        kind = candidates[0].piece.kind
        placed = two_ply._placements(board, kind)
        succs1, vals1 = placed[0], placed[1]
        lines1 = placed[2] if len(placed) > 3 else [0] * len(vals1)
        if not vals1:
            return [float("-inf")] * len(candidates)
        # Map viable-list position -> candidate index (lockouts excluded
        # from _placements; search_locks order matches candidates order).
        viable_pos = [i for i, c in enumerate(candidates)
                      if not board.locks_out(c.piece)]
        # Next-piece kind for the 2nd ply: prefer the explicit index the
        # runner passes; fall back to the candidate kind approximation.
        next_kind = self._kind_from_idx(next_piece_idx, fallback=kind)
        out = [float("-inf")] * len(candidates)
        for local, cand_idx in enumerate(viable_pos):
            placed2 = two_ply._placements(succs1[local], next_kind)
            succs2, vals2 = placed2[0], placed2[1]
            if not vals2:
                out[cand_idx] = self._dead
            else:
                bonus = float(two_ply.line_bonus.get(int(lines1[local]), 0.0))
                out[cand_idx] = (vals1[local] + bonus
                                 + two_ply.gamma_ply * max(vals2))
        return out

    @staticmethod
    def _kind_from_idx(idx, *, fallback):
        import deathbot_sim as _ds
        if idx is None:
            return fallback
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                return fallback
        order = [_ds.PieceType.I, _ds.PieceType.O, _ds.PieceType.T,
                 _ds.PieceType.S, _ds.PieceType.Z, _ds.PieceType.J,
                 _ds.PieceType.L]
        if 0 <= idx < len(order):
            return order[idx]
        return fallback


class GameRunner:
    """Plays one Death-mode game step at a time using a scorer."""

    def __init__(self, scorer: Scorer, rng: random.Random | None = None,
                 *, epsilon: float = 0.0) -> None:
        self.scorer = scorer
        self.rng = rng or random.Random()
        self.epsilon = epsilon
        self.piece_queue = PieceQueue(self.rng)
        self.reset()

    def reset(self) -> None:
        self.board = ds.Board()
        self.current_piece: Optional[ds.PieceType] = None
        self.next_piece: Optional[ds.PieceType] = None
        self.current_candidates: list = []
        self.current_scores: list[float] = []
        self.best_idx: int = 0
        self.pieces_placed: int = 0
        self.lines_cleared: int = 0
        self.alive: bool = True
        self.piece_queue.reset()
        self._spawn_new_piece()

    def _spawn_new_piece(self) -> None:
        # Advance the queue: the old next-piece becomes the new current piece.
        self.current_piece = self.piece_queue.current
        self.next_piece = self.piece_queue.next_piece
        self.current_candidates = self.board.search_locks(self.current_piece)
        if not self.current_candidates:
            self.alive = False
            self.current_scores = []
            self.best_idx = 0
            return
        cur_idx = _PIECE_INDEX.get(self.current_piece, 0)
        nxt_idx = _PIECE_INDEX.get(self.next_piece, 0)
        if hasattr(self.scorer, "score_batch"):
            self.current_scores = self.scorer.score_batch(
                self.board, self.current_candidates,
                current_piece_idx=cur_idx, next_piece_idx=nxt_idx)
        else:
            self.current_scores = [
                self.scorer.score(self.board, c) for c in self.current_candidates]
        self.best_idx = max(range(len(self.current_scores)),
                            key=lambda i: self.current_scores[i])

    def step(self) -> Optional["ds.LockCandidate"]:
        if not self.alive or not self.current_candidates:
            self.alive = False
            return None
        if self.rng.random() < self.epsilon:
            cand = self.rng.choice(self.current_candidates)
        else:
            cand = self.current_candidates[self.best_idx]
        self.board.lock(cand.piece)
        lines = self.board.clear_lines()
        self.lines_cleared += lines
        self.pieces_placed += 1
        self.piece_queue.advance()  # next-piece becomes current
        self._spawn_new_piece()
        return cand

    @property
    def level(self) -> int:
        """TAP level with level-stop: level increases by 1 per piece placed
        and per line cleared, but PAUSES at every 100 boundary (99, 199, 299,
        ...) until a line is cleared. Once a line is cleared at a stop, the
        accumulated pieces advance the level past the boundary.

        TAP Death starts at level 500; we display the offset from 500 for
        readability (so the first piece is level 501). Actually TAP Death
        starts at 0 and the internal level is 500, but for visualization we
        just show the raw count with the stop logic.
        """
        raw_level = self.pieces_placed + self.lines_cleared
        # Find the last level-stop boundary we've crossed.
        # Level stops at 99, 199, 299, ... (every 100 levels, 0-indexed 99).
        # The level cannot advance past 99, 199, etc. until a line is cleared.
        # We track whether we've cleared a line since the last stop.
        #
        # Simplified: if we haven't cleared a line since hitting the boundary,
        # cap at the boundary. We approximate this by checking if lines_cleared
        # is enough to "unlock" each boundary we've passed.
        #
        # Each boundary at B=99,199,299,... requires at least 1 line clear
        # to pass. So the max boundaries we can pass = lines_cleared.
        # The effective level is:
        #   raw_level, but capped at the (lines_cleared + 1)-th boundary.
        if raw_level < 100:
            return raw_level
        # Number of 100-boundaries we've hit: floor(raw_level / 100)
        # But each boundary past the first requires a line clear to unlock.
        # Boundaries we're allowed to pass = lines_cleared (each line unlocks
        # one boundary). Actually TAP is: at each x99, you need a line clear
        # to proceed. So the number of boundaries you can cross = lines_cleared
        # (assuming you cleared a line at each). If you haven't cleared enough
        # lines, you're stuck at the last boundary.
        boundaries_hit = raw_level // 100  # how many x99 boundaries we'd cross
        boundaries_unlocked = self.lines_cleared  # each line clear unlocks one
        # If we've hit more boundaries than we've unlocked, we're stuck at
        # the last unlocked boundary (which is x99).
        if boundaries_hit > boundaries_unlocked:
            # Stuck at the (boundaries_unlocked)-th boundary.
            # The boundary value is boundaries_unlocked * 100 - 1 (x99).
            return boundaries_unlocked * 100 + 99
        return raw_level

    @property
    def survival(self) -> int:
        return self.pieces_placed


class GameStats:
    """Tracks aggregate stats across multiple games."""

    def __init__(self) -> None:
        self.games_played: int = 0
        self.max_level: int = 0
        self.max_survival: int = 0
        self.max_lines: int = 0
        self.total_pieces: int = 0
        self.total_lines: int = 0
        self.recent_survivals: list[int] = []

    def record_game(self, runner: GameRunner) -> None:
        self.games_played += 1
        self.total_pieces += runner.pieces_placed
        self.total_lines += runner.lines_cleared
        if runner.level > self.max_level:
            self.max_level = runner.level
        if runner.survival > self.max_survival:
            self.max_survival = runner.survival
        if runner.lines_cleared > self.max_lines:
            self.max_lines = runner.lines_cleared
        self.recent_survivals.append(runner.survival)
        if len(self.recent_survivals) > 20:
            self.recent_survivals.pop(0)

    @property
    def avg_survival(self) -> float:
        if not self.recent_survivals:
            return 0.0
        return sum(self.recent_survivals) / len(self.recent_survivals)


# ---- Live game panel ------------------------------------------------------

def draw_game_panel(
    surface: pygame.Surface,
    runner: GameRunner,
    stats: GameStats,
    scorer_name: str,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    rect: pygame.Rect,
    game_step_timer: float,
    game_step_interval: float,
) -> None:
    pygame.draw.rect(surface, PANEL_BG, rect)
    pygame.draw.rect(surface, GRID, rect, 1)

    title = font.render("Live game (network play)", True, (220, 220, 230))
    surface.blit(title, (rect.x + 8, rect.y + 6))
    status = "PLAYING" if runner.alive else "GAME OVER"
    status_color = STAT_GOOD if runner.alive else (255, 100, 100)
    status_txt = small_font.render(status, True, status_color)
    surface.blit(status_txt, (rect.right - status_txt.get_width() - 8, rect.y + 10))

    cell_px = 20
    board_x = rect.x + 16
    board_y = rect.y + 36
    bw = runner.board.width * cell_px
    bh = runner.board.visible_rows * cell_px
    board_rect = pygame.Rect(board_x, board_y, bw, bh)

    # Ghost piece (where the network will place the current piece).
    ghost_cells: set[tuple[int, int]] = set()
    if runner.alive and runner.current_candidates:
        best_cand = runner.current_candidates[runner.best_idx]
        ghost_cells = {(int(c), int(r)) for c, r in best_cand.piece.cells()}

    pygame.draw.rect(surface, BG, board_rect)
    for r in range(runner.board.visible_rows):
        for c in range(runner.board.width):
            cx = board_x + c * cell_px
            cy = board_y + (runner.board.visible_rows - 1 - r) * cell_px
            crect = pygame.Rect(cx, cy, cell_px, cell_px)
            if (c, r) in ghost_cells:
                pygame.draw.rect(surface, GHOST, crect)
            elif runner.board.cell(c, r):
                pygame.draw.rect(surface, LOCKED, crect)
            else:
                pygame.draw.rect(surface, BG, crect)
            pygame.draw.rect(surface, GRID, crect, 1)

    # Next-piece preview.
    next_x = board_x + bw + 12
    next_y = board_y
    surface.blit(small_font.render("NEXT", True, TEXT_DIM), (next_x, next_y))
    if runner.next_piece is not None:
        next_idx = _PIECE_INDEX.get(runner.next_piece, 0)
        next_color = PIECE_COLORS.get(next_idx, ACTIVE)
        mini_cell = 14
        mini_x = next_x
        mini_y = next_y + 18
        p = ds.Piece(runner.next_piece, 0, 0, 0)
        cells = {(int(c), int(r)) for c, r in p.cells()}
        if cells:
            min_c = min(c for c, _ in cells)
            min_r = min(r for _, r in cells)
            max_c = max(c for c, _ in cells)
            max_r = max(r for _, r in cells)
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    cx = mini_x + (c - min_c) * mini_cell
                    cy = mini_y + (max_r - r) * mini_cell
                    crect = pygame.Rect(cx, cy, mini_cell, mini_cell)
                    if (c, r) in cells:
                        pygame.draw.rect(surface, next_color, crect)
                    else:
                        pygame.draw.rect(surface, BG, crect)
                    pygame.draw.rect(surface, GRID, crect, 1)

    # Stats.
    stats_y = board_y + bh + 12
    stats_x = board_x

    def stat_line(label: str, value: str, y: int, color=TEXT) -> None:
        surface.blit(small_font.render(f"{label}:", True, TEXT_DIM), (stats_x, y))
        surface.blit(small_font.render(value, True, color), (stats_x + 80, y))

    stat_line("level", f"{runner.level}", stats_y, STAT_GOOD)
    stat_line("pieces", f"{runner.pieces_placed}", stats_y + 16)
    stat_line("lines", f"{runner.lines_cleared}", stats_y + 32)
    if runner.pieces_placed:
        stat_line("l/100", f"{100.0 * runner.lines_cleared / runner.pieces_placed:.1f}",
                  stats_y + 48)

    agg_x = stats_x + 180

    def agg_line(label: str, value: str, y: int, color=TEXT) -> None:
        surface.blit(small_font.render(f"{label}:", True, TEXT_DIM), (agg_x, y))
        surface.blit(small_font.render(value, True, color), (agg_x + 90, y))

    agg_line("games", f"{stats.games_played}", stats_y)
    agg_line("max level", f"{stats.max_level}", stats_y + 16, STAT_GOOD)
    agg_line("max survive", f"{stats.max_survival}", stats_y + 32, STAT_GOOD)
    agg_line("avg survive", f"{stats.avg_survival:.1f}", stats_y + 48)
    agg_line("max lines", f"{stats.max_lines}", stats_y + 64)

    # Recent survivals mini bar chart.
    chart_y = stats_y + 90
    surface.blit(small_font.render("recent survivals:", True, TEXT_DIM),
                 (stats_x, chart_y))
    bar_x = stats_x + 110
    bar_max_h = 30
    max_s = max(stats.recent_survivals) if stats.recent_survivals else 1
    for i, s in enumerate(stats.recent_survivals):
        h = int(bar_max_h * s / max(1, max_s))
        bx = bar_x + i * 6
        by = chart_y + 16 + (bar_max_h - h)
        color = STAT_GOOD if s == max_s else STAT_DIM
        pygame.draw.rect(surface, color, (bx, by, 5, h))

    # Game step progress bar.
    if runner.alive:
        progress = min(1.0, game_step_timer / game_step_interval)
        bar_w = int(bw * progress)
        pygame.draw.rect(surface, (30, 60, 90),
                         (board_x, board_y + bh + 2, bw, 3))
        pygame.draw.rect(surface, ACTIVE,
                         (board_x, board_y + bh + 2, bar_w, 3))


# ---- Testing panel (kept from before) ------------------------------------

def board_rect(x: int, y: int, cell_px: int, board: "ds.Board") -> pygame.Rect:
    return pygame.Rect(x, y, board.width * cell_px, board.visible_rows * cell_px)


def draw_board(surface, board, piece, rect, cell_px, piece_color=ACTIVE, *,
               show_invisible=False) -> None:
    pygame.draw.rect(surface, BG, rect)
    rows = board.visible_rows if not show_invisible else board.height
    pc = {(int(c), int(r)) for c, r in piece.cells()} if piece else set()
    for r in range(rows):
        for c in range(board.width):
            cx = rect.x + c * cell_px
            cy = rect.y + (rows - 1 - r) * cell_px
            crect = pygame.Rect(cx, cy, cell_px, cell_px)
            if (c, r) in pc:
                pygame.draw.rect(surface, piece_color, crect)
            elif board.cell(c, r):
                pygame.draw.rect(surface, LOCKED, crect)
            else:
                pygame.draw.rect(surface, BG, crect)
            pygame.draw.rect(surface, GRID, crect, 1)


def draw_testing_panel(surface, board, candidates, scores, current_idx, best_idx,
                       font, small_font, rect, net,
                       candidate_lines: list[int] | None = None) -> None:
    pygame.draw.rect(surface, PANEL_BG, rect)
    pygame.draw.rect(surface, GRID, rect, 1)
    surface.blit(font.render("Testing (lock candidates)", True, TEXT),
                 (rect.x + 8, rect.y + 6))

    cell_px = 18
    left_x = rect.x + 16
    left_y = rect.y + 36
    left_rect = board_rect(left_x, left_y, cell_px, board)
    cur_piece = candidates[current_idx].piece if candidates else None
    draw_board(surface, board, cur_piece, left_rect, cell_px, ACTIVE)
    surface.blit(small_font.render(
        f"cycling ({current_idx + 1}/{len(candidates)})", True, TEXT_DIM),
        (left_x, left_y + left_rect.height + 4))
    if cur_piece and candidates:
        surface.blit(small_font.render(
            f"inputs: {candidates[current_idx].inputs_str()}", True, TEXT_DIM),
            (left_x, left_y + left_rect.height + 20))

    right_x = left_x + left_rect.width + 32
    right_y = left_y
    right_rect = board_rect(right_x, right_y, cell_px, board)
    best_piece = candidates[best_idx].piece if candidates else None
    draw_board(surface, board, best_piece, right_rect, cell_px, BEST)
    surface.blit(small_font.render("network's pick (argmax)", True, TEXT_DIM),
                 (right_x, right_y + right_rect.height + 4))
    if best_piece and candidates:
        surface.blit(small_font.render(
            f"inputs: {candidates[best_idx].inputs_str()}", True, TEXT_DIM),
            (right_x, right_y + right_rect.height + 20))

    scores_x = right_x + right_rect.width + 24
    scores_y = right_y
    surface.blit(small_font.render("top scores", True, TEXT_DIM),
                 (scores_x, scores_y))
    ranked = sorted(enumerate(scores), key=lambda kv: -kv[1])[:5]
    for i, (orig_idx, sc) in enumerate(ranked):
        y = scores_y + 20 + i * 18
        color = BEST if orig_idx == best_idx else TEXT
        c = candidates[orig_idx]
        line_tag = ""
        if candidate_lines is not None and orig_idx < len(candidate_lines):
            n = int(candidate_lines[orig_idx])
            if n == 4:
                line_tag = "  TETRIS"
                color = (255, 220, 80)
            elif n > 0:
                line_tag = f"  +{n}"
        surface.blit(small_font.render(
            f"#{i+1}  s={sc:+.3f}  st={c.piece.state} x={c.piece.x}{line_tag}", True, color),
            (scores_x, y))
        bar_w = max(0, min(120, int(sc * 40 + 60)))
        pygame.draw.rect(surface, SCORE_BAR, (scores_x + 180, y + 4, bar_w, 8))

    if candidates and hasattr(net, "score"):
        feats = extract_features(board, candidates[current_idx])
        feats_y = scores_y + 140
        surface.blit(small_font.render("features (current):", True, TEXT_DIM),
                     (scores_x, feats_y))
        names = ["agg_h", "holes", "bump", "lines"]
        for i, (n, v) in enumerate(zip(names, feats)):
            y = feats_y + 18 + i * 16
            surface.blit(small_font.render(f"  {n} = {v:.1f}", True, TEXT_DIM),
                         (scores_x, y))


# ---- Demo boards + scorer factory ----------------------------------------

DEMO_BOARDS = {
    "stack": "........../........../........../........../........../.##.#...../.######.../#######.../######..../#########.",
    "empty": "........../........../........../........../........../........../........../........../........../..........",
}


def make_scorer(use_nn: bool, weights_path: str | None = None,
                 two_ply: bool = False, gamma_ply: float = 0.999,
                 line_bonus: dict | None = None):
    if use_nn:
        if not _TORCH_AVAILABLE:
            print("torch not available; falling back to MockScorer")
            return MockNetwork()
        net = TAPNet()
        if weights_path is not None:
            p = Path(weights_path)
            if not p.is_absolute():
                p = HERE.parent / weights_path
            if p.exists():
                net.load_state_dict(torch.load(p, map_location="cpu"))
                print(f"loaded NN weights from {p}")
            else:
                print(f"weights file {p} not found; using untrained net")
        net.eval()
        if two_ply:
            print(f"using full-expansion 2-ply lookahead (gamma={gamma_ply}, "
                  f"line_bonus={line_bonus})")
            return TwoPlyVizScorer(net, gamma_ply=gamma_ply,
                                   line_bonus=line_bonus)
        return NeuralScorer(net=net)
    return MockNetwork()


# ---- Main loop ------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    piece_name = "L"
    board_name = "stack"
    use_nn = False
    weights_path: str | None = None
    use_two_ply = False
    gamma_ply = 0.999
    line_bonus: dict | None = None
    for i, a in enumerate(args):
        if a == "--piece" and i + 1 < len(args):
            piece_name = args[i + 1].upper()
        elif a == "--board" and i + 1 < len(args):
            board_name = args[i + 1]
        elif a.startswith("--board="):
            board_name = a.split("=", 1)[1]
        elif a == "--nn":
            use_nn = True
        elif a == "--weights" and i + 1 < len(args):
            weights_path = args[i + 1]
        elif a.startswith("--weights="):
            weights_path = a.split("=", 1)[1]
        elif a == "--two-ply":
            use_two_ply = True
        elif a == "--gamma-ply" and i + 1 < len(args):
            gamma_ply = float(args[i + 1])
        elif a.startswith("--gamma-ply="):
            gamma_ply = float(a.split("=", 1)[1])
        elif a == "--line-bonus" and i + 1 < len(args):
            # "1,3,5,8" -> {1:1.0, 2:3.0, 3:5.0, 4:8.0} (z-units)
            vals = [float(x) for x in args[i + 1].split(",")]
            line_bonus = {n + 1: v for n, v in enumerate(vals)}
        elif a.startswith("--line-bonus="):
            vals = [float(x) for x in a.split("=", 1)[1].split(",")]
            line_bonus = {n + 1: v for n, v in enumerate(vals)}
    if piece_name not in PIECE_KINDS:
        print(f"unknown piece {piece_name!r}; choose from {list(PIECE_KINDS)}")
        sys.exit(1)

    if board_name in DEMO_BOARDS:
        test_board = _parse_board(DEMO_BOARDS[board_name])
    else:
        test_board = _parse_board(board_name)
    test_kind = PIECE_KINDS[piece_name]

    scorer = make_scorer(use_nn, weights_path, two_ply=use_two_ply,
                         gamma_ply=gamma_ply, line_bonus=line_bonus)
    scorer_name = ("TwoPly(snapshot9)" if (use_nn and use_two_ply)
                   else "NeuralScorer" if use_nn else "MockScorer")

    rng = random.Random(42)
    runner = GameRunner(scorer, rng, epsilon=0.0)
    stats = GameStats()

    test_candidates = test_board.search_locks(test_kind)
    test_cur_idx = _PIECE_INDEX.get(test_kind, 0)
    if hasattr(scorer, "score_batch") and test_candidates:
        test_scores = scorer.score_batch(test_board, test_candidates,
                                          current_piece_idx=test_cur_idx)
    else:
        test_scores = [scorer.score(test_board, c) for c in test_candidates]
    try:
        _res = test_board.lock_candidates_full(test_kind)
        test_lines = [int(x) for x in _res[5]]
        if len(test_lines) != len(test_candidates):
            test_lines = [0] * len(test_candidates)
    except Exception:
        test_lines = [0] * len(test_candidates)
    test_best_idx = (max(range(len(test_scores)), key=lambda i: test_scores[i])
                     if test_scores else 0)
    test_current_idx = 0

    pygame.init()
    pygame.display.set_caption("deathbot — live game + testing viz")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 12)

    game_rect = pygame.Rect(PANEL_MARGIN, PANEL_MARGIN, GAME_PANEL_W,
                            WINDOW_H - PANEL_MARGIN * 2)
    test_rect = pygame.Rect(PANEL_MARGIN * 2 + GAME_PANEL_W, PANEL_MARGIN,
                            TEST_PANEL_W, WINDOW_H - PANEL_MARGIN * 2)

    paused = False
    reduced_motion = False
    game_step_timer = 0.0
    game_step_interval = 0.1
    game_over_pause = 0.0
    game_over_interval = 1.5
    test_cycle_speed = 8.0
    test_time_accum = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_LEFT:
                    if test_candidates:
                        test_current_idx = (test_current_idx - 1) % len(test_candidates)
                elif event.key == pygame.K_RIGHT:
                    if test_candidates:
                        test_current_idx = (test_current_idx + 1) % len(test_candidates)
                elif event.key == pygame.K_r:
                    reduced_motion = not reduced_motion
                elif event.key == pygame.K_n:
                    use_nn = not use_nn
                    scorer = make_scorer(use_nn, weights_path,
                                         two_ply=use_two_ply,
                                         gamma_ply=gamma_ply,
                                         line_bonus=line_bonus)
                    scorer_name = ("TwoPly(snapshot9)"
                                   if (use_nn and use_two_ply)
                                   else "NeuralScorer" if use_nn
                                   else "MockScorer")
                    runner.scorer = scorer
                    if hasattr(scorer, "score_batch") and test_candidates:
                        test_scores = scorer.score_batch(
                            test_board, test_candidates,
                            current_piece_idx=test_cur_idx)
                    else:
                        test_scores = [scorer.score(test_board, c) for c in test_candidates]
                    test_best_idx = (max(range(len(test_scores)),
                                         key=lambda i: test_scores[i])
                                     if test_scores else 0)
                elif event.key == pygame.K_g:
                    if not runner.alive:
                        stats.record_game(runner)
                    runner.reset()
                    game_over_pause = 0.0

        # Auto-cycle testing panel.
        if not paused and test_candidates:
            speed = test_cycle_speed * (0.4 if reduced_motion else 1.0)
            test_time_accum += dt * speed
            if test_time_accum >= 1.0:
                steps = int(test_time_accum)
                test_current_idx = (test_current_idx + steps) % len(test_candidates)
                test_time_accum -= steps

        # Advance the live game.
        if not paused:
            if runner.alive:
                interval = game_step_interval * (2.0 if reduced_motion else 1.0)
                game_step_timer += dt
                if game_step_timer >= interval:
                    runner.step()
                    game_step_timer = 0.0
            else:
                game_over_pause += dt
                if game_over_pause >= game_over_interval:
                    stats.record_game(runner)
                    runner.reset()
                    game_over_pause = 0.0
                    game_step_timer = 0.0

        # Render.
        screen.fill(BG)
        draw_game_panel(screen, runner, stats, scorer_name, font, small_font,
                        game_rect, game_step_timer, game_step_interval)
        draw_testing_panel(screen, test_board, test_candidates, test_scores,
                           test_current_idx, test_best_idx,
                           font, small_font, test_rect, scorer,
                           candidate_lines=test_lines)

        hint = small_font.render(
            f"{scorer_name} | SPACE pause | <-/-> test step | R slow | "
            f"N swap | G new game | ESC quit", True, TEXT_DIM)
        screen.blit(hint, (PANEL_MARGIN, WINDOW_H - 18))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
