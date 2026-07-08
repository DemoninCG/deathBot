"""Live pygame visualizer for the deathbot training + testing process.

Two panels in one window:
  - LEFT (training): a colored visualization of the network's neurons and
    connections on a black background. Neuron color encodes activation
    (phase 2 will feed real activations; for now a mock pulse). Shows the
    layer sizes and a "training step" counter.
  - RIGHT (testing): cycles through all reachable lock positions for a
    hard-coded board + next piece, drawing each candidate in light blue on
    a white-on-black board. Below it, a separate board shows the network's
    predicted best placement, with the top-5 scores listed.

Run:
    python -m viz_live                 # default hard-coded board + I piece
    python -m viz_live --piece T       # different piece
    python -m viz_live --board SPEC    # custom board (see viz_pathfind._parse_board)

Controls:
    SPACE  - pause/resume the lock cycling
    LEFT / RIGHT - manually step through candidates when paused
    R      - toggle reduced-motion mode (slower cycling)
    ESC    - quit
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import pygame

import deathbot_sim as ds
from scorer import MockNetwork, Scorer, extract_features

# Make viz_pathfind importable for the board parser.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from viz_pathfind import _parse_board  # noqa: E402


# ---- Constants ------------------------------------------------------------

# Window layout.
WINDOW_W = 1280
WINDOW_H = 720
PANEL_MARGIN = 16
TRAIN_PANEL_W = 480  # left panel
TEST_PANEL_W = WINDOW_W - TRAIN_PANEL_W - PANEL_MARGIN * 3

# Colors (RGB).
BG = (8, 8, 12)           # near-black background
PANEL_BG = (16, 16, 24)
GRID = (40, 40, 50)
TEXT = (220, 220, 230)
TEXT_DIM = (140, 140, 150)
LOCKED = (200, 200, 200)  # white-ish for locked blocks
ACTIVE = (120, 200, 255)  # light blue for the active piece being cycled
BEST = (120, 255, 180)    # green for the network's pick
NEURON_LOW = (30, 60, 120)
NEURON_HIGH = (120, 200, 255)
NEURON_EDGE = (50, 80, 130)
SCORE_BAR = (120, 200, 255)

PIECE_KINDS = {
    "I": ds.PieceType.I,
    "O": ds.PieceType.O,
    "T": ds.PieceType.T,
    "S": ds.PieceType.S,
    "Z": ds.PieceType.Z,
    "J": ds.PieceType.J,
    "L": ds.PieceType.L,
}


# ---- Training panel (network viz) -----------------------------------------

def draw_network(
    surface: pygame.Surface,
    net: "MockNetwork | Scorer",
    step: int,
    font: pygame.font.Font,
    rect: pygame.Rect,
) -> None:
    """Draw the network's neurons and connections on a black background.

    `net` must have a `layer_sizes` attribute (list of ints) for the layout.
    Activations are a mock pulse for now; phase 2 will pass real activations.
    """
    pygame.draw.rect(surface, PANEL_BG, rect)
    pygame.draw.rect(surface, GRID, rect, 1)

    # Title.
    title = font.render("Network (training)", True, TEXT)
    surface.blit(title, (rect.x + 8, rect.y + 6))
    step_txt = font.render(f"step {step}", True, TEXT_DIM)
    surface.blit(step_txt, (rect.right - step_txt.get_width() - 8, rect.y + 6))

    # Layout.
    if not hasattr(net, "layer_sizes"):
        # Fallback: draw a label saying "no layer info".
        lbl = font.render("(scorer has no layer_sizes; using mock)", True, TEXT_DIM)
        surface.blit(lbl, (rect.x + 8, rect.y + 30))
        return
    layer_sizes = net.layer_sizes
    n_layers = len(layer_sizes)
    if n_layers == 0:
        return

    # Compute neuron positions.
    layer_x = [
        rect.x + 40 + i * ((rect.width - 80) // max(1, n_layers - 1))
        for i in range(n_layers)
    ]
    max_neurons = max(layer_sizes)
    neuron_r = max(6, min(18, (rect.height - 80) // (max(2, max_neurons) * 2)))
    neuron_positions: list[list[tuple[int, int]]] = []
    for li, count in enumerate(layer_sizes):
        xs = layer_x[li]
        total_h = count * (neuron_r * 2 + 8)
        start_y = rect.y + 50 + (rect.height - 70 - total_h) // 2
        positions = []
        for ni in range(count):
            ny = start_y + ni * (neuron_r * 2 + 8) + neuron_r
            positions.append((xs, ny))
        neuron_positions.append(positions)

    # Draw connections (edges) first so neurons sit on top.
    for li in range(n_layers - 1):
        for (x1, y1) in neuron_positions[li]:
            for (x2, y2) in neuron_positions[li + 1]:
                pygame.draw.line(surface, NEURON_EDGE, (x1, y1), (x2, y2), 1)

    # Draw neurons with a mock activation pulse.
    t = time.time()
    for li, positions in enumerate(neuron_positions):
        for ni, (x, y) in enumerate(positions):
            # Mock activation: a wave traveling left-to-right + per-neuron phase.
            phase = (t * 0.8 + li * 0.3 + ni * 0.15) % (2 * math.pi)
            activation = 0.5 + 0.5 * math.sin(phase)
            r_col = int(NEURON_LOW[0] + (NEURON_HIGH[0] - NEURON_LOW[0]) * activation)
            g_col = int(NEURON_LOW[1] + (NEURON_HIGH[1] - NEURON_LOW[1]) * activation)
            b_col = int(NEURON_LOW[2] + (NEURON_HIGH[2] - NEURON_LOW[2]) * activation)
            pygame.draw.circle(surface, (r_col, g_col, b_col), (x, y), neuron_r)
            pygame.draw.circle(surface, NEURON_EDGE, (x, y), neuron_r, 1)

    # Layer labels.
    for li, size in enumerate(layer_sizes):
        x = layer_x[li]
        lbl = font.render(f"L{li}({size})", True, TEXT_DIM)
        surface.blit(lbl, (x - lbl.get_width() // 2, rect.bottom - 18))


# ---- Testing panel (board cycling + best pick) ----------------------------

def board_rect(x: int, y: int, cell_px: int, board: "ds.Board") -> pygame.Rect:
    return pygame.Rect(x, y, board.width * cell_px, board.visible_rows * cell_px)


def draw_board(
    surface: pygame.Surface,
    board: "ds.Board",
    piece: Optional["ds.Piece"],
    rect: pygame.Rect,
    cell_px: int,
    piece_color: tuple[int, int, int] = ACTIVE,
    *,
    show_invisible: bool = False,
) -> None:
    """Draw a board (visible rows only) with locked blocks in white-ish and
    an optional active piece in the given color. Black background."""
    pygame.draw.rect(surface, BG, rect)
    # Visible rows only (row 0 at bottom -> drawn last).
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


def draw_testing_panel(
    surface: pygame.Surface,
    board: "ds.Board",
    candidates: list["ds.LockCandidate"],
    scores: list[float],
    current_idx: int,
    best_idx: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    rect: pygame.Rect,
    net: "MockNetwork | Scorer",
) -> None:
    """Draw the testing panel: cycling board (left) + best pick (right) + scores."""
    pygame.draw.rect(surface, PANEL_BG, rect)
    pygame.draw.rect(surface, GRID, rect, 1)

    # Title.
    title = font.render("Testing (lock candidates)", True, TEXT)
    surface.blit(title, (rect.x + 8, rect.y + 6))

    cell_px = 18
    # Left board: cycling through candidates.
    left_x = rect.x + 16
    left_y = rect.y + 36
    left_rect = board_rect(left_x, left_y, cell_px, board)
    cur_piece = candidates[current_idx].piece if candidates else None
    draw_board(surface, board, cur_piece, left_rect, cell_px, ACTIVE)

    lbl = small_font.render(f"cycling candidates ({current_idx + 1}/{len(candidates)})", True, TEXT_DIM)
    surface.blit(lbl, (left_x, left_y + left_rect.height + 4))
    if cur_piece:
        path = candidates[current_idx].inputs_str()
        path_txt = small_font.render(f"inputs: {path}", True, TEXT_DIM)
        surface.blit(path_txt, (left_x, left_y + left_rect.height + 20))

    # Right board: best pick.
    right_x = left_x + left_rect.width + 32
    right_y = left_y
    right_rect = board_rect(right_x, right_y, cell_px, board)
    best_piece = candidates[best_idx].piece if candidates else None
    draw_board(surface, board, best_piece, right_rect, cell_px, BEST)

    lbl2 = small_font.render("network's pick (argmax)", True, TEXT_DIM)
    surface.blit(lbl2, (right_x, right_y + right_rect.height + 4))
    if best_piece:
        path2 = candidates[best_idx].inputs_str()
        path_txt2 = small_font.render(f"inputs: {path2}", True, TEXT_DIM)
        surface.blit(path_txt2, (right_x, right_y + right_rect.height + 20))

    # Scores list (top 5).
    scores_x = right_x + right_rect.width + 24
    scores_y = right_y
    surface.blit(small_font.render("top scores", True, TEXT_DIM), (scores_x, scores_y))
    # Sort by score descending, take top 5.
    ranked = sorted(enumerate(scores), key=lambda kv: -kv[1])[:5]
    for i, (orig_idx, sc) in enumerate(ranked):
        y = scores_y + 20 + i * 18
        color = BEST if orig_idx == best_idx else TEXT
        c = candidates[orig_idx]
        txt = small_font.render(
            f"#{i+1}  s={sc:+.3f}  st={c.piece.state} x={c.piece.x}", True, color
        )
        surface.blit(txt, (scores_x, y))
        # Mini score bar.
        bar_w = max(0, min(120, int(sc * 40 + 60)))
        pygame.draw.rect(surface, SCORE_BAR, (scores_x + 180, y + 4, bar_w, 8))

    # Feature readout for the current candidate (training-side debugging).
    if candidates and hasattr(net, "score"):
        feats = extract_features(board, candidates[current_idx])
        feats_y = scores_y + 140
        surface.blit(small_font.render("features (current):", True, TEXT_DIM), (scores_x, feats_y))
        names = ["agg_h", "holes", "bump", "lines"]
        for i, (n, v) in enumerate(zip(names, feats)):
            y = feats_y + 18 + i * 16
            surface.blit(small_font.render(f"  {n} = {v:.1f}", True, TEXT_DIM), (scores_x, y))


# ---- Main loop ------------------------------------------------------------

DEMO_BOARDS = {
    # A near-full board with a gap on the right, good for showing the piece
    # weaving into a hole.
    "stack": "........../........../........../........../........#/......####/######.###/#########/#########/#########",
    # Empty board.
    "empty": "........../........../........../........../........../........../........../........../........../..........",
}


def main() -> None:
    args = sys.argv[1:]
    piece_name = "I"
    board_name = "stack"
    for i, a in enumerate(args):
        if a == "--piece" and i + 1 < len(args):
            piece_name = args[i + 1].upper()
        elif a == "--board" and i + 1 < len(args):
            board_name = args[i + 1]
        elif a.startswith("--board="):
            board_name = a.split("=", 1)[1]
    if piece_name not in PIECE_KINDS:
        print(f"unknown piece {piece_name!r}; choose from {list(PIECE_KINDS)}")
        sys.exit(1)

    # Resolve board: either a named demo or a spec string.
    if board_name in DEMO_BOARDS:
        board = _parse_board(DEMO_BOARDS[board_name])
    else:
        board = _parse_board(board_name)

    kind = PIECE_KINDS[piece_name]

    # Set up the scorer + "network".
    net = MockNetwork()

    # Compute all lock candidates + their scores up front.
    candidates = board.search_locks(kind)
    scores = [net.score(board, c) for c in candidates]
    if candidates:
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
    else:
        best_idx = 0

    pygame.init()
    pygame.display.set_caption("deathbot — live training + testing viz")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 12)

    train_rect = pygame.Rect(
        PANEL_MARGIN, PANEL_MARGIN, TRAIN_PANEL_W, WINDOW_H - PANEL_MARGIN * 2
    )
    test_rect = pygame.Rect(
        PANEL_MARGIN * 2 + TRAIN_PANEL_W, PANEL_MARGIN,
        TEST_PANEL_W, WINDOW_H - PANEL_MARGIN * 2,
    )

    current_idx = 0
    paused = False
    reduced_motion = False
    cycle_speed = 8.0  # candidates per second
    time_accum = 0.0
    step = 0

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
                    if candidates:
                        current_idx = (current_idx - 1) % len(candidates)
                elif event.key == pygame.K_RIGHT:
                    if candidates:
                        current_idx = (current_idx + 1) % len(candidates)
                elif event.key == pygame.K_r:
                    reduced_motion = not reduced_motion

        # Auto-cycle candidates.
        if not paused and candidates:
            speed = cycle_speed * (0.4 if reduced_motion else 1.0)
            time_accum += dt * speed
            if time_accum >= 1.0:
                steps = int(time_accum)
                current_idx = (current_idx + steps) % len(candidates)
                time_accum -= steps

        # Mock training step counter (phase 2 will drive this for real).
        step += 1

        # Render.
        screen.fill(BG)
        draw_network(screen, net, step, font, train_rect)
        draw_testing_panel(
            screen, board, candidates, scores, current_idx, best_idx,
            font, small_font, test_rect, net,
        )

        # Controls hint.
        hint = small_font.render(
            "SPACE pause | ←/→ step | R reduced-motion | ESC quit",
            True, TEXT_DIM,
        )
        screen.blit(hint, (PANEL_MARGIN, WINDOW_H - 18))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
