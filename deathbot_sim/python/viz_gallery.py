"""Rotation gallery: render every piece's spawn position and all rotation
states, side by side, so the cell-offset tables in piece.rs can be eyeballed
against the real TAP game and adjusted.

Three views:
  - `gallery_text()` -> prints a compact ASCII gallery to stdout. Each piece
    gets a row of mini-boards (one per rotation state) showing the piece's
    cells as 'O' on an empty 4x4 bounding box. Good for a quick terminal
    check.
  - `gallery_spawn_text(board=None)` -> prints each piece at its SPAWN
    position on a full 10x21 board (with the invisible row). This is what
    you'll compare against TAP spawn screenshots.
  - `gallery_tk()` -> opens a Tkinter window with a grid of mini-boards,
    one per (piece, state). Click a piece name to see it at spawn position
    on a full board. Interactive; falls back to text if no display.

Run as:
    python -m viz_gallery               # text gallery (all pieces, all states)
    python -m viz_gallery --spawn       # spawn positions on full boards
    python -m viz_gallery --tk          # Tkinter interactive gallery
    python -m viz_gallery --piece T     # single piece, all states
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import deathbot_sim as ds

# Make the local viz module importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from viz import render_text, print_board  # noqa: E402


PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_KINDS = {
    "I": ds.PieceType.I,
    "O": ds.PieceType.O,
    "T": ds.PieceType.T,
    "S": ds.PieceType.S,
    "Z": ds.PieceType.Z,
    "J": ds.PieceType.J,
    "L": ds.PieceType.L,
}


def _mini_board_text(piece: "ds.Piece", size: int = 4) -> list[str]:
    """Render a piece in a `size`x`size` bounding box, top row first.

    Cells of the piece are drawn as 'O'; empty cells as '.'. The piece's
    origin (x, y) is NOT used — we just render the relative cell offsets so
    the shape is visible regardless of position.
    """
    cells = {(int(c), int(r)) for c, r in piece.cells()}
    # Normalize so the minimum col and row are 0 (shift into the box).
    if not cells:
        return ["." * size for _ in range(size)]
    min_c = min(c for c, _ in cells)
    min_r = min(r for _, r in cells)
    norm = {(c - min_c, r - min_r) for c, r in cells}
    width = max(c for c, _ in norm) + 1
    height = max(r for _, r in norm) + 1
    # Use a box big enough for the piece but at least `size`.
    box_w = max(size, width)
    box_h = max(size, height)
    lines = []
    for r in range(box_h - 1, -1, -1):  # top row first
        line = []
        for c in range(box_w):
            line.append("O" if (c, r) in norm else ".")
        lines.append("".join(line))
    return lines


def gallery_text(pieces: Optional[list[str]] = None) -> None:
    """Print a compact ASCII gallery: one row per piece, one mini-board per
    rotation state."""
    names = pieces or PIECE_NAMES
    for name in names:
        kind = PIECE_KINDS[name]
        num_states = ds.Piece(kind, 0, 0, 0).kind  # noqa: F841 — just to probe
        # Determine num states by rotating until we wrap.
        states = []
        seen = set()
        p = ds.Piece(kind, 0, 0, 0)
        while True:
            if p.state in seen:
                break
            seen.add(p.state)
            states.append(p)
            p = p.rotate_cw()
        # Render each state as a mini-board and stack horizontally.
        mini_boards = [_mini_board_text(s) for s in states]
        max_h = max(len(m) for m in mini_boards)
        print(f"--- {name} ({len(states)} states) ---")
        for row_idx in range(max_h):
            parts = []
            for mb in mini_boards:
                if row_idx < len(mb):
                    parts.append(mb[row_idx])
                else:
                    parts.append(" " * len(mb[0]) if mb else "")
            print("  ".join(parts))
        # Print cell offsets for each state (for verification against tables).
        for i, s in enumerate(states):
            cells_str = " ".join(f"({c},{r})" for c, r in s.cells())
            print(f"  state {i}: {cells_str}")
        print()


def gallery_spawn_text(board: Optional["ds.Board"] = None) -> None:
    """Print each piece at its SPAWN position on a full 10x21 board.

    This is the view to compare against TAP spawn screenshots: the piece
    appears in the invisible row / top of the field, exactly as it would
    at the moment of spawning (before any input).
    """
    board = board or ds.Board()
    for name in PIECE_NAMES:
        kind = PIECE_KINDS[name]
        p = ds.Piece.spawn(kind)
        print(f"=== {name} spawn: x={p.x} y={p.y} state={p.state} ===")
        print_board(board, p)
        print()


def gallery_tk(pieces: Optional[list[str]] = None) -> None:
    """Open a Tkinter window with a grid of mini-boards, one per (piece, state).

    Layout: 7 rows (one per piece) x up to 4 columns (one per state). Each
    mini-board is a 4x4 grid of cells; filled = black, empty = white. The
    piece name and state number are labelled.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Tkinter not available; falling back to text gallery.")
        gallery_text(pieces)
        return

    names = pieces or PIECE_NAMES
    cell_px = 18
    box_size = 4
    label_h = 16

    root = tk.Tk()
    root.title("deathbot_sim — ARS rotation gallery")

    canvas = tk.Canvas(
        root,
        width=len(names) * (box_size * cell_px + 30) + 10,
        height=5 * (box_size * cell_px + label_h + 20) + 10,
        bg="#f0f0f0",
        highlightthickness=0,
    )
    canvas.pack(padx=4, pady=4)

    for col, name in enumerate(names):
        kind = PIECE_KINDS[name]
        # Determine states.
        states = []
        seen = set()
        p = ds.Piece(kind, 0, 0, 0)
        while True:
            if p.state in seen:
                break
            seen.add(p.state)
            states.append(p)
            p = p.rotate_cw()

        x0 = 10 + col * (box_size * cell_px + 30)
        canvas.create_text(
            x0 + box_size * cell_px // 2,
            8,
            text=name,
            font=("Consolas", 12, "bold"),
            anchor="n",
        )

        for row, s in enumerate(states):
            y0 = 24 + row * (box_size * cell_px + label_h + 8)
            # Label.
            canvas.create_text(
                x0 + box_size * cell_px // 2,
                y0,
                text=f"state {s.state}",
                font=("Consolas", 9),
                anchor="n",
            )
            # Mini-board.
            cells = {(int(c), int(r)) for c, r in s.cells()}
            min_c = min(c for c, _ in cells)
            min_r = min(r for _, r in cells)
            norm = {(c - min_c, r - min_r) for c, r in cells}
            for r in range(box_size):
                for c in range(box_size):
                    cx0 = x0 + c * cell_px
                    cy0 = y0 + label_h + (box_size - 1 - r) * cell_px
                    cx1 = cx0 + cell_px
                    cy1 = cy0 + cell_px
                    fill = "#000000" if (c, r) in norm else "#ffffff"
                    canvas.create_rectangle(
                        cx0, cy0, cx1, cy1, fill=fill, outline="#888888"
                    )

    ttk.Button(root, text="Close", command=root.destroy).pack(pady=4)
    root.mainloop()


# Also a full-board spawn gallery in Tk for screenshot comparison.
def gallery_spawn_tk() -> None:
    """Open a Tkinter window showing each piece at spawn on a full 10x21 board.
    Press Right/Left arrow to cycle through pieces."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Tkinter not available; falling back to text spawn gallery.")
        gallery_spawn_text()
        return

    board = ds.Board()
    idx = [0]

    cell_px = 22
    root = tk.Tk()
    root.title("deathbot_sim — spawn positions (←/→ to cycle)")
    canvas = tk.Canvas(
        root,
        width=10 * cell_px + 2,
        height=21 * cell_px + 2,
        bg="white",
        highlightthickness=0,
    )
    canvas.pack(padx=4, pady=4)
    info = ttk.Label(root, text="")
    info.pack(pady=2)
    ttk.Label(root, text="← / → to cycle pieces, Esc to quit").pack()

    def redraw() -> None:
        canvas.delete("all")
        name = PIECE_NAMES[idx[0]]
        kind = PIECE_KINDS[name]
        p = ds.Piece.spawn(kind)
        cells = {(int(c), int(r)) for c, r in p.cells()}
        for r in range(21):
            for c in range(10):
                cx0 = 1 + c * cell_px
                cy0 = 1 + (20 - r) * cell_px
                cx1 = cx0 + cell_px
                cy1 = cy0 + cell_px
                if (c, r) in cells:
                    fill = "#444444"
                elif board.cell(c, r):
                    fill = "#000000"
                else:
                    fill = "#ffffff"
                canvas.create_rectangle(cx0, cy0, cx1, cy1, fill=fill, outline="#cccccc")
            if r == 20:
                y_under = 1 + (20 - r) * cell_px + cell_px
                canvas.create_line(1, y_under, 10 * cell_px + 1, y_under, fill="red", width=2)
        info.config(text=f"{name} spawn: x={p.x} y={p.y} state={p.state}  ({idx[0]+1}/{len(PIECE_NAMES)})")

    def on_key(event: "tk.Event") -> None:
        if event.keysym == "Right":
            idx[0] = (idx[0] + 1) % len(PIECE_NAMES)
            redraw()
        elif event.keysym == "Left":
            idx[0] = (idx[0] - 1) % len(PIECE_NAMES)
            redraw()
        elif event.keysym == "Escape":
            root.destroy()

    root.bind("<Key>", on_key)
    redraw()
    root.mainloop()


def main() -> None:
    args = sys.argv[1:]
    if "--tk" in args:
        if "--spawn" in args:
            gallery_spawn_tk()
        else:
            pieces = None
            for i, a in enumerate(args):
                if a == "--piece" and i + 1 < len(args):
                    pieces = [args[i + 1].upper()]
                    break
            gallery_tk(pieces)
    elif "--spawn" in args:
        gallery_spawn_text()
    else:
        pieces = None
        for i, a in enumerate(args):
            if a == "--piece" and i + 1 < len(args):
                pieces = [args[i + 1].upper()]
                break
        gallery_text(pieces)


if __name__ == "__main__":
    main()
