"""Simple black & white board visualizer for the deathbot_sim simulator.

Two modes:
  - `render_text(board, piece=None)` -> list[str]: ASCII art, top row first.
      '#' = filled cell, '.' = empty. A live piece (if given) is drawn as
      'O' so it's distinguishable from locked blocks. The invisible buffer
      row (row 20) is included and labelled.
  - `show_tk(board, piece=None)`: pops a small Tkinter window with a 1-bit
      black/white grid. Filled = black, empty = white. A live piece is drawn
      as a distinct gray. Tkinter is stdlib so no install needed; if the
      environment is headless, the text renderer is the fallback.

Also exposes `play_sequence(steps)` for driving a list of (board, piece)
snapshots through both renderers — useful for stepping through a BFS path
when debugging the pathfinder.

Coordinate convention matches the Rust side: row 0 = bottom, row 20 = the
invisible buffer. The renderers flip this so the top of the screen = row 20.
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional, Tuple

import deathbot_sim as ds

# Type aliases.
Cell = Tuple[int, int]


def _piece_cells(piece: "ds.Piece") -> set[Cell]:
    return {(int(c), int(r)) for c, r in piece.cells()}


def render_text(
    board: "ds.Board",
    piece: Optional["ds.Piece"] = None,
    *,
    show_invisible: bool = True,
) -> list[str]:
    """Render the board as ASCII art, top row first.

    '#' = locked block, 'O' = live piece cell, '.' = empty.
    The invisible buffer row (row 20) is included by default and is the
    first line of output; it's marked with a trailing '  <- invisible'.
    """
    pc = _piece_cells(piece) if piece is not None else set()
    width = board.width
    height = board.height
    top = board.visible_rows if not show_invisible else height
    lines: list[str] = []
    for r in range(top - 1, -1, -1):
        row_chars = []
        for c in range(width):
            if (c, r) in pc:
                row_chars.append("O")
            elif board.cell(c, r):
                row_chars.append("#")
            else:
                row_chars.append(".")
        line = "".join(row_chars)
        if show_invisible and r == board.visible_rows:
            line += "  <- invisible (row 20)"
        lines.append(line)
    return lines


def print_board(
    board: "ds.Board",
    piece: Optional["ds.Piece"] = None,
    *,
    show_invisible: bool = True,
    file=None,
) -> None:
    """Print render_text() output to stdout (or a file)."""
    out = file or sys.stdout
    for line in render_text(board, piece, show_invisible=show_invisible):
        print(line, file=out)
    # Footer column ruler.
    print("0123456789", file=out)


# ----- Tkinter GUI (optional, stdlib) --------------------------------------

def show_tk(
    board: "ds.Board",
    piece: Optional["ds.Piece"] = None,
    *,
    cell_px: int = 24,
    title: str = "deathbot_sim",
) -> None:
    """Open a Tkinter window showing the board in black & white.

    Black = locked, white = empty, gray = live piece. The invisible row is
    drawn with a thin red border to remind you it's above the field.
    Raises if a display isn't available (e.g. headless CI).
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as e:
        raise RuntimeError("Tkinter not available; use render_text() instead") from e

    root = tk.Tk()
    root.title(title)
    width = board.width
    height = board.height
    pc = _piece_cells(piece) if piece is not None else set()

    canvas = tk.Canvas(
        root,
        width=width * cell_px + 2,
        height=height * cell_px + 2,
        bg="white",
        highlightthickness=0,
    )
    canvas.pack(padx=4, pady=4)

    for r in range(height):
        for c in range(width):
            x0 = 1 + c * cell_px
            y0 = 1 + (height - 1 - r) * cell_px  # flip: row 0 at bottom
            x1 = x0 + cell_px
            y1 = y0 + cell_px
            if (c, r) in pc:
                fill = "#888888"  # live piece
            elif board.cell(c, r):
                fill = "#000000"  # locked
            else:
                fill = "#ffffff"  # empty
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#cccccc")
        # Mark the invisible buffer row with a red underline.
        if r == board.visible_rows:
            y_under = 1 + (height - 1 - r) * cell_px + cell_px
            canvas.create_line(1, y_under, width * cell_px + 1, y_under, fill="red", width=2)

    info = ttk.Label(root, text=f"{board!r}  piece={piece!r}")
    info.pack(pady=2)

    close = ttk.Button(root, text="Close", command=root.destroy)
    close.pack(pady=2)

    root.mainloop()


def play_sequence(
    steps: Iterable[Tuple["ds.Board", Optional["ds.Piece"]]],
    *,
    use_tk: bool = False,
    pause: bool = True,
) -> None:
    """Step through a sequence of (board, piece) snapshots.

    With use_tk=False each step is printed to stdout and the user presses
    Enter to advance. With use_tk=True the first step is shown in a Tk
    window; subsequent steps require closing & re-calling (kept simple for
    now — a proper animation loop can come later).
    """
    for i, (board, piece) in enumerate(steps):
        print(f"--- step {i} ---")
        print_board(board, piece)
        if pause:
            try:
                input("[enter] next > ")
            except EOFError:
                break


# ----- CLI entry point ------------------------------------------------------

def _demo() -> None:
    """Drop a couple of pieces on an empty board and render the result."""
    board = ds.Board()
    # Drop an O at the left wall.
    o = ds.Piece.spawn(ds.PieceType.O).translate(-3, 0)
    grounded = board.drop_20g(o)
    board.lock(grounded)
    # Drop a T on top, centered.
    t = ds.Piece.spawn(ds.PieceType.T)
    grounded_t = board.drop_20g(t)
    print("After locking O (left) and dropping T (centered):")
    print_board(board, grounded_t)


if __name__ == "__main__":
    if "--tk" in sys.argv:
        # Demo with the Tk GUI.
        board = ds.Board()
        t = ds.Piece.spawn(ds.PieceType.T)
        grounded = board.drop_20g(t)
        show_tk(board, grounded, title="deathbot_sim — T drop demo")
    else:
        _demo()
