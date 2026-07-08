"""Pathfinder visualizer: render all reachable lock positions for a piece on
a board, so you can see the BFS coverage and verify the search is finding
sensible placements.

Views:
  - `render_locks_text(board, kind)` -> prints the board with every lock
    candidate's footprint overlaid as 'x' marks. Cells reachable as a lock
    position get 'x'; existing locked blocks stay '#'. Useful for a quick
    "where can this piece go?" overview.
  - `gallery_locks_text(board, kind, max_show=12)` -> prints up to `max_show`
    individual lock candidates, each on its own mini-board with the piece's
    cells as 'O' and the input path printed below. Good for verifying input
    sequences.
  - `show_locks_tk(board, kind)` -> Tkinter window cycling through lock
    candidates with Left/Right arrows. Each shows the piece (gray) on the
    board with its input path labelled.

Run as:
    python -m viz_pathfind T                  # all T locks on empty board
    python -m viz_pathfind T --gallery        # individual candidates
    python -m viz_pathfind T --tk             # interactive cycling
    python -m viz_pathfind --board 'J........./.########/.########' T
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


PIECE_KINDS = {
    "I": ds.PieceType.I,
    "O": ds.PieceType.O,
    "T": ds.PieceType.T,
    "S": ds.PieceType.S,
    "Z": ds.PieceType.Z,
    "J": ds.PieceType.J,
    "L": ds.PieceType.L,
}


def render_locks_overlay(board: "ds.Board", kind: "ds.PieceType") -> list[str]:
    """Render the board with every lock candidate's footprint as 'x'.

    Cells that are part of at least one reachable lock position get 'x';
    existing locked blocks stay '#'; empty unreachable cells are '.'.
    The invisible row is included.
    """
    locks = board.search_locks(kind)
    reachable: set[tuple[int, int]] = set()
    for c in locks:
        for cell in c.piece.cells():
            reachable.add((int(cell[0]), int(cell[1])))

    lines = []
    for r in range(board.height - 1, -1, -1):
        row_chars = []
        for c in range(board.width):
            if board.cell(c, r):
                row_chars.append("#")
            elif (c, r) in reachable:
                row_chars.append("x")
            else:
                row_chars.append(".")
        line = "".join(row_chars)
        if r == board.visible_rows:
            line += "  <- invisible (row 20)"
        lines.append(line)
    return lines


def render_locks_text(board: "ds.Board", kind: "ds.PieceType") -> None:
    """Print the lock-candidate overlay for `kind` on `board`."""
    name = _kind_name(kind)
    locks = board.search_locks(kind)
    print(f"=== {name}: {len(locks)} lock candidates ===")
    for line in render_locks_overlay(board, kind):
        print(line)
    print("0123456789")
    print("'x' = reachable lock cell, '#' = existing block, '.' = empty")
    print()


def gallery_locks_text(
    board: "ds.Board", kind: "ds.PieceType", max_show: int = 12
) -> None:
    """Print up to `max_show` individual lock candidates, each on its own
    mini-board with the piece as 'O' and the input path below."""
    name = _kind_name(kind)
    locks = board.search_locks(kind)
    print(f"=== {name}: {len(locks)} lock candidates (showing first {min(max_show, len(locks))}) ===")
    for i, c in enumerate(locks[:max_show]):
        print(f"--- candidate {i} ---")
        print_board(board, c.piece)
        path = c.inputs_str() if hasattr(c, "inputs_str") else [str(x) for x in c.inputs]
        print(f"  inputs: {path}")
        print()


def show_locks_tk(board: "ds.Board", kind: "ds.PieceType") -> None:
    """Tkinter window cycling through lock candidates with Left/Right arrows."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Tkinter not available; falling back to text gallery.")
        gallery_locks_text(board, kind)
        return

    locks = board.search_locks(kind)
    if not locks:
        print("No lock candidates; not opening Tk window.")
        return
    idx = [0]
    cell_px = 22
    name = _kind_name(kind)

    root = tk.Tk()
    root.title(f"deathbot_sim — {name} lock candidates (←/→ to cycle)")
    canvas = tk.Canvas(
        root,
        width=board.width * cell_px + 2,
        height=board.height * cell_px + 2,
        bg="white",
        highlightthickness=0,
    )
    canvas.pack(padx=4, pady=4)
    info = ttk.Label(root, text="")
    info.pack(pady=2)
    ttk.Label(root, text="← / → to cycle, Esc to quit").pack()

    def redraw() -> None:
        canvas.delete("all")
        c = locks[idx[0]]
        piece_cells = {(int(x), int(y)) for x, y in c.piece.cells()}
        for r in range(board.height):
            for col in range(board.width):
                cx0 = 1 + col * cell_px
                cy0 = 1 + (board.height - 1 - r) * cell_px
                cx1 = cx0 + cell_px
                cy1 = cy0 + cell_px
                if (col, r) in piece_cells:
                    fill = "#666666"
                elif board.cell(col, r):
                    fill = "#000000"
                else:
                    fill = "#ffffff"
                canvas.create_rectangle(cx0, cy0, cx1, cy1, fill=fill, outline="#cccccc")
            if r == board.visible_rows:
                y_under = 1 + (board.height - 1 - r) * cell_px + cell_px
                canvas.create_line(1, y_under, board.width * cell_px + 1, y_under, fill="red", width=2)
        path = c.inputs_str() if hasattr(c, "inputs_str") else [str(x) for x in c.inputs]
        info.config(
            text=f"{name} lock {idx[0] + 1}/{len(locks)}: state={c.piece.state} x={c.piece.x} y={c.piece.y}  inputs={path}"
        )

    def on_key(event: "tk.Event") -> None:
        if event.keysym == "Right":
            idx[0] = (idx[0] + 1) % len(locks)
            redraw()
        elif event.keysym == "Left":
            idx[0] = (idx[0] - 1) % len(locks)
            redraw()
        elif event.keysym == "Escape":
            root.destroy()

    root.bind("<Key>", on_key)
    redraw()
    root.mainloop()


def _kind_name(kind: "ds.PieceType") -> str:
    for name, k in PIECE_KINDS.items():
        if k == kind:
            return name
    return "?"


def _parse_board(spec: str) -> "ds.Board":
    """Parse a board from a spec string of '/'-separated rows, top row first.
    Each row is 10 chars; '#' = filled, anything else = empty. Rows beyond 21
    or short specs are padded."""
    b = ds.Board()
    rows = spec.split("/")
    # spec is top-row-first; board row 0 is bottom.
    for i, row in enumerate(rows):
        r = len(rows) - 1 - i  # convert to bottom-indexed
        if r < 0 or r >= b.height:
            continue
        for c, ch in enumerate(row[:b.width]):
            if ch == "#":
                b.set_cell(c, r)
    return b


def main() -> None:
    args = sys.argv[1:]
    board_spec: Optional[str] = None
    piece_name: Optional[str] = None
    mode = "overlay"  # default
    for i, a in enumerate(args):
        if a == "--board" and i + 1 < len(args):
            board_spec = args[i + 1]
        elif a == "--gallery":
            mode = "gallery"
        elif a == "--tk":
            mode = "tk"
        elif a in PIECE_KINDS:
            piece_name = a.upper()
    if piece_name is None:
        print("Usage: python -m viz_pathfind <PIECE> [--gallery|--tk] [--board SPEC]")
        print(f"  PIECE one of: {', '.join(PIECE_KINDS)}")
        sys.exit(1)
    board = _parse_board(board_spec) if board_spec else ds.Board()
    kind = PIECE_KINDS[piece_name]
    if mode == "gallery":
        gallery_locks_text(board, kind)
    elif mode == "tk":
        show_locks_tk(board, kind)
    else:
        render_locks_text(board, kind)


if __name__ == "__main__":
    main()
