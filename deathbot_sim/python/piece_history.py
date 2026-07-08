"""TAP piece generator with history-based rerolling.

TAP doesn't use pure random piece selection. It keeps a history of the last 4
pieces (initially [Z, Z, S, S]) and when generating a new piece, it rerolls
up to 6 times if the drawn piece is in the history. On the 6th reroll it
accepts whatever it gets. This reduces the chance of long droughts and
repeated pieces, making the game more forgiving.

This is shared between selfplay.py (training) and viz_live.py (visualizer)
so both use the same piece distribution.
"""

from __future__ import annotations

import random
from collections import deque

import deathbot_sim as ds


# Initial history per TAP: [Z, Z, S, S]
_INITIAL_HISTORY = [ds.PieceType.Z, ds.PieceType.Z, ds.PieceType.S, ds.PieceType.S]

# All 7 piece types.
ALL_PIECES = [
    ds.PieceType.I, ds.PieceType.O, ds.PieceType.T,
    ds.PieceType.S, ds.PieceType.Z, ds.PieceType.J, ds.PieceType.L,
]

# Max reroll attempts before accepting.
_MAX_REROLLS = 6


class PieceHistory:
    """TAP piece generator with 4-piece history rerolling.

    Usage:
        hist = PieceHistory(rng)
        piece = hist.next()  # get the next piece, updating history
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.history: deque = deque(_INITIAL_HISTORY, maxlen=4)

    def next(self) -> "ds.PieceType":
        """Draw the next piece using TAP's history-based reroll."""
        for attempt in range(_MAX_REROLLS):
            piece = self.rng.choice(ALL_PIECES)
            if piece not in self.history:
                # Accepted — update history and return.
                self.history.append(piece)
                return piece
        # 6th reroll: accept whatever we got.
        self.history.append(piece)
        return piece

    def reset(self) -> None:
        """Reset to the initial history [Z, Z, S, S]."""
        self.history = deque(_INITIAL_HISTORY, maxlen=4)


class PieceQueue:
    """A piece queue with one-piece lookahead, backed by PieceHistory.

    Maintains a current piece and a next-piece preview. When you call
    `advance()`, the current next-piece becomes the new current piece, and
    a new next-piece is drawn. This ensures the displayed "next" piece is
    exactly the piece that becomes active on the next step (unlike drawing
    two fresh pieces each time, which discards the preview).
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self.hist = PieceHistory(rng)
        # Prime the queue: draw the first current + next.
        self.current = self.hist.next()
        self.next_piece = self.hist.next()

    def reset(self) -> None:
        """Reset to the initial state and re-prime the queue."""
        self.hist.reset()
        self.current = self.hist.next()
        self.next_piece = self.hist.next()

    def advance(self) -> "ds.PieceType":
        """Advance: the next-piece becomes current, draw a new next-piece.
        Returns the new current piece."""
        self.current = self.next_piece
        self.next_piece = self.hist.next()
        return self.current
