"""Python-side tests for the deathbot_sim Rust extension.

These mirror and extend the Rust unit tests in src/board.rs and tests/sim_tests.rs.
Run with:
    python -m pytest tests/test_sim.py -v

The goal here is twofold:
  1. Catch any regressions in the Python-facing API surface (PyO3 bindings).
  2. Make piece-placement / movement bugs easy to diagnose visually — each
     failure prints the board state via viz.render_text() so you can see
     exactly where the piece ended up.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

# Make the local viz module importable when running from the crate root.
HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import deathbot_sim as ds  # noqa: E402
from viz import render_text  # noqa: E402


ALL_PIECES = [
    ds.PieceType.I,
    ds.PieceType.O,
    ds.PieceType.T,
    ds.PieceType.S,
    ds.PieceType.Z,
    ds.PieceType.J,
    ds.PieceType.L,
]


def _describe(board: "ds.Board", piece: "ds.Piece | None" = None) -> str:
    """Render the board as text for failure messages."""
    return "\n".join(render_text(board, piece, show_invisible=True))


# ----- spawn & validity ----------------------------------------------------

class TestSpawn:
    def test_empty_board_spawn_is_valid(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            p = ds.Piece.spawn(kind)
            assert board.is_valid(p), f"{kind!r} spawn invalid on empty board"

    def test_spawn_state_is_zero(self):
        for kind in ALL_PIECES:
            assert ds.Piece.spawn(kind).state == 0

    def test_spawn_not_locked_out_on_empty_board(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            p = ds.Piece.spawn(kind)
            assert not board.locks_out(p), f"{kind!r} falsely reports lockout"

    def test_spawn_x_is_three(self):
        # Placeholder expectation: spawn column is 3 for all pieces in the
        # current scaffold. Adjust once TAP-exact spawn columns are verified.
        for kind in ALL_PIECES:
            assert ds.Piece.spawn(kind).x == 3, f"{kind!r} spawn x != 3"


# ----- 20G drop ------------------------------------------------------------

class TestDrop20G:
    def test_all_pieces_reach_bottom_on_empty_board(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            p = ds.Piece.spawn(kind)
            grounded = board.drop_20g(p)
            min_row = min(r for _c, r in grounded.cells())
            assert min_row == 0, (
                f"{kind!r} did not reach bottom under 20G (min_row={min_row})\n"
                + _describe(board, grounded)
            )

    def test_drop_rests_on_existing_stack(self):
        board = ds.Board()
        # Build a 3-high wall in columns 4-9, leaving cols 0-3 open.
        for c in range(4, 10):
            for r in range(3):
                board.set_cell(c, r)
        # Drop an I (flat, occupies 4 consecutive cols at dy=1). At spawn
        # x=3 its cells are cols 3..6. The piece is a rigid horizontal bar,
        # so even though col 3 is empty down to row 0, the cells in cols
        # 4-6 rest on top of the 3-high stack at row 3. The whole piece
        # therefore sits at row 3 (all cells share dy=1).
        i = ds.Piece.spawn(ds.PieceType.I)
        grounded = board.drop_20g(i)
        rows = [r for _c, r in grounded.cells()]
        assert min(rows) == 3 and max(rows) == 3, (
            f"flat I should rest on the 3-high stack at row 3 "
            f"(got min={min(rows)}, max={max(rows)})\n"
            + _describe(board, grounded)
        )


# ----- lock & line clear ---------------------------------------------------

class TestLockAndClear:
    def test_lock_then_clear_single_line(self):
        board = ds.Board()
        # Bottom row full except cols 0-1.
        for c in range(2, 10):
            board.set_cell(c, 0)
        # Drop an O at x=-1 so it fills cols 0-1 of row 0 (and row 1).
        o = ds.Piece(ds.PieceType.O, 0, -1, 20)
        grounded = board.drop_20g(o)
        board.lock(grounded)
        cleared = board.clear_lines()
        assert cleared == 1, f"expected 1 line cleared, got {cleared}\n" + _describe(board)
        # The O's upper cells (cols 0-1, row 1) compact into row 0.
        assert board.cell(0, 0), f"col 0 row 0 should be set after compact\n" + _describe(board)
        assert board.cell(1, 0), f"col 1 row 0 should be set after compact\n" + _describe(board)
        assert not board.cell(0, 1), f"col 0 row 1 should be empty after compact\n" + _describe(board)

    def test_clear_full_row_directly(self):
        board = ds.Board()
        for c in range(10):
            board.set_cell(c, 0)
        assert board.clear_lines() == 1
        assert not board.cell(0, 0)

    def test_no_clear_when_row_incomplete(self):
        board = ds.Board()
        for c in range(9):  # leave one empty
            board.set_cell(c, 0)
        assert board.clear_lines() == 0

    def test_double_clear_compacts_correctly(self):
        board = ds.Board()
        # Fill rows 0 and 1 fully, plus a stray block at row 2 col 0.
        for c in range(10):
            board.set_cell(c, 0)
            board.set_cell(c, 1)
        board.set_cell(0, 2)
        cleared = board.clear_lines()
        assert cleared == 2
        # The stray block should fall from row 2 to row 0.
        assert board.cell(0, 0)
        assert not board.cell(0, 2)


# ----- lockout / game over -------------------------------------------------

class TestLockout:
    def test_full_board_spawn_is_invalid_or_lockout(self):
        board = ds.Board()
        for r in range(board.height):
            for c in range(board.width):
                board.set_cell(c, r)
        for kind in ALL_PIECES:
            p = ds.Piece.spawn(kind)
            # On a completely full board the spawn is either invalid
            # (overlaps) or a lockout (cells above the field). Either is
            # an acceptable "game over" signal.
            assert not board.is_valid(p) or board.locks_out(p), (
                f"{kind!r} should be invalid or lockout on full board\n"
                + _describe(board, p)
            )

    def test_piece_above_field_is_lockout(self):
        board = ds.Board()
        # Vertical I (state 1) at y=20 puts cells at rows 20..23 — row 21+
        # are above the field.
        high = ds.Piece(ds.PieceType.I, 1, 3, 20)
        assert board.locks_out(high), (
            "vertical I at y=20 should lock out (cells above row 20)\n"
            + _describe(board, high)
        )

    def test_piece_in_invisible_row_is_not_lockout(self):
        board = ds.Board()
        # Vertical I at y=17: top cell at row 20 (the invisible row, in-field).
        in_field = ds.Piece(ds.PieceType.I, 1, 3, 17)
        assert not board.locks_out(in_field), (
            "vertical I at y=17 is in the invisible row, not above the field\n"
            + _describe(board, in_field)
        )


# ----- rotation (placeholder, to be filled in 1.2) -------------------------

class TestRotation:
    def test_rotate_cw_increments_state(self):
        p = ds.Piece(ds.PieceType.T, 0, 0, 0)
        assert p.rotate_cw().state == 1
        assert p.rotate_cw().rotate_cw().state == 2

    def test_rotate_ccw_decrements_state(self):
        p = ds.Piece(ds.PieceType.T, 2, 0, 0)
        assert p.rotate_ccw().state == 1

    def test_i_has_two_states(self):
        p = ds.Piece(ds.PieceType.I, 0, 0, 0)
        assert p.rotate_cw().state == 1
        assert p.rotate_cw().rotate_cw().state == 0  # wraps

    def test_o_has_one_state(self):
        p = ds.Piece(ds.PieceType.O, 0, 0, 0)
        assert p.rotate_cw().state == 0
        assert p.rotate_ccw().state == 0

    def test_cells_count_is_four_for_all_states(self):
        for kind in ALL_PIECES:
            for state in range(4):
                p = ds.Piece(kind, state, 0, 0)
                assert len(p.cells()) == 4, (
                    f"{kind!r} state {state} should have 4 cells"
                )


# ----- translation ---------------------------------------------------------

class TestTranslate:
    def test_translate_moves_origin(self):
        p = ds.Piece(ds.PieceType.T, 0, 3, 19)
        moved = p.translate(1, -1)
        assert moved.x == 4
        assert moved.y == 18

    def test_translate_negative(self):
        p = ds.Piece(ds.PieceType.T, 0, 3, 19)
        moved = p.translate(-1, 0)
        assert moved.x == 2
        assert moved.y == 19


# ----- visualization (smoke tests for the harness itself) ------------------

class TestViz:
    def test_render_text_returns_21_lines(self):
        board = ds.Board()
        lines = render_text(board, show_invisible=True)
        assert len(lines) == 21
        # Each line is 10 chars of board, plus the invisible-row line has
        # a trailing label. Strip any label suffix before checking width.
        for line in lines:
            board_part = line.split("  <-")[0]
            assert len(board_part) == 10, (
                f"line not 10 wide: {line!r} (board part {len(board_part)})"
            )

    def test_render_text_marks_filled_cells(self):
        board = ds.Board()
        board.set_cell(0, 0)
        lines = render_text(board, show_invisible=True)
        # Bottom row is the last line (row 0 = bottom).
        assert lines[-1][0] == "#"
        assert lines[-1][1] == "."

    def test_render_text_draws_piece_as_O(self):
        board = ds.Board()
        p = ds.Piece(ds.PieceType.O, 0, 0, 0)  # cells at cols 1-2 rows 0-1
        lines = render_text(board, p, show_invisible=False)
        # Bottom row: cols 1-2 should be 'O'.
        assert lines[-1][1] == "O"
        assert lines[-1][2] == "O"


# ----- rotation + wall kicks (phase 1.2) -----------------------------------

class TestRotationApi:
    def test_try_rotate_cw_on_empty_board_succeeds(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            if kind == ds.PieceType.O:
                continue
            p = ds.Piece.spawn(kind)
            r = board.try_rotate(p, True)
            assert r is not None, f"{kind!r} CW rotation failed on empty board"
            assert r.state != p.state, f"{kind!r} state should change after CW"

    def test_try_rotate_ccw_on_empty_board_succeeds(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            if kind == ds.PieceType.O:
                continue
            p = ds.Piece.spawn(kind)
            r = board.try_rotate(p, False)
            assert r is not None, f"{kind!r} CCW rotation failed on empty board"
            assert r.state != p.state, f"{kind!r} state should change after CCW"

    def test_o_piece_never_rotates(self):
        board = ds.Board()
        p = ds.Piece.spawn(ds.PieceType.O)
        assert board.try_rotate(p, True) is None
        assert board.try_rotate(p, False) is None

    def test_bare_rotation_uses_zero_kick_on_empty_board(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            if kind == ds.PieceType.O:
                continue
            p = ds.Piece.spawn(kind)
            kick = board.kick_used(p, True)
            assert kick == (0, 0), f"{kind!r} should use (0,0) kick on empty board, got {kick}"

    def test_right_kick_chosen_when_left_blocked(self):
        board = ds.Board()
        # L state 3 at (3,0) CW -> state 0 bare (5,0),(3,1),(4,1),(5,1).
        # Block (3,1) (scan cell 4, off-center): bare fails, kicks allowed.
        # +1 -> x=4: (6,0),(4,1),(5,1),(6,1) open. -1 -> x=2 includes
        # (3,1) blocked, fails. So +1.
        l = ds.Piece(ds.PieceType.L, 3, 3, 0)
        board.set_cell(3, 1)
        kick = board.kick_used(l, True)
        assert kick == (1, 0), f"should use +1 right kick, got {kick}\n" + _describe(board, l)

    def test_left_kick_chosen_when_right_blocked(self):
        board = ds.Board()
        # Same L3->L0 CW: block (5,0) (scan 9, off-center, bare cell) so
        # bare fails with kicks allowed; block (6,0) too (outside the box,
        # kills +1 only). -1 -> x=2 all open. So -1.
        l = ds.Piece(ds.PieceType.L, 3, 3, 0)
        board.set_cell(5, 0)
        board.set_cell(6, 0)
        kick = board.kick_used(l, True)
        assert kick == (-1, 0), f"should use -1 left kick, got {kick}\n" + _describe(board, l)

    def test_i_piece_never_kicks(self):
        board = ds.Board()
        # I spawn CW -> state 1 needs (5,18..21). Block (5,19): bare fails
        # while x=4 (+1) would fit — must still fail in TAP.
        i = ds.Piece.spawn(ds.PieceType.I)
        board.set_cell(5, 19)
        assert board.try_rotate(i, True) is None
        assert board.kick_used(i, True) is None

    def test_no_double_kicks(self):
        board = ds.Board()
        # L3->L0 CW blocked at bare, +1 AND -1: block (3,1) [bare/-1],
        # (6,0) [+1]. ±2 cells open, but TAP must still fail.
        l = ds.Piece(ds.PieceType.L, 3, 3, 0)
        board.set_cell(3, 1)
        board.set_cell(6, 0)
        assert board.try_rotate(l, True) is None

    def test_center_column_blocks_kick(self):
        board = ds.Board()
        # T state 0 at (3,0) CW -> state 1 bare (4,0),(4,1),(5,1),(4,2).
        # Box (3,0), scan 1..9: block (4,2) (scan cell 2, CENTER). Bare
        # fails and the first collision is center -> no kicks, even though
        # +1 -> (5,0),(5,1),(6,1),(5,2) would fit.
        t = ds.Piece(ds.PieceType.T, 0, 3, 0)
        board.set_cell(4, 2)
        assert board.try_rotate(t, True) is None
        assert board.kick_used(t, True) is None

    def test_center_column_allows_offcenter_first_hit(self):
        board = ds.Board()
        # L3->L0 CW with (3,1) blocked (scan 4, off-center, bare cell):
        # bare fails, first hit off-center, kicks allowed, +1 wins.
        l = ds.Piece(ds.PieceType.L, 3, 3, 0)
        board.set_cell(3, 1)
        kick = board.kick_used(l, True)
        assert kick == (1, 0), f"off-center first hit should allow +1, got {kick}\n" + _describe(board, l)

    def test_rotation_fails_when_all_kicks_blocked(self):
        board = ds.Board()
        # Block every column the I piece could rotate into.
        for c in range(1, 9):
            for r in range(17, board.height):
                board.set_cell(c, r)
        i = ds.Piece.spawn(ds.PieceType.I)
        assert board.try_rotate(i, True) is None, "I rotation should fail when all kicks blocked"


# ----- tick action application (phase 1.2) ---------------------------------

class TestApplyTick:
    def test_neutral_on_grounded_piece_keeps_position(self):
        board = ds.Board()
        p = board.drop_20g(ds.Piece.spawn(ds.PieceType.T))
        piece, grounded = board.apply_tick(p, None, None)
        assert grounded, "neutral on grounded piece should report grounded"
        assert piece == p

    def test_move_left_then_drop(self):
        board = ds.Board()
        p = ds.Piece.spawn(ds.PieceType.O)
        piece, grounded = board.apply_tick(p, None, "left")
        assert grounded, "O should be grounded after one tick on empty board"
        assert piece.x == p.x - 1

    def test_move_right_then_drop(self):
        board = ds.Board()
        p = ds.Piece.spawn(ds.PieceType.O)
        piece, grounded = board.apply_tick(p, None, "right")
        assert grounded
        assert piece.x == p.x + 1

    def test_rotate_cw_then_drop(self):
        board = ds.Board()
        p = ds.Piece.spawn(ds.PieceType.T)
        piece, grounded = board.apply_tick(p, "cw", None)
        assert piece.state == 1, f"T should be state 1 after CW, got {piece.state}"
        assert grounded, "T should be grounded after one tick on empty board"

    def test_rotate_and_move_same_tick(self):
        board = ds.Board()
        p = ds.Piece.spawn(ds.PieceType.T)
        piece, grounded = board.apply_tick(p, "cw", "left")
        assert piece.state == 1, "rotate should happen before move"
        assert piece.x == p.x - 1, "move should shift left after rotate"

    def test_move_blocked_by_wall_stays_put(self):
        board = ds.Board()
        # Fill col 0 so a Left move from spawn is blocked.
        for r in range(board.height):
            board.set_cell(0, r)
        p = ds.Piece.spawn(ds.PieceType.O)  # x=3, cells cols 4-5
        # Move left repeatedly — eventually hits the wall.
        piece = p
        for _ in range(10):
            piece, _ = board.apply_tick(piece, None, "left")
        # The piece should not overlap the wall (col 0).
        for c, r in piece.cells():
            assert c >= 0, f"piece cell at col {c} should not be negative"


# ----- IRS root states (phase 1.2) -----------------------------------------

class TestIrsRoots:
    def test_irs_count_matches_num_states_on_empty_board(self):
        board = ds.Board()
        # Use string keys since PieceType hashing was added in 1.2 but we
        # keep this test robust against binding changes.
        expected = {
            "I": 2,
            "O": 1,
            "T": 4,
            "S": 2,
            "Z": 2,
            "J": 4,
            "L": 4,
        }
        kind_by_name = {
            "I": ds.PieceType.I,
            "O": ds.PieceType.O,
            "T": ds.PieceType.T,
            "S": ds.PieceType.S,
            "Z": ds.PieceType.Z,
            "J": ds.PieceType.J,
            "L": ds.PieceType.L,
        }
        for name, count in expected.items():
            kind = kind_by_name[name]
            roots = board.irs_root_states(kind)
            assert len(roots) == count, (
                f"{name} should have {count} IRS roots, got {len(roots)}"
            )

    def test_irs_roots_have_distinct_states(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            roots = board.irs_root_states(kind)
            states = {r.state for r in roots}
            assert len(states) == len(roots), f"{kind!r} IRS roots have duplicate states"

    def test_irs_roots_all_valid_on_empty_board(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            for root in board.irs_root_states(kind):
                assert board.is_valid(root), f"{kind!r} IRS root {root} is invalid"


# ----- rotation gallery (visualizer smoke tests) ---------------------------

class TestGallery:
    def test_mini_board_renders_piece_shape(self):
        # Import here so the test doesn't fail if run from a different cwd.
        sys.path.insert(0, str(PYTHON_DIR))
        from viz_gallery import _mini_board_text

        # T state 0: cells (1,0),(0,1),(1,1),(2,1). The 3-wide row is row 1
        # (the dy=1 row), which in the mini-board (top row first) is the
        # second-to-last line. The bottom line (row 0) only has the single
        # cell (1,0).
        t = ds.Piece(ds.PieceType.T, 0, 0, 0)
        lines = _mini_board_text(t)
        # The 3-wide flat row should contain 'OOO' somewhere.
        assert any("OOO" in line for line in lines), (
            f"T should have a 3-wide row, got: {lines}"
        )


# ----- pathfinder BFS (phase 1.3) ------------------------------------------

class TestPathfinder:
    def test_empty_board_has_candidates_for_all_pieces(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            locks = board.search_locks(kind)
            assert len(locks) > 0, f"{kind!r} should have reachable locks on empty board"

    def test_count_locks_matches_search(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            assert board.count_locks(kind) == len(board.search_locks(kind))

    def test_o_piece_reaches_nine_horizontal_positions(self):
        # O can't rotate; on an empty board it can occupy cols 0-1 through 8-9.
        board = ds.Board()
        locks = board.search_locks(ds.PieceType.O)
        xs = {c.piece.x for c in locks}
        assert len(xs) == 9, f"O should reach 9 x-positions, got {len(xs)}: {sorted(xs)}"
        for c in locks:
            assert c.piece.state == 0, "O candidate should be state 0"

    def test_bare_spawn_drop_is_a_candidate(self):
        # The IRS root with no inputs (empty path) should be present.
        board = ds.Board()
        for kind in ALL_PIECES:
            locks = board.search_locks(kind)
            bare = [c for c in locks if not c.inputs]
            assert len(bare) >= 1, f"{kind!r} should have a bare-drop candidate (empty inputs)"

    def test_candidates_reach_floor_on_empty_board(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            for c in board.search_locks(kind):
                min_r = min(r for _c, r in c.piece.cells())
                assert min_r == 0, f"{kind!r} candidate not at floor: {c.piece}"

    def test_no_duplicate_state_x_pairs(self):
        # BFS dedups by (state, x); no two candidates share that key.
        board = ds.Board()
        for kind in ALL_PIECES:
            locks = board.search_locks(kind)
            keys = {(c.piece.state, c.piece.x) for c in locks}
            assert len(keys) == len(locks), f"{kind!r} has duplicate (state, x) candidates"

    def test_t_reaches_all_four_rotation_states(self):
        board = ds.Board()
        locks = board.search_locks(ds.PieceType.T)
        states = {c.piece.state for c in locks}
        assert states == {0, 1, 2, 3}, f"T should reach states 0-3, got {sorted(states)}"

    def test_i_reaches_both_rotation_states(self):
        board = ds.Board()
        locks = board.search_locks(ds.PieceType.I)
        states = {c.piece.state for c in locks}
        assert states == {0, 1}, f"I should reach states 0-1, got {sorted(states)}"

    def test_s_reaches_both_rotation_states(self):
        board = ds.Board()
        locks = board.search_locks(ds.PieceType.S)
        states = {c.piece.state for c in locks}
        assert states == {0, 1}, f"S should reach states 0-1, got {sorted(states)}"

    def test_inputs_are_valid_enum_values(self):
        board = ds.Board()
        for kind in ALL_PIECES:
            for c in board.search_locks(kind):
                for inp in c.inputs:
                    assert inp in (
                        ds.Input.Left,
                        ds.Input.Right,
                        ds.Input.RotateCw,
                        ds.Input.RotateCcw,
                        ds.Input.Neutral,
                    ), f"unexpected input {inp} for {kind!r}"

    def test_bfs_finds_shortest_paths_first(self):
        # The bare drop has 0 inputs; at least one 1-input candidate exists.
        board = ds.Board()
        locks = board.search_locks(ds.PieceType.T)
        lens = [len(c.inputs) for c in locks]
        assert 0 in lens, "should be a 0-input candidate (bare drop)"
        assert 1 in lens, "should be at least one 1-input candidate"

    def test_full_board_candidates_have_empty_inputs(self):
        # On a full board the piece can't move, so all candidates (if any)
        # have empty input paths.
        board = ds.Board()
        for r in range(board.height):
            for c in range(board.width):
                board.set_cell(c, r)
        for kind in ALL_PIECES:
            for c in board.search_locks(kind):
                assert not c.inputs, f"{kind!r} should have no inputs on full board"

    def test_locks_decrease_when_stack_blocks_movement(self):
        # A board with walls should yield fewer locks than an empty board.
        empty = ds.Board()
        walled = ds.Board()
        # Build a 5-high wall in columns 3-6 across the spawn rows.
        for c in range(3, 7):
            for r in range(5):
                walled.set_cell(c, r)
        for kind in ALL_PIECES:
            n_empty = empty.count_locks(kind)
            n_walled = walled.count_locks(kind)
            assert n_walled <= n_empty, (
                f"{kind!r}: walled board ({n_walled}) should have <= "
                f"empty board ({n_empty}) locks"
            )


# ----- pathfinder visualizer (smoke tests) --------------------------------

class TestPathfinderViz:
    def test_render_locks_overlay_returns_21_lines(self):
        sys.path.insert(0, str(PYTHON_DIR))
        from viz_pathfind import render_locks_overlay

        board = ds.Board()
        lines = render_locks_overlay(board, ds.PieceType.T)
        assert len(lines) == 21
        for line in lines:
            board_part = line.split("  <-")[0]
            assert len(board_part) == 10

    def test_overlay_marks_reachable_cells(self):
        sys.path.insert(0, str(PYTHON_DIR))
        from viz_pathfind import render_locks_overlay

        board = ds.Board()
        lines = render_locks_overlay(board, ds.PieceType.T)
        # At least one 'x' should appear (the T has reachable locks).
        assert any("x" in line for line in lines), "overlay should mark reachable cells with 'x'"
