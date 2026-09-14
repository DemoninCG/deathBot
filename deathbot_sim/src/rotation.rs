//! Arika Rotation System: rotation tables + wall kicks (TAP).
//!
//! ARS wall-kick model (classic TAP, i.e. TGM1/TAP — NOT TGM3):
//! - When a rotation would place the piece in an invalid spot, the engine
//!   tries at most TWO kicks: one cell right, then one cell left, and picks
//!   the first that yields a valid position.
//! - TAP checks the RIGHT kick first, then the LEFT. This is asymmetric and
//!   is one of the defining quirks of ARS vs. SRS (which uses diagonal kicks).
//! - The O piece never rotates (one state, rotation is a no-op).
//! - The I piece NEVER kicks in TAP (I wall/floor kicks were added in
//!   TGM3 Terror-Instinct). I rotation succeeds only as the bare rotation.
//! - L, J, T are subject to the CENTER COLUMN rule: the 3x3 box of the
//!   bare target position is scanned in 1..9 order (top-left to
//!   bottom-right); if the FIRST colliding cell (stack block or wall/floor)
//!   is in the center column (cells 2, 5, 8), the rotation fails outright
//!   and no kick is tried. An off-center first collision allows the normal
//!   right-then-left kicks.
//! - No ±2 kicks and no floor kicks exist in TAP.
//!
//! Kick sequence per rotation attempt (in order):
//!   1. (0, 0)   — the bare rotation, no nudge
//!   2. (+1, 0)  — nudge one cell right
//!   3. (-1, 0)  — nudge one cell left

use crate::board::Board;
use crate::piece::{Piece, PieceType};

/// The wall-kick offset sequence, applied in order. TAP convention: right
/// kick first, then left. Each entry is `(dx, dy)` — dy is always 0 for ARS
/// (purely horizontal kicks). TAP has no ±2 or floor kicks.
pub const KICKS: [(i8, i8); 3] = [(0, 0), (1, 0), (-1, 0)];

/// 3x3 box scan order for the center-column rule (cell numbers 1..9,
/// row-major top-to-bottom). Entries are `(dx, dy)` offsets from the box
/// origin plus whether the cell is in the center column (dx == 1).
const CENTER_SCAN: [((i8, i8), bool); 9] = [
    ((0, 2), false), // 1
    ((1, 2), true),  // 2 center
    ((2, 2), false), // 3
    ((0, 1), false), // 4
    ((1, 1), true),  // 5 center
    ((2, 1), false), // 6
    ((0, 0), false), // 7
    ((1, 0), true),  // 8 center
    ((2, 0), false), // 9
];

/// True if a box cell is colliding (wall/floor/stack). Cells above the
/// field (r >= HEIGHT) count as empty — same convention as
/// `Board::is_valid`, which allows high spawn/kick probes.
fn box_cell_blocked(board: &Board, c: i8, r: i8) -> bool {
    if c < 0 || c >= crate::board::WIDTH as i8 || r < 0 {
        return true;
    }
    if (r as usize) >= crate::board::HEIGHT {
        return false;
    }
    board.rows[r as usize] & (1 << c) != 0
}

/// Center-column gate for L/J/T bare rotations. Scans the 3x3 box of the
/// bare target position in 1..9 order; returns true when the FIRST colliding
/// cell is a center-column cell (rotation must fail, no kicks allowed).
/// Returns false when the first collision is off-center (kicks allowed) or
/// when nothing collides (caller already handled the bare-success case).
fn center_column_blocks(board: &Board, target: &Piece) -> bool {
    for &((dx, dy), is_center) in CENTER_SCAN.iter() {
        let c = target.x + dx;
        let r = target.y + dy;
        if box_cell_blocked(board, c, r) {
            return is_center;
        }
    }
    false
}

/// Attempt to rotate `p` (clockwise if `cw`, else counter-clockwise) and apply
/// the TAP ARS wall-kick sequence. Returns the first valid resulting position,
/// or `None` if the rotation fails (piece stays in its current state).
///
/// The returned piece (if any) has the *new* rotation state and the kick
/// offset already applied. The caller is responsible for any subsequent 20G
/// drop.
pub fn try_rotate(board: &Board, p: &Piece, cw: bool) -> Option<Piece> {
    // O piece cannot rotate — no kicks to try.
    if p.kind == PieceType::O {
        return None;
    }

    let target_state = if cw { p.rotate_cw() } else { p.rotate_ccw() };

    // I piece never kicks in TAP: bare rotation or nothing.
    if p.kind == PieceType::I {
        let bare = target_state;
        return if board.is_valid(&bare) {
            Some(bare)
        } else {
            None
        };
    }

    // Bare rotation first for every other piece.
    let bare = target_state;
    if board.is_valid(&bare) {
        return Some(bare);
    }

    // Center-column rule gates kicks for L/J/T only (S/Z skip it).
    if matches!(p.kind, PieceType::L | PieceType::J | PieceType::T)
        && center_column_blocks(board, &bare)
    {
        return None;
    }

    for &(dx, dy) in KICKS.iter().skip(1) {
        let candidate = target_state.translate(dx, dy);
        if board.is_valid(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// Which kick offset actually produced a successful rotation? Returns the
/// first fitting `(dx, dy)`, or `None` if rotation failed entirely. Useful
/// for the visualizer / debugging. Mirrors `try_rotate` exactly (I never
/// kicks; L/J/T gated by the center-column rule; no ±2 kicks).
pub fn kick_used(board: &Board, p: &Piece, cw: bool) -> Option<(i8, i8)> {
    if p.kind == PieceType::O {
        return None;
    }
    let target_state = if cw { p.rotate_cw() } else { p.rotate_ccw() };
    if p.kind == PieceType::I {
        return if board.is_valid(&target_state) {
            Some((0, 0))
        } else {
            None
        };
    }
    if board.is_valid(&target_state) {
        return Some((0, 0));
    }
    if matches!(p.kind, PieceType::L | PieceType::J | PieceType::T)
        && center_column_blocks(board, &target_state)
    {
        return None;
    }
    for &k in KICKS.iter().skip(1) {
        if board.is_valid(&target_state.translate(k.0, k.1)) {
            return Some(k);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn o_piece_never_rotates() {
        let b = Board::new();
        let o = Piece::spawn(PieceType::O);
        assert!(try_rotate(&b, &o, true).is_none());
        assert!(try_rotate(&b, &o, false).is_none());
    }

    #[test]
    fn bare_rotation_succeeds_on_empty_board() {
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            if kind == PieceType::O {
                continue;
            }
            let p = Piece::spawn(kind);
            let r = try_rotate(&b, &p, true);
            assert!(r.is_some(), "{:?} CW rotation failed on empty board", kind);
            let used = kick_used(&b, &p, true);
            assert_eq!(used, Some((0, 0)), "{:?} should not need a kick on empty board", kind);
        }
    }

    #[test]
    fn right_kick_rescues_blocked_rotation() {
        // L state 3 at (3,0) CW -> state 0 bare (5,0),(3,1),(4,1),(5,1).
        // Block (3,1) (scan cell 4, off-center): bare fails, kicks stay
        // allowed. +1 -> x=4: (6,0),(4,1),(5,1),(6,1) open. -1 -> x=2:
        // (4,0),(2,1),(3,1),(4,1): (3,1) blocked, fails. +1 WINS.
        let mut b = Board::new();
        let l = Piece::new(PieceType::L, 3, 3, 0);
        assert!(b.is_valid(&l));
        b.set_cell(3, 1);
        let used = kick_used(&b, &l, true);
        assert_eq!(used, Some((1, 0)), "should use the +1 right kick, got {:?}", used);
    }

    #[test]
    fn left_kick_used_when_right_side_blocked() {
        // Same L3->L0 CW (box origin (3,0)): block (3,1) (scan 4, off)
        // for bare-fail, plus (6,1) (the +1 kick) so +1 fails. -1 -> x=2
        // -> (4,0),(2,1),(3,1),(4,1): (3,1) blocked, fails too. Use (5,1)
        // instead?? (5,1) is scan 6 AND a bare cell: bare fails, first hit
        // ... scan: (3,2)e,(4,2)e,(5,2)e,(3,1)e?? (3,1) empty. (4,1)e,
        // (5,1)B -> first hit (5,1) off-center, kicks allowed. Bare fails.
        // +1 needs (6,1): open. -1 needs (3,1): open. +1 WINS (not -1).
        // For -1: block (3,1) [bare-fail, scan4, kills -1 too — no].
        // Block (6,1) [+1 cell, scan order: (3,2)e,(4,2)e,(5,2)e,(3,1)e,
        // (4,1)e,(5,1)e,(3,0)e,(4,0)e,(5,0)e — (6,1) is NOT in the box,
        // so scan never sees it; bare SUCCEEDS]. Hmm. Block (5,0) (scan 9,
        // off, bare cell): first hit (5,0)?? scan: 1..8 empty, 9 BLOCKED
        // -> off-center, kicks allowed; bare fails. +1 -> x=4:
        // (6,0),(4,1),(5,1),(6,1): open, WINS again. Block (6,0) too
        // (not in box, kills +1 only): +1 needs (6,0) blocked fails; -1 ->
        // x=2: (4,0),(2,1),(3,1),(4,1): all open -> -1 WINS.
        let mut b = Board::new();
        let l = Piece::new(PieceType::L, 3, 3, 0);
        assert!(b.is_valid(&l));
        b.set_cell(5, 0);
        b.set_cell(6, 0);
        let used = kick_used(&b, &l, true);
        assert_eq!(used, Some((-1, 0)), "should use the -1 left kick, got {:?}", used);
    }

    #[test]
    fn rotation_fails_when_bare_blocked_i_piece() {
        // I never kicks in TAP: any blocked bare rotation fails outright,
        // even when a ±1 nudge would fit.
        let mut b = Board::new();
        let i = Piece::spawn(PieceType::I);
        // I spawn state 0 at x=3: cells (3..6, 20). CW -> state 1 at x=3:
        // cells (5,18..21). Block (5,19) so bare fails while x=4 (+1)
        // would be open — must still fail.
        b.set_cell(5, 19);
        assert!(try_rotate(&b, &i, true).is_none());
        assert!(kick_used(&b, &i, true).is_none());
    }

    #[test]
    fn center_column_rule_blocks_kick() {
        // T at low height so the whole 3x3 box is in-bounds. State 0 at
        // (3,0): cells (4,0),(3,1),(4,1),(5,1). CW -> state 1 bare cells
        // (4,0),(4,1),(5,1),(4,2). Box origin (3,0): scan 1..9 =
        // (3,2),(4,2),(5,2),(3,1),(4,1),(5,1),(3,0),(4,0),(5,0).
        // Block (4,2) (scan cell 2, CENTER): bare fails and the first
        // collision is center -> no kicks. +1 kick ->
        // (5,0),(5,1),(6,1),(5,2) would fit, but must be vetoed.
        let mut b = Board::new();
        let t = Piece::new(PieceType::T, 0, 3, 0);
        assert!(b.is_valid(&t));
        b.set_cell(4, 2);
        assert!(try_rotate(&b, &t, true).is_none(),
            "center-column first collision must block the kick, got {:?}",
            kick_used(&b, &t, true));
    }

    #[test]
    fn center_column_rule_allows_offcenter_first_collision() {
        // Same low T: block (3,2) (scan cell 1, off-center) AND (4,0)
        // (a bare piece cell, scan cell 8) so bare fails. Scan hits
        // (3,2) first -> off-center -> kicks allowed. +1 kick needs
        // (5,0),(5,1),(6,1),(5,2): all open -> +1 wins.
        let mut b = Board::new();
        let t = Piece::new(PieceType::T, 0, 3, 0);
        assert!(b.is_valid(&t));
        b.set_cell(3, 2);
        b.set_cell(4, 0);
        let used = kick_used(&b, &t, true);
        assert_eq!(used, Some((1, 0)), "off-center first collision should allow +1 kick, got {:?}", used);
    }

    #[test]
    fn center_column_wall_counts_as_offcenter() {
        // Bare target half outside the left wall: the wall cells are
        // off-center in scan order, so kicks stay allowed. T state 3 at
        // x=-1 CW -> state 0 bare cells (-1..1, y..y+1): box scan hits
        // cell 1 (x=-1,y+2, wall) before any center cell.
        let b = Board::new();
        let t = Piece::new(PieceType::T, 3, -1, 0);
        let used = kick_used(&b, &t, true);
        assert_eq!(used, Some((1, 0)), "wall-first collision is off-center, +1 kick should apply, got {:?}", used);
    }

    #[test]
    fn no_double_kicks() {
        // A position reachable only via ±2 is unreachable in TAP. T state
        // 0 at x=3 CW blocked at bare, +1 AND -1 must both fail.
        let mut b = Board::new();
        let t = Piece::spawn(PieceType::T);
        // Bare state-1 cells (4,19),(4,20),(5,20),(4,21): block (4,19).
        b.set_cell(4, 19);
        // +1 kick cells (5,19),(5,20),(6,20),(5,21): block (5,19).
        b.set_cell(5, 19);
        // -1 kick cells (3,19),(3,20),(4,20),(3,21): block (3,19).
        b.set_cell(3, 19);
        // ±2 cells are open, but TAP must still fail.
        assert!(try_rotate(&b, &t, true).is_none());
    }

    #[test]
    fn rotation_fails_when_all_kicks_blocked() {
        let mut b = Board::new();
        // Block every column the I piece could rotate into across the spawn
        // rows, so every kick in the table fails.
        for c in 1..=8 {
            for r in 17..crate::board::HEIGHT {
                b.set_cell(c, r);
            }
        }
        let i = Piece::spawn(PieceType::I);
        assert!(try_rotate(&b, &i, true).is_none(), "I rotation should fail when all kicks blocked");
    }
}
