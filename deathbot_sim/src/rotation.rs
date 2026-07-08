//! Arika Rotation System: rotation tables + wall kicks (phase 1.2).
//!
//! ARS wall-kick model (classic TAP):
//! - When a rotation would place the piece in an invalid spot, the engine
//!   tries a sequence of horizontal "kicks" — nudges by (dx, 0) — and picks
//!   the first that yields a valid position.
//! - TAP checks the RIGHT kick first, then the LEFT. This is asymmetric and
//!   is one of the defining quirks of ARS vs. SRS (which uses diagonal kicks).
//! - The O piece never kicks (it has one state, so rotation is a no-op).
//! - The I piece uses the same kick sequence but only has 2 states.
//!
//! Kick sequence per rotation attempt (in order):
//!   1. (0, 0)   — the bare rotation, no nudge
//!   2. (+1, 0)  — nudge one cell right
//!   3. (-1, 0)  — nudge one cell left
//!   4. (+2, 0)  — nudge two cells right (rarely needed; conservative superset
//!                  covering observed TAP I-piece spins against walls)
//!   5. (-2, 0)  — nudge two cells left
//!
//! The 2-cell kicks are a conservative superset. Adjust here if you find
//! discrepancies against the real game.

use crate::board::Board;
use crate::piece::{Piece, PieceType};

/// The wall-kick offset sequence, applied in order. TAP convention: right
/// kick first, then left. Each entry is `(dx, dy)` — dy is always 0 for ARS
/// (purely horizontal kicks).
pub const KICKS: [(i8, i8); 5] = [
    (0, 0),
    (1, 0),
    (-1, 0),
    (2, 0),
    (-2, 0),
];

/// Attempt to rotate `p` (clockwise if `cw`, else counter-clockwise) and apply
/// the ARS wall-kick sequence. Returns the first valid resulting position, or
/// `None` if every kick in the table fails (piece stays in its current state).
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

    for &(dx, dy) in KICKS.iter() {
        let candidate = target_state.translate(dx, dy);
        if board.is_valid(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// Which kick offset actually produced a successful rotation? Returns the
/// first fitting `(dx, dy)`, or `None` if rotation failed entirely. Useful
/// for the visualizer / debugging.
pub fn kick_used(board: &Board, p: &Piece, cw: bool) -> Option<(i8, i8)> {
    if p.kind == PieceType::O {
        return None;
    }
    let target_state = if cw { p.rotate_cw() } else { p.rotate_ccw() };
    for &k in KICKS.iter() {
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
        let mut b = Board::new();
        // T spawn state 0 at x=3: cells (4,19),(3,20),(4,20),(5,20).
        // CW -> state 1 at x=3: cells (4,19),(4,20),(5,20),(4,21).
        // We want to force the +1 right kick to be the first that succeeds.
        // Block the bare rotation's (4,19) AND (4,20) so (0,0) fails. The +1
        // kick shifts to x=4 -> cells (5,19),(5,20),(6,20),(5,21): open.
        // The -1 kick (tried after +1) shifts to x=2 -> (3,19),(3,20),(4,20),
        // (3,21): (4,20) blocked, fails. So +1 should be selected.
        b.set_cell(4, 19);
        b.set_cell(4, 20);
        let t = Piece::spawn(PieceType::T);
        let used = kick_used(&b, &t, true);
        assert_eq!(used, Some((1, 0)), "should use the +1 right kick, got {:?}", used);
    }

    #[test]
    fn left_kick_used_when_right_side_blocked() {
        let mut b = Board::new();
        // Same T spawn/CW setup, but block the +1 kick's cells instead so the
        // -1 left kick is the first that succeeds. Block (5,19) and (5,20)
        // and (6,20) (the +1 kick's cols) and also the bare rotation's
        // (4,19)/(4,20). The -1 kick -> x=2 -> (3,19),(3,20),(4,20),(3,21):
        // (4,20) is blocked, so -1 fails too. To make -1 succeed we must NOT
        // block (4,20). But then the bare rotation would succeed... T state 1
        // bare cells are (4,19),(4,20),(5,20),(4,21). If we block only (4,19)
        // and (5,20) and (6,20), then bare fails ((4,19) blocked), +1 ->
        // (5,19),(5,20),(6,20),(5,21): (5,20),(6,20) blocked, fails. -1 ->
        // (3,19),(3,20),(4,20),(3,21): all open. Succeeds with -1.
        b.set_cell(4, 19);
        b.set_cell(5, 20);
        b.set_cell(6, 20);
        let t = Piece::spawn(PieceType::T);
        let used = kick_used(&b, &t, true);
        assert_eq!(used, Some((-1, 0)), "should use the -1 left kick, got {:?}", used);
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
