//! Integration tests for the deathbot_sim crate.
//!
//! These run via `cargo test` and exercise the public API end-to-end. Phase
//! 1.1 covers board + piece basics; 1.2 and 1.3 will add rotation-kick and
//! BFS-path tests here.

use deathbot_sim::board::Board;
use deathbot_sim::piece::{Piece, PieceType};

#[test]
fn spawn_and_drop_all_pieces() {
    for &kind in PieceType::ALL.iter() {
        let b = Board::new();
        let p = Piece::spawn(kind);
        assert!(b.is_valid(&p), "{:?} spawn invalid on empty board", kind);
        let g = b.drop_20g(&p);
        // After 20G on an empty board, the lowest cell must be at row 0.
        let min_r = g.cells_iter().map(|(_c, r)| r).min().unwrap();
        assert_eq!(min_r, 0, "{:?} did not reach bottom under 20G", kind);
    }
}

#[test]
fn lockout_on_full_board() {
    let mut b = Board::new();
    const COL_MASK: u32 = (1u32 << 10) - 1;
    for r in 0..21 {
        b.rows[r] = COL_MASK;
    }
    for &kind in PieceType::ALL.iter() {
        let p = Piece::spawn(kind);
        assert!(
            b.locks_out(&p) || !b.is_valid(&p),
            "{:?} should lock out or be invalid on a full board",
            kind
        );
    }
}

#[test]
fn line_clear_compacts_above_rows() {
    let mut b = Board::new();
    const COL_MASK: u32 = (1u32 << 10) - 1;
    // Row 0 full except cols 0-1. Drop an O at x=-1: its bottom cells (cols
    // 0-1, row 0) complete row 0, and its upper cells (cols 0-1, row 1)
    // remain. After clearing row 0, those upper bits compact into row 0.
    b.rows[0] = COL_MASK & !0b11;
    let o = Piece::new(PieceType::O, 0, -1, 20);
    let g = b.drop_20g(&o);
    b.lock(&g);
    let cleared = b.clear_lines();
    assert_eq!(cleared, 1);
    // After clearing row 0, the O's upper cells (cols 0,1 = 0b11) fall to row 0.
    assert_eq!(b.rows[0], 0b11);
    assert_eq!(b.rows[1], 0);
}
