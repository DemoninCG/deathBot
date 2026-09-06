//! Bulk candidate locking: clone+lock+clear for every lock candidate in ONE
//! FFI crossing (phase D5 of the 2026-09 training plan).
//!
//! The Python trainer scores every candidate by cloning the board, locking,
//! and clearing — previously N FFI calls plus N separate `row_bits` calls.
//! This module does all of it in Rust and returns packed rows as u32
//! bitmasks (matching `Board.row_bits()`), so Python can vectorize the
//! encoding with numpy.

use crate::board::{Board, HEIGHT, WIDTH};
use crate::pathfind::{search_placements, LockCandidate};
use crate::piece::Piece;

/// Result rows for one candidate after lock+clear.
pub struct CandidateResult {
    /// 21 row bitmasks (bottom-first), post-lock post-clear.
    pub rows: [u32; 21],
    pub lines_cleared: usize,
}

/// Clone+lock+clear every candidate; returns packed rows in candidate order.
///
/// Mirrors `Board::lock` + `Board::clear_lines` semantics exactly: cells
/// above the field (r >= HEIGHT) are silently dropped by lock, and full rows
/// compact downward.
pub fn lock_candidates(board: &Board, candidates: &[LockCandidate]) -> Vec<CandidateResult> {
    candidates
        .iter()
        .map(|cand| {
            let mut rows = board.rows;
            for (c, r) in cand.piece.cells_iter() {
                if (0..WIDTH as i8).contains(&c) && (0..HEIGHT as i8).contains(&r) {
                    rows[r as usize] |= 1 << c;
                }
            }
            // Compact filled rows in place (same as Board::clear_lines).
            let mut cleared = 0usize;
            let mut write = 0usize;
            for read in 0..HEIGHT {
                if rows[read] & COL_MASK == COL_MASK {
                    cleared += 1;
                } else {
                    if write != read {
                        rows[write] = rows[read];
                    }
                    write += 1;
                }
            }
            for r in write..HEIGHT {
                rows[r] = 0;
            }
            CandidateResult { rows, lines_cleared: cleared }
        })
        .collect()
}

const COL_MASK: u32 = (1u32 << WIDTH) - 1;

/// Reconstruct a board from 21 row bitmasks (bottom-first).
pub fn board_from_rows(rows: &[u32]) -> Board {
    let mut b = Board::new();
    for (r, &bits) in rows.iter().take(HEIGHT).enumerate() {
        b.rows[r] = bits & COL_MASK;
    }
    b
}

/// Reconstruct many boards at once (item 4): one allocation for N boards,
/// avoiding N separate FFI crossings from Python.
pub fn boards_from_rows_flat(flat: &[u32]) -> Vec<Board> {
    flat.chunks(HEIGHT)
        .map(|rows| board_from_rows(rows))
        .collect()
}

/// Lock+clear for path-free placements (item 5 fast path): same semantics as
/// `lock_candidates` but takes grounded `Piece`s directly (from
/// `search_placements`), skipping LockCandidate/input-path construction.
pub fn lock_pieces(board: &Board, pieces: &[Piece]) -> Vec<CandidateResult> {
    pieces
        .iter()
        .map(|piece| {
            let mut rows = board.rows;
            for (c, r) in piece.cells_iter() {
                if (0..WIDTH as i8).contains(&c) && (0..HEIGHT as i8).contains(&r) {
                    rows[r as usize] |= 1 << c;
                }
            }
            let mut cleared = 0usize;
            let mut write = 0usize;
            for read in 0..HEIGHT {
                if rows[read] & COL_MASK == COL_MASK {
                    cleared += 1;
                } else {
                    if write != read {
                        rows[write] = rows[read];
                    }
                    write += 1;
                }
            }
            for r in write..HEIGHT {
                rows[r] = 0;
            }
            CandidateResult { rows, lines_cleared: cleared }
        })
        .collect()
}

/// Full scoring fast path (items 4/5): search placements + lock + clear +
/// lockout flags in one call, with no input-path allocation. Returns
/// (pieces, results, lockouts) in the same BFS order as `search_locks`.
pub fn search_and_lock(
    board: &Board,
    kind: crate::piece::PieceType,
) -> (Vec<Piece>, Vec<CandidateResult>, Vec<bool>) {
    let pieces = search_placements(board, kind);
    let results = lock_pieces(board, &pieces);
    let lockouts = pieces.iter().map(|p| board.locks_out(p)).collect();
    (pieces, results, lockouts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::pathfind::search_locks;
    use crate::piece::PieceType;

    #[test]
    fn bulk_matches_per_candidate_sim() {
        let mut b = Board::new();
        for c in 2..10 {
            b.set_cell(c, 0);
        }
        let cands = search_locks(&b, PieceType::O);
        assert!(!cands.is_empty());
        let results = lock_candidates(&b, &cands);
        assert_eq!(results.len(), cands.len());
        for (cand, res) in cands.iter().zip(results.iter()) {
            let mut sim = b.clone();
            sim.lock(&cand.piece);
            let lines = sim.clear_lines();
            assert_eq!(sim.rows, res.rows, "lock+clear rows parity for {cand:?}");
            assert_eq!(lines, res.lines_cleared, "lines parity for {cand:?}");
        }
    }

    #[test]
    fn bulk_on_empty_board_matches_sim() {
        let b = Board::new();
        for kind in [PieceType::I, PieceType::T, PieceType::O] {
            let cands = search_locks(&b, kind);
            let results = lock_candidates(&b, &cands);
            for (cand, res) in cands.iter().zip(results.iter()) {
                let mut sim = b.clone();
                sim.lock(&cand.piece);
                sim.clear_lines();
                assert_eq!(sim.rows, res.rows);
                assert_eq!(res.lines_cleared, 0);
            }
        }
    }
}