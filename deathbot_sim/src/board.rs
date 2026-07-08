//! 10x21 bitboard + collision / line-clear / lockout logic (phase 1.1).
//!
//! Layout:
//! - 21 rows total. Row 0 = bottom (visible). Rows 0..=19 = the 20 visible
//!   rows. Row 20 = the single invisible buffer row above the field.
//! - Anything forced above row 20 (i.e. row >= 21) is "deleted" per TAP rules
//!   and constitutes a lockout/game-over.
//! - Each row is a `u32`; bits 0..=9 represent columns 0..=9. Bits 10..=31
//!   are unused slack, which lets us OR a piece's column mask in without
//!   masking and detect overflow cheaply via `row & FULL == FULL`.

use crate::piece::Piece;

pub const WIDTH: usize = 10;
pub const VISIBLE_ROWS: usize = 20;
pub const HEIGHT: usize = VISIBLE_ROWS + 1; // 21: includes one invisible row.
pub const INVISIBLE_ROW: usize = VISIBLE_ROWS; // index 20.

const COL_MASK: u32 = (1u32 << WIDTH) - 1; // 0b1111111111

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    /// `rows[r]` bit r-th row. `rows[0]` is the bottom.
    pub rows: [u32; HEIGHT],
}

impl Default for Board {
    fn default() -> Self {
        Self { rows: [0; HEIGHT] }
    }
}

impl Board {
    pub fn new() -> Self {
        Self::default()
    }

    /// True if column `c` is set in row `r`. Returns false for out-of-range.
    pub fn cell(&self, c: i8, r: i8) -> bool {
        if (0..WIDTH as i8).contains(&c) && (0..HEIGHT as i8).contains(&r) {
            self.rows[r as usize] & (1 << c) != 0
        } else {
            false
        }
    }

    /// Set a single cell (col, row). No-op if out of range. Useful for
    /// building test setups and pathfinder "what-if" probes.
    pub fn set_cell(&mut self, col: usize, row: usize) {
        if col < WIDTH && row < HEIGHT {
            self.rows[row] |= 1 << col;
        }
    }

    /// A piece position is valid iff every cell is in-bounds horizontally and
    /// does not overlap an existing block. Cells above the field (r >= HEIGHT)
    /// are allowed *only* when probing spawns/kicks — they indicate the piece
    /// is sticking out the top, which `locks_out` will later flag as game over.
    /// Cells below 0 are never valid.
    pub fn is_valid(&self, p: &Piece) -> bool {
        for &(dx, dy) in p.kind.cells(p.state) {
            let c = p.x + dx;
            let r = p.y + dy;
            if c < 0 || c >= WIDTH as i8 {
                return false;
            }
            if r < 0 {
                return false;
            }
            // r may be >= HEIGHT during a high spawn probe; that's allowed by
            // is_valid but will be caught by locks_out at lock time.
            if (r as usize) < HEIGHT && self.rows[r as usize] & (1 << c) != 0 {
                return false;
            }
        }
        true
    }

    /// True if locking `p` would put any cell strictly above the invisible row
    /// (r >= HEIGHT). In TAP this is the "block out" / stack-overflow death.
    pub fn locks_out(&self, p: &Piece) -> bool {
        p.cells_iter().any(|(_c, r)| r >= HEIGHT as i8)
    }

    /// Drop the piece under 20G gravity: shift it down as far as it stays
    /// valid. Returns the grounded piece. Assumes the piece is currently
    /// valid; if not, returns it unchanged.
    pub fn drop_20g(&self, p: &Piece) -> Piece {
        let mut cur = *p;
        loop {
            let next = cur.translate(0, -1);
            if self.is_valid(&next) {
                cur = next;
            } else {
                break;
            }
        }
        cur
    }

    /// Merge a grounded piece into the board. Does NOT clear lines. Caller is
    /// responsible for checking `locks_out` first if game-over detection is
    /// needed. Cells above the field are simply ignored (already "deleted").
    pub fn lock(&mut self, p: &Piece) {
        for (c, r) in p.cells_iter() {
            if (0..WIDTH as i8).contains(&c) && (0..HEIGHT as i8).contains(&r) {
                self.rows[r as usize] |= 1 << c;
            }
        }
    }

    /// Clear all full rows and return how many were cleared. Rows above each
    /// cleared row shift down by the number of cleared rows below them.
    pub fn clear_lines(&mut self) -> usize {
        let mut cleared = 0usize;
        // Walk from bottom up; compact filled-removed rows in place.
        let mut write = 0usize;
        for read in 0..HEIGHT {
            if self.rows[read] & COL_MASK == COL_MASK {
                cleared += 1;
            } else {
                if write != read {
                    self.rows[write] = self.rows[read];
                }
                write += 1;
            }
        }
        // Zero out the rows at the top that were vacated.
        for r in write..HEIGHT {
            self.rows[r] = 0;
        }
        cleared
    }

    /// Height of the stack in column `c` (0..WIDTH), counting from the bottom.
    /// Returns 0 for an empty column.
    pub fn column_height(&self, c: usize) -> usize {
        assert!(c < WIDTH);
        for r in (0..HEIGHT).rev() {
            if self.rows[r] & (1 << c) != 0 {
                return r + 1;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::PieceType;

    #[test]
    fn empty_board_valid_spawn() {
        let b = Board::new();
        let p = Piece::spawn(PieceType::T);
        assert!(b.is_valid(&p));
        assert!(!b.locks_out(&p));
    }

    #[test]
    fn drop_20g_lands_at_bottom() {
        let b = Board::new();
        let p = Piece::spawn(PieceType::O).translate(0, 0);
        let g = b.drop_20g(&p);
        // O spawn cells occupy rows 20-21 of origin => with y=20 that's rows
        // 20 and 21. After 20G the bottom cells should be at row 0/1.
        let bottom_row = g.cells_iter().map(|(_, r)| r).min().unwrap();
        assert_eq!(bottom_row, 0);
    }

    #[test]
    fn lock_then_clear_single_line() {
        let mut b = Board::new();
        // Build a near-full bottom row missing columns 0 and 1, then drop an O
        // so its bottom cells complete the line. O at x=-1 places cells in
        // cols 0-1. The O is 2 rows tall, so its upper cells also land in
        // row 1 (cols 0-1); after clearing row 0 that content compacts down.
        for c in 2..WIDTH {
            b.rows[0] |= 1 << c;
        }
        let o = Piece::new(PieceType::O, 0, -1, 20);
        let g = b.drop_20g(&o);
        b.lock(&g);
        assert_eq!(b.rows[0] & COL_MASK, COL_MASK, "row 0 should be full");
        assert_eq!(b.rows[1], 0b11, "row 1 should hold the O's upper cells");
        assert_eq!(b.clear_lines(), 1);
        // After clearing row 0, row 1's content (cols 0,1) falls into row 0.
        assert_eq!(b.rows[0], 0b11, "row 1 content should compact to row 0");
        assert_eq!(b.rows[1], 0, "row 1 should now be empty");
    }

    #[test]
    fn lockout_detected_when_above_field() {
        let mut b = Board::new();
        // Fill everything up to and including the invisible row.
        for r in 0..HEIGHT {
            b.rows[r] = COL_MASK;
        }
        // The I piece in spawn state 0 is flat (all cells dy=1). At spawn
        // y=19 its cells sit in row 20 (the invisible row, in-field), so on a
        // full board it is *invalid* (overlaps) but NOT locked out — no cell
        // is above the field. lockout is specifically about cells at row >=
        // HEIGHT (i.e. row 21+, the "deleted" region per TAP rules).
        let p = Piece::spawn(PieceType::I);
        assert!(!b.is_valid(&p), "spawn overlaps a full board");
        assert!(!b.locks_out(&p), "flat I at row 20 is not above the field");

        // A piece whose cells actually sit above the field (row >= 21) IS a
        // lockout, regardless of board contents.
        let high = Piece::new(PieceType::I, 0, 3, 20); // cells at row 21
        assert!(b.locks_out(&high));

        // A tall piece (e.g. I vertical, cells dy 0..3) at y=20 puts its top
        // cell at row 23 — also above the field.
        let vertical = Piece::new(PieceType::I, 1, 3, 20);
        assert!(b.locks_out(&vertical));

        // Sanity: the same vertical I at y=17 has top cell at row 20 — in the
        // invisible row, not above, so not a lockout on an empty board.
        let empty = Board::new();
        let in_field = Piece::new(PieceType::I, 1, 3, 17);
        assert!(!empty.locks_out(&in_field));
    }
}
