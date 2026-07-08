//! Action application: rotate -> move -> 20G gravity (phase 1.2).
//!
//! TAP input model per the phase-1 spec:
//! - Initial Rotation System (IRS): a rotation input applied before the first
//!   tick rotates the spawn piece for free. We seed the BFS root with all
//!   rotation states reachable from spawn via IRS.
//! - No Initial Movement System: TAP has no IRS-equivalent for horizontal
//!   movement before the first tick. The first horizontal move happens on-tick.
//! - Per-tick action ordering: rotate -> move -> 20G gravity. The piece then
//!   either locks (if it can't fall further) or remains live for the next tick.
//!
//! The `Input` enum below labels BFS edges so each reachable lock state can
//! be replayed as an exact input sequence during real gameplay.

use crate::board::Board;
use crate::piece::{Piece, PieceType};
use crate::rotation::try_rotate;

/// A single discrete input. Matches the buttons a TGM cabinet exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Input {
    RotateCw,
    RotateCcw,
    Left,
    Right,
    /// No input this tick; if the piece is grounded it locks.
    Neutral,
}

/// The action model the pathfinder and the live player both use. Each variant
/// is one tick's worth of input. Neutral with a grounded piece => lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TickAction {
    pub rotate: Option<RotationInput>,
    pub move_dir: Option<MoveInput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RotationInput {
    Cw,
    Ccw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoveInput {
    Left,
    Right,
}

impl TickAction {
    pub const NEUTRAL: Self = Self {
        rotate: None,
        move_dir: None,
    };

    /// All single-button tick actions (no combos): neutral, 2 rotations,
    /// 2 moves. The full combo space (rotate+move same tick) is 1 + 2 + 2 +
    /// 2*2 = 9; we expose that via `all_actions` below.
    pub const SINGLE: [Self; 5] = [
        Self::NEUTRAL,
        Self { rotate: Some(RotationInput::Cw), move_dir: None },
        Self { rotate: Some(RotationInput::Ccw), move_dir: None },
        Self { rotate: None, move_dir: Some(MoveInput::Left) },
        Self { rotate: None, move_dir: Some(MoveInput::Right) },
    ];
}

/// All 9 tick actions: neutral + 2 rotations + 2 moves + 4 rotate+move combos.
/// The BFS enumerates these at every node. (TAP allows rotate+move on the
/// same tick since they're independent buttons.)
pub fn all_actions() -> Vec<TickAction> {
    let mut v = Vec::with_capacity(9);
    v.push(TickAction::NEUTRAL);
    for r in [RotationInput::Cw, RotationInput::Ccw] {
        v.push(TickAction { rotate: Some(r), move_dir: None });
        for m in [MoveInput::Left, MoveInput::Right] {
            v.push(TickAction { rotate: Some(r), move_dir: Some(m) });
        }
    }
    for m in [MoveInput::Left, MoveInput::Right] {
        v.push(TickAction { rotate: None, move_dir: Some(m) });
    }
    v
}

/// Result of applying one tick's action to a live piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickOutcome {
    /// The piece after rotate -> move -> 20G. If rotation failed, the piece
    /// is unchanged in rotation but may still move and drop.
    pub piece: Piece,
    /// True if the piece cannot fall further (i.e. drop_20g was a no-op).
    /// A grounded piece + Neutral input on the next tick = lock.
    pub grounded: bool,
}

/// Apply one tick's action to `p` in the order: rotate -> move -> 20G gravity.
///
/// Rotation uses the ARS wall-kick table (right kick first, then left). If
/// rotation fails entirely, the piece keeps its current state. Movement is
/// a single-column shift; if blocked, the piece stays put. Finally the piece
/// drops under 20G and we report whether it's grounded.
pub fn apply_tick(board: &Board, p: &Piece, action: TickAction) -> TickOutcome {
    // 1. Rotate (with kicks).
    let mut cur = *p;
    if let Some(r) = action.rotate {
        let cw = matches!(r, RotationInput::Cw);
        if let Some(rotated) = try_rotate(board, &cur, cw) {
            cur = rotated;
        }
        // If rotation failed, cur stays as-is (state unchanged).
    }

    // 2. Move one column (no IHS in TAP — horizontal move is on-tick only).
    if let Some(m) = action.move_dir {
        let dx: i8 = match m {
            MoveInput::Left => -1,
            MoveInput::Right => 1,
        };
        let moved = cur.translate(dx, 0);
        if board.is_valid(&moved) {
            cur = moved;
        }
    }

    // 3. 20G gravity.
    let grounded_before = !board.is_valid(&cur.translate(0, -1));
    let dropped = board.drop_20g(&cur);
    let grounded = !board.is_valid(&dropped.translate(0, -1));
    // If the piece couldn't drop at all, grounded == grounded_before == true.
    let _ = grounded_before; // kept for clarity/debugging
    TickOutcome { piece: dropped, grounded }
}

/// The IRS root states for a freshly spawned piece: the spawn state plus all
/// rotation states reachable by free pre-tick rotations (IRS).
///
/// TAP allows the player to hold a rotation button before the piece spawns;
/// the piece then appears already rotated, at no input cost. We model this by
/// seeding the BFS with every distinct state the spawn piece can be in.
///
/// Returns a vec of (state, piece) pairs — one per distinct rotation state
/// (O has 1, I has 2, J/L/S/T/Z have 4). The piece coordinates are the spawn
/// coordinates; IRS does not move the piece horizontally.
pub fn irs_root_states(board: &Board, kind: PieceType) -> Vec<Piece> {
    let spawn = Piece::spawn(kind);
    let n = kind.num_states();
    let mut roots = Vec::with_capacity(n as usize);
    let mut seen = std::collections::HashSet::new();
    // Try every state directly — IRS can reach any state, but the spawn
    // coordinates may make some states invalid (e.g. a piece whose rotated
    // form pokes out the side at spawn x). Skip those.
    for s in 0..n {
        let candidate = Piece::new(kind, s, spawn.x, spawn.y);
        if board.is_valid(&candidate) && seen.insert((s, spawn.x, spawn.y)) {
            roots.push(candidate);
        }
    }
    // Also probe states reachable via kicks from spawn (in case a bare
    // rotated spawn is invalid but a kick makes it valid — rare but possible
    // at the wall). We chase rotations BFS-style from spawn.
    let mut frontier: Vec<Piece> = roots.clone();
    while let Some(p) = frontier.pop() {
        for cw in [true, false] {
            if let Some(r) = try_rotate(board, &p, cw) {
                let key = (r.state, r.x, r.y);
                if seen.insert(key) {
                    roots.push(r);
                    frontier.push(r);
                }
            }
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::PieceType;

    #[test]
    fn all_actions_has_nine_entries() {
        let v = all_actions();
        assert_eq!(v.len(), 9, "expected 9 tick actions (neutral + 2 rot + 2 move + 4 combos)");
    }

    #[test]
    fn neutral_on_grounded_piece_is_ground_state() {
        let b = Board::new();
        let p = Piece::spawn(PieceType::T);
        // First drop the piece to ground it.
        let grounded_piece = b.drop_20g(&p);
        let out = apply_tick(&b, &grounded_piece, TickAction::NEUTRAL);
        assert!(out.grounded, "neutral on grounded piece should report grounded");
        assert_eq!(out.piece, grounded_piece);
    }

    #[test]
    fn move_then_drop_translates_piece() {
        let b = Board::new();
        let p = Piece::spawn(PieceType::O);
        let out = apply_tick(&b, &p, TickAction { rotate: None, move_dir: Some(MoveInput::Left) });
        assert!(out.grounded, "O should be grounded after one tick on empty board");
        assert_eq!(out.piece.x, p.x - 1, "O should have moved left one column");
    }

    #[test]
    fn rotate_cw_changes_state_on_empty_board() {
        let b = Board::new();
        let p = Piece::spawn(PieceType::T);
        let out = apply_tick(&b, &p, TickAction { rotate: Some(RotationInput::Cw), move_dir: None });
        assert_eq!(out.piece.state, 1, "T should be in state 1 after CW rotation");
    }

    #[test]
    fn irs_roots_count_matches_num_states_on_empty_board() {
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            let roots = irs_root_states(&b, kind);
            assert_eq!(
                roots.len(),
                kind.num_states() as usize,
                "{:?} should have {} IRS root states on empty board, got {}",
                kind,
                kind.num_states(),
                roots.len()
            );
        }
    }
}
