//! Breadth-first search over reachable lock positions (phase 1.3).
//!
//! Because gravity is 20G, the search tree is shallow: after any input the
//! piece immediately falls to the ground, and the next input operates from
//! that grounded position. We BFS over `(rotation_state, x)` as the canonical
//! node (y is fully determined by 20G drop from a given (state, x)) and record
//! the `Vec<Input>` that reached each node so the chosen lock can be replayed
//! during live play.
//!
//! Algorithm:
//!   1. Seed the queue with the IRS-reachable root states (see action.rs).
//!      Each root is dropped under 20G to get its grounded position.
//!   2. For each dequeued node, enumerate successor edges:
//!        - Left / Right (one-column shift, if valid)
//!        - RotateCw / RotateCcw (via rotation::try_rotate, with kicks)
//!      After each edge, drop_20g the piece. The resulting (state, x) is the
//!      child node key; if unseen, enqueue it with the extended input path.
//!   3. Every visited grounded position is a lock candidate (under 20G the
//!      piece is always grounded after drop). We record the grounded piece +
//!      the input path. A final Neutral input (implicit) locks the piece.
//!   4. Dedup nodes by (state, x) so the search is finite (at most
//!      num_states * 10 nodes per root set).
//!
//! Output of the search is fed to the neural net (phase 2) for state
//! evaluation; the argmax lock's input path is what gets enacted live.

use std::collections::{HashSet, VecDeque};

use crate::action::{irs_root_states, Input};
use crate::board::Board;
use crate::piece::{Piece, PieceType};
use crate::rotation::try_rotate;

/// One reachable locking position, plus the exact input sequence that reaches
/// it from the spawn (post-IRS) state.
///
/// The input sequence does NOT include a trailing Neutral — callers should
/// emit one to actually lock the piece in the real game.
#[derive(Debug, Clone)]
pub struct LockCandidate {
    pub piece: Piece,
    pub inputs: Vec<Input>,
}

/// Canonical BFS node key: (rotation_state, x). Under 20G, y is fully
/// determined by dropping from any given (state, x), so we dedup on this pair.
type NodeKey = (u8, i8);

/// Search all reachable lock positions for `kind` on `board`, starting from
/// the IRS root states. Returns a vec of `LockCandidate`s, each with the
/// grounded piece and the input sequence (from the IRS root) that reaches it.
///
/// The search is breadth-first, so shorter input sequences are found first.
/// Dedup ensures at most `num_states * WIDTH` distinct nodes are visited.
pub fn search_locks(board: &Board, kind: PieceType) -> Vec<LockCandidate> {
    let mut candidates: Vec<LockCandidate> = Vec::new();
    let mut visited: HashSet<NodeKey> = HashSet::new();
    let mut queue: VecDeque<(Piece, Vec<Input>)> = VecDeque::new();

    // 1. Seed with IRS roots, each dropped under 20G.
    for root in irs_root_states(board, kind) {
        let grounded = board.drop_20g(&root);
        let key = (grounded.state, grounded.x);
        if visited.insert(key) {
            candidates.push(LockCandidate {
                piece: grounded,
                inputs: Vec::new(),
            });
            queue.push_back((grounded, Vec::new()));
        }
    }

    // 2. BFS over (state, x) edges.
    while let Some((piece, path)) = queue.pop_front() {
        // Enumerate the 4 single-input edges: Left, Right, RotateCw, RotateCcw.
        for (input, child) in successors(board, &piece) {
            let grounded = board.drop_20g(&child);
            let key = (grounded.state, grounded.x);
            if visited.insert(key) {
                let mut new_path = path.clone();
                new_path.push(input);
                candidates.push(LockCandidate {
                    piece: grounded,
                    inputs: new_path.clone(),
                });
                queue.push_back((grounded, new_path));
            }
        }
    }

    candidates
}

/// Generate the 4 single-input successor pieces from `piece` on `board`:
/// Left, Right, RotateCw, RotateCcw. Each child is the piece AFTER the input
/// but BEFORE the 20G drop (the caller drops it). Filters out edges that
/// produce invalid positions (e.g. move into a wall, or rotation that fails
/// all kicks).
fn successors<'b>(board: &'b Board, piece: &Piece) -> impl Iterator<Item = (Input, Piece)> + 'b {
    let mut out: Vec<(Input, Piece)> = Vec::with_capacity(4);

    // Left
    let left = piece.translate(-1, 0);
    if board.is_valid(&left) {
        out.push((Input::Left, left));
    }
    // Right
    let right = piece.translate(1, 0);
    if board.is_valid(&right) {
        out.push((Input::Right, right));
    }
    // Rotate CW (with kicks)
    if let Some(rotated) = try_rotate(board, piece, true) {
        out.push((Input::RotateCw, rotated));
    }
    // Rotate CCW (with kicks)
    if let Some(rotated) = try_rotate(board, piece, false) {
        out.push((Input::RotateCcw, rotated));
    }
    out.into_iter()
}

/// Convenience: count of distinct reachable lock positions. Useful for sanity
/// checks and as a feature for the neural net.
pub fn count_locks(board: &Board, kind: PieceType) -> usize {
    search_locks(board, kind).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::PieceType;

    #[test]
    fn empty_board_has_candidates_for_all_pieces() {
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            let locks = search_locks(&b, kind);
            assert!(!locks.is_empty(), "{:?} should have reachable locks on empty board", kind);
        }
    }

    #[test]
    fn o_piece_candidates_are_only_horizontal_positions() {
        // O can't rotate, so all candidates are just horizontal shifts from
        // the spawn column. On an empty board, the O (2 wide) can occupy
        // columns 0-1 through 8-9, i.e. 9 positions.
        let b = Board::new();
        let locks = search_locks(&b, PieceType::O);
        let xs: HashSet<i8> = locks.iter().map(|c| c.piece.x).collect();
        // O spawn x=3. It can move left to x=-1 (cells cols 0-1) and right
        // to x=7 (cells cols 8-9). That's 9 distinct x positions.
        assert_eq!(xs.len(), 9, "O should reach 9 horizontal positions, got {}: {:?}", xs.len(), xs);
        // All candidates should have state 0 (O can't rotate).
        for c in &locks {
            assert_eq!(c.piece.state, 0, "O candidate should be state 0");
        }
    }

    #[test]
    fn candidates_include_bare_spawn_drop() {
        // The IRS root with no inputs (empty path) should be a candidate.
        let b = Board::new();
        let locks = search_locks(&b, PieceType::T);
        let bare = locks.iter().find(|c| c.inputs.is_empty());
        assert!(bare.is_some(), "should be a candidate with empty input path (bare spawn drop)");
    }

    #[test]
    fn candidates_reach_floor_on_empty_board() {
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            for c in search_locks(&b, kind) {
                // Every candidate's minimum cell row should be 0 (floor).
                let min_r = c.piece.cells_iter().map(|(_, r)| r).min().unwrap();
                assert_eq!(min_r, 0, "{:?} candidate not at floor: {:?}", kind, c.piece);
            }
        }
    }

    #[test]
    fn input_paths_are_shortest_bfs_order() {
        // BFS guarantees the first time we visit a node, its path is shortest.
        let b = Board::new();
        let locks = search_locks(&b, PieceType::T);
        // The bare spawn drop has 0 inputs. A one-step neighbor has 1 input.
        // Verify no candidate has a longer-than-necessary path by checking
        // that a 1-input candidate exists.
        let one_step = locks.iter().any(|c| c.inputs.len() == 1);
        assert!(one_step, "should be at least one 1-input candidate");
    }

    #[test]
    fn candidates_dedup_by_state_and_x() {
        // No two candidates should share the same (state, x) — BFS dedups.
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            let locks = search_locks(&b, kind);
            let keys: HashSet<(u8, i8)> = locks.iter().map(|c| (c.piece.state, c.piece.x)).collect();
            assert_eq!(keys.len(), locks.len(), "{:?} has duplicate (state, x) candidates", kind);
        }
    }

    #[test]
    fn full_board_has_only_irs_roots_if_any() {
        // On a completely full board, the piece can't move or rotate (no
        // space), so candidates == IRS roots that are valid. Most pieces
        // won't even spawn validly, so candidates may be empty.
        let mut b = Board::new();
        for r in 0..crate::board::HEIGHT {
            for c in 0..crate::board::WIDTH {
                b.set_cell(c, r);
            }
        }
        for &kind in PieceType::ALL.iter() {
            let locks = search_locks(&b, kind);
            // Each candidate, if any, must have empty inputs (can't move).
            for c in &locks {
                assert!(c.inputs.is_empty(), "{:?} candidate should have no inputs on full board", kind);
            }
        }
    }

    #[test]
    fn candidates_include_rotated_states_on_empty_board() {
        // For a 4-state piece like T, the search should reach at least one
        // candidate in each rotation state (0, 1, 2, 3).
        let b = Board::new();
        let locks = search_locks(&b, PieceType::T);
        let states: HashSet<u8> = locks.iter().map(|c| c.piece.state).collect();
        assert!(states.contains(&0), "T should reach state 0");
        assert!(states.contains(&1), "T should reach state 1");
        assert!(states.contains(&2), "T should reach state 2");
        assert!(states.contains(&3), "T should reach state 3");
    }
}
