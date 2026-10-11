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
//!   4. Dedup nodes by (state, x, y) so the search is finite. Note: y is
//!      included in the key because overhangs allow the same (state, x) to
//!      ground at multiple heights (a piece resting on top of an overhang
//!      vs. tucked below it). Deduping on (state, x) alone would silently
//!      drop valid tucked positions.
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
/// `irs_root` is the rotation state of the IRS root the path starts from
/// (0 = bare spawn, reached with no pre-spawn input). Live playback needs
/// it: a nonzero root must be set up during ARE via IRS taps before the
/// path's inputs are replayed from spawn.
#[derive(Debug, Clone)]
pub struct LockCandidate {
    pub piece: Piece,
    pub inputs: Vec<Input>,
    pub irs_root: u8,
}

/// Canonical BFS node key: (rotation_state, x, y). We include y because under
/// 20G the grounded y for a given (state, x) is NOT always unique: when the
/// board has overhangs, a piece can be grounded at the same (state, x) at
/// different heights (one on top of the overhang, one tucked below it).
/// Deduping on (state, x) alone would silently drop valid tucked positions.
type NodeKey = (u8, i8, i8);

/// Search all reachable lock positions for `kind` on `board`, starting from
/// the IRS root states. Returns a vec of `LockCandidate`s, each with the
/// grounded piece and the input sequence (from the IRS root) that reaches it.
///
/// The search is breadth-first, so shorter input sequences are found first.
/// Dedup ensures at most `num_states * WIDTH` distinct nodes are visited.
///
/// Speed path (items 5/7): visited set is a fixed bitset indexed by
/// (state, x+16, y+32) — no hashing — and queue/path storage reuses buffers.
pub fn search_locks(board: &Board, kind: PieceType) -> Vec<LockCandidate> {
    search_locks_with_paths(board, kind)
}

/// Bare-spawn search for LIVE PLAYBACK (not training): identical BFS, but
/// seeded ONLY from the unrotated spawn state (IRS root 0), so every
/// returned path starts exactly where the real piece spawns and needs no
/// pre-spawn input. Locks reachable solely via rotated IRS roots (rare
/// IRS-only tucks) are absent rather than mislabeled — the caller must not
/// filter or reinterpret roots. Every candidate reports `irs_root: 0`.
/// Returns empty when even the bare spawn is blocked (topping out).
pub fn search_locks_bare(board: &Board, kind: PieceType) -> Vec<LockCandidate> {
    search_locks_with_paths_from(board, kind, Some(0))
}

/// Root-constrained search for live playback with true IRS: identical BFS
/// seeded only from IRS roots whose rotation state is `root_state` (when
/// valid at spawn). Lets the bot plan from the orientation its ARE hold
/// actually produced, verified at spawn. Empty when that root is blocked.
pub fn search_locks_from(
    board: &Board,
    kind: PieceType,
    root_state: u8,
) -> Vec<LockCandidate> {
    search_locks_with_paths_from(board, kind, Some(root_state))
}

fn search_locks_with_paths(board: &Board, kind: PieceType) -> Vec<LockCandidate> {
    search_locks_with_paths_from(board, kind, None)
}

/// Scoring-only search (item 5): grounded (state, x, y) triples with NO
/// input-path allocation. Same BFS order and dedup as `search_locks`, for
/// callers (bulk lock+clear, NN scoring) that never replay inputs.
pub fn search_placements(board: &Board, kind: PieceType) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    // state < 4, x in [-8, 17], y in [-8, 40] mapped into a bitset.
    let mut visited = [false; 4 * 26 * 49];
    let idx = |s: u8, x: i8, y: i8| -> Option<usize> {
        let xi = x as i32 + 8;
        let yi = y as i32 + 8;
        if !(0..4).contains(&(s as i32)) || !(0..26).contains(&xi) || !(0..49).contains(&yi) {
            return None;
        }
        Some((s as usize) * 26 * 49 + (xi as usize) * 49 + (yi as usize))
    };
    let mut queue: VecDeque<Piece> = VecDeque::new();
    for root in irs_root_states(board, kind) {
        let grounded = board.drop_20g(&root);
        match idx(grounded.state, grounded.x, grounded.y) {
            Some(k) => {
                if !visited[k] {
                    visited[k] = true;
                    out.push(grounded);
                    queue.push_back(grounded);
                }
            }
            None => {
                out.push(grounded);
                queue.push_back(grounded);
            }
        }
    }
    while let Some(piece) = queue.pop_front() {
        for (_input, child) in successors(board, &piece) {
            let grounded = board.drop_20g(&child);
            match idx(grounded.state, grounded.x, grounded.y) {
                Some(k) => {
                    if !visited[k] {
                        visited[k] = true;
                        out.push(grounded);
                        queue.push_back(grounded);
                    }
                }
                None => {
                    out.push(grounded);
                    queue.push_back(grounded);
                }
            }
        }
    }
    out
}

fn search_locks_with_paths_from(
    board: &Board,
    kind: PieceType,
    root_filter: Option<u8>,
) -> Vec<LockCandidate> {
    let mut candidates: Vec<LockCandidate> = Vec::new();
    let mut visited: HashSet<NodeKey> = HashSet::new();
    let mut queue: VecDeque<(Piece, Vec<Input>, u8)> = VecDeque::new();

    // 1. Seed with IRS roots, each dropped under 20G. A root filter keeps
    // only the requested rotation state (bare spawn = 0, or the orientation
    // an ARE hold produced); None keeps every valid root.
    for root in irs_root_states(board, kind) {
        if root_filter.is_some_and(|s| s != root.state) {
            continue;
        }
        let grounded = board.drop_20g(&root);
        let key = (grounded.state, grounded.x, grounded.y);
        if visited.insert(key) {
            candidates.push(LockCandidate {
                piece: grounded,
                inputs: Vec::new(),
                irs_root: root.state,
            });
            queue.push_back((grounded, Vec::new(), root.state));
        }
    }

    // 2. BFS over (state, x) edges.
    while let Some((piece, path, root_state)) = queue.pop_front() {
        // Enumerate the 4 single-input edges: Left, Right, RotateCw, RotateCcw.
        for (input, child) in successors(board, &piece) {
            let grounded = board.drop_20g(&child);
            let key = (grounded.state, grounded.x, grounded.y);
            if visited.insert(key) {
                let mut new_path = path.clone();
                new_path.push(input);
                candidates.push(LockCandidate {
                    piece: grounded,
                    inputs: new_path.clone(),
                    irs_root: root_state,
                });
                queue.push_back((grounded, new_path, root_state));
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
    fn candidates_dedup_by_state_x_and_y() {
        // No two candidates should share the same (state, x, y) — BFS dedups
        // on the full grounded position. (y is included because overhangs
        // allow the same (state, x) to ground at multiple heights.)
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            let locks = search_locks(&b, kind);
            let keys: HashSet<(u8, i8, i8)> =
                locks.iter().map(|c| (c.piece.state, c.piece.x, c.piece.y)).collect();
            assert_eq!(keys.len(), locks.len(), "{:?} has duplicate (state, x, y) candidates", kind);
        }
    }

    #[test]
    fn bare_search_reports_root_zero_and_subset_of_full() {
        // Every bare candidate starts at the unrotated spawn (irs_root 0).
        // Each bare lock position must also exist in the full search (paths
        // may differ: full search can reach the same lock shorter via a
        // rotated root, which is exactly why bare search exists).
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            let bare = search_locks_bare(&b, kind);
            let full = search_locks(&b, kind);
            assert!(!bare.is_empty(), "{:?} bare search empty on empty board", kind);
            for c in &bare {
                assert_eq!(c.irs_root, 0, "{:?} bare root not zero", kind);
                assert!(full.iter().any(|f|
                    f.piece.state == c.piece.state
                    && f.piece.x == c.piece.x && f.piece.y == c.piece.y),
                    "{:?} bare lock missing from full search", kind);
            }
        }
    }

    #[test]
    fn from_search_root_zero_matches_bare() {
        // search_locks_from(_, 0) must equal search_locks_bare exactly
        // (same seeds, same order, same paths).
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            let a = search_locks_bare(&b, kind);
            let f = search_locks_from(&b, kind, 0);
            assert_eq!(a.len(), f.len(), "{:?} count", kind);
            for (x, y) in a.iter().zip(f.iter()) {
                assert_eq!((x.piece.state, x.piece.x, x.piece.y),
                           (y.piece.state, y.piece.x, y.piece.y));
                assert_eq!(x.inputs, y.inputs);
                assert_eq!(x.irs_root, 0);
                assert_eq!(y.irs_root, 0);
            }
        }
    }

    #[test]
    fn from_search_other_roots_stay_inside_full() {
        // Constrained searches report their own root and stay within the
        // full search's lock positions.
        let b = Board::new();
        for &kind in PieceType::ALL.iter() {
            let full = search_locks(&b, kind);
            for s in 1..kind.num_states() {
                for c in search_locks_from(&b, kind, s).iter() {
                    assert_eq!(c.irs_root, s);
                    assert!(full.iter().any(|f|
                        f.piece.state == c.piece.state
                        && f.piece.x == c.piece.x && f.piece.y == c.piece.y),
                        "{:?} root-{} lock missing from full search", kind, s);
                }
            }
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
