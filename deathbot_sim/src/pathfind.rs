//! Breadth-first search over reachable lock positions (phase 1.3 — TODO).
//!
//! Because gravity is 20G, the search tree is shallow: from any spawn state
//! the piece can only slide horizontally or rotate before it locks. We BFS
//! over (rotation_state, x) as the canonical node — y is fully determined by
//! 20G drop from a given (state, x) — and record the `Vec<Input>` that reached
//! each node so the chosen lock can be replayed during live play.
//!
//! Plan (to implement in 1.3):
//!   1. Seed the queue with the IRS-reachable root states (see action.rs).
//!   2. For each dequeued node, enumerate successor edges:
//!        - RotateCw / RotateCcw (via rotation::try_rotate)
//!        - Left / Right (one column, if valid)
//!      Each edge produces a new (state, x) and is labeled with an Input.
//!   3. After every edge, drop_20g the piece. If grounded, record a lock
//!      candidate (state, x, y, input_path). If still airborne (shouldn't
//!      happen under 20G unless kicked above the stack), keep exploring.
//!   4. De-dupe nodes by (state, x) so the search is finite.
//!   5. Return the set of all distinct lock candidates with their input paths.
//!
//! Output of the search is fed to the neural net (phase 2) for state
//! evaluation; the argmax lock's input path is what gets enacted live.

use crate::action::Input;
use crate::board::Board;
use crate::piece::{Piece, PieceType};

/// One reachable locking position, plus the exact input sequence that reaches
/// it from the spawn (post-IRS) state.
#[derive(Debug, Clone)]
pub struct LockCandidate {
    pub piece: Piece,
    pub inputs: Vec<Input>,
}

/// TODO(1.3): the actual BFS. Stub returns an empty vec so the crate compiles
/// and the Python module can be imported while 1.2 is still in flight.
pub fn search_locks(_board: &Board, _kind: PieceType) -> Vec<LockCandidate> {
    Vec::new()
}
