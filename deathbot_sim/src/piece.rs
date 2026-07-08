//! Piece type, rotation state, and grid coordinates (phase 1.1).
//!
//! ARS uses a 4-state table for J/L/S/T/Z, a 2-state table for I, and a
//! 1-state table for O. We store explicit per-state cell offsets (relative to
//! the piece origin `(x, y)`) and index states modulo `num_states`, so e.g. the
//! I piece's state 2 wraps back to state 0.
//!
//! NOTE: The cell-offset tables below are the standard SRS spawn layouts and
//! serve as a placeholder. Phase 1.2 will replace rotation *transitions* and
//! the wall-kick table with the verified Arika Rotation System set; the spawn
//! (state 0) flat orientations are the same in ARS, so only the rotated states
//! need re-verification.

/// The seven tetromino types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    I = 0,
    O = 1,
    T = 2,
    S = 3,
    Z = 4,
    J = 5,
    L = 6,
}

impl PieceType {
    pub const ALL: [PieceType; 7] = [
        PieceType::I,
        PieceType::O,
        PieceType::T,
        PieceType::S,
        PieceType::Z,
        PieceType::J,
        PieceType::L,
    ];

    /// Number of distinct rotation states: 4 for J/L/T, 2 for I/S/Z, 1 for O.
    /// S and Z have only 2 states in ARS (they toggle between horizontal and
    /// vertical); their state 2 wraps to state 0, state 3 wraps to state 1.
    pub fn num_states(self) -> u8 {
        match self {
            PieceType::I => 2,
            PieceType::O => 1,
            PieceType::S => 2,
            PieceType::Z => 2,
            _ => 4, // J, L, T
        }
    }

    /// Cell offsets `(dx, dy)` for a given rotation state, relative to the
    /// piece origin. `state` is taken modulo `num_states`.
    pub fn cells(self, state: u8) -> &'static [(i8, i8)] {
        let s = state % self.num_states();
        match (self, s) {
            (PieceType::I, 0) => &I_0,
            (PieceType::I, 1) => &I_1,
            (PieceType::O, 0) => &O_0,
            (PieceType::T, 0) => &T_0,
            (PieceType::T, 1) => &T_1,
            (PieceType::T, 2) => &T_2,
            (PieceType::T, 3) => &T_3,
            (PieceType::S, 0) => &S_0,
            (PieceType::S, 1) => &S_1,
            (PieceType::Z, 0) => &Z_0,
            (PieceType::Z, 1) => &Z_1,
            (PieceType::J, 0) => &J_0,
            (PieceType::J, 1) => &J_1,
            (PieceType::J, 2) => &J_2,
            (PieceType::J, 3) => &J_3,
            (PieceType::L, 0) => &L_0,
            (PieceType::L, 1) => &L_1,
            (PieceType::L, 2) => &L_2,
            (PieceType::L, 3) => &L_3,
            _ => unreachable!(),
        }
    }

    /// TAP spawn column for the piece origin. Pieces spawn roughly centered.
    /// TODO(1.2): verify exact TAP spawn x/y against the game.
    pub fn spawn_x(self) -> i8 {
        // Origin offset; cells are relative. 3 is the conventional center spawn.
        3
    }
}

// I piece — 2 states. Horizontal / vertical.
static I_0: [(i8, i8); 4] = [(0, 2), (1, 2), (2, 2), (3, 2)];
static I_1: [(i8, i8); 4] = [(2, 0), (2, 1), (2, 2), (2, 3)];

// O piece — 1 state. Immovable 2x2 block.
static O_0: [(i8, i8); 4] = [(1, 0), (2, 0), (1, 1), (2, 1)];

// T piece — 4 states.
static T_0: [(i8, i8); 4] = [(1, 0), (0, 1), (1, 1), (2, 1)];
static T_1: [(i8, i8); 4] = [(1, 0), (1, 1), (2, 1), (1, 2)];
static T_2: [(i8, i8); 4] = [(0, 1), (1, 1), (2, 1), (1, 2)];
static T_3: [(i8, i8); 4] = [(1, 0), (0, 1), (1, 1), (1, 2)];

// S piece — 2 states (horizontal / vertical).
static S_0: [(i8, i8); 4] = [(1, 0), (2, 0), (0, 1), (1, 1)];
static S_1: [(i8, i8); 4] = [(1, 0), (1, 1), (2, 1), (2, 2)];

// Z piece — 2 states (horizontal / vertical).
static Z_0: [(i8, i8); 4] = [(0, 0), (1, 0), (1, 1), (2, 1)];
static Z_1: [(i8, i8); 4] = [(1, 0), (0, 1), (1, 1), (0, 2)];

// J piece — 4 states.
static J_0: [(i8, i8); 4] = [(0, 0), (0, 1), (1, 1), (2, 1)];
static J_1: [(i8, i8); 4] = [(1, 0), (2, 0), (1, 1), (1, 2)];
static J_2: [(i8, i8); 4] = [(0, 0), (1, 0), (2, 0), (2, 1)];
static J_3: [(i8, i8); 4] = [(1, 0), (1, 1), (0, 2), (1, 2)];

// L piece — 4 states.
static L_0: [(i8, i8); 4] = [(2, 0), (0, 1), (1, 1), (2, 1)];
static L_1: [(i8, i8); 4] = [(1, 0), (1, 1), (1, 2), (2, 2)];
static L_2: [(i8, i8); 4] = [(0, 0), (1, 0), (2, 0), (0, 1)];
static L_3: [(i8, i8); 4] = [(0, 0), (1, 0), (1, 1), (1, 2)];

/// A live piece on the field: type, current rotation state, and origin coords.
///
/// Coordinates use signed integers so wall-kick probes with negative offsets
/// don't underflow. `(x, y)` is the piece origin; actual cell positions are
/// `(x + dx, y + dy)` for each `(dx, dy)` in `PieceType::cells(state)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub kind: PieceType,
    pub state: u8,
    pub x: i8,
    pub y: i8,
}

impl Piece {
    pub fn new(kind: PieceType, state: u8, x: i8, y: i8) -> Self {
        Self { kind, state, x, y }
    }

    /// Spawn a piece in state 0 at its conventional spawn column.
    ///
    /// Spawn y is chosen so the piece's top cell (max dy in state 0) sits in
    /// the invisible buffer row (row 20). Most pieces have max dy = 1, so
    /// y = 19 puts their top cell at row 20. The I piece has max dy = 2, so
    /// it spawns at y = 18 to keep its top cell at row 20.
    /// TODO: verify exact TAP spawn x/y against the game; some pieces may
    /// spawn with a different y offset.
    pub fn spawn(kind: PieceType) -> Self {
        let max_dy = kind.cells(0).iter().map(|&(_, dy)| dy).max().unwrap_or(0);
        let y = 20 - max_dy; // top cell lands at row 20 (invisible buffer)
        Self::new(kind, 0, kind.spawn_x(), y)
    }

    /// Iterate absolute cell positions `(col, row)` occupied by this piece.
    pub fn cells_iter(&self) -> impl Iterator<Item = (i8, i8)> + '_ {
        self.kind
            .cells(self.state)
            .iter()
            .map(move |&(dx, dy)| (self.x + dx, self.y + dy))
    }

    /// Rotate clockwise (state+1 modulo num_states). Does not apply kicks.
    pub fn rotate_cw(&self) -> Self {
        let n = self.kind.num_states();
        Self::new(self.kind, (self.state + 1) % n, self.x, self.y)
    }

    /// Rotate counter-clockwise (state-1 modulo num_states). No kicks.
    pub fn rotate_ccw(&self) -> Self {
        let n = self.kind.num_states();
        Self::new(self.kind, (self.state + n - 1) % n, self.x, self.y)
    }

    /// Translate by `(dx, dy)`.
    pub fn translate(&self, dx: i8, dy: i8) -> Self {
        Self::new(self.kind, self.state, self.x + dx, self.y + dy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_state_wrap() {
        assert_eq!(PieceType::I.num_states(), 2);
        assert_eq!(PieceType::O.num_states(), 1);
        assert_eq!(PieceType::S.num_states(), 2);
        assert_eq!(PieceType::Z.num_states(), 2);
        assert_eq!(PieceType::T.num_states(), 4);
        assert_eq!(PieceType::J.num_states(), 4);
        assert_eq!(PieceType::L.num_states(), 4);
        // I wraps state 2 -> 0.
        let p = Piece::new(PieceType::I, 0, 0, 0);
        assert_eq!(p.rotate_cw().state, 1);
        assert_eq!(p.rotate_cw().rotate_cw().state, 0);
        // S wraps state 2 -> 0.
        let s = Piece::new(PieceType::S, 0, 0, 0);
        assert_eq!(s.rotate_cw().state, 1);
        assert_eq!(s.rotate_cw().rotate_cw().state, 0);
        // O never changes state.
        let o = Piece::new(PieceType::O, 0, 0, 0);
        assert_eq!(o.rotate_cw().state, 0);
    }

    #[test]
    fn cells_count_is_four() {
        for &kind in PieceType::ALL.iter() {
            for s in 0..4u8 {
                assert_eq!(kind.cells(s).len(), 4, "{:?} state {}", kind, s);
            }
        }
    }
}
