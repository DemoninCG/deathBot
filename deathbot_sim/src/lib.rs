//! Tetris: The Grand Master 2 - The Absolute Plus (TAP)
//! Death-mode gameplay simulator and pathfinder.
//!
//! Phase 1 scaffolding. Board + piece + 20G drop are functional (1.1); the
//! rotation system (1.2) and BFS pathfinder (1.3) are stubbed.

pub mod board;
pub mod piece;
pub mod rotation;
pub mod action;
pub mod pathfind;
pub mod bulk;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Python-facing entry point. Exposes Board, Piece, PieceType, Input, and
/// LockCandidate so the visualization harness and pytest suite can drive the
/// simulator and pathfinder directly.
#[pymodule]
fn deathbot_sim(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyBoard>()?;
    m.add_class::<PyPiece>()?;
    m.add_class::<PyLockCandidate>()?;
    m.add("PieceType", _py.get_type_bound::<PyPieceType>())?;
    m.add("Input", _py.get_type_bound::<PyInput>())?;
    Ok(())
}

// ---- Input enum (Python-facing) -----------------------------------------

/// A single discrete input, matching the BFS edge labels. Use these to
/// interpret a `LockCandidate.inputs` path.
#[pyclass(name = "Input", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PyInput {
    RotateCw = 0,
    RotateCcw = 1,
    Left = 2,
    Right = 3,
    Neutral = 4,
}

impl PyInput {
    fn from_inner(i: action::Input) -> Self {
        match i {
            action::Input::RotateCw => PyInput::RotateCw,
            action::Input::RotateCcw => PyInput::RotateCcw,
            action::Input::Left => PyInput::Left,
            action::Input::Right => PyInput::Right,
            action::Input::Neutral => PyInput::Neutral,
        }
    }
}

#[pymethods]
impl PyInput {
    fn __repr__(&self) -> String {
        format!("Input::{:?}", self)
    }
}

// ---- LockCandidate (Python-facing) --------------------------------------

/// One reachable locking position + the input sequence that reaches it from
/// the IRS root. The input sequence does NOT include a trailing Neutral —
/// emit one to actually lock the piece in the real game.
/// `irs_root` is the rotation state of the IRS root the path starts from
/// (0 = bare spawn). Live playback sets it up with IRS taps during ARE.
#[pyclass(name = "LockCandidate")]
#[derive(Debug, Clone)]
pub struct PyLockCandidate {
    pub piece: PyPiece,
    pub inputs: Vec<PyInput>,
    pub irs_root: u8,
}

#[pymethods]
impl PyLockCandidate {
    #[getter]
    fn piece(&self) -> PyPiece {
        PyPiece { inner: self.piece.inner }
    }

    /// The input sequence (list of Input enum values) from the IRS root.
    #[getter]
    fn inputs(&self) -> Vec<PyInput> {
        self.inputs.clone()
    }

    /// The input sequence as a list of human-readable strings, e.g.
    /// ["Left", "RotateCw", "Neutral"]. Convenience for debugging.
    fn inputs_str(&self) -> Vec<String> {
        self.inputs
            .iter()
            .map(|i| format!("{:?}", i))
            .collect()
    }

    /// Rotation state of the IRS root this path starts from (0 = bare
    /// spawn, no pre-spawn input needed).
    #[getter]
    fn irs_root(&self) -> u8 {
        self.irs_root
    }

    fn __repr__(&self) -> String {
        format!("LockCandidate({:?}, {:?})", self.piece.inner, self.inputs)
    }
}

// ---- PieceType enum (Python-facing) ---------------------------------------

#[pyclass(name = "PieceType", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PyPieceType {
    I = 0,
    O = 1,
    T = 2,
    S = 3,
    Z = 4,
    J = 5,
    L = 6,
}

impl PyPieceType {
    fn to_inner(self) -> piece::PieceType {
        match self {
            PyPieceType::I => piece::PieceType::I,
            PyPieceType::O => piece::PieceType::O,
            PyPieceType::T => piece::PieceType::T,
            PyPieceType::S => piece::PieceType::S,
            PyPieceType::Z => piece::PieceType::Z,
            PyPieceType::J => piece::PieceType::J,
            PyPieceType::L => piece::PieceType::L,
        }
    }
}

#[pymethods]
impl PyPieceType {
    /// Return the list of [dx, dy] cell offsets for state 0.
    fn spawn_cells(&self, py: Python<'_>) -> PyObject {
        let cells = self.to_inner().cells(0);
        cells
            .iter()
            .map(|&(dx, dy)| (dx as i32, dy as i32))
            .collect::<Vec<_>>()
            .to_object(py)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

// ---- Piece (Python-facing) ------------------------------------------------

#[pyclass(name = "Piece", eq)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyPiece {
    pub inner: piece::Piece,
}

#[pymethods]
impl PyPiece {
    #[new]
    fn new(kind: PyPieceType, state: u8, x: i8, y: i8) -> Self {
        Self {
            inner: piece::Piece::new(kind.to_inner(), state, x, y),
        }
    }

    #[staticmethod]
    fn spawn(kind: PyPieceType) -> Self {
        Self {
            inner: piece::Piece::spawn(kind.to_inner()),
        }
    }

    #[getter]
    fn kind(&self) -> PyPieceType {
        match self.inner.kind {
            piece::PieceType::I => PyPieceType::I,
            piece::PieceType::O => PyPieceType::O,
            piece::PieceType::T => PyPieceType::T,
            piece::PieceType::S => PyPieceType::S,
            piece::PieceType::Z => PyPieceType::Z,
            piece::PieceType::J => PyPieceType::J,
            piece::PieceType::L => PyPieceType::L,
        }
    }

    #[getter]
    fn state(&self) -> u8 {
        self.inner.state
    }

    #[getter]
    fn x(&self) -> i8 {
        self.inner.x
    }

    #[getter]
    fn y(&self) -> i8 {
        self.inner.y
    }

    /// Absolute (col, row) cell positions as a list of [c, r] pairs.
    fn cells(&self, py: Python<'_>) -> PyObject {
        let v: Vec<[i32; 2]> = self
            .inner
            .cells_iter()
            .map(|(c, r)| [c as i32, r as i32])
            .collect();
        v.to_object(py)
    }

    fn rotate_cw(&self) -> Self {
        Self { inner: self.inner.rotate_cw() }
    }

    fn rotate_ccw(&self) -> Self {
        Self { inner: self.inner.rotate_ccw() }
    }

    fn translate(&self, dx: i8, dy: i8) -> Self {
        Self { inner: self.inner.translate(dx, dy) }
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    fn __repr__(&self) -> String {
        format!(
            "Piece({:?}, state={}, x={}, y={})",
            self.inner.kind, self.inner.state, self.inner.x, self.inner.y
        )
    }
}

// ---- Board (Python-facing) ------------------------------------------------

#[pyclass(name = "Board")]
#[derive(Debug, Clone)]
pub struct PyBoard {
    pub inner: board::Board,
}

#[pymethods]
impl PyBoard {
    #[new]
    fn new() -> Self {
        Self { inner: board::Board::new() }
    }

    /// Number of visible columns (always 10).
    #[getter]
    fn width(&self) -> usize {
        board::WIDTH
    }

    /// Number of rows including the single invisible buffer row (always 21).
    #[getter]
    fn height(&self) -> usize {
        board::HEIGHT
    }

    /// Number of visible rows (always 20).
    #[getter]
    fn visible_rows(&self) -> usize {
        board::VISIBLE_ROWS
    }

    /// True if the piece position is valid (in-bounds horizontally, no
    /// overlap with existing blocks). Cells above the field are allowed by
    /// is_valid but flagged separately by locks_out.
    fn is_valid(&self, p: &PyPiece) -> bool {
        self.inner.is_valid(&p.inner)
    }

    /// True if locking the piece would put any cell strictly above the
    /// invisible row (row >= 21). That's the TAP "block out" death condition.
    fn locks_out(&self, p: &PyPiece) -> bool {
        self.inner.locks_out(&p.inner)
    }

    /// Drop the piece under 20G gravity; returns the grounded piece.
    fn drop_20g(&self, p: &PyPiece) -> PyPiece {
        PyPiece { inner: self.inner.drop_20g(&p.inner) }
    }

    /// Merge a (grounded) piece into the board. Does not clear lines.
    fn lock(&mut self, p: &PyPiece) {
        self.inner.lock(&p.inner)
    }

    /// Clear all full rows; returns the count cleared.
    fn clear_lines(&mut self) -> usize {
        self.inner.clear_lines()
    }

    /// Set a single cell (col, row). Raises if out of range. Useful for
    /// building test board setups from Python.
    fn set_cell(&mut self, col: usize, row: usize) -> PyResult<()> {
        if col >= board::WIDTH || row >= board::HEIGHT {
            return Err(PyValueError::new_err(format!(
                "cell ({}, {}) out of range ({}x{})",
                col, row, board::WIDTH, board::HEIGHT
            )));
        }
        self.inner.rows[row] |= 1 << col;
        Ok(())
    }

    /// True if cell (col, row) is occupied. False for out-of-range.
    fn cell(&self, col: i32, row: i32) -> bool {
        self.inner.cell(col as i8, row as i8)
    }

    /// Stack height in a column (0..10), counting from the bottom.
    fn column_height(&self, col: usize) -> usize {
        self.inner.column_height(col)
    }

    /// Render the board as a list of 21 strings, top row first, each 10 chars
    /// wide. '#' = filled, '.' = empty. The invisible row (row 20) is the
    /// last element. Useful for text-based debugging and golden-image tests.
    fn to_text(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(board::HEIGHT);
        for r in (0..board::HEIGHT).rev() {
            let mut s = String::with_capacity(board::WIDTH);
            for c in 0..board::WIDTH {
                s.push(if self.inner.rows[r] & (1 << c) != 0 { '#' } else { '.' });
            }
            out.push(s);
        }
        out
    }

    // ---- Fast board access (training speed optimizations) -----------------
    //
    // These exist to eliminate the O(210) per-cell Python→Rust FFI calls
    // that dominated self-play training time. Each is a single FFI call.

    /// Deep-copy the board. One FFI call instead of 210 set_cell calls.
    /// Equivalent to the old Python loop:
    ///   sim = Board(); for r in range(h): for c in range(w): if b.cell(c,r): sim.set_cell(c,r)
    fn clone_board(&self) -> Self {
        Self { inner: self.inner.clone() }
    }

    /// Return the raw row bitmasks as a list of 21 ints. Bit `c` of row `r`
    /// is set iff cell (c, r) is filled. This lets Python build a tensor in
    /// one vectorized pass instead of 200 individual cell() calls.
    ///
    /// Row 0 (bottom) is element 0; row 20 (invisible) is element 20.
    fn row_bits(&self) -> Vec<u32> {
        self.inner.rows.to_vec()
    }

    // ---- Rotation / action API (phase 1.2) --------------------------------

    /// Attempt to rotate `p` (CW if `cw` else CCW) applying the ARS wall-kick
    /// table. Returns the rotated+kicked piece, or None if every kick fails.
    fn try_rotate(&self, p: &PyPiece, cw: bool) -> Option<PyPiece> {
        rotation::try_rotate(&self.inner, &p.inner, cw).map(|pp| PyPiece { inner: pp })
    }

    /// Which (dx, dy) kick offset would be used to rotate `p`? Returns None
    /// if rotation would fail entirely. Useful for the visualizer/debugging.
    fn kick_used(&self, p: &PyPiece, cw: bool) -> Option<(i8, i8)> {
        rotation::kick_used(&self.inner, &p.inner, cw)
    }

    /// Apply one tick's action to `p` in TAP order: rotate -> move -> 20G.
    /// `rotate` is None | "cw" | "ccw"; `move_dir` is None | "left" | "right".
    /// Returns (piece, grounded: bool).
    fn apply_tick(
        &self,
        p: &PyPiece,
        rotate: Option<&str>,
        move_dir: Option<&str>,
    ) -> (PyPiece, bool) {
        let act = action::TickAction {
            rotate: match rotate {
                Some("cw") => Some(action::RotationInput::Cw),
                Some("ccw") => Some(action::RotationInput::Ccw),
                _ => None,
            },
            move_dir: match move_dir {
                Some("left") => Some(action::MoveInput::Left),
                Some("right") => Some(action::MoveInput::Right),
                _ => None,
            },
        };
        let out = action::apply_tick(&self.inner, &p.inner, act);
        (PyPiece { inner: out.piece }, out.grounded)
    }

    /// The IRS root states for a freshly spawned piece: spawn state plus all
    /// rotation states reachable by free pre-tick rotations (IRS). Returns a
    /// list of pieces, one per distinct reachable state.
    fn irs_root_states(&self, kind: PyPieceType) -> Vec<PyPiece> {
        action::irs_root_states(&self.inner, kind.to_inner())
            .into_iter()
            .map(|p| PyPiece { inner: p })
            .collect()
    }

    // ---- Pathfinder (phase 1.3) -------------------------------------------

    /// BFS over all reachable lock positions for `kind` on this board.
    /// Returns a list of `LockCandidate`s, each with the grounded piece and
    /// the input sequence (from the IRS root) that reaches it. Shortest paths
    /// are found first (BFS order). Dedup by (state, x).
    fn search_locks(&self, kind: PyPieceType) -> Vec<PyLockCandidate> {
        pathfind::search_locks(&self.inner, kind.to_inner())
            .into_iter()
            .map(|c| PyLockCandidate {
                piece: PyPiece { inner: c.piece },
                inputs: c.inputs.iter().map(|&i| PyInput::from_inner(i)).collect(),
                irs_root: c.irs_root,
            })
            .collect()
    }

    /// Convenience: count of distinct reachable lock positions.
    fn count_locks(&self, kind: PyPieceType) -> usize {
        pathfind::count_locks(&self.inner, kind.to_inner())
    }

    /// Bare-spawn BFS for live playback: same search seeded only from the
    /// unrotated spawn, so every path replays with no pre-spawn input.
    /// Same element type as `search_locks` (all report `irs_root: 0`).
    fn search_locks_bare(&self, kind: PyPieceType) -> Vec<PyLockCandidate> {
        pathfind::search_locks_bare(&self.inner, kind.to_inner())
            .into_iter()
            .map(|c| PyLockCandidate {
                piece: PyPiece { inner: c.piece },
                inputs: c.inputs.iter().map(|&i| PyInput::from_inner(i)).collect(),
                irs_root: c.irs_root,
            })
            .collect()
    }

    /// Root-constrained BFS for live playback with true IRS: same search
    /// seeded only from IRS roots with the given rotation state (the
    /// orientation an ARE hold produced). Empty when that root is blocked.
    fn search_locks_from(
        &self,
        kind: PyPieceType,
        root_state: u8,
    ) -> Vec<PyLockCandidate> {
        pathfind::search_locks_from(&self.inner, kind.to_inner(), root_state)
            .into_iter()
            .map(|c| PyLockCandidate {
                piece: PyPiece { inner: c.piece },
                inputs: c.inputs.iter().map(|&i| PyInput::from_inner(i)).collect(),
                irs_root: c.irs_root,
            })
            .collect()
    }

    /// Bulk lock+clear every reachable candidate for `kind` in ONE FFI call.
    ///
    /// Returns `(rows, lines)` where `rows` is a flat list of
    /// `len(candidates) * 21` u32 bitmasks (candidate-major order, matching
    /// the order returned by `search_locks`) and `lines` is one line-count
    /// per candidate. This replaces the N×(clone_board + lock + clear_lines
    /// + row_bits) FFI round-trips in the Python trainer with a single call.
    ///
    /// NOTE: lockout (cells above the field) is NOT checked here — mirrors
    /// the historical semantics where the caller must filter. Python-side
    /// rollout code checks `locks_out` before committing.
    fn lock_candidates_rows(&self, kind: PyPieceType) -> (Vec<u32>, Vec<usize>) {
        let candidates = pathfind::search_locks(&self.inner, kind.to_inner());
        let results = bulk::lock_candidates(&self.inner, &candidates);
        let mut rows = Vec::with_capacity(candidates.len() * board::HEIGHT);
        let mut lines = Vec::with_capacity(candidates.len());
        for res in &results {
            rows.extend_from_slice(&res.rows);
            lines.push(res.lines_cleared);
        }
        (rows, lines)
    }

    /// Bulk search + lock + clear with FULL per-candidate metadata, all in
    /// one FFI crossing:
    ///   (state, x, y, locks_out, rows_flat, lines)
    ///   - state/x/y per candidate (the grounded piece)
    ///   - locks_out: whether locking this candidate would end the game
    ///   - rows_flat: len(candidates) * 21 u32 bitmasks (post-lock, post-clear)
    ///   - lines: one line count per candidate
    #[pyo3(signature = (kind))]
    fn lock_candidates_full(
        &self,
        kind: PyPieceType,
    ) -> (Vec<i8>, Vec<i8>, Vec<i8>, Vec<bool>, Vec<u32>, Vec<usize>) {
        let candidates = pathfind::search_locks(&self.inner, kind.to_inner());
        let results = bulk::lock_candidates(&self.inner, &candidates);
        let n = candidates.len();
        let mut states = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut lockouts = Vec::with_capacity(n);
        let mut rows = Vec::with_capacity(n * board::HEIGHT);
        let mut lines = Vec::with_capacity(n);
        for (cand, res) in candidates.iter().zip(results.iter()) {
            states.push(cand.piece.state as i8);
            xs.push(cand.piece.x);
            ys.push(cand.piece.y);
            lockouts.push(self.inner.locks_out(&cand.piece));
            rows.extend_from_slice(&res.rows);
            lines.push(res.lines_cleared);
        }
        (states, xs, ys, lockouts, rows, lines)
    }

    /// Reconstruct a board from 21 row bitmasks (bottom-first), e.g. the
    /// packed rows returned by `lock_candidates_rows`.
    #[staticmethod]
    fn from_row_bits(rows: Vec<u32>) -> PyResult<PyBoard> {
        if rows.len() != board::HEIGHT {
            return Err(PyValueError::new_err(format!(
                "expected {} row bitmasks, got {}",
                board::HEIGHT,
                rows.len()
            )));
        }
        Ok(PyBoard { inner: bulk::board_from_rows(&rows) })
    }

    /// Reconstruct MANY boards in one FFI call (item 4): `flat` is
    /// N*21 u32 bitmasks (candidate-major). Returns N Board objects.
    #[staticmethod]
    fn from_row_bits_bulk(flat: Vec<u32>) -> PyResult<Vec<PyBoard>> {
        if flat.len() % board::HEIGHT != 0 {
            return Err(PyValueError::new_err(format!(
                "expected a multiple of {} row bitmasks, got {}",
                board::HEIGHT,
                flat.len()
            )));
        }
        Ok(bulk::boards_from_rows_flat(&flat)
            .into_iter()
            .map(|inner| PyBoard { inner })
            .collect())
    }

    /// Scoring-only bulk path (item 5): search + lock + clear with NO
    /// input-path allocation. Same BFS order as `search_locks`.
    /// Returns (state, x, y, locks_out, rows_flat, lines).
    fn lock_candidates_fast(
        &self,
        kind: PyPieceType,
    ) -> (Vec<i8>, Vec<i8>, Vec<i8>, Vec<bool>, Vec<u32>, Vec<usize>) {
        let (pieces, results, lockouts) =
            bulk::search_and_lock(&self.inner, kind.to_inner());
        let n = pieces.len();
        let mut states = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut rows = Vec::with_capacity(n * board::HEIGHT);
        let mut lines = Vec::with_capacity(n);
        for ((p, res), _lo) in pieces.iter().zip(results.iter()).zip(lockouts.iter()) {
            states.push(p.state as i8);
            xs.push(p.x);
            ys.push(p.y);
            rows.extend_from_slice(&res.rows);
            lines.push(res.lines_cleared);
        }
        (states, xs, ys, lockouts, rows, lines)
    }

    fn __repr__(&self) -> String {
        format!("Board({}x{})", board::WIDTH, board::HEIGHT)
    }
}
