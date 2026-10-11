"""1-ply greedy suggestion for the TGM2 mirror (snapshot12, plain greedy).

Board + piece IDs come from the BizHawk state snapshots (TGM IDs:
I=0 Z=1 S=2 J=3 L=4 O=5 T=6). This module maps them onto deathbot_sim
types, enumerates reachable locks, scores afterstates with snapshot12,
and returns the argmax placement. No style bonuses (burn/line-bonus are a
TwoPlyScorer playing-time concern, not part of the weights).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Perf: pin torch to a single thread BEFORE any inference. The default
# (4 threads here) makes each 1-ply batch burst across all cores and
# starves the MAME thread, which shows up as slower-than-normal play +
# audio cutouts. Single-thread uses ~1/4 of the CPU per inference
# (isolated batches are a few ms slower, but the emulator stays smooth).
# set_num_interop_threads() may only be called once before parallel work
# starts, so do it at import (guarded); set_num_threads() is re-applied
# in ensure_net() since it can be set at any time.
try:
    import torch as _torch

    try:
        _torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        _torch.set_num_threads(1)
    except RuntimeError:
        pass
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "deathbot_sim" / "python"))

import deathbot_sim as ds  # noqa: E402
from model import TAPNet, piece_to_index  # noqa: E402
from neural_scorer import NeuralScorer  # noqa: E402

WEIGHTS = ROOT / "model_weights_snapshot12.pt"

# TGM wire ID -> sim piece type.
#
# VERIFIED from ARS diagram pixel reads + sprite captures + palette hues:
# the sim tables use SRS naming, but TGM/ARS mirrors both chiral pairs.
# ARS-L-spawn ###/#.. == sim-J_0, ARS-J-spawn ###/..# == sim-L_0,
# ARS-S-spawn .##/##. == sim-Z_0, ARS-Z-spawn ##./.## == sim-S_0 (all in
# screen frame; I/O/T are mirror-symmetric so they are unaffected). The
# bridge therefore crosses BOTH pairs: wire-J uses sim-L geometry+index,
# wire-L uses sim-J, wire-S uses sim-Z, wire-Z uses sim-S. Each net one-hot
# index stays paired with the geometry it learned in-sim, so no retraining
# is needed.
TGM2SIM = [ds.PieceType.I, ds.PieceType.S, ds.PieceType.Z, ds.PieceType.L,
           ds.PieceType.J, ds.PieceType.O, ds.PieceType.T]

_net = None
_scorer = None
_load_error = None


def ensure_net():
    """Load snapshot12 once. Returns (scorer_or_None, error_or_None)."""
    global _net, _scorer, _load_error
    if _scorer is not None or _load_error is not None:
        return _scorer, _load_error
    try:
        import torch
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
        _net = TAPNet()
        _net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        _net.eval()
        _scorer = NeuralScorer(net=_net)
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
    return _scorer, _load_error


def board_from_rows(rows) -> "ds.Board":
    """rows: 20 strings top-first of IZSJLOT/. -> sim Board (row 0 = bottom)."""
    b = ds.Board()
    for r_top in range(20):
        row = rows[r_top]
        for c in range(10):
            if row[c] != ".":
                b.set_cell(c, 19 - r_top)
    return b


# Rotation-state counts per sim piece type (kept for reference; v1 uses
# bare-spawn roots only, see below).
NUMSTATES = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}

# Rotation-direction truth (verified cell-by-cell against the ARS diagram
# and the button probe): sim state+1 cycles VISUALLY COUNTERCLOCKWISE for
# the 4-state pieces T/J/L, i.e. sim "RotateCw" behaves like the TGM CCW
# button and vice versa. (S/Z/I have 2 states / O has 1, so direction is
# outcome-free there.) The sim is internally self-consistent — search,
# scoring, and paths agree with each other — so the fix is purely the
# bridge translation below: sim steps/IRS roots are emitted as TGM-side
# directions ("CW"/"CCW" = what the buttons do), keyed per sim type.
INVERTED = {"T", "J", "L"}


def _irs_for(root: int, n: int, inverted: bool) -> str | None:
    """Kept for a future true-IRS implementation (pre-spawn hold during
    ARE). v1 does not use IRS at all — see suggest()."""
    if root == 0:
        return "-"
    if root == 1:
        return "CCW" if inverted else "CW"
    if n == 4 and root == 3:
        return "CW" if inverted else "CCW"
    return None


def _tgm_step(sim_step: str, inverted: bool) -> str:
    """Translate a sim input name to TGM-side direction names."""
    if sim_step == "RotateCw":
        return "CCW" if inverted else "CW"
    if sim_step == "RotateCcw":
        return "CW" if inverted else "CCW"
    return sim_step  # Left/Right unchanged


def _sim_name(t) -> str:
    return str(t).split(".")[-1]


def sim_info(tgm_id: int) -> tuple:
    """(sim type name, num_states) for a TGM wire ID."""
    if not isinstance(tgm_id, int) or not 0 <= tgm_id < 7:
        raise ValueError(f"bad TGM piece id: {tgm_id!r}")
    t = TGM2SIM[tgm_id]
    name = _sim_name(t)
    return name, NUMSTATES[name]


def _allowed_roots(n: int) -> set:
    """IRS roots reachable with at most one pre-spawn rotation."""
    if n <= 1:
        return {0}
    if n == 2:
        return {0, 1}
    return {0, 1, 3}


def _argmax(board, cands, ci: int, ni: int):
    """(best_index, scores) with TwoPlyScorer-compatible tie-breaking."""
    scorer, err = ensure_net()
    if scorer is None:
        return None, None
    import torch
    scores = scorer.score_batch(board, cands, ci, ni)
    best = max(range(len(scores)), key=lambda j: (scores[j], -j))
    return best, scores


def choose_root(rows, cur_tgm: int, next_approx_tgm: int) -> dict:
    """Lock-time ROOT choice for true IRS (ARE hold).

    Full search, argmax with an approximated next piece, restricted to
    single-tap roots. Returns {"ok", "root" (sim state), "hold" (TGM-side
    "CW"/"CCW"/"-"), "value", "n"}.
    """
    if not (isinstance(cur_tgm, int) and isinstance(next_approx_tgm, int)
            and 0 <= cur_tgm < 7 and 0 <= next_approx_tgm < 7):
        return {"ok": False,
                "reason": f"bad ids cur={cur_tgm} next={next_approx_tgm}"}
    cur_t = TGM2SIM[cur_tgm]
    nxt_t = TGM2SIM[next_approx_tgm]
    try:
        board = board_from_rows(rows)
        n = NUMSTATES[_sim_name(cur_t)]
        allowed = _allowed_roots(n)
        cands = [c for c in board.search_locks(cur_t)
                 if not board.locks_out(c.piece) and c.irs_root in allowed]
        if not cands:
            return {"ok": False, "reason": "no allowed-root locks"}
        from model import piece_to_index
        best, scores = _argmax(board, cands, piece_to_index(cur_t),
                               piece_to_index(nxt_t))
        if best is None:
            return {"ok": False, "reason": "net unavailable"}
        root = cands[best].irs_root
        inv = _sim_name(cur_t) in INVERTED
        # root==1: CW normally, CCW when inverted; root==3 (n==4):
        # CCW normally, CW when inverted.
        if root == 0:
            hold = "-"
        elif root == 1:
            hold = "CCW" if inv else "CW"
        else:  # root == 3, n == 4
            hold = "CW" if inv else "CCW"
        return {"ok": True, "root": root, "hold": hold,
                "value": scores[best], "n": len(cands)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"root choice failed: {exc}"}


def root_for_hold(hold: str, sim_name: str, n: int) -> int:
    """Sim root state a TGM-side ARE hold produces (probe: hold rotates
    exactly once). "-" (or O piece) means bare spawn 0."""
    if hold == "-" or n <= 1:
        return 0
    inv = sim_name in INVERTED
    if hold == "CW":
        return n - 1 if inv else 1
    return 1 if inv else n - 1  # hold == "CCW"


def suggest(rows, cur_tgm: int, next_tgm: int, root: int = 0) -> dict:
    """Greedy 1-ply argmax from a fixed IRS root (default bare spawn 0).

    Uses root-constrained BFS, so every path starts exactly where the real
    piece is — the bare spawn, or the orientation an ARE hold produced
    (adjudicated by Lua via the state's held field, never assumed).
    Returns a dict (ok=False with reason on failure). On success includes
    irs="-", steps (TGM-side names), root, ghost, top3, n, eval_ms.
    """
    t0 = time.perf_counter()
    scorer, err = ensure_net()
    if scorer is None:
        return {"ok": False, "reason": f"net load failed: {err}"}
    if not (isinstance(cur_tgm, int) and isinstance(next_tgm, int)
            and 0 <= cur_tgm < 7 and 0 <= next_tgm < 7):
        return {"ok": False, "reason": f"bad ids cur={cur_tgm} next={next_tgm}"}
    cur_t = TGM2SIM[cur_tgm]
    nxt_t = TGM2SIM[next_tgm]
    inverted = _sim_name(cur_t) in INVERTED
    try:
        board = board_from_rows(rows)
        cands = board.search_locks_from(cur_t, root)
        n_total = len(cands)
        playable = []
        for c in cands:
            if board.locks_out(c.piece):
                continue
            playable.append((c, "-"))
        if not playable:
            return {"ok": False, "reason": "no v1-playable locks",
                    "eval_ms": (time.perf_counter() - t0) * 1000.0,
                    "n": n_total}
        scores = scorer.score_batch(
            board, [c for c, _ in playable],
            piece_to_index(cur_t), piece_to_index(nxt_t))
        order = sorted(range(len(scores)),
                       key=lambda j: (-scores[j], j))
        best = order[0]
        best_c, best_irs = playable[best]
        ghost = [(c, 19 - r) for c, r in best_c.piece.cells()]
        steps_tgm = [_tgm_step(s, inverted) for s in best_c.inputs_str()]
        top3 = [(scores[j], playable[j][0].piece.state,
                 playable[j][0].piece.x, playable[j][0].piece.y)
                for j in order[:3]]
        return {
            "ok": True,
            "best_score": scores[best],
            "best_state": best_c.piece.state,
            "best_x": best_c.piece.x,
            "best_y": best_c.piece.y,
            "ghost": ghost,
            "top3": top3,
            "n": len(playable),
            "n_total": n_total,
            "irs": best_irs,
            "steps": steps_tgm,
            "root": best_c.irs_root,
            "eval_ms": (time.perf_counter() - t0) * 1000.0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"eval failed: {exc}"}
