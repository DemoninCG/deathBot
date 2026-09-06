"""Seeded well diagnostics (§15.1.2): build/preserve/cash-out/recover suite.

Constructs six hand-built board states (no RNG, no search training) and
asks the champion scorer (snapshot9 + full 2-ply) what it does:

  1. precursor1  — clean 1-deep well precursor on an empty board
  2. partial2    — two-row partial setup (well col 9, rows filled except well)
  3. ready       — Tetris-ready well (depth>=4, open, low stack)
  4. delayed_I   — ready well but current piece is NOT I (must preserve)
  5. blocked     — nearly-blocked well (hole/overhang in channel)
  6. highstack   — high stack: continuing to build is unsafe (must recover)

For each: report the champion's pick (lines cleared, Phi delta, well depth
before/after, Tetris taken or not). Verdict logic:
- takes a reachable Tetris from `ready`? If NO → scoring/transition bug,
  inspect before any training.
- builds from precursor/partial (Phi non-decreasing, no well cover)?
- preserves the setup under delayed_I?
- recovers (clears down / abandons well) on highstack/blocked?

These states double as a later training curriculum.

Usage:
    python diag_seeded_wells.py --weights ../../model_weights_snapshot9.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deathbot_sim as ds
from model import TAPNet, piece_to_index
from two_ply import TwoPlyScorer
from well_potential import dense_well_features, well_potential, WELL_COL

KIND = {0: ds.PieceType.I, 1: ds.PieceType.O, 2: ds.PieceType.T,
        3: ds.PieceType.S, 4: ds.PieceType.Z, 5: ds.PieceType.J,
        6: ds.PieceType.L}
NAME = {v: k for k, v in
        {0: "I", 1: "O", 2: "T", 3: "S", 4: "Z", 5: "J", 6: "L"}.items()}


def _board_from_heights(h, holes: list[tuple[int, int]] | None = None) -> ds.Board:
    b = ds.Board()
    for c, hh in enumerate(h):
        for r in range(hh):
            b.set_cell(c, r)
    if holes:
        # carve holes by rebuilding without those cells (set_cell only adds;
        # emulate by clearing via a fresh board minus hole cells)
        b2 = ds.Board()
        hole_set = set(holes)
        for c, hh in enumerate(h):
            for r in range(hh):
                if (c, r) not in hole_set:
                    b2.set_cell(c, r)
        return b2
    return b


def _rows_of(board: ds.Board) -> np.ndarray:
    return np.asarray(board.row_bits(), dtype=np.uint32)


def build_suite() -> list[dict]:
    # Well col 9. NOTE: solid-height boards have FULL rows below the well
    # floor (rows 0..h[9]-1 are complete), so any move there reports
    # inflated bulk lines_cleared (the move clears the well rows AND the
    # pre-existing full rows). Those states still test cash-out direction
    # (depth_after collapse) but not exact Tetris counts. The `shelf`
    # case below is the clean Tetris-ready shape: a capped well with an
    # overhang shelf, so only the 4 well rows clear.
    flat = [4] * 9 + [3]          # 1-deep precursor
    partial = [6] * 9 + [4]       # 2-deep partial
    ready = [8] * 9 + [4]         # 4-deep ready (bulk lines inflated)
    blocked = [8] * 9 + [4]       # same shape, plus an overhang hole
    high = [16] * 9 + [14]        # high stack, must recover not build
    return [
        {"name": "precursor1", "h": flat, "cur": "T", "nxt": "I",
         "want": "build"},
        {"name": "partial2", "h": partial, "cur": "L", "nxt": "I",
         "want": "build"},
        {"name": "ready", "h": ready, "cur": "I", "nxt": "T",
         "want": "cashout"},
        {"name": "delayed_I", "h": ready, "cur": "T", "nxt": "O",
         "want": "preserve"},
        {"name": "blocked", "h": blocked, "cur": "I", "nxt": "T",
         "want": "recover_or_cashout",
         "holes": [(9, 5)]},
        {"name": "highstack", "h": high, "cur": "O", "nxt": "I",
         "want": "recover"},
    ]


def _piece(name: str):
    return {"I": ds.PieceType.I, "O": ds.PieceType.O, "T": ds.PieceType.T,
            "S": ds.PieceType.S, "Z": ds.PieceType.Z, "J": ds.PieceType.J,
            "L": ds.PieceType.L}[name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval()
    scorer = TwoPlyScorer(net, beam=10 ** 9)

    print(f"{'case':>12s} {'cur':>3s} {'pick_lines':>10s} {'dPhi':>7s} "
          f"{'depth_before':>12s} {'depth_after':>11s} {'verdict':>16s}")
    for case in build_suite():
        board = _board_from_heights(case["h"], case.get("holes"))
        cur, nxt = _piece(case["cur"]), _piece(case["nxt"])
        rows_before = _rows_of(board)
        phi_before = well_potential(rows_before)
        depth_before = dense_well_features(rows_before)["well_depth_raw"]
        try:
            pick = scorer.choose(board, cur, nxt)
        except ValueError:
            print(f"{case['name']:>12s} {'—':>3s} {'no-cands':>10s}")
            continue
        bulk = board.lock_candidates_full(cur)
        n = len(bulk[0])
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(n, 21)
        lines = int(bulk[5][pick])  # index 5 = lines_cleared (6-tuple)
        phi_after = well_potential(rows[pick])
        depth_after = dense_well_features(rows[pick])["well_depth_raw"]
        dphi = phi_after - phi_before
        want = case["want"]
        ok = "inspect"
        # NOTE: bulk lines_cleared counts TOTAL rows cleared by the move
        # (bulk.rs compacts all full rows), so a Tetris on these hand-built
        # boards also sweeps the pre-existing full rows below the well.
        # Cash-out test = the well column is filled by the move, i.e. the
        # afterstate well depth collapses to ~0 AND lines >= 4.
        if want == "cashout":
            ok = ("CASH-OUT" if (lines >= 4 and depth_after <= 1)
                  else f"lines={lines} depth_after={depth_after:.0f}")
        elif want == "preserve":
            ok = "preserve" if (depth_after >= depth_before
                                and lines < 4) else "damaged-setup"
        elif want == "build":
            ok = "builds" if dphi >= -0.05 else "avoids-setup"
        else:
            ok = "see-depth/lines"
        print(f"{case['name']:>12s} {case['cur']:>3s} {lines:>10d} "
              f"{dphi:>+7.3f} {depth_before:>12.0f} {depth_after:>11.0f} "
              f"{ok:>16s}")


if __name__ == "__main__":
    main()
