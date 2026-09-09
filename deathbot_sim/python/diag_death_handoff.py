"""Death-handoff autopsy: 20851 + 20857 controlled continuations (§40.1).

User design (2026-09-09): the two commit_loose cap-1500 deaths do NOT by
themselves implicate rescue — elapsed time is not causal; recoverability
at the intervention point is. For each death tape, save states around:

  - first substantial accumulation,
  - RECOVER entry / RECOVER exit,
  - final re-entries before death,

with the EXACT recorded queue continuation (positional PieceQueue replay,
E4-validated). From IDENTICAL states, compare:

  A. original commitment controller — must REPRODUCE the failure
     (determinism sanity check).
  B. permanent survival fallback (RECOVER policy, re-entry DISABLED) —
     is recovery itself sufficient, and is re-entry the problem?
  C. canonical survival champion (bonus-free full 2-ply) — is this
     position still recoverable at all?
     (B and C differ by construction: B carries the completion bonus,
     C does not. Keep both to isolate bonus-vs-reentry effects.)

Interpretation matrix (pre-registered):
  B caps + C caps, A dies  → re-entry is the defect; recovery works.
  B dies + C dies           → position unrecoverable at that handoff;
                             trigger fired too late.
  B caps + C dies           → completion bonus harmful in recovery;
                             use bonus-free fallback.
  A caps from early handoff, dies from late → trigger threshold late,
    not the recovery policy.

NO panels (death-tape replays only), NO training. Selection of handoff
positions comes from an instrumented replay (per-piece mode/occ/top +
transition log) — event detection is mechanical, not eyeballed.

Usage:
    python diag_death_handoff.py --weights ../../model_weights_snapshot9.pt \\
        --out ../_diag_out/death_handoff.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
DEATHS = (20851, 20857)
RESCUE_CAP = 1500
ACCUM_WINDOW = 100  # occ gain over trailing 100 pcs marking "accumulation"
ACCUM_THRESH = 25.0


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"not found: {p}")


def _occ_top(board):
    rows = np.asarray(board.row_bits(), dtype=np.uint32)
    occ = sum(bin(int(v)).count("1") for v in rows)
    top = 0
    for r in range(20, -1, -1):
        if int(rows[r]) & 0b1111111111:
            top = r + 1
            break
    return occ, top


def instrumented_replay(net, seed, *, abort_top=16, abort_occ=110,
                        reenter_occ=60):
    """Replay the commit_loose death tape with a per-piece event log.

    Returns (trace, death_pos): trace rows {pos, occ, top, committed,
    transition} where transition marks RECOVER entry/exit edges.
    """
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer
    from piece_history import PieceQueue
    import deathbot_sim as ds
    base = TwoPlyScorer(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS))
    ctrl = CommitScorer(base, abort_top=abort_top, abort_occ=abort_occ,
                        reenter_occ=reenter_occ)
    pq = PieceQueue(random.Random(seed))
    board = ds.Board()
    trace: list[dict] = []
    prev_committed = True
    pos = 0
    while True:
        bulk = board.lock_candidates_full(pq.current)
        nc = len(bulk[0])
        if nc == 0 or all(bulk[3]):
            return trace, pos
        occ, top = _occ_top(board)
        # peek transition: choose() mutates committed; record edge after.
        pick = ctrl.choose(board, pq.current, pq.next_piece)
        edge = None
        if prev_committed and not ctrl.committed:
            edge = "RECOVER-entry"
        elif not prev_committed and ctrl.committed:
            edge = "RECOVER-exit"
        prev_committed = ctrl.committed
        trace.append({"pos": pos, "occ": occ, "top": top,
                      "committed": ctrl.committed, "edge": edge})
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pos += 1
        if pos > 1600:
            return trace, pos


def pick_handoffs(trace, death_pos) -> list[dict]:
    """Mechanical event selection: first accumulation, RECOVER edges, and
    the final two re-entries before death."""
    occ = [t["occ"] for t in trace]
    first_accum = None
    for i in range(ACCUM_WINDOW, len(occ)):
        if occ[i] - occ[i - ACCUM_WINDOW] >= ACCUM_THRESH:
            first_accum = trace[i]["pos"]
            break
    entries = [t["pos"] for t in trace if t["edge"] == "RECOVER-entry"]
    exits = [t["pos"] for t in trace if t["edge"] == "RECOVER-exit"]
    # final two re-entries (exits) before death; fall back to last entries.
    late = [p for p in exits if p < death_pos][-2:]
    wants: list[tuple[str, int | None]] = [
        ("first-accumulation", first_accum),
        ("first-RECOVER-entry", entries[0] if entries else None),
        ("first-RECOVER-exit", exits[0] if exits else None),
        ("final-reentry-1", late[0] if len(late) > 1 else None),
        ("final-reentry-2", late[-1] if late else None),
    ]
    # final pre-death state (10 pieces before terminal) as the hopelessness
    # probe — expected unrecoverable; bounds the window.
    wants.append(("pre-death-10", death_pos - 10 if death_pos >= 10 else None))
    out = []
    seen = set()
    for label, pos in wants:
        if pos is None or pos in seen or pos < 0 or pos >= death_pos:
            out.append({"label": label, "pos": pos, "skipped": True})
            continue
        seen.add(pos)
        out.append({"label": label, "pos": pos})
    return out


def run_continuation(net, seed, pos, mode, *, rescue_cap=RESCUE_CAP) -> dict:
    """From the state at (seed, pos): replay EXACTLY to pos with the commit
    controller (reproduces the recorded trajectory), then continue with:

      original : the same commitment controller (must reproduce death),
      noreentry: RECOVER fallback (bonus-carrying base scorer) with the
                 commitment latched OFF at handoff,
      champion : canonical bonus-free full 2-ply from handoff.
    """
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer
    from piece_history import PieceQueue
    import deathbot_sim as ds
    base_bonus = TwoPlyScorer(net, beam=10 ** 9,
                              line_bonus=dict(COMPLETION_BONUS))
    ctrl = CommitScorer(base_bonus, abort_top=16, abort_occ=110,
                        reenter_occ=60)
    pq = PieceQueue(random.Random(seed))
    board = ds.Board()
    for _ in range(pos):
        bulk = board.lock_candidates_full(pq.current)
        nc = len(bulk[0])
        assert nc and not all(bulk[3]), f"replay died early at {seed}"
        pick = ctrl.choose(board, pq.current, pq.next_piece)
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
    occ0, top0 = _occ_top(board)
    pieces = 0
    reason = "max_pieces"
    if mode == "original":
        scorer = ctrl  # same object: commitment state carries over
        latched = None
    elif mode == "noreentry":
        ctrl.committed = False  # latch recovery ON at handoff...
        scorer = ctrl
        latched = "reentry-disabled"
        # ...and keep it off: monkey-guard re-entry by pinning the exit
        # threshold unreachable for the rest of this continuation.
        ctrl.reenter_occ = -1
    elif mode == "champion":
        scorer = TwoPlyScorer(net, beam=10 ** 9)  # bonus-free canonical
        latched = None
    else:
        raise ValueError(mode)
    while pieces < rescue_cap:
        bulk = board.lock_candidates_full(pq.current)
        nc = len(bulk[0])
        if nc == 0:
            reason = "no_candidates"
            break
        if all(bulk[3]):
            reason = "lockout"
            break
        pick = scorer.choose(board, pq.current, pq.next_piece)
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    return {"mode": mode, "seed": seed, "handoff_pos": pos,
            "occ_at_handoff": occ0, "top_at_handoff": top0,
            "pieces_after": pieces, "reason": reason,
            "latched": latched}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_all = time.perf_counter()
    weights = str(resolve(args.weights))
    from model import TAPNet
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()

    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "deaths": {}, "wall_seconds": 0.0}
    for seed in DEATHS:
        trace, death_pos = instrumented_replay(net, seed)
        assert trace, f"empty trace for {seed}"
        handoffs = pick_handoffs(trace, death_pos)
        rows = []
        for h in handoffs:
            if h.get("skipped"):
                rows.append({**h, "continuations": None})
                continue
            conts = {}
            for mode in ("original", "noreentry", "champion"):
                conts[mode] = run_continuation(net, seed, h["pos"], mode)
            # reproduction check inline
            rep = conts["original"]
            rows.append({**h,
                         "occ": next(t["occ"] for t in trace
                                     if t["pos"] == h["pos"]),
                         "top": next(t["top"] for t in trace
                                     if t["pos"] == h["pos"]),
                         "continuations": conts,
                         "reproduced": bool(
                             rep["pieces_after"] + h["pos"] == death_pos
                             and rep["reason"] != "max_pieces")})
            c = conts
            print(f"  seed {seed} {h['label']}@{h['pos']}: "
                  f"orig {c['original']['pieces_after']} "
                  f"({c['original']['reason']}) | "
                  f"noreentry {c['noreentry']['pieces_after']} "
                  f"({c['noreentry']['reason']}) | "
                  f"champ {c['champion']['pieces_after']} "
                  f"({c['champion']['reason']})",
                  flush=True)
        out["deaths"][str(seed)] = {
            "death_pos": death_pos,
            "n_trace": len(trace),
            "handoffs": rows}
        print(f"[autopsy] seed {seed}: death @{death_pos}, "
              f"{len([h for h in handoffs if not h.get('skipped')])} handoffs",
              flush=True)
    out["wall_seconds"] = round(time.perf_counter() - t_all, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {op} in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
