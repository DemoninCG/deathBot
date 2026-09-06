"""Sampled multi-step rollouts with a style-capable continuation (§15.1.4).

Two-ply cannot see empty-board→Tetris plans. This driver does sampled
lookahead with COMMON RANDOM NUMBERS:

  For each ROOT candidate a of the current piece:
    for each of K sampled future tapes (TAP PieceHistory continuations):
      simulate D future placements with the well CONTROLLER continuation
      (NOT the flat champion — the champion dismantles setups and reports
      zero setup value, §15.1.4)
    score(a) = mean over tapes of [tetris_lines + gamma^d * Phi(end)]
               + Q_base,2ply(s, a) anchor (keeps survival grounding)

Preview-correct: the actual next piece (true preview) is move 0 of every
tape; beyond it, pieces are sampled from the TAP randomizer (history
reroll), never the evaluator's hidden tape.

Triggered selectively (§15.1.4): this module scores ONE decision; callers
invoke it at well start/abandon, competing plans, drought, or risk
thresholds — not every placement.

Usage:
    python diag_rollout.py --weights ../../model_weights_snapshot9.pt \
        --games 10 --seed0 20350 --depth 6 --tapes 4 \
        --out ../_diag_out/rollout_probe.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from well_potential import well_potential, WELL_COL  # noqa: E402


def _rows_array(flat_rows, n):
    return np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)


def rollout_value(board, tapes: list[list], depth: int,
                  well_col: int = WELL_COL,
                  gamma: float = 0.99) -> float:
    """Simulate the controller for `depth` pieces on each tape; return the
    mean [tetris-line count + gamma^depth * Phi(end)] over tapes."""
    from well_controller import controller_choose
    vals = []
    for tape in tapes:
        b = board.clone_board()
        tetris_lines = 0
        for d, kind in enumerate(tape[:depth]):
            bulk = b.lock_candidates_full(kind)
            n = len(bulk[0])
            if n == 0 or all(bulk[3]):
                tetris_lines -= 50  # death during rollout: heavy penalty
                break
            pick = controller_choose(b, kind, None, well_col=well_col)
            nl = int(bulk[5][pick])
            if nl >= 4:
                tetris_lines += nl
            rows = _rows_array(bulk[4], n)[pick]
            b = __import__("deathbot_sim").Board.from_row_bits(
                [int(v) for v in rows])
        end_phi = well_potential(np.asarray(b.row_bits(), dtype=np.uint32),
                                 well_col)
        vals.append(tetris_lines + (gamma ** depth) * end_phi)
    return statistics.fmean(vals) if vals else 0.0


def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, seed, games_cfg) = payload
    sys.path.insert(0, here)
    import random as _r
    import numpy as _np
    import deathbot_sim as _ds
    from model import TAPNet as _N
    from two_ply import TwoPlyScorer as _S
    from piece_history import PieceHistory as _PH
    from well_controller import controller_choose as _cc
    (cap, depth, tapes, gamma, well_col, use_rollout) = games_cfg
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    scorer = _S(net, beam=10 ** 9)
    rng = _r.Random(seed)
    from piece_history import PieceQueue as _PQ
    pq = _PQ(rng)
    board = _ds.Board()
    pieces = lines = 0
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    t0 = time.perf_counter()
    while pieces < cap:
        bulk = board.lock_candidates_full(pq.current)
        n = len(bulk[0])
        if n == 0:
            reason = "no_candidates"
            break
        if all(bulk[3]):
            reason = "lockout"
            break
        viable = [i for i, lo in enumerate(bulk[3]) if not lo]
        if not viable:
            reason = "lockout"
            break
        if use_rollout and len(viable) > 1:
            # Root candidates: top-6 by 1-ply value (diverse shortlist that
            # retains setup moves: top-3 + 3 sampled from the rest).
            res = board.lock_candidates_full(pq.current)
            rows0 = _np.asarray(res[4], dtype=_np.uint32).reshape(n, 21)
            from afterstate import flat_rows_to_tensor as _frt
            from model import piece_to_index as _pi
            boards = _frt(rows0[viable], len(viable))
            cur = torch.full((len(viable),), _pi(pq.current))
            with torch.no_grad():
                v1 = net(boards, cur, cur).flatten().tolist()
            order = sorted(range(len(viable)), key=lambda j: -v1[j])
            short = order[:3] + order[3:6]
            short = [j for j in short if j < len(viable)]
            # Common random numbers: same tapes for every root candidate.
            tape_rng = _r.Random(seed * 100003 + pieces)
            tapes_list = []
            for _ in range(tapes):
                # Move 0 = TRUE preview; beyond: TAP-randomizer samples.
                ph = _PH(tape_rng)
                tape = [pq.next_piece] + [ph.next() for _ in range(depth - 1)]
                tapes_list.append(tape)
            import well_potential as _wp
            best_j, best_s = short[0], float("-inf")
            for j in short:
                succ = _ds.Board.from_row_bits(
                    [int(v) for v in rows0[viable[j]]])
                rv = rollout_value(succ, tapes_list, depth - 1,
                                   well_col, gamma)
                s = v1[j] + rv
                if s > best_s:
                    best_s, best_j = s, j
            pick = viable[best_j]
        else:
            pick = scorer.choose(board, pq.current, pq.next_piece)
        nl = int(bulk[5][pick])
        lines += nl
        clears[nl] = clears.get(nl, 0) + 1
        rows = _np.asarray(bulk[4], dtype=_np.uint32).reshape(n, 21)[pick]
        board = _ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        pieces += 1
    else:
        reason = "max_pieces"
    return {"seed": seed, "pieces": pieces, "lines": lines,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=20350)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--tapes", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--well-col", type=int, default=WELL_COL)
    ap.add_argument("--no-rollout", action="store_true",
                    help="baseline: pure champion 2-ply on the same seeds")
    ap.add_argument("--workers", type=int, default=4,
                    help="default 4 (not 8): rollout workers are RAM-heavy "
                         "(net + sim per worker); never co-run with a sweep")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def _resolve(p):
        from pathlib import Path as _P
        s = _P(p).expanduser()
        for c in (s, HERE.parent / s, HERE.parent.parent / s,
                  HERE.parent.parent.parent / s):
            if c.is_file():
                return str(c.resolve())
        raise FileNotFoundError(p)

    weights = _resolve(args.weights)
    cfg = (args.max_pieces, args.depth, args.tapes, args.gamma,
           args.well_col, not args.no_rollout)
    seeds = [args.seed0 + i for i in range(args.games)]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    payloads = [(str(HERE), weights, s, cfg) for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_worker, p) for p in payloads]):
            rows.append(fut.result())
    elapsed = time.perf_counter() - t0
    rows.sort(key=lambda r: r["seed"])
    pieces = [r["pieces"] for r in rows]
    lines = [r["lines"] for r in rows]
    tet = sum(int(r["clears"].get("4", 0)) for r in rows)
    scoring = sum(int(r["clears"].get(k, 0)) for r in rows
                  for k in ("1", "2", "3", "4"))
    out = {"weights": weights, "games": args.games, "seed0": args.seed0,
           "depth": args.depth, "tapes": args.tapes,
           "rollout": not args.no_rollout,
           "mean_pieces": statistics.fmean(pieces),
           "mean_lines": statistics.fmean(lines),
           "tetrises": tet,
           "tetris_per_100": 100.0 * tet / max(1, sum(pieces)),
           "tetris_share": (tet / scoring) if scoring else 0.0,
           "capped": sum(1 for p in pieces if p >= args.max_pieces),
           "wall_s": round(elapsed, 1), "rows": rows}
    print(f"rollout={out['rollout']}: {out['mean_pieces']:.1f} pcs / "
          f"{out['mean_lines']:.1f} lines, tetris {tet} "
          f"({out['tetris_per_100']:.2f}/100, share {out['tetris_share']:.3f}), "
          f"cap {out['capped']}/{args.games} (wall {elapsed:.0f}s)")
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
