"""Collect 2-ply+Phi teacher trajectories for DAgger (§15.1.5).

Runs the dense-Phi teacher (full 2-ply + style_fn λ + completion bonus) on
N games and stores every visited (pre-board, candidates, teacher pick) so
diag_dagger can train the decoupled head on the teacher's OWN distribution
— not the flat champion's bank (§13.3 lesson).

Usage:
    python collect_teacher_bank.py --weights ../../model_weights_snapshot9.pt \\
        --lambda 20 --games 16 --seed0 20400 \\
        --out ../_diag_out/teacher_bank_lam20.npz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=20.0)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=20400)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from model import TAPNet
    from two_ply import TwoPlyScorer
    from well_potential import style_fn_factory
    from piece_history import PieceQueue
    import deathbot_sim as ds

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval()
    style_fn = style_fn_factory(scale=float(args.lam))
    scorer = TwoPlyScorer(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS),
                          style_fn=style_fn, style_alpha=1.0)
    pre_list, pick_list, lines_list, seed_list = [], [], [], []
    clears_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    t0 = time.perf_counter()
    for g in range(args.games):
        seed = args.seed0 + g
        rng = random.Random(seed)
        pq = PieceQueue(rng)
        board = ds.Board()
        pieces = 0
        while pieces < args.max_pieces:
            bulk = board.lock_candidates_full(pq.current)
            n = len(bulk[0])
            if n == 0 or all(bulk[3]):
                break
            pick = scorer.choose(board, pq.current, pq.next_piece)
            pre_list.append(np.asarray(board.row_bits(), dtype=np.uint32))
            pick_list.append(int(pick))
            # candidates of this state (packed) for later feature extraction
            rows = np.asarray(bulk[4], dtype=np.uint32).reshape(n, 21)
            lines_list.append(np.asarray(
                [len(rows)] + [int(x) for x in bulk[5]] +
                rows.reshape(-1).tolist(), dtype=np.uint32))
            seed_list.append(seed)
            nl = int(bulk[5][pick])
            clears_total[nl] = clears_total.get(nl, 0) + 1
            board = ds.Board.from_row_bits([int(v) for v in rows[pick]])
            pq.advance()
            pieces += 1
        print(f"game {seed}: {pieces} pcs", flush=True)
    # Ragged candidate lists → store as flat + offsets.
    print(f"teacher games done: {len(pre_list)} states, "
          f"clears {clears_total} in {time.perf_counter()-t0:.0f}s")
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        op, pre_rows=np.stack(pre_list),
        cand_blob=np.concatenate(lines_list),
        picks=np.asarray(pick_list, dtype=np.int64),
        seeds=np.asarray(seed_list, dtype=np.int64),
        meta_json=json.dumps({"lam": args.lam, "seed0": args.seed0,
                              "games": args.games,
                              "clears": clears_total}))
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
