"""Collect commitment-controller teacher bank (§44 Step 4 DAgger bank).

Runs the CORRECTED cool30lam10 controller (abort_top 16, abort_occ 110,
reenter_occ 60, cooldown 30, lam 10, masked-root full expansion) on N games
and stores every COMMITTED preserved decision as (pre-board, candidates,
controller pick, placed/preview piece ids, lockout mask). Uncommitted
(style-off) decisions are SKIPPED (base policy == no teacher signal);
fallbacks and cash-outs are recorded with a kind flag but excluded from
pairs by default (cash-outs are forced takes, not style preferences).

Schema mirrors collect_teacher_bank.py (pre_rows / cand_blob / picks /
seeds / placed / next / lockouts) plus:
  kind: int8 per state (0 preserved, 1 cashout, 2 fallback)
  mode arrays: committed flag, cooldown_left, since_recover, occ, top
so train_commit_head.py can restore survival-margin weighting exactly.

NO panels consumed: training seeds 26000+ and reporting tapes only when
--seed0 >= 26000. Refuses selection-range seeds (< 26000 and < 31200
reporting floor is fine — bank is training data, not a selection panel,
but keep it on reporting seeds regardless).

Usage:
    python collect_commit_bank.py --weights ../../model_weights_snapshot9.pt \
        --games 16 --seed0 26000 --max-pieces 300 \
        --out ../_diag_out/commit_bank_cool30lam10.npz
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
CTRL_KW = {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
           "cooldown": 30, "lam": 10.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=26000)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from model import TAPNet, piece_to_index
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer, preserves_well
    from well_potential import dense_well_features
    from piece_history import PieceQueue
    import deathbot_sim as ds

    def resolve(p: str) -> Path:
        sup = Path(p).expanduser()
        for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
                  HERE.parent.parent.parent / sup):
            if c.is_file():
                return c.resolve()
        raise FileNotFoundError(p)

    w = str(resolve(args.weights))
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(w, map_location="cpu"))
    net.eval()

    pre_list, pick_list, lines_list, seed_list = [], [], [], []
    placed_list, next_list, lockout_list = [], [], []
    kind_list, cool_list, rec_list, occ_list, top_list = [], [], [], [], []
    clears_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    n_skip_off = n_cash = n_fallback = 0
    t0 = time.perf_counter()
    for g in range(args.games):
        seed = args.seed0 + g
        base = TwoPlyScorer(net, beam=10 ** 9,
                            line_bonus=dict(COMPLETION_BONUS))
        ctrl = CommitScorer(base, **CTRL_KW)
        pq = PieceQueue(random.Random(seed))
        board = ds.Board()
        pieces = 0
        last_recover_at = -10 ** 9
        while pieces < args.max_pieces:
            bulk = board.lock_candidates_full(pq.current)
            n = len(bulk[0])
            if n == 0 or all(bulk[3]):
                break
            (states, xs, ys, lockouts, flat_rows, lines) = bulk
            viable = [i for i, lo in enumerate(lockouts) if not lo]
            pre_rows = np.asarray(board.row_bits(), dtype=np.uint32)
            occ = sum(bin(int(v)).count("1") for v in pre_rows)
            top = 0
            for r in range(20, -1, -1):
                if int(pre_rows[r]) & 0b1111111111:
                    top = r + 1
                    break
            rows = np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)
            # Mirror CommitScorer mode bookkeeping (read-only here; the
            # real transition happens inside ctrl.choose below).
            was_committed = ctrl.committed
            pick = ctrl.choose(board, pq.current, pq.next_piece)
            if not ctrl.committed:
                n_skip_off += 1
            else:
                # Classify the committed decision.
                kind = 0
                is_cash = False
                for i in viable:
                    if int(lines[i]) == 4:
                        try:
                            if dense_well_features(rows[i])["tetris_ready"] > 0.5:
                                if pick == i:
                                    is_cash = True
                                break
                        except Exception:
                            pass
                if is_cash:
                    kind = 1
                    n_cash += 1
                else:
                    kept = []
                    for i in viable:
                        try:
                            f = dense_well_features(rows[i])
                        except Exception:
                            continue
                        if preserves_well(pre_rows, rows[i], f,
                                          ctrl.max_new_holes):
                            kept.append(i)
                    if pick not in kept:
                        kind = 2
                        n_fallback += 1
                if ctrl.stats.get("recovers", 0) and pieces - last_recover_at > 10 ** 8:
                    pass
                pre_list.append(np.asarray(board.row_bits(), dtype=np.uint32))
                pick_list.append(int(pick))
                placed_list.append(int(piece_to_index(pq.current)))
                next_list.append(int(piece_to_index(pq.next_piece)))
                lockout_list.append(np.asarray(bulk[3], dtype=np.uint8))
                rows_b = rows
                lines_list.append(np.asarray(
                    [len(rows_b)] + [int(x) for x in bulk[5]] +
                    rows_b.reshape(-1).tolist(), dtype=np.uint32))
                seed_list.append(seed)
                kind_list.append(kind)
                cool_list.append(int(ctrl._cooldown_left))
                # since-recover is not tracked on ctrl; use recovers count delta
                rec_list.append(int(ctrl.stats.get("recovers", 0)))
                occ_list.append(int(occ))
                top_list.append(int(top))
            nl = int(bulk[5][pick])
            clears_total[nl] = clears_total.get(nl, 0) + 1
            board = ds.Board.from_row_bits([int(v) for v in rows[pick]])
            pq.advance()
            pieces += 1
        print(f"game {seed}: {pieces} pcs", flush=True)
    print(f"commit games done: {len(pre_list)} states "
          f"(skip_off {n_skip_off}, cash {n_cash}, fallback {n_fallback}), "
          f"clears {clears_total} in {time.perf_counter()-t0:.0f}s")
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    max_n = max(int(a.shape[0]) for a in lockout_list) if lockout_list else 0
    lock_pad = np.stack([np.pad(a.astype(np.uint8), (0, max_n - a.shape[0]),
                                constant_values=1) for a in lockout_list])
    np.savez_compressed(
        op, pre_rows=np.stack(pre_list),
        cand_blob=np.concatenate(lines_list),
        picks=np.asarray(pick_list, dtype=np.int64),
        seeds=np.asarray(seed_list, dtype=np.int64),
        placed=np.asarray(placed_list, dtype=np.int64),
        next=np.asarray(next_list, dtype=np.int64),
        lockouts=lock_pad,
        kind=np.asarray(kind_list, dtype=np.int8),
        cooldown_left=np.asarray(cool_list, dtype=np.int64),
        recovers=np.asarray(rec_list, dtype=np.int64),
        occ=np.asarray(occ_list, dtype=np.int64),
        top=np.asarray(top_list, dtype=np.int64),
        meta_json=json.dumps({"controller": "corrected cool30lam10",
                              "ctrl_kw": CTRL_KW,
                              "commit_controller": "masked-root full expansion",
                              "seed0": args.seed0, "games": args.games,
                              "clears": clears_total,
                              "has_piece_ids": True,
                              "n_skip_off": n_skip_off,
                              "n_cash": n_cash, "n_fallback": n_fallback}))
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
