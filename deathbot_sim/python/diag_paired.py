"""Experiment 3: paired-rollout counterfactual labels (plan branch C2).

For selected decision states, clone the simulator and compare the incumbent's
chosen move against its top-1ply rivals under IDENTICAL sampled future piece
sequences. The label is observed survival per candidate — a within-state
causal comparison, immune to the across-state regression problem that killed
value learning (Phase 2).

Procedure per selected state:
  1. Rebuild the exact pre-move state (bank stores packed pre-boards; piece
     identities are recoverable by queue replay).
  2. Candidates: the incumbent's chosen move + top-`alts` 1-ply rivals
     (skipping lockout rivals — instant losses make uninformative labels).
  3. For each candidate: commit it, then continue greedily (1-ply incumbent)
     for `horizon` pieces. All candidates share the SAME `reps` future tapes
     (aligned by forcing the queue so the continuation starts with the known
     next piece).
  4. Label(c) = mean survival over reps.

Output: a counterfactual label bank (.npz) for later training arms, plus a
summary of the within-state spread (how much there is to learn).

Usage:
    python diag_paired.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \
        --weights ../../model_weights_snapshot9.pt \
        --states 120 --alts 2 --reps 3 --horizon 80 \
        --out ../_diag_out/paired_labels.npz
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

import deathbot_sim as ds
from model import TAPNet, piece_to_index  # noqa: E402
from diag_phase2 import Bank  # noqa: E402
from afterstate import flat_rows_to_tensor  # noqa: E402
from piece_history import PieceQueue  # noqa: E402


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def greedy_continuation(scorer, board: "ds.Board", pq: PieceQueue,
                        max_pieces: int) -> int:
    """Greedy 1-ply continuation from (board, pq). Returns placements made."""
    pieces = 0
    while pieces < max_pieces:
        cands = board.search_locks(pq.current)
        if not cands or all(board.locks_out(c.piece) for c in cands):
            break
        cur_idx = piece_to_index(pq.current)
        nxt_idx = piece_to_index(pq.next_piece)
        scores = scorer.score_batch(board, cands, current_piece_idx=cur_idx,
                                    next_piece_idx=nxt_idx)
        viable = [i for i, c in enumerate(cands) if not board.locks_out(c.piece)]
        pick = max(viable, key=lambda i: (scores[i], -i))
        rows = np.asarray(
            board.lock_candidates_full(pq.current)[4],
            dtype=np.uint32).reshape(len(cands), 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        board.clear_lines()
        pq.advance()
        pieces += 1
    return pieces


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--alts", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--tape-seed0", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    bank = Bank(args.bank)
    weights_path = resolve_weights(args.weights)
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()
    from neural_scorer import NeuralScorer
    scorer = NeuralScorer(net=net, device="cpu")
    print(f"Bank: {bank.n_states} states; weights {weights_path.name}")

    n = bank.n_states
    cand_offsets = bank.cand_offsets
    n_cand = int(cand_offsets[-1])
    placed_ids, next_ids = bank._replay_piece_ids()

    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
        cand_cur[lo:hi] = piece_to_index(placed_ids[s])
        cand_nxt[lo:hi] = piece_to_index(next_ids[s])
    inc_vals = torch.empty(n_cand)
    with torch.no_grad():
        for lo in range(0, n_cand, 4096):
            hi = min(lo + 4096, n_cand)
            inc_vals[lo:hi] = net(cand_boards[lo:hi], cand_cur[lo:hi],
                                  cand_nxt[lo:hi]).flatten()

    ambiguous: list[tuple[float, int]] = []
    for s in range(n):
        lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
        if hi - lo < 2:
            continue
        v = inc_vals[lo:hi]
        top2 = torch.topk(v, 2).values
        gap = float(top2[0] - top2[1])
        if gap <= args.margin:
            ambiguous.append((gap, s))
    rng = random.Random(args.seed)
    rng.shuffle(ambiguous)
    picked = [s for _g, s in ambiguous[: args.states]]
    print(f"ambiguous states (gap <= {args.margin}): {len(ambiguous)}; "
          f"picked {len(picked)}")

    labels_states: list[int] = []
    labels_offsets: list[int] = [0]
    labels_rows: list[np.ndarray] = []
    labels_values: list[float] = []
    labels_chosen_pos: list[int] = []

    t0 = time.perf_counter()
    done = 0
    for s in picked:
        lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
        pre_board = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        kind = placed_ids[s]
        next_kind = next_ids[s]
        live = pre_board.search_locks(kind)
        c_states = bank.cand_state[lo:hi]
        c_xs = bank.cand_x[lo:hi]
        c_ys = bank.cand_y[lo:hi]
        geom_to_live = {(c.piece.state, c.piece.x, c.piece.y): c
                        for c in live}

        chosen_bank = int(bank.chosen_idx[s])
        order = sorted(range(hi - lo), key=lambda j: -float(inc_vals[lo + j]))
        picked_bank = [chosen_bank] + [j for j in order
                                       if j != chosen_bank][: args.alts]
        chosen_live = geom_to_live.get(
            (int(c_states[chosen_bank]), int(c_xs[chosen_bank]),
             int(c_ys[chosen_bank])))
        if chosen_live is None:
            continue
        to_test = [chosen_live]
        picked_bank_list = [chosen_bank]
        for j in picked_bank[1:]:
            geom = (int(c_states[j]), int(c_xs[j]), int(c_ys[j]))
            c = geom_to_live.get(geom)
            if c is not None and not pre_board.locks_out(c.piece):
                to_test.append(c)
                picked_bank_list.append(j)

        tape_seeds = [args.tape_seed0 + args.seed * 100_000 + done * 10 + r
                      for r in range(args.reps)]
        vals: list[float] = []
        for c in to_test:
            (st, xs_, ys_, lockouts, flat_rows, lines) = \
                pre_board.lock_candidates_full(kind)
            rows_per = np.asarray(flat_rows, dtype=np.uint32).reshape(
                len(st), 21)
            row = None
            for i in range(len(st)):
                if (st[i] == c.piece.state and xs_[i] == c.piece.x
                        and ys_[i] == c.piece.y):
                    row = rows_per[i]
                    break
            assert row is not None
            survivals = []
            for tape_seed in tape_seeds:
                b2 = ds.Board.from_row_bits([int(v) for v in row])
                pq = PieceQueue(random.Random(tape_seed))
                for _ in range(8):
                    if pq.current == next_kind:
                        break
                    pq.advance()
                survivals.append(greedy_continuation(scorer, b2, pq,
                                                     args.horizon))
            vals.append(statistics.fmean(survivals))

        for j, c, val in zip(picked_bank_list, to_test, vals):
            labels_rows.append(
                bank.cand_rows[(lo + j) * 21:(lo + j) * 21 + 21])
            labels_values.append(val)
        labels_offsets.append(labels_offsets[-1] + len(to_test))
        labels_states.append(s)
        labels_chosen_pos.append(0)
        done += 1
        if done % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {done}/{len(picked)} states ({el:.0f}s, "
                  f"{el / done:.1f}s/state)", flush=True)

    packed = (np.stack(labels_rows) if labels_rows
              else np.zeros((0, 21), np.uint32))
    spread = [max(labels_values[a:b]) - min(labels_values[a:b])
              for a, b in zip(labels_offsets, labels_offsets[1:])]
    meta = {
        "weights": str(weights_path),
        "states_requested": args.states, "alts": args.alts,
        "reps": args.reps, "horizon": args.horizon, "margin": args.margin,
        "tape_seed0": args.tape_seed0, "seed": args.seed,
        "n_states_labeled": len(labels_states),
        "mean_spread": statistics.fmean(spread) if spread else 0.0,
        "median_spread": statistics.median(spread) if spread else 0.0,
    }
    np.savez_compressed(
        args.out,
        state_ids=np.asarray(labels_states, dtype=np.int32),
        cand_offsets=np.asarray(labels_offsets, dtype=np.int64),
        cand_rows=packed.reshape(-1),
        cand_values=np.asarray(labels_values, dtype=np.float32),
        chosen_pos=np.asarray(labels_chosen_pos, dtype=np.int16),
        meta=np.asarray(json.dumps(meta)))
    print(f"\nlabeled {len(labels_states)} states in "
          f"{time.perf_counter() - t0:.0f}s "
          f"({(time.perf_counter() - t0) / max(1, len(labels_states)):.1f} "
          f"s/state)")
    print(f"within-state survival spread: mean {meta['mean_spread']:.1f}, "
          f"median {meta['median_spread']:.1f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
