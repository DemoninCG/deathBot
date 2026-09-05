"""Reference state bank builder (phase A5 of the 2026-09 training plan).

Plays greedy games with a fixed checkpoint on deterministic seeds and stores
a stratified "bank" of decision states for later diagnostic use:

  * For every step: the pre-move board (packed row bitmasks), placed/preview
    piece indices, the FULL candidate list (state/x/y + lockout flags),
    teacher (heuristic) scores, and the chosen candidate.
  * Games are stratified into opening / midgame / endgame states so Phase 2
    update-damage experiments can measure ranking drift on identical inputs.
  * Everything is stored compactly (packed uint32 rows) so a bank of tens of
    thousands of states is a few MB.

Usage:
    python bank_builder.py --weights ../model_weights_snapshot7.pt \
        --games 32 --out ../_diag_out/state_bank_snapshot7.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STRATA = [(0, 15, "opening"), (15, 60, "midgame"), (60, 10**9, "endgame")]


def resolve_weights(path: str) -> Path:
    supplied = Path(path).expanduser()
    for candidate in (supplied, HERE.parent / supplied,
                      HERE.parent.parent / supplied):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"weights file not found: {path}")


def build_bank(weights_path: str, *, games: int, seed: int = 42,
               max_pieces: int = 1500, max_states_per_game: int | None = None,
               verbose: bool = False) -> dict:
    """Play `games` greedy games; return packed state-bank arrays.

    Every visited decision state is recorded regardless of which candidate
    the policy chose, so later experiments can evaluate ANY candidate.
    """
    import torch

    from model import TAPNet, piece_to_index
    from neural_scorer import NeuralScorer
    from heuristic import HeuristicScorer
    from afterstate import flat_rows_to_tensor

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()
    scorer = NeuralScorer(net=net, device="cpu")
    heur = HeuristicScorer()

    import deathbot_sim as ds
    from piece_history import PieceQueue

    pre_rows: list[np.ndarray] = []       # [n, 21] uint32
    cand_offsets: list[int] = [0]         # running offset into flat arrays
    cand_state: list[int] = []
    cand_x: list[int] = []
    cand_y: list[int] = []
    cand_lockout: list[bool] = []
    cand_rows: list[np.ndarray] = []      # flat n*21 uint32
    cand_lines: list[int] = []
    teacher_scores: list[float] = []
    chosen_idx: list[int] = []
    step_pos: list[int] = []
    game_ids: list[int] = []
    game_lengths: list[int] = []

    for g in range(games):
        game_seed = seed + g
        rng = random.Random(game_seed)
        board = ds.Board()
        pq = PieceQueue(rng)
        pos = 0
        kept = 0
        while pos < max_pieces:
            current = pq.current
            candidates = board.search_locks(current)
            if not candidates:
                break
            cur_idx = piece_to_index(current)
            nxt_idx = piece_to_index(pq.next_piece)
            net_scores = scorer.score_batch(board, candidates,
                                            current_piece_idx=cur_idx,
                                            next_piece_idx=nxt_idx)
            pick = max(range(len(candidates)), key=lambda i: net_scores[i])

            # One bulk call serves BOTH recording and committing.
            (states, xs, ys, lockouts, flat_rows, lines) = \
                board.lock_candidates_full(current)
            n = len(states)

            # Record the state + full candidate set.
            if max_states_per_game is None or kept < max_states_per_game:
                pre_rows.append(np.asarray(board.row_bits(), dtype=np.uint32))
                for c in candidates:
                    cand_state.append(c.piece.state)
                    cand_x.append(c.piece.x)
                    cand_y.append(c.piece.y)
                    cand_lockout.append(board.locks_out(c.piece))
                cand_rows.append(np.asarray(flat_rows, dtype=np.uint32))
                cand_lines.extend(lines)
                # Teacher scores per candidate (post-clear afterstates) via
                # the packed-rows batch path (exact parity with per-board
                # reward, including invisible-row content).
                packed = np.asarray(flat_rows, dtype=np.uint32).reshape(
                    len(candidates), 21)
                teacher_scores.extend(
                    heur._weighted_reward_batch(
                        packed, [int(x) for x in lines])["rewards"])
                chosen_idx.append(pick)
                step_pos.append(pos)
                game_ids.append(g)
                cand_offsets.append(cand_offsets[-1] + len(candidates))
                kept += 1

            # Commit the chosen candidate: rebuild board from its rows.
            chosen_rows = np.asarray(flat_rows, dtype=np.uint32).reshape(
                n, 21)[pick]
            board = ds.Board.from_row_bits([int(v) for v in chosen_rows])
            pq.advance()
            pos += 1
        game_lengths.append(pos)
        if verbose:
            print(f"  game {g} (seed {game_seed}): {pos} pieces", flush=True)

    packed_pre = np.stack(pre_rows) if pre_rows else np.zeros((0, 21), np.uint32)
    packed_cand = (np.concatenate(cand_rows) if cand_rows
                   else np.zeros(0, np.uint32))
    return {
        "pre_rows": packed_pre,
        "cand_offsets": np.asarray(cand_offsets, dtype=np.int64),
        "cand_state": np.asarray(cand_state, dtype=np.int8),
        "cand_x": np.asarray(cand_x, dtype=np.int8),
        "cand_y": np.asarray(cand_y, dtype=np.int8),
        "cand_lockout": np.asarray(cand_lockout, dtype=bool),
        "cand_rows": packed_cand,
        "cand_lines": np.asarray(cand_lines, dtype=np.int16),
        "teacher_scores": np.asarray(teacher_scores, dtype=np.float32),
        "chosen_idx": np.asarray(chosen_idx, dtype=np.int16),
        "step_pos": np.asarray(step_pos, dtype=np.int32),
        "game_ids": np.asarray(game_ids, dtype=np.int32),
        "game_lengths": np.asarray(game_lengths, dtype=np.int32),
        "meta": {
            "weights": str(weights_path),
            "weights_sha256": hashlib.sha256(
                Path(weights_path).read_bytes()).hexdigest(),
            "games": games,
            "seed0": seed,
            "max_pieces": max_pieces,
            "n_states": int(packed_pre.shape[0]),
            "n_candidates": int(cand_offsets[-1]) if cand_offsets else 0,
            "mean_game_length": statistics.fmean(game_lengths) if game_lengths else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pieces", type=int, default=1500)
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    weights = resolve_weights(args.weights)
    t0 = time.perf_counter()
    bank = build_bank(str(weights), games=args.games, seed=args.seed,
                      max_pieces=args.max_pieces, verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(bank["meta"], indent=2)
    arrays = {k: v for k, v in bank.items() if k != "meta"}
    np.savez_compressed(out, meta_json=np.asarray(meta_json), **arrays)

    print(f"\nBank: {bank['meta']['n_states']} states, "
          f"{bank['meta']['n_candidates']} candidates, "
          f"mean game {bank['meta']['mean_game_length']:.0f} pieces")
    print(f"Wrote {out} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
