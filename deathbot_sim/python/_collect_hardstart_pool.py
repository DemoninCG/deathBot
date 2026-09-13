"""Hard-start state pool for loop-gen-2 (2026-09-12 audit build).

Plays the working champion from recovery-suite start boards and records
the FIRST `max_states` decision states per game (the recovery phase —
after that it is normal play, already covered by the base bank).

Suites (audit §52.6 mixing guidance — bands where snap10 dies 2-3x as
often but snap11 survives; flat9/flat6 dropped as no-signal):
    flat12 flat15 flat18 rand12 holes4 holes6 holes8 drought

Start boards come from `_recovery_suite.build_start_board` with game
indices 100+ (the suite itself uses 0-19 — no board overlap, so the
pool never trains on the suite's eval boards). TAP pieces throughout
(drought suite: 6-forced-non-I prefix, then TAP — same as the suite).
True-preview greedy 1-ply, corrected lockout, separate-RNG convention —
the SAME policy the suite measures (2026-09-12 audit fix).

Output NPZ (viable candidates only; singles/all-dead flagged, not stored):
    pre_rows      [H, 21] uint32 (pre-move boards)
    cand_rows     flat uint32 (viable candidate rows, H states)
    cand_offsets  [H+1] int64
    placed_idx    [H] int8 (0=I..6=L, ALL_PIECES order)
    nxt_idx       [H] int8
    suite         [H] unicode (suite name per state)
    alldead       [H] bool (no viable candidates — skipped in fits)
    meta_json     {champion sha, suites, games_per_suite, max_states, seed}
Fit with `_selfdistill_loop2.py` (70% base bank + 30% this pool).

Usage:
    python _collect_hardstart_pool.py --weights ../_diag_out/_loop_champ_r1.pt \
        --out ../_diag_out/_hardstart_pool.npz --games 10 --max-states 300
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
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"

# Audit §52.6: signal bands only (flat9/flat6 carry no signal).
MIX_SUITES = ("flat12", "flat15", "flat18", "rand12",
              "holes4", "holes6", "holes8", "drought")
POOL_GAME0 = 100  # suite eval uses 0-19; pool uses 100+ (no overlap)


def resolve_weights(p: str) -> Path:
    supplied = Path(p).expanduser()
    for cand in (supplied, HERE / supplied, HERE.parent / supplied,
                 ROOT / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights file not found: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default=str(D / "_hardstart_pool.npz"))
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--max-states", type=int, default=300)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--master-seed", type=int, default=202_600)
    ap.add_argument("--suites", nargs="*", default=list(MIX_SUITES))
    args = ap.parse_args()

    from model import TAPNet, piece_to_index
    from _recovery_suite import SUITES, build_start_board
    from piece_history import PieceQueue
    from _recovery_suite import _DroughtQueue
    from afterstate import flat_rows_to_tensor
    import deathbot_sim as ds

    wpath = resolve_weights(args.weights)
    sha = hashlib.sha256(wpath.read_bytes()).hexdigest()[:16]
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(str(wpath), map_location="cpu"))
    net.eval()
    print(f"champion {wpath.name} (sha {sha})", flush=True)

    # Per-suite index must match the SUITES tuple (it feeds the seed).
    for s in args.suites:
        assert s in SUITES, f"unknown suite {s}"

    pre_rows, cand_chunks, offsets = [], [], [0]
    placed_idx, nxt_idx, suite_of, alldead = [], [], [], []
    n_games, n_full = 0, 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for suite in args.suites:
            for gi in range(args.games):
                game_index = POOL_GAME0 + gi
                board0, dk = build_start_board(suite, game_index,
                                               args.master_seed)
                # Rebuild the identical board the suite would use, then
                # play with the suite's own seed convention for the queue.
                seed = (args.master_seed + SUITES.index(suite) * 10_000
                        + game_index)
                board = board0.clone_board()
                prng = random.Random(seed + 7_000_000_001)
                pq = (_DroughtQueue(prng, dk) if dk
                      else PieceQueue(prng))
                kept = 0
                pieces = 0
                while pieces < args.cap and kept < args.max_states:
                    bulk = board.lock_candidates_full(pq.current)
                    nc = len(bulk[0])
                    if nc == 0:
                        break
                    viable = [i for i, lo in enumerate(bulk[3]) if not lo]
                    if not viable:
                        alldead.append(True)
                        break
                    rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
                    vrows = rows[viable]
                    boards = flat_rows_to_tensor(vrows.reshape(-1),
                                                 len(viable))
                    cur_t = torch.full((len(viable),),
                                       piece_to_index(pq.current),
                                       dtype=torch.long)
                    nxt_t = torch.full((len(viable),),
                                       piece_to_index(pq.next_piece),
                                       dtype=torch.long)
                    v = net(boards, cur_t, nxt_t).flatten().tolist()
                    order = sorted(range(len(viable)),
                                   key=lambda j: (-v[j], j))
                    pick = viable[order[0]]
                    # Record AFTER picking (state + viable set + ids).
                    pre_rows.append(np.asarray(board.row_bits(),
                                               dtype=np.uint32))
                    cand_chunks.append(np.asarray(vrows, dtype=np.uint32))
                    offsets.append(offsets[-1] + len(viable))
                    placed_idx.append(piece_to_index(pq.current))
                    nxt_idx.append(piece_to_index(pq.next_piece))
                    suite_of.append(suite)
                    alldead.append(False)
                    kept += 1
                    board = ds.Board.from_row_bits(
                        [int(x) for x in rows[pick]])
                    pq.advance()
                    pieces += 1
                n_games += 1
                if kept >= args.max_states:
                    n_full += 1
                print(f"  {suite} game {game_index}: {kept} states "
                      f"({pieces} pcs, {time.perf_counter() - t0:.0f}s)",
                      flush=True)

    H = len(pre_rows)
    out = Path(args.out)
    meta = {"champion": str(wpath), "sha": sha, "suites": list(args.suites),
            "games_per_suite": args.games, "game0": POOL_GAME0,
            "max_states": args.max_states, "cap": args.cap,
            "master_seed": args.master_seed, "n_states": H,
            "n_full_games": n_full, "n_games": n_games,
            "policy": "true-preview greedy 1-ply, corrected lockout, "
                      "separate-RNG (suite-identical)"}
    np.savez_compressed(
        str(out),
        pre_rows=np.stack(pre_rows) if H else np.zeros((0, 21), np.uint32),
        cand_rows=(np.concatenate([c.reshape(-1) for c in cand_chunks])
                   if H else np.zeros(0, np.uint32)),
        cand_offsets=np.asarray(offsets, dtype=np.int64),
        placed_idx=np.asarray(placed_idx, dtype=np.int8),
        nxt_idx=np.asarray(nxt_idx, dtype=np.int8),
        suite=np.asarray(suite_of, dtype="U8"),
        alldead=np.asarray(alldead, dtype=bool),
        meta_json=np.asarray(json.dumps(meta)))
    print(f"Wrote {out.name}: {H} states from {n_games} games "
          f"({n_full} full) in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
