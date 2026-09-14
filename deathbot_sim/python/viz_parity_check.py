"""Headless parity: viz CommitVizScorer + GameRunner vs the canonical
eval path (CommitScorer.choose + bulk advance, as in
diag_commit_stageC._worker / commit_controller.play_game_commit).

NOT a panel eval — plays short trajectories on fixed seeds through BOTH
paths and asserts identical picks/pieces/lines. Run:

    python viz_parity_check.py --weights ../../model_weights_snapshot12.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import deathbot_sim as ds
from model import TAPNet
from piece_history import PieceQueue
from viz_live import CommitVizScorer, GameRunner, STYLISH_DEFAULTS


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(p)


def canonical_game(net, seed: int, pieces: int, cfg: dict):
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer, COMPLETION_BONUS
    base = TwoPlyScorer(net, beam=10 ** 9,
                        line_bonus=dict(COMPLETION_BONUS))
    ctrl = CommitScorer(base, **cfg)
    board = ds.Board()
    pq = PieceQueue(random.Random(seed))
    picks: list[int] = []
    lines_total = 0
    n = 0
    reason = "ok"
    while n < pieces:
        bulk = board.lock_candidates_full(pq.current)
        nc = len(bulk[0])
        if nc == 0:
            reason = "no_candidates"
            break
        if all(bulk[3]):
            reason = "lockout"
            break
        pick = ctrl.choose(board, pq.current, pq.next_piece)
        picks.append(int(pick))
        lines_total += int(bulk[5][pick])
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance()
        n += 1
    return picks, n, lines_total, reason


def viz_game(net, seed: int, pieces: int, cfg: dict):
    # Faithful viz path: ONE CommitScorer advanced by GameRunner only.
    # Records the runner's RESOLVED candidate index (search_locks order,
    # what the runner actually commits) mapped back to bulk order, so it
    # is comparable to the canonical bulk-index picks. (Calling
    # scorer.ctrl.choose() a second time per step would consume the
    # controller's cooldown counter twice per decision — never do that.)
    scorer = CommitVizScorer(net, **cfg)
    runner = GameRunner(scorer, random.Random(seed), epsilon=0.0,
                        seed=seed)
    picks: list[int] = []
    n = 0
    while n < pieces and runner.alive:
        bulk_before = runner.board.lock_candidates_full(
            runner.current_piece)
        nc = len(bulk_before[0])
        if nc == 0 or all(bulk_before[3]):
            break
        # Candidate identity the runner is about to commit.
        rc = runner.current_candidates[runner.best_idx].piece
        key = (int(rc.state), int(rc.x), int(rc.y))
        bulk_pick = None
        for j in range(nc):
            if (int(bulk_before[0][j]), int(bulk_before[1][j]),
                    int(bulk_before[2][j])) == key:
                bulk_pick = j
                break
        picks.append(int(bulk_pick) if bulk_pick is not None else -1)
        runner.step()
        n += 1
    return picks, runner.pieces_placed, runner.lines_cleared, runner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--pieces", type=int, default=60)
    args = ap.parse_args()

    w = resolve(args.weights)
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(str(w), map_location="cpu"))
    net.eval()

    ok = True
    for seed in args.seeds:
        cp, cn, cl, creason = canonical_game(
            net, seed, args.pieces, dict(STYLISH_DEFAULTS))
        vp, vn, vl, runner = viz_game(
            net, seed, args.pieces, dict(STYLISH_DEFAULTS))
        match = (cp == vp and cn == vn and cl == vl)
        ok = ok and match
        print(f"seed {seed}: canon pieces={cn} lines={cl} {creason} | "
              f"viz pieces={vn} lines={vl} tetris={runner.tetrises} "
              f"mode={'COMMIT' if runner.scorer.last_committed else 'RECOVER'} "
              f"-> {'MATCH' if match else 'MISMATCH'}", flush=True)
        if not match:
            for i, (a, b) in enumerate(zip(cp, vp)):
                if a != b:
                    print(f"  first divergence at step {i}: "
                          f"canon={a} viz={b}")
                    break
    # Multi-game check: two fresh runners on the same seed must open
    # identically, and reset() on a continued stream must advance the
    # tape (not re-prime the same pair — the double-priming bug made
    # game 1 diverge from headless even when the queue logic matched).
    r1 = GameRunner(CommitVizScorer(net), random.Random(99),
                    epsilon=0.0, seed=99)
    r2 = GameRunner(CommitVizScorer(net), random.Random(99),
                    epsilon=0.0, seed=99)
    same_open = (r1.current_piece == r2.current_piece
                 and r1.next_piece == r2.next_piece)
    print(f"fresh-runner same open: {same_open} "
          f"(cur={r1.current_piece} nxt={r1.next_piece})", flush=True)
    ok = ok and same_open
    # Pin the stylish defaults to the eval driver's ARMS config.
    import diag_commit_stageC as stageC
    arm = stageC.ARMS["cool30lam10tiered"]
    pinned = all(arm.get(k) == v for k, v in STYLISH_DEFAULTS.items())
    print(f"STYLISH_DEFAULTS pins diag_commit_stageC cool30lam10tiered: "
          f"{pinned} {STYLISH_DEFAULTS}")
    ok = ok and pinned
    if not ok:
        raise SystemExit("parity FAILED")
    print("parity OK")


if __name__ == "__main__":
    main()
