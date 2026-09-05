"""Value-function health probe: is a collapsed network 'opening-blind'?

For each checkpoint, play a few greedy games and, at every step, compare the
network's candidate ranking against the heuristic teacher's ranking:

  spread      - std of network scores across candidates (0 => constant output)
  scale       - mean |score| (output magnitude)
  top1_agree  - fraction of steps where net argmax == heuristic argmax
  kendall     - Kendall tau between net and heuristic candidate rankings

Bins are by pieces-placed-so-far, so we can see WHERE in the game a network
is sane (healthy late-game, garbage opening => the FIFO-overflow signature).

Run:  python diag_value_probe.py --out ../_diag_out/value_probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

BINS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 10_000)]
BIN_LABELS = ["0-9", "10-24", "25-49", "50-99", "100+"]


def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    order = sorted(range(n), key=lambda i: a[i])
    rb = [b[i] for i in order]
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rb[i] < rb[j]:
                conc += 1
            elif rb[i] > rb[j]:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2)


def _probe_task(payload):
    """Play greedy games with one checkpoint; return per-step ranking stats."""
    (weights_path, seeds, max_pieces) = payload
    torch.set_num_threads(1)
    from model import TAPNet
    from neural_scorer import NeuralScorer
    from heuristic import HeuristicScorer
    from afterstate import make_afterstate
    from selfplay import play_game

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()
    scorer = NeuralScorer(net=net, device="cpu")
    heur = HeuristicScorer()

    steps = []
    for seed in seeds:
        rng = random.Random(seed)
        # Re-walk each game manually so we can inspect candidates per step.
        import deathbot_sim as ds
        from piece_history import PieceQueue
        from model import piece_to_index

        board = ds.Board()
        pq = PieceQueue(rng)
        pos = 0
        while pos < max_pieces:
            candidates = board.search_locks(pq.current)
            if not candidates:
                break
            cur_idx = piece_to_index(pq.current)
            nxt_idx = piece_to_index(pq.next_piece)
            scores = scorer.score_batch(board, candidates,
                                        current_piece_idx=cur_idx,
                                        next_piece_idx=nxt_idx)
            h_scores = []
            for c in candidates:
                a = make_afterstate(board, c)
                h_scores.append(heur.score_afterstate(a.board, a.lines_cleared))
            net_best = max(range(len(candidates)), key=lambda i: scores[i])
            h_best = max(range(len(candidates)), key=lambda i: h_scores[i])
            steps.append({
                "pos": pos,
                "n_cand": len(candidates),
                "spread": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                "scale": statistics.fmean(abs(s) for s in scores),
                "agree": 1.0 if net_best == h_best else 0.0,
                "tau": kendall_tau(scores, h_scores),
            })
            pick = sorted(range(len(candidates)),
                          key=lambda i: (-scores[i], i))[0]
            a = make_afterstate(board, candidates[pick])
            board = a.board
            pq.advance()
            pos += 1
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pieces", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    here = HERE
    jobs = []
    for w in args.weights:
        p = Path(w)
        if not p.is_file():
            p = here.parent / w
        jobs.append((str(p.resolve()),
                     list(range(args.seed, args.seed + args.games)),
                     args.max_pieces))

    t0 = time.perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (weights_path, _, _), fut in zip(jobs, pool.map(_probe_task, jobs)):
            name = Path(weights_path).name
            steps = fut
            per_bin = []
            for (lo, hi), label in zip(BINS, BIN_LABELS):
                b = [s for s in steps if lo <= s["pos"] < hi]
                if not b:
                    per_bin.append({"bin": label, "n": 0})
                    continue
                per_bin.append({
                    "bin": label,
                    "n": len(b),
                    "n_cand": round(statistics.fmean(s["n_cand"] for s in b), 1),
                    "spread": round(statistics.fmean(s["spread"] for s in b), 4),
                    "scale": round(statistics.fmean(s["scale"] for s in b), 3),
                    "top1_agree": round(statistics.fmean(s["agree"] for s in b), 3),
                    "kendall": round(statistics.fmean(
                        s["tau"] for s in b if s["tau"] == s["tau"]), 3),
                })
            total_pieces = max((s["pos"] for s in steps), default=0) + 1
            results[name] = {"games": args.games, "max_pieces": args.max_pieces,
                             "steps": len(steps), "last_pos": total_pieces - 1,
                             "bins": per_bin}
            print(f"\n=== {name} (steps={len(steps)}, "
                  f"longest game reached piece {total_pieces}) ===")
            print(f"{'bin':>8s} {'n':>6s} {'cand':>6s} {'spread':>8s} "
                  f"{'scale':>7s} {'top1':>6s} {'tau':>6s}")
            for b in per_bin:
                if b["n"] == 0:
                    print(f"{b['bin']:>8s} {0:6d} {'-':>6s}")
                    continue
                print(f"{b['bin']:>8s} {b['n']:6d} {b['n_cand']:6.1f} "
                      f"{b['spread']:8.4f} {b['scale']:7.3f} "
                      f"{b['top1_agree']:6.3f} {b['kendall']:6.3f}")
    print(f"\n(probe finished in {time.perf_counter() - t0:.0f}s)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
