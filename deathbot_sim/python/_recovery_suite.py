"""Randomized recovery suite (2026-09-12).

Motivation: gen-2's promotion gain is concentrated in long-game survival
(61% vs 18% cap-3000), cap-1500 is saturating, and the style program
needs to know whether the new champion is genuinely better at RECOVERY
before any height-adjustable-lambda work. A recovery suite isolates the
skill directly: start the net in deliberately bad positions and measure
how long it survives from there.

Design (user request: random stack shapes + random holes so recovery is
evaluated GENERALLY, not on hand-built special cases):

  Suites (per-game start boards are built ONCE from a seeded RNG and
  reused identically across arms — fully paired and reproducible):
    flat{6,9,12,15}   flat stack at fixed height (pure dig test)
    rand{9,12}        random per-column heights in [base-2, base+2]
    holes{2,4,6}      flat height-12 stack with k random buried holes
    drought           flat height-9 stack; first 6 queue pieces forced
                      non-I (drought recovery), then TAP distribution
  Every game: standard TAP piece generator (PieceQueue), corrected
  lockout semantics, cap 1500.

  Metrics per (arm, suite): mean/median pieces survived, deaths (<300),
  capped count; paired per-seed deltas vs the first arm.

NOT a gate and NOT a promotion claim: read-only diagnostic; results feed
the style program (which height/hole bands have headroom) and the
hard-benchmark decision.

Usage:
    python _recovery_suite.py --weights ../../model_weights_snapshot11.pt \
        ../../model_weights_snapshot10.pt --names snap11 snap10 \
        --out ../_diag_out/_recovery_suite.json
Incremental: writes after each (arm, suite); interrupt-safe; resume-safe.
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
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"

from afterstate import flat_rows_to_tensor  # noqa: E402


def resolve_weights(p: str) -> Path:
    supplied = Path(p).expanduser()
    for cand in (supplied, HERE / supplied, HERE.parent / supplied,
                 ROOT / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights file not found: {p}")

GAMES_PER_SUITE = 20
CAP = 1500
MASTER_SEED = 202_600
# NOTE: append-only — suite indices feed the per-game seed
# (master + index*10000 + game), so inserting in the middle would
# re-seed every later suite. flat18/holes8 appended 2026-09-12.
SUITES = ("flat6", "flat9", "flat12", "flat15",
          "rand9", "rand12", "holes2", "holes4", "holes6", "drought",
          "flat18", "holes8")


def build_start_board(suite: str, game_index: int,
                      master_seed: int) -> tuple["ds.Board", int]:
    """Deterministic start board for (suite, game). Returns (board, drought_k).

    GUARANTEE (2026-09-12 fix, user catch): every stack leaves ONE random
    column completely EMPTY. Reason: a board with all 10 columns filled to
    any common height contains complete rows that auto-clear on the first
    lock — the recovery state would erase itself. The open column
    guarantees every row is incomplete. Randomness: open-column placement,
    per-column heights (rand suites), and buried holes (holes suites) come
    from a seeded RNG, so every arm faces identical starts and the suite
    is reproducible.
    """
    import deathbot_sim as ds
    # stable per-game seed (hash() is salted per process — never use it)
    seed = master_seed + SUITES.index(suite) * 10_000 + game_index
    rng = random.Random(seed)
    open_col = rng.randrange(10)

    if suite.startswith("flat"):
        heights = [int(suite[4:])] * 10
        holes: set[tuple[int, int]] = set()
        drought_k = 0
    elif suite.startswith("rand"):
        base = int(suite[4:])
        heights = [max(0, min(19, base + rng.randint(-2, 2)))
                   for _ in range(10)]
        holes = set()
        drought_k = 0
    elif suite.startswith("holes"):
        k = int(suite[5:])
        heights = [12] * 10
        holes = set()
        made = 0
        for c in rng.sample(range(10), 10):
            if made >= k or c == open_col:
                continue
            # buried hole: missing cell with a filled cell above it
            r = rng.randint(1, heights[c] - 2)
            holes.add((c, r))
            made += 1
        drought_k = 0
    elif suite == "drought":
        heights = [9] * 10
        holes = set()
        drought_k = 6
    else:
        raise ValueError(suite)

    heights[open_col] = 0  # the open column: guarantees no complete rows
    holes = {(c, r) for (c, r) in holes if c != open_col}
    board = ds.Board()
    for c, hh in enumerate(heights):
        for r in range(hh):
            if (c, r) in holes:
                continue
            board.set_cell(c, r)
    return board, drought_k


class _DroughtQueue:
    """PieceQueue with the first k draws forced to non-I pieces, then the
    standard TAP history-reroll distribution."""

    def __init__(self, rng: random.Random, k: int) -> None:
        from piece_history import PieceHistory
        self.rng = rng
        self.k = k
        self.hist = PieceHistory(rng)
        self._drawn = 0
        self.current = self._next()
        self.next_piece = self._next()

    def _next(self):
        import deathbot_sim as ds
        from piece_history import ALL_PIECES
        if self._drawn < self.k:
            self._drawn += 1
            cands = [p for p in ALL_PIECES if p != ds.PieceType.I]
            p = self.rng.choice(cands)
            self.hist.history.append(p)
            return p
        return self.hist.next()

    def advance(self) -> "ds.PieceType":
        self.current = self.next_piece
        self.next_piece = self._next()
        return self.current

    def reset(self) -> None:
        pass


def play_recovery_game(net, board0: "ds.Board", drought_k: int, seed: int,
                       cap: int = CAP) -> dict:
    """One recovery game: fixed start board, TAP pieces (first k forced
    non-I if drought_k), greedy 1-ply, corrected lockout semantics."""
    import deathbot_sim as ds
    from model import piece_to_index
    from piece_history import PieceQueue
    import random as _random

    board = board0.clone_board()
    prng = _random.Random(seed + 7_000_000_001)  # separate-RNG convention
    pq = _DroughtQueue(prng, drought_k) if drought_k else PieceQueue(prng)

    pieces = 0
    reason = "max_pieces"
    while pieces < cap:
        bulk = board.lock_candidates_full(pq.current)
        nc = len(bulk[0])
        if nc == 0:
            reason = "no_candidates"
            break
        viable = [i for i, lo in enumerate(bulk[3]) if not lo]
        if not viable:
            reason = "lockout"
            break
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
        boards = flat_rows_to_tensor(rows[viable].reshape(-1), len(viable))
        cur_t = torch.full((len(viable),), piece_to_index(pq.current),
                           dtype=torch.long)
        nxt_t = torch.full((len(viable),), piece_to_index(pq.next_piece),
                           dtype=torch.long)
        # 2026-09-12 audit fix: score with the TRUE preview (same policy
        # as the survival sweep harness via NeuralScorer.score_batch, which
        # only uses score_batch_fast when cur == nxt). The old code called
        # score_batch_fast unconditionally, duplicating cur into both
        # slots and silently discarding nxt_t (the versioned 2-ply
        # approximation) — a preview-blind policy on exactly the states
        # where preview matters most.
        with torch.inference_mode():
            v = net(boards, cur_t, nxt_t).flatten().tolist()
        order = sorted(range(len(viable)), key=lambda j: (-v[j], j))
        pick = viable[order[0]]
        board = ds.Board.from_row_bits([int(x) for x in rows[pick]])
        pq.advance()
        pieces += 1
    return {"pieces": pieces, "reason": reason, "capped": pieces >= cap}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--names", nargs="*",
                    help="short names for the arms (order matches)")
    ap.add_argument("--out", default=str(D / "_recovery_suite.json"))
    ap.add_argument("--games", type=int, default=GAMES_PER_SUITE)
    ap.add_argument("--master-seed", type=int, default=MASTER_SEED)
    ap.add_argument("--suites", nargs="*", default=list(SUITES))
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}: arms "
              f"{sorted(results.get('arms', {}))}", flush=True)
    results.setdefault("note",
                       "Randomized recovery suite (read-only diagnostic; "
                       "not a gate); identical start boards across arms; "
                       "TAP pieces; cap 1500. Tape convention: separate-RNG "
                       "(Random(seed+7_000_000_001)) with direct greedy "
                       "argmax — do NOT compare these tape labels against "
                       "sweep-harness (aliased) tapes (§33).")
    results.setdefault("arms", {})
    results.setdefault("arm_sha", {})
    results.setdefault("games", args.games)
    results.setdefault("master_seed", args.master_seed)

    arms = []
    for i, w in enumerate(args.weights):
        name = args.names[i] if args.names and i < len(args.names) \
            else Path(w).stem
        arms.append((name, resolve_weights(w)))
    print("arms:", [a[0] for a in arms], flush=True)

    from model import TAPNet

    # render one sample board per suite for the log
    for suite in args.suites:
        b, dk = build_start_board(suite, 0, args.master_seed)
        print(f"{suite:8s} sample heights "
              f"{[b.column_height(c) for c in range(10)]} "
              f"drought {dk}", flush=True)

    base_arm = arms[0][0]
    for name, wpath in arms:
        arm = results["arms"].setdefault(name, {})
        results["arm_sha"].setdefault(name, hashlib.sha256(
            wpath.read_bytes()).hexdigest()[:16])
        net = TAPNet(output="linear")
        net.load_state_dict(torch.load(str(wpath), map_location="cpu"))
        net.eval()
        for suite in args.suites:
            sdata = arm.setdefault(suite, {})
            for g in range(args.games):
                if str(g) in sdata:
                    continue
                seed = args.master_seed + SUITES.index(suite) * 10_000 + g
                b, dk = build_start_board(suite, g, args.master_seed)
                t0 = time.perf_counter()
                res = play_recovery_game(net, b, dk, seed)
                res["wall_s"] = round(time.perf_counter() - t0, 2)
                sdata[str(g)] = res
            pieces = [sdata[str(g)]["pieces"] for g in range(args.games)]
            sdata["mean"] = round(statistics.fmean(pieces), 1)
            sdata["median"] = statistics.median(pieces)
            sdata["deaths_lt300"] = sum(p < 300 for p in pieces)
            sdata["capped"] = sum(p >= CAP for p in pieces)
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"{name:8s} {suite:8s} mean {sdata['mean']:7.1f} "
                  f"med {sdata['median']:7.1f} deaths "
                  f"{sdata['deaths_lt300']:2d}/{args.games} cap "
                  f"{sdata['capped']:2d}/{args.games} — SAVED", flush=True)

    results["paired_vs_" + base_arm] = {}
    for name, _ in arms[1:]:
        row = results["paired_vs_" + base_arm].setdefault(name, {})
        for suite in args.suites:
            a = [results["arms"][name][suite][str(g)]["pieces"]
                 for g in range(args.games)]
            b = [results["arms"][base_arm][suite][str(g)]["pieces"]
                 for g in range(args.games)]
            dl = [x - y for x, y in zip(a, b)]
            m = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
            row[suite] = {"delta": round(m, 1), "t": round(t, 2),
                          "wins": sum(x > 0 for x in dl),
                          "losses": sum(x < 0 for x in dl)}
        op.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"paired {name} - {base_arm}:", flush=True)
        for suite in args.suites:
            r = results["paired_vs_" + base_arm][name][suite]
            print(f"  {suite:8s} {r['delta']:+8.1f} t {r['t']:+.2f} "
                  f"W/L {r['wins']}/{r['losses']}", flush=True)

    print("\nNOTE: read-only diagnostic — feeds the style program and the "
          "hard-benchmark decision; no gate, no promotion.", flush=True)


if __name__ == "__main__":
    main()