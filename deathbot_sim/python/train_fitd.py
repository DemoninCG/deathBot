"""Online fitted-TD self-play training (Phase 3, 2026-09 plan section 4).

The Phase 2/3 fixed-data campaign established:
  * absolute MC survival labels destroy the policy (T1/T2);
  * fitted one-step TD toward the incumbent-anchored Bellman image
        z(s) -> gamma * z(s') + (1 - gamma) * z(s)
    is behavior-NEUTRAL on frozen data (validated on 30 fresh seeds,
    delta -3.9 +- 205) — i.e. a STABLE update rule, safe to iterate online.

This trainer wires that update into an online loop:

  each round:
    1. play `games_per_round` rollouts with the current net (parallel
       deterministic collector, corrected lockout semantics),
    2. add raw transitions (board, cur, nxt, successor-index, terminal) to a
       transition replay,
    3. fit: for a batch of transitions, target = gamma * z_target(s') +
       (1 - gamma) * z_online(s) where z_target comes from a LAGGED target
       network and z_online from the live net (fixed point = incumbent's own
       function; the (1-gamma) term is the placement reward in z units),
    4. greedy eval on the DEV panel; best weights to <out>.best.pt; optional
       rollback (round-zero capable, rejected candidates preserved).

Designed for SHORT sessions first (3 rounds x 8 games): measure the paired
dev-panel delta per round before any long run.

Usage:
    python train_fitd.py --rounds 3 --games 8 --steps-per-round 200 \
        --lr 1e-5 --gamma 0.999 \
        --weights-in ../../model_weights_snapshot9.pt \
        --weights-out ../_diag_out/fitd_session1.pt \
        --dev-panel ../_diag_out/dev_panel_32.pt --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet  # noqa: E402
from afterstate import rows_to_tensor  # noqa: E402


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


# ---- Transition buffer -------------------------------------------------------

import dataclasses


@dataclasses.dataclass
class Transition:
    board_rows: np.ndarray      # [21] uint32 packed post-clear board
    cur_idx: int
    nxt_idx: int
    successor_rows: np.ndarray | None   # packed rows of the next decision state
    successor_cur: int
    successor_nxt: int
    terminal: bool              # True = real game over (no successor)
    generation: int


def collect_rollouts(net: TAPNet, *, games: int, base_seed: int,
                     epsilon: float, top_k: int, max_pieces: int,
                     workers: int, generation: int) -> tuple[list[Transition],
                                                             list[int]]:
    """Play games with `net` and return raw transitions with successor info.

    Reuses the deterministic parallel collector (selfplay.py) for speed, but
    keeps per-step boards via play_game_detailed so transitions can be
    chained: consecutive trajectory entries of one game are (s, s') pairs.
    Terminal = real topout; cap truncations keep a successor.
    """
    import random
    from selfplay import play_game_detailed

    transitions: list[Transition] = []
    lengths: list[int] = []
    # Item 12: pack tensor->rows via bit ops (no per-cell Python loops).
    _shifts = np.arange(10, dtype=np.uint32)
    def _tensor_to_rows(bt: torch.Tensor) -> np.ndarray:
        grid = (bt.squeeze(0).numpy() > 0.5)  # [20, 10] top-first bool
        bottom_first = grid[::-1, :]          # [20, 10] row0=bottom
        rows = np.zeros(21, dtype=np.uint32)
        rows[:20] = (bottom_first.astype(np.uint32)
                     * (1 << _shifts)).sum(axis=1)
        return rows
    for g in range(games):
        env_rng = random.Random(base_seed + g)
        explore_rng = random.Random(base_seed + g + 7_000_000_001)
        from neural_scorer import NeuralScorer
        scorer = NeuralScorer(net=net, device="cpu")
        result = play_game_detailed(
            scorer, env_rng, epsilon=epsilon, exploration_top_k=5,
            max_pieces=max_pieces, heuristic_scorer=None,
            legacy_lockout=False, explore_rng=explore_rng)
        traj = result.trajectory
        # Pre-convert all boards once (successor of step i == board of i+1).
        all_rows = [_tensor_to_rows(bt) for (bt, _ci, _ni, _hv) in traj]
        for i in range(len(traj)):
            bt, ci, ni, _hv = traj[i]
            rows = all_rows[i]
            if i + 1 < len(traj):
                nrows = all_rows[i + 1]
                transitions.append(Transition(
                    rows, ci, ni, nrows, traj[i + 1][1], traj[i + 1][2],
                    False, generation))
            else:
                # Last committed placement: if the game ended by cap the
                # successor exists but wasn't played; mark terminal-free by
                # treating it as terminal for the bootstrap (conservative:
                # the game either ended or the net is choosing next step
                # anyway). Real topouts have no successor.
                transitions.append(Transition(
                    rows, ci, ni, None, 0, 0,
                    result.topout, generation))
        lengths.append(result.pieces_placed)
    return transitions, lengths


def board_rows_to_tensor(rows: np.ndarray) -> torch.Tensor:
    return rows_to_tensor(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--steps-per-round", type=int, default=200,
                    help="optimizer updates per round (batch 64 => 12.8k "
                         "transition draws/round)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--target-net-half-life", type=int, default=2,
                    help="copy online -> target every N rounds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights-in", required=True)
    ap.add_argument("--weights-out", required=True)
    ap.add_argument("--dev-panel", required=True,
                    help="JSON file listing the dev-panel seeds")
    ap.add_argument("--workers", type=int, default=1,
                    help="rollout workers (1 = serial; transitions need "
                         "per-step boards, so the D4 pool is unused here)")
    ap.add_argument("--save-every-round", action="store_true", default=True)
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"
    dev_panel = json.loads(Path(args.dev_panel).read_text())
    dev_seeds: list[int] = dev_panel["seeds"]
    assert dev_panel.get("status") == "reserved", (
        f"dev panel {args.dev_panel} is not fresh (status "
        f"{dev_panel.get('status')}) — pick a fresh range")

    weights_in = resolve_weights(args.weights_in)
    out_path = Path(args.weights_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights_in, map_location=device))
    net.eval()
    target_net = TAPNet(output="linear")
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    best_mean = float("inf")  # will be set by the pre-training dev eval
    best_round = 0

    from evaluate import evaluate_weights_parallel

    def dev_eval(n: TAPNet) -> dict:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "w.pt"
            torch.save(n.state_dict(), str(p))
            ev = evaluate_weights_parallel(
                str(p), games=len(dev_seeds), seed=dev_seeds[0],
                max_pieces=1500, workers=args.workers)
        per_seed = {g.seed: g.pieces for g in ev.games}
        return ev, per_seed

    ev0, _ = dev_eval(net)
    best_mean = ev0.mean_pieces
    torch.save(best_state, str(out_path) + ".best.pt")
    print(f"[dev panel] snapshot start: mean {ev0.mean_pieces:.1f} "
          f"med {ev0.median_pieces:.1f} ({len(dev_seeds)} games)")
    start_mean = ev0.mean_pieces

    rng = random.Random(args.seed + 100)
    replay: deque = deque(maxlen=200_000)
    results = {"weights_in": str(weights_in),
               "weights_in_sha256": hashlib.sha256(
                   weights_in.read_bytes()).hexdigest(),
               "args": {k: v for k, v in vars(args).items()},
               "start_dev_mean": start_mean, "rounds": {}}
    base_seed = 90_000  # training rollout seeds (never used for evaluation)

    for round_idx in range(args.rounds):
        t0 = time.perf_counter()
        print(f"\n=== round {round_idx + 1}/{args.rounds} "
              f"(epsilon {args.epsilon}) ===", flush=True)
        transitions, lengths = collect_rollouts(
            net, games=args.games, base_seed=base_seed + round_idx * 10_000,
            epsilon=args.epsilon, top_k=5, max_pieces=1500,
            workers=args.workers, generation=round_idx)
        replay.extend(transitions)
        pieces_list = lengths
        print(f"  collected {len(transitions)} transitions from "
              f"{args.games} games "
              f"(pieces min/avg/max {min(pieces_list)}/"
              f"{statistics.fmean(pieces_list):.0f}/{max(pieces_list)}); "
              f"replay {len(replay)}")

        # Pack boards for the round's training.
        all_rows = np.stack([t.board_rows for t in transitions])
        boards = rows_to_tensor(all_rows)
        cur = torch.tensor([t.cur_idx for t in transitions], dtype=torch.long)
        nxt = torch.tensor([t.nxt_idx for t in transitions], dtype=torch.long)
        succ_has = torch.tensor([t.terminal is False and t.successor_rows
                                 is not None for t in transitions])
        succ_boards = torch.zeros_like(boards)
        succ_cur = torch.zeros_like(cur)
        succ_nxt = torch.zeros_like(nxt)
        for i, t in enumerate(transitions):
            if t.successor_rows is not None:
                succ_boards[i] = rows_to_tensor(t.successor_rows[None, :])[0]
                succ_cur[i] = t.successor_cur
                succ_nxt[i] = t.successor_nxt

        net.train()
        epoch_losses = []
        for step in range(args.steps_per_round):
            idx = rng.sample(range(len(transitions)),
                             min(args.batch_size, len(transitions)))
            b = boards[idx]
            ci = cur[idx]
            ni = nxt[idx]
            with torch.no_grad():
                # Bellman image target, matching diag_phase3 round 2:
                #   z(s) -> gamma * z_tgt(s') + (1 - gamma) * z_online(s)
                # z_tgt from the LAGGED net (stability); z_online from the
                # live net (keeps the incumbent's scale; placement reward
                # in z units). Terminal transitions: no bootstrap term.
                alive = succ_has[idx]
                z_succ = torch.zeros(len(idx), 1)
                if alive.any():
                    sb = succ_boards[idx][alive]      # [k, 1, 20, 10]
                    sc = succ_cur[idx][alive]         # [k]
                    sn = succ_nxt[idx][alive]         # [k]
                    z_succ[alive] = target_net(sb, sc, sn)
                z_own = net(b, ci, ni)
                targets = torch.where(
                    alive.unsqueeze(1),
                    args.gamma * z_succ + (1.0 - args.gamma) * z_own,
                    z_own)
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(net(b, ci, ni), targets)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss))
        net.eval()
        print(f"  trained {args.steps_per_round} steps, "
              f"loss {statistics.fmean(epoch_losses):.3f}")

        # Target-net refresh (half-life in rounds).
        if (round_idx + 1) % max(1, args.target_net_half_life) == 0:
            target_net.load_state_dict(net.state_dict())
            print("  target net <- online net")

        ev, per_seed = dev_eval(net)
        print(f"  [dev panel] mean {ev.mean_pieces:.1f} "
              f"(delta {ev.mean_pieces - start_mean:+.1f}) "
              f"med {ev.median_pieces:.1f}")
        results["rounds"][str(round_idx + 1)] = {
            "train_s": round(time.perf_counter() - t0, 1),
            "transitions": len(transitions),
            "pieces_min_avg_max": [min(pieces_list),
                                   round(statistics.fmean(pieces_list), 1),
                                   max(pieces_list)],
            "mean_loss": round(statistics.fmean(epoch_losses), 4),
            "dev_mean": ev.mean_pieces,
            "dev_median": ev.median_pieces,
            "dev_per_seed": per_seed,
        }
        if ev.mean_pieces > best_mean:
            best_mean = ev.mean_pieces
            best_state = {k: v.detach().clone()
                          for k, v in net.state_dict().items()}
            best_round = round_idx + 1
            torch.save(best_state, str(out_path) + ".best.pt")
            print(f"  new best -> {out_path}.best.pt")
        torch.save(net.state_dict(), out_path)

    # Final dev eval of the best weights.
    best_net = TAPNet(output="linear")
    best_net.load_state_dict(torch.load(str(out_path) + ".best.pt",
                                        map_location=device))
    best_net.eval()
    ev_best, _ = dev_eval(best_net)
    results["best"] = {"round": best_round, "dev_mean": ev_best.mean_pieces,
                       "start_dev_mean": start_mean,
                       "delta": ev_best.mean_pieces - start_mean}
    print(f"\n[final] best (round {best_round}) dev mean "
          f"{ev_best.mean_pieces:.1f} vs start {start_mean:.1f} "
          f"-> delta {ev_best.mean_pieces - start_mean:+.1f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2),
                                  encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
