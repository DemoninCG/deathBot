"""Phase 3 driver: controlled policy improvement from snapshot9 (2026-09 plan).

Phase 2 established, on fixed data:
  * absolute MC survival labels destroy the policy (T1/T2/T3 arms);
  * fitting a consistent transform of the incumbent's own outputs is safe
    and even beneficial (C3 arm -> snapshot9);
  * the mechanism is trajectory smoothing, so any new target that fights the
    incumbent's candidate ORDERING is dangerous.

Phase 3 arms therefore combine value learning with ordering preservation:

  P1_rank           pure teacher-ranking regularizer (no value target):
                    candidate-softmax cross-entropy toward the heuristic's
                    per-state ranking, temperature-scaled. Tests whether
                    ordering-only learning moves behavior at all.
  P2_rank_mc        T3 anchored-MC value target + P1 ranking loss, weight
                    --rank-weight. The plan's "C1 branch": MC improvement
                    with the incumbent's ordering as a guardrail.
  P3_fitd           fitted one-step TD in an incumbent-transformed space:
                    targets = a*Q_hat(s') + b with Q_hat = incumbent's own
                    scores (calibration), i.e. regression toward the Bellman
                    image of the incumbent's function. A conservative,
                    order-preserving bootstrap.
  P4_rank_fitd      P3 + ranking regularizer.

All arms train on the frozen snapshot9 bank with paired minibatch schedules,
detached targets everywhere, and are screened on a FRESH panel (registry:
seeds 6000-6011; validation reserve 7000-7029).

Usage:
    python diag_phase3.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \
        --weights ../../model_weights_snapshot9.pt \
        --arms P1_rank P2_rank_mc P3_fitd P4_rank_fitd \
        --steps 400 --lr 1e-5 --rank-weight 0.5 --rank-temp 50 \
        --screen-games 12 --screen-seed0 6000 \
        --out ../_diag_out/phase3_round1.json
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
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet, piece_to_index  # noqa: E402
from diag_phase2 import Bank, bank_inputs, ranking_drift  # noqa: E402


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def teacher_rank_targets(bank: Bank) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-state (candidate_offsets, teacher_scores) lists for the ranking
    loss. Returns (offsets [S+1] long, scores [C] float)."""
    return (torch.tensor(np.asarray(bank.cand_offsets), dtype=torch.long),
            torch.tensor(np.asarray(bank.teacher_scores), dtype=torch.float32))


def rank_loss(net: TAPNet, bank: Bank, state_indices: list[int],
              boards_all: torch.Tensor, cur_all: torch.Tensor,
              nxt_all: torch.Tensor, teacher_scores: torch.Tensor,
              cand_offsets: torch.Tensor, *, temperature: float,
              device: str = "cpu") -> torch.Tensor:
    """Candidate-softmax CE between the policy's candidate scores and the
    teacher's per-state ranking (Boltzmann over temperature-scaled ranks).

    The teacher's ABSOLUTE scores are never fit — only their softmax
    distribution (with sharpness controlled by temperature), and only the
    ordering matters after temperature. Uses the recorded full candidate
    sets (all candidates, not just the chosen one).
    """
    total = boards_all.new_zeros(())
    n_terms = 0
    for s in state_indices:
        lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
        k = hi - lo
        if k < 2:
            continue
        b = boards_all[lo:hi].to(device)
        ci = cur_all[lo:hi].to(device)
        ni = nxt_all[lo:hi].to(device)
        pred = net(b, ci, ni).flatten()
        t_scores = teacher_scores[lo:hi].to(device)
        # Softmax over teacher scores at the given temperature. Rows are
        # candidates of ONE state, so this is a valid per-state distribution.
        target_probs = F.softmax(t_scores / temperature, dim=0)
        log_probs = F.log_softmax(pred / 1.0, dim=0)
        total = total + -(target_probs * log_probs).sum()
        n_terms += 1
    if n_terms == 0:
        return boards_all.new_zeros(())
    return total / n_terms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arms", nargs="+", default=[
        "P1_rank", "P2_rank_mc", "P3_fitd", "P4_rank_fitd"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank-weight", type=float, default=0.5,
                    help="weight of the ranking loss relative to value loss")
    ap.add_argument("--rank-temp", type=float, default=50.0,
                    help="softmax temperature over ranking scores (higher = "
                         "closer to uniform = gentler ordering signal)")
    ap.add_argument("--rank-source", type=str, default="incumbent",
                    choices=["incumbent", "heuristic"],
                    help="whose candidate rankings to imitate. Phase 3 "
                         "round 1 showed heuristic rankings HURT snapshot9 "
                         "(it already beats the teacher); incumbent rankings "
                         "are the Phase-2-proven safe ordering.")
    ap.add_argument("--rank-batch", type=int, default=8,
                    help="states per ranking-loss draw (each needs a full "
                         "candidate forward pass)")
    ap.add_argument("--gamma", type=float, default=0.999,
                    help="discount for the fitted-TD bootstrap (plan C3)")
    ap.add_argument("--td-scale", type=float, default=10.0)
    ap.add_argument("--td-shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-games", type=int, default=12)
    ap.add_argument("--screen-seed0", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rank-states", type=int, default=200)
    ap.add_argument("--save-arms", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    bank = Bank(args.bank)
    print(f"Bank: {bank.n_states} states / {int(bank.cand_offsets[-1])} "
          f"candidates")

    weights_path = resolve_weights(args.weights)
    incumbent = TAPNet(output="linear")
    incumbent.load_state_dict(torch.load(weights_path, map_location=device))
    incumbent.eval()
    weights_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    print(f"Incumbent: {weights_path.name} ({weights_sha[:12]}…)")

    # Paired schedule over CHOSEN states (value loss) — identical per arm.
    rng = random.Random(args.seed)
    n = bank.n_states
    schedule = [rng.sample(range(n), min(args.batch_size, n))
                for _ in range(args.steps)]
    # Ranking draws: per step, which states contribute ranking gradients.
    rank_schedule = [rng.sample(range(n), min(args.rank_batch, n))
                     for _ in range(args.steps)]
    rank_states = rng.sample(range(n), min(args.rank_states, n))

    # Chosen-action inputs for the value loss.
    boards, cur, nxt = bank_inputs(bank)
    # Full candidate inputs for the ranking loss (precomputed once).
    from afterstate import flat_rows_to_tensor
    n_cand = int(bank.cand_offsets[-1])
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_offsets, teacher_scores = teacher_rank_targets(bank)
    placed_ids, next_ids = bank._replay_piece_ids()
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
        cand_cur[lo:hi] = piece_to_index(placed_ids[s])
        cand_nxt[lo:hi] = piece_to_index(next_ids[s])
    if args.rank_source == "incumbent":
        # Recompute ranking scores from the INCUMBENT on identical candidate
        # sets (Phase 2 lesson: snapshot9 already beats the heuristic, so its
        # own ordering is the one worth preserving).
        with torch.no_grad():
            inc_scores = torch.empty(n_cand)
            chunk = 4096
            for lo in range(0, n_cand, chunk):
                hi = min(lo + chunk, n_cand)
                inc_scores[lo:hi] = incumbent(
                    cand_boards[lo:hi], cand_cur[lo:hi],
                    cand_nxt[lo:hi]).flatten()
        teacher_scores = inc_scores
        print("ranking source: incumbent (snapshot9's own candidate order)")
    else:
        print("ranking source: heuristic teacher")
    print(f"Value inputs {tuple(boards.shape)}, ranking inputs "
          f"{tuple(cand_boards.shape)}")

    with torch.no_grad():
        incumbent_out = incumbent(boards, cur, nxt)

    # ---- Targets ----------------------------------------------------------
    # Fitted one-step TD in the incumbent-transformed space (plan C3/C4):
    #   z-space: target_z(s) = gamma * z(s')  where z = incumbent scores
    #   (terminal handling: the bank records only non-terminal transitions,
    #    so all bootstraps are alive; the +1 placement reward is folded in
    #    as the shift — see below)
    # Raw-space equivalent: Q_raw(s) = 1 + gamma * Q_raw(s') approximated by
    #   the incumbent's own raw scores. We fit z(s) toward z(s')*gamma in
    #   transformed space so the ordering is preserved exactly at the target.
    steps_after = torch.tensor(
        [float(bank.game_lengths[int(g)] - int(p))
         for g, p in zip(bank.game_ids, bank.step_pos)],
        dtype=torch.float32)
    # Successor boards for the bootstrap: the chosen afterstate of the NEXT
    # step in the same game (missing for the final step of each game).
    next_state_idx = torch.full((n,), -1, dtype=torch.long)
    # Map (game, pos) -> state index.
    pos_index = {}
    for s in range(n):
        pos_index.setdefault(int(bank.game_ids[s]), {})[
            int(bank.step_pos[s])] = s
    for s in range(n):
        nxt_pos = int(bank.step_pos[s]) + 1
        game = int(bank.game_ids[s])
        if nxt_pos in pos_index.get(game, {}):
            next_state_idx[s] = pos_index[game][nxt_pos]
    has_next = next_state_idx >= 0
    print(f"Bootstrap coverage: {int(has_next.sum())}/{n} states have a "
          f"recorded successor")

    # z(s') for successors, defaulting to the state's own value when absent.
    with torch.no_grad():
        z_next = torch.empty(n, 1)
        idx_next = next_state_idx.clamp(min=0)
        z_next[has_next] = incumbent(
            boards[idx_next][has_next], cur[idx_next][has_next],
            nxt[idx_next][has_next])
        z_next[~has_next] = incumbent_out[~has_next]
    # Alive-masked bootstrap target: z(s) -> gamma*z(s') + (1-gamma)*z(s)
    # for states with a successor — the incumbent-anchored Bellman image
    # (keeps values on the incumbent's scale; the (1-gamma)*z(s) term is the
    # discounted '1 placement reward' expressed in z units, so the target is
    # a consistent transform, not a shrink-toward-zero). States without a
    # successor keep their own value (no gradient toward anything new).
    fitd_target = torch.where(
        has_next.unsqueeze(1),
        args.gamma * z_next + (1.0 - args.gamma) * incumbent_out.detach(),
        incumbent_out.detach())
    # Anchored MC (for P2): same recipe as Phase 2 T3.
    disc = (1.0 - args.gamma ** steps_after.clamp(min=0)) / (1.0 - args.gamma)
    anchor_mean = float(incumbent_out.mean())
    anchor_std = max(float(incumbent_out.std(unbiased=False)), 1.0)
    norm = (disc - disc.mean()) / disc.std(unbiased=False).clamp(min=1e-6)
    anchor_mc_target = (anchor_mean + anchor_std * norm).unsqueeze(1)

    value_targets: dict[str, torch.Tensor | None] = {
        "P1_rank": None,
        "P2_rank_mc": anchor_mc_target,
        "P3_fitd": fitd_target,
        "P4_rank_fitd": fitd_target,
    }

    results = {
        "weights": str(weights_path), "weights_sha256": weights_sha,
        "bank": str(Path(args.bank).resolve()),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "rank_weight": args.rank_weight, "rank_temp": args.rank_temp,
        "rank_batch": args.rank_batch, "gamma": args.gamma,
        "td_scale": args.td_scale, "td_shift": args.td_shift,
        "seed": args.seed, "bootstrap_coverage": int(has_next.sum()),
        "arms": {},
    }

    from evaluate import evaluate_weights_parallel
    import tempfile

    for arm in args.arms:
        print(f"\n=== {arm} ===", flush=True)
        net = TAPNet(output="linear")
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.train()
        t0 = time.perf_counter()
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        value_target = value_targets[arm]
        use_rank = arm in ("P1_rank", "P2_rank_mc", "P4_rank_fitd")
        for step in range(args.steps):
            v_idx = schedule[step]
            opt.zero_grad()
            value_loss = boards.new_zeros(())
            if value_target is not None:
                b = boards[v_idx].to(device)
                ci = cur[v_idx].to(device)
                ni = nxt[v_idx].to(device)
                t = value_target[v_idx].to(device)
                pred = net(b, ci, ni)
                value_loss = F.smooth_l1_loss(pred, t)
            rank_term = boards.new_zeros(())
            if use_rank:
                r_states = rank_schedule[step]
                rank_term = rank_loss(
                    net, bank, r_states, cand_boards, cand_cur, cand_nxt,
                    teacher_scores, cand_offsets,
                    temperature=args.rank_temp, device=device)
            loss = value_loss + args.rank_weight * rank_term
            loss.backward()
            opt.step()
            if (step + 1) % 100 == 0:
                print(f"  step {step + 1}/{args.steps}: "
                      f"value={float(value_loss):.4f} "
                      f"rank={float(rank_term):.4f}", flush=True)
        net.eval()
        dt = time.perf_counter() - t0

        drift = ranking_drift(net, incumbent, bank, rank_states, device=device)
        tmp = Path(tempfile.gettempdir()) / f"_phase3_{arm}_{args.seed}.pt"
        torch.save(net.state_dict(), str(tmp))
        ev = evaluate_weights_parallel(
            str(tmp), games=args.screen_games, seed=args.screen_seed0,
            max_pieces=1500, workers=args.workers)
        tmp.unlink(missing_ok=True)

        if args.save_arms:
            sp = Path(args.out).resolve().parent / (
                f"phase3_{arm}_seed{args.seed}.pt")
            torch.save(net.state_dict(), str(sp))
            print(f"  saved arm checkpoint to {sp}")

        results["arms"][arm] = {
            "train_s": round(dt, 1),
            "mean_pieces": ev.mean_pieces,
            "median_pieces": ev.median_pieces,
            "capped": ev.capped_games,
            "per_seed": [g.pieces for g in ev.games],
            **drift,
        }
        print(f"  eval {ev.mean_pieces:.1f} | tau {drift['mean_kendall']:.3f} "
              f"| top1 {drift['top1_agreement']:.3f}")

    # References on the same panel.
    ref9 = evaluate_weights_parallel(
        str(weights_path), games=args.screen_games, seed=args.screen_seed0,
        max_pieces=1500, workers=args.workers)
    results["ref_snapshot9"] = {
        "mean_pieces": ref9.mean_pieces,
        "per_seed": [g.pieces for g in ref9.games],
    }
    print(f"\n[snapshot9 on same panel] {ref9.mean_pieces:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
