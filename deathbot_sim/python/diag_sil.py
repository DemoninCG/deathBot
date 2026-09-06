"""Experiment 2: policy-gradient-style labels (SIL) on the snapshot9 bank.

Rationale: every prior training method regressed a VALUE function onto
outcome labels; for a strong policy those labels sit below the policy, and
regression drags the value function down (Phase 2 T1/T2/T3; two online
fitd sessions showed the stabilized version is only a stabilizer).

Policy-gradient methods avoid this by moving action PREFERENCES, not values.
The faithful cheap version on frozen data is Self-Imitation Learning (Oh et
al. 2018) shaped to this project:

    A(s, a_chosen) = [gamma * z(s') + (1 - gamma) * z(s)] - z(s)   (z units)

i.e. the Bellman-image advantage of the action actually taken, measured
against the incumbent's own expectation. Positive advantage -> raise that
candidate's softmax probability; negative -> lower it. The value head is
untouched (no regression anywhere), so the value function cannot degrade.

Loss per state (all candidates of the state, from the bank):
    L = -A * log softmax(candidate_scores / T)[a_chosen]
    with A detached. A > 0 weights up the taken action; A < 0 down-weights.

Arms:
    SIL1  T = 1.0 (sharp; effectively push toward argmax when A>0)
    SIL3  T = 3.0
    SIL10 T = 10.0 (soft)

Screened on fresh panel 18000-18011 (snapshot9 reference measured there too).
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet, piece_to_index  # noqa: E402
from diag_phase2 import Bank, bank_inputs  # noqa: E402


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--temps", nargs="+", type=float, default=[1.0, 3.0, 10.0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--adv-scale", type=float, default=1.0,
                    help="multiplier on the (already z-united) advantage")
    ap.add_argument("--adv-clip", type=float, default=20.0,
                    help="clip |A| to keep single-sample gradients sane")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-games", type=int, default=12)
    ap.add_argument("--screen-seed0", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-arms", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    bank = Bank(args.bank)
    n = bank.n_states
    print(f"Bank: {n} states / {int(bank.cand_offsets[-1])} candidates")

    weights_path = resolve_weights(args.weights)
    incumbent = TAPNet(output="linear")
    incumbent.load_state_dict(torch.load(weights_path, map_location=device))
    incumbent.eval()
    print(f"Incumbent: {weights_path.name}")

    # Full candidate tensors (precomputed once).
    from afterstate import flat_rows_to_tensor
    n_cand = int(bank.cand_offsets[-1])
    cand_offsets = torch.tensor(np.asarray(bank.cand_offsets), dtype=torch.long)
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    placed_ids, next_ids = bank._replay_piece_ids()
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
        cand_cur[lo:hi] = piece_to_index(placed_ids[s])
        cand_nxt[lo:hi] = piece_to_index(next_ids[s])

    # Chosen-action inputs.
    boards, cur, nxt = bank_inputs(bank)

    # Incumbent z on ALL candidates (chunked) and on chosen actions.
    def batched_values(net, b, c, x, chunk=4096):
        out = torch.empty(b.shape[0])
        with torch.no_grad():
            for lo in range(0, b.shape[0], chunk):
                hi = min(lo + chunk, b.shape[0])
                out[lo:hi] = net(b[lo:hi], c[lo:hi], x[lo:hi]).flatten()
        return out

    inc_cand = batched_values(incumbent, cand_boards, cand_cur, cand_nxt)
    with torch.no_grad():
        inc_chosen = incumbent(boards, cur, nxt).flatten()

    # Advantages: A = [gamma * z(s') + (1 - gamma) * z(s)] - z(s), z-units.
    # z(s') = incumbent value of the CHOSEN action's successor board (the
    # bank's next state = chosen afterstate of the next step).
    next_state_idx = torch.full((n,), -1, dtype=torch.long)
    pos_index = {}
    for s in range(n):
        pos_index.setdefault(int(bank.game_ids[s]), {})[
            int(bank.step_pos[s])] = s
    for s in range(n):
        nxt_pos = int(bank.step_pos[s]) + 1
        if nxt_pos in pos_index.get(int(bank.game_ids[s]), {}):
            next_state_idx[s] = pos_index[int(bank.game_ids[s])][nxt_pos]
    has_next = next_state_idx >= 0
    z_succ = torch.zeros(n)
    idx_next = next_state_idx.clamp(min=0)
    z_succ[has_next] = inc_chosen[idx_next][has_next]
    bellman = args.gamma * z_succ + (1.0 - args.gamma) * inc_chosen
    adv = (bellman - inc_chosen) * args.adv_scale
    adv = adv.clamp(-args.adv_clip, args.adv_scale * args.adv_clip)
    adv = adv.detach()
    adv[~has_next] = 0.0  # endgame edge: no successor -> no preference signal
    print(f"advantages: mean {adv.mean():+.3f} std {adv.std():.3f} "
          f"pos {int((adv > 0).sum())} neg {int((adv < 0).sum())} "
          f"(coverage {int(has_next.sum())}/{n})")

    # Per-state candidate offsets + chosen index as tensors.
    chosen_idx = torch.tensor(np.asarray(bank.chosen_idx), dtype=torch.long)

    results = {
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "bank": str(Path(args.bank).resolve()),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "gamma": args.gamma, "adv_scale": args.adv_scale,
        "adv_clip": args.adv_clip, "seed": args.seed,
        "adv_stats": {"mean": float(adv.mean()), "std": float(adv.std())},
        "arms": {},
    }

    rng = random.Random(args.seed)
    schedule = [rng.sample(range(n), min(args.batch_size, n))
                for _ in range(args.steps)]

    from evaluate import evaluate_weights_parallel
    import tempfile

    for temp in args.temps:
        arm = f"SIL{temp:g}"
        print(f"\n=== {arm} ===", flush=True)
        net = TAPNet(output="linear")
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        t0 = time.perf_counter()
        losses = []
        for step in range(args.steps):
            idx = schedule[step]
            opt.zero_grad()
            # Accumulate policy-gradient loss over the batch's states.
            total = boards.new_zeros(())
            for s in idx:
                lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
                k = hi - lo
                if k < 2:
                    continue
                cb = cand_boards[lo:hi]
                cc = cand_cur[lo:hi]
                cn = cand_nxt[lo:hi]
                scores = net(cb, cc, cn).flatten()
                logp = F.log_softmax(scores / temp, dim=0)
                a = adv[s]
                if a == 0:
                    continue
                total = total + (-a * logp[int(chosen_idx[s])])
            if total.requires_grad:
                total = total / max(1, len(idx))
                total.backward()
                opt.step()
            losses.append(float(total))
            if (step + 1) % 100 == 0:
                print(f"  step {step + 1}/{args.steps}: "
                      f"loss={statistics.fmean(losses[-100:]):.4f}", flush=True)
        net.eval()
        dt = time.perf_counter() - t0

        from diag_phase2 import ranking_drift
        rank_states = rng.sample(range(n), min(200, n))
        drift = ranking_drift(net, incumbent, bank, rank_states, device=device)

        tmp = Path(tempfile.gettempdir()) / f"_sil_{temp}_{args.seed}.pt"
        torch.save(net.state_dict(), str(tmp))
        ev = evaluate_weights_parallel(
            str(tmp), games=args.screen_games, seed=args.screen_seed0,
            max_pieces=1500, workers=args.workers)
        tmp.unlink(missing_ok=True)
        if args.save_arms:
            sp = Path(args.out).resolve().parent / (
                f"sil_T{temp:g}_seed{args.seed}.pt")
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

    # Snapshot9 reference on the same panel.
    ref = evaluate_weights_parallel(
        str(weights_path), games=args.screen_games, seed=args.screen_seed0,
        max_pieces=1500, workers=args.workers)
    results["ref_snapshot9"] = {
        "mean_pieces": ref.mean_pieces,
        "per_seed": [g.pieces for g in ref.games],
    }
    print(f"\n[snapshot9 on same panel] {ref.mean_pieces:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
