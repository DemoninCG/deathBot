"""Phase 2 driver: fixed-data update-damage controls (2026-09 training plan).

Everything runs on a FROZEN dataset (the reference state bank built by
``bank_builder.py``) with identical minibatch schedules, so any behavioral
difference between arms is caused by the target/loss/update change under
test — not by distribution drift from fresh self-play.

Controls (in order of diagnostic value):

  C0_noop          load + save + re-evaluate (catches checkpoint/eval bugs)
  C1_affine        analytically rescale the final layer positively — must NOT
                   change any greedy argmax (scale != policy)
  C2_selfdistill   train toward the incumbent's own outputs (zero residual);
                   the pipeline should be a no-op for behavior
  C3_affine_fit    train toward the affine-transformed outputs (tests whether
                   optimization itself distorts rankings)
  T1_raw_mc        raw blended MC targets on the bank's chosen actions
  T2_norm_mc       z-normalized MC targets (the snapshot8-style squash)
  T3_anchor_mc     anchored MC targets (the current code's scheme)
  L1_mse_raw_mc    raw MC targets with MSE instead of SmoothL1 (loss axis)

Every arm:
  * starts from an independent copy of the incumbent weights,
  * fits ONLY on the bank's recorded states/actions (no fresh rollouts),
  * uses identical minibatch indices per step (paired across arms),
  * reports candidate-ranking drift on a held-out slice of the bank,
  * is evaluated greedily on a matched seed panel before/after.

Usage:
    python diag_phase2.py --bank ../_diag_out/state_bank_snapshot7_32games.npz \
        --weights ../model_weights_snapshot7.pt --arms C0_noop C1_affine C2_selfdistill \
        --steps 100 --out ../_diag_out/phase2_pilot.json
"""

from __future__ import annotations

import argparse
import copy
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet  # noqa: E402


# ---- Bank loading -----------------------------------------------------------

class Bank:
    """In-memory view of a state bank produced by bank_builder.py."""

    def __init__(self, npz_path: str | Path) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self.pre_rows: np.ndarray = data["pre_rows"]          # [S, 21] uint32
        self.cand_offsets: np.ndarray = data["cand_offsets"]  # [S+1] int64
        self.cand_state: np.ndarray = data["cand_state"]
        self.cand_x: np.ndarray = data["cand_x"]
        self.cand_y: np.ndarray = data["cand_y"]
        self.cand_lockout: np.ndarray = data["cand_lockout"]
        self.cand_rows: np.ndarray = data["cand_rows"]        # [C*21] uint32
        self.cand_lines: np.ndarray = data["cand_lines"]
        self.teacher_scores: np.ndarray = data["teacher_scores"]
        self.chosen_idx: np.ndarray = data["chosen_idx"]
        self.step_pos: np.ndarray = data["step_pos"]
        self.game_ids: np.ndarray = data["game_ids"]
        self.game_lengths: np.ndarray = data["game_lengths"]
        self.meta = json.loads(str(data["meta_json"]))

    @property
    def n_states(self) -> int:
        return int(self.pre_rows.shape[0])

    def candidate_slice(self, state: int) -> tuple[int, int]:
        """[start, end) index range of candidates for ``state``."""
        return int(self.cand_offsets[state]), int(self.cand_offsets[state + 1])

    def _replay_piece_ids(self) -> tuple[list, list]:
        """Replay each game's piece queue to recover (placed, preview) piece
        identities for every recorded step (deterministic; same seeds)."""
        from piece_history import PieceQueue
        placed: list = []
        nxt: list = []
        per_game_steps: dict[int, list[int]] = {}
        for s, g in enumerate(self.game_ids):
            per_game_steps.setdefault(int(g), []).append(s)
        for g, steps in per_game_steps.items():
            rng = random.Random(self.meta["seed0"] + g)
            pq = PieceQueue(rng)
            pos = 0
            steps_sorted = sorted(steps, key=lambda s: self.step_pos[s])
            max_pos = max(self.step_pos[s] for s in steps_sorted) + 1
            while pos < max_pos:
                if steps_sorted and self.step_pos[steps_sorted[0]] == pos:
                    s = steps_sorted.pop(0)
                    placed.append(pq.current)
                    nxt.append(pq.next_piece)
                pq.advance()
                pos += 1
        return placed, nxt


def bank_inputs(bank: Bank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(boards, cur_idx, next_idx) for the CHOSEN candidate of every state."""
    from afterstate import rows_to_tensor
    from model import piece_to_index

    placed, nxt = bank._replay_piece_ids()
    n = bank.n_states
    boards = torch.empty((n, 1, 20, 10))
    cur = torch.empty(n, dtype=torch.long)
    nxt_t = torch.empty(n, dtype=torch.long)
    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        rows = bank.cand_rows[lo * 21:hi * 21].reshape(hi - lo, 21)
        boards[s] = rows_to_tensor(rows[int(bank.chosen_idx[s])])
        cur[s] = piece_to_index(placed[s])
        nxt_t[s] = piece_to_index(nxt[s])
    return boards, cur, nxt_t


# ---- Ranking diagnostics ----------------------------------------------------

def _rankdata(scores: torch.Tensor) -> torch.Tensor:
    """Average ranks (0-based, ties averaged) for one candidate vector."""
    n = scores.numel()
    order = torch.argsort(scores)
    ranks = torch.empty(n)
    ranks[order] = torch.arange(n, dtype=torch.float32)
    # Average tied ranks.
    vals, counts = torch.unique(scores, return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        if c > 1:
            mask = scores == v
            ranks[mask] = ranks[mask].mean()
    return ranks


def kendall_tau(a: torch.Tensor, b: torch.Tensor) -> float:
    """Tie-aware Kendall tau between two candidate-score vectors.

    Concordant/discordant pairs use the SIGN of rank differences; using raw
    products (an earlier bug) is not bounded to [-1, 1].
    """
    ra, rb = _rankdata(a), _rankdata(b)
    n = a.numel()
    if n < 2:
        return float("nan")
    conc = disc = tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = (ra[i] - ra[j]).item()
            sb = (rb[i] - rb[j]).item()
            if sa == 0 or sb == 0:
                tied += 1  # tie in one ranking: neither concordant nor discordant
            elif (sa > 0) == (sb > 0):
                conc += 1
            else:
                disc += 1
    denom = n * (n - 1) / 2 - tied
    if denom <= 0:
        return float("nan")
    return (conc - disc) / denom


def ranking_drift(net, ref_net, bank: Bank, state_indices: list[int],
                  device: str = "cpu") -> dict:
    """Compare candidate rankings of ``net`` vs ``ref_net`` on identical
    candidate sets. Tie-aware tau averaged over states; top-1 agreement;
    mean margin change of the incumbent's top pick."""
    from afterstate import flat_rows_to_tensor
    from model import piece_to_index
    taus: list[float] = []
    agree = 0
    compared = 0
    margin_deltas: list[float] = []
    for s in state_indices:
        lo, hi = bank.candidate_slice(s)
        rows = bank.cand_rows[lo * 21:hi * 21].reshape(hi - lo, 21)
        boards = flat_rows_to_tensor(rows.reshape(-1), hi - lo).to(device)
        placed, nxt = bank._replay_piece_ids()
        cur = torch.full((hi - lo,), piece_to_index(placed[s]), dtype=torch.long,
                         device=device)
        nxt_t = torch.full((hi - lo,), piece_to_index(nxt[s]), dtype=torch.long,
                           device=device)
        with torch.no_grad():
            new_scores = net(boards, cur, nxt_t).flatten().cpu()
            ref_scores = ref_net(boards, cur, nxt_t).flatten().cpu()
        if new_scores.numel() < 2:
            continue
        tau = kendall_tau(new_scores, ref_scores)
        if tau == tau:  # not NaN
            taus.append(tau)
        if int(new_scores.argmax()) == int(ref_scores.argmax()):
            agree += 1
        compared += 1
        pick = int(ref_scores.argmax())
        if new_scores.numel() > 1:
            sorted_new, _ = torch.sort(new_scores, descending=True)
            margin_new = float(sorted_new[0] - sorted_new[1])
            sorted_ref, _ = torch.sort(ref_scores, descending=True)
            margin_ref = float(sorted_ref[0] - sorted_ref[1])
            margin_deltas.append(margin_new - margin_ref)
    return {
        "states_compared": compared,
        "mean_kendall": (statistics.fmean(taus) if taus else float("nan")),
        "top1_agreement": (agree / compared if compared else float("nan")),
        "mean_top_margin_delta": (statistics.fmean(margin_deltas)
                                  if margin_deltas else float("nan")),
    }


# ---- Arms -------------------------------------------------------------------

def apply_affine(net: TAPNet, scale: float, shift: float) -> None:
    """Exactly y -> scale*y + shift via the final layer (no optimization)."""
    with torch.no_grad():
        net.fc2.weight.mul_(scale)
        net.fc2.bias.add_(shift)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arms", nargs="+", default=[
        "C0_noop", "C1_affine", "C2_selfdistill", "T1_raw_mc", "T2_norm_mc",
        "T3_anchor_mc"])
    ap.add_argument("--steps", type=int, default=400,
                    help="optimizer updates per training arm")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="pilot lr (plan: 1e-5 conservative; 1e-4 separate)")
    ap.add_argument("--loss", type=str, default="smoothl1",
                    choices=["smoothl1", "mse"],
                    help="loss for training arms (MSE arm is separate)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=9001)
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--rank-states", type=int, default=150,
                    help="held-out states for ranking drift")
    ap.add_argument("--save-arms", action="store_true",
                    help="save each arm's trained weights next to --out "
                         "(phase2_arm_<arm>_seed<seed>.pt) for panel validation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    bank = Bank(args.bank)
    print(f"Bank: {bank.n_states} states, "
          f"{int(bank.cand_offsets[-1])} candidates "
          f"(games={bank.meta['games']}, seed0={bank.meta['seed0']})")

    weights_path = Path(args.weights).expanduser()
    if not weights_path.is_absolute():
        # Resolve like the other tools: cwd, then crate root.
        for cand in (weights_path, HERE.parent / weights_path,
                     HERE.parent.parent / weights_path):
            if cand.is_file():
                weights_path = cand.resolve()
                break
        else:
            raise FileNotFoundError(f"weights not found: {args.weights}")
    weights_path = weights_path.resolve()
    incumbent = TAPNet(output="linear")
    incumbent.load_state_dict(torch.load(weights_path, map_location=device))
    incumbent.eval()
    weights_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    # Fixed paired minibatch schedule: identical for every training arm.
    rng = random.Random(args.seed)
    n = bank.n_states
    schedule: list[list[int]] = []
    for _ in range(args.steps):
        schedule.append(rng.sample(range(n), min(args.batch_size, n)))
    # Held-out ranking slice (disjoint from nothing — ranking uses all
    # candidates, training uses chosen actions; overlap is fine and wanted).
    rank_states = rng.sample(range(n), min(args.rank_states, n))

    boards, cur, nxt = bank_inputs(bank)
    print(f"Bank inputs: boards {tuple(boards.shape)}, "
          f"placed ids recovered")

    # Teacher (heuristic) target per chosen state, from recorded scores.
    teacher_target = torch.tensor(
        [bank.teacher_scores[bank.candidate_slice(s)[0] + int(bank.chosen_idx[s])]
         for s in range(n)], dtype=torch.float32).unsqueeze(1)
    # Incumbent's own outputs (self-distill target + affine source).
    with torch.no_grad():
        incumbent_out = incumbent(boards, cur, nxt)
    affine_scale = 10.0
    affine_shift = 5.0
    affine_target = incumbent_out * affine_scale + affine_shift

    # Survival MC targets per chosen state: the bank records whole-game
    # length; remaining-survival labels need the per-step tape, so the bank's
    # step_pos + game_lengths give it exactly:
    #   steps_after(step) = length - pos
    steps_after = torch.tensor(
        [float(bank.game_lengths[int(g)] - int(p))
         for g, p in zip(bank.game_ids, bank.step_pos)],
        dtype=torch.float32).unsqueeze(1)
    gamma = 0.99
    disc = (1.0 - gamma ** steps_after.clamp(min=0)) / (1.0 - gamma)
    # z-normalized survival (T2): the snapshot8 squash recipe.
    mu, sd = disc.mean(), disc.std(unbiased=False).clamp(min=1e-6)
    norm_target = (disc - mu) / sd
    # Anchored target (T3): onto the incumbent's output scale.
    anchor_mean = float(incumbent_out.mean())
    anchor_std = max(float(incumbent_out.std(unbiased=False)), 1.0)
    anchor_target = anchor_mean + anchor_std * norm_target

    targets_by_arm: dict[str, torch.Tensor] = {
        "C2_selfdistill": incumbent_out,
        "C3_affine_fit": affine_target,
        "T1_raw_mc": disc,
        "T2_norm_mc": norm_target,
        "T3_anchor_mc": anchor_target,
        "L1_mse_raw_mc": disc,
    }

    loss_fn_cls = nn.SmoothL1Loss if args.loss == "smoothl1" else nn.MSELoss

    def make_net() -> TAPNet:
        net = TAPNet(output="linear")
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.eval()
        return net

    results: dict = {
        "weights": str(weights_path), "weights_sha256": weights_sha,
        "bank": str(Path(args.bank).resolve()), "steps": args.steps,
        "batch_size": args.batch_size, "lr": args.lr, "loss": args.loss,
        "seed": args.seed, "anchor_mean": anchor_mean,
        "anchor_std": anchor_std,
        "arms": {},
    }

    for arm in args.arms:
        print(f"\n=== arm {arm} ===", flush=True)
        net = make_net()
        t0 = time.perf_counter()
        if arm == "C0_noop":
            pass  # load + (save/eval) only
        elif arm == "C1_affine":
            apply_affine(net, affine_scale, affine_shift)
        else:
            target = targets_by_arm[arm]
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            loss_fn = (nn.MSELoss() if arm.startswith("L1_")
                       else loss_fn_cls())
            net.train()
            for step, idx in enumerate(schedule):
                b = boards[idx].to(device)
                ci = cur[idx].to(device)
                ni = nxt[idx].to(device)
                t = target[idx].to(device)
                opt.zero_grad()
                pred = net(b, ci, ni)
                loss = loss_fn(pred, t)
                loss.backward()
                opt.step()
                if (step + 1) % 100 == 0:
                    print(f"  step {step + 1}/{args.steps}: loss={loss.item():.4f}",
                          flush=True)
            net.eval()
        dt = time.perf_counter() - t0

        drift = ranking_drift(net, incumbent, bank, rank_states, device=device)
        # Greedy behavior on a matched panel (uses current corrected
        # semantics; the incumbent reference is evaluated once below).
        from evaluate import evaluate_weights_parallel
        import tempfile, os
        tmp_pt = Path(tempfile.gettempdir()) / f"_phase2_{arm}_{args.seed}.pt"
        torch.save(net.state_dict(), str(tmp_pt))
        ev = evaluate_weights_parallel(
            str(tmp_pt), games=args.eval_games, seed=args.eval_seed,
            max_pieces=1500, workers=args.eval_workers)
        tmp_pt.unlink(missing_ok=True)

        if args.save_arms:
            save_path = Path(args.out).resolve().parent / (
                f"phase2_arm_{arm}_seed{args.seed}.pt")
            torch.save(net.state_dict(), str(save_path))
            print(f"  saved arm checkpoint to {save_path}")

        results["arms"][arm] = {
            "train_s": round(dt, 1),
            "mean_pieces": ev.mean_pieces,
            "median_pieces": ev.median_pieces,
            "min_pieces": ev.min_pieces,
            "max_pieces": ev.max_pieces,
            "per_seed": [g.pieces for g in ev.games],
            **drift,
        }
        print(f"  eval mean {ev.mean_pieces:.1f} | "
              f"kendall {drift['mean_kendall']:.4f} | "
              f"top1 agree {drift['top1_agreement']:.3f} | "
              f"margin delta {drift['mean_top_margin_delta']:.2f}")

    # Incumbent reference on the same panel.
    from evaluate import evaluate_weights_parallel
    ref_ev = evaluate_weights_parallel(
        str(weights_path), games=args.eval_games, seed=args.eval_seed,
        max_pieces=1500, workers=args.eval_workers)
    results["incumbent"] = {
        "mean_pieces": ref_ev.mean_pieces,
        "per_seed": [g.pieces for g in ref_ev.games],
    }
    print(f"\nIncumbent on same panel: {ref_ev.mean_pieces:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
