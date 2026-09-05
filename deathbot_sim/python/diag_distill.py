"""Distillation sweep driver: push the snapshot9 self-distillation finding.

Background (2026-09-05): fitting a checkpoint's own outputs stretched to
``scale*f(x)+shift`` for a few hundred Adam steps ("C3 arm") produced a
genuinely stronger policy (snapshot9: ~740 vs ~510 fresh-panel mean). The
leading hypothesis is a born-again-network / self-distillation smoothing
effect: SGD re-projects the value function onto a smoother hypothesis-space
solution without adding environmental information.

This driver runs distillation variants and screens them on FRESH seed panels
(panel discipline: the pilot panel (9001-9008 / seed42-eval) and the Phase-2
validation panels (1000-1029, 42-71) are dev-contaminated; screening uses
2000+, replication 3000+, final validation 4000+ — each used at most once).

Arms (all trained on bank states, paired minibatch schedule):
  D1_iter2          distill snapshot9 again (born-again round 2)
  D2_ensemble567    distill mean output of snapshots 5+6+7 (ensemble teacher)
  D3_scratch        same target as D1 but from a scratch init (mechanism:
                    is the gain the *target* or the *trajectory*?)
  D4_offpolicy      D1 target but trained on train.py-style random boards
                    (mechanism: on-distribution-only vs global smoothing)

Usage:
    python diag_distill.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \
        --arms D1_iter2 D2_ensemble567 --scale 10 --shift 5 --steps 400 \
        --screen-games 12 --screen-seed0 2000 \
        --out ../_diag_out/distill_round2.json
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
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet, piece_to_index  # noqa: E402


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied, HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def load_net(path: Path) -> TAPNet:
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def bank_inputs_on_policy(bank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(boards, cur, nxt) for each state's chosen candidate (bank-frozen)."""
    from afterstate import rows_to_tensor
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


def bank_inputs_off_policy(bank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as on-policy but uses UNCHOSEN candidates too (all candidates of
    every state), giving a much wider board distribution. Boards still come
    from the bank (real champion-visited pre-boards) but the (board,
    afterstate) pairs cover placements the policy rejected."""
    from afterstate import flat_rows_to_tensor
    n_cand = int(bank.cand_offsets[-1])
    rows = bank.cand_rows.reshape(n_cand, 21)
    boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    # Piece identities: replay queues and expand per candidate count.
    placed, nxt = bank._replay_piece_ids()
    cur = torch.empty(n_cand, dtype=torch.long)
    nxt_t = torch.empty(n_cand, dtype=torch.long)
    for s in range(bank.n_states):
        lo, hi = bank.candidate_slice(s)
        cur[lo:hi] = piece_to_index(placed[s])
        nxt_t[lo:hi] = piece_to_index(nxt[s])
    return boards, cur, nxt_t


def compute_targets_chunked(nets: list[TAPNet], boards: torch.Tensor,
                            cur: torch.Tensor, nxt: torch.Tensor,
                            *, chunk: int = 4096) -> torch.Tensor:
    """Mean teacher output over ``nets`` for every input row, computed in
    chunks (a 400k-row single forward pass needs ~20 GB of conv activations
    and OOMs; chunked it needs ~200 MB). Returns [N, 1]."""
    n = boards.shape[0]
    out = torch.empty((n, 1))
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            preds = torch.stack([m(boards[lo:hi], cur[lo:hi], nxt[lo:hi])
                                 for m in nets]).mean(0)
            out[lo:hi] = preds
    return out


def distill(target_net: TAPNet, init_net: TAPNet, boards: torch.Tensor,
            cur: torch.Tensor, nxt: torch.Tensor, *, steps: int, batch_size: int,
            lr: float, scale: float, shift: float, seed: int,
            precomputed_base: torch.Tensor | None = None) -> TAPNet:
    """Train init_net toward scale*f(x)+shift of target_net on fixed inputs."""
    net = init_net
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    n = boards.shape[0]
    if precomputed_base is None:
        precomputed_base = compute_targets_chunked([target_net], boards, cur, nxt)
    targets = precomputed_base * scale + shift
    rng = random.Random(seed)
    net.train()
    for _step in range(steps):
        idx = rng.sample(range(n), min(batch_size, n))
        b, ci, ni, t = boards[idx], cur[idx], nxt[idx], targets[idx]
        opt.zero_grad()
        pred = net(b, ci, ni)
        loss = loss_fn(pred, t)
        loss.backward()
        opt.step()
    net.eval()
    return net


def screen(net: TAPNet, *, games: int, seed0: int, workers: int,
           tag: str) -> dict:
    from evaluate import evaluate_weights_parallel
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"_distill_{tag}_{seed0}.pt"
    torch.save(net.state_dict(), str(tmp))
    ev = evaluate_weights_parallel(str(tmp), games=games, seed=seed0,
                                   max_pieces=1500, workers=workers)
    tmp.unlink(missing_ok=True)
    return {
        "mean_pieces": ev.mean_pieces, "median_pieces": ev.median_pieces,
        "min_pieces": ev.min_pieces, "max_pieces": ev.max_pieces,
        "capped": ev.capped_games, "per_seed": [g.pieces for g in ev.games],
    }


def paired_stats(a: list[int], b: list[int]) -> dict:
    deltas = [x - y for x, y in zip(a, b)]
    n = len(deltas)
    mean_d = statistics.fmean(deltas)
    sd_d = statistics.stdev(deltas) if n > 1 else 0.0
    t = mean_d / (sd_d / n ** 0.5) if sd_d > 0 else float("nan")
    boot_rng = random.Random(0)
    boot = []
    for _ in range(5000):
        boot.append(statistics.fmean(
            deltas[boot_rng.randrange(n)] for _ in range(n)))
    boot.sort()
    return {
        "mean_delta": mean_d, "t": t,
        "ci95": [boot[int(0.025 * 5000)], boot[int(0.975 * 5000)]],
        "wins": sum(d > 0 for d in deltas),
        "losses": sum(d < 0 for d in deltas),
        "ties": sum(d == 0 for d in deltas),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--arms", nargs="+", default=["D1_iter2", "D2_ensemble567"])
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-games", type=int, default=12)
    ap.add_argument("--screen-seed0", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-arms", action="store_true",
                    help="save trained arm weights next to --out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    # Bank import: reuse diag_phase2's Bank loader.
    sys.path.insert(0, str(HERE))
    from diag_phase2 import Bank
    bank = Bank(args.bank)
    print(f"Bank: {bank.n_states} states / {int(bank.cand_offsets[-1])} "
          f"candidates ({bank.meta['weights'].split(chr(92))[-1]})")

    snap9 = resolve_weights("..\\..\\model_weights_snapshot9.pt")
    snap7 = resolve_weights("..\\..\\model_weights_snapshot7.pt")
    w = {"snap9": snap9, "snap7": snap7,
         "snap5": snap7.parent / "model_weights_snapshot5.pt",
         "snap6": snap7.parent / "model_weights_snapshot6.pt"}
    nets = {}
    for k, p in w.items():
        if p.is_file():
            nets[k] = load_net(p)
    if "snap9" not in nets:
        raise FileNotFoundError("snapshot9 required")

    on_boards, on_cur, on_nxt = bank_inputs_on_policy(bank)
    print(f"on-policy inputs: {tuple(on_boards.shape)}")

    # Precompute ensemble targets (mean of available 5/6/7 outputs).
    ens_members = [nets[k] for k in ("snap5", "snap6", "snap7") if k in nets]
    print(f"ensemble members: {[k for k in ('snap5','snap6','snap7') if k in nets]}")

    results = {"args": {k: v for k, v in vars(args).items()},
               "snap9": str(snap9), "arms": {}}

    # Reference screens on the SAME panel (paired).
    ref9 = screen(nets["snap9"], games=args.screen_games,
                  seed0=args.screen_seed0, workers=args.workers, tag="snap9")
    print(f"[snap9 ref] {args.screen_games} games seed0={args.screen_seed0}: "
          f"{ref9['mean_pieces']:.1f}")
    results["ref_snap9"] = ref9
    if "snap7" in nets:
        ref7 = screen(nets["snap7"], games=args.screen_games,
                      seed0=args.screen_seed0, workers=args.workers, tag="snap7")
        print(f"[snap7 ref] {ref7['mean_pieces']:.1f}")
        results["ref_snap7"] = ref7

    for arm in args.arms:
        t0 = time.perf_counter()
        print(f"\n=== {arm} ===", flush=True)
        if arm == "D1_iter2":
            base9 = compute_targets_chunked([nets["snap9"]], on_boards,
                                            on_cur, on_nxt)
            net = distill(nets["snap9"], load_net(snap9), on_boards, on_cur,
                          on_nxt, steps=args.steps, batch_size=args.batch_size,
                          lr=args.lr, scale=args.scale, shift=args.shift,
                          seed=args.seed, precomputed_base=base9)
        elif arm == "D2_ensemble567":
            if len(ens_members) < 2:
                print("  skipped: ensemble members missing")
                continue
            # Custom distill with mean teacher: build targets once (chunked).
            init = load_net(snap9)
            opt = torch.optim.Adam(init.parameters(), lr=args.lr)
            loss_fn = nn.SmoothL1Loss()
            n = on_boards.shape[0]
            base = compute_targets_chunked(ens_members, on_boards, on_cur,
                                           on_nxt).flatten()
            # Keep the [N, 1] column shape so the loss never broadcasts
            # (a [64] target vs [64,1] input silently mis-shapes the loss).
            targets = (base * args.scale + args.shift).unsqueeze(1)
            rng = random.Random(args.seed)
            init.train()
            for _ in range(args.steps):
                idx = rng.sample(range(n), min(args.batch_size, n))
                opt.zero_grad()
                loss = loss_fn(init(on_boards[idx], on_cur[idx], on_nxt[idx]),
                               targets[idx])
                loss.backward()
                opt.step()
            init.eval()
            net = init
        elif arm == "D3_scratch":
            torch.manual_seed(0)
            scratch = TAPNet(output="linear")
            net = distill(nets["snap9"], scratch, on_boards, on_cur, on_nxt,
                          steps=args.steps * 5,  # scratch needs longer
                          batch_size=args.batch_size, lr=args.lr * 3,
                          scale=args.scale, shift=args.shift, seed=args.seed)
        elif arm == "D4_offpolicy":
            off_boards, off_cur, off_nxt = bank_inputs_off_policy(bank)
            print(f"  off-policy inputs: {tuple(off_boards.shape)}")
            net = distill(nets["snap9"], load_net(snap9), off_boards, off_cur,
                          off_nxt, steps=args.steps, batch_size=args.batch_size,
                          lr=args.lr, scale=args.scale, shift=args.shift,
                          seed=args.seed)
        else:
            print(f"  unknown arm {arm}")
            continue
        dt = time.perf_counter() - t0

        res = screen(net, games=args.screen_games, seed0=args.screen_seed0,
                     workers=args.workers, tag=arm)
        res["train_s"] = round(dt, 1)
        res["paired_vs_snap9"] = paired_stats(res["per_seed"],
                                              ref9["per_seed"])
        if args.save_arms:
            save_path = Path(args.out).resolve().parent / (
                f"distill_{arm}_scale{args.scale:g}_steps{args.steps}"
                f"_seed{args.seed}.pt")
            torch.save(net.state_dict(), str(save_path))
            print(f"  saved arm checkpoint to {save_path}")
        results["arms"][arm] = res
        p = res["paired_vs_snap9"]
        print(f"  eval {res['mean_pieces']:.1f} "
              f"(vs snap9 {ref9['mean_pieces']:.1f}) "
              f"delta {p['mean_delta']:+.1f} t={p['t']:.2f} "
              f"CI95 {p['ci95'][0]:+.0f}..{p['ci95'][1]:+.0f} "
              f"W/L {p['wins']}/{p['losses']} ({dt:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
