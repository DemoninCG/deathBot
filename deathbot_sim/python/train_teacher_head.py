"""Train the decoupled style head on the λ=20 teacher bank (§15.1.5 DAgger-0).

Round 0 is OFFLINE (no student loop yet): pairs come from the teacher's OWN
trajectories (collect_teacher_bank.py), teacher pick vs the champion 1-ply
pick on the same states. This is the correct-distribution version of the
Phase-B recipe: same margin-ranking + survival-weighted + standardized
features + veto chassis, but the teacher is well-building (14 Tetrises in
4683 states) instead of flat.

Screens at cap 300 on FRESH seeds (default 20450+, never used before):
  1-ply + alpha*style with veto — the fast student. The teacher itself
  needs full 2-ply; the student test is how much style transfers to 1-ply.

Usage:
    python train_teacher_head.py --bank ../_diag_out/teacher_bank_lam20.npz \\
        --weights ../../model_weights_snapshot9.pt --steps 400 --seed 42 \\
        --screen-games 30 --screen-seed0 20450 \\
        --out ../_diag_out/teacher_head.json
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
from style_head import (StyleHead, style_features_from_rows,  # noqa: E402
                        snapshot9_frozen_copy, fit_feature_stats,
                        apply_stats)


def _decode_bank(path: Path):
    data = np.load(path, allow_pickle=False)
    pre = data["pre_rows"]
    blob = data["cand_blob"]
    picks = data["picks"]
    meta = json.loads(str(data["meta_json"]))
    # cand_blob rows: [ncands, lines..., rows_flat...] per state
    states = []
    off = 0
    for _ in range(len(pre)):
        n = int(blob[off])
        lines = [int(x) for x in blob[off + 1:off + 1 + n]]
        rows = blob[off + 1 + n:off + 1 + n + n * 21].reshape(n, 21)
        off += 1 + n + n * 21
        states.append((rows, lines))
    return pre, states, picks, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0, 2.0, 4.0])
    ap.add_argument("--veto-margin", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-games", type=int, default=30)
    ap.add_argument("--screen-seed0", type=int, default=20450)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-head", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    pre, states, picks, meta = _decode_bank(Path(args.bank))
    print(f"Teacher bank: {len(pre)} states, lam={meta.get('lam')}, "
          f"clears={meta.get('clears')}")
    surv = snapshot9_frozen_copy()
    print("Survival net frozen: snapshot9")

    # 1-ply champion pick per state (student side of the pair).
    from afterstate import flat_rows_to_tensor
    pairs, margins = [], []
    t0 = time.perf_counter()
    for s, ((rows, lines), pick_t) in enumerate(zip(states, picks)):
        k = rows.shape[0]
        boards = flat_rows_to_tensor(rows.reshape(-1), k)
        # Piece ids unknown in teacher bank — recover: score with cur=nxt=0
        # is WRONG. Instead use survival values via the same path the bank
        # was built with is unavailable; fall back to comparing FEATURES of
        # teacher pick vs the lowest-Phi candidate as the "flat" side.
        # (Piece-id-free: Phi needs no net; survival margin via heuristic.)
        phis = [float(__import__("well_potential").well_potential(rows[j]))
                for j in range(k)]
        flat_j = int(np.argmin(phis))
        if int(pick_t) == flat_j:
            continue
        pairs.append((style_features_from_rows(rows[int(pick_t)], lines[int(pick_t)]),
                      style_features_from_rows(rows[flat_j], lines[flat_j])))
        margins.append(0.0)  # teacher-bank pairs: no survival margin available
        if (s + 1) % 1000 == 0:
            print(f"  paired {s + 1}/{len(pre)} (pairs {len(pairs)})", flush=True)
    print(f"pairs: {len(pairs)}/{len(pre)} in {time.perf_counter()-t0:.0f}s")
    if not pairs:
        raise SystemExit("no pairs — nothing to train")

    f2 = torch.tensor(np.stack([p[0] for p in pairs]))
    f1 = torch.tensor(np.stack([p[1] for p in pairs]))
    names = ["wheight", "holes", "holedepth", "cav", "wellcost", "spawn",
             "lines", "deepwell"]
    d = (f2 - f1).mean(0)
    print("mean(featTeacher - featFlat): " +
          ", ".join(f"{k}={v:+.3f}" for k, v in zip(names, d.tolist())))
    fmean, fstd = fit_feature_stats(torch.cat([f2, f1]))
    z2, z1 = apply_stats(f2, fmean, fstd), apply_stats(f1, fmean, fstd)

    head = StyleHead()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = [rng.sample(range(len(pairs)), min(args.batch_size, len(pairs)))
             for _ in range(args.steps)]
    losses = []
    head.train()
    for step in range(args.steps):
        idx = sched[step]
        opt.zero_grad()
        s2 = head(z2[idx]).flatten()
        s1 = head(z1[idx]).flatten()
        loss = F.softplus(-(s2 - s1) + args.margin).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    head.eval()
    print(f"trained {args.steps} steps, final loss "
          f"{statistics.fmean(losses[-50:]):.4f}")

    # Screen: reuse diag_style_head's worker via import.
    import diag_style_head as _dsh
    head_state = {k: v.detach().cpu().clone()
                  for k, v in head.state_dict().items()}
    feat_stats = {"mean": fmean.tolist(), "std": fstd.tolist()}
    wp = str(Path(args.weights).expanduser().resolve())
    if not Path(wp).is_file():
        for cand in (HERE.parent / args.weights, HERE.parent.parent / args.weights):
            if cand.is_file():
                wp = str(cand.resolve())
                break
    from concurrent.futures import ProcessPoolExecutor
    screens = {}
    for a in args.alphas:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_dsh._style_worker,
                                (wp, head_state, feat_stats,
                                 args.screen_seed0 + i, args.max_pieces,
                                 (a, args.veto_margin)))
                    for i in range(args.screen_games)]
            rows_a = [f.result() for f in futs]
        pieces_a = [r[0] for r in rows_a]
        lines_a = [r[1] for r in rows_a]
        tet_a = sum(int(r[2].get(4, 0)) for r in rows_a)
        scoring_a = sum(int(r[2].get(k, 0)) for r in rows_a for k in (1, 2, 3, 4))
        screens[str(a)] = {
            "mean_pieces": statistics.fmean(pieces_a),
            "mean_lines": statistics.fmean(lines_a),
            "tetrises": tet_a,
            "tetris_share": (tet_a / scoring_a) if scoring_a else 0.0,
            "capped": sum(1 for p in pieces_a if p >= args.max_pieces),
        }
        print(f"  alpha={a}: {statistics.fmean(pieces_a):.1f} pcs / "
              f"{statistics.fmean(lines_a):.1f} lines, "
              f"cap {screens[str(a)]['capped']}/{args.screen_games}, "
              f"tetris {tet_a}", flush=True)
    res = {"bank": str(args.bank), "teacher_lam": meta.get("lam"),
           "pairs": len(pairs), "steps": args.steps, "lr": args.lr,
           "feat_direction": {k: round(float(v), 4)
                              for k, v in zip(names, d.tolist())},
           "final_loss": round(statistics.fmean(losses[-50:]), 4),
           "screens": screens,
           "screen_seed0": args.screen_seed0, "games": args.screen_games}
    if args.save_head:
        sp = Path(args.out).resolve().parent / f"teacher_head_seed{args.seed}.pt"
        torch.save({"state_dict": head.state_dict(),
                    "feat_mean": fmean.tolist(), "feat_std": fstd.tolist()},
                   str(sp))
        res["head_checkpoint"] = str(sp)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
