"""Train the decoupled style head on the λ=20 teacher bank (§15.1.5 DAgger-0/1).

Round 0 was OFFLINE on the teacher's OWN trajectories
(collect_teacher_bank.py): pairs are teacher pick vs the champion 1-ply
pick on the same states, with survival-margin weighting (Phase-B fix),
standardized features, and the veto chassis. Round-1 banks store
(placed, preview) piece ids + lockout masks, so the 1-ply argmax and its
survival margin are exact. Legacy banks without piece ids fall back to
the round-0 lowest-Phi heuristic (unweighted).

Screens at cap 300 on FRESH seeds (default 20480+, never used before):
1-ply + alpha*style over a veto x alpha grid — the fast student. The
teacher itself needs full 2-ply; the student test is how much style
transfers to 1-ply without breaking the survival gate.

Usage:
    python train_teacher_head.py --bank ../_diag_out/teacher_bank_lam20.npz \
        --weights ../../model_weights_snapshot9.pt --steps 400 --seed 42 \
        --screen-games 30 --screen-seed0 20480 \
        --alphas 0 0.5 1.0 --veto-margins 10 25 \
        --out ../_diag_out/teacher_head_r1.json
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
    placed = np.asarray(data["placed"]) if "placed" in data else None
    nxt_ids = np.asarray(data["next"]) if "next" in data else None
    lockouts = np.asarray(data["lockouts"]) if "lockouts" in data else None
    # cand_blob rows: [ncands, lines..., rows_flat...] per state
    states = []
    off = 0
    for _ in range(len(pre)):
        n = int(blob[off])
        lines = [int(x) for x in blob[off + 1:off + 1 + n]]
        rows = blob[off + 1 + n:off + 1 + n + n * 21].reshape(n, 21)
        off += 1 + n + n * 21
        states.append((rows, lines))
    return pre, states, picks, meta, placed, nxt_ids, lockouts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--veto-margins", nargs="+", type=float, default=[10.0, 25.0])
    ap.add_argument("--veto-margin", type=float, default=None)
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
    pre, states, picks, meta, placed_ids, next_ids, lockouts = _decode_bank(
        Path(args.bank))
    has_ids = (placed_ids is not None and next_ids is not None
               and lockouts is not None)
    print(f"Teacher bank: {len(pre)} states, lam={meta.get('lam')}, "
          f"clears={meta.get('clears')}, piece_ids={has_ids}")
    surv = snapshot9_frozen_copy()
    print("Survival net frozen: snapshot9")

    # 1-ply champion pick per state (student side of the pair).
    # Round-1 banks carry exact (placed, preview) ids + lockout masks:
    # score every viable candidate with frozen snapshot9 and take the
    # argmax as the flat side, with the survival margin as its weight
    # (Phase-B recipe). Legacy banks fall back to lowest-Phi (unweighted).
    from afterstate import flat_rows_to_tensor
    pairs, margins = [], []
    use_exact = has_ids
    t0 = time.perf_counter()
    surv.eval()
    with torch.no_grad():
        for s, ((rows, lines), pick_t) in enumerate(zip(states, picks)):
            k = rows.shape[0]
            if use_exact:
                lo_mask = lockouts[s, :k].astype(bool)
                viable = [i for i in range(k) if not lo_mask[i]]
                if not viable:
                    continue
                boards = flat_rows_to_tensor(rows.reshape(-1), k)
                cur = torch.full((k,), int(placed_ids[s]), dtype=torch.long)
                nxt = torch.full((k,), int(next_ids[s]), dtype=torch.long)
                sv = surv(boards, cur, nxt).flatten()
                sv_list = [float(sv[i]) for i in range(k)]
                flat_j = max(viable, key=lambda i: (sv_list[i], -i))
                margin = float(sv_list[int(pick_t)]) - float(
                    sv_list[flat_j]) if not lo_mask[int(pick_t)] else -1e6
            else:
                boards = flat_rows_to_tensor(rows.reshape(-1), k)
                phis = [float(__import__("well_potential").well_potential(
                    rows[j])) for j in range(k)]
                flat_j = int(np.argmin(phis))
                margin = 0.0
            if int(pick_t) == flat_j:
                continue
            pairs.append((style_features_from_rows(rows[int(pick_t)],
                                                   lines[int(pick_t)]),
                          style_features_from_rows(rows[flat_j],
                                                   lines[flat_j])))
            margins.append(margin)
            if (s + 1) % 1000 == 0:
                print(f"  paired {s + 1}/{len(pre)} (pairs {len(pairs)})",
                      flush=True)
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
    if use_exact:
        surv_margin = torch.tensor(margins, dtype=torch.float32)
        pair_w = torch.exp(surv_margin / 10.0).clamp(max=1.0)
        print(f"pair survival margins: mean {float(surv_margin.mean()):+.2f}, "
              f"frac>=0 {int((surv_margin >= 0).sum())}/{len(surv_margin)}, "
              f"mean weight {float(pair_w.mean()):.3f}")
    else:
        pair_w = torch.ones(len(pairs), dtype=torch.float32)
        print("legacy bank: no survival margins, weights=1.0")
    losses = []
    head.train()
    for step in range(args.steps):
        idx = sched[step]
        opt.zero_grad()
        s2 = head(z2[idx]).flatten()
        s1 = head(z1[idx]).flatten()
        per = F.softplus(-(s2 - s1) + args.margin)
        loss = (per * pair_w[idx]).mean()
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
    veto_list = (args.veto_margins if args.veto_margins
                 else [args.veto_margin if args.veto_margin is not None
                       else 25.0])
    screens = {}
    for vm in veto_list:
        for a in args.alphas:
            key = f"a{a}_v{vm:g}"
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(_dsh._style_worker,
                                    (wp, head_state, feat_stats,
                                     args.screen_seed0 + i, args.max_pieces,
                                     (a, vm)))
                        for i in range(args.screen_games)]
                rows_a = [f.result() for f in futs]
            pieces_a = [r[0] for r in rows_a]
            lines_a = [r[1] for r in rows_a]
            tet_a = sum(int(r[2].get(4, 0)) for r in rows_a)
            scoring_a = sum(int(r[2].get(k, 0)) for r in rows_a
                            for k in (1, 2, 3, 4))
            screens[key] = {
                "alpha": a, "veto_margin": vm,
                "mean_pieces": statistics.fmean(pieces_a),
                "mean_lines": statistics.fmean(lines_a),
                "tetrises": tet_a,
                "tetris_share": (tet_a / scoring_a) if scoring_a else 0.0,
                "capped": sum(1 for p in pieces_a if p >= args.max_pieces),
                "per_seed": pieces_a, "per_seed_lines": lines_a,
            }
            print(f"  alpha={a} veto={vm:g}: "
                  f"{statistics.fmean(pieces_a):.1f} pcs / "
                  f"{statistics.fmean(lines_a):.1f} lines, "
                  f"cap {screens[key]['capped']}/{args.screen_games}, "
                  f"tetris {tet_a}", flush=True)
    res = {"bank": str(args.bank), "teacher_lam": meta.get("lam"),
           "pairs": len(pairs), "steps": args.steps, "lr": args.lr,
           "exact_pairs": bool(use_exact),
           "feat_direction": {k: round(float(v), 4)
                              for k, v in zip(names, d.tolist())},
           "final_loss": round(statistics.fmean(losses[-50:]), 4),
           "screens": screens,
           "screen_seed0": args.screen_seed0, "games": args.screen_games,
           "veto_margins": veto_list, "alphas": list(args.alphas)}
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
