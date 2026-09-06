"""Phase A: train the style head on 2-ply-vs-1-ply disagreement pairs.

Frozen snapshot9 supplies survival values; the head learns the FEATURE
direction of the 2-ply pick via margin ranking loss:
  L = softplus(-(style[pick2ply] - style[pick1ply]))   (no value targets)
Only states where 2-ply disagrees with 1-ply carry signal (864/4000 in the
P2T run). Survivors-only combination at screen time preserves the gate.

Usage:
    python diag_style_head.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \
        --pairs 4000 --steps 400 --alpha 15 --screen-games 30 \
        --screen-seed0 20194 --out ../_diag_out/style_head_A.json
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
from diag_phase2 import Bank  # noqa: E402
from style_head import (StyleHead, style_features_from_rows,  # noqa: E402
                        snapshot9_frozen_copy)


def _style_worker(payload) -> tuple:
    """Module-level worker (picklable): 1-ply + alpha*style game.

    Foundational fixes applied at screen time (must match training):
    - features STANDARDIZED with the saved stats (raw spans O(100)->O(1));
    - style CALIBRATED per state (minus state mean): only within-state
      ranking reaches the combination, never a global bias;
    - SURVIVAL VETO: candidates > veto_margin below best v are ineligible
      before style is added (payload alpha tuple: (alpha, veto_margin)).
    """
    import numpy as _np
    import torch as _t
    import deathbot_sim as _ds
    from model import TAPNet as _N, piece_to_index as _pi
    from style_head import (StyleHead as _H, style_features_from_rows as _sf,
                            apply_stats as _ap)
    from afterstate import flat_rows_to_tensor as _frt
    from piece_history import PieceQueue as _PQ
    import random as _r
    (weights_path_s, head_state, feat_stats, seed, max_pieces,
     alpha_veto) = payload
    alpha, veto_margin = alpha_veto
    m = _N(output="linear")
    m.load_state_dict(_t.load(weights_path_s, map_location="cpu"))
    m.eval()
    h = _H(); h.load_state_dict(head_state); h.eval()
    fmean = _t.tensor(feat_stats["mean"])
    fstd = _t.tensor(feat_stats["std"])
    board = _ds.Board()
    pq = _PQ(_r.Random(seed))
    pieces = lines = 0
    clears = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    while pieces < max_pieces:
        cands = board.search_locks(pq.current)
        if not cands:
            return (pieces, lines, clears, "no_candidates")
        viable = [i for i, c in enumerate(cands)
                  if not board.locks_out(c.piece)]
        if not viable:
            return (pieces, lines, clears, "lockout")
        res = board.lock_candidates_full(pq.current)
        n_c = len(res[0])
        rows = _np.asarray(res[4], dtype=_np.uint32).reshape(n_c, 21)
        boards = _frt(rows.reshape(-1), n_c)
        cur = _t.full((n_c,), _pi(pq.current), dtype=_t.long)
        nxt = _t.full((n_c,), _pi(pq.next_piece), dtype=_t.long)
        with _t.no_grad():
            sv = m(boards, cur, nxt).flatten()
            feats = _ap(_t.tensor(_np.stack(
                [_sf(rows[i], int(res[5][i])) for i in range(n_c)])),
                fmean, fstd)
            st = h(feats).flatten()
            st = st - st[viable].mean()  # per-state calibration
        best = max(float(sv[i]) for i in viable)
        fin = {}
        for i in range(n_c):
            if i not in viable or (best - float(sv[i])) > veto_margin:
                fin[i] = float("-inf")
            else:
                fin[i] = float(sv[i]) + alpha * float(st[i])
        pick = max(range(n_c), key=lambda i: (fin[i], -i))
        lines += int(res[5][pick])
        clears[int(res[5][pick])] += 1
        board = _ds.Board.from_row_bits([int(v) for v in rows[pick]])
        pq.advance()
        pieces += 1
    return (pieces, lines, clears, "max_pieces")


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
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=15.0,
                    help="search-time style weight (z-units, ~k15 scale)")
    ap.add_argument("--veto-margin", type=float, default=25.0,
                    help="survival veto: candidates further than this (z-units) "
                         "below best v are ineligible before style applies")
    ap.add_argument("--alphas", nargs="+", type=float, default=None,
                    help="screen a sweep of alphas (e.g. --alphas 0 1 3 10); "
                         "each is evaluated on the same panel, no extra training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-games", type=int, default=30)
    ap.add_argument("--screen-seed0", type=int, default=20194)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-head", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    bank = Bank(args.bank)
    n = bank.n_states
    print(f"Bank: {n} states / {int(bank.cand_offsets[-1])} candidates")
    weights_path = resolve_weights(args.weights)
    surv = snapshot9_frozen_copy()
    print(f"Survival net frozen: {weights_path.name}")

    rng = random.Random(args.seed)
    states = rng.sample(range(n), min(args.pairs, n))

    # Teacher: 2-ply argmax per state; keep only disagreements with 1-ply.
    # Also record the SURVIVAL margin (v[pick2] - v[pick1]) per pair so
    # training can downweight survival-harmful teacher picks (Phase A fix).
    import deathbot_sim as ds
    from two_ply import TwoPlyScorer
    from afterstate import flat_rows_to_tensor
    scorer = TwoPlyScorer(surv, beam=10 ** 9)
    placed, nxt = bank._replay_piece_ids()
    pairs: list[tuple[np.ndarray, np.ndarray]] = []  # (feat_2ply, feat_1ply)
    triplets: list[tuple[int, int, int]] = []  # (state, pick2, pick1)
    surv_vals: dict[int, torch.Tensor] = {}
    t0 = time.perf_counter()
    for si, s in enumerate(states):
        lo, hi = bank.candidate_slice(s)
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        kind, nk = placed[s], nxt[s]
        try:
            pick2 = scorer.choose(pre, kind, nk)
        except ValueError:
            continue
        pick1 = int(bank.chosen_idx[s])
        if pick2 == pick1:
            continue
        rows = bank.cand_rows[lo * 21:hi * 21].reshape(hi - lo, 21)
        lines = [int(x) for x in bank.cand_lines[lo:hi]]
        pairs.append((style_features_from_rows(rows[pick2], lines[pick2]),
                      style_features_from_rows(rows[pick1], lines[pick1])))
        triplets.append((s, pick2, pick1))
        with torch.no_grad():
            boards = flat_rows_to_tensor(rows.reshape(-1), hi - lo)
            cur = torch.full((hi - lo,), piece_to_index(placed[s]))
            nx = torch.full((hi - lo,), piece_to_index(nxt[s]))
            surv_vals[s] = surv(boards, cur, nx).flatten()
        if (si + 1) % 500 == 0:
            print(f"  teacher {si + 1}/{len(states)} (pairs {len(pairs)})", flush=True)
    print(f"teacher done in {time.perf_counter() - t0:.0f}s; "
          f"{len(pairs)}/{len(states)} disagreement pairs")
    if not pairs:
        raise SystemExit("no disagreement pairs — nothing to train")

    f2 = torch.tensor(np.stack([p[0] for p in pairs]))
    f1 = torch.tensor(np.stack([p[1] for p in pairs]))
    # Feature direction check: mean(2ply - 1ply) per dim.
    d = (f2 - f1).mean(0)
    names = ["wheight", "holes", "holedepth", "cav", "wellcost", "spawn",
             "lines", "deepwell"]
    print("mean(feat2ply - feat1ply): " +
          ", ".join(f"{k}={v:+.3f}" for k, v in zip(names, d.tolist())))

    # Foundational fix 1: standardize features (raw spans O(100)->O(1)).
    # Foundational fix 2: weight pairs by the SURVIVAL margin — pairs where
    # the teacher pick is far below the 1-ply pick in survival value are the
    # ones that killed Phase A (style memorized survival-harmful picks).
    from style_head import fit_feature_stats, apply_stats
    fmean, fstd = fit_feature_stats(torch.cat([f2, f1]))
    print("feat std: " +
          ", ".join(f"{k}={v:.3f}" for k, v in zip(names, fmean.tolist())))
    print("feat std: " +
          ", ".join(f"{k}={v:.3f}" for k, v in zip(names, fstd.tolist())))
    surv_margin = torch.tensor(
        [float(surv_vals[s][p2]) - float(surv_vals[s][p1])
         for (s, p2, p1) in triplets])
    pair_w = torch.exp(surv_margin / 10.0).clamp(max=1.0)
    print(f"pair survival margins: mean {float(surv_margin.mean()):+.2f}, "
          f"frac>=0 {int((surv_margin >= 0).sum())}/{len(surv_margin)}, "
          f"mean weight {float(pair_w.mean()):.3f}")
    z2 = apply_stats(f2, fmean, fstd)
    z1 = apply_stats(f1, fmean, fstd)

    head = StyleHead()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = [rng.sample(range(len(pairs)), min(args.batch_size, len(pairs)))
             for _ in range(args.steps)]
    losses = []
    t1 = time.perf_counter()
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
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{args.steps}: "
                  f"loss={statistics.fmean(losses[-100:]):.4f}", flush=True)
    head.eval()
    dt = time.perf_counter() - t1
    # Persist standardization stats alongside the head checkpoint.
    feat_stats = {"mean": fmean.tolist(), "std": fstd.tolist()}

    # Screen: 1-ply + alpha*style, survivors-only combination, cap-300 panel.
    # An --alphas sweep reuses the ONE trained head across alphas (no extra
    # training): alpha=0 must reproduce the survival policy (gate check).
    from concurrent.futures import ProcessPoolExecutor
    head_state = {k: v.detach().cpu().clone()
                  for k, v in head.state_dict().items()}
    wp_s = str(weights_path)
    sweep = args.alphas if args.alphas else [args.alpha]
    screens = {}
    for a in sweep:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_style_worker,
                                (wp_s, head_state, feat_stats,
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
            "lines_per_100": 100.0 * sum(lines_a) / max(1, sum(pieces_a)),
            "tetrises": tet_a,
            "tetris_share": (tet_a / scoring_a) if scoring_a else 0.0,
            "capped": sum(1 for p in pieces_a if p >= args.max_pieces),
            "per_seed": pieces_a, "per_seed_lines": lines_a,
        }
        print(f"  alpha={a}: {statistics.fmean(pieces_a):.1f} pcs / "
              f"{statistics.fmean(lines_a):.1f} lines, "
              f"cap {screens[str(a)]['capped']}/{args.screen_games}, "
              f"tetris {tet_a}", flush=True)
    # Back-compat top-level fields follow the primary --alpha (or first sweep).
    primary = str(args.alpha if not args.alphas else sweep[0])
    rows = None  # per-alpha rows live in screens
    res = {
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "pairs": len(pairs), "steps": args.steps, "lr": args.lr,
        "margin": args.margin, "alpha": args.alpha, "seed": args.seed,
        "train_s": round(dt, 1),
        "final_loss": round(statistics.fmean(losses[-50:]), 4),
        "feat_direction": {k: round(float(v), 4) for k, v in zip(names, d.tolist())},
        "screens": screens,
        "mean_pieces": screens[primary]["mean_pieces"],
        "mean_lines": screens[primary]["mean_lines"],
        "lines_per_100": screens[primary]["lines_per_100"],
        "tetrises": screens[primary]["tetrises"],
        "tetris_share": screens[primary]["tetris_share"],
        "capped": screens[primary]["capped"],
        "per_seed": screens[primary]["per_seed"],
        "per_seed_lines": screens[primary]["per_seed_lines"],
    }
    if args.save_head:
        sp = Path(args.out).resolve().parent / f"style_head_seed{args.seed}.pt"
        torch.save({"state_dict": head.state_dict(),
                    "feat_mean": fmean.tolist(), "feat_std": fstd.tolist()},
                   str(sp))
        res["head_checkpoint"] = str(sp)
        print(f"  saved head to {sp}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
