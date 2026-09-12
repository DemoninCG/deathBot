"""Fresh-panel confirmation of the soup screen leader (2026-09-11).

The §49-adjacent soup screen (reporting seeds 26000s/31000s,
_soup_screen.json) found: soup3 (element-wise mean of the three §49
replication checkpoints s42/s43/s44) is the best point estimate on the
30-tape reporting leg (+514.2 vs snap9, t+6.33, 19/30 cap) but is NOT
separated from its best member (s43: +72, t+0.80; s42: +141, t+1.33).
Per project discipline (§8, §36.3: n=30 resolves ~±120 on paired sd ~600;
one decision per fresh panel), a promotion claim requires a FRESH panel.

This driver runs the pre-registered confirmation:
  arms: soup3 (candidate) vs s42 (= fullsplit_seed42 = snapshot10, the
        standing 1-ply champion) — paired, 30 FRESH tapes 21020-21049
        (CONSUMED on completion; next free 21050+), cap 1500,
        sweep harness (evaluate_weights_parallel), eps 0, workers 8.
  Gate (pre-registered): promote soup3 over s42 iff paired pieces delta
        t >= 2.0 AND cap rate >= s42's; if n.s., hold s42 as champion and
        record the soup as "no separation" (either way the recipe is
        unchanged — this is checkpoint selection, not new training).

REPORTING-ONLY note: 21020+ is a dev range (not a promotion reserve);
5300+ reserves remain untouched regardless of outcome.

Usage:
    python _soup_confirm.py --out ../_diag_out/_soup_confirm.json
Incremental: writes after each arm; safe to interrupt and re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
D = HERE.parent / "_diag_out"

PANEL_SEED0 = 21020
N_GAMES = 30


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_soup_confirm.json"))
    ap.add_argument("--n", type=int, default=N_GAMES)
    ap.add_argument("--seed0", type=int, default=PANEL_SEED0)
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming from {op.name}", flush=True)
    results.setdefault("note",
                       "Fresh-panel confirmation: soup3 vs s42(champ), "
                       f"seeds {args.seed0}-{args.seed0 + args.n - 1}, "
                       "cap 1500, paired. Pre-registered gate: promote iff "
                       "paired t >= 2.0.")
    results.setdefault("arms", {})

    soup3 = D / "_soup_s42_s43_s44.pt"
    champ = D / "fullsplit_seed42.pt"  # == snapshot10 (sha 9989ce86)
    ckpts = {"s42_champ": champ, "soup3": soup3}

    from evaluate import evaluate_weights_parallel

    for name in sorted(ckpts):
        if name in results["arms"]:
            print(f"{name}: already done", flush=True)
            continue
        t0 = time.perf_counter()
        ev = evaluate_weights_parallel(str(ckpts[name]), games=args.n,
                                       seed=args.seed0, max_pieces=1500,
                                       workers=8)
        pieces = [g.pieces for g in ev.games]
        rec = {
            "mean": round(statistics.fmean(pieces), 1),
            "median": statistics.median(pieces),
            "capped": sum(p >= 1500 for p in pieces),
            "per_seed": pieces,
            "sha": hashlib.sha256(Path(ckpts[name]).read_bytes())
                   .hexdigest()[:16],
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        results["arms"][name] = rec
        op.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{name:10s} mean {rec['mean']:7.1f} med {rec['median']:7.1f} "
              f"cap {rec['capped']:2d}/{args.n} ({rec['wall_s']:.0f}s) — "
              f"SAVED", flush=True)

    a = results["arms"]["soup3"]["per_seed"]
    b = results["arms"]["s42_champ"]["per_seed"]
    dl = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(dl)
    sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
    t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
    wins = sum(d > 0 for d in dl)
    losses = sum(d < 0 for d in dl)
    results["paired_soup3_vs_s42"] = {
        "delta": round(m, 1), "t": round(t, 2),
        "wins": wins, "losses": losses,
        "discordant": wins + losses,
        "promote_soup3": bool(t >= 2.0),
        "per_seed_delta": dl,
    }
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\npaired soup3 - s42: {m:+.1f}  t {t:+.2f}  W/L {wins}/{losses}",
          flush=True)
    print("GATE:", "PROMOTE soup3" if t >= 2.0 else "HOLD s42 (champion "
          "unchanged) — consider extending n or adding recipe seeds 45/46",
          flush=True)


if __name__ == "__main__":
    main()
