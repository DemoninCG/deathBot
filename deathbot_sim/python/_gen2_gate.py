"""Gen-2 fresh-panel gate (pre-registered; 2026-09-11).

Context: gen-2 (champion fullsplit_seed42 start, fs2ply = its own full
2-ply expansion as teacher, validated dose) screened at gen2 - champ =
+277.6 (t+2.75) / +238.2 (t+2.28) / +314.8 (t+3.39) on reporting seeds
31000s for schedule seeds 42/43/44 respectively — three independent
training seeds, all positive, all t >= 2.28. No degradation on consumed
training tapes (26000s: +6/+8/-48, all n.s.).

Per the 2026-09-11 soup lesson (a +130 lean on 31000s evaporated at
pooled n=90), 31000s has been viewed many times and cannot support a
promotion. This driver runs the PRE-REGISTERED fresh-panel gate:

  Panel: seeds 21110-21139 (30 fresh dev tapes; CONSUMED on completion;
         next free 21150+; 5300+ promotion reserves untouched).
  Arms:  s42_champ (standing champion) + gen2_seed42/43/44.
  Pre-registered decision (written before any fresh eval):
    PRIMARY: pooled per-tape mean-of-the-three-gen2-seeds delta vs champ.
             PROMOTE the gen-2 recipe iff pooled t >= 2.0 AND every
             individual seed's paired delta > 0 on this panel.
    SECONDARY: per-seed deltas/t, cap rates, per-seed W/L.
    If PASS: promotion-grade confirmation on the 5300+ reserve (n=300,
             matching the §30 standard) before any champion change.
    If FAIL: record no-separation; champion unchanged; recipe recorded
             as "screen-positive, gate-negative" (third n=30 mirage
             instance would be noted).

Reporting-only ranges: 26000s/31000s; this gate consumes 21110-21139.

Usage:
    python _gen2_gate.py --out ../_diag_out/_gen2_gate.json
Incremental: writes after each arm; interrupt-safe.
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

SEED0 = 21110
N = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_gen2_gate.json"))
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--seed0", type=int, default=SEED0)
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}", flush=True)
    results.setdefault("note",
                       "PRE-REGISTERED fresh-panel gate: gen-2 recipe vs "
                       "champion. PROMOTE iff pooled mean-of-3-seeds "
                       "per-tape delta t >= 2.0 AND all three per-seed "
                       f"deltas > 0. Panel {args.seed0}-"
                       f"{args.seed0 + args.n - 1} (fresh dev).")
    results.setdefault("arms", {})

    ckpts = {
        "s42_champ": D / "fullsplit_seed42.pt",
        "gen2_seed42": D / "_gen2_seed42.pt",
        "gen2_seed43": D / "_gen2_seed43.pt",
        "gen2_seed44": D / "_gen2_seed44.pt",
    }

    from evaluate import evaluate_weights_parallel

    for name in sorted(ckpts):
        if name in results["arms"]:
            print(f"{name}: done", flush=True)
            continue
        ev = evaluate_weights_parallel(str(ckpts[name]), games=args.n,
                                       seed=args.seed0, max_pieces=1500,
                                       workers=8)
        pieces = [g.pieces for g in ev.games]
        results["arms"][name] = {
            "mean": round(statistics.fmean(pieces), 1),
            "capped": sum(p >= 1500 for p in pieces),
            "per_seed": pieces,
        }
        op.write_text(json.dumps(results, indent=2), encoding="utf-8")
        a = results["arms"][name]
        print(f"{name:12s} mean {a['mean']:7.1f} cap {a['capped']:2d}/"
              f"{args.n} — SAVED", flush=True)

    champ = results["arms"]["s42_champ"]["per_seed"]
    per_seed_stats = {}
    pooled_rows = []
    all_positive = True
    for name in sorted(ckpts):
        if name == "s42_champ":
            continue
        pieces = results["arms"][name]["per_seed"]
        dl = [a - b for a, b in zip(pieces, champ)]
        m = statistics.fmean(dl)
        sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
        t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
        all_positive = all_positive and m > 0
        per_seed_stats[name] = {"delta": round(m, 1), "t": round(t, 2),
                                "wins": sum(x > 0 for x in dl),
                                "losses": sum(x < 0 for x in dl)}
        print(f"  {name:12s} vs champ {m:+7.1f} t {t:+.2f} "
              f"W/L {per_seed_stats[name]['wins']}/"
              f"{per_seed_stats[name]['losses']}", flush=True)
    # pooled: per-tape mean across the three gen2 seeds
    gen2_names = [n for n in sorted(ckpts) if n != "s42_champ"]
    pooled_dl = []
    for i in range(args.n):
        tape_mean = statistics.fmean(results["arms"][g]["per_seed"][i]
                                     for g in gen2_names)
        pooled_dl.append(tape_mean - champ[i])
    m = statistics.fmean(pooled_dl)
    sd = statistics.stdev(pooled_dl)
    t = m / (sd / len(pooled_dl) ** 0.5) if sd > 0 else float("nan")
    verdict = "PROMOTE-to-promotion-panel" if (t >= 2.0 and all_positive) \
        else "NO-PROMOTION (champion unchanged)"
    results["gate"] = {
        "pooled_delta": round(m, 1), "pooled_t": round(t, 2),
        "all_three_positive": all_positive,
        "per_seed": per_seed_stats,
        "verdict": verdict,
    }
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nPOOLED mean-of-3 gen2 - champ: {m:+.1f} t {t:+.2f} "
          f"(all positive: {all_positive})", flush=True)
    print("GATE:", verdict, flush=True)


if __name__ == "__main__":
    main()