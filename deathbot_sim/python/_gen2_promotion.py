"""Gen-2 PROMOTION panel (§30-grade; pre-registered 2026-09-12).

User decision (2026-09-12): promote-test gen2_seed44 — the seed that
leads BOTH independent panels (31000s: +314.8 t+3.39; gate 21110s:
+208.5 t+1.95, cap 25/30). Acknowledged bias: best-of-3 selection
inflates the prior; the fresh panel below is the actual test of the
named checkpoint, and the choice is made BEFORE seeing any 5300s data.

Protocol (pre-registered before any 5300s data exists):
  Panel:  seeds 5300-5399 (100 fresh tapes, PROMOTION RESERVE, first
          use), cap 3000 (user call: 1500-cap rate is saturating —
          champ 15/30, gen2 23-25/30 at cap 1500 — and longer games
          carry more signal per tape). NOTE: cap-3000 means are NOT
          comparable to any historical cap-1500 table.
  Arms:   s42_champ (standing champion fullsplit_seed42) vs
          gen2_seed44. snap9 included as harness reference only
          (does NOT enter the decision).
  DECISION: PROMOTE gen2_seed44 to 1-ply champion iff paired per-tape
          delta (gen2 - champ) t >= 2.0 at n=100 AND gen2 cap rate is
          not lower than champ's.
  EXTENSION (pre-registered, exactly one): if delta > 0 but t < 2.0,
          run ONE more block of n=200 fresh tapes (5400-5599) and test
          the POOLED per-tape deltas at n=300; promote iff pooled
          t >= 2.0. If pooled delta <= 0 at any point, STOP: champion
          unchanged, no further blocks.
  If PROMOTED: copy gen2_seed44 -> model_weights_snapshot11.pt (new
          champion lineage name), update PROJECT_STATE.md, keep
          fullsplit_seed42/snapshot10 as the fallback champion.

Usage:
    python _gen2_promotion.py --out ../_diag_out/_gen2_promotion.json
Incremental: writes after each arm; interrupt-safe; resume-safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"

SEED0 = 5300
N = 100
CAP = 3000


def paired(a: list[int], b: list[int]) -> dict:
    dl = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(dl)
    sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
    t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
    return {"delta": round(m, 1), "t": round(t, 2),
            "wins": sum(x > 0 for x in dl), "losses": sum(x < 0 for x in dl),
            "ties": sum(x == 0 for x in dl), "n": len(dl)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_gen2_promotion.json"))
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--arm", default="gen2_seed44",
                    help="which gen-2 checkpoint to test (pre-named: 44)")
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}", flush=True)
    results.setdefault("note",
                       "PRE-REGISTERED promotion panel: gen2_seed44 vs "
                       "champion, fresh 5300s reserve, cap 3000. PROMOTE "
                       "iff paired t >= 2.0 AND cap rate not lower; one "
                       "pre-authorized extension block (+200) if 0 < t < 2.")
    results.setdefault("arms", {})

    cand = D / f"_gen2_seed{args.arm.replace('gen2_seed', '')}.pt"
    ckpts = {"s42_champ": D / "fullsplit_seed42.pt",
             args.arm: cand,
             "snap9_ref": ROOT / "model_weights_snapshot9.pt"}

    from evaluate import evaluate_weights_parallel

    for name in sorted(ckpts):
        if name in results["arms"]:
            print(f"{name}: done", flush=True)
            continue
        t0 = time.perf_counter()
        ev = evaluate_weights_parallel(str(ckpts[name]), games=args.n,
                                       seed=args.seed0, max_pieces=args.cap,
                                       workers=8)
        pieces = [g.pieces for g in ev.games]
        results["arms"][name] = {
            "mean": round(statistics.fmean(pieces), 1),
            "median": statistics.median(pieces),
            "capped": sum(p >= args.cap for p in pieces),
            "per_seed": pieces,
            "sha": hashlib.sha256(Path(ckpts[name]).read_bytes())
                   .hexdigest()[:16],
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        op.write_text(json.dumps(results, indent=2), encoding="utf-8")
        a = results["arms"][name]
        print(f"{name:12s} mean {a['mean']:7.1f} med {a['median']:7.1f} "
              f"cap {a['capped']:3d}/{args.n} ({a['wall_s']:.0f}s) — SAVED",
              flush=True)

    champ = results["arms"]["s42_champ"]
    gen2 = results["arms"][args.arm]
    pr = paired(gen2["per_seed"], champ["per_seed"])
    pr["cap_gen2"] = gen2["capped"]
    pr["cap_champ"] = champ["capped"]
    promote = pr["t"] >= 2.0 and gen2["capped"] >= champ["capped"]
    pr["verdict"] = ("PROMOTE" if promote else
                     ("EXTEND (one pre-authorized +200 block)"
                      if pr["delta"] > 0 else
                      "NO-PROMOTION (champion unchanged)"))
    results["decision"] = pr
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\npaired {args.arm} - champ @cap{args.cap}: {pr['delta']:+.1f} "
          f"t {pr['t']:+.2f} W/T/L {pr['wins']}/{pr['ties']}/{pr['losses']} "
          f"cap {pr['cap_gen2']}/{args.n} vs {pr['cap_champ']}/{args.n}",
          flush=True)
    print("DECISION:", pr["verdict"], flush=True)

    if promote:
        dst = ROOT / "model_weights_snapshot11.pt"
        if not dst.exists():
            shutil.copyfile(str(cand), str(dst))
            results["promoted_to"] = str(dst)
            results["promoted_sha"] = hashlib.sha256(
                dst.read_bytes()).hexdigest()[:16]
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"copied {cand.name} -> {dst.name} (new champion "
                  f"lineage file; snapshot10/fullsplit_seed42 kept as "
                  f"fallback)", flush=True)


if __name__ == "__main__":
    main()