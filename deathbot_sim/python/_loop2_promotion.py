"""Loop-gen-2 PROMOTION panel (§52.8-grade; pre-registered 2026-09-13).

Gate authorization: `_loop2_final_gate.json` verdict PROMOTE-TO-5400-PANEL
(G1 +294.4 t+2.20 AND G2 +42.6 t+1.23, both PASS on dev 21270-21329).

Protocol (same §51.4 rule as the gen-2 promotion):
  Panel:  seeds 5400-5499 (100 fresh tapes, PROMOTION RESERVE — 5300-5399
          consumed by the §51.4 gen-2 promotion; first use of 5400s),
          cap 3000. NOTE: cap-3000 means are NOT comparable to any
          historical cap-1500 table.
  Arms:   snapshot11 (standing 1-ply champion) vs loop2_r3 (final loop
          working champion `_loop2_champ_r3.pt`). No reference arm enters
          the decision.
  DECISION: PROMOTE loop2_r3 to 1-ply champion iff paired per-tape delta
          (challenger - champion) t >= 2.0 at n=100 AND challenger cap
          rate is not lower than champion's.
  EXTENSION (pre-registered, exactly one): if delta > 0 but t < 2.0, run
          ONE more block of n=200 fresh tapes (5500-5699) and test the
          POOLED per-tape deltas at n=300; promote iff pooled t >= 2.0.
          If pooled delta <= 0 at any point, STOP: champion unchanged.
  If PROMOTED: copy `_loop2_champ_r3.pt` -> `model_weights_snapshot12.pt`
          (new champion lineage name), keep snapshot11 as fallback.

Usage:
    python _loop2_promotion.py --out ../_diag_out/_loop2_promotion.json
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

SEED0 = 5400
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
    ap.add_argument("--out", default=str(D / "_loop2_promotion.json"))
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--challenger", default=str(D / "_loop2_champ_r3.pt"))
    ap.add_argument("--champion",
                    default=str(ROOT / "model_weights_snapshot11.pt"))
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}", flush=True)
    results.setdefault("note",
                       "PRE-REGISTERED loop-gen-2 promotion panel: "
                       "loop2_r3 vs snapshot11, fresh 5400s reserve, cap "
                       "3000. PROMOTE iff paired t >= 2.0 AND cap rate "
                       "not lower; one pre-authorized extension block "
                       "(+200) if 0 < t < 2.")
    results.setdefault("arms", {})

    ckpts = {"challenger": Path(args.challenger),
             "champion": Path(args.champion)}

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

    champ = results["arms"]["champion"]
    chal = results["arms"]["challenger"]
    pr = paired(chal["per_seed"], champ["per_seed"])
    pr["cap_challenger"] = chal["capped"]
    pr["cap_champion"] = champ["capped"]
    pr["seeds"] = f"{args.seed0}-{args.seed0 + args.n - 1}"
    promote = pr["t"] >= 2.0 and chal["capped"] >= champ["capped"]
    pr["verdict"] = ("PROMOTE" if promote else
                     ("EXTEND (one pre-authorized +200 block: 5500-5699)"
                      if pr["delta"] > 0 else
                      "NO-PROMOTION (champion unchanged)"))
    results["decision"] = pr
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\npaired challenger - champion @cap{args.cap}: "
          f"{pr['delta']:+.1f} t {pr['t']:+.2f} "
          f"W/T/L {pr['wins']}/{pr['ties']}/{pr['losses']} "
          f"cap {pr['cap_challenger']}/{args.n} vs "
          f"{pr['cap_champion']}/{args.n}", flush=True)
    print("DECISION:", pr["verdict"], flush=True)

    if promote:
        dst = ROOT / "model_weights_snapshot12.pt"
        if not dst.exists():
            shutil.copyfile(str(ckpts["challenger"]), str(dst))
            results["promoted_to"] = str(dst)
            results["promoted_sha"] = hashlib.sha256(
                dst.read_bytes()).hexdigest()[:16]
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"copied _loop2_champ_r3.pt -> {dst.name} (new champion "
                  f"lineage file; snapshot11 kept as fallback)", flush=True)


if __name__ == "__main__":
    main()
