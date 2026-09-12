"""RESUME of the interrupted seed-replication eval phase (2026-09-11).

The original run (python/_distill_replicate.py, launched 2026-09-11) died
when the terminal window was closed during the EVAL phase. The expensive
phases survived on disk:

  - _diag_out/_fullsplit_repl_seed43.pt  (fit done 15:32)
  - _diag_out/_fullsplit_repl_seed44.pt  (fit done 16:07)

The JSON is written only at the very end of _distill_replicate.py, so the
per-step fit history (loss/dW/agree3k) and both eval legs were lost with the
process. Per-step history is NOT recoverable without re-fitting (~1.5 h) and
is not needed for the decisive question (recipe robustness). This driver
redoes ONLY the eval phase, with the same arms / harness / seed ranges as
the original, and — lesson applied — writes partial results to disk after
EVERY arm so another interruption cannot lose completed work.

Arms (identical to the original run):
  snap9               = model_weights_snapshot9.pt   (paired baseline)
  fullsplit_seed42    = _diag_out/fullsplit_seed42.pt (the +468 original)
  repl_seed43         = _diag_out/_fullsplit_repl_seed43.pt
  repl_seed44         = _diag_out/_fullsplit_repl_seed44.pt

Eval (REPORTING ONLY, paired vs snap9 in the sweep harness):
  train26000  = seeds 26000-26007 (8 games; consumed as training data §26.2)
  report31000 = seeds 31000-31029 (30 games; §28 demo-reporting range)
Both legs at cap 1500, workers=8, epsilon=0 — identical to
_diagill_replicate.py's run_eval wrapper (evaluate_weights_parallel).

Usage:
    python _distill_replicate_resume.py \
        --out ../_diag_out/_fullsplit_replicate.json
Writes per-arm JSON incrementally; safe to interrupt and re-run (skips
arms already recorded).
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
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"


def run_eval(weights: Path, games: int, seed0: int) -> list[int]:
    from evaluate import evaluate_weights_parallel
    ev = evaluate_weights_parallel(str(weights), games=games, seed=seed0,
                                   max_pieces=1500, workers=8)
    return [g.pieces for g in ev.games]


def summarize(pieces: list[int], base: list[int]) -> dict:
    dl = [x - y for x, y in zip(pieces, base)]
    m = statistics.fmean(dl)
    sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
    t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
    return {
        "mean": round(statistics.fmean(pieces), 1),
        "capped": sum(p >= 1500 for p in pieces),
        "delta_vs_snap9": round(m, 1),
        "t": round(t, 2),
        "wins": sum(d > 0 for d in dl),
        "losses": sum(d < 0 for d in dl),
        "per_seed": pieces,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_fullsplit_replicate.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    ap.add_argument("--include-fit-history", action="store_true",
                    help="re-fit to recover per-step history (SLOW, ~35 min "
                         "per seed; NOT needed for the decisive question)")
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming from existing {op.name}: "
              f"{sorted(results.keys())}", flush=True)
    results.setdefault("note",
                       "Eval-phase resume after window close (2026-09-11); "
                       "fit history lost, checkpoints + eval legs authoritative.")
    results["arms"] = results.get("arms", {})
    results.setdefault("eval_train26000", {})
    results.setdefault("eval_report31000", {})

    snap9 = ROOT / "model_weights_snapshot9.pt"
    ckpts: dict[str, Path] = {
        "snap9": snap9,
        "fullsplit_seed42": D / "fullsplit_seed42.pt",
    }
    for seed in args.seeds:
        p = D / f"_fullsplit_repl_seed{seed}.pt"
        if p.exists():
            ckpts[f"repl_seed{seed}"] = p
        else:
            print(f"WARNING: {p.name} missing — run _distill_replicate.py "
                  f"for seed {seed} first", flush=True)
    print("arms:", {k: str(v.name) for k, v in ckpts.items()}, flush=True)

    for tag, count, seed0 in (("eval_train26000", 8, 26000),
                              ("eval_report31000", 30, 31000)):
        # snap9 FIRST (it is the paired baseline; sorted order would put
        # fullsplit_seed42 before it and summarize() would get base=None).
        base = None
        ordered = ["snap9"] + [n for n in sorted(ckpts) if n != "snap9"]
        for name in ordered:
            if name in results[tag]:
                print(f"{tag} {name}: already done, skipping", flush=True)
                if name == "snap9":
                    base = results[tag][name]["per_seed"]
                continue
            t0 = time.perf_counter()
            pieces = run_eval(ckpts[name], count, seed0)
            if name == "snap9":
                base = pieces
            results[tag][name] = summarize(pieces, base)
            results[tag][name]["wall_s"] = round(time.perf_counter() - t0, 1)
            results[tag][name]["checkpoint_sha"] = hashlib.sha256(
                Path(ckpts[name]).read_bytes()).hexdigest()[:16]
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            r = results[tag][name]
            print(f"  {tag} {name:18s} mean {r['mean']:7.1f} "
                  f"cap {r['capped']:3d}/{count} "
                  f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                  f"W/L {r['wins']}/{r['losses']} "
                  f"({r['wall_s']:.0f}s) — SAVED", flush=True)

    print("wrote", op, flush=True)
    print("\n=== SUMMARY (paired vs snap9, same harness both arms) ===",
          flush=True)
    for tag in ("eval_train26000", "eval_report31000"):
        if not results[tag]:
            continue
        print(tag)
        for name, r in results[tag].items():
            print(f"  {name:18s} mean {r['mean']:7.1f} cap {r['capped']:3d} "
                  f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                  f"W/L {r['wins']}/{r['losses']}")


if __name__ == "__main__":
    main()
