"""Extend the soup3-vs-champion confirmation to n=60/90 (2026-09-11).

§49-adjacent context: the pre-registered fresh-panel gate (21020-21049,
_soup_confirm.json) gave soup3 - s42 = +128.5, t+1.36, W/L 12/7 — the same
lean as the reporting screen (+141.3, t+1.33 on 31000s). Two independent
panels agree in direction but neither reaches t >= 2 at n=30. Per the
§36.3 precedent (extend the SAME pre-registered decision with fresh tapes;
do not re-decide on a new panel), this driver appends further fresh blocks
and POOLS per-seed deltas with the original block.

Blocks (each CONSUMED on completion):
  block 1: 21020-21049 (already in _soup_confirm.json)
  block 2: 21050-21079 (this run, --blocks 2)
  block 3: 21080-21109 (optional, --blocks 3)

Gate stays pre-registered: promote soup3 iff POOLED paired t >= 2.0.
Reporting ranges only (dev); 5300+ reserves untouched.

Usage:
    python _soup_extend.py --blocks 2            # adds 21050-21079
    python _soup_extend.py --blocks 2 3          # adds both blocks
Incremental: writes after each (block, arm); safe to interrupt.
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

BLOCKS = {2: 21050, 3: 21080}  # block id -> seed0 (30 tapes each)


def paired(a: list[int], b: list[int]) -> dict:
    dl = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(dl)
    sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
    t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
    return {"delta": round(m, 1), "t": round(t, 2),
            "wins": sum(x > 0 for x in dl), "losses": sum(x < 0 for x in dl),
            "n": len(dl)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_soup_confirm.json"))
    ap.add_argument("--blocks", nargs="+", type=int, default=[2])
    args = ap.parse_args()

    op = Path(args.out)
    results = json.loads(op.read_text(encoding="utf-8"))
    results.setdefault("blocks", {})
    soup3 = D / "_soup_s42_s43_s44.pt"
    champ = D / "fullsplit_seed42.pt"
    ckpts = {"s42_champ": champ, "soup3": soup3}

    from evaluate import evaluate_weights_parallel

    for block in args.blocks:
        seed0 = BLOCKS[block]
        blk = results["blocks"].setdefault(str(block), {"seed0": seed0})
        for name in sorted(ckpts):
            if name in blk:
                print(f"block{block} {name}: done", flush=True)
                continue
            t0 = time.perf_counter()
            ev = evaluate_weights_parallel(
                str(ckpts[name]), games=30, seed=seed0, max_pieces=1500,
                workers=8)
            pieces = [g.pieces for g in ev.games]
            blk[name] = {"mean": round(statistics.fmean(pieces), 1),
                         "capped": sum(p >= 1500 for p in pieces),
                         "per_seed": pieces,
                         "sha": hashlib.sha256(
                             Path(ckpts[name]).read_bytes())
                         .hexdigest()[:16],
                         "wall_s": round(time.perf_counter() - t0, 1)}
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"block{block} {name:10s} mean {blk[name]['mean']:7.1f} "
                  f"cap {blk[name]['capped']:2d}/30 — SAVED", flush=True)

    # pooled: block 1 arms + all added blocks
    pooled = {"soup3": list(results["arms"]["soup3"]["per_seed"]),
              "s42_champ": list(results["arms"]["s42_champ"]["per_seed"])}
    for block in sorted(results["blocks"], key=int):
        blk = results["blocks"][block]
        pooled["soup3"].extend(blk["soup3"]["per_seed"])
        pooled["s42_champ"].extend(blk["s42_champ"]["per_seed"])
    pr = paired(pooled["soup3"], pooled["s42_champ"])
    pr["cap_soup3"] = sum(p >= 1500 for p in pooled["soup3"])
    pr["cap_s42"] = sum(p >= 1500 for p in pooled["s42_champ"])
    results["pooled_soup3_vs_s42"] = pr
    results["pooled_per_seed"] = pooled
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nPOOLED n={pr['n']}: soup3 - s42 {pr['delta']:+.1f} "
          f"t {pr['t']:+.2f} W/L {pr['wins']}/{pr['losses']} "
          f"cap {pr['cap_soup3']}/{pr['n']} vs {pr['cap_s42']}/{pr['n']}",
          flush=True)
    print("GATE:", "PROMOTE soup3" if pr["t"] >= 2.0 else
          "HOLD s42 — champion unchanged", flush=True)


if __name__ == "__main__":
    main()
