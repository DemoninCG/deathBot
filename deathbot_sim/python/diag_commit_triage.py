"""Rung-2 triage: usability-gated re-entry + controller-λ sweep (§41 fix).

Stage A (cheap triage, TRAINING seeds 26000-26007, reporting only):
  wide variant grid, 8 games each at cap 300. Variants span three axes:
    reentry gate : occ-only (legacy) vs top-gated vs usable-channel vs
                   usable+cooldown
    cooldown     : 0 / 30 / 60 pieces after RECOVER exit
    lam-in-commit: 0 (pure commit) vs 10 (constraint + gradient combo)
  Metric: (T/100, cap-rate) Pareto movement vs commit_loose baseline.
  NO panels consumed (training seeds = reporting only, same status as
  the 20400 teacher bank).

Stage B (promotion, FRESH panel, only for ≤2 winners): cap-300 paired
screen vs commit_loose + lam10 on the next free dev range.

Controller-λ note: lam inside commit biases WITHIN the preserved set
(constraint sets admissibility, gradient orders inside it). Mechanically
distinct from pre-constraint scalar λ, which also re-ranks abandoning
moves — the combo should hold style with fewer fallbacks, at the risk of
re-creating λ-style over-eager digging inside the preserved set.

Usage:
    python diag_commit_triage.py --weights ../../model_weights_snapshot9.pt \\
        --out ../_diag_out/commit_triage.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TRAIN_SEEDS = list(range(26000, 26008))
CAP = 300

# (name, kwargs for play_game_commit). Baseline first.
VARIANTS: list[tuple[str, dict]] = [
    ("base_loose", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60}),
    ("top10", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "reenter_top": 10}),
    ("top8", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
              "reenter_top": 8}),
    ("cool30", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                "cooldown": 30}),
    ("cool60", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                "cooldown": 60}),
    ("top10_cool30", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                      "reenter_top": 10, "cooldown": 30}),
    ("lam10", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "lam": 10.0}),
    ("lam10_top10", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                     "reenter_top": 10, "lam": 10.0}),
    ("lam20", {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "lam": 20.0}),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(Path(args.weights).expanduser())
    for cand in (Path(weights), HERE.parent / args.weights,
                 HERE.parent.parent / args.weights):
        if cand.is_file():
            weights = str(cand.resolve())
            break
    from model import TAPNet
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    from commit_controller import play_game_commit
    import random

    t0 = time.perf_counter()
    out = {"weights": weights,
           "weights_sha256": hashlib.sha256(
               Path(weights).read_bytes()).hexdigest(),
           "seeds": TRAIN_SEEDS, "cap": CAP, "variants": {}}
    print(f"{'variant':>14s} {'cap':>5s} {'pcs':>7s} {'tet':>4s} "
          f"{'T/100':>6s} {'off':>6s} {'rec':>5s} {'denT':>5s} {'denC':>5s}")
    for name, kw in VARIANTS:
        tets = pcs = offs = 0
        recs = denT = denC = 0
        capped = 0
        for seed in TRAIN_SEEDS:
            pieces, _lines, clears, reason, stats = play_game_commit(
                net, random.Random(seed), max_pieces=CAP, **kw)
            pcs += pieces
            tets += int(clears.get(4, 0))
            capped += int(pieces >= CAP)
            offs += stats["style_off"] / max(1, stats["decisions"])
            recs += stats["recovers"]
            denT += stats.get("reentry_denied_top", 0)
            denC += stats.get("reentry_denied_cool", 0)
        n = len(TRAIN_SEEDS)
        t100 = 100 * tets / max(1, pcs)
        row = {"capped": capped, "mean_pieces": round(pcs / n, 1),
               "tetrises": tets, "t_per_100": round(t100, 3),
               "style_off_frac": round(offs / n, 3),
               "mean_recovers": round(recs / n, 2),
               "mean_denied_top": round(denT / n, 2),
               "mean_denied_cool": round(denC / n, 2), "kwargs": kw}
        out["variants"][name] = row
        print(f"{name:>14s} {capped:3d}/{n:<3d} {pcs / n:7.1f} {tets:4d} "
              f"{t100:6.3f} {offs / n:6.3f} {recs / n:5.1f} "
              f"{denT / n:5.1f} {denC / n:5.1f}", flush=True)
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {op} in {out['wall_seconds']}s "
          f"(training seeds only — no panels consumed)")


if __name__ == "__main__":
    main()
