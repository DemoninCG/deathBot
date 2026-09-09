"""E3 dev-panel gate: 1-ply paired play, start vs armA vs armB (§32.3 E3).

All three checkpoints play the SAME fresh dev tapes (default 20632+,
next free after E1's 20612-20631) at cap 1500, epsilon=0, corrected
lockout, sweep harness (play_game_detailed, aliased convention — SAME
harness all arms, so the comparison is valid per E0 §33).

Gate: an arm must beat the START checkpoint (fullsplit_seed42, the
promoted 1-ply champion) on paired pieces with a convincing effect at
n=20 (t >= ~2), OR show a clear cap-rate separation, to justify a
fresh-panel final. Otherwise E3 closes (the step does not compound).

Usage:
    python diag_e3_gate.py --seed0 20632 --games 20 --cap 1500 \\
        --out ../_diag_out/e3_gate.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

E3_SEED0 = 20632  # next free dev after E1 (20612-20631 CONSUMED)
REPORTING_SEED0 = 31200


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"not found: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="../_diag_out/fullsplit_seed42.pt")
    ap.add_argument("--arma", default="../_diag_out/e3_round.armA.pt")
    ap.add_argument("--armb", default="../_diag_out/e3_round.armB.pt")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset to play: any of start armA armB "
                         "(default: all three)")
    ap.add_argument("--seed0", type=int, default=E3_SEED0)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from model import TAPNet
    from neural_scorer import NeuralScorer
    from selfplay import play_game_detailed

    arms = [("start", resolve(args.start)), ("armA", resolve(args.arma)),
            ("armB", resolve(args.armb))]
    if args.arms:
        want = set(args.arms)
        arms = [(n, p) for (n, p) in arms if n in want]
        if "start" not in [n for n, _ in arms]:
            print("[E3gate] start is always included as the paired baseline.")
            sys.exit(2)
    nets = {}
    for name, path in arms:
        net = TAPNet(output="linear")
        net.load_state_dict(torch.load(str(path), map_location="cpu"))
        net.eval()
        nets[name] = NeuralScorer(net=net, device="cpu")
    seeds = [args.seed0 + i for i in range(args.games)]
    if args.seed0 < E3_SEED0:
        print(f"[E3gate] seed0 {args.seed0} < {E3_SEED0} — refusing "
              f"(registry rules).")
        sys.exit(2)
    tag = "reporting-only" if args.seed0 >= REPORTING_SEED0 else \
        f"SELECTION dev panel {seeds[0]}-{seeds[-1]} (CONSUMED)"
    print(f"[E3gate] {tag}.")

    t0 = time.perf_counter()
    rows: dict[str, list] = {name: [] for name, _ in arms}
    for seed in seeds:
        for name, _ in arms:
            r = play_game_detailed(nets[name], random.Random(seed),
                                   epsilon=0.0, max_pieces=args.cap)
            rows[name].append({"seed": seed, "pieces": r.pieces_placed,
                               "reason": r.reason})
        print(f"  seed {seed}: " + " / ".join(
            f"{n}={next(x['pieces'] for x in rows[n] if x['seed']==seed)}"
            for n, _ in arms), flush=True)

    def paired(a, b):
        da = {r["seed"]: r["pieces"] for r in a}
        db = {r["seed"]: r["pieces"] for r in b}
        d = [db[s] - da[s] for s in da]
        m = statistics.fmean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
        return m, (m / se if se else 0.0), d

    out = {"seed0": args.seed0, "games": args.games, "cap": args.cap,
           "selection_panel": args.seed0 < REPORTING_SEED0,
           "arms": {}, "wall_seconds": 0.0}
    print(f"\n{'arm':>6s} {'mean':>7s} {'med':>6s} {'cap':>7s}")
    for name, path in arms:
        pcs = [r["pieces"] for r in rows[name]]
        print(f"{name:>6s} {statistics.fmean(pcs):7.1f} "
              f"{statistics.median(pcs):6.0f} "
              f"{sum(1 for p in pcs if p >= args.cap):3d}/{len(pcs):<3d}")
        out["arms"][name] = {
            "weights": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "mean_pieces": round(statistics.fmean(pcs), 1),
            "median_pieces": statistics.median(pcs),
            "capped": sum(1 for p in pcs if p >= args.cap),
            "per_seed": {str(r["seed"]): {"pieces": r["pieces"],
                                          "reason": r["reason"]}
                         for r in rows[name]}}
    for other in [n for n, _ in arms if n != "start"]:
        m, t, d = paired(rows["start"], rows[other])
        w = sum(1 for x in d if x > 0)
        l = sum(1 for x in d if x < 0)
        print(f"{other} vs start: {m:+.1f} t{t:+.2f} W/L {w}/{l}")
        key = {"armA": "armA_vs_start", "armB": "armB_vs_start"}[other]
        out[key] = {"paired_d": round(m, 1),
                    "paired_t": round(t, 2),
                    "W": w, "L": l, "deltas": d}
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {op} in {out['wall_seconds']}s")


if __name__ == "__main__":
    main()
