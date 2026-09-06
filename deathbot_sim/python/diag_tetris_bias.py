"""Section 2 screen: Tetris-bias line_bonus sweep on full 2-ply (cap 300, fresh 20134+).

Design (matches the agreed plan):
- Full-expansion 2-ply (beam=100), snapshot9, corrected lockout.
- line_bonus applies to SURVIVING branches only (two_ply.py); dying/lockout
  penalties untouched -> survival ordering preserved by construction.
- Bonus scale: within-state v1 spread ~10.5 (bank probe), so k=5/15/40 spans
  sub-margin / decisive / dominant. Guideline-shaped {1:k/8, 2:3k/8, 3:5k/8,
  4:k} rewards Tetrises most.
- Panel: 30 fresh games 20134-20163, cap 300 (every game caps; style carries
  the signal). Paired per-seed vs bonus-free beam-100 on the SAME seeds.
- Safety gate: cap rate must stay 30/30; any non-capped game = regression.
"""
import sys, json, statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

W = str(Path(r"C:\My Things\HTML\Demonin's Item Shop\other\deathBot\model_weights_snapshot9.pt"))
SEED0, GAMES, CAP = 20134, 30, 300

ARMS = {
    "base": None,
    "k5": {1: 5/8, 2: 15/8, 3: 25/8, 4: 5.0},
    "k15": {1: 15/8, 2: 45/8, 3: 75/8, 4: 15.0},
    "k40": {1: 5.0, 2: 15.0, 3: 25.0, 4: 40.0},
}

def worker(payload):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    here, weights, arm, bonus, seed, cap = payload
    sys.path.insert(0, here)
    import random
    from model import TAPNet
    from two_ply import play_game_2ply_style
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    import time
    t0 = time.perf_counter()
    pieces, lines, clears, reason = play_game_2ply_style(
        net, random.Random(seed), beam=100, line_bonus=bonus, max_pieces=cap)
    return {"arm": arm, "seed": seed, "pieces": pieces, "lines": lines,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "s": time.perf_counter() - t0}

def main():
    payloads = [(str(HERE), W, arm, bonus, SEED0 + i, CAP)
                for arm, bonus in ARMS.items() for i in range(GAMES)]
    out = {}
    import time
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for fut in as_completed([pool.submit(worker, p) for p in payloads]):
            r = fut.result()
            out.setdefault(r["arm"], []).append(r)
    print(f"wall {time.perf_counter()-t0:.0f}s")
    res = {"weights": W, "seed0": SEED0, "games": GAMES, "max_pieces": CAP,
           "arms": {}, "wall_s": round(time.perf_counter()-t0, 1)}
    base_lines = {r["seed"]: r["lines"] for r in out["base"]}
    print(f"{'arm':>5s} {'pieces':>7s} {'lines':>7s} {'l/100':>6s} "
          f"{'tetris':>6s} {'tshr':>6s} {'cap':>4s} {'s/g':>6s} | paired dLines t")
    for arm in ARMS:
        rows = sorted(out[arm], key=lambda r: r["seed"])
        pieces = [r["pieces"] for r in rows]
        lines = [r["lines"] for r in rows]
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        scoring = sum(int(r["clears"].get(k, 0)) for r in rows for k in ("1","2","3","4"))
        tshr = tet / scoring if scoring else 0.0
        l100 = 100.0 * sum(lines) / max(1, sum(pieces))
        cap = sum(1 for p in pieces if p >= CAP)
        dl = [r["lines"] - base_lines[r["seed"]] for r in rows] if arm != "base" else [0]*len(rows)
        if arm == "base":
            extra = "—"
        else:
            md = statistics.fmean(dl); sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            se = sd / (len(dl) ** 0.5)
            extra = f"{md:+.1f} t{md/se:+.2f}" if se else f"{md:+.1f} t—"
        print(f"{arm:>5s} {statistics.fmean(pieces):7.1f} {statistics.fmean(lines):7.1f} "
              f"{l100:6.1f} {tet:6d} {tshr:6.3f} {cap:4d} "
              f"{statistics.fmean([r['s'] for r in rows]):6.1f} | {extra}")
        res["arms"][arm] = {
            "bonus": ARMS[arm],
            "mean_pieces": statistics.fmean(pieces),
            "mean_lines": statistics.fmean(lines),
            "lines_per_100": l100,
            "tetrises": tet, "tetris_share": tshr, "capped": cap,
            "paired_dlines": dl,
            "per_seed": {str(r["seed"]): r["pieces"] for r in rows},
            "per_seed_lines": {str(r["seed"]): r["lines"] for r in rows},
            "per_seed_clears": {str(r["seed"]): r["clears"] for r in rows},
        }
    op = HERE.parent / "_diag_out" / "two_ply_tetris_k.json"
    op.write_text(json.dumps(res, indent=2))
    print(f"Wrote {op}")

if __name__ == "__main__":
    main()
