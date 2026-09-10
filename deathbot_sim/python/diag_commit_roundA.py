"""Reporting-only Round A for controller ideas after corrected cool30lam10.

All cells retain the corrected masked-root expansion. They isolate controller
state changes on matched reporting tapes; this is a rough triage, not a
promotion experiment. Never use a seed below REPORTING_SEED0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPORTING_SEED0 = 31000
ARMS = {
    "baseline": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                 "cooldown": 30, "lam": 10.0},
    "reenter_top10": {"abort_top": 16, "abort_occ": 110,
                        "reenter_occ": 60, "reenter_top": 10,
                        "cooldown": 30, "lam": 10.0},
    "debt30": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "cooldown": 30, "lam": 10.0, "debt_limit": 30.0},
    "debt80": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "cooldown": 30, "lam": 10.0, "debt_limit": 80.0},
    "watch80": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                "cooldown": 30, "lam": 10.0, "progress_patience": 80},
    "uniform": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                "cooldown": 30, "lam": 10.0,
                "uniform_commit_scoring": True},
    "cashout_q": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
                  "cooldown": 30, "lam": 10.0, "rank_cashouts": True},
    "reentry_cert": {"abort_top": 16, "abort_occ": 110,
                      "reenter_occ": 60, "reenter_top": 10,
                      "reenter_holes": 2, "reentry_min_progress": 1.0,
                      "cooldown": 30, "lam": 10.0},
    "ready8": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
               "cooldown": 30, "lam": 10.0, "ready_patience": 8},
    "debt30_top10": {"abort_top": 16, "abort_occ": 110,
                      "reenter_occ": 60, "reenter_top": 10,
                      "cooldown": 30, "lam": 10.0, "debt_limit": 30.0},
}


def resolve(path: str) -> Path:
    supplied = Path(path).expanduser()
    for candidate in (supplied, HERE.parent / supplied, HERE.parent.parent / supplied,
                      HERE.parent.parent.parent / supplied):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _worker(payload) -> dict:
    import os
    import random

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    here, weights, arm, seed, cap = payload
    sys.path.insert(0, here)
    from commit_controller import play_game_commit
    from model import TAPNet

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    start = time.perf_counter()
    pieces, lines, clears, reason, stats = play_game_commit(
        net, random.Random(seed), max_pieces=cap, **ARMS[arm])
    return {"arm": arm, "seed": seed, "pieces": pieces, "lines": lines,
            "clears": {str(key): int(value) for key, value in clears.items()},
            "reason": reason, "seconds": round(time.perf_counter() - start, 2),
            "stats": stats}


def _paired_tetrises(rows, baseline) -> tuple[float, float]:
    deltas = [int(row["clears"].get("4", 0))
              - int(baseline[row["seed"]]["clears"].get("4", 0))
              for row in rows]
    mean = statistics.fmean(deltas)
    if len(deltas) < 2:
        return mean, 0.0
    sd = statistics.stdev(deltas)
    return mean, mean / (sd / len(deltas) ** 0.5) if sd else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--seed0", type=int, default=32000)
    parser.add_argument("--seeds", default="",
                        help="comma-separated explicit reporting replay seeds")
    parser.add_argument("--cap", type=int, default=300)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--allow-replay-seeds", action="store_true",
                        help="permit known prior-panel tapes as reporting replays")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="comma-separated subset of: " + ",".join(ARMS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seeds = ([int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
             if args.seeds else list(range(args.seed0, args.seed0 + args.games)))
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    if min(seeds) < REPORTING_SEED0 and not args.allow_replay_seeds:
        parser.error(f"Round A is reporting-only; seed0 must be >= {REPORTING_SEED0}")
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if not arms or unknown:
        parser.error("unknown Round A arm(s): " + ", ".join(unknown))

    weights = str(resolve(args.weights))
    payloads = [(str(HERE), weights, arm, seed, args.cap)
                for arm in arms for seed in seeds]
    results: dict[str, list[dict]] = {arm: [] for arm in arms}
    start = time.perf_counter()
    print(f"[roundA] reporting-only: seeds {seeds[0]}-{seeds[-1]}, arms={arms}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results[row["arm"]].append(row)
            if completed % len(arms) == 0:
                print(f"  {completed}/{len(payloads)} ({time.perf_counter() - start:.0f}s)",
                      flush=True)

    if "baseline" not in results:
        parser.error("the selected arms must include baseline")
    baseline = {row["seed"]: row for row in results["baseline"]}
    output = {"weights": weights,
              "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
              "seed0": args.seed0, "seeds": seeds, "games": len(seeds), "cap": args.cap,
              "reporting_only": True,
              "replay_of_prior_tapes": bool(args.allow_replay_seeds),
              "arms": {}}
    print(f"\n{'arm':>14} {'cap':>7} {'pcs':>7} {'tet':>5} {'T/100':>6} {'dTet':>7} {'t':>6} {'exit':>9}")
    for arm in arms:
        rows = sorted(results[arm], key=lambda row: row["seed"])
        pieces = [row["pieces"] for row in rows]
        tetrises = sum(int(row["clears"].get("4", 0)) for row in rows)
        delta, t_value = _paired_tetrises(rows, baseline)
        caps = sum(piece >= args.cap for piece in pieces)
        exit_count = sum(row["stats"].get("debt_exits", 0)
                         + row["stats"].get("progress_exits", 0) for row in rows)
        print(f"{arm:>14} {caps:3}/{len(rows):<3} {statistics.fmean(pieces):7.1f} "
              f"{tetrises:5d} {100 * tetrises / max(1, sum(pieces)):6.3f} "
              f"{delta:+7.2f} {t_value:+6.2f} {exit_count:9d}")
        output["arms"][arm] = {"params": ARMS[arm], "capped": caps,
                                 "mean_pieces": round(statistics.fmean(pieces), 2),
                                 "tetrises": tetrises,
                                 "t_per_100": round(100 * tetrises / max(1, sum(pieces)), 4),
                                 "paired_dtet_vs_baseline": round(delta, 4),
                                 "paired_t_vs_baseline": round(t_value, 4),
                                 "controller_exits": exit_count,
                                 "per_seed": rows}
    output["wall_seconds"] = round(time.perf_counter() - start, 2)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {target} in {output['wall_seconds']}s")


if __name__ == "__main__":
    main()