"""Instrumented short training runs to localize the self-play collapse.

Runs several configurations side by side (one subprocess each, 1 torch
thread). Each experiment is a CHAIN of rounds: round r resumes from round
r-1's checkpoint, exactly like a real training session. After every round
the checkpoint is greedy-evaluated on the standard eval seeds.

  E1 current-defaults : current selfplay.py defaults, resume snapshot7
  E2 legacy-code      : _legacy/selfplay.py (session-1 code), resume snapshot7
  E3 no-heuristic     : current code, pure survival target
  E4 fresh-init       : current code, training from scratch (control)

Per-round results (mean/median/max pieces, buffer size, target stats, loss)
are written to a JSON file.

Usage:
    python diag_train_probe.py --rounds 3 --games 25 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def eval_checkpoint(weights: Path, games: int, seed: int,
                    max_pieces: int) -> list[int]:
    """Greedy-eval a checkpoint in a fresh subprocess; return pieces per game."""
    code = f"""
import sys, random, json
sys.path.insert(0, {str(HERE)!r})
import torch
torch.set_num_threads(1)
from model import TAPNet
from neural_scorer import NeuralScorer
from selfplay import play_game
net = TAPNet(output='linear')
net.load_state_dict(torch.load({str(weights)!r}, map_location='cpu'))
net.eval()
scorer = NeuralScorer(net=net, device='cpu')
out = []
for s in range({seed}, {seed} + {games}):
    traj, lines = play_game(scorer, random.Random(s), epsilon=0.0,
                            max_pieces={max_pieces})
    out.append(len(traj))
print(json.dumps(out))
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       cwd=str(HERE), timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"eval failed for {weights}:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def train_one_round(script: Path, cli: list[str], log_path: Path) -> None:
    """Run one round of training in a subprocess, logging output."""
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run([PY, str(script), *cli],
                              cwd=str(script.parent), stdout=logf,
                              stderr=subprocess.STDOUT, timeout=7200)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"train failed ({script.name} {' '.join(cli)}):\n{tail}")


def parse_round_stats(log_path: Path) -> dict:
    """Extract survival/target/loss stats from a training log."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    m = re.search(r"Playing (\d+) games", text)
    if m:
        out["games"] = int(m.group(1))
    m = re.search(r"pieces: min=(\d+)\s+avg=([\d.]+)\s+max=(\d+)", text)
    if m:
        out["train_min"] = int(m.group(1))
        out["train_avg"] = float(m.group(2))
        out["train_max"] = int(m.group(3))
    m = re.search(r"Buffer: (\d+)", text)
    if m:
        out["buffer"] = int(m.group(1))
    m = re.search(r"Targets\s+[-\u2014] min=([-\d.]+) avg=([-\d.]+) max=([-\d.]+)", text)
    if m:
        out["target_min"] = float(m.group(1))
        out["target_avg"] = float(m.group(2))
        out["target_max"] = float(m.group(3))
    # Legacy (session-1) log format: "Target stats: min=.. max=.. mean=.."
    m = re.search(r"Target stats: min=([-\d.]+)\s+max=([-\d.]+)\s+mean=([-\d.]+)", text)
    if m:
        out["target_min"] = float(m.group(1))
        out["target_max"] = float(m.group(2))
        out["target_avg"] = float(m.group(3))
    losses = re.findall(r"epoch \d+/\d+: loss=([\d.]+)", text)
    if losses:
        out["loss"] = [float(x) for x in losses]
    m = re.search(r"epsilon=([\d.]+)", text)
    if m:
        out["epsilon"] = float(m.group(1))
    m = re.search(r"survival mean=([-\d.]+) std=([\d.]+)", text)
    if m:
        out["norm_survival_mean"] = float(m.group(1))
        out["norm_survival_std"] = float(m.group(2))
    return out


def run_experiment(name: str, script: Path, base_cli: list[str],
                   rounds: int, eval_games: int, out_dir: Path,
                   start_weights: Path) -> dict:
    """Chain `rounds` rounds of training, evaluating after each."""
    ckpt_dir = out_dir / f"rounds_{name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"name": name, "script": str(script), "base_cli": base_cli,
                    "rounds": {}}
    current_weights = start_weights
    t0 = time.perf_counter()
    for rd in range(1, rounds + 1):
        out_ckpt = ckpt_dir / f"r{rd}.pt"
        log = out_dir / f"train_{name}_r{rd}.log"
        cli = [*base_cli, "--weights-in", str(current_weights),
               "--weights-out", str(out_ckpt), "--rounds", "1"]
        try:
            train_one_round(script, cli, log)
        except Exception as exc:  # noqa: BLE001
            result["rounds"][str(rd)] = {"error": str(exc)[:400]}
            break
        stats = parse_round_stats(log)
        try:
            pieces = eval_checkpoint(out_ckpt, eval_games, 42, 1500)
            stats.update({
                "eval_mean": round(statistics.fmean(pieces), 1),
                "eval_median": round(statistics.median(pieces), 1),
                "eval_min": min(pieces), "eval_max": max(pieces),
            })
        except Exception as exc:  # noqa: BLE001
            stats["eval_error"] = str(exc)[:300]
        result["rounds"][str(rd)] = stats
        current_weights = out_ckpt
        em = stats.get("eval_mean", float("nan"))
        print(f"  [{name}] round {rd}/{rounds}: eval_mean={em} "
              f"train_avg={stats.get('train_avg')} "
              f"buffer={stats.get('buffer')} ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    result["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4,
                    help="experiments run in parallel (each 1 torch thread)")
    ap.add_argument("--weights-in", default="model_weights_snapshot7.pt")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="subset: E1 E2 E3 E4")
    ap.add_argument("--out", default="../_diag_out/train_probe.json")
    args = ap.parse_args()

    out_dir = (HERE / args.out).resolve().parent
    out_file = HERE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    snap7 = (HERE.parent / args.weights_in).resolve()
    legacy = HERE / "_legacy" / "selfplay.py"
    current = HERE / "selfplay.py"

    defs = {
        "E1_current_defaults": (current, []),
        "E2_legacy_code": (legacy, []),
        "E3_no_heuristic": (current, ["--survival-weight", "1",
                                      "--heuristic-weight", "0"]),
        "E4_fresh_init": (current, []),
    }
    names = args.experiments or list(defs)
    experiments = []
    for name in names:
        script, extra = defs[name]
        start = (out_dir / "empty_init.pt") if name == "E4_fresh_init" else snap7
        experiments.append((name, script, extra, start))

    if not (out_dir / "empty_init.pt").is_file():
        code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); "
                "import torch; from model import TAPNet; "
                "torch.manual_seed(0); "
                f"torch.save(TAPNet(output='linear').state_dict(), "
                f"{str(out_dir / 'empty_init.pt')!r})")
        subprocess.run([PY, "-c", code], check=True, cwd=str(HERE))

    print(f"{len(experiments)} experiments x {args.rounds} rounds, "
          f"{args.games} games/round, {args.epochs} epochs/round, "
          f"eval on {args.eval_games} seeds, {args.workers} parallel.",
          flush=True)

    all_results: dict[str, dict] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(run_experiment, name, script,
                        ["--games", str(args.games),
                         "--epochs", str(args.epochs), "--seed", "42",
                         *extra],
                        args.rounds, args.eval_games, out_dir, start): name
            for name, script, extra, start in experiments
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                all_results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001
                all_results[name] = {"error": str(exc)[:500]}
            print(f"=== {name} done ({all_results[name].get('elapsed_s', '?')}s)",
                  flush=True)

    # Final summary table.
    print("\n=== Per-round eval (mean pieces) ===")
    hdr = "experiment".ljust(24) + "".join(f"r{r}".rjust(10)
                                           for r in range(1, args.rounds + 1))
    print(hdr)
    for name, res in all_results.items():
        row = name.ljust(24)
        for r in range(1, args.rounds + 1):
            rd = res.get("rounds", {}).get(str(r), {})
            v = rd.get("eval_mean", "-")
            row += f"{v!s:>{10}}"
        print(row)

    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_file} (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
