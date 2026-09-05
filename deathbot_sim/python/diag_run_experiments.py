"""Driver: faithful in-process training reproductions with per-round capture.

Each experiment runs ONE continuous training process (replay buffer, epsilon
decay, and RNG carried across rounds, as in a real session). Per-round
checkpoints are captured via the PER_ROUND_CAPTURE module-level hook that the
diag instrumentation added to selfplay.py / selfplay_refit.py /
_legacy/selfplay.py. After training, every captured round checkpoint is
greedy-evaluated on the standard eval seeds. Results go to JSON.

Experiments:
  E1_current         : current selfplay.py, defaults, resume snapshot7
  E2_legacy          : _legacy/selfplay.py (session-1/2 code), resume snapshot7
  E3_refit           : selfplay_refit.py (refit normalizer per round), resume snapshot7
  E4_current_scratch : current code, from-scratch init (control)

Usage:
    python diag_run_experiments.py --rounds 4 --games 25 --epochs 3
    python diag_run_experiments.py --experiments E1_current E2_legacy
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

EVAL_CODE = """
import sys, random, json
sys.path.insert(0, {here!r})
import torch
torch.set_num_threads(1)
from model import TAPNet
from neural_scorer import NeuralScorer
from selfplay import play_game
net = TAPNet(output='linear')
net.load_state_dict(torch.load({weights!r}, map_location='cpu'))
net.eval()
scorer = NeuralScorer(net=net, device='cpu')
out = []
for s in range({seed}, {seed} + {games}):
    traj, lines = play_game(scorer, random.Random(s), epsilon=0.0,
                            max_pieces={max_pieces})
    out.append(len(traj))
print(json.dumps(out))
"""


def eval_checkpoint(weights: Path, games: int, seed: int = 42,
                    max_pieces: int = 1500) -> list[int]:
    code = EVAL_CODE.format(here=str(HERE), weights=str(weights),
                            seed=seed, games=games, max_pieces=max_pieces)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       cwd=str(HERE), timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"eval failed for {weights}:\n{r.stderr[-1500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def run_training_subprocess(script: Path, argv: list[str], round_dir: Path,
                            log_path: Path) -> list[str]:
    """Run one experiment in an ISOLATED subprocess with the capture hook.

    Subprocess isolation is mandatory: the legacy and current python/ trees
    have same-named modules with different APIs (e.g. heuristic.py), and
    importing both into one interpreter poisons sys.modules. Runs in-process
    on one worker thread per experiment, torch pinned to 1 thread.
    """
    worker = HERE / "_capture_worker.py"
    cmd = [PY, str(worker), str(script), str(round_dir), *argv]
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.PIPE,
                              text=True, cwd=str(HERE), timeout=14400)
    err = proc.stderr or ""
    # The worker prints "CAPTURED n: [paths]" on stderr after training.
    captured: list[str] = []
    for line in err.splitlines():
        if line.startswith("CAPTURED "):
            import ast
            try:
                captured = ast.literal_eval(line.split(":", 1)[1].strip())
            except (SyntaxError, ValueError):
                pass
    if proc.returncode != 0 or not captured:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(
            f"training failed rc={proc.returncode}; captured={len(captured)}\n"
            f"stderr tail: {err[-600:]}\nlog tail: {tail}")
    return captured


def run_experiment(name: str, script: Path, extra_cli: list[str],
                   start_weights: Path, rounds: int, games: int, epochs: int,
                   eval_games: int, out_dir: Path,
                   legacy_lockout: bool = False) -> dict:
    # Unique per-run directory: reusing round dirs across runs let stale
    # rNN.pt files from an older experiment be evaluated as if current.
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    round_dir = out_dir / f"rounds_{name}_{run_stamp}"
    round_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"train_{name}_{run_stamp}.log"
    argv = ["--rounds", str(rounds), "--games", str(games),
            "--epochs", str(epochs), "--seed", "42",
            "--weights-in", str(start_weights),
            "--weights-out", str(out_dir / f"final_{name}_{run_stamp}.pt"),
            *extra_cli]
    if legacy_lockout and script.name == "selfplay.py":
        argv.append("--legacy-lockout")
    result: dict = {"name": name, "script": str(script), "argv": argv,
                    "start_weights": str(start_weights),
                    "run_stamp": run_stamp,
                    "round_dir": str(round_dir), "rounds": {}}
    t0 = time.perf_counter()
    try:
        run_training_subprocess(script, argv, round_dir, log_path)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{exc}\n{traceback.format_exc()[-1200:]}"
        return result
    result["train_s"] = round(time.perf_counter() - t0, 1)

    for rd in range(1, rounds + 1):
        ckpt = round_dir / f"r{rd:02d}.pt"
        if not ckpt.is_file():
            result["rounds"][str(rd)] = {"error": "checkpoint missing"}
            continue
        try:
            te = time.perf_counter()
            pieces = eval_checkpoint(ckpt, eval_games)
            result["rounds"][str(rd)] = {
                "eval_mean": round(statistics.fmean(pieces), 1),
                "eval_median": round(statistics.median(pieces), 1),
                "eval_min": min(pieces), "eval_max": max(pieces),
                "eval_s": round(time.perf_counter() - te, 1),
            }
        except Exception as exc:  # noqa: BLE001
            result["rounds"][str(rd)] = {"error": str(exc)[:400]}
        em = result["rounds"][str(rd)].get("eval_mean", "ERR")
        print(f"  [{name}] round {rd}: eval_mean={em}", flush=True)
    return result


def resolve_weights(path_str: str) -> Path:
    """Resolve a weights path like the training scripts do, but LOUDLY.

    Tries: as-given (absolute or cwd-relative), then relative to the crate
    root, then relative to the workspace root. Raises if none exist — a
    silent from-scratch fallback would invalidate the experiment.
    """
    supplied = Path(path_str).expanduser()
    candidates = [supplied, HERE.parent / supplied,
                  HERE.parent.parent / supplied]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"weights not found (tried {len(candidates)} locations): {path_str}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4,
                    help="experiments in parallel; each pins torch to 1 thread")
    ap.add_argument("--weights-in", default="model_weights_snapshot7.pt")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="subset: E1_current E2_legacy E3_refit E4_current_scratch")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: _diag_out/experiments_<stamp>.json "
                         "— a fresh file per run, so old results are never "
                         "overwritten)")
    ap.add_argument("--legacy-lockout", action="store_true",
                    help="pass --legacy-lockout to current-code experiments "
                         "(historical game-over semantics)")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        args.out = f"../_diag_out/experiments_{stamp}.json"
    out_file = (HERE / args.out).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_file.parent
    snap7 = resolve_weights(args.weights_in)
    print(f"Resume weights: {snap7}")
    empty = out_dir / "empty_init.pt"
    if not empty.is_file():
        code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); "
                "import torch; from model import TAPNet; "
                "torch.manual_seed(0); "
                f"torch.save(TAPNet(output='linear').state_dict(), {str(empty)!r})")
        subprocess.run([PY, "-c", code], check=True, cwd=str(HERE))

    defs = {
        # NOTE: E3_refit (selfplay_refit.py) and E7_raw_targets
        # (selfplay_rawnorm.py) referenced modules deleted during the
        # 2026-09-05 cleanup; their archived results remain in _diag_out/.
        # Current-code experiments below all support the A2/A3/A4 fixes.
        "E1_current": (HERE / "selfplay.py", [], snap7),
        "E2_legacy": (HERE / "_legacy" / "selfplay.py", [], snap7),
        "E4_current_scratch": (HERE / "selfplay.py", [], empty),
        "E5_legacy_mp500": (HERE / "_legacy" / "selfplay.py",
                            ["--max-pieces", "500"], snap7),
        "E6_legacy_bigbuffer": (HERE / "_legacy" / "selfplay.py",
                                ["--replay-capacity", "400000"], snap7),
        "E8_legacy_fullscale": (HERE / "_legacy" / "selfplay.py", [], snap7),
        "E9_corrected_lockout": (HERE / "selfplay.py",
                                 [], snap7),
    }
    names = args.experiments or list(defs)
    jobs = [(name, defs[name][0], defs[name][1], defs[name][2])
            for name in names]

    print(f"{len(jobs)} experiments x {args.rounds} rounds | "
          f"{args.games} games/round | {args.epochs} epochs/round | "
          f"eval {args.eval_games} seeds | {args.workers} workers",
          flush=True)

    all_results: dict[str, dict] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(run_experiment, name, script, extra, start,
                        args.rounds, args.games, args.epochs,
                        args.eval_games, out_dir,
                        args.legacy_lockout): name
            for name, script, extra, start in jobs
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                all_results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001
                all_results[name] = {"error": str(exc)[:800]}
            print(f"=== {name} done", flush=True)

    print(f"\n=== eval mean pieces per round (baseline: "
          f"{args.weights_in} scored ~503 on 30 seeds) ===")
    for name in names:
        res = all_results.get(name, {"error": "missing"})
        if "error" in res and "rounds" not in res:
            print(f"  {name:20s} ERROR: {res['error'].splitlines()[0]}")
            continue
        cells = []
        for rd in range(1, args.rounds + 1):
            rd_res = res.get("rounds", {}).get(str(rd), {})
            v = rd_res.get("eval_mean", "ERR")
            cells.append(f"r{rd}={v}")
        print(f"  {name:20s} " + "  ".join(cells))

    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_file} (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
