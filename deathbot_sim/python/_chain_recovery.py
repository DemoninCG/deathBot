"""Chain launcher: run the recovery suite after loop cycle 1 completes.

The overnight self-distillation loop (_selfdistill_loop.json) is running.
The user asked for the randomized recovery suite to run AFTER cycle 1
completes (its burst + screen + adopt/revert land, dev seed 21150-21179
consumed) so the suite reads the post-burst champion state.

This driver polls _selfdistill_loop.json until cycle r1 has a screen
(i.e., cycle 1 completed), then launches the full recovery suite
(10 suites x 20 games x 2 arms, cap 1500, ~40-70 min) via subprocess.
It then exits — the loop keeps running its own cycles afterwards.

Notes:
  - The suite runs arms snapshot11 (champion) vs snapshot10 (fallback);
    snapshot9 can be added with --extra-arms.
  - The suite's start boards are fixed by master seed, so if cycle 1
    adopts its burst, the suite measures the NEW champion; if it reverts,
    it measures snapshot11 — either way the comparison vs snapshot10 is
    the recovery question the user asked about.
  - Sequential execution (user lesson 2026-09-11): nothing heavy runs
    concurrently.

Usage:
    python _chain_recovery.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"
LOOP_JSON = D / "_selfdistill_loop.json"
POLL_S = 60


def cycle1_done(results: dict) -> bool:
    c1 = results.get("cycles", {}).get("r1", {})
    return bool(c1.get("screen"))


def main() -> None:
    print("waiting for loop cycle 1 to complete "
          "(screen recorded)...", flush=True)
    while True:
        if LOOP_JSON.exists():
            try:
                results = json.loads(LOOP_JSON.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                results = {}
            if cycle1_done(results):
                c1 = results["cycles"]["r1"]
                print(f"cycle 1 done: screen {c1['screen']['pre_mean']} -> "
                      f"{c1['screen']['post_mean']} "
                      f"adopted={c1['screen']['adopted']}", flush=True)
                break
        time.sleep(POLL_S)

    # launch the full recovery suite
    # 2026-09-12 audit fix: measure the POST-cycle-1 working champion
    # (_loop_champ_r1.pt — cycle 1 adopted) alongside the lineage
    # endpoints. The old snap11-vs-snap10-only comparison measured the
    # pre-burst champion even after an adoption.
    loop_champ = D / "_loop_champ_r1.pt"
    weights = [str(ROOT / "model_weights_snapshot11.pt"),
               str(ROOT / "model_weights_snapshot10.pt")]
    names = ["snap11", "snap10"]
    if loop_champ.is_file():
        weights.append(str(loop_champ))
        names.append("loop_r1")
    cmd = [sys.executable, str(HERE / "_recovery_suite.py"),
           "--weights", *weights,
           "--names", *names,
           "--games", "20",
           "--out", str(D / "_recovery_suite.json")]
    print("launching recovery suite:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(HERE))
    print("recovery suite complete; loop continues independently",
          flush=True)


if __name__ == "__main__":
    main()