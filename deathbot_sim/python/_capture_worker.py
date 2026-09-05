"""Subprocess shim: run ONE training experiment with the capture hook.

Invoked by diag_run_experiments.py so each experiment gets its own process
(and thus its own sys.modules — the legacy and current python/ packages
conflict when imported into the same interpreter).

Usage:
    python _capture_worker.py <script> <round_dir> <extra json args...>
The caller passes normal selfplay CLI args after the two paths. The final
captured-round checkpoints are written to <round_dir>/rNN.pt.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


def main() -> None:
    script = Path(sys.argv[1])
    round_dir = Path(sys.argv[2])
    argv = sys.argv[3:]
    round_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    captured: list[str] = []

    def capture(round_idx: int, net) -> None:
        path = round_dir / f"r{round_idx + 1:02d}.pt"
        torch.save(net.state_dict(), path)
        captured.append(str(path))

    g = {"__name__": "__main__", "__file__": str(script),
         "PER_ROUND_CAPTURE": capture}
    old_argv = sys.argv
    sys.argv = [str(script), *argv]
    try:
        runpy.run_path(str(script), run_name="__main__", init_globals=g)
    finally:
        sys.argv = old_argv
    print(f"CAPTURED {len(captured)}: {captured}", file=sys.stderr)


if __name__ == "__main__":
    main()
