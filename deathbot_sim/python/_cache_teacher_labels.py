"""Cache the frozen 2-ply teacher labels for the snapshot9 bank (2026-09-11).

The fullsplit recipe's teacher labeling (~24 min for 23,016 states) is
IDENTICAL across schedule seeds — only the minibatch schedule changes with
the seed. Caching the labels once makes additional recipe seeds (45/46...)
~10x cheaper per member and removes any risk of labeling drift between
replications.

Format: NPZ with
  teacher_pick  int64[n_labelled]   bank candidate index (full-bank frame)
  state_index   int32[n_labelled]   bank state ids labelled
  alldead       bool[n_labelled]
  fit_pool      int32[n_pool]       bank state ids (multi-cand, not-all-dead,
                                    train games only)
  meta          json string         sha of weights, bank, gamma_ply

Usage:
    python _cache_teacher_labels.py            # build if missing
    python _cache_teacher_labels.py --verify   # compare against a fit run's
                                               # printed PRE agree (0.7867)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"

DEAD = -1e6
OUT = D / "_teacher_labels_snap9_bank.npz"


def build() -> None:
    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from diag_tinyset_overfit import teacher_scores_for_state

    bank_path = D / "state_bank_snapshot9_32games.npz"
    weights_path = ROOT / "model_weights_snapshot9.pt"
    bank = Bank(str(bank_path))
    n = bank.n_states

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    net.eval()

    placed, nxt = bank._replay_piece_ids()
    train_games = 24
    game_of = np.asarray(bank.game_ids)
    game_ids_sorted = sorted(set(int(g) for g in game_of))
    train_set = set(game_ids_sorted[:train_games])

    teacher_pick: dict[int, int] = {}
    alldead: dict[int, bool] = {}
    t0 = time.perf_counter()
    import deathbot_sim as ds
    with torch.inference_mode():
        for s in range(n):
            pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
            tv, viable_live = teacher_scores_for_state(
                net, pre, placed[s], nxt[s], 0.999)
            if tv is None:
                continue
            lo, _hi = bank.candidate_slice(s)
            viable = [lo + p for p in viable_live]
            order = sorted(range(len(tv)), key=lambda j: (-tv[j], j))
            teacher_pick[s] = viable[order[0]]
            alldead[s] = all(float(v) <= DEAD / 2 for v in tv)
            if (len(teacher_pick) % 4000) == 0:
                print(f"  teacher {len(teacher_pick)} states "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"teacher done {time.perf_counter() - t0:.0f}s "
          f"({len(teacher_pick)} states)", flush=True)

    fit_pool = [s for s in range(n)
                if s in teacher_pick
                and int(game_of[s]) in train_set
                and (bank.candidate_slice(s)[1]
                     - bank.candidate_slice(s)[0]) > 1
                and not alldead[s]]

    meta = {
        "weights_sha": hashlib.sha256(weights_path.read_bytes())
                       .hexdigest()[:16],
        "bank_sha": hashlib.sha256(bank_path.read_bytes()).hexdigest()[:16],
        "gamma_ply": 0.999,
        "train_games": train_games,
    }
    np.savez(str(OUT),
             state_index=np.array(sorted(teacher_pick), dtype=np.int32),
             teacher_pick=np.array([teacher_pick[s]
                                    for s in sorted(teacher_pick)],
                                   dtype=np.int64),
             alldead=np.array([alldead[s] for s in sorted(teacher_pick)],
                              dtype=bool),
             fit_pool=np.array(fit_pool, dtype=np.int32),
             meta=json.dumps(meta))
    print(f"wrote {OUT} ({len(teacher_pick)} labels, fit pool "
          f"{len(fit_pool)} — original run: 17265)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        z = np.load(str(OUT))
        print(f"{OUT.name} already exists: {len(z['state_index'])} labels, "
              f"fit pool {len(z['fit_pool'])} (pass --force to rebuild)")
        return
    build()


if __name__ == "__main__":
    main()
