"""E4 λ10 risk work (§32.3 E4): failure/recovery bank + rescue test + hysteresis.

Question (§32.1.8): does the survival policy RECOVER from λ10-generated
states, or only avoid reaching them? That determines whether a rescue
controller is even possible.

Protocol (NO panels, NO training — all reporting/training seeds):
  1. FAILURE/RECOVERY BANK: replay the λ10 31207 death tape (the only λ10
     death in 40 cap-3000 games) under λ10, snapshotting pre-boards every
     100 pieces from 2000 on (+ the 31207 champ-tape states as controls).
     Also snapshot the top-3 cap-1500 spike games (highest end-occ) at
     their peak windows. Bank = packed pre-boards + (placed, preview) ids
     + occ/top/holes + trajectory position. Written to
     ../_diag_out/lam10_risk_bank.npz.
  2. RESCUE TEST: from each bank state, play the SURVIVAL policy
     (snapshot9 full 2-ply, bonus-free) to cap 1500 or death, on the TRUE
     continuation tape (same env-RNG state the game had — reconstructed by
     replaying the seed to that position, then switching scorer). Report
     per-state: survival pieces after handoff, reason, occ trajectory.
     Control: same handoff with the λ10 scorer (does λ10 itself survive
     from its own states?).
  3. HYSTERESIS GRID (<=4 params): entry/exit occupancy thresholds ×
     lambda scale × bonus toggle, evaluated as closed-loop policies on
     training seeds 26000-26007 (reporting only): T/100 vs failures vs
     style-off fraction. Pareto report, no promotion.

Usage:
    python diag_lam10_risk.py --weights ../../model_weights_snapshot9.pt \\
        --out ../_diag_out/lam10_risk.json
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

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
DEATH_SEED = 31207
SPIKE_SEEDS_1500 = (31207,)  # death tape doubles as the spike source
SNAP_EVERY = 100
SNAP_FROM = 2000
RESCUE_CAP = 1500
TRAIN_SEEDS = list(range(26000, 26008))


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"not found: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--bank-out", default="../_diag_out/lam10_risk_bank.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_all = time.perf_counter()
    weights = str(resolve(args.weights))
    from model import TAPNet, piece_to_index
    from two_ply import TwoPlyScorer
    from well_potential import style_fn_factory, dense_well_features
    from piece_history import PieceQueue
    import deathbot_sim as ds

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    surv = TwoPlyScorer(net, beam=10 ** 9)
    lam10 = TwoPlyScorer(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS),
                         style_fn=style_fn_factory(scale=float(args.lam)),
                         style_alpha=1.0)

    def occ_of(board):
        return sum(bin(int(v)).count("1") for v in board.row_bits())

    # ---- 1. bank: replay λ10 death tape, snapshot states ----
    snaps: list[dict] = []
    for seed, scorer, tag in ((DEATH_SEED, lam10, "lam10-death"),):
        pq = PieceQueue(random.Random(seed))
        board = ds.Board()
        pos = 0
        while True:
            bulk = board.lock_candidates_full(pq.current)
            nc = len(bulk[0])
            if nc == 0 or all(bulk[3]):
                snaps.append({"seed": seed, "pos": pos, "tag": tag + "-death",
                              "terminal": True,
                              "pre_rows": np.asarray(board.row_bits(),
                                                     dtype=np.uint32),
                              "kind": str(pq.current),
                              "nk": str(pq.next_piece),
                              "kind_idx": piece_to_index(pq.current),
                              "nk_idx": piece_to_index(pq.next_piece)})
                break
            if pos >= SNAP_FROM and pos % SNAP_EVERY == 0:
                feats = dense_well_features(
                    np.asarray(board.row_bits(), dtype=np.uint32))
                snaps.append({"seed": seed, "pos": pos, "tag": tag,
                              "terminal": False,
                              "pre_rows": np.asarray(board.row_bits(),
                                                     dtype=np.uint32),
                              "kind": str(pq.current),
                              "nk": str(pq.next_piece),
                              "kind_idx": piece_to_index(pq.current),
                              "nk_idx": piece_to_index(pq.next_piece),
                              "occ": occ_of(board), "top": float(feats["top"]),
                              "holes": float(feats["holes_outside"])})
            pick = scorer.choose(board, pq.current, pq.next_piece)
            rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)[pick]
            board = ds.Board.from_row_bits([int(v) for v in rows])
            pq.advance()
            pos += 1
            if pos > 3000:
                break
    print(f"bank: {len(snaps)} snapshots from seed {DEATH_SEED} "
          f"(death at {[s['pos'] for s in snaps if s.get('terminal')]})",
          flush=True)
    bank_path = Path(args.bank_out)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bank_path,
        pre_rows=np.stack([s["pre_rows"] for s in snaps]),
        kind_idx=np.array([s["kind_idx"] for s in snaps], dtype=np.int64),
        nk_idx=np.array([s["nk_idx"] for s in snaps], dtype=np.int64),
        pos=np.array([s["pos"] for s in snaps], dtype=np.int64),
        meta_json=json.dumps({"seed": DEATH_SEED, "lam": args.lam,
                              "tags": [s["tag"] for s in snaps]}))

    # ---- 2. rescue test: survival policy from λ10 states ----
    # TRUE continuation tape: replay env RNG to (seed, pos) via a fresh
    # PieceQueue advanced pos times, then hand the live board + queue to
    # the rescue scorer. (PieceQueue draws are deterministic per seed, so
    # advancing pos times reconstructs the exact queue state.)
    def rescue(snap, scorer, label):
        seed, pos = snap["seed"], snap["pos"]
        if snap.get("terminal"):
            return {"label": label, "seed": seed, "pos": pos,
                    "skipped": "terminal"}
        pq = PieceQueue(random.Random(seed))
        board = ds.Board.from_row_bits([int(v) for v in snap["pre_rows"]])
        for _ in range(pos):
            pq.advance()  # queue state at handoff (draws are positional)
        # NOTE: PieceQueue history depends on draws, not boards, so
        # advancing pos times from the seed reconstructs the exact queue.
        pieces = 0
        occ_traj = []
        while pieces < RESCUE_CAP:
            bulk = board.lock_candidates_full(pq.current)
            nc = len(bulk[0])
            if nc == 0:
                reason = "no_candidates"
                break
            if all(bulk[3]):
                reason = "lockout"
                break
            pick = scorer.choose(board, pq.current, pq.next_piece)
            rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)[pick]
            board = ds.Board.from_row_bits([int(v) for v in rows])
            pq.advance()
            pieces += 1
            if pieces % 100 == 0:
                occ_traj.append(occ_of(board))
        else:
            reason = "max_pieces"
        return {"label": label, "seed": seed, "pos": pos, "pieces": pieces,
                "reason": reason, "occ_traj": occ_traj,
                "occ_at_handoff": occ_of(
                    ds.Board.from_row_bits([int(v) for v in snap["pre_rows"]]))}

    live_snaps = [s for s in snaps if not s.get("terminal")]
    rescue_rows = []
    for snap in live_snaps:
        r1 = rescue(snap, surv, "survival")
        r2 = rescue(snap, lam10, "lam10-control")
        rescue_rows.append({"pos": snap["pos"], "occ": snap.get("occ"),
                            "survival": r1, "lam10": r2})
        print(f"  pos {snap['pos']} occ {snap.get('occ')}: "
              f"surv {r1['pieces']} ({r1['reason']}) vs "
              f"lam10 {r2['pieces']} ({r2['reason']})", flush=True)

    # ---- 3. hysteresis grid (<=4 params, training seeds, reporting) ----
    # Policy: play λ10 while occ < ENTER; switch to survival (bonus off,
    # style off) while occ >= ENTER until occ <= EXIT. Grid: ENTER x EXIT.
    grid = [(70, 40), (90, 50), (60, 30)]
    hyst_rows = []
    for enter, exit_ in grid:
        tet = fails = off = tot = 0
        for seed in TRAIN_SEEDS:
            pq = PieceQueue(random.Random(seed))
            board = ds.Board()
            in_rescue = False
            pieces = 0
            while pieces < 1500:
                bulk = board.lock_candidates_full(pq.current)
                nc = len(bulk[0])
                if nc == 0 or all(bulk[3]):
                    fails += 1
                    break
                occ = occ_of(board)
                if not in_rescue and occ >= enter:
                    in_rescue = True
                elif in_rescue and occ <= exit_:
                    in_rescue = False
                sc = surv if in_rescue else lam10
                off += int(in_rescue)
                pick = sc.choose(board, pq.current, pq.next_piece)
                rows = np.asarray(bulk[4], dtype=np.uint32).reshape(
                    nc, 21)[pick]
                nl = int(bulk[5][pick])
                if nl == 4:
                    tet += 1
                board = ds.Board.from_row_bits([int(v) for v in rows])
                pq.advance()
                pieces += 1
                tot += 1
        hyst_rows.append({"enter": enter, "exit": exit_,
                          "t_per_100": round(100 * tet / max(1, tot), 3),
                          "failures": fails,
                          "style_off_frac": round(off / max(1, tot), 3),
                          "games": len(TRAIN_SEEDS)})
        print(f"  hyst enter {enter}/exit {exit_}: T/100 "
              f"{hyst_rows[-1]['t_per_100']}, fails {fails}, "
              f"off-frac {hyst_rows[-1]['style_off_frac']}", flush=True)

    res = {
        "weights": weights,
        "weights_sha256": hashlib.sha256(
            Path(weights).read_bytes()).hexdigest(),
        "lam": args.lam, "death_seed": DEATH_SEED,
        "n_snaps": len(snaps),
        "bank": str(bank_path.resolve()),
        "rescue": rescue_rows,
        "hysteresis": hyst_rows,
        "train_seeds": TRAIN_SEEDS,
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "note": ("NO panels, NO training. Rescue uses TRUE continuation "
                 "tapes (queue reconstructed positionally). Hysteresis on "
                 "training seeds = reporting only."),
    }
    op = Path(args.out)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op} + {bank_path} in {res['wall_seconds']}s")


if __name__ == "__main__":
    main()
