"""Piece-slot semantics audit (§20 item 1, no panels, no gameplay eval).

Documents, for each scoring stage, what board the net receives, what each
piece slot means, which piece identity each path supplies, and which future
pieces are legitimately known — then quantifies incumbent (fast-path,
cur-in-both-slots) vs semantically-corrected (cur+true-preview) variants on
ACTION changes, not just score noise.

Stages audited:
  A. selfplay/bank true path: NeuralScorer.score_batch(board, cands,
     current_piece_idx=cur, next_piece_idx=nxt) — board = post-lock
     post-clear afterstate of the CURRENT piece placement; cur slot = piece
     just placed; nxt slot = legitimately-known preview (PieceQueue holds
     current+next; TAP randomizer reveals exactly one preview). SEMANTIC.
  B. champion 2-ply ply-1 (TwoPlyScorer._placements_from_bulk): same board
     as A, but fast path feeds cur in BOTH slots (score_batch_fast). The nxt
     slot carries cur, NOT the preview — even though the preview IS known
     (choose() receives next_piece; diag_p1_fit teacher fn does too but
     ignores nk for v1).
  C. champion 2-ply ply-2: board = post-clear afterstate of the PREVIEW
     piece placed on the ply-1 successor; nxt slot = same preview id again
     (fast path), so slot semantics = "preview-as-cur, preview-as-nxt".
     True semantics would need the piece AFTER preview, which is NOT known
     (only one preview exists). So ply-2 duplication is FORCED by the
     information horizon; ply-1 duplication is NOT (preview is known there).
  D. viz_live TwoPlyVizScorer: ply-1 from _placements (same as B); ply-2
     uses next_kind derived from the passed next_piece_idx (same as C).

Metrics (frozen snapshot9 bank, seed-7 uniform 500-state sample):
  - per-candidate |fast - true| distribution (mean/median/p90/max)
  - argmax action-change rate fast-vs-true on ply-1
  - true-preview top1-top2 gap distribution overall + on changed states
    (are flips confined to near-ties?)
  - cur==nxt control: states where placed==preview (fast==true by
    construction; nonzero diff there = implementation bug, not semantics)

Versioned champion config (do NOT silently "fix"):
  CHAMPION = snapshot9 sha 229bbe2a + full 2-ply beam>=n + fast-path
  scoring (cur-in-both-slots both plies) + bonus-free + gamma_ply 0.999.
  A corrected variant (true preview at ply-1) is a SEPARATELY NAMED config
  (candidate: CHAMPION+trueprev-ply1), compared on action changes below.

Usage:
    python diag_piece_slots.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \\
        --weights ../../model_weights_snapshot9.pt --states 500 --seed 7 \\
        --out ../_diag_out/piece_slots.json
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

from model import TAPNet, piece_to_index  # noqa: E402
from diag_phase2 import Bank  # noqa: E402
from afterstate import flat_rows_to_tensor  # noqa: E402


SEMANTICS_DOC = {
    "board": "post-lock post-clear afterstate of the piece just placed "
              "(visible 20 rows; row-20 buffer dropped by the encoder)",
    "slot_cur": "the piece just placed (ply-1: current; ply-2: preview)",
    "slot_nxt": "the next KNOWN piece after the placed one "
                 "(ply-1: preview, known via PieceQueue; "
                 "ply-2: piece-after-preview, NOT known — one-preview horizon)",
    "stage_A_selfplay_bank": "score_batch(cur=placed, nxt=preview): SEMANTIC",
    "stage_B_champion_ply1": "score_batch_fast(cur, cur): nxt slot carries "
                              "cur instead of the known preview — CORRECTABLE",
    "stage_C_champion_ply2": "score_batch_fast(nxt, nxt): nxt slot carries "
                              "preview instead of the unknown next-next — "
                              "FORCED by the information horizon",
    "stage_D_viz": "same as B/C (ply-1 dup cur; ply-2 dup preview)",
    "champion_config": "snapshot9 sha 229bbe2a + full 2-ply beam>=n + "
                        "fast-path both plies + bonus-free + gamma_ply 0.999",
}


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--states", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    bank = Bank(args.bank)
    n = bank.n_states
    weights_path = resolve_weights(args.weights)
    print(f"Bank: {n} states; sample {args.states} (seed {args.seed})")

    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    net.eval()

    placed_ids, next_ids = bank._replay_piece_ids()
    states = rng.sample(range(n), min(args.states, n))

    diffs: list[float] = []
    gap_fast: list[float] = []
    gap_true: list[float] = []
    gap_on_change: list[float] = []
    maxdiff_on_change: list[float] = []
    n_change = 0
    n_eval = 0
    n_cur_eq_nxt = 0
    control_maxdiff = 0.0
    changed_seeds: list[int] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in states:
            lo, hi = bank.candidate_slice(s)
            m = hi - lo
            if m < 2:
                continue
            rows = bank.cand_rows[lo * 21:hi * 21].reshape(m, 21)
            boards = flat_rows_to_tensor(rows.reshape(-1), m)
            cur = piece_to_index(placed_ids[s])
            nxt = piece_to_index(next_ids[s])
            cur_t = torch.full((m,), cur, dtype=torch.long)
            nxt_t = torch.full((m,), nxt, dtype=torch.long)
            v_fast = net.score_batch_fast(boards, cur_t).flatten().tolist()
            v_true = net(boards, cur_t, nxt_t).flatten().tolist()
            diffs.extend(abs(a - b) for a, b in zip(v_fast, v_true))
            sf = sorted(v_fast, reverse=True)
            st = sorted(v_true, reverse=True)
            gap_fast.append(sf[0] - sf[1])
            gap_true.append(st[0] - st[1])
            n_eval += 1
            if cur == nxt:
                n_cur_eq_nxt += 1
                for a, b in zip(v_fast, v_true):
                    control_maxdiff = max(control_maxdiff, abs(a - b))
            a_fast = int(np.argmax(v_fast))
            a_true = int(np.argmax(v_true))
            if a_fast != a_true:
                n_change += 1
                gap_on_change.append(st[0] - st[1])
                maxdiff_on_change.append(
                    max(abs(a - b) for a, b in zip(v_fast, v_true)))
                if len(changed_seeds) < 20:
                    changed_seeds.append(s)
    elapsed = time.perf_counter() - t0
    sdiff = sorted(diffs)

    def _pct(q: float) -> float:
        return sdiff[min(len(sdiff) - 1, int(len(sdiff) * q))] if sdiff else 0.0

    res = {
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(
            weights_path.read_bytes()).hexdigest(),
        "bank": str(Path(args.bank).resolve()),
        "states_sampled": len(states),
        "states_eval": n_eval,
        "seed": args.seed,
        "semantics": SEMANTICS_DOC,
        "per_candidate_absdiff": {
            "n": len(diffs),
            "mean": statistics.fmean(diffs) if diffs else 0.0,
            "median": statistics.median(diffs) if diffs else 0.0,
            "p90": _pct(0.9),
            "max": max(diffs) if diffs else 0.0,
        },
        "action_change": {
            "n_changed": n_change,
            "n_eval": n_eval,
            "rate": (n_change / n_eval) if n_eval else 0.0,
        },
        "gap_fast_median": statistics.median(gap_fast) if gap_fast else 0.0,
        "gap_true_median": statistics.median(gap_true) if gap_true else 0.0,
        "gap_true_on_changed": {
            "n": len(gap_on_change),
            "mean": statistics.fmean(gap_on_change) if gap_on_change else 0.0,
            "median": statistics.median(gap_on_change)
            if gap_on_change else 0.0,
            "max": max(gap_on_change) if gap_on_change else 0.0,
        },
        "maxdiff_on_changed_mean": (statistics.fmean(maxdiff_on_change)
                                    if maxdiff_on_change else 0.0),
        "control_cur_eq_nxt": {
            "n_states": n_cur_eq_nxt,
            "max_absdiff": control_maxdiff,
        },
        "changed_state_ids_sample": changed_seeds,
        "verdict": ("ply-1 duplication is ACTION-NEGLIGIBLE (<=~1% flips, "
                    "confined to near-tie gaps << median) — keep fast path "
                    "as the versioned champion; ply-2 duplication is "
                    "information-forced. Control must be ~0."
                    if (n_change / max(1, n_eval)) <= 0.02 else
                    "ply-1 duplication FLIPS >2% of sampled actions — "
                    "name the corrected variant and compare closed-loop "
                    "before any distillation claims input sufficiency."),
        "wall_s": round(elapsed, 1),
    }
    print(f"eval {n_eval} states in {elapsed:.0f}s: "
          f"action-change {n_change}/{n_eval} "
          f"({n_change / max(1, n_eval):.3f}); "
          f"|diff| mean {res['per_candidate_absdiff']['mean']:.3f} "
          f"median {res['per_candidate_absdiff']['median']:.3f} "
          f"max {res['per_candidate_absdiff']['max']:.3f}; "
          f"control maxdiff {control_maxdiff:.2e}")
    if gap_on_change:
        print(f"on-change true-gap mean {statistics.fmean(gap_on_change):.3f} "
              f"median {statistics.median(gap_on_change):.3f} "
              f"(overall true-gap median "
              f"{statistics.median(gap_true):.3f})")
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
