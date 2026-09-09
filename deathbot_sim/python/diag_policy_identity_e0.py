"""E0 quantitative probe: aliased vs separate explore-RNG tapes (§32.3 E0).

Replays the 8 training seeds 26000-26007 (REPORTING ONLY — never a
selection panel) through play_game_detailed under BOTH RNG conventions:

  A. aliased   : play_game_detailed(scorer, Random(seed), epsilon=0.0)
                 — the historical harness convention (sweep/evaluate).
  B. separate  : ... explore_rng=Random(seed + 7_000_000_001)
                 — the train_fitd / parallel-worker convention, and the
                   implicit convention of the §26.2 collector (no RNG draw).

Also replays the same seeds through the LOCKSTEP scoring-parity probe
(sweep score_batch path vs collector bulk-forward path, shared queue,
canonical (state,x,y) alignment, stop at first difference) at cap 400,
on seed 26001 (a tape where A and B differ) as well as 26000.

Writes _diag_out/policy_identity_e0.json. No panels, no checkpoints,
no training. Runtime ~2-4 min (8 seeds x 2 conventions, student net).

Usage:
    python diag_policy_identity_e0.py --out ../_diag_out/policy_identity_e0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXPLORE_OFFSET = 7_000_000_001
SEEDS = list(range(26000, 26008))
CAP = 1500
PROBE_CAP = 400


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", default="../_diag_out/fullsplit_seed42.pt")
    ap.add_argument("--incumbent", default="../../model_weights_snapshot9.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch

    from model import TAPNet, piece_to_index
    from neural_scorer import NeuralScorer
    from selfplay import play_game_detailed
    from afterstate import flat_rows_to_tensor, make_afterstate
    from piece_history import PieceQueue
    import deathbot_sim as ds

    def resolve(p: str) -> Path:
        sup = Path(p).expanduser()
        for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
                  HERE.parent.parent.parent / sup):
            if c.is_file():
                return c.resolve()
        raise FileNotFoundError(f"not found: {p}")

    student_path = resolve(args.student)
    incumb_path = resolve(args.incumbent)
    student = TAPNet(output="linear")
    student.load_state_dict(torch.load(str(student_path), map_location="cpu"))
    student.eval()
    incumb = TAPNet(output="linear")
    incumb.load_state_dict(torch.load(str(incumb_path), map_location="cpu"))
    incumb.eval()
    s_scorer = NeuralScorer(net=student, device="cpu")
    i_scorer = NeuralScorer(net=incumb, device="cpu")

    def canon_key(piece):
        return (int(piece.state), int(piece.x), int(piece.y))

    t0 = time.perf_counter()
    tapes: dict[str, dict] = {}
    for seed in SEEDS:
        row: dict[str, dict] = {}
        for name, net, scorer in (("student", student, s_scorer),
                                  ("incumbent", incumb, i_scorer)):
            a = play_game_detailed(scorer, random.Random(seed), epsilon=0.0,
                                   max_pieces=CAP)
            b = play_game_detailed(scorer, random.Random(seed), epsilon=0.0,
                                   max_pieces=CAP,
                                   explore_rng=random.Random(seed + EXPLORE_OFFSET))
            row[name] = {
                "aliased": {"pieces": a.pieces_placed, "reason": a.reason},
                "separate": {"pieces": b.pieces_placed, "reason": b.reason},
            }
        tapes[str(seed)] = row
        print(f"  seed {seed}: student A={row['student']['aliased']['pieces']} "
              f"B={row['student']['separate']['pieces']} | incumb "
              f"A={row['incumbent']['aliased']['pieces']} "
              f"B={row['incumbent']['separate']['pieces']}", flush=True)

    # Lockstep parity probe on 26000 (agreeing tape) + 26001 (diverging tape).
    parity: dict[str, dict] = {}
    for seed in (26000, 26001):
        net = student
        scorer = NeuralScorer(net=net, device="cpu")
        pqA = PieceQueue(random.Random(seed))
        boardA = ds.Board()
        pqB = PieceQueue(random.Random(seed))
        boardB = ds.Board()
        verdict = {"diverged": False, "steps_compared": 0,
                   "max_score_diff": 0.0}
        for step in range(PROBE_CAP):
            assert (pqA.current, pqA.next_piece) == (pqB.current, pqB.next_piece)
            cur_idx = piece_to_index(pqA.current)
            nxt_idx = piece_to_index(pqA.next_piece)
            candsA = boardA.search_locks(pqA.current)
            if not candsA:
                verdict.update({"terminal": "no_candidates", "at_step": step})
                break
            scoresA_all = scorer.score_batch(
                boardA, candsA, current_piece_idx=cur_idx,
                next_piece_idx=nxt_idx)
            viableA = [i for i, c in enumerate(candsA)
                       if not boardA.locks_out(c.piece)]
            if not viableA:
                verdict.update({"terminal": "lockout", "at_step": step})
                break
            keysA_all = [canon_key(c.piece) for c in candsA]
            bestA = max(viableA, key=lambda i: (scoresA_all[i], -i))
            bulk = boardB.lock_candidates_full(pqB.current)
            nc = len(bulk[0])
            assert nc == len(candsA), (nc, len(candsA))
            rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
            viableB = [i for i, lo in enumerate(bulk[3]) if not lo]
            assert viableB, "B all-dead while A viable"
            candsB = boardB.search_locks(pqB.current)
            keysB_all = [canon_key(c.piece) for c in candsB]
            # key -> position inside the viable-only v1 list (NOT the bulk
            # index: v1 has one entry per viable candidate).
            posB = {keysB_all[i]: j for j, i in enumerate(viableB)}
            boards = flat_rows_to_tensor(rows[viableB].reshape(-1), len(viableB))
            cur_t = torch.full((len(viableB),), cur_idx, dtype=torch.long)
            nxt_t = torch.full((len(viableB),), nxt_idx, dtype=torch.long)
            with torch.no_grad():
                v1 = net(boards, cur_t, nxt_t).flatten().tolist()
            keyA = keysA_all[bestA]
            pickB_viable = max(range(len(viableB)), key=lambda j: (v1[j], -j))
            keyB = keysB_all[viableB[pickB_viable]]
            posA = {keysA_all[i]: i for i in viableA}
            assert {keysA_all[i] for i in viableA} == \
                {keysB_all[i] for i in viableB}, "viable-set mismatch"
            maxdiff = 0.0
            for k in posA:
                d = abs(scoresA_all[posA[k]] - v1[posB[k]])
                maxdiff = max(maxdiff, d)
            verdict["max_score_diff"] = max(verdict["max_score_diff"], maxdiff)
            assert maxdiff <= 1e-3, (step, maxdiff)
            assert keyA == keyB, (step, keyA, keyB)
            verdict["steps_compared"] = step + 1
            afterA = make_afterstate(boardA, candsA[bestA])
            boardA = afterA.board
            boardB = ds.Board.from_row_bits(
                [int(v) for v in rows[viableB[pickB_viable]]])
            pqA.advance()
            pqB.advance()
        else:
            verdict.update({"terminal": None})
        parity[str(seed)] = verdict
        print(f"  parity seed {seed}: {verdict['steps_compared']} steps, "
              f"maxdiff {verdict['max_score_diff']:.2e}", flush=True)

    res = {
        "explore_offset": EXPLORE_OFFSET,
        "cap": CAP,
        "probe_cap": PROBE_CAP,
        "student": str(student_path),
        "student_sha256": hashlib.sha256(
            student_path.read_bytes()).hexdigest(),
        "incumbent": str(incumb_path),
        "incumbent_sha256": hashlib.sha256(
            incumb_path.read_bytes()).hexdigest(),
        "tapes": tapes,
        "parity": parity,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "note": ("26000-26007 are TRAINING seeds (reporting only). "
                 "select_candidate_index draws explore_rng.random() per "
                 "placement even at epsilon=0; aliased play_group != "
                 "separate tapes. Collector (§26.2, no RNG draw) == "
                 "separate by construction. E3 must fix one convention "
                 "for both arms."),
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {op} in {res['wall_seconds']}s")


if __name__ == "__main__":
    main()
