"""Policy-identity probe: sweep harness vs collector, same board+queue.

§29 reported the same student net scoring 1347.2 (5/8 cap) under the sweep
harness vs 1003.5 (4/8 cap) under the diag_student_aggregate collector on
the SAME training seeds. This probe replays ONE seeded game through both
implementations simultaneously and reports the FIRST divergence, aligned
by canonical (state, x, y) geometry of the grounded piece:

  A. sweep path  : search_locks + NeuralScorer.score_batch
                   (selfplay.play_game_detailed internals)
  B. collector   : lock_candidates_full + flat_rows_to_tensor + net forward
                   (diag_student_aggregate internals)

Checked per step: identical piece streams, identical viable masks (by
geometry), aligned per-candidate scores, identical argmax pick. The probe
STOPS at the first difference (review priority 1: stop at first difference,
not first death).

Known differences to watch (from code inspection):
  * A scores with make_afterstate(board).tensor (per-candidate clone+lock
    +clear, board_to_tensor); B scores bulk rows from lock_candidates_full
    (Rust bulk lock+clear). Both should be the same post-clear board -
    parity was verified in Phase 1 (D5), but only via from_row_bits paths.
  * A's score_batch fast path requires cur == nxt to use score_batch_fast;
    at ply-1 cur != nxt (preview), so A uses net(boards, cur, nxt) - same
    as B. If a hidden caller passes cur==nxt, the fast path duplicates the
    CURRENT piece into the preview slot (champion 2-ply semantics).
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import TAPNet, piece_to_index
from afterstate import flat_rows_to_tensor, make_afterstate
from piece_history import PieceQueue
from neural_scorer import NeuralScorer
import deathbot_sim as ds

STUDENT = HERE.parent / "_diag_out" / "fullsplit_seed42.pt"
SEED = 26000  # collector: capped at 1500 in 26.2 (a long tape)
CAP = 400


def canon_key(piece):
    return (int(piece.state), int(piece.x), int(piece.y))


def main() -> None:
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(str(STUDENT), map_location="cpu"))
    net.eval()
    scorer = NeuralScorer(net=net, device="cpu")

    pqA = PieceQueue(random.Random(SEED))
    boardA = ds.Board()
    pqB = PieceQueue(random.Random(SEED))
    boardB = ds.Board()

    for step in range(CAP):
        assert (pqA.current, pqA.next_piece) == (pqB.current, pqB.next_piece)
        cur_idx = piece_to_index(pqA.current)
        nxt_idx = piece_to_index(pqA.next_piece)

        # ---- Path A (sweep) ----
        candsA = boardA.search_locks(pqA.current)
        if not candsA:
            print(f"step {step}: A terminal no_candidates")
            break
        scoresA_all = scorer.score_batch(boardA, candsA,
                                         current_piece_idx=cur_idx,
                                         next_piece_idx=nxt_idx)
        viableA = [i for i, c in enumerate(candsA)
                   if not boardA.locks_out(c.piece)]
        if not viableA:
            print(f"step {step}: A terminal lockout (all candidates lock out)")
            break
        keysA_all = [canon_key(c.piece) for c in candsA]
        bestA = max(viableA, key=lambda i: (scoresA_all[i], -i))
        keyA = keysA_all[bestA]

        # ---- Path B (collector) ----
        bulk = boardB.lock_candidates_full(pqB.current)
        nc = len(bulk[0])
        if nc == 0:
            print(f"step {step}: B terminal no_candidates")
            break
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
        viableB = [i for i, lo in enumerate(bulk[3]) if not lo]
        if not viableB:
            print(f"step {step}: B terminal lockout (all candidates lock out)")
            break
        candsB = boardB.search_locks(pqB.current)
        keysB_all = [canon_key(c.piece) for c in candsB]
        assert len(candsB) == nc, (len(candsB), nc)  # 19 pin: same order
        boards = flat_rows_to_tensor(rows[viableB].reshape(-1), len(viableB))
        cur_t = torch.full((len(viableB),), cur_idx, dtype=torch.long)
        nxt_t = torch.full((len(viableB),), nxt_idx, dtype=torch.long)
        with torch.no_grad():
            v1 = net(boards, cur_t, nxt_t).flatten().tolist()
        pickB_viable = max(range(len(viableB)), key=lambda j: (v1[j], -j))
        keyB = keysB_all[viableB[pickB_viable]]

        # ---- Compare ----
        setA = {keysA_all[i] for i in viableA}
        setB = {keysB_all[i] for i in viableB}
        if setA != setB:
            print(f"step {step}: VIABLE-SET MISMATCH "
                  f"(|A|={len(setA)}, |B|={len(setB)})")
            print("  only in A:", sorted(setA - setB)[:10])
            print("  only in B:", sorted(setB - setA)[:10])
            break
        # aligned scores on the union of viable keys
        posA = {keysA_all[i]: i for i in viableA}
        posB = {keysB_all[i]: i for i in viableB}
        maxdiff = 0.0
        worst = None
        for k in setA:
            ja, jb = posA[k], posB[k]
            with torch.no_grad():
                vv = net(boards[jb:jb + 1], cur_t[jb:jb + 1],
                         nxt_t[jb:jb + 1]).item()
            d = abs(scoresA_all[ja] - vv)
            if d > maxdiff:
                maxdiff, worst = d, (k, scoresA_all[ja], vv)
        if maxdiff > 1e-3:
            print(f"step {step}: SCORE MISMATCH max {maxdiff:.6f} at {worst}")
            break
        if keyA != keyB:
            sa, sb = scoresA_all[bestA], v1[pickB_viable]
            print(f"step {step}: PICK DIVERGENCE (scores agree to "
                  f"{maxdiff:.6f})")
            print(f"  A pick {keyA} (score {sa:.4f}); "
                  f"B pick {keyB} (score {sb:.4f})")
            print(f"  A's score for B's pick: {scoresA_all[posA[keyB]]:.4f}")
            print(f"  B's score for A's pick: {v1[posB[keyA]]:.4f}")
            break
        # commit both via their own paths
        afterA = make_afterstate(boardA, candsA[bestA])
        boardA = afterA.board
        boardB = ds.Board.from_row_bits(
            [int(v) for v in rows[viableB[pickB_viable]]])
        pqA.advance()
        pqB.advance()
    else:
        print(f"NO divergence in {CAP} steps on seed {SEED}: same viable "
              f"sets (by geometry), scores aligned <1e-3, same picks.")


if __name__ == "__main__":
    main()
