"""E2 student-proposal shortlist probe (§32.3 E2, offline, NO panels).

Question: can the promoted student (fullsplit_seed42, trained to predict
the snapshot9 full-2-ply teacher's argmax) serve as a SEARCH PROPOSER —
i.e. does a small union shortlist built from student + incumbent 1-ply
rankings cover the full teacher's winner at smaller k than v1-top-k alone?

Protocol (all frozen, no gameplay, no training):
  states  = bank states (snapshot9 bank, incumbent-trajectory) — sample N
          + student-visited states: REPLAYED on the fly from training seeds
            26000-26007 under the separate-RNG convention (== collector
            tapes per E0, §33). Replayed states are rebuilt identically
            (pre-board packed rows + placed/preview ids), NOT re-labeled by
            a new teacher: the frozen snapshot9 full-2-ply teacher labels
            every state here.
  teacher = versioned CHAMPION (snapshot9 full 2-ply, fast-path both plies,
            gamma 0.999) — same semantics as teacher_scores_for_state.
  shortlists (candidate indices in viable order):
    v1_k{k}        : top-k by incumbent 1-ply value (k in --ks)
    union_{a}_{b}  : top-a incumbent + top-b student + lowest-top viable
                     (dedup; size <= a+b+1)
  metrics per shortlist: coverage (teacher winner in shortlist), regret|miss
    (positive-regret, live only; dead-miss counted separately), mean size,
    size histogram. Stratified by teacher margin buckets + high-stack
    (top>=12) states where proposals matter most.

Gate (§32.3): closed-loop ONLY if a union list beats v1-top-8's coverage
(1.000 on bank states per §32.1.6) at SMALLER mean size, or wins on
student-visited/high-stack states. Otherwise E2 closes as a negative.

Usage:
    python diag_student_propose.py --bank ../_diag_out/state_bank_snapshot9_32games.npz \\
        --incumbent ../../model_weights_snapshot9.pt \\
        --student ../_diag_out/fullsplit_seed42.pt \\
        --bank-states 400 --replay-games 8 --seed0 26000 \\
        --out ../_diag_out/student_propose.json
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
from diag_tinyset_overfit import teacher_scores_for_state  # noqa: E402

DEAD_PENALTY = -1e6
MARGIN_BUCKETS = [(0.0, 1.0, "<1"), (1.0, 5.0, "1-5"),
                  (5.0, 20.0, "5-20"), (20.0, float("inf"), "20+")]
EXPLORE_OFFSET = 7_000_000_001


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"not found: {p}")


def v1_scores(net, boards, cur_idx, nxt_idx):
    cur = torch.full((boards.shape[0],), cur_idx, dtype=torch.long)
    nxt = torch.full((boards.shape[0],), nxt_idx, dtype=torch.long)
    with torch.inference_mode():
        return net(boards, cur, nxt).flatten().tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--bank-states", type=int, default=400)
    ap.add_argument("--replay-games", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=26000)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 8])
    ap.add_argument("--unions", nargs="+", type=str,
                    default=["3_3", "2_2", "3_2", "5_3"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    incumb = TAPNet(output="linear")
    incumb_path = resolve(args.incumbent)
    incumb.load_state_dict(torch.load(str(incumb_path), map_location="cpu"))
    incumb.eval()
    student = TAPNet(output="linear")
    student_path = resolve(args.student)
    student.load_state_dict(torch.load(str(student_path), map_location="cpu"))
    student.eval()

    import deathbot_sim as ds
    from piece_history import PieceQueue

    bank = Bank(str(resolve(args.bank)))
    n_bank = bank.n_states
    placed_ids, next_ids = bank._replay_piece_ids()
    n_cand = int(bank.cand_offsets[-1])
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n_bank):
        lo, hi = bank.candidate_slice(s)
        cand_cur[lo:hi] = piece_to_index(placed_ids[s])
        cand_nxt[lo:hi] = piece_to_index(next_ids[s])

    # ---- collect states: (boards, cur, nxt, viable_bank_idx, source) ----
    states: list[dict] = []
    bank_sample = rng.sample(range(n_bank), min(args.bank_states, n_bank))
    for s in bank_sample:
        lo, hi = bank.candidate_slice(s)
        lock = bank.cand_lockout[lo:hi]
        viable = [i for i, lk in enumerate(lock) if not lk]
        if len(viable) < 2:
            continue
        states.append({"boards": cand_boards[lo:hi], "cur": int(cand_cur[lo]),
                       "nxt": int(cand_nxt[lo]), "viable": viable,
                       "source": "bank", "key": f"bank{s}"})

    # student-visited states: replay training seeds with the STUDENT 1-ply
    # greedy policy under the separate-RNG convention (E0: == collector).
    n_replay = 0
    for g in range(args.replay_games):
        seed = args.seed0 + g
        pq = PieceQueue(random.Random(seed))
        board = ds.Board()
        pos = 0
        while pos < args.cap:
            bulk = board.lock_candidates_full(pq.current)
            nc = len(bulk[0])
            if nc == 0 or all(bulk[3]):
                break
            rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
            viable = [i for i, lo in enumerate(bulk[3]) if not lo]
            if len(viable) < 2:
                board = ds.Board.from_row_bits(
                    [int(v) for v in rows[viable[0]]]) if viable else board
                pq.advance()
                pos += 1
                continue
            boards = flat_rows_to_tensor(rows[viable].reshape(-1), len(viable))
            ci, ni = piece_to_index(pq.current), piece_to_index(pq.next_piece)
            sv = v1_scores(student, boards, ci, ni)
            pick = max(range(len(viable)), key=lambda j: (sv[j], -j))
            states.append({"boards": boards, "cur": ci, "nxt": ni,
                           "viable": list(range(len(viable))),
                           "source": "student", "key": f"{seed}@{pos}"})
            n_replay += 1
            board = ds.Board.from_row_bits(
                [int(v) for v in rows[viable[pick]]])
            pq.advance()
            pos += 1
    print(f"states: {len([s for s in states if s['source']=='bank'])} bank + "
          f"{n_replay} student-visited", flush=True)

    # ---- per-state teacher label + proposal metrics ----
    def shortlists(iv, sv, tops, n):
        out = {}
        for k in args.ks:
            out[f"v1_k{k}"] = sorted(
                range(n), key=lambda j: (-iv[j], j))[:k]
        for spec in args.unions:
            a, b = (int(x) for x in spec.split("_"))
            ti = sorted(range(n), key=lambda j: (-iv[j], j))[:a]
            ts = sorted(range(n), key=lambda j: (-sv[j], j))[:b]
            union = list(dict.fromkeys(ti + ts + [tops]))
            out[f"union_{spec}"] = union
        return out

    results: dict[str, dict] = {}
    misses: dict[str, list] = {}
    n_eval = n_skip = n_alldead = 0
    t_lab = time.perf_counter()
    with torch.no_grad():
        for st in states:
            boards, ci, ni = st["boards"], st["cur"], st["nxt"]
            # teacher label needs a live pre-board: rebuild from bank rows
            # for bank states; for replay states rebuild via bulk rows.
            iv = v1_scores(incumb, boards, ci, ni)
            sv = v1_scores(student, boards, ci, ni)
            n = boards.shape[0]
            st["_iv"], st["_sv"] = iv, sv
            # lowest-top viable index (geometry hedge member): fewest filled
            # cells in the visible tensor (proxy for stack height).
            grid = (boards.squeeze(1).numpy() > 0.5)  # [n,20,10]
            st["_tops"] = min(range(n),
                              key=lambda j: (float(grid[j].sum()), -iv[j], j))
    print(f"v1 pre-score done in {time.perf_counter()-t_lab:.0f}s", flush=True)

    # Teacher labeling, split by source (bank states: frozen fn on live
    # pre-boards; replay states: live re-expansion with identical semantics).
    def teacher_for_bank_state(s):
        lo, hi = bank.candidate_slice(s)
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        tv, viable_live = teacher_scores_for_state(
            incumb, pre, placed_ids[s], next_ids[s], args.gamma_ply)
        if tv is None:
            return None, None
        # viable_live are live positions; bank order == live order (pinned
        # §19), viable set here == st viable by construction.
        return tv, viable_live

    # label bank states
    labeled = 0
    bank_key_to_state = {st["key"]: st for st in states
                         if st["source"] == "bank"}
    for key, st in bank_key_to_state.items():
        s = int(key[4:])
        tv, viable_live = teacher_for_bank_state(s)
        if tv is None:
            st["_skip"] = True
            continue
        st["_tv"] = tv
        labeled += 1
    print(f"bank teacher labels: {labeled}/{len(bank_key_to_state)} "
          f"({time.perf_counter()-t_lab:.0f}s)", flush=True)

    # label replay states via live re-expansion (identical champion semantics:
    # v1 fast-path cur-slot + gamma*max(v2 true-preview), dead -> penalty).
    t_re = time.perf_counter()
    n_re_lab = 0
    for g in range(args.replay_games):
        seed = args.seed0 + g
        pq = PieceQueue(random.Random(seed))
        board = ds.Board()
        # walk the same student trajectory to each recorded state
        recs = [st for st in states if st["source"] == "student"
                and st["key"].startswith(f"{seed}@")]
        rec_by_pos = {int(st["key"].split("@")[1]): st for st in recs}
        pos = 0
        while pos < args.cap and rec_by_pos:
            bulk = board.lock_candidates_full(pq.current)
            nc = len(bulk[0])
            if nc == 0 or all(bulk[3]):
                break
            rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nc, 21)
            viable = [i for i, lo in enumerate(bulk[3]) if not lo]
            if pos in rec_by_pos and len(viable) >= 2:
                st = rec_by_pos[pos]
                boards = flat_rows_to_tensor(rows[viable].reshape(-1),
                                             len(viable))
                ci = piece_to_index(pq.current)
                ni = piece_to_index(pq.next_piece)
                cur = torch.full((len(viable),), ci, dtype=torch.long)
                fast = getattr(incumb, "score_batch_fast", None)
                with torch.inference_mode():
                    v1 = (fast(boards, cur) if fast is not None
                          else incumb(boards, cur, cur)).flatten().tolist()
                # ply-2 over true preview successors
                mk = getattr(ds.Board, "from_row_bits_bulk", None)
                succs = (mk([int(v) for vv in rows[viable] for v in vv])
                         if mk is not None else
                         [ds.Board.from_row_bits([int(v) for v in rows[i]])
                          for i in viable])
                segs, counts = [], []
                for b in succs:
                    r2 = b.lock_candidates_full(pq.next_piece)
                    n2 = len(r2[0])
                    if n2 == 0:
                        counts.append(0)
                        continue
                    rr = np.asarray(r2[4], dtype=np.uint32).reshape(n2, 21)
                    keep = [i for i in range(n2) if not r2[3][i]]
                    if not keep:
                        counts.append(0)
                        continue
                    counts.append(len(keep))
                    segs.extend(rr[i] for i in keep)
                tv = list(v1)
                if segs:
                    big = flat_rows_to_tensor(np.stack(segs), len(segs))
                    nt = torch.full((len(segs),), ni, dtype=torch.long)
                    with torch.inference_mode():
                        allv = (fast(big, nt) if fast is not None
                                else incumb(big, nt, nt)).flatten().tolist()
                    off = 0
                    for j in range(len(viable)):
                        if counts[j] == 0:
                            tv[j] = DEAD_PENALTY
                        else:
                            seg = allv[off:off + counts[j]]
                            off += counts[j]
                            tv[j] = v1[j] + args.gamma_ply * max(seg)
                else:
                    tv = [DEAD_PENALTY] * len(viable)
                st["_tv"] = tv
                n_re_lab += 1
            # advance along the STUDENT trajectory (same greedy pick)
            if len(viable) >= 2:
                boards = flat_rows_to_tensor(rows[viable].reshape(-1),
                                             len(viable))
                ci = piece_to_index(pq.current)
                ni = piece_to_index(pq.next_piece)
                sv = v1_scores(student, boards, ci, ni)
                pick = max(range(len(viable)), key=lambda j: (sv[j], -j))
                board = ds.Board.from_row_bits(
                    [int(v) for v in rows[viable[pick]]])
            elif viable:
                board = ds.Board.from_row_bits(
                    [int(v) for v in rows[viable[0]]])
            pq.advance()
            pos += 1
            if pos % 500 == 0:
                print(f"  replay label game {seed} pos {pos} ...", flush=True)
    print(f"replay teacher labels: {n_re_lab} "
          f"({time.perf_counter()-t_re:.0f}s)", flush=True)

    # ---- metrics ----
    names: list[str] = ([f"v1_k{k}" for k in args.ks]
                        + [f"union_{u}" for u in args.unions])
    agg: dict[str, dict] = {nm: {"n": 0, "cover": 0, "regret": [],
                                 "dead_miss": 0, "sizes": [],
                                 "by_source": {}, "by_bucket": {},
                                 "highstack": {"n": 0, "cover": 0}}
                            for nm in names}
    for st in states:
        tv = st.get("_tv")
        if tv is None:
            n_skip += 1
            continue
        iv, sv = st["_iv"], st["_sv"]
        n = len(tv)
        if all(float(x) <= DEAD_PENALTY / 2 for x in tv):
            n_alldead += 1
            continue
        t_best = max(range(n), key=lambda j: (tv[j], -j))
        best = float(tv[t_best])
        margin = best - max((float(tv[j]) for j in range(n) if j != t_best),
                            default=best)
        bucket = next(lbl for lo, hi, lbl in MARGIN_BUCKETS if lo <= margin < hi
                      or (hi == float("inf") and margin >= lo))
        # lowest-top viable index (geometry hedge member, precomputed)
        tops = st["_tops"]
        sl = shortlists(iv, sv, tops, n)
        n_eval += 1
        for nm in names:
            members = sl[nm]
            a = agg[nm]
            a["n"] += 1
            a["sizes"].append(len(members))
            hit = t_best in members
            a["cover"] += int(hit)
            src = a["by_source"].setdefault(st["source"], {"n": 0, "cover": 0})
            src["n"] += 1
            src["cover"] += int(hit)
            bkt = a["by_bucket"].setdefault(bucket, {"n": 0, "cover": 0})
            bkt["n"] += 1
            bkt["cover"] += int(hit)
            if not hit:
                q = float(tv[members[0]]) if members else float("-inf")
                if q <= DEAD_PENALTY / 2:
                    a["dead_miss"] += 1
                else:
                    a["regret"].append(best - q)
    print(f"\n{'shortlist':>12s} {'cover':>7s} {'miss':>5s} "
          f"{'reg|miss':>8s} {'dead':>5s} {'size':>6s} "
          f"{'bank':>7s} {'stud':>7s}")
    out_variants = {}
    for nm in names:
        a = agg[nm]
        cov = a["cover"] / max(1, a["n"])
        reg = statistics.fmean(a["regret"]) if a["regret"] else 0.0
        size = statistics.fmean(a["sizes"]) if a["sizes"] else 0.0
        bc = a["by_source"].get("bank", {"n": 0, "cover": 0})
        sc = a["by_source"].get("student", {"n": 0, "cover": 0})
        bcover = bc["cover"] / max(1, bc["n"])
        scover = sc["cover"] / max(1, sc["n"])
        print(f"{nm:>12s} {cov:7.4f} {a['n']-a['cover']:5d} "
              f"{reg:8.2f} {a['dead_miss']:5d} {size:6.2f} "
              f"{bcover:7.4f} {scover:7.4f}")
        out_variants[nm] = {
            "coverage": round(cov, 4), "n": a["n"],
            "n_miss": a["n"] - a["cover"],
            "regret_miss_mean": round(reg, 3),
            "regret_miss_max": round(max(a["regret"]), 3) if a["regret"] else 0.0,
            "dead_miss": a["dead_miss"],
            "mean_size": round(size, 3),
            "bank_coverage": round(bcover, 4),
            "student_coverage": round(scover, 4),
            "by_bucket": {k: {"n": v["n"],
                              "coverage": round(v["cover"] / max(1, v["n"]), 4)}
                          for k, v in a["by_bucket"].items()},
        }
    res = {
        "incumbent": str(incumb_path),
        "incumbent_sha256": hashlib.sha256(
            incumb_path.read_bytes()).hexdigest(),
        "student": str(student_path),
        "student_sha256": hashlib.sha256(
            student_path.read_bytes()).hexdigest(),
        "bank": str(resolve(args.bank)),
        "n_eval": n_eval, "n_skip": n_skip, "n_alldead": n_alldead,
        "seed0": args.seed0, "replay_games": args.replay_games,
        "cap": args.cap, "gamma_ply": args.gamma_ply,
        "variants": out_variants,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "note": ("training seeds 26000+ replayed (reporting only); "
                 "v1_k8 bank coverage is the bar (§32.1.6: 1.000). "
                 "Closed-loop only if a union list wins at smaller size."),
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nWrote {op} in {res['wall_seconds']}s "
          f"(eval {n_eval}, skip {n_skip}, alldead {n_alldead})")


if __name__ == "__main__":
    main()
