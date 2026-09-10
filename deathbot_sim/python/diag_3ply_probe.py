"""Phase 3 (§44): top-k 8x5 3-ply frozen-bank probe (OFFLINE, NO panels, NO training).

Narrowed per review §44.1 D:
  - states from CONSEQUENTIAL moments, not random bank rows: preservation
    conflicts (base pick excluded from kept), cash-out opportunities
    (reachable Tetris present), re-entry choices (uncommitted->committed
    edge), pre-death recovery states. Collected via live replays on
    REPORTING seeds only (26000-26002 + 20851/20857 death tapes — all
    reporting/training, never selection).
  - preview-correct: ply-1 = current (known), ply-2 = true preview (known),
    ply-3 = TAP-randomizer suffix samples beyond the preview, with COMMON
    random numbers across roots (same suffix pieces per k) and AVERAGED
    continuation returns across K suffixes (never max over samples — that
    would pick lucky queues, not expected value).
  - history-correct suffixes: PieceQueue history deque + RNG state captured
    per state; suffix pieces drawn through the real PieceHistory reroll
    logic from that history.

Teacher: snapshot9 bonus-free full 2-ply (fast-path, gamma 0.999) = Q2.
3-ply: Q3(a) = v1(a) + bonus(a) + gamma * E_k[ max_b ( v(s',b) + bonus(b)
    + gamma * max_c v(s'',c_k) ) ], b over nxt placements, c over suffix-k
    placements. Top-8 roots by Q2 (full Q2 computed for all viable roots),
    top-5 b per root by v1 (bounded cost), K=2 suffixes.
Cost bound: PROMOTE to Phase 4 only if 3-ply labeling cost <= ~3x 2-ply
labeling AND disagreement concentrates where 2-ply is demonstrably wrong
(large-margin flips with a survival/style mechanism, not near-tie noise).

Usage:
    python diag_3ply_probe.py --weights ../../model_weights_snapshot9.pt \
        --states 120 --suffixes 2 --out ../_diag_out/3ply_probe.json
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
from collections import deque
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
DEAD = -1e6
GAMMA = 0.999
K_ROOTS = 8
K_B = 5

def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup, HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(p)

def collect_states(net, device, seeds_caps, ctrl_kwargs, max_states=120):
    """Replay controller on reporting tapes; capture consequential states
    with (board rows, cur, nxt, history list, rng state)."""
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer, preserves_well
    from well_potential import dense_well_features
    from piece_history import PieceQueue, PieceHistory, ALL_PIECES
    import deathbot_sim as ds
    base = TwoPlyScorer(net, beam=10**9, line_bonus=dict(COMPLETION_BONUS))
    states = []
    for seed, cap in seeds_caps:
        ctrl = CommitScorer(base, **ctrl_kwargs)
        pq = PieceQueue(random.Random(seed))
        board = ds.Board()
        pieces = 0
        prev_committed = True
        while pieces < cap and len(states) < max_states:
            bulk = board.lock_candidates_full(pq.current)
            n = len(bulk[0])
            if n == 0 or all(bulk[3]):
                break
            (st, xs, ys, lockouts, flat_rows, lines) = bulk
            viable = [i for i, lo in enumerate(lockouts) if not lo]
            rows = np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)
            pre_rows = np.asarray(board.row_bits(), dtype=np.uint32)
            occ = sum(bin(int(v)).count("1") for v in pre_rows)
            top = 0
            for r in range(20, -1, -1):
                if int(pre_rows[r]) & 0b1111111111:
                    top = r+1; break
            # controller bookkeeping (mirror)
            if ctrl.committed and (top >= ctrl.abort_top or occ >= ctrl.abort_occ):
                ctrl.committed = False; ctrl.stats["recovers"] += 1; ctrl._cooldown_left = ctrl.cooldown
            elif not ctrl.committed:
                if ctrl._cooldown_left > 0: ctrl._cooldown_left -= 1
                if occ <= ctrl.reenter_occ:
                    if ctrl.cooldown and ctrl._cooldown_left > 0:
                        ctrl.stats["reentry_denied_cool"] += 1
                    elif ctrl.reenter_top is not None and top > ctrl.reenter_top:
                        ctrl.stats["reentry_denied_top"] += 1
                    else:
                        if ctrl._usable_channel(rows, viable):
                            ctrl.committed = True; ctrl.stats["reentries"] += 1
                        else:
                            ctrl.stats["reentry_denied_top"] += 1
            edge = None
            if prev_committed and not ctrl.committed: edge = "entry"
            elif not prev_committed and ctrl.committed: edge = "reentry"
            prev_committed = ctrl.committed
            # classify consequential
            kind = None
            base_pick = base.choose(board, pq.current, pq.next_piece)
            if ctrl.committed:
                kept = []
                for i in viable:
                    try: f = dense_well_features(rows[i])
                    except Exception: continue
                    if preserves_well(pre_rows, rows[i], f, ctrl.max_new_holes):
                        kept.append(i)
                conflict = bool(kept) and (base_pick not in kept)
                cash = any(int(lines[i])==4 for i in viable)
                try:
                    cash = cash and any(int(lines[i])==4 and dense_well_features(rows[i])["tetris_ready"]>0.5 for i in viable)
                except Exception: pass
                if conflict: kind = "conflict"
                elif cash: kind = "cashout"
                elif edge == "reentry": kind = "reentry"
            else:
                if edge == "entry" or top >= 12: kind = "recovery"
            pick = ctrl.choose(board, pq.current, pq.next_piece)
            if kind and len(states) < max_states:
                states.append({"seed": seed, "pos": pieces, "kind": kind,
                    "pre_rows": pre_rows.copy(), "cur": str(pq.current),
                    "cur_id": pq.current, "nxt_id": pq.next_piece,
                    "hist": list(pq.hist.history),
                    "rngstate": pq.hist.rng.getstate(),
                    "occ": occ, "top": top})
            rows_pick = rows[pick]
            board = ds.Board.from_row_bits([int(v) for v in rows_pick])
            pq.advance(); pieces += 1
    return states

def draw_suffix(hist_list, rng_state, k_idx, state_tag):
    """Deterministic suffix piece via real TAP reroll logic, common across roots.

    Reconstructs a PieceHistory from the captured (history, RNG state),
    then draws ONE piece through the real next() reroll path. Same suffix
    for every root (common random numbers); different k give different
    suffixes, averaged downstream (never maximized)."""
    from piece_history import PieceHistory
    h = PieceHistory.__new__(PieceHistory)
    h.rng = random.Random()
    h.rng.setstate(rng_state)
    h.history = deque(hist_list, maxlen=4)
    _ = (k_idx, state_tag)  # distinct k handled by sequential draws
    return h.next()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--suffixes", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = str(resolve(args.weights))
    from model import TAPNet, piece_to_index
    from afterstate import flat_rows_to_tensor
    import deathbot_sim as ds
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(w, map_location="cpu")); net.eval()
    device = "cpu"
    t0 = time.perf_counter()
    ctrl_kwargs = {"abort_top":16,"abort_occ":110,"reenter_occ":60,"cooldown":30,"lam":10.0}
    print("[3ply] collecting consequential states (reporting tapes only)...", flush=True)
    states = collect_states(net, device, [(20851,1500),(20857,1500),(26000,300),(26001,300),(26002,300)],
                            ctrl_kwargs, max_states=args.states)
    kinds = {}
    for s in states: kinds[s["kind"]] = kinds.get(s["kind"],0)+1
    print(f"[3ply] {len(states)} states: {kinds}", flush=True)
    # label each state: full Q2 for all viable roots + Q3 for top-8
    results = []
    t_q2 = t_q3 = 0.0
    for si, s in enumerate(states):
        board = ds.Board.from_row_bits([int(v) for v in s["pre_rows"]])
        cur, nxt = s["cur_id"], s["nxt_id"]
        bulk = board.lock_candidates_full(cur)
        (st, xs, ys, lockouts, flat_rows, lines) = bulk
        n = len(st)
        viable = [i for i, lo in enumerate(lockouts) if not lo]
        if not viable:
            continue
        rows = np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)
        # Q2 full (replicate TwoPlyScorer exactly, bonus-free surviving branches)
        tq = time.perf_counter()
        cur_idx = piece_to_index(cur); nxt_idx = piece_to_index(nxt)
        sub = rows[viable]
        boards = flat_rows_to_tensor(sub.reshape(-1), len(viable)).to(device)
        cur_t = torch.full((len(viable),), cur_idx, dtype=torch.long, device=device)
        fast = getattr(net, "score_batch_fast", None)
        with torch.inference_mode():
            v1 = (fast(boards, cur_t).flatten().tolist() if fast is not None
                  else net(boards, cur_t, cur_t).flatten().tolist())
        # ply-2 expansion for all viable (for exact Q2)
        mk = ds.Board.from_row_bits_bulk if hasattr(ds.Board, "from_row_bits_bulk") else None
        succs = mk([int(v) for v in np.concatenate([np.asarray(rows[i],dtype=np.uint32) for i in viable]).astype(np.uint32).tolist()]) if mk else [ds.Board.from_row_bits([int(t) for t in rows[i]]) for i in viable]
        branch_rows, counts, flags = [], [], []
        for b in succs:
            b2 = b.lock_candidates_full(nxt)
            n2 = len(b2[0])
            if n2 == 0:
                flags.append(False); counts.append(0); continue
            r2 = np.asarray(b2[4], dtype=np.uint32).reshape(n2, 21)
            keep = [i for i in range(n2) if not b2[3][i]]
            if not keep:
                flags.append(True); counts.append(0); continue
            flags.append(True); counts.append(len(keep))
            for i in keep: branch_rows.append(r2[i])
        Q2 = {}
        if branch_rows:
            big = flat_rows_to_tensor(np.stack(branch_rows), len(branch_rows)).to(device)
            nxt_t = torch.full((len(branch_rows),), nxt_idx, dtype=torch.long, device=device)
            with torch.inference_mode():
                all_v = (fast(big, nxt_t).flatten().tolist() if fast is not None
                         else net(big, nxt_t, nxt_t).flatten().tolist())
            off = 0
            for pos, vi in enumerate(viable):
                if counts[pos] == 0:
                    Q2[vi] = DEAD
                else:
                    seg = all_v[off:off+counts[pos]]; off += counts[pos]
                    Q2[vi] = v1[pos] + float(COMPLETION_BONUS.get(int(lines[vi]),0.0)) + GAMMA*max(seg)
        else:
            for vi in viable: Q2[vi] = DEAD
        t_q2 += time.perf_counter()-tq
        order2 = sorted(viable, key=lambda i: (-Q2[i], i))
        q2_pick = order2[0]
        q2_margin = float(Q2[order2[0]] - Q2[order2[1]]) if len(order2)>1 else 0.0
        # Q3 on top-8 roots by Q2
        tq3 = time.perf_counter()
        top8 = order2[:K_ROOTS]
        # suffix pieces: K SEQUENTIAL draws from the reconstructed history
        # (draw k+1 consumes the first k, matching live TAP queue order).
        sufs = []
        for k in range(args.suffixes):
            from piece_history import PieceHistory
            hh = PieceHistory.__new__(PieceHistory)
            hh.rng = random.Random()
            hh.rng.setstate(s["rngstate"])
            hh.history = deque(s["hist"], maxlen=4)
            suf = None
            for _ in range(k + 1):
                suf = hh.next()
            sufs.append(suf)
        Q3 = {}
        for vi in top8:
            # ply-2 candidates of root vi under nxt
            bsucc = ds.Board.from_row_bits([int(t) for t in rows[vi]])
            b2 = bsucc.lock_candidates_full(nxt)
            n2 = len(b2[0])
            if n2 == 0:
                Q3[vi] = DEAD; continue
            r2 = np.asarray(b2[4], dtype=np.uint32).reshape(n2, 21)
            keep2 = [i for i in range(n2) if not b2[3][i]]
            if not keep2:
                Q3[vi] = DEAD; continue
            # top-5 b by v1 (bounded)
            bb = flat_rows_to_tensor(r2[keep2].reshape(-1), len(keep2)).to(device)
            nt = torch.full((len(keep2),), nxt_idx, dtype=torch.long, device=device)
            with torch.inference_mode():
                v_b = (fast(bb, nt).flatten().tolist() if fast is not None
                       else net(bb, nt, nt).flatten().tolist())
            b_rank = sorted(range(len(keep2)), key=lambda j: (-v_b[j], j))[:K_B]
            # E over suffixes of max_b continuation
            cont = []
            for suf in sufs:
                suf_idx = piece_to_index(suf)
                best_b = None
                for j in b_rank:
                    r2j = r2[keep2[j]]
                    cboard = ds.Board.from_row_bits([int(t) for t in r2j])
                    c3 = cboard.lock_candidates_full(suf)
                    n3 = len(c3[0])
                    if n3 == 0:
                        val = DEAD
                    else:
                        r3 = np.asarray(c3[4], dtype=np.uint32).reshape(n3, 21)
                        keep3 = [i for i in range(n3) if not c3[3][i]]
                        if not keep3:
                            val = DEAD
                        else:
                            cb = flat_rows_to_tensor(r3[keep3].reshape(-1), len(keep3)).to(device)
                            st3 = torch.full((len(keep3),), suf_idx, dtype=torch.long, device=device)
                            with torch.inference_mode():
                                v_c = (fast(cb, st3).flatten().tolist() if fast is not None
                                       else net(cb, st3, st3).flatten().tolist())
                            val = v_b[j] + float(COMPLETION_BONUS.get(int(b2[5][keep2[j]]),0.0)) + GAMMA*max(v_c)
                    if best_b is None or val > best_b: best_b = val
                cont.append(best_b if best_b is not None else DEAD)
            live = [c for c in cont if c > DEAD/2]
            if not live:
                Q3[vi] = DEAD
            else:
                # Q3 = v1 + bonus + gamma * mean continuation
                pos = viable.index(vi)
                Q3[vi] = v1[pos] + float(COMPLETION_BONUS.get(int(lines[vi]),0.0)) + GAMMA*(sum(live)/len(live))
        t_q3 += time.perf_counter()-tq3
        order3 = sorted(top8, key=lambda i: (-Q3[i], i))
        q3_pick = order3[0]
        flip = int(q3_pick != q2_pick)
        # is q2 pick outside top8? no (top8 by Q2 contains it) — report Q3 rank of q2 pick
        q3_rank_of_q2 = order3.index(q2_pick)+1 if q2_pick in order3 else -1
        gap2 = float(Q2[q2_pick]-Q2[q3_pick]) if flip else 0.0
        gap3 = float(Q3[q3_pick]-Q3[q2_pick]) if flip else 0.0
        results.append({"state": si, "kind": s["kind"], "seed": s["seed"], "pos": s["pos"],
                        "occ": s["occ"], "top": s["top"], "viable": len(viable),
                        "q2_pick": int(q2_pick), "q3_pick": int(q3_pick), "flip": flip,
                        "q2_margin": round(q2_margin,3), "q2_gap": round(gap2,3), "q3_gap": round(gap3,3),
                        "q3_rank_of_q2": q3_rank_of_q2,
                        "sufs": [str(x) for x in sufs]})
        if (si+1) % 20 == 0:
            print(f"[3ply] {si+1}/{len(states)} flips={sum(r['flip'] for r in results)}", flush=True)
    nflip = sum(r["flip"] for r in results)
    print(f"[3ply] flips {nflip}/{len(results)}; Q2-label {t_q2:.0f}s, Q3-label {t_q3:.0f}s, ratio {t_q3/max(1e-9,t_q2):.2f}x")
    out = {"weights": w, "weights_sha256": hashlib.sha256(Path(w).read_bytes()).hexdigest(),
           "ctrl": ctrl_kwargs, "k_roots": K_ROOTS, "k_b": K_B, "suffixes": args.suffixes,
           "n_states": len(results), "n_flips": nflip,
           "q2_label_s": round(t_q2,1), "q3_label_s": round(t_q3,1),
           "cost_ratio": round(t_q3/max(1e-9,t_q2),3),
           "results": results, "wall_seconds": round(time.perf_counter()-t0,1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[3ply] wrote {args.out}")

if __name__ == "__main__":
    main()
