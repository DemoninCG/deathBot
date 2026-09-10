"""Audit A (§44.1): constrained-root scoring approximation check (NO panels, NO training).

Question: CommitScorer, when the unconstrained full-2-ply base pick violates
the well constraint, ranks the preserved set by 1-ply v1 (+ optional Phi)
instead of exact restricted Q_2ply = argmax_{a in kept} Q_2ply(a).
Does the approximation change consequential choices?

Protocol (reporting tapes only):
  - seeds 20851, 20857 (commit_loose death tapes, cap 1500 — reproduce deaths)
  - seeds 26000, 26001, 26002 (reporting/training seeds, cap 300 — cheap)
  - variants: cool30 (16/110/60+cooldown30) and cool30lam10 (+lam10)
  - for every COMMITTED decision with non-empty kept: record preserved size,
    whether base pick was excluded (conflict), approx pick (v1+phi) vs exact
    restricted Q2ply (+phi when lam), gaps, and game outcome context.

Exact Q2ply replicates TwoPlyScorer.choose ply-2 semantics exactly:
  final = v1 + completion_bonus + gamma*max(v2_next), dead -> DEAD_PENALTY,
  ply-2 scored with fast(big, nxt) (cur=nxt duplication, versioned champion).
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, random
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
DEAD_PENALTY = -1e6
GAMMA = 0.999

def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(p)

def exact_q2ply_for_kept(net, device, kept_rows, kept_lines, next_piece):
    """Exact restricted Q2ply for kept roots (bulk-indexed inputs)."""
    from afterstate import flat_rows_to_tensor
    from model import piece_to_index
    import deathbot_sim as ds
    n = len(kept_rows)
    if n == 0:
        return []
    sub = np.stack(kept_rows)  # (n,21)
    boards = flat_rows_to_tensor(sub.reshape(-1), n).to(device)
    # v1 needs current piece — caller passes cur via closure; here we recompute
    # v1 outside. This helper returns max(v2) per kept root + dead flags.
    nxt_idx = piece_to_index(next_piece)
    maxv2 = []
    dead = []
    # batch ply-2 across kept roots (same as TwoPlyScorer, per-root max)
    branch_rows = []
    counts = []
    flags = []
    mk = ds.Board.from_row_bits_bulk if hasattr(ds.Board, "from_row_bits_bulk") else None
    succs = []
    if mk is not None:
        flat = np.concatenate([np.asarray(r, dtype=np.uint32) for r in kept_rows]).astype(np.uint32).tolist()
        succs = mk([int(v) for v in flat])
    else:
        succs = [ds.Board.from_row_bits([int(t) for t in r]) for r in kept_rows]
    for b in succs:
        bulk2 = b.lock_candidates_full(next_piece)
        (s2, _x, _y, lo2, flat2, _ln) = bulk2
        n2 = len(s2)
        if n2 == 0:
            flags.append(False); counts.append(0); continue
        rows2 = np.asarray(flat2, dtype=np.uint32).reshape(n2, 21)
        keep = [i for i in range(n2) if not lo2[i]]
        if not keep:
            flags.append(True); counts.append(0); continue
        flags.append(True); counts.append(len(keep))
        for i in keep:
            branch_rows.append(rows2[i])
    if branch_rows:
        big = flat_rows_to_tensor(np.stack(branch_rows), len(branch_rows)).to(device)
        nxt_t = torch.full((len(branch_rows),), nxt_idx, dtype=torch.long, device=device)
        fast = getattr(net, "score_batch_fast", None)
        with torch.inference_mode():
            if fast is not None:
                all_v = fast(big, nxt_t).flatten().tolist()
            else:
                all_v = net(big, nxt_t, nxt_t).flatten().tolist()
        off = 0
        for pos in range(n):
            if counts[pos] == 0:
                maxv2.append(None); dead.append(True)
            else:
                seg = all_v[off:off+counts[pos]]; off += counts[pos]
                maxv2.append(max(seg)); dead.append(False)
    else:
        for _ in range(n):
            maxv2.append(None); dead.append(True)
    return maxv2, dead

def run_game(net, device, seed, cap, ctrl_kwargs):
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer, preserves_well
    from well_potential import dense_well_features, style_fn_factory
    from piece_history import PieceQueue
    import deathbot_sim as ds
    base = TwoPlyScorer(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS))
    ctrl = CommitScorer(base, **ctrl_kwargs)
    phi_fn = style_fn_factory(scale=ctrl_kwargs.get("lam", 0.0)) if ctrl_kwargs.get("lam", 0.0) else None
    pq = PieceQueue(random.Random(seed))
    board = ds.Board()
    # per-decision records (committed + kept non-empty only)
    recs = []
    pieces = 0
    reason = "max_pieces"
    from model import piece_to_index
    from afterstate import flat_rows_to_tensor
    while pieces < cap:
        bulk = board.lock_candidates_full(pq.current)
        n = len(bulk[0])
        if n == 0:
            reason = "no_candidates"; break
        if all(bulk[3]):
            reason = "lockout"; break
        # replicate CommitScorer.choose with instrumentation
        (states, xs, ys, lockouts, flat_rows, lines) = bulk
        viable = [i for i, lo in enumerate(lockouts) if not lo]
        pre_rows = np.asarray(board.row_bits(), dtype=np.uint32)
        occ = sum(bin(int(v)).count("1") for v in pre_rows)
        top = 0
        for r in range(20, -1, -1):
            if int(pre_rows[r]) & 0b1111111111:
                top = r + 1; break
        rows = np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)
        # recover/reentry bookkeeping (mirror CommitScorer)
        if ctrl.committed and (top >= ctrl.abort_top or occ >= ctrl.abort_occ):
            ctrl.committed = False; ctrl.stats["recovers"] += 1; ctrl._cooldown_left = ctrl.cooldown
        elif not ctrl.committed:
            if ctrl._cooldown_left > 0:
                ctrl._cooldown_left -= 1
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
        committed = ctrl.committed
        if not committed:
            ctrl.stats["style_off"] += 1
            pick = base.choose(board, pq.current, pq.next_piece)
        else:
            # cash-out
            took_cash = None
            for i in viable:
                if int(lines[i]) == 4:
                    try:
                        if dense_well_features(rows[i])["tetris_ready"] > 0.5:
                            took_cash = i; break
                    except Exception:
                        pass
            if took_cash is not None:
                ctrl.stats["cashouts"] += 1
                pick = took_cash
            else:
                kept = []
                for i in viable:
                    try:
                        f = dense_well_features(rows[i])
                    except Exception:
                        continue
                    if preserves_well(pre_rows, rows[i], f, ctrl.max_new_holes):
                        kept.append(i)
                if not kept:
                    ctrl.stats["fallbacks"] += 1
                    pick = base.choose(board, pq.current, pq.next_piece)
                else:
                    base_pick = base.choose(board, pq.current, pq.next_piece)
                    ctrl.stats["decisions"] += 1
                    if base_pick in kept:
                        ctrl.stats["preserved_decisions"] += 1
                        pick = base_pick
                    else:
                        # CONFLICT: approx (v1+phi) vs exact (Q2ply+phi)
                        cur_idx = piece_to_index(pq.current)
                        sub = rows[kept]
                        boards = flat_rows_to_tensor(sub.reshape(-1), len(kept)).to(device)
                        cur = torch.full((len(kept),), cur_idx, dtype=torch.long, device=device)
                        fast = getattr(net, "score_batch_fast", None)
                        with torch.inference_mode():
                            if fast is not None:
                                v1 = fast(boards, cur).flatten().tolist()
                            else:
                                v1 = net(boards, cur, cur).flatten().tolist()
                        phi = [0.0]*len(kept)
                        if phi_fn is not None:
                            try:
                                phi = list(phi_fn([rows[j] for j in kept], [0]*len(kept)))
                            except Exception:
                                pass
                        approx = [a+b for a, b in zip(v1, phi)]
                        approx_pos = max(range(len(kept)), key=lambda j: (approx[j], -j))
                        # exact Q2ply
                        kept_lines = [int(lines[j]) for j in kept]
                        maxv2, dead = exact_q2ply_for_kept(net, device, [rows[j] for j in kept], kept_lines, pq.next_piece)
                        exact = []
                        for j in range(len(kept)):
                            if dead[j]:
                                exact.append(DEAD_PENALTY)
                            else:
                                bonus = float(COMPLETION_BONUS.get(kept_lines[j], 0.0))
                                exact.append(v1[j] + bonus + GAMMA*maxv2[j] + phi[j])
                        exact_pos = max(range(len(kept)), key=lambda j: (exact[j], -j))
                        flip = int(approx_pos != exact_pos)
                        # gaps
                        order = sorted(range(len(kept)), key=lambda j: (-exact[j], j))
                        qgap = float(exact[order[0]] - exact[approx_pos]) if flip else 0.0
                        vgap = float(max(v1)-v1[approx_pos])
                        phigap = float(max(phi)-phi[approx_pos]) if phi_fn else 0.0
                        recs.append({"pos": pieces, "kept_size": len(kept),
                                     "viable_size": len(viable),
                                     "flip": flip, "qgap": round(qgap,3),
                                     "vgap": round(vgap,3), "phigap": round(phigap,3),
                                     "approx_bulk": int(kept[approx_pos]),
                                     "exact_bulk": int(kept[exact_pos]),
                                     "base_bulk": int(base_pick),
                                     "occ": int(occ), "top": int(top)})
                        ctrl.stats["preserved_decisions"] += 1
                        pick = kept[approx_pos]  # play the CURRENT approx (faithful replay)
        # advance
        ctrl.stats["decisions"] = ctrl.stats.get("decisions", 0)
        rows_pick = rows[pick]
        board = ds.Board.from_row_bits([int(v) for v in rows_pick])
        pq.advance()
        pieces += 1
    return {"seed": seed, "pieces": pieces, "reason": reason,
            "conflicts": recs, "stats": dict(ctrl.stats)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = str(resolve(args.weights))
    net_fn = __import__("model", fromlist=["TAPNet"]).TAPNet
    net = net_fn(output="linear")
    net.load_state_dict(torch.load(w, map_location="cpu")); net.eval()
    device = "cpu"
    variants = {
        "cool30": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30},
        "cool30lam10": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0},
    }
    # (seed, cap): death tapes full, reporting tapes cheap
    jobs = [(20851, 1500), (20857, 1500), (26000, 300), (26001, 300), (26002, 300)]
    out = {"weights": w, "weights_sha256": hashlib.sha256(Path(w).read_bytes()).hexdigest(),
           "jobs": [], "wall_seconds": 0.0}
    t0 = time.perf_counter()
    for vname, kw in variants.items():
        for seed, cap in jobs:
            print(f"[auditA] {vname} seed {seed} cap {cap} ...", flush=True)
            r = run_game(net, device, seed, cap, kw)
            r["variant"] = vname; r["cap"] = cap
            nconf = len(r["conflicts"])
            nflip = sum(c["flip"] for c in r["conflicts"])
            qgaps = [c["qgap"] for c in r["conflicts"] if c["flip"]]
            print(f"  -> pieces {r['pieces']} {r['reason']}, conflicts {nconf}, flips {nflip}, "
                  f"meanQgap {sum(qgaps)/len(qgaps) if qgaps else 0:.2f}", flush=True)
            # drop bulky per-decision approx/exact bulk ids? keep (small)
            out["jobs"].append(r)
    out["wall_seconds"] = round(time.perf_counter()-t0, 1)
    # summary
    for vname in variants:
        rows = [j for j in out["jobs"] if j["variant"]==vname]
        nc = sum(len(j["conflicts"]) for j in rows)
        nf = sum(sum(c["flip"] for c in j["conflicts"]) for j in rows)
        print(f"[auditA] {vname}: {nf}/{nc} flips ({100*nf/max(1,nc):.1f}%)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[auditA] wrote {args.out}")

if __name__ == "__main__":
    main()
