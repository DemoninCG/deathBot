"""Step-4 gate B (§44.1-B): controller-state sufficiency audit (NO panels, NO training).

Question: the decoupled style head sees only 8 candidate-afterstate
features. It does NOT receive BUILD/RECOVER mode, cooldown remaining, or
pre-state commitment context. A reversible controller can deliberately rank
similar afterstates differently in those modes, so labels from its
trajectories may be contradictory for that head. This is distinct from the
already-closed question of whether style labels can improve a 1-ply net.

Protocol (reporting tapes only — 26000s training seeds + 20851/20857 death
tapes, cap 300/1500, corrected cool30lam10 controller):
  For every committed preserved decision, compute the 8-dim style feature
  vector of the teacher (controller) pick vs the flat (1-ply argmax) pick.
  Group pairs by (committed mode, cooldown bucket, recovery reason):
    - mode: committed vs style-off is excluded (uncommitted == base policy,
      no teacher signal); committed only.
    - cooldown bucket: cooldown_left == 0 vs > 0.
    - recovery reason: pieces-since-last-recover bucket (fresh <=50,
      mid 51-200, settled >200) + occ/top bucket at decision time.
  Within each group AND across groups, measure:
    - collisions: near-identical delta-feature vectors (cosine > 0.95)
      with OPPOSING labels (teacher pick swaps sides).
    - opposing-label rate: fraction of pairs whose delta vector points
      opposite (cosine < 0) to the group's mean direction.
  Verdict bins (pre-registered):
    - WELL-POSED: opposing-label rate < 20% in every group AND no
      cross-group sign flips on the dominant direction -> proceed to
      commitment-shaping distillation with the plain decoupled head.
    - MODE-CONDITIONED: opposing labels concentrate by group (one group
      > 30% opposing while others < 10%, or dominant directions flip sign
      across groups) -> student must be mode-conditioned (or distinct
      BUILD/RECOVER heads) before any distill.
    - CONTRADICTORY: opposing > 30% everywhere -> labels are noise for
      this head; do NOT distill (close the distill line, keep inference).

Usage:
    python diag_sufficiency_audit.py --weights ../../model_weights_snapshot9.pt \
        --out ../_diag_out/sufficiency_audit.json
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

COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
CTRL_KW = {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60,
           "cooldown": 30, "lam": 10.0}
# (seed, cap): reporting/training only, never selection.
JOBS = [(26000, 300), (26001, 300), (26002, 300), (20851, 1500),
        (20857, 1500)]


def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup,
              HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(p)


def _cos(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_job(net, seed, cap):
    import random
    from two_ply import TwoPlyScorer
    from commit_controller import CommitScorer, preserves_well
    from well_potential import dense_well_features
    from style_head import style_features_from_rows
    from afterstate import flat_rows_to_tensor
    from model import piece_to_index
    from piece_history import PieceQueue
    import deathbot_sim as ds
    base = TwoPlyScorer(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS))
    ctrl = CommitScorer(base, **CTRL_KW)
    pq = PieceQueue(random.Random(seed))
    board = ds.Board()
    pairs = []  # (delta8, group_key, aux)
    pieces = 0
    last_recover_at = -10 ** 9
    net.eval()
    with torch.inference_mode():
        while pieces < cap:
            bulk = board.lock_candidates_full(pq.current)
            n = len(bulk[0])
            if n == 0 or all(bulk[3]):
                break
            (states, xs, ys, lockouts, flat_rows, lines) = bulk
            viable = [i for i, lo in enumerate(lockouts) if not lo]
            pre_rows = np.asarray(board.row_bits(), dtype=np.uint32)
            occ = sum(bin(int(v)).count("1") for v in pre_rows)
            top = 0
            for r in range(20, -1, -1):
                if int(pre_rows[r]) & 0b1111111111:
                    top = r + 1
                    break
            rows = np.asarray(flat_rows, dtype=np.uint32).reshape(n, 21)
            if ctrl.committed and (top >= ctrl.abort_top
                                   or occ >= ctrl.abort_occ):
                ctrl.committed = False
                ctrl.stats["recovers"] += 1
                ctrl._cooldown_left = ctrl.cooldown
                last_recover_at = pieces
            elif not ctrl.committed:
                if ctrl._cooldown_left > 0:
                    ctrl._cooldown_left -= 1
                if occ <= ctrl.reenter_occ:
                    if not (ctrl.cooldown and ctrl._cooldown_left > 0) and not (
                            ctrl.reenter_top is not None
                            and top > ctrl.reenter_top):
                        if ctrl._usable_channel(rows, viable):
                            ctrl.committed = True
                            ctrl.stats["reentries"] += 1
            if not ctrl.committed:
                ctrl.stats["style_off"] += 1
                pick = base.choose(board, pq.current, pq.next_piece)
            else:
                took_cash = None
                for i in viable:
                    if int(lines[i]) == 4:
                        try:
                            if dense_well_features(rows[i])["tetris_ready"] > 0.5:
                                took_cash = i
                                break
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
                        if preserves_well(pre_rows, rows[i], f,
                                          ctrl.max_new_holes):
                            kept.append(i)
                    if not kept:
                        ctrl.stats["fallbacks"] += 1
                        pick = base.choose(board, pq.current, pq.next_piece)
                    else:
                        pick = ctrl.choose(board, pq.current, pq.next_piece)
                        # Pair: controller pick vs exact 1-ply argmax.
                        cur_idx = piece_to_index(pq.current)
                        nxt_idx = piece_to_index(pq.next_piece)
                        sub = rows[viable]
                        boards = flat_rows_to_tensor(
                            sub.reshape(-1), len(viable)).to("cpu")
                        cur = torch.full((len(viable),), cur_idx,
                                         dtype=torch.long)
                        nxt = torch.full((len(viable),), nxt_idx,
                                         dtype=torch.long)
                        fast = getattr(net, "score_batch_fast", None)
                        if fast is not None:
                            v1 = fast(boards, cur).flatten().tolist()
                        else:
                            v1 = net(boards, cur, nxt).flatten().tolist()
                        flat_pos = max(range(len(viable)),
                                       key=lambda j: (v1[j], -j))
                        flat_bulk = viable[flat_pos]
                        if flat_bulk != pick:
                            f2 = style_features_from_rows(
                                rows[pick], int(lines[pick])).astype(float)
                            f1 = style_features_from_rows(
                                rows[flat_bulk],
                                int(lines[flat_bulk])).astype(float)
                            since_rec = pieces - last_recover_at
                            rec_bucket = ("fresh" if since_rec <= 50
                                          else "mid" if since_rec <= 200
                                          else "settled")
                            cool_bucket = ("cool" if ctrl._cooldown_left > 0
                                           else "free")
                            occ_bucket = ("lo" if occ <= 60
                                          else "mid" if occ <= 110
                                          else "hi")
                            pairs.append({
                                "delta": (f2 - f1).tolist(),
                                "group": f"{rec_bucket}/{cool_bucket}/{occ_bucket}",
                                "cool_left": int(ctrl._cooldown_left),
                                "since_rec": int(since_rec),
                                "occ": int(occ), "top": int(top),
                            })
            rows_pick = rows[pick]
            board = ds.Board.from_row_bits([int(v) for v in rows_pick])
            pq.advance()
            pieces += 1
    return {"seed": seed, "pieces": pieces, "pairs": pairs,
            "stats": dict(ctrl.stats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = str(resolve(args.weights))
    from model import TAPNet
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(w, map_location="cpu"))
    net.eval()
    t0 = time.perf_counter()
    jobs = []
    all_pairs = []
    for seed, cap in JOBS:
        print(f"[suff] seed {seed} cap {cap} ...", flush=True)
        r = run_job(net, seed, cap)
        print(f"  -> pieces {r['pieces']}, pairs {len(r['pairs'])}",
              flush=True)
        jobs.append({"seed": seed, "pieces": r["pieces"],
                     "n_pairs": len(r["pairs"]), "stats": r["stats"]})
        all_pairs.extend(r["pairs"])
    # Group analysis.
    groups: dict[str, list] = {}
    for p in all_pairs:
        groups.setdefault(p["group"], []).append(np.asarray(
            p["delta"], dtype=float))
    names = ["wheight", "holes", "holedepth", "cav", "wellcost", "spawn",
             "lines", "deepwell"]
    overall = np.stack([np.asarray(p["delta"], dtype=float)
                        for p in all_pairs]) if all_pairs else np.zeros((0, 8))
    mean_dir = overall.mean(0) if len(overall) else np.zeros(8)
    group_rows = []
    for g, vecs in sorted(groups.items()):
        M = np.stack(vecs)
        mu = M.mean(0)
        cos_to_overall = _cos(mu, mean_dir)
        cos_to_own = [_cos(v, mu) for v in vecs]
        opp_own = sum(1 for c in cos_to_own if c < 0) / max(1, len(vecs))
        cos_to_global = [_cos(v, mean_dir) for v in vecs]
        opp_global = sum(1 for c in cos_to_global if c < 0) / max(1, len(vecs))
        group_rows.append({
            "group": g, "n": len(vecs),
            "mean_delta": [round(float(x), 3) for x in mu.tolist()],
            "cos_group_mean_to_overall": round(cos_to_overall, 3),
            "opposing_vs_own": round(opp_own, 3),
            "opposing_vs_global": round(opp_global, 3)})
        print(f"[suff] {g} n={len(vecs)} cos2overall={cos_to_overall:+.3f} "
              f"oppOwn={opp_own:.3f} oppGlobal={opp_global:.3f}", flush=True)
    opp_all = (sum(1 for p in all_pairs
                   if _cos(np.asarray(p["delta"], dtype=float),
                           mean_dir) < 0) / max(1, len(all_pairs)))
    # Collision scan: near-identical deltas (cos > 0.95) with opposing
    # projections onto the global direction (sign flip).
    units = [v / (np.linalg.norm(v) + 1e-12) for v in
             (np.asarray(p["delta"], dtype=float) for p in all_pairs)]
    collisions = 0
    checked = 0
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if int(all_pairs[i]["occ"] / 20) != int(all_pairs[j]["occ"] / 20):
                continue
            checked += 1
            if float(units[i] @ units[j]) > 0.95:
                si = float(units[i] @ (mean_dir / (np.linalg.norm(mean_dir) + 1e-12)))
                sj = float(units[j] @ (mean_dir / (np.linalg.norm(mean_dir) + 1e-12)))
                if si * sj < 0:
                    collisions += 1
    if opp_all < 0.20 and all(r["opposing_vs_global"] < 0.20
                              for r in group_rows) and not any(
            r["cos_group_mean_to_overall"] < 0 for r in group_rows):
        verdict = "WELL-POSED"
    elif any(r["opposing_vs_global"] > 0.30 for r in group_rows) or any(
            r["cos_group_mean_to_overall"] < 0 for r in group_rows):
        verdict = "MODE-CONDITIONED"
    elif opp_all > 0.30:
        verdict = "CONTRADICTORY"
    else:
        verdict = "WELL-POSED"
    out = {"weights": w,
           "weights_sha256": hashlib.sha256(
               Path(w).read_bytes()).hexdigest(),
           "controller": dict(CTRL_KW), "jobs": jobs,
           "n_pairs": len(all_pairs),
           "feature_names": names,
           "overall_mean_delta": [round(float(x), 3)
                                  for x in mean_dir.tolist()],
           "opposing_vs_global_all": round(opp_all, 3),
           "groups": group_rows,
           "collision_checked": checked, "collisions": collisions,
           "verdict": verdict,
           "wall_seconds": round(time.perf_counter() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[suff] n={len(all_pairs)} opp={opp_all:.3f} "
          f"collisions={collisions}/{checked} -> {verdict}")
    print(f"[suff] wrote {args.out}")


if __name__ == "__main__":
    main()
