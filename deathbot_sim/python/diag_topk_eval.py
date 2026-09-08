"""Conclusive top-k shortlist evaluation (§27.3 follow-up, 2026-09-08).

The §27.3 probe compared closed-loop beam-8/beam-5 against the beam-100
champion while `TwoPlyScorer.choose` had a COMPARISON BUG: only the top-k
candidates were expanded to `v1 + gamma*max(v2)` but the final argmax ran
over ALL candidates (unexpanded ones keep raw v1). Since max(v2) is almost
always negative on this value scale, any 0<beam<n systematically PLAYED the
rank-(k+1) move — beam5/beam8 deaths at 30-39 pieces were the bug, not
evidence about v1-top-k coverage. Fixed in two_ply.py (decision restricted
to the expanded shortlist); shortlist_mode adds diverse/v1_safe hedges.

This driver now measures the FIXED variants conclusively:

Stage 1 (frozen bank, states 0..--states):
  agreement with full-expansion champion + positive regret (§20.3 metrics:
  never average sentinels) + wall time per decision. A variant with
  agreement ~1 and ~0 regret is behaviorally indistinguishable from the
  champion at a fraction of the cost — that is the speed prize.

Stage 2 (closed-loop games):
  cap-rate / pieces / lines for champion vs top-k variants on fresh tapes.
  Survival gate: cap rate must match the champion (§13/§15 protocol).

Panels: frozen bank for stage 1 (always allowed); stage 2 defaults to
REPORTING seeds 31000+ (demo-only range) unless --seed0 is passed; pass a
CONSUMED-free range explicitly to make a selection decision.

Usage:
    python diag_topk_eval.py --weights ../../model_weights_snapshot9.pt \
        --states 150 --stage2-games 8 --seed0 31000 --cap 1500 --workers 4 \
        --out ../_diag_out/topk_eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import piece_to_index  # noqa: E402
from afterstate import flat_rows_to_tensor  # noqa: E402
from two_ply import DEAD_PENALTY  # noqa: E402

DEMO_SEED0 = 31000  # reporting-only range (§28 rule 1)


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


# ---- Stage 1: frozen-bank agreement / regret / cost -------------------------

def stage1(args, weights: str) -> dict:
    from diag_phase2 import Bank
    from model import TAPNet
    from two_ply import TwoPlyScorer
    import deathbot_sim as ds

    bank = Bank(str(Path(args.bank)))
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    placed, nxt = bank._replay_piece_ids()
    n = min(args.states, bank.n_states)

    champ = TwoPlyScorer(net, beam=10 ** 9)
    variants = {}
    for k in args.beams:
        for mode in args.modes:
            variants[f"b{k}_{mode}"] = TwoPlyScorer(
                net, beam=k, shortlist_mode=mode)

    # Teacher vector per state for regret (fast-path champion semantics).
    def teacher_scores(pre, kind, nk):
        bulk = pre.lock_candidates_full(kind)
        nv = len(bulk[0])
        if nv == 0:
            return None, None
        lockouts = list(bulk[3])
        viable = [i for i, lk in enumerate(lockouts) if not lk]
        if not viable:
            return None, None
        rows = np.asarray(bulk[4], dtype=np.uint32).reshape(nv, 21)
        cur = torch.full((len(viable),), piece_to_index(kind), dtype=torch.long)
        nxtt = torch.full((len(viable),), piece_to_index(nk), dtype=torch.long)
        boards = flat_rows_to_tensor(rows[viable], len(viable))
        fast = getattr(net, "score_batch_fast", None)
        with torch.inference_mode():
            v1 = (fast(boards, cur) if fast is not None
                  else net(boards, cur, cur)).flatten().tolist()
        succ_rows, counts = [], []
        mk = getattr(ds.Board, "from_row_bits_bulk", None)
        succs = (mk([int(v) for vv in rows[viable] for v in vv])
                 if mk is not None else
                 [ds.Board.from_row_bits([int(v) for v in rows[i]])
                  for i in viable])
        for b in succs:
            r2 = b.lock_candidates_full(nk)
            n2 = len(r2[0])
            if n2:
                rr = np.asarray(r2[4], dtype=np.uint32).reshape(n2, 21)
                keep = [i for i in range(n2) if not r2[3][i]]
                counts.append(len(keep))
                succ_rows.extend(rr[i] for i in keep)
            else:
                counts.append(0)
        final = list(v1)
        if succ_rows:
            big = flat_rows_to_tensor(np.stack(succ_rows), len(succ_rows))
            nt = torch.full((len(succ_rows),), piece_to_index(nk),
                            dtype=torch.long)
            with torch.inference_mode():
                allv = (fast(big, nt) if fast is not None
                        else net(big, nt, nt)).flatten().tolist()
            off = 0
            for j in range(len(viable)):
                if counts[j] == 0:
                    final[j] = DEAD_PENALTY
                else:
                    seg = allv[off:off + counts[j]]
                    off += counts[j]
                    final[j] = v1[j] + args.gamma_ply * max(seg)
        else:
            for j in range(len(viable)):
                final[j] = DEAD_PENALTY
        return viable, final

    agree = {name: 0 for name in variants}
    reg_dis: dict[str, list] = {name: [] for name in variants}
    reg_all: dict[str, list] = {name: [] for name in variants}
    champ_t: list[float] = []
    var_t: dict[str, list] = {name: [] for name in variants}
    picks_by = {name: {} for name in variants}

    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        kind, nk = placed[s], nxt[s]
        viable, tv = teacher_scores(pre, kind, nk)
        if tv is None:
            continue
        t0 = time.perf_counter()
        champ_pick = champ.choose(pre, kind, nk)
        champ_t.append(time.perf_counter() - t0)
        best = max(tv)
        pos_of = {cand: j for j, cand in enumerate(viable)}
        cp_pos = next((j for j, cand in enumerate(viable)
                       if cand == champ_pick), None)
        for name, sc in variants.items():
            t0 = time.perf_counter()
            pick = sc.choose(pre, kind, nk)
            var_t[name].append(time.perf_counter() - t0)
            picks_by[name][s] = pick
            if pick == champ_pick:
                agree[name] += 1
                continue
            p_pos = pos_of.get(pick)
            if p_pos is None:
                reg = best  # lockout pick: regret == full gap (never neg)
            else:
                q = float(tv[p_pos])
                reg = best - q if q > DEAD_PENALTY / 2 else best
            reg_dis[name].append(reg)
            reg_all[name].append(reg)

    rows = {
        "states": n,
        "champ_sec_per_decision": round(statistics.fmean(champ_t) * 1000, 1),
        "variants": {},
    }
    print(f"\n{'variant':>14s} {'agree':>7s} {'n_dis':>6s} "
          f"{'reg|dis':>8s} {'reg|dis p90':>11s} {'ms/dec':>7s} {'speedup':>8s}")
    champ_ms = statistics.fmean(champ_t) * 1000
    for name in variants:
        ag = agree[name] / max(1, n)
        dis = reg_dis[name]
        allr = reg_all[name]
        ms = statistics.fmean(var_t[name]) * 1000 if var_t[name] else 0.0
        p90 = float(np.percentile(dis, 90)) if dis else 0.0
        rows["variants"][name] = {
            "agreement": round(ag, 4),
            "n_disagree": len(dis),
            "reg_dis_mean": round(statistics.fmean(dis), 3) if dis else 0.0,
            "reg_dis_p90": round(p90, 3),
            "reg_dis_max": round(max(dis), 3) if dis else 0.0,
            "reg_all_mean": round(statistics.fmean(allr), 3) if allr else 0.0,
            "ms_per_decision": round(ms, 2),
            "speedup_vs_full": round(champ_ms / ms, 2) if ms else None,
            "per_state_picks": picks_by[name],
        }
        print(f"{name:>14s} {ag:7.3f} {len(dis):6d} "
              f"{rows['variants'][name]['reg_dis_mean']:8.2f} {p90:11.2f} "
              f"{ms:7.1f} {rows['variants'][name]['speedup_vs_full']:7.2f}x")
    return rows


# ---- Stage 2: closed-loop games ---------------------------------------------

def _game_worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, name, beam, mode, seed, cap) = payload
    sys.path.insert(0, here)
    import random
    from model import TAPNet
    from two_ply import play_game_2ply_style
    net = TAPNet(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    t0 = time.perf_counter()
    pieces, lines, clears, reason = play_game_2ply_style(
        net, random.Random(seed), beam=beam, shortlist_mode=mode,
        max_pieces=cap)
    return {"config": name, "seed": seed, "pieces": pieces, "lines": lines,
            "clears": {str(k): int(v) for k, v in clears.items()},
            "reason": reason, "s": round(time.perf_counter() - t0, 1)}


def stage2(args, weights: str) -> dict:
    configs = [("champ", 10 ** 9, "v1")]
    for k in args.beams:
        for mode in args.modes:
            configs.append((f"b{k}_{mode}", k, mode))
    seeds = [args.seed0 + i for i in range(args.stage2_games)]
    if args.seed0 < DEMO_SEED0:
        print(f"[stage2] seed0 {args.seed0} < demo range {DEMO_SEED0}: "
              f"treating as a SELECTION panel (registry rules apply).")
    else:
        print(f"[stage2] reporting-only seeds {seeds[0]}-{seeds[-1]} "
              f"(demo range {DEMO_SEED0}+).")
    payloads = [(str(HERE), weights, nm, b, m, s, args.cap)
                for (nm, b, m) in configs for s in seeds]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: dict[str, list] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_game_worker, p) for p in payloads]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.setdefault(r["config"], []).append(r)
            done += 1
            print(f"  stage2 {done}/{len(payloads)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    base = {r["seed"]: r["pieces"] for r in results["champ"]}
    out = {"seed0": args.seed0, "games": args.stage2_games, "cap": args.cap,
           "selection_panel": args.seed0 < DEMO_SEED0, "configs": {}}
    print(f"\n{'config':>14s} {'cap':>6s} {'pcs':>7s} {'lines':>7s} "
          f"{'tet':>4s} {'s/g':>6s} {'dPieces':>16s}")
    for (nm, _b, _m) in configs:
        rows = sorted(results[nm], key=lambda r: r["seed"])
        pcs = [r["pieces"] for r in rows]
        capn = sum(1 for p in pcs if p >= args.cap)
        tet = sum(int(r["clears"].get("4", 0)) for r in rows)
        d = [r["pieces"] - base[r["seed"]] for r in rows if nm != "champ"]
        if d:
            md = statistics.fmean(d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
            dtxt = f"{md:+7.1f} t{md / se:+5.2f}" if se else f"{md:+7.1f} t—"
        else:
            dtxt = "—"
        print(f"{nm:>14s} {capn:3d}/{len(rows):<3d} "
              f"{statistics.fmean(pcs):7.1f} "
              f"{statistics.fmean([r['lines'] for r in rows]):7.1f} "
              f"{tet:4d} {statistics.fmean([r['s'] for r in rows]):6.1f} "
              f"{dtxt:>16s}")
        out["configs"][nm] = {
            "capped": capn, "mean_pieces": statistics.fmean(pcs),
            "mean_lines": statistics.fmean([r["lines"] for r in rows]),
            "tetrises": tet, "sec_per_game": round(statistics.fmean(
                [r["s"] for r in rows]), 1),
            "per_seed": {str(r["seed"]): {"pieces": r["pieces"],
                                          "lines": r["lines"],
                                          "clears": r["clears"],
                                          "reason": r["reason"]}
                         for r in rows},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bank", default="../_diag_out/"
                                       "state_bank_snapshot9_32games.npz")
    ap.add_argument("--states", type=int, default=150)
    ap.add_argument("--beams", nargs="+", type=int, default=[5, 8, 16])
    ap.add_argument("--modes", nargs="+",
                    default=["v1", "diverse", "v1_safe"])
    ap.add_argument("--gamma-ply", type=float, default=0.999)
    ap.add_argument("--stage2-games", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=DEMO_SEED0)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--skip-stage2", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = str(resolve_weights(args.weights))
    out: dict = {"weights": weights,
                 "weights_sha256": hashlib.sha256(
                     Path(weights).read_bytes()).hexdigest(),
                 "gamma_ply": args.gamma_ply}
    if not args.skip_stage1:
        print("=== stage 1: frozen-bank agreement / regret / cost ===",
              flush=True)
        out["stage1"] = stage1(args, weights)
    if not args.skip_stage2:
        print("\n=== stage 2: closed-loop games ===", flush=True)
        out["stage2"] = stage2(args, weights)
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {op}")


if __name__ == "__main__":
    main()
