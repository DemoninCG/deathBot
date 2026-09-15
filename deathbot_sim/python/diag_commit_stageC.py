"""Stage C (conclusive, §44 Phase 1): cool30 vs cool30lam10 + cap-3000 leg.

Design (per §44 + review §44.1 C):
  - 30 FRESH tapes 20877-20906 (dev; CONSUMED), MATCHED across 4 arms:
    cool30 / cool30lam10 (contenders) + base_loose (rung-1 ref) + lam10 (style ref).
  - cap-1500 primary leg on ALL arms; cap-3000 leg on the SAME 30 tapes for
    cool30 vs cool30lam10 only (horizon confirmation, §32.3 item 6).
  - Endpoints: paired cap (McNemar-style discordant pairs + restricted-mean
    pieces), paired Tetris deltas, T/100, T per capped game, longest
    no-Tetris gap, mode transitions (recovers/reentries), constrained-pick /
    fallback / re-entry counts, recovery duration, off-frac.
  - Replay contract (§44.1 C, cheap form): per-seed piece-tape hash
    (PieceQueue draw log is deterministic from seed — record seed + policy
    config + per-step board hash chain head/tail + chosen bulk index).
    Full per-step artifacts are NOT stored (30 tapes x 1500 steps x 2 caps
    is GB-scale); the seed + pinned config + code version IS the replay.
    Board-hash chain (FNV over row bits per step, head+tail recorded)
    guards against future RNG-identity confusion (E0 lesson).

Promotion rule (pre-registered, §44): promote the winner as THE stylish
config iff cap-rate separates (paired t>=2 on pieces, or discordant-pair
exact p<=0.05) or style does (paired dTet t>=2 with cap>=28/30); else hold
cool30lam10 on the stage-B point estimate and CONTINUE (no rerun — n=15+30
pooled decides).

Usage:
    python diag_commit_stageC.py --weights ../../model_weights_snapshot9.pt \
        --games 30 --seed0 20877 --cap 1500 --cap2 3000 --workers 8 \
        --out ../_diag_out/commit_stageC.json
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STAGEC_SEED0 = 20877
REPORTING_SEED0 = 31200
COMPLETION_BONUS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
ARMS = {
    "base_loose": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60},
    "cool30": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30},
    "cool30lam10": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0},
    "cool30lam20": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 20.0},
    "cool30lam30": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 30.0},
    "watch80": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0, "progress_patience": 80},
    "reentry_cert": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "reenter_top": 10,
                     "reenter_holes": 2, "reentry_min_progress": 1.0, "cooldown": 30, "lam": 10.0},
    "a120_r50": {"abort_top": 16, "abort_occ": 120, "reenter_occ": 50, "cooldown": 30, "lam": 10.0},
    "cool30lam10tiered": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0, "tiered_preserve": True},
    "cool15lam10tiered": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 15, "lam": 10.0, "tiered_preserve": True},
    "cool15lam10tiered_burn4": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 15, "lam": 10.0, "tiered_preserve": True, "burn_penalty": 4.0},
    "cool20lam10tiered": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 20, "lam": 10.0, "tiered_preserve": True},
    "cool30lam10tiered_burn2": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0, "tiered_preserve": True, "burn_penalty": 2.0},
    "cool30lam10tiered_burn4": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0, "tiered_preserve": True, "burn_penalty": 4.0},
    "cool30lam10tiered_burn6": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0, "tiered_preserve": True, "burn_penalty": 6.0},
    "cool30tiered": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 0.0, "tiered_preserve": True},
    "cool30lam5tiered": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 5.0, "tiered_preserve": True},
    "cool30lam15tiered": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 15.0, "tiered_preserve": True},
    "cool30lam20tiered": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 20.0, "tiered_preserve": True},
    "tiered_lo5_hi15": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0, "tiered_preserve": True, "lam_lo": 5.0, "lam_hi": 15.0},
    "tiered_lo5_hi20": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0, "tiered_preserve": True, "lam_lo": 5.0, "lam_hi": 20.0},
    "tiered_lo10_hi20": {"abort_top": 16, "abort_occ": 110, "reenter_occ": 60, "cooldown": 30, "lam": 10.0, "tiered_preserve": True, "lam_lo": 10.0, "lam_hi": 20.0},
}

def resolve(p: str) -> Path:
    sup = Path(p).expanduser()
    for c in (sup, HERE.parent / sup, HERE.parent.parent / sup, HERE.parent.parent.parent / sup):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(p)

def _fnv(rows) -> int:
    h = 0x811C9DC5
    for v in rows:
        h ^= int(v) & 0xFFFFFFFF
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def _worker(payload) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    (here, weights, arm, seed, cap, window) = payload
    sys.path.insert(0, here)
    import random
    import numpy as _np
    from model import TAPNet as _N
    from two_ply import TwoPlyScorer as _S
    from well_potential import style_fn_factory as _ff
    from well_potential import dense_well_features as _dwf
    from commit_controller import CommitScorer as _C
    from piece_history import PieceQueue as _PQ
    import deathbot_sim as _ds
    net = _N(output="linear")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    if arm == "lam10":
        scorer = _S(net, beam=100, line_bonus=dict(COMPLETION_BONUS),
                    style_fn=_ff(scale=10.0), style_alpha=1.0)
        lam_style = True
    else:
        base = _S(net, beam=10 ** 9, line_bonus=dict(COMPLETION_BONUS))
        scorer = _C(base, **ARMS[arm])
        lam_style = False
    pq = _PQ(random.Random(seed))
    board = _ds.Board()
    clears = {0:0,1:0,2:0,3:0,4:0}
    lines_total = pieces = 0
    reason = "max_pieces"
    tetris_at = []
    windows = []
    def _new_acc(): return {"n":0,"lines":0,"tet":0,"occ":0,"top":0.0,"holes":0.0}
    def _flush(a):
        n = max(1, a["n"])
        return {"pieces":a["n"],"lines":a["lines"],"tetrises":a["tet"],
                "mean_occ":a["occ"]/n,"mean_top":a["top"]/n,"mean_holes":a["holes"]/n}
    acc = _new_acc()
    off_n = off_d = 0
    chain = 0x811C9DC5
    first_hash = None
    t0 = time.perf_counter()
    while pieces < cap:
        bulk = board.lock_candidates_full(pq.current)
        n_cands = len(bulk[0])
        if n_cands == 0:
            reason = "no_candidates"; break
        if all(bulk[3]):
            reason = "lockout"; break
        pick = scorer.choose(board, pq.current, pq.next_piece)
        rows = _np.asarray(bulk[4], dtype=_np.uint32).reshape(n_cands, 21)[pick]
        if first_hash is None:
            first_hash = _fnv(rows)
        chain ^= _fnv(rows); chain = (chain*0x01000193)&0xFFFFFFFF
        nl = int(bulk[5][pick])
        lines_total += nl; clears[nl] = clears.get(nl,0)+1
        if nl == 4: tetris_at.append(pieces+1)
        occ = sum(bin(int(v)).count("1") for v in rows)
        feats = _dwf(rows)
        acc["n"]+=1; acc["lines"]+=nl
        if nl==4: acc["tet"]+=1
        acc["occ"]+=occ; acc["top"]+=float(feats["top"]); acc["holes"]+=float(feats["holes_outside"])
        if not lam_style:
            off_d+=1; off_n+=int(not scorer.committed)
        board = _ds.Board.from_row_bits([int(v) for v in rows])
        pq.advance(); pieces+=1
        if pieces % window == 0:
            windows.append(_flush(acc)); acc=_new_acc()
    if acc["n"]: windows.append(_flush(acc))
    fin_rows = _np.asarray(board.row_bits(), dtype=_np.uint32)
    fin_feats = _dwf(fin_rows)
    # longest no-Tetris gap (in pieces, incl. edges)
    marks = [0]+tetris_at+[pieces]
    gap = max([marks[i+1]-marks[i] for i in range(len(marks)-1)] or [pieces])
    out = {"arm":arm,"seed":seed,"pieces":pieces,"lines":lines_total,
           "clears":{str(k):int(v) for k,v in clears.items()},"reason":reason,
           "windows":windows,"tetris_at":tetris_at,
           "longest_no_tetris_gap":gap,
           "final":{"occ":sum(bin(int(v)).count("1") for v in fin_rows),
                    "top":float(fin_feats["top"]),"holes":float(fin_feats["holes_outside"])},
           "board_chain_head": first_hash, "board_chain": chain,
           "s":round(time.perf_counter()-t0,1)}
    if not lam_style:
        out["style_off_frac"]=off_n/max(1,off_d)
        out["ctrl_stats"]=scorer.stats
    return out

def _paired_t(xs):
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs)>1 else 0.0
    se = sd/len(xs)**0.5 if len(xs)>1 else 0.0
    return m, (m/se if se else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=STAGEC_SEED0)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--cap2", type=int, default=3000)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", type=str, default="all",
                    help="comma-separated subset of base_loose,cool30,cool30lam10,cool30lam20,cool30lam30,lam10 (default all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = str(resolve(args.weights))
    if args.seed0 < STAGEC_SEED0:
        print(f"[stageC] seed0 {args.seed0} < {STAGEC_SEED0} refusing"); sys.exit(2)
    tag = "reporting-only" if args.seed0 >= REPORTING_SEED0 else f"SELECTION {args.seed0}-{args.seed0+args.games-1}"
    seeds = [args.seed0+i for i in range(args.games)]
    if args.arms.strip().lower() == "all":
        leg1_arms = list(ARMS)+["lam10"]
    else:
        leg1_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        for a in leg1_arms:
            if a != "lam10" and a not in ARMS:
                print(f"[stageC] unknown arm {a!r}"); sys.exit(2)
    print(f"[stageC] {tag}; leg1 cap {args.cap} arms {leg1_arms}; leg2 cap {args.cap2} cool30 vs cool30lam10.")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    def run_leg(arms, cap):
        res = {}
        pays = [(str(HERE), weights, a, s, cap, args.window) for a in arms for s in seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_worker, p) for p in pays]
            done = 0
            for f in as_completed(futs):
                r = f.result(); res.setdefault(r["arm"], []).append(r)
                done += 1
                if done % 20 == 0:
                    print(f"  cap{cap}: {done}/{len(pays)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        return res
    leg1 = run_leg(leg1_arms, args.cap)
    leg2 = run_leg(["cool30","cool30lam10"], args.cap2) if args.cap2 else {}
    out = {"weights":weights,"weights_sha256":hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
           "seed0":args.seed0,"games":args.games,"cap":args.cap,"cap2":args.cap2,
           "selection_panel": args.seed0 < REPORTING_SEED0,
           "arms_config": ARMS, "legs":{}, "wall_seconds":0.0}
    def summarize(res, cap):
        summ = {}
        ref = {r["seed"]: int(r["clears"].get("4",0)) for r in res.get("lam10", [])}
        refp = {r["seed"]: r["pieces"] for r in res.get("lam10", [])}
        # Head-to-head fallback when lam10 is absent (minimal decision run):
        # pair cool30lam10 directly against cool30 on the same seeds.
        h2h_tet = {}
        h2h_pcs = {}
        if not ref and "cool30" in res and "cool30lam10" in res:
            c30 = {r["seed"]: r for r in res["cool30"]}
            for r in res["cool30lam10"]:
                q = c30.get(r["seed"])
                if q is not None:
                    h2h_tet[r["seed"]] = int(r["clears"].get("4", 0)) - int(q["clears"].get("4", 0))
                    h2h_pcs[r["seed"]] = r["pieces"] - q["pieces"]
        for arm, rows0 in res.items():
            rows = sorted(rows0, key=lambda r: r["seed"])
            pcs = [r["pieces"] for r in rows]
            capn = sum(1 for p in pcs if p >= cap)
            tet = sum(int(r["clears"].get("4",0)) for r in rows)
            t100 = 100*tet/max(1,sum(pcs))
            capped_t = [int(r["clears"].get("4",0)) for r in rows if r["pieces"]>=cap]
            t_per_capped = (sum(capped_t)/len(capped_t)) if capped_t else 0.0
            gaps = [r["longest_no_tetris_gap"] for r in rows]
            d = [int(r["clears"].get("4",0))-ref[r["seed"]] for r in rows] if arm!="lam10" and ref else []
            dp = [r["pieces"]-refp[r["seed"]] for r in rows] if arm!="lam10" and refp else []
            if not ref and h2h_tet:
                if arm == "cool30lam10":
                    dd = [h2h_tet[s] for s in sorted(h2h_tet)]
                    ddp = [h2h_pcs[s] for s in sorted(h2h_pcs)]
                    mt, tt = _paired_t(dd)
                    mp, tp = _paired_t(ddp)
                elif arm == "cool30":
                    mt, tt, mp, tp = 0.0, 0.0, 0.0, 0.0
                    d = []
                    dp = []
            else:
                mt, tt = _paired_t(d) if d else (0.0, 0.0)
                mp, tp = _paired_t(dp) if dp else (0.0, 0.0)
            # discordant cap pairs vs lam10 (or head-to-head cool30lam10 vs cool30)
            disc = ""
            if not ref and h2h_pcs and arm == "cool30lam10":
                a_cap = {r["seed"]: (r["pieces"] >= cap) for r in rows}
                c_cap = {r["seed"]: (r["pieces"] >= cap)
                         for r in res.get("cool30", [])}
                disc = (f"{sum(1 for s in h2h_pcs if a_cap.get(s) and not c_cap.get(s))}"
                        f"/{sum(1 for s in h2h_pcs if (not a_cap.get(s)) and c_cap.get(s))}"
                        " vs cool30")
            if arm != "lam10" and refp:
                a_cap = {r["seed"]: (r["pieces"]>=cap) for r in rows}
                b_cap = {s: (refp[s]>=cap) for s in refp}
                n10 = sum(1 for s in refp if a_cap.get(s) and not b_cap.get(s))
                n01 = sum(1 for s in refp if (not a_cap.get(s)) and b_cap.get(s))
                disc = f"{n10}/{n01}"
            extra = {}
            if rows and "style_off_frac" in rows[0]:
                extra["style_off_frac"]=round(statistics.fmean(r["style_off_frac"] for r in rows),3)
                cs = [r["ctrl_stats"] for r in rows if r.get("ctrl_stats")]
                if cs:
                    extra["mean_recovers"]=round(statistics.fmean(c["recovers"] for c in cs),2)
                    extra["mean_reentries"]=round(statistics.fmean(c["reentries"] for c in cs),2)
                    extra["mean_fallbacks"]=round(statistics.fmean(c["fallbacks"] for c in cs),2)
                    extra["mean_cashouts"]=round(statistics.fmean(c["cashouts"] for c in cs),3)
            print(f"{arm:>12s} cap{cap}: {capn:3d}/{len(rows)} pcs{statistics.fmean(pcs):7.1f} "
                  f"tet{tet:4d} T/100 {t100:.3f} T/cap {t_per_capped:.2f} gap{statistics.fmean(gaps):.0f} "
                  f"dTet {mt:+.2f} t{tt:+.2f} dPc {mp:+.1f} t{tp:+.2f} disc {disc}")
            summ[arm]={"capped":capn,"mean_pieces":round(statistics.fmean(pcs),1),
                       "mean_lines":round(statistics.fmean(r["lines"] for r in rows),1),
                       "tetrises":tet,"t_per_100":round(t100,4),
                       "t_per_capped":round(t_per_capped,3),
                       "mean_longest_gap":round(statistics.fmean(gaps),1),
                       "paired_dTet":round(mt,3),"paired_dTet_t":round(tt,2),
                       "paired_dPieces":round(mp,1),"paired_dPieces_t":round(tp,2),
                       "discordant_cap_vs_lam10":disc,
                       "per_seed":{str(r["seed"]):{"pieces":r["pieces"],"lines":r["lines"],
                          "clears":r["clears"],"reason":r["reason"],"windows":r["windows"],
                          "tetris_at":r["tetris_at"],"longest_no_tetris_gap":r["longest_no_tetris_gap"],
                          "final":r["final"],"board_chain_head":r["board_chain_head"],
                          "board_chain":r["board_chain"],
                          **({"style_off_frac":r.get("style_off_frac"),"ctrl_stats":r.get("ctrl_stats")} if "style_off_frac" in r else {})} for r in rows},
                       **extra}
        return summ
    print("--- leg1 ---")
    out["legs"]["cap1500"]=summarize(leg1, args.cap)
    if leg2:
        print("--- leg2 ---")
        out["legs"]["cap3000"]=summarize(leg2, args.cap2)
    out["wall_seconds"]=round(time.perf_counter()-t0,1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[stageC] wrote {args.out}")

if __name__ == "__main__":
    main()
