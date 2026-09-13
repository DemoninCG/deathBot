"""Continuous self-distillation loop GEN-2 (2026-09-12 audit build).

What changed vs gen-1 (`_selfdistill_loop.py`):
  1. MIXED-BANK FITS: 70% base bank (snapshot9-era normal games) + 30%
     hard-start pool (`_collect_hardstart_pool.py` output — champion
     continuations from recovery-suite start boards, game indices 100+,
     never overlapping the suite's eval boards 0-19). Gen-1 learned
     nothing about recovery states (audit §52.6 item 2); gen-2 does.
  2. DUAL SCREEN per cycle (both paired, both must pass):
       survival: 30 fresh dev tapes cap 3000 vs pre-burst (same as gen-1)
       recovery: 4 suites (flat15/flat18/holes6/drought) x 8 games,
         game indices 200-207 (overlap-free vs pool 100-109 and suite
         eval 0-19), paired pre-vs-post on identical boards.
     ADOPT iff (m_s >= 0 AND t_s > -1.5) AND (m_r > -100 AND t_r > -1.5).
     The recovery bar is a damage net (paired-sd ~600/game, so -100 is
     ~1/6 sd), not a promotion claim.
  3. All scoring true-preview (audit §52.6 item 1) — no fast path anywhere.
  4. Gate unchanged (base-bank dis/new-targets/afford vs previous cycle's
     labels); hard-pool probe metrics reported, not gating.

PRE-REGISTERED FINAL-GATE RULE (fires once, in a SEPARATE step after the
loop stops — nothing here promotes anything):
  G1 dev gate: 60 fresh tapes (the NEXT untouched dev range after the
    loop's last consumed seed), cap 3000, final working champion vs
    snapshot11, 1-ply greedy, same harness both arms. PASS iff paired
    delta > 0 AND t >= 2.0 AND cap rate not lower.
  G2 recovery gate: full 12-suite x 20-game suite on the same two arms
    (202600 seed space, identical boards). PASS iff paired mean across
    all 240 games >= 0 with no suite worse than t < -2.0.
  PROMOTE both gates pass -> n=100 cap-3000 promotion panel on the 5400+
    reserve (same rule as §51.4: PROMOTE iff t >= 2.0 AND cap not lower;
    one pre-authorized +200 extension iff 0 < t < 2; stop iff delta <= 0).
  Either gate fails -> HOLD snapshot11, no promotion panel spent.

Hard rules (same as gen-1): NEVER writes lineage files; working champs in
_diag_out as _loop2_champ_rN.pt; dev seeds continue from gen-1's
next_dev_seed (21180); every stage writes the JSON incrementally.
Resume-safe: reads the JSON, continues from recorded state.

Usage:
    python _collect_hardstart_pool.py --weights ../_diag_out/_loop_champ_r1.pt \
        --out ../_diag_out/_hardstart_pool.npz --games 10 --max-states 300
    python _selfdistill_loop2.py --out ../_diag_out/_selfdistill_loop2.json \
        --hard-pool ../_diag_out/_hardstart_pool.npz \
        --init-from ../_diag_out/_selfdistill_loop.json \
        --max-cycles 8 --deadline-hours 7.5
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"

ORIGIN = ROOT / "model_weights_snapshot11.pt"
OLD_LABELS_R1 = D / "_teacher_labels_fs2ply_bank.npz"
DEV_SEED0 = 21150
SCREEN_GAMES = 30
SCREEN_CAP = 3000
STEPS = 1200
BATCH = 256
LR = 1e-4
GAMMA_PLY = 0.999
HARD_FRAC = 0.30
# Recovery leg: signal bands only (§52.6), overlap-free game indices.
REC_SUITES = ("flat15", "flat18", "holes6", "drought")
REC_GAME0 = 200
REC_GAMES = 8


def log(op: Path, results: dict) -> None:
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")


def paired_stats(dl: list[float]) -> tuple[float, float]:
    m = statistics.fmean(dl)
    sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
    t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
    return m, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_selfdistill_loop2.json"))
    ap.add_argument("--init-from",
                    default=str(D / "_selfdistill_loop.json"),
                    help="gen-1 JSON to inherit champion/dev-seed from")
    ap.add_argument("--max-cycles", type=int, default=8)
    ap.add_argument("--deadline-hours", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--fit-seed0", type=int, default=100,
                    help="schedule seeds: fit_seed0 + cycle")
    ap.add_argument("--bank",
                    default=str(D / "state_bank_snapshot9_32games.npz"))
    ap.add_argument("--hard-pool",
                    default=str(D / "_hardstart_pool.npz"))
    ap.add_argument("--old-labels", default=str(OLD_LABELS_R1))
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}: cycles done = "
              f"{sorted(results.get('cycles', {}))}", flush=True)
    elif Path(args.init_from).exists():
        prev = json.loads(Path(args.init_from).read_text(encoding="utf-8"))
        results["current_champion"] = prev.get(
            "current_champion", str(ORIGIN))
        results["next_dev_seed"] = prev.get("next_dev_seed", DEV_SEED0)
        results["gen1_final_sha"] = prev.get("final_sha")
        print(f"init from {Path(args.init_from).name}: champion "
              f"{Path(results['current_champion']).name}, dev seed "
              f"{results['next_dev_seed']}", flush=True)
    results.setdefault("note",
                       "Loop gen-2: mixed-bank (70/30) fits + dual "
                       "survival/recovery screen; working champs in "
                       "_diag_out only; final gate is separate "
                       "(see module docstring).")
    results.setdefault("cycles", {})
    t_start = time.perf_counter()

    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from diag_tinyset_overfit import teacher_scores_for_state
    from afterstate import flat_rows_to_tensor
    from evaluate import evaluate_weights_parallel
    from _recovery_suite import SUITES, build_start_board
    from _recovery_suite import play_recovery_game
    from piece_history import ALL_PIECES
    import deathbot_sim as ds

    # ---- base bank ----
    bank = Bank(args.bank)
    n = bank.n_states
    placed, nxt = bank._replay_piece_ids()
    n_cand = int(bank.cand_offsets[-1])
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = bank.candidate_slice(s)
        cand_cur[lo:hi] = piece_to_index(placed[s])
        cand_nxt[lo:hi] = piece_to_index(nxt[s])
    game_of = np.asarray(bank.game_ids)
    train_set = set(sorted(set(int(g) for g in game_of))[:24])
    base_pool_all = [s for s in range(n)
                     if int(game_of[s]) in train_set
                     and (bank.candidate_slice(s)[1]
                          - bank.candidate_slice(s)[0]) > 1]

    # ---- hard-start pool ----
    hz = np.load(str(args.hard_pool))
    h_pre = hz["pre_rows"]
    h_flat = hz["cand_rows"]
    h_off = hz["cand_offsets"]
    h_placed = hz["placed_idx"]
    h_nxt = hz["nxt_idx"]
    h_suites = [str(x) for x in hz["suite"]]
    h_alldead = hz["alldead"]
    H = int(h_pre.shape[0])
    h_total = int(h_off[-1])
    h_boards = flat_rows_to_tensor(h_flat, h_total) if h_total else None
    h_kinds = [ALL_PIECES[int(i)] for i in h_placed]
    h_nks = [ALL_PIECES[int(i)] for i in h_nxt]
    print(f"hard pool: {H} states "
          f"({json.loads(str(hz['meta_json']))['sha']})", flush=True)

    champ_path = Path(results.get("current_champion", str(ORIGIN)))
    cycle = max((int(k[1:]) for k in results["cycles"]), default=0)
    dev_seed = results.get("next_dev_seed", DEV_SEED0)
    print(f"loop2 start: champion {Path(champ_path).name} "
          f"(sha {hashlib.sha256(champ_path.read_bytes()).hexdigest()[:16]}), "
          f"cycle {cycle + 1}, dev seed {dev_seed}", flush=True)

    while cycle < args.max_cycles:
        if (time.perf_counter() - t_start) / 3600 > args.deadline_hours:
            print("deadline reached — stopping", flush=True)
            break
        cycle += 1
        tag = f"r{cycle}"
        cyc = results["cycles"].setdefault(tag, {})
        cyc["started_from"] = str(champ_path)
        cyc["started_sha"] = hashlib.sha256(
            champ_path.read_bytes()).hexdigest()[:16]
        log(op, results)

        champ = TAPNet(output="linear")
        champ.load_state_dict(torch.load(str(champ_path), map_location="cpu"))
        champ.eval()

        # ---- 1+2: label BOTH pools with the current champion's 2-ply ----
        lab_path = D / f"_loop2_labels_{tag}.npz"
        hlab_path = D / f"_loop2_hlabels_{tag}.npz"
        # 2026-09-13 fix: gate `old` must be the labels the working
        # champion was last TRAINED toward (last adopted cycle's labels),
        # not blindly r{cycle-1}. After a REVERT the champion is unchanged,
        # so r{cycle-1}'s labels are bit-identical to the fresh ones and
        # new_targets is necessarily 0 — a gate artifact, not exhaustion.
        if cycle == 1:
            old_path = Path(args.old_labels)
        else:
            adopted_tag = None
            for k in range(cycle - 1, 0, -1):
                ck = results["cycles"].get(f"r{k}", {})
                if ck.get("adopted_checkpoint") or \
                        ck.get("screen", {}).get("adopted"):
                    adopted_tag = f"r{k}"
                    break
            old_path = (D / f"_loop2_labels_{adopted_tag}.npz"
                        if adopted_tag else Path(args.old_labels))
        cyc["gate_old"] = Path(old_path).name
        zold = np.load(str(old_path))
        old_pick = {int(s): int(p) for s, p in
                    zip(zold["state_index"], zold["teacher_pick"])}
        t0 = time.perf_counter()
        if lab_path.exists() and hlab_path.exists():
            z = np.load(str(lab_path))
            new_pick = {int(s): int(p) for s, p in
                        zip(z["state_index"], z["teacher_pick"])}
            alldead = {int(s): bool(a) for s, a in
                       zip(z["state_index"], z["alldead"])}
            zh = np.load(str(hlab_path))
            h_pick = {int(s): int(p) for s, p in
                      zip(zh["state_index"], zh["teacher_pick"])}
            h_dead = {int(s): bool(a) for s, a in
                      zip(zh["state_index"], zh["alldead"])}
            print(f"{tag}: labels cached ({len(new_pick)} base, "
                  f"{len(h_pick)} hard)", flush=True)
        else:
            new_pick, alldead = {}, {}
            with torch.inference_mode():
                for s in range(n):
                    pre = ds.Board.from_row_bits(
                        [int(v) for v in bank.pre_rows[s]])
                    tv, viable_live = teacher_scores_for_state(
                        champ, pre, placed[s], nxt[s], GAMMA_PLY)
                    if tv is None:
                        continue
                    lo, _hi = bank.candidate_slice(s)
                    viable = [lo + p for p in viable_live]
                    order = sorted(range(len(tv)),
                                   key=lambda j: (-tv[j], j))
                    new_pick[s] = viable[order[0]]
                    alldead[s] = all(float(v) <= -5e5 for v in tv)
                    if len(new_pick) % 4000 == 0:
                        print(f"  {tag} base-label {len(new_pick)} "
                              f"({time.perf_counter() - t0:.0f}s)", flush=True)
            np.savez(str(lab_path),
                     state_index=np.array(sorted(new_pick), dtype=np.int32),
                     teacher_pick=np.array(
                         [new_pick[s] for s in sorted(new_pick)],
                         dtype=np.int64),
                     alldead=np.array(
                         [alldead[s] for s in sorted(new_pick)], dtype=bool))
            h_pick, h_dead = {}, {}
            with torch.inference_mode():
                for h in range(H):
                    if bool(h_alldead[h]):
                        continue
                    lo, hi = int(h_off[h]), int(h_off[h + 1])
                    if hi - lo < 2:
                        continue
                    pre = ds.Board.from_row_bits(
                        [int(v) for v in h_pre[h]])
                    tv, viable_live = teacher_scores_for_state(
                        champ, pre, h_kinds[h], h_nks[h], GAMMA_PLY)
                    if tv is None:
                        continue
                    order = sorted(range(len(tv)),
                                   key=lambda j: (-tv[j], j))
                    # teacher frame == stored viable order (bulk order)
                    assert len(tv) == hi - lo
                    h_pick[h] = order[0]  # local viable index
                    h_dead[h] = all(float(v) <= -5e5 for v in tv)
                    if len(h_pick) % 4000 == 0:
                        print(f"  {tag} hard-label {len(h_pick)} "
                              f"({time.perf_counter() - t0:.0f}s)", flush=True)
            np.savez(str(hlab_path),
                     state_index=np.array(sorted(h_pick), dtype=np.int32),
                     teacher_pick=np.array(
                         [h_pick[h] for h in sorted(h_pick)], dtype=np.int64),
                     alldead=np.array(
                         [h_dead[h] for h in sorted(h_pick)], dtype=bool))
            print(f"{tag}: labeled {len(new_pick)} base + {len(h_pick)} "
                  f"hard ({time.perf_counter() - t0:.0f}s)", flush=True)

        # ---- gate (base bank vs previous cycle's labels, gen-1 rule) ----
        student_pick: dict[int, int] = {}
        with torch.inference_mode():
            for s in sorted(new_pick):
                lo, hi = bank.candidate_slice(s)
                sc = champ(cand_boards[lo:hi], cand_cur[lo:hi],
                           cand_nxt[lo:hi]).flatten()
                student_pick[s] = lo + int(sc.argmax())
        both = sorted(new_pick)
        dis = [s for s in both if student_pick[s] != new_pick[s]]
        new_targets = [s for s in dis if student_pick[s] == old_pick.get(s)]
        gaps = []
        for s in dis:
            lo, hi = bank.candidate_slice(s)
            with torch.inference_mode():
                v = champ(cand_boards[lo:hi], cand_cur[lo:hi],
                          cand_nxt[lo:hi]).flatten().tolist()
            gaps.append(v[(list(range(lo, hi))).index(new_pick[s])]
                        - max(v))
        afford_med = statistics.median(gaps) if gaps else -99.0
        gate = {"dis_rate": round(len(dis) / max(1, len(both)), 4),
                "new_targets": len(new_targets),
                "afford_median": round(float(afford_med), 2)}
        # hard-pool probe metrics (informational, not gating)
        h_dis = h_new = 0
        with torch.inference_mode():
            for h in sorted(h_pick):
                lo, hi = int(h_off[h]), int(h_off[h + 1])
                cur_t = torch.full((hi - lo,), int(h_placed[h]),
                                   dtype=torch.long)
                nxt_t = torch.full((hi - lo,), int(h_nxt[h]), dtype=torch.long)
                sc = champ(h_boards[lo:hi], cur_t, nxt_t).flatten()
                sp = int(sc.argmax())
                if sp != h_pick[h]:
                    h_dis += 1
                    # "new" has no old-label meaning on hard states;
                    # report raw disagreement only.
                    h_new += 1
        gate["hard_states"] = len(h_pick)
        gate["hard_dis"] = h_dis
        gate["hard_dis_rate"] = round(h_dis / max(1, len(h_pick)), 4)
        gate_pass = (gate["dis_rate"] >= 0.05
                     and gate["new_targets"] >= 300
                     and gate["afford_median"] > -8)
        cyc["gate"] = gate
        cyc["gate_pass"] = bool(gate_pass)
        log(op, results)
        print(f"{tag} gate: {gate} -> {'PASS' if gate_pass else 'FAIL'}",
              flush=True)
        if not gate_pass:
            results["stop_reason"] = f"{tag}: gate FAIL (exhausted)"
            results["current_champion"] = str(champ_path)
            log(op, results)
            break

        # ---- 3: mixed burst fit (70% base + 30% hard) ----
        fit_base = [s for s in both
                    if int(game_of[s]) in train_set
                    and (bank.candidate_slice(s)[1]
                         - bank.candidate_slice(s)[0]) > 1
                    and not alldead.get(s, False)]
        fit_hard = [h for h in sorted(h_pick) if not h_dead.get(h, False)]
        n_hard = int(BATCH * HARD_FRAC)
        n_base = BATCH - n_hard
        rng = random.Random(args.fit_seed0 + cycle)
        student = TAPNet(output="linear")
        student.load_state_dict(torch.load(str(champ_path),
                                           map_location="cpu"))
        student.train()
        opt = torch.optim.Adam(student.parameters(), lr=LR)
        w0 = [p.detach().clone() for p in student.parameters()]
        losses: list[float] = []
        hist = []
        for step in range(args.steps):
            idx_b = rng.sample(fit_base, min(n_base, len(fit_base)))
            idx_h = rng.sample(fit_hard, min(n_hard, len(fit_hard)))
            bl, cl, nl, tl, lens = [], [], [], [], []
            for s in idx_b:
                lo, hi = bank.candidate_slice(s)
                bl.append(cand_boards[lo:hi])
                cl.append(cand_cur[lo:hi])
                nl.append(cand_nxt[lo:hi])
                tl.append(new_pick[s] - lo)
                lens.append(hi - lo)
            for h in idx_h:
                lo, hi = int(h_off[h]), int(h_off[h + 1])
                bl.append(h_boards[lo:hi])
                cl.append(torch.full((hi - lo,), int(h_placed[h]),
                                     dtype=torch.long))
                nl.append(torch.full((hi - lo,), int(h_nxt[h]),
                                     dtype=torch.long))
                tl.append(h_pick[h])
                lens.append(hi - lo)
            b, c, nx = torch.cat(bl), torch.cat(cl), torch.cat(nl)
            opt.zero_grad()
            scores = student(b, c, nx).flatten()
            total = scores.new_zeros(())
            off = 0
            for li, t in zip(lens, tl):
                seg = scores[off:off + li]
                off += li
                total = total + F.cross_entropy(
                    seg.unsqueeze(0), torch.tensor([t]))
            loss = total / len(lens)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 200 == 0:
                dW = float(sum(((a - bb) ** 2).sum()
                               for a, bb in zip(w0, student.parameters())
                               ).sqrt().detach())
                hist.append({"step": step + 1,
                             "loss": round(statistics.fmean(losses[-100:]), 4),
                             "dW": round(dW, 3)})
                print(f"  {tag} step {step + 1}/{args.steps}: "
                      f"loss {hist[-1]['loss']:.4f} dW {dW:.2f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        burst_path = D / f"_loop2_burst_{tag}.pt"
        torch.save(student.state_dict(), str(burst_path))
        assert burst_path.exists()
        cyc["burst"] = {
            "steps": args.steps,
            "fit_base": len(fit_base), "fit_hard": len(fit_hard),
            "mix": f"{1 - HARD_FRAC:.0%}/{(HARD_FRAC):.0%}",
            "final_loss": round(statistics.fmean(losses[-50:]), 4),
            "dW": hist[-1]["dW"] if hist else 0.0,
            "checkpoint": str(burst_path),
            "sha": hashlib.sha256(burst_path.read_bytes()).hexdigest()[:16],
            "history": hist,
        }
        log(op, results)
        print(f"{tag} burst saved ({burst_path.name})", flush=True)

        # ---- 4a: survival screen (safety net, same as gen-1) ----
        champ_pre = [g.pieces for g in evaluate_weights_parallel(
            str(champ_path), games=SCREEN_GAMES, seed=dev_seed,
            max_pieces=SCREEN_CAP, workers=8).games]
        cand_post = [g.pieces for g in evaluate_weights_parallel(
            str(burst_path), games=SCREEN_GAMES, seed=dev_seed,
            max_pieces=SCREEN_CAP, workers=8).games]
        dl_s = [a - b for a, b in zip(cand_post, champ_pre)]
        m_s, t_s = paired_stats(dl_s)

        # ---- 4b: recovery leg (paired, identical boards) ----
        burst_net = TAPNet(output="linear")
        burst_net.load_state_dict(torch.load(str(burst_path),
                                             map_location="cpu"))
        burst_net.eval()
        dl_r: list[float] = []
        rec_detail = {}
        with torch.inference_mode():
            for suite in REC_SUITES:
                wins = losses = 0
                for gi in range(REC_GAMES):
                    game_index = REC_GAME0 + gi
                    b0, dk = build_start_board(
                        suite, game_index, 202_600)
                    seed = (202_600 + SUITES.index(suite) * 10_000
                            + game_index)
                    r_pre = play_recovery_game(champ, b0, dk, seed,
                                               cap=1500)["pieces"]
                    r_post = play_recovery_game(burst_net, b0, dk, seed,
                                                cap=1500)["pieces"]
                    dl_r.append(r_post - r_pre)
                    wins += r_post > r_pre
                    losses += r_post < r_pre
                rec_detail[suite] = {"wins": wins, "losses": losses}
        m_r, t_r = paired_stats(dl_r)

        adopt = (m_s >= 0 and t_s > -1.5) and (m_r > -100 and t_r > -1.5)
        cyc["screen"] = {
            "dev_seeds": f"{dev_seed}-{dev_seed + SCREEN_GAMES - 1}",
            "cap": SCREEN_CAP,
            "survival": {
                "pre_mean": round(statistics.fmean(champ_pre), 1),
                "post_mean": round(statistics.fmean(cand_post), 1),
                "delta": round(m_s, 1), "t": round(t_s, 2),
                "wins": sum(x > 0 for x in dl_s),
                "losses": sum(x < 0 for x in dl_s),
                "capped_pre": sum(p >= SCREEN_CAP for p in champ_pre),
                "capped_post": sum(p >= SCREEN_CAP for p in cand_post),
            },
            "recovery": {
                "suites": list(REC_SUITES),
                "games": f"{REC_GAME0}-{REC_GAME0 + REC_GAMES - 1}",
                "delta": round(m_r, 1), "t": round(t_r, 2),
                "wins": sum(x > 0 for x in dl_r),
                "losses": sum(x < 0 for x in dl_r),
                "per_suite": rec_detail,
            },
            "adopted": bool(adopt),
        }
        if adopt:
            new_champ = D / f"_loop2_champ_{tag}.pt"
            torch.save(student.state_dict(), str(new_champ))
            assert new_champ.exists()
            champ_path = new_champ
            results["current_champion"] = str(new_champ)
            cyc["adopted_checkpoint"] = str(new_champ)
        else:
            results["non_adopt_streak"] = \
                results.get("non_adopt_streak", 0) + 1
        cyc["wall_s"] = round(time.perf_counter() - t0, 1)
        log(op, results)
        print(f"{tag} screen: surv {m_s:+.1f} t {t_s:+.2f} | rec "
              f"{m_r:+.1f} t {t_r:+.2f} -> "
              f"{'ADOPT' if adopt else 'REVERT'} ({cyc['wall_s']:.0f}s)",
              flush=True)
        dev_seed += SCREEN_GAMES
        results["next_dev_seed"] = dev_seed
        if not adopt:
            if results["non_adopt_streak"] >= 2:
                results["stop_reason"] = "2 consecutive non-adoptions"
                log(op, results)
                print("stop: ", results["stop_reason"], flush=True)
                break
        else:
            results["non_adopt_streak"] = 0

    results["final_champion"] = str(champ_path)
    results["final_sha"] = hashlib.sha256(
        champ_path.read_bytes()).hexdigest()[:16]
    results["total_wall_s"] = round(time.perf_counter() - t_start, 1)
    log(op, results)
    print(f"\nLOOP2 DONE: {cycle} cycles, final champion "
          f"{Path(champ_path).name} (sha {results['final_sha'][:16]}); "
          f"reason: {results.get('stop_reason', 'max cycles')}", flush=True)
    print("NEXT (pre-registered): G1 dev gate (60 fresh tapes cap 3000 vs "
          "snapshot11, PASS iff delta>0 AND t>=2.0 AND cap not lower) + G2 "
          "full-suite recovery gate (PASS iff 240-game paired mean >= 0, "
          "no suite t<-2.0); both pass -> n=100 promotion panel on 5400+ "
          "reserve (§51.4 rule).", flush=True)


if __name__ == "__main__":
    main()
