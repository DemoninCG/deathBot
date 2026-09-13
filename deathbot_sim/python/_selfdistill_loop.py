"""Continuous self-distillation loop (overnight; 2026-09-12).

Validated mechanism (§51): distilling a champion's own full-2-ply
expansion back into its 1-ply weights produced the largest promotion in
project history (gen2_seed44 -> snapshot11, +768 t+5.97 at cap 3000).
The gen-3 probe (_teacher3_probe.json) PASSED its gate (559 new targets,
affordability median -1.05), justifying continued bursts.

Loop structure (per cycle):
  1. LABEL: label the bank with the working champion's full-2-ply
     (cycle 1 reuses the gen-3 probe's cached labels — already PASS).
  2. PROBE GATE (re-evaluated EVERY cycle vs the previous cycle's
     labels): dis >= 5% AND new_targets >= 300 AND afford median > -8.
     FAIL = loop exhausted -> stop (E3 lesson: never grind an exhausted
     start).
  3. BURST: validated-dose CE (lr 1e-4, batch 256, 1200 steps) from the
     working champion toward the new labels (schedule seed = base + N).
  4. SCREEN (safety net, not a promotion claim): 30 fresh dev tapes at
     cap 3000, paired vs the pre-burst champion. ADOPT iff delta >= 0
     AND t > -1.5 (2026-09-12 audit: the old delta > -100 bar kept
     true-zero bursts that merely pull the champion toward
     already-extracted labels while burning dev tapes; a revert costs
     nothing since labels advance anyway).
     Two consecutive non-adoptions -> stop.
  5. Next cycle labels with the ADOPTED champion's 2-ply (the probe
     contraction is measured fresh each cycle).

Hard rules:
  - NEVER writes model_weights_snapshot11.pt / model_weights.pt /
    any workspace-root lineage file. Working champions live in
    _diag_out as _loop_champ_r{N}.pt.
  - Dev seeds advance sequentially from 21150 (30 per cycle); the FINAL
    working champion gets a pre-registered fresh-panel gate + user-gated
    promotion panel (5400+ reserve) in a SEPARATE step — nothing here
    promotes anything.
  - Save-first discipline: each burst checkpoint is written before any
    results assembly; every stage writes the JSON incrementally.
  - All fits start from the working champion (NOT from scratch); a
    reverted cycle leaves the working champion untouched.

Stop conditions: gate fail | 2 consecutive non-adoptions |
--max-cycles | wall-clock deadline.

Usage:
    python _selfdistill_loop.py --out ../_diag_out/_selfdistill_loop.json \
        --max-cycles 8 --deadline-hours 7.5
Resume-safe: reads the JSON, continues from the recorded state.
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
OLD_LABELS_R1 = D / "_teacher_labels_fs2ply_bank.npz"  # gen-2's teacher
DEV_SEED0 = 21150
SCREEN_GAMES = 30
SCREEN_CAP = 3000
STEPS = 1200
BATCH = 256
LR = 1e-4
GAMMA_PLY = 0.999


def log(op: Path, results: dict) -> None:
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_selfdistill_loop.json"))
    ap.add_argument("--max-cycles", type=int, default=8)
    ap.add_argument("--deadline-hours", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--fit-seed0", type=int, default=45,
                    help="schedule seeds: fit_seed0 + cycle (42/43/44 used)")
    ap.add_argument("--bank", default=str(D / "state_bank_snapshot9_32games.npz"))
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming {op.name}: cycles done = "
              f"{sorted(results.get('cycles', {}))}", flush=True)
    results.setdefault("note",
                       "Continuous self-distillation loop; working champs "
                       "in _diag_out only; screens are safety nets, not "
                       "promotions; final gate is separate.")
    results.setdefault("cycles", {})
    t_start = time.perf_counter()

    from model import TAPNet, piece_to_index
    from diag_phase2 import Bank
    from diag_tinyset_overfit import teacher_scores_for_state
    from afterstate import flat_rows_to_tensor
    from evaluate import evaluate_weights_parallel
    import deathbot_sim as ds

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

    # ---- working champion state ----
    champ_path = Path(results.get("current_champion",
                                  str(ORIGIN)))
    cycle = max((int(k[1:]) for k in results["cycles"]), default=0)
    dev_seed = results.get("next_dev_seed", DEV_SEED0)
    print(f"loop start: champion {Path(champ_path).name} "
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

        # ---- 1+2: label with the current champion's 2-ply + gate ----
        lab_path = D / f"_loop_labels_{tag}.npz"
        old_path = (OLD_LABELS_R1 if cycle == 1
                    else D / f"_loop_labels_r{cycle - 1}.npz")
        zold = np.load(str(old_path))
        old_pick = {int(s): int(p) for s, p in
                    zip(zold["state_index"], zold["teacher_pick"])}
        if lab_path.exists():
            z = np.load(str(lab_path))
            new_pick = {int(s): int(p) for s, p in
                        zip(z["state_index"], z["teacher_pick"])}
            alldead = {int(s): bool(a) for s, a in
                       zip(z["state_index"], z["alldead"])}
            print(f"{tag}: labels cached ({len(new_pick)})", flush=True)
        else:
            new_pick: dict[int, int] = {}
            alldead: dict[int, bool] = {}
            t0 = time.perf_counter()
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
                        print(f"  {tag} label {len(new_pick)} "
                              f"({time.perf_counter() - t0:.0f}s)",
                              flush=True)
            np.savez(str(lab_path),
                     state_index=np.array(sorted(new_pick), dtype=np.int32),
                     teacher_pick=np.array([new_pick[s]
                                            for s in sorted(new_pick)],
                                           dtype=np.int64),
                     alldead=np.array([alldead[s]
                                       for s in sorted(new_pick)],
                                      dtype=bool))
            print(f"{tag}: labeled {len(new_pick)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

        # student picks + affordability (probe metrics, cheap)
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
        # 2026-09-12 audit fix: true-preview forward (survival-harness
        # policy). score_batch_fast is the versioned 2-ply cur-duplication
        # approximation and must not score 1-ply student affordability.
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
        gate_pass = (gate["dis_rate"] >= 0.05
                     and gate["new_targets"] >= 300
                     and gate["afford_median"] > -8)
        cyc["gate"] = gate
        cyc["gate_pass"] = bool(gate_pass)
        log(op, results)
        print(f"{tag} gate: {gate} -> {'PASS' if gate_pass else 'FAIL'}",
              flush=True)
        if not gate_pass:
            results["stop_reason"] = f"{tag}: gate FAIL (information exhausted)"
            results["current_champion"] = str(champ_path)
            log(op, results)
            break

        # ---- 3: burst fit (validated dose) ----
        fit_pool = [s for s in both
                    if int(game_of[s]) in train_set
                    and (bank.candidate_slice(s)[1]
                         - bank.candidate_slice(s)[0]) > 1
                    and not alldead.get(s, False)]
        rng = random.Random(args.fit_seed0 + cycle)
        student = TAPNet(output="linear")
        student.load_state_dict(torch.load(str(champ_path),
                                           map_location="cpu"))
        student.train()
        opt = torch.optim.Adam(student.parameters(), lr=LR)
        w0 = [p.detach().clone() for p in student.parameters()]
        losses: list[float] = []
        hist = []
        t1 = time.perf_counter()
        for step in range(args.steps):
            idx = rng.sample(fit_pool, min(256, len(fit_pool)))
            bl, cl, nl, tl, lens = [], [], [], [], []
            for s in idx:
                lo, hi = bank.candidate_slice(s)
                bl.append(cand_boards[lo:hi])
                cl.append(cand_cur[lo:hi])
                nl.append(cand_nxt[lo:hi])
                tl.append(new_pick[s] - lo)
                lens.append(hi - lo)
            b, c, nx = torch.cat(bl), torch.cat(cl), torch.cat(nl)
            tlt = torch.tensor(tl, dtype=torch.long)
            opt = torch.optim.Adam(student.parameters(), lr=1e-4) \
                if step == 0 else opt
            opt.zero_grad()
            scores = student(b, c, nx).flatten()
            total = scores.new_zeros(())
            off = 0
            for li, t in zip(lens, tl):
                seg = scores[off:off + li]
                off += li
                total = total + F.cross_entropy(seg.unsqueeze(0),
                                                torch.tensor([t]))
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
        # SAVE FIRST (§51 process lesson)
        burst_path = D / f"_loop_burst_{tag}.pt"
        torch.save(student.state_dict(), str(burst_path))
        assert burst_path.exists()
        cyc["burst"] = {
            "steps": args.steps, "fit_pool": len(fit_pool),
            "final_loss": round(statistics.fmean(losses[-50:]), 4),
            "dW": hist[-1]["dW"] if hist else 0.0,
            "checkpoint": str(burst_path),
            "sha": hashlib.sha256(burst_path.read_bytes()).hexdigest()[:16],
            "history": hist,
        }
        log(op, results)
        print(f"{tag} burst saved ({burst_path.name})", flush=True)

        # ---- 4: screen (safety net) + adopt/revert ----
        champ_pre = [g.pieces for g in evaluate_weights_parallel(
            str(champ_path), games=SCREEN_GAMES, seed=dev_seed,
            max_pieces=SCREEN_CAP, workers=8).games]
        cand_post = [g.pieces for g in evaluate_weights_parallel(
            str(burst_path), games=SCREEN_GAMES, seed=dev_seed,
            max_pieces=SCREEN_CAP, workers=8).games]
        dl = [a - b for a, b in zip(cand_post, champ_pre)]
        m = statistics.fmean(dl)
        sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
        t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
        adopt = m >= 0 and t > -1.5
        cyc["screen"] = {
            "dev_seeds": f"{dev_seed}-{dev_seed + SCREEN_GAMES - 1}",
            "cap": SCREEN_CAP,
            "pre_mean": round(statistics.fmean(champ_pre), 1),
            "post_mean": round(statistics.fmean(cand_post), 1),
            "delta": round(m, 1), "t": round(t, 2),
            "wins": sum(x > 0 for x in dl),
            "losses": sum(x < 0 for x in dl),
            "capped_pre": sum(p >= SCREEN_CAP for p in champ_pre),
            "capped_post": sum(p >= SCREEN_CAP for p in cand_post),
            "adopted": bool(adopt),
        }
        if adopt:
            new_champ = D / f"_loop_champ_{tag}.pt"
            torch.save(student.state_dict(), str(new_champ))
            assert new_champ.exists()
            champ_path = new_champ
            results["current_champion"] = str(new_champ)
            cyc["adopted_checkpoint"] = str(new_champ)
        else:
            results.setdefault("non_adopt_streak", 0)
            results["non_adopt_streak"] = \
                results.get("non_adopt_streak", 0) + 1
        cyc["wall_s"] = round(time.perf_counter() - t0, 1)
        log(op, results)
        print(f"{tag} screen: pre {cyc['screen']['pre_mean']} -> post "
              f"{cyc['screen']['post_mean']} (delta {m:+.1f} t {t:+.2f}) "
              f"-> {'ADOPT' if adopt else 'REVERT'} "
              f"({cyc['wall_s']:.0f}s)", flush=True)
        dev_seed += SCREEN_GAMES
        results["next_dev_seed"] = dev_seed
        results["cycle1_complete"] = True  # audit-friendly stop marker
        if not adopt:
            if results["non_adopt_streak"] >= 2:
                results["stop_reason"] = "2 consecutive non-adoptions"
                log(op, results)
                print("stop: ", results["stop_reason"], flush=True)
                break
        else:
            results["non_adopt_streak"] = 0
        w0 = None  # reset for next cycle
        # next cycle fits from the (possibly adopted) champion
        # (2026-09-12: user plans to interrupt after cycle 1 for a full
        # plan audit — KeyboardInterrupt here is SAFE: the JSON holds the
        # complete state, the next cycle starts cleanly from
        # current_champion/next_dev_seed on resume.)

    results["final_champion"] = str(champ_path)
    results["final_sha"] = hashlib.sha256(
        champ_path.read_bytes()).hexdigest()[:16]
    results["total_wall_s"] = round(time.perf_counter() - t_start, 1)
    log(op, results)
    print(f"\nLOOP DONE: {cycle} cycles, final champion "
          f"{Path(champ_path).name} (sha {results['final_sha'][:16]}); "
          f"reason: {results.get('stop_reason', 'max cycles')}", flush=True)
    print("NEXT: pre-registered fresh-panel gate on the final champion vs "
          "snapshot11 (dev 21150+ continues), then user-gated promotion "
          "panel on the 5400+ reserve if it passes.", flush=True)


if __name__ == "__main__":
    main()