"""2-ply distillation for 1-ply improvement (2026-09-06).

Goal: teach the 1-ply greedy policy to imitate full-expansion 2-ply choices
on frozen bank states. If the student matches the teacher's argmax on states
the champion visits, 1-ply play should approach 2-ply play at 1-ply cost.

Why this might work where SIL/fitd didn't: the teacher (2-ply argmax) is a
STRONGER policy than the student (validated +505), so unlike self-imitation
or own-function bootstraps there is real new ranking information. The update
is preference-only (softmax CE toward the teacher's pick) — no value
regression anywhere, so the Phase-2 failure mode cannot occur.

Arms (all from snapshot9, paired minibatch schedule, 400 steps, lr 1e-5):
  P2T1   distill 2-ply argmax, T=1.0 (sharp)
  P2T3   distill 2-ply argmax, T=3.0 (soft)

Teacher labels computed once with TwoPlyScorer(beam=100) on bank states
(states subsampled to --teacher-states for cost; labels cached in JSON).

Screen: 30 fresh games at cap 300 (style-era protocol: survival saturated,
style+safety carry the signal) + cap-rate gate. Selection on panel 20164+.
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

from model import TAPNet, piece_to_index  # noqa: E402
from diag_phase2 import Bank  # noqa: E402


def _ev_game_1ply(payload) -> tuple:
    """Module-level worker (picklable): 1-ply style game for one seed."""
    import random as _r
    import torch as _t
    from model import TAPNet as _N
    from two_ply import play_game_2ply_style as _pg
    (wpath, seed, max_pieces) = payload
    m = _N(output="linear")
    m.load_state_dict(_t.load(wpath, map_location="cpu"))
    m.eval()
    p, ln, cl, rs = _pg(m, _r.Random(seed), beam=0, max_pieces=max_pieces)
    return (p, ln, {str(k): int(v) for k, v in cl.items()}, rs)


def resolve_weights(path_str: str) -> Path:
    supplied = Path(path_str).expanduser()
    for cand in (supplied, HERE.parent / supplied,
                 HERE.parent.parent / supplied,
                 HERE.parent.parent.parent / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights not found: {path_str}")


def compute_teacher_labels(bank: Bank, net: TAPNet, states: list[int]) -> dict[int, int]:
    """2-ply argmax (bank-candidate index space) for each state in `states`."""
    import deathbot_sim as ds
    from two_ply import TwoPlyScorer
    scorer = TwoPlyScorer(net, beam=10 ** 9)
    placed, nxt = bank._replay_piece_ids()
    labels: dict[int, int] = {}
    for si, s in enumerate(states):
        lo, hi = bank.candidate_slice(s)
        pre = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s]])
        kind = placed[s]
        nk = nxt[s]
        try:
            pick = scorer.choose(pre, kind, nk)
        except ValueError:
            continue
        # choose() returns search_locks-order index; bank order matches the
        # bulk order (bank_builder records from lock_candidates_full, same
        # BFS order) — verify alignment via geometry on a sample.
        labels[s] = int(pick)
        if (si + 1) % 500 == 0:
            print(f"  teacher {si + 1}/{len(states)}", flush=True)
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--teacher-states", type=int, default=4000,
                    help="bank states to label with the 2-ply teacher")
    ap.add_argument("--temps", nargs="+", type=float, default=[1.0, 3.0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--screen-games", type=int, default=30)
    ap.add_argument("--screen-seed0", type=int, default=20164)
    ap.add_argument("--max-pieces", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-arms", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    bank = Bank(args.bank)
    n = bank.n_states
    print(f"Bank: {n} states / {int(bank.cand_offsets[-1])} candidates")

    weights_path = resolve_weights(args.weights)
    incumbent = TAPNet(output="linear")
    incumbent.load_state_dict(torch.load(weights_path, map_location=device))
    incumbent.eval()

    # Full candidate tensors (same layout as diag_sil).
    from afterstate import flat_rows_to_tensor
    n_cand = int(bank.cand_offsets[-1])
    cand_offsets = torch.tensor(np.asarray(bank.cand_offsets), dtype=torch.long)
    cand_boards = flat_rows_to_tensor(bank.cand_rows, n_cand)
    placed_ids, next_ids = bank._replay_piece_ids()
    cand_cur = torch.empty(n_cand, dtype=torch.long)
    cand_nxt = torch.empty(n_cand, dtype=torch.long)
    for s in range(n):
        lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
        cand_cur[lo:hi] = piece_to_index(placed_ids[s])
        cand_nxt[lo:hi] = piece_to_index(next_ids[s])
    chosen_idx = torch.tensor(np.asarray(bank.chosen_idx), dtype=torch.long)

    # Teacher states: random subsample (deterministic on --seed).
    rng = random.Random(args.seed)
    label_states = rng.sample(range(n), min(args.teacher_states, n))

    # Geometry sanity: bank candidate order vs live search_locks order.
    import deathbot_sim as ds
    s0 = label_states[0]
    lo0, hi0 = bank.candidate_slice(s0)
    live = ds.Board.from_row_bits([int(v) for v in bank.pre_rows[s0]]).search_locks(placed_ids[s0])
    assert len(live) == hi0 - lo0, (
        f"bank/live candidate count mismatch: {hi0 - lo0} vs {len(live)}")
    for j, c in enumerate(live):
        assert (c.piece.state, c.piece.x, c.piece.y) == (
            int(bank.cand_state[lo0 + j]), int(bank.cand_x[lo0 + j]),
            int(bank.cand_y[lo0 + j])), "bank/live candidate ORDER mismatch"
    print(f"geometry check OK on state {s0} ({hi0 - lo0} candidates)")

    # 1-ply agreement baseline on the labeled states (how much is there to learn?).
    with torch.no_grad():
        agree0 = 0
        margins = []
        for s in label_states:
            lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
            sc = incumbent(cand_boards[lo:hi], cand_cur[lo:hi], cand_nxt[lo:hi]).flatten()
            top2 = torch.topk(sc, min(2, sc.numel())).values
            agree0 += int(int(sc.argmax()) == int(chosen_idx[s]))
            if sc.numel() > 1:
                margins.append(float(top2[0] - top2[1]))
    print(f"1-ply argmax == bank chosen on {agree0}/{len(label_states)} labeled states; "
          f"mean top-2 margin {statistics.fmean(margins):.2f}")

    print(f"Computing 2-ply teacher labels for {len(label_states)} states ...")
    t0 = time.perf_counter()
    teacher = compute_teacher_labels(bank, incumbent, label_states)
    print(f"teacher done in {time.perf_counter() - t0:.0f}s; "
          f"labels for {len(teacher)}/{len(label_states)} states")
    agree_t = sum(1 for s, p in teacher.items() if p == int(chosen_idx[s]))
    print(f"2-ply argmax == 1-ply chosen on {agree_t}/{len(teacher)} states "
          f"({agree_t / max(1, len(teacher)):.3f}) — disagreement = learnable signal")

    results = {
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "bank": str(Path(args.bank).resolve()),
        "teacher_states": len(teacher),
        "agree_2ply_vs_1ply": agree_t / max(1, len(teacher)),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "screen_games": args.screen_games,
        "screen_seed0": args.screen_seed0, "max_pieces": args.max_pieces,
        "arms": {},
    }

    schedule = [rng.sample(sorted(teacher), min(args.batch_size, len(teacher)))
                for _ in range(args.steps)]

    from evaluate import evaluate_weights_parallel
    from two_ply import play_game_2ply_style, style_summary
    import tempfile

    for temp in args.temps:
        arm = f"P2T{temp:g}"
        print(f"\n=== {arm} ===", flush=True)
        net = TAPNet(output="linear")
        net.load_state_dict(torch.load(weights_path, map_location=device))
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        losses = []
        # Item 10: precompute per-state slices once; each step does ONE
        # forward over the concatenated candidates of the whole minibatch,
        # then segments log_softmax per state (no per-state net calls).
        step_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor, list[int]]] = []
        for idx in schedule:
            boards_list, cur_list, nxt_list, tgt_list, lens = [], [], [], [], []
            for s in idx:
                lo, hi = int(cand_offsets[s]), int(cand_offsets[s + 1])
                if hi - lo < 2:
                    continue
                boards_list.append(cand_boards[lo:hi])
                cur_list.append(cand_cur[lo:hi])
                nxt_list.append(cand_nxt[lo:hi])
                tgt_list.append(int(teacher[s]))
                lens.append(hi - lo)
            if not boards_list:
                step_batches.append(None)
                continue
            step_batches.append((torch.cat(boards_list), torch.cat(cur_list),
                                 torch.cat(nxt_list),
                                 torch.tensor(tgt_list, dtype=torch.long),
                                 lens))
        t1 = time.perf_counter()
        for step in range(args.steps):
            pre = step_batches[step]
            if pre is None:
                losses.append(0.0)
                continue
            b_all, c_all, n_all, tgt_all, lens = pre
            opt.zero_grad()
            scores_all = net(b_all, c_all, n_all).flatten()
            # Segmented log_softmax: split concatenated scores per state.
            total = scores_all.new_zeros(())
            off = 0
            for li, t in zip(lens, tgt_all):
                seg = scores_all[off:off + li]
                off += li
                logp = F.log_softmax(seg / temp, dim=0)
                total = total + (-logp[int(t)])
            total = total / max(1, len(lens))
            total.backward()
            opt.step()
            losses.append(float(total.detach()))
            if (step + 1) % 100 == 0:
                print(f"  step {step + 1}/{args.steps}: "
                      f"loss={statistics.fmean(losses[-100:]):.4f}", flush=True)
        net.eval()
        dt = time.perf_counter() - t1

        from diag_phase2 import ranking_drift
        rank_states = rng.sample(range(n), min(200, n))
        drift = ranking_drift(net, incumbent, bank, rank_states, device=device)

        tmp = Path(tempfile.gettempdir()) / f"_p2t_{temp}_{args.seed}.pt"
        torch.save(net.state_dict(), str(tmp))
        # Style screen: fixed-data gain must show up in real play; judge on
        # lines + cap safety at cap 300 (survival-era protocol).
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_ev_game_1ply,
                                (str(tmp), args.screen_seed0 + i,
                                 args.max_pieces))
                    for i in range(args.screen_games)]
            rows = [f.result() for f in futs]
        pieces = [r[0] for r in rows]
        lines = [r[1] for r in rows]
        tet = sum(int(r[2].get("4", 0)) for r in rows)
        scoring = sum(int(r[2].get(k, 0)) for r in rows for k in ("1", "2", "3", "4"))
        tmp.unlink(missing_ok=True)
        if args.save_arms:
            sp = Path(args.out).resolve().parent / f"p2t_T{temp:g}_seed{args.seed}.pt"
            torch.save(net.state_dict(), str(sp))
            print(f"  saved arm checkpoint to {sp}")
        results["arms"][arm] = {
            "train_s": round(dt, 1),
            "mean_pieces": statistics.fmean(pieces),
            "mean_lines": statistics.fmean(lines),
            "lines_per_100": 100.0 * sum(lines) / max(1, sum(pieces)),
            "tetrises": tet,
            "tetris_share": (tet / scoring) if scoring else 0.0,
            "capped": sum(1 for p in pieces if p >= args.max_pieces),
            "per_seed": pieces,
            "per_seed_lines": lines,
            **drift,
        }
        print(f"  1-ply eval {statistics.fmean(pieces):.1f} pcs / "
              f"{statistics.fmean(lines):.1f} lines | tau {drift['mean_kendall']:.3f} "
              f"| top1 {drift['top1_agreement']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
