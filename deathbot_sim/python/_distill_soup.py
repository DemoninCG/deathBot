"""Cheap improvement tests on the validated distillation recipe (2026-09-11).

Context (§49): the fullsplit recipe (CE toward the frozen 2-ply teacher's
argmax, lr 1e-4, batch 256, 1200 steps, from snapshot9) is replicated 3/3
schedule seeds (+337..+617 paired vs snap9 on 31000s). The user reopened
the path for a little more testing. This driver runs the two cheapest
never-tested cells plus a new-recipe-seed batch, all REPORTING-ONLY
(seeds 31000-31029 / 26000-26007 — demo-reporting + training ranges;
no dev/promotion panels consumed; no champion change implied):

ARMS
  soup42/43/44 : element-wise mean of the three same-basin recipe
                 checkpoints (L2 1.5-1.6 apart, cos 0.999 — §49 replication
                 products). Motivation: their failures are anti-correlated
                 per tape (31009: +619/-721/-55; 31012: -718/+0/-1100), and
                 weight averaging never redistributes mass across basins
                 (same basin by construction). This is the classic
                 "model soup" (Wortsman et al. 2022) setting: several
                 fine-tunes of ONE init in ONE basin.
  soup_s42_s43 : pairwise soup (in case 3-way averaging over-smooths).
  ensemble344  : NOT a weights file — a value-ensemble scorer (mean of the
                 three nets' outputs per candidate) played through the same
                 sweep harness, as the alternative combination method.
  snap9        : paired baseline.
  s42/s43/s44  : the members themselves (fresh harness numbers for
                 consistency with the resumed §49.3 eval).

If a soup/ensemble beats its best member on this panel, the next step is a
fresh-panel confirmation (21020+). If not, the recipe-seeds-45/46 leg
(also in this driver) adds two more same-recipe members — both to test
recipe robustness further and to give the soup more ingredients.

Weight-averaging validity: TAPNet is pure conv/linear (no BN/dropout), so
the element-wise mean is exactly a network; teacher labels are unchanged;
nothing is retrained.

Usage:
    python _distill_soup.py --out ../_diag_out/_soup_screen.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"


def make_soup(paths: list[Path], out: Path) -> Path:
    sds = [torch.load(str(p), map_location="cpu") for p in paths]
    keys = sorted(sds[0].keys())
    avg = {}
    for k in keys:
        avg[k] = sum(sd[k].float() for sd in sds) / len(sds)
    avg = {k: v.to(sds[0][k].dtype) for k, v in avg.items()}
    torch.save(avg, str(out))
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "_soup_screen.json"))
    ap.add_argument("--seeds4546", action="store_true",
                    help="also run the recipe at schedule seeds 45/46 "
                         "(~50 min fit each + labeling ~24 min once)")
    ap.add_argument("--skip-eval-existing", action="store_true")
    args = ap.parse_args()

    op = Path(args.out)
    results: dict = {}
    if op.exists():
        results = json.loads(op.read_text(encoding="utf-8"))
        print(f"resuming from {op.name}: "
              f"{sorted(results.get('eval', {}).keys())}", flush=True)
    results.setdefault("note",
                       "Soup/ensemble screen over §49 replication products; "
                       "reporting-only seeds 31000s + 26000s; no panels.")
    results.setdefault("eval", {})

    # ---- build soups (cheap; idempotent) ----
    m42 = D / "fullsplit_seed42.pt"
    m43 = D / "_fullsplit_repl_seed43.pt"
    m44 = D / "_fullsplit_repl_seed44.pt"
    soup3 = D / "_soup_s42_s43_s44.pt"
    soup2 = D / "_soup_s42_s43.pt"
    if not soup3.exists():
        make_soup([m42, m43, m44], soup3)
        print(f"built {soup3.name}", flush=True)
    if not soup2.exists():
        make_soup([m42, m43], soup2)
        print(f"built {soup2.name}", flush=True)

    ckpts: dict[str, Path | None] = {
        "snap9": ROOT / "model_weights_snapshot9.pt",
        "s42": m42, "s43": m43, "s44": m44,
        "soup3": soup3, "soup2_43": soup2,
        "ensemble344": m42,  # sentinel path; arm handled specially below
    }

    # ---- optional: two more recipe members at schedule seeds 45/46 ----
    if args.seeds4546:
        for seed in (45, 46):
            ck = D / f"_fullsplit_repl_seed{seed}.pt"
            if ck.exists():
                ckpts[f"s{seed}"] = ck
                continue
            print(f"=== fitting recipe seed {seed} ===", flush=True)
            import subprocess
            cmd = [sys.executable, str(HERE / "_distill_replicate.py"),
                   "--seeds", str(seed),
                   "--out", str(D / f"_fullsplit_replicate_s{seed}.json")]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=str(HERE))
            ckpts[f"s{seed}"] = ck
        # rebuild the 5-way soup if all five members exist
        members5 = [m42, m43, m44] + [D / f"_fullsplit_repl_seed{s}.pt"
                                      for s in (45, 46)]
        if all(p.exists() for p in members5):
            soup5 = D / "_soup_s42_s43_s44_s45_s46.pt"
            if not soup5.exists():
                make_soup(members5, soup5)
                print(f"built {soup5.name}", flush=True)
            ckpts["soup5"] = soup5

    # ---- value-ensemble scorer (mean of 3 nets per candidate) ----
    class EnsembleScorer:
        """Drop-in NeuralScorer substitute: mean of N nets' scores.

        Mirrors NeuralScorer.score_batch's fast path so the sweep harness
        and this driver's eval share one code path. cur==nxt fast path
        preserved; the three nets are evaluated on the same batch.
        """

        def __init__(self, paths: list[Path]) -> None:
            from model import TAPNet
            self.nets = []
            for p in paths:
                net = TAPNet(output="linear")
                net.load_state_dict(torch.load(str(p), map_location="cpu"))
                net.eval()
                self.nets.append(net)

        def score_batch(self, board, candidates, current_piece_idx=None,
                        next_piece_idx=None):
            from model import piece_to_index
            from afterstate import flat_rows_to_tensor
            cur = current_piece_idx if current_piece_idx is not None \
                else piece_to_index(candidates[0].piece.kind)
            nxt = next_piece_idx if next_piece_idx is not None else cur
            bulk = getattr(board, "lock_candidates_fast", None) \
                or getattr(board, "lock_candidates_full", None)
            if bulk is not None and candidates and \
                    getattr(candidates[0], "piece", None) is not None:
                try:
                    kind = candidates[0].piece.kind
                    (states, xs, ys, lockouts, flat_rows, lines) = bulk(kind)
                    n = len(states)
                    matches = (
                        n == len(candidates)
                        and all(s == c.piece.state and x == c.piece.x
                                and y == c.piece.y
                                for s, x, y, c in
                                zip(states, xs, ys, candidates))
                    )
                except Exception:
                    matches = False
                if matches:
                    boards = flat_rows_to_tensor(flat_rows, n)
                    cur_t = torch.full((n,), cur, dtype=torch.long)
                    nxt_t = torch.full((n,), nxt, dtype=torch.long)
                    with torch.inference_mode():
                        outs = [net(boards, cur_t, nxt_t).flatten()
                                for net in self.nets]
                    mean = torch.stack(outs).mean(0)
                    return [float(x) for x in mean.tolist()]
            # fallback (rare): per-candidate
            from afterstate import make_afterstate
            boards = torch.stack([make_afterstate(board, c).tensor
                                  for c in candidates], dim=0)
            cur_t = torch.full((len(candidates),), cur, dtype=torch.long)
            nxt_t = torch.full((len(candidates),), nxt, dtype=torch.long)
            with torch.inference_mode():
                outs = [net(boards, cur_t, nxt_t).flatten()
                        for net in self.nets]
            mean = torch.stack(outs).mean(0)
            return [float(x) for x in mean.tolist()]

    # ---- paired eval (REPORTING ONLY; same harness as §49.3 resume) ----
    from selfplay import play_game_detailed
    import random as _random
    import deathbot_sim  # noqa: F401  (ensure Rust ext is importable)

    def eval_scorer(scorer, games: int, seed0: int) -> list[int]:
        out = []
        for s in range(seed0, seed0 + games):
            res = play_game_detailed(scorer, _random.Random(s), epsilon=0.0,
                                     max_pieces=1500)
            out.append(res.pieces_placed)
        return out

    def eval_weights(path: Path, games: int, seed0: int) -> list[int]:
        from evaluate import evaluate_weights_parallel
        ev = evaluate_weights_parallel(str(path), games=games, seed=seed0,
                                       max_pieces=1500, workers=8)
        return [g.pieces for g in ev.games]

    LEGS = (("train26000", 8, 26000), ("report31000", 30, 31000))

    for tag, count, seed0 in LEGS:
        leg = results["eval"].setdefault(tag, {})
        # snap9 (baseline via evaluate_weights_parallel on its path)
        if "snap9" not in leg:
            t0 = time.perf_counter()
            pieces = eval_weights(ckpts["snap9"], count, seed0)
            leg["snap9"] = {"mean": round(statistics.fmean(pieces), 1),
                            "per_seed": pieces,
                            "wall_s": round(time.perf_counter() - t0, 1)}
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"{tag} snap9 mean {leg['snap9']['mean']} — SAVED",
                  flush=True)
        base = leg["snap9"]["per_seed"]

        for name, ck in sorted(ckpts.items()):
            if name == "snap9":
                continue
            if name in leg and not args.skip_eval_existing:
                print(f"{tag} {name}: already done", flush=True)
                continue
            t0 = time.perf_counter()
            if name == "ensemble344":
                pieces = eval_scorer(EnsembleScorer([m42, m43, m44]),
                                     count, seed0)
            else:
                pieces = eval_weights(ck, count, seed0)
            dl = [a - b for a, b in zip(pieces, base)]
            m = statistics.fmean(dl)
            sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
            t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
            leg[name] = {
                "mean": round(statistics.fmean(pieces), 1),
                "capped": sum(p >= 1500 for p in pieces),
                "delta_vs_snap9": round(m, 1), "t": round(t, 2),
                "wins": sum(d > 0 for d in dl),
                "losses": sum(d < 0 for d in dl),
                "per_seed": pieces,
                "wall_s": round(time.perf_counter() - t0, 1),
                "sha": (hashlib.sha256(Path(ck).read_bytes()).hexdigest()[:16]
                        if name != "ensemble344" else "ensemble"),
            }
            op.write_text(json.dumps(results, indent=2), encoding="utf-8")
            r = leg[name]
            print(f"{tag} {name:12s} mean {r['mean']:7.1f} "
                  f"cap {r['capped']:3d}/{count} "
                  f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                  f"W/L {r['wins']}/{r['losses']} — SAVED", flush=True)

    print("\n=== SOUP SCREEN SUMMARY (paired vs snap9) ===", flush=True)
    for tag, _, _ in LEGS:
        leg = results["eval"].get(tag, {})
        if not leg:
            continue
        print(tag)
        for name, r in leg.items():
            extra = (f"cap {r.get('capped','-'):>4}  "
                     f"delta {r['delta_vs_snap9']:+7.1f} t {r['t']:+.2f} "
                     f"W/L {r['wins']}/{r['losses']}"
                     if "delta_vs_snap9" in r else "(baseline)")
            print(f"  {name:12s} mean {r['mean']:7.1f} {extra}")


if __name__ == "__main__":
    main()
