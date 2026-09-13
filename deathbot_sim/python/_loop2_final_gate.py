"""Pre-registered final-gate driver for loop-gen-2 output (2026-09-12).

G1 DEV GATE: 60 fresh tapes cap 3000, 1-ply greedy, same sweep harness
both arms — final working champion (`--challenger`, a _loop2_champ_rN.pt)
vs snapshot11 (`--incumbent`). PASS iff paired delta > 0 AND t >= 2.0
AND challenger cap rate >= incumbent cap rate.

G2 RECOVERY GATE: full 12-suite x 20-game suite (202600 seed space,
identical boards) on the same two arms via _recovery_suite semantics.
PASS iff 240-game paired mean >= 0 AND no suite paired t < -2.0.

PROMOTION (user-gated, separate step, 5400+ reserve): n=100 cap 3000,
PROMOTE iff paired t >= 2.0 AND cap not lower; one pre-authorized +200
extension iff 0 < t < 2; stop iff delta <= 0 (the §51.4 rule).

Panel accounting: caller passes --dev-seeds START (default 21180 = the
next free dev seed after gen-1 cycle 1 consumed 21150-21179). On PASS
this driver APPENDS the consumed range to PANEL_REGISTRY.md itself, so
the range can never be silently reused.

Usage:
    python _loop2_final_gate.py --challenger ../_diag_out/_loop2_champ_r2.pt \
        --dev-seeds 21180 --out ../_diag_out/_loop2_final_gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
D = HERE.parent / "_diag_out"

GATE_GAMES = 60
GATE_CAP = 3000
REGISTRY = D / "PANEL_REGISTRY.md"


def resolve_weights(p: str) -> Path:
    supplied = Path(p).expanduser()
    for cand in (supplied, HERE / supplied, HERE.parent / supplied,
                 ROOT / supplied):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"weights file not found: {p}")


def paired(dl: list[float]) -> tuple[float, float, int, int]:
    m = statistics.fmean(dl)
    sd = statistics.stdev(dl) if len(dl) > 1 else 0.0
    t = m / (sd / len(dl) ** 0.5) if sd > 0 else float("nan")
    return m, t, sum(x > 0 for x in dl), sum(x < 0 for x in dl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", required=True,
                    help="final _loop2_champ_rN.pt")
    ap.add_argument("--incumbent",
                    default=str(ROOT / "model_weights_snapshot11.pt"))
    ap.add_argument("--dev-seeds", type=int, default=21180)
    ap.add_argument("--out", default=str(D / "_loop2_final_gate.json"))
    args = ap.parse_args()

    op = Path(args.out)
    if op.exists():
        raise SystemExit(f"refusing to overwrite existing gate file {op} "
                         f"(one decision per fresh panel — pick a new --out)")

    from evaluate import evaluate_weights_parallel
    from _recovery_suite import SUITES, build_start_board, play_recovery_game
    from model import TAPNet
    import torch

    t0 = time.perf_counter()
    chal = resolve_weights(args.challenger)
    inc = resolve_weights(args.incumbent)
    results: dict = {
        "note": ("PRE-REGISTERED loop-gen-2 final gate. G1 PASS iff paired "
                 "delta>0 AND t>=2.0 AND cap not lower. G2 PASS iff "
                 "240-game paired mean>=0 AND no suite t<-2.0. Promotion "
                 "is user-gated on the 5400+ reserve (§51.4 rule)."),
        "challenger": str(chal),
        "challenger_sha": hashlib.sha256(chal.read_bytes()).hexdigest()[:16],
        "incumbent": str(inc),
        "incumbent_sha": hashlib.sha256(inc.read_bytes()).hexdigest()[:16],
        "dev_seeds": f"{args.dev_seeds}-{args.dev_seeds + GATE_GAMES - 1}",
        "cap": GATE_CAP,
    }

    # ---- G1: survival dev gate ----
    pre = evaluate_weights_parallel(str(inc), games=GATE_GAMES,
                                    seed=args.dev_seeds, max_pieces=GATE_CAP,
                                    workers=8)
    post = evaluate_weights_parallel(str(chal), games=GATE_GAMES,
                                     seed=args.dev_seeds, max_pieces=GATE_CAP,
                                     workers=8)
    a = [g.pieces for g in post.games]
    b = [g.pieces for g in pre.games]
    dl = [x - y for x, y in zip(a, b)]
    m, t, w, l = paired(dl)
    cap_post = sum(p >= GATE_CAP for p in a)
    cap_pre = sum(p >= GATE_CAP for p in b)
    g1_pass = bool(m > 0 and t >= 2.0 and cap_post >= cap_pre)
    results["G1_survival"] = {
        "challenger_mean": round(statistics.fmean(a), 1),
        "incumbent_mean": round(statistics.fmean(b), 1),
        "delta": round(m, 1), "t": round(t, 2),
        "wins": w, "losses": l,
        "capped": f"{cap_post}/{GATE_GAMES} vs {cap_pre}/{GATE_GAMES}",
        "pass": g1_pass,
    }
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"G1: {statistics.fmean(a):.1f} vs {statistics.fmean(b):.1f} "
          f"(delta {m:+.1f} t {t:+.2f}, cap {cap_post} vs {cap_pre}) -> "
          f"{'PASS' if g1_pass else 'FAIL'}", flush=True)

    # ---- G2: recovery gate (identical boards, paired) ----
    nets = {}
    for name, path in (("challenger", chal), ("incumbent", inc)):
        net = TAPNet(output="linear")
        net.load_state_dict(torch.load(str(path), map_location="cpu"))
        net.eval()
        nets[name] = net
    all_dl: list[float] = []
    per_suite = {}
    worst_t = 0.0
    with torch.inference_mode():
        for suite in SUITES:
            dl_s: list[float] = []
            for g in range(20):
                b0, dk = build_start_board(suite, g, 202_600)
                seed = 202_600 + SUITES.index(suite) * 10_000 + g
                r_post = play_recovery_game(nets["challenger"], b0, dk,
                                            seed, cap=1500)["pieces"]
                r_pre = play_recovery_game(nets["incumbent"], b0, dk,
                                           seed, cap=1500)["pieces"]
                dl_s.append(r_post - r_pre)
            all_dl.extend(dl_s)
            ms, ts, ws, ls = paired(dl_s)
            per_suite[suite] = {"delta": round(ms, 1), "t": round(ts, 2),
                                "wins": ws, "losses": ls}
            worst_t = min(worst_t, ts)
            print(f"  {suite:8s} {ms:+7.1f} t {ts:+.2f} W/L {ws}/{ls}",
                  flush=True)
    mg, tg, wg, lg = paired(all_dl)
    g2_pass = bool(mg >= 0 and worst_t >= -2.0)
    results["G2_recovery"] = {
        "paired_mean_240": round(mg, 1), "t": round(tg, 2),
        "wins": wg, "losses": lg, "worst_suite_t": round(worst_t, 2),
        "per_suite": per_suite, "pass": g2_pass,
    }
    results["verdict"] = ("PROMOTE-TO-5400-PANEL"
                           if (g1_pass and g2_pass) else "HOLD")
    results["wall_s"] = round(time.perf_counter() - t0, 1)
    op.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"G2: 240-game delta {mg:+.1f} t {tg:+.2f}, worst suite t "
          f"{worst_t:+.2f} -> {'PASS' if g2_pass else 'FAIL'}", flush=True)
    print(f"VERDICT: {results['verdict']}", flush=True)

    # ---- registry: the dev range is consumed whatever the verdict ----
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(
            f"\n- seeds: {args.dev_seeds}-{args.dev_seeds + GATE_GAMES - 1}\n"
            f"  protocol: corrected-lockout, cap 3000 "
            f"(loop-gen-2 final gate G1, §52.7)\n"
            f"  status: CONSUMED\n"
            f"  history: {Path(chal).name} "
            f"(sha {results['challenger_sha']}) vs snapshot11, "
            f"{GATE_GAMES} fresh tapes: "
            f"delta {m:+.1f} t {t:+.2f}, cap {cap_post} vs {cap_pre} — "
            f"G1 {'PASS' if g1_pass else 'FAIL'}; G2 "
            f"{'PASS' if g2_pass else 'FAIL'} "
            f"(240-game {mg:+.1f} t {tg:+.2f}, worst suite "
            f"{worst_t:+.2f}) — VERDICT {results['verdict']} "
            f"(file: {op.name}). "
            f"Next free: {args.dev_seeds + GATE_GAMES}+ (dev); "
            f"5400+ reserves for promotion.\n")
    print(f"registry: {args.dev_seeds}-{args.dev_seeds + GATE_GAMES - 1} "
          f"CONSUMED", flush=True)


if __name__ == "__main__":
    main()
