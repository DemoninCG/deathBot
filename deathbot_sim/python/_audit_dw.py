"""One-off audit: per-checkpoint weight delta from its start point, by layer.

Purpose (2026-09-11): quantify the weight-space displacement of every
surviving distillation checkpoint from its starting point, to test the
"trajectory vs budget vs loss-surface" hypotheses for why distillation
almost never worked. Read-only over checkpoints; writes a small JSON.

Runs from deathbot_sim/python; standalone (no panel, no gameplay).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
D = HERE.parent / "_diag_out"
ROOT = HERE.parent.parent  # deathBot/

snap9 = ROOT / "model_weights_snapshot9.pt"
snap7 = ROOT / "model_weights_snapshot7.pt"

ENTRIES = [
    # (label, checkpoint path, start-weights path)
    ("C3_seed42 SUCCESS snap7->+231", D / "phase2_arm_C3_affine_fit_seed42.pt", snap7),
    ("fullsplit_seed42 SUCCESS snap9->+468", D / "fullsplit_seed42.pt", snap9),
    ("P2T_T1 FAIL neutral (2-ply CE T=1)", D / "p2t_T1_seed42.pt", snap9),
    ("P2T_T3 FAIL neutral (2-ply CE T=3)", D / "p2t_T3_seed42.pt", snap9),
    ("E3_armA FAIL (continued CE from fullsplit)", D / "e3_round.armA.pt", D / "fullsplit_seed42.pt"),
    ("E3_armB FAIL (aggregate CE from fullsplit)", D / "e3_round.armB.pt", D / "fullsplit_seed42.pt"),
    ("D1_iter2_round2 (FAIL n.s. snap9->snap9)", D / "distill_D1_iter2_scale30_steps400_seed42.pt", snap9),
]


def load_sd(p: Path) -> dict:
    sd = torch.load(str(p), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def main() -> None:
    ref_cache: dict[str, dict] = {}

    def get_ref(p: Path) -> dict:
        k = str(p)
        if k not in ref_cache:
            ref_cache[k] = load_sd(p)
        return ref_cache[k]

    out = []
    for label, ckpt, start in ENTRIES:
        if not ckpt.exists():
            print(f"MISSING {label}: {ckpt.name}")
            continue
        sd = load_sd(ckpt)
        rec = {
            "label": label,
            "ckpt": str(ckpt),
            "ckpt_sha": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16],
            "start": str(start),
            "start_sha": hashlib.sha256(start.read_bytes()).hexdigest()[:16],
        }
        if start.exists():
            ref = get_ref(start)
            per_layer = {}
            tot = 0.0
            for k in sd:
                if k in ref:
                    dv = sd[k].float() - ref[k].float()
                    n = float(dv.norm())
                    per_layer[k] = round(n, 4)
                    tot += n * n
            rec["dW_total_l2"] = round(tot ** 0.5, 3)
            rec["layers"] = per_layer
            rel = {}
            for k, v in per_layer.items():
                base = float(ref[k].norm()) or 1.0
                rel[k] = round(v / base, 5)
            rec["rel_per_layer"] = rel
        out.append(rec)
        print(f"{label}: dW={rec.get('dW_total_l2')}")

    dest = HERE / "_audit_dw_out.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
