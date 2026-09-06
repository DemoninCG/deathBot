"""Decoupled style head: frozen survival value + trainable style module.

Rationale (2026-09-06 night, §14): shared-weight P2T distillation was
neutral-to-negative (§13.3) — fitting style preferences into snapshot9's
weights moves survival-critical rankings. This module keeps snapshot9
FROZEN and trains a SEPARATE small MLP over exact heuristic features;
combination happens only at search time (v + alpha * style, survivors
only), so survival cannot regress through shared weights by construction.

Style MLP input (8 dims, exact, no learning needed to extract):
  [weighted_height, holes, hole_depth, accessible_cavities, well_cost,
   spawn_danger, lines_cleared, deepest_well]
Output: scalar style bonus in z-units. Zero-init final layer => starts as
the survival policy, style grows only via training signal.

Training signal (Phase A, search-free): pairwise preference on bank states
- teacher = snapshot9's own 2-ply argmax vs its 1-ply argmax. On states
  where they disagree, the style head learns the FEATURE direction of the
  2-ply pick (margin ranking loss). No value regression, no absolute
  labels, no shared weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STYLE_DIM = 8


def style_features_from_rows(rows21: np.ndarray, lines: int) -> np.ndarray:
    """Exact heuristic features + deepest well, from packed [21] rows.

    Pure-numpy replica of heuristic.py semantics (bottom-first bitmasks):
    heights, weighted height, holes/accessible split, hole depth, well
    triangular cost, spawn danger, plus deepest single well depth.
    Returns float32 [8].
    """
    from heuristic import HEIGHT_WEIGHTS, SPAWN_COLUMNS, SPAWN_DANGER_HEIGHT
    rows = np.asarray(rows21, dtype=np.uint32).reshape(-1)
    assert rows.shape[0] == 21
    occ = np.zeros((21, 10), dtype=bool)
    for r in range(21):
        bits = int(rows[r])
        for c in range(10):
            if bits & (1 << c):
                occ[r, c] = True
    heights = np.zeros(10, dtype=int)
    for c in range(10):
        col = np.where(occ[:, c])[0]
        heights[c] = int(col.max()) + 1 if len(col) else 0
    weighted_height = float(sum(heights[c] * HEIGHT_WEIGHTS[c] for c in range(10)))
    holes = acc = 0
    hole_depth = 0
    max_h = int(heights.max()) if heights.max() > 0 else 0
    for row in range(max_h):
        row_occ = occ[row]
        left = np.cumprod(~row_occ)
        right = np.cumprod(~row_occ[::-1])[::-1]
        for col in range(10):
            if row >= heights[col] or row_occ[col]:
                continue
            if left[col] or right[col]:
                acc += 1
            else:
                holes += 1
                hole_depth += int(occ[row + 1:, col].sum())
    wells = []
    for col, level in enumerate(heights):
        if col == 0:
            d = heights[1] - level
        elif col == 9:
            d = heights[8] - level
        else:
            d = min(heights[col - 1], heights[col + 1]) - level
        if d > 0:
            wells.append(int(d))
    well_cost = float(sum((d - 1) * d // 2 for d in wells))
    deepest = float(max(wells)) if wells else 0.0
    spawn = float(sum(max(0, heights[c] - SPAWN_DANGER_HEIGHT) ** 2
                      for c in SPAWN_COLUMNS))
    return np.array([weighted_height, float(holes), float(hole_depth),
                     float(acc), well_cost, spawn, float(lines),
                     deepest], dtype=np.float32)


class StyleHead(nn.Module):
    """Small MLP: 8 heuristic features -> scalar style bonus (z-units).

    Inputs must be STANDARDIZED (see fit_feature_stats/apply_stats):
    raw features span O(100) (weighted_height) to O(1) (lines), so an
    unnormalized MLP lets the large-scale dims dominate and learns a big
    positive bias (Phase A: mean +86). Standardization keeps every dim in
    the game and the ranking loss focused on direction, not scale.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(STYLE_DIM, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        # Zero-init output: starts as pure survival policy.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(feats)))

    def calibrated(self, feats: torch.Tensor, ref_feats: torch.Tensor) -> torch.Tensor:
        """Zero-mean calibrated output: style(feats) - style(ref).mean().

        Raw head outputs carry a large positive bias (Phase A: mean +86 over
        candidates), so alpha*style acts as a per-state constant that drowns
        survival margins. Subtracting the reference (batch/state) mean keeps
        only the WITHIN-state ranking signal the combination needs.
        """
        out = self.forward(feats)
        with torch.no_grad():
            ref_mean = self.forward(ref_feats).mean()
        return out - ref_mean


def fit_feature_stats(feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim mean/std for standardization (std floored at 1e-3)."""
    mean = feats.mean(0)
    std = feats.std(0, unbiased=False).clamp(min=1e-3)
    return mean.detach(), std.detach()


def apply_stats(feats: torch.Tensor, mean: torch.Tensor,
                std: torch.Tensor) -> torch.Tensor:
    """Standardize: (feats - mean) / std."""
    return (feats - mean.to(feats.device)) / std.to(feats.device)


def state_dict_to_vector(sd: dict) -> torch.Tensor:
    return torch.cat([v.detach().flatten().float() for v in sd.values()])


def snapshot9_frozen_copy(device: str = "cpu"):
    """Load snapshot9 and freeze every parameter (survival net never trains)."""
    from model import TAPNet
    net = TAPNet(output="linear")
    w = HERE.parent.parent / "model_weights_snapshot9.pt"
    net.load_state_dict(torch.load(w, map_location=device))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def combined_scores(survival_vals: torch.Tensor, style_vals: torch.Tensor,
                    alpha: float, veto_margin: float = 25.0) -> torch.Tensor:
    """Search-time combination with a survival veto gate.

    Candidates more than `veto_margin` (z-units) below the best survival
    value are ineligible (set to -inf) BEFORE style is applied; style only
    re-ranks veto survivors. Additive v + alpha*style alone lets a large
    style term overrule survival (Phase A collapse: 0/30 cap); the veto
    makes survival-first structural instead of scale-dependent.
    """
    best = survival_vals.max()
    eligible = (best - survival_vals) <= veto_margin
    out = survival_vals + alpha * style_vals
    return torch.where(eligible, out,
                         torch.full_like(out, float("-inf")))
