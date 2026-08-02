"""Shift ◆ (per-step view) — step-to-step output divergence.

Fidelity: Shiftⱼ = JSD(Qⱼ₋₁, Qⱼ), the Jensen-Shannon divergence between
consecutive steps' output distributions. Within a single forward pass: tall bars
mark abrupt vocabulary pivots (the field of viable tokens reorganized sharply),
low bars mark smooth continuation. Computed from the stored top-K distributions,
aligned by token id over the union of the two steps' candidates.

MEASUREMENT CAVEAT (real, not cosmetic): JSD is computed only over the stored
top-K candidates, not the full vocabulary. When two consecutive steps' top-K
sets share little or no overlap, JSD saturates at exactly 1 bit regardless of
how similar the true full-vocabulary distributions are — disjoint support alone
is enough to hit the ceiling. A chart where most bars sit near 1 more often
reflects narrow top-K supports failing to overlap than genuine maximal
divergence. The chart surfaces the top-K overlap fraction (in hover, and a
banner when overlap is low) so this isn't silently mistaken for "everything is
maximally different."

Backing data: consecutive ``output_side.steps[].topk`` (≥ 2 steps).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import na_figure, save_fig, signal_title
from hif.viz._theme import VIOLET, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL, GLYPH = "Shift", "◆"
_NEEDS = "Requires at least two generation steps with top-K distributions."


def available(profile: BehavioralRangeProfile) -> str | None:
    steps = getattr(profile.output_side, "steps", None) or []
    return None if len(steps) >= 2 else _NEEDS


def _jsd(p: dict[int, float], q: dict[int, float]) -> float:
    """Jensen-Shannon divergence (bits) between two sparse top-K distributions,
    aligned over the union of their token ids and renormalized."""
    keys = set(p) | set(q)
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    ps, qs = pv.sum(), qv.sum()
    if ps <= 0 or qs <= 0:
        return 0.0
    pv, qv = pv / ps, qv / qs
    m = 0.5 * (pv + qv)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return max(0.0, min(1.0, 0.5 * _kl(pv, m) + 0.5 * _kl(qv, m)))


def _overlap_frac(p: dict[int, float], q: dict[int, float]) -> float:
    """|shared top-K tokens| / |union| between two steps — explains saturation."""
    union = set(p) | set(q)
    if not union:
        return 0.0
    return len(set(p) & set(q)) / len(union)


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, GLYPH, reason), output_path, formats)

    steps = profile.output_side.steps
    dists = [{e.token_id: e.prob for e in s.topk} for s in steps]
    jsd = [_jsd(dists[i - 1], dists[i]) for i in range(1, len(dists))]
    overlap = [_overlap_frac(dists[i - 1], dists[i]) for i in range(1, len(dists))]
    x = list(range(1, len(dists)))
    toks = [steps[i].selected_token_str for i in x]
    mean = float(np.mean(jsd)) if jsd else 0.0
    median_overlap = float(np.median(overlap)) if overlap else 1.0

    hover = [
        f"Step {i-1} → {i}<br>chose {steps[i].selected_token_str!r}<br>JSD: {j:.4f}<br>"
        f"top-K overlap: {int(o*100)}%"
        + (" — low overlap alone can saturate JSD near 1" if o < 0.3 else "")
        for i, j, o in zip(x, jsd, overlap)
    ]

    fig = go.Figure(go.Bar(x=x, y=jsd, marker_color=VIOLET, opacity=0.85,
                           hovertext=hover, hoverinfo="text"))
    fig.add_hline(y=mean, line_dash="dash", line_color=VIOLET, opacity=0.6,
                  annotation_text=f"mean {mean:.3f}", annotation_position="top left")

    subtitle = ("Step-to-step output divergence JSD(Qⱼ₋₁, Qⱼ) — tall bars are abrupt vocabulary "
                "pivots within one forward pass")
    if median_overlap < 0.5:
        subtitle += (f" · caveat: median top-K overlap between steps is only "
                     f"{int(median_overlap*100)}% — JSD saturates near 1 bit from disjoint top-K "
                     f"support alone, not necessarily true maximal divergence (hover per bar)")

    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, GLYPH, profile.model.name, subtitle),
        xaxis=dict(title="Generation step (transition into)"),
        yaxis=dict(title="JSD (bits)", rangemode="tozero", range=[0, max(jsd + [0.01]) * 1.15]),
        height=520, showlegend=False, margin=dict(t=150, b=60),
    ))
    return save_fig(fig, output_path, formats)
