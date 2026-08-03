"""Step-to-step output divergence (per-step view).

Fidelity: Shiftⱼ = JSD(Qⱼ₋₁, Qⱼ), the Jensen-Shannon divergence between
consecutive steps' output distributions. Within a single forward pass: tall bars
mark abrupt vocabulary pivots (the field of viable tokens reorganized sharply),
low bars mark smooth continuation. Computed from the stored top-K distributions,
aligned by token id over the union of the two steps' candidates.

This module DRAWS the quantity; it does not define it. The computation lives in
``hif/metrics/shift.py`` and is shared with the ``output_step_jsd_bits``
measurement, so the bars here and the number in a machine record are the same
arithmetic and cannot drift.

MEASUREMENT CAVEAT (real, not cosmetic): JSD is computed only over the stored
top-K candidates, not the full vocabulary. When two consecutive steps' top-K
sets share little or no overlap, JSD saturates at exactly 1 bit regardless of
how similar the true full-vocabulary distributions are — disjoint support alone
is enough to hit the ceiling. A chart where most bars sit near 1 more often
reflects narrow top-K supports failing to overlap than genuine maximal
divergence. The chart surfaces the top-K overlap fraction (in hover, and a
banner when overlap is low) so this isn't silently mistaken for "everything is
maximally different." The record carries the same fact as the companion
measurement ``output_step_topk_overlap_fraction``.

Backing data: consecutive ``output_side.steps[].topk`` (≥ 2 steps, with real
distributions — a selected-only backend has none, and the panel says so).
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from hif.metrics.shift import LABEL, shift_summary, shift_trace, unavailable_reason
from hif.viz.base import na_figure, save_fig, signal_title
from hif.viz._theme import VIOLET, dark_layout
from hif.profile.schema import BehavioralRangeProfile


def available(profile: BehavioralRangeProfile) -> str | None:
    steps = getattr(profile.output_side, "steps", None) or []
    return unavailable_reason(steps)


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, reason), output_path, formats)

    steps = profile.output_side.steps
    trace = shift_trace(steps)
    summary = shift_summary(steps)
    x = [t.step for t in trace]
    jsd = [t.jsd_bits for t in trace]
    mean = summary.mean_jsd_bits
    median_overlap = summary.median_overlap_fraction

    hover = [
        f"Step {t.step-1} → {t.step}<br>chose {steps[t.step].selected_token_str!r}<br>"
        f"JSD: {t.jsd_bits:.4f}<br>"
        f"top-K overlap: {int(t.topk_overlap_fraction*100)}%"
        + (" — low overlap alone can saturate JSD near 1"
           if t.topk_overlap_fraction < 0.3 else "")
        for t in trace
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
        title=signal_title(LABEL, profile.model.name, subtitle),
        xaxis=dict(title="Generation step (transition into)"),
        yaxis=dict(title="JSD (bits)", rangemode="tozero", range=[0, max(jsd + [0.01]) * 1.15]),
        height=520, showlegend=False, margin=dict(t=150, b=60),
    ))
    return save_fig(fig, output_path, formats)
