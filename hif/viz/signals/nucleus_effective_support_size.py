"""Nucleus effective support size (aggregate view) — width of the active choice set per step.

Fidelity: mean nucleus Effective Support Size (ESS = 2^nucleus_entropy, at the
pinned 0.95 nucleus) across generation steps — the average number of tokens
meaningfully in play. The faithful chart is the per-step ESS trace with the
mean line; peaks are exploratory steps, troughs are committed ones.

The basis is in the name because it decides the ceiling: this counts tokens
inside the renormalized 95% nucleus, so it is bounded by the nucleus size and
not by the vocabulary. It is a different count from the display face of
`output_entropy_bits`, which is over the top-K distribution as the backend
exposed it, and the two must not be read against each other. See
`effective_support_size()` in hif/metrics/distribution.py for why the
transform is 2^H (Grendar 2006).

Backing data: ``metrics.distribution[].nucleus_effective_support_size``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import NEEDS_DISTRIBUTION, na_figure, save_fig, signal_title
from hif.viz._theme import INDIGO, AMBER, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL = "Nucleus effective support size"


def available(profile: BehavioralRangeProfile) -> str | None:
    dist = getattr(profile.metrics, "distribution", None) or []
    return None if len(dist) > 0 else NEEDS_DISTRIBUTION


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, reason), output_path, formats)

    dist = profile.metrics.distribution
    steps = list(range(len(dist)))
    ess = [d.nucleus_effective_support_size for d in dist]
    out_steps = profile.output_side.steps
    toks = [out_steps[i].selected_token_str if i < len(out_steps) else "" for i in steps]
    mean = float(np.mean(ess)) if ess else 0.0

    hover = [
        f"Step {i} → {tok!r}<br>Effective support: {e:.1f} tokens<br>"
        f"(nucleus entropy {d.nucleus_entropy_bits:.2f} bits)"
        for i, tok, e, d in zip(steps, toks, ess, dist)
    ]

    fig = go.Figure()
    # "mean" spelled out explicitly — a dashed line alone reads as an arbitrary
    # threshold, not obviously the average of the series it cuts through.
    fig.add_hline(y=mean, line_dash="dash", line_color=AMBER,
                  annotation_text=f"mean = {mean:.1f} tokens",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=steps, y=ess, mode="lines+markers",
        line=dict(color=INDIGO, width=2.2), marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.10)",
        hovertext=hover, hoverinfo="text", name="Nucleus effective support size",
    ))
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, profile.model.name,
                           "Effective support size of the 95% nucleus per step (2^nucleus-entropy) — how "
                           "many tokens were genuinely in play · high = exploratory, low = committed · shown "
                           "in generation order — the trend across the response is the signal, not just the peak"),
        xaxis=dict(title="Generation step (chronological)"),
        yaxis=dict(title="Nucleus effective support (tokens)", rangemode="tozero"),
        height=480, showlegend=False, margin=dict(t=110, b=60),
    ))
    return save_fig(fig, output_path, formats)
