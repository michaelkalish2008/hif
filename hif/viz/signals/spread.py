"""Spread ■ (per-step view) — attention spread over context.

Fidelity: attention-row entropy per generated token — how evenly the model's
attention was distributed across prior context when generating each token.
A value of k bits ≈ 2^k context positions receiving meaningful weight. High =
diffuse attention; low = concentrated. Measured in context-position space
(orthogonal to Entropy ●, which is vocabulary space).

Backing data: ``profile.attention`` (continuation attention) — requires
attention capture (open HuggingFace models).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import NEEDS_ATTENTION, na_figure, save_fig, signal_title
from hif.viz._theme import AMBER, dark_layout
from hif.viz.signals._attention import get_attention_map, row_entropy_trace
from hif.profile.schema import BehavioralRangeProfile

LABEL, GLYPH = "Output attention-row entropy (bits)", "■"


def available(profile: BehavioralRangeProfile) -> str | None:
    tokens, weights = get_attention_map(profile, "output")
    return None if weights else NEEDS_ATTENTION


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    tokens, weights = get_attention_map(profile, "output")
    if not weights:
        return save_fig(na_figure(LABEL, GLYPH, NEEDS_ATTENTION), output_path, formats)

    trace = row_entropy_trace(weights) or []
    x = list(range(len(trace)))
    toks = tokens or [""] * len(trace)
    mean = float(np.mean(trace)) if trace else 0.0

    hover = [f"Token {i} — {t!r}<br>Attention spread: {v:.3f} bits (≈{2**v:.1f} positions)"
             for i, t, v in zip(x, toks, trace)]

    fig = go.Figure(go.Scatter(x=x, y=trace, mode="lines+markers",
                               line=dict(color=AMBER, width=2.2), marker=dict(size=6),
                               fill="tozeroy", fillcolor="rgba(245,158,11,0.10)",
                               hovertext=hover, hoverinfo="text"))
    fig.add_hline(y=mean, line_dash="dash", line_color=AMBER, opacity=0.5,
                  annotation_text=f"mean {mean:.2f} bits", annotation_position="top left")
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, GLYPH, profile.model.name,
                           "Attention spread over context positions per token — high = attention diffuse "
                           "across many positions, low = concentrated"),
        xaxis=dict(title="Token position"),
        yaxis=dict(title="Attention entropy (bits)", rangemode="tozero"),
        height=460, showlegend=False, margin=dict(t=90, b=60),
    ))
    return save_fig(fig, output_path, formats)
