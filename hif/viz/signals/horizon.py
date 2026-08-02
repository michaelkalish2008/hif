"""Input attention-row entropy — self-attention diffuseness per prompt position.

H(ā_{i,0:i}) in bits: the Shannon entropy of a prompt position's self-attention
over its causal prefix. Low = the position's attention concentrates on a few
prior tokens; high = it spreads broadly across the prefix.

Reported in raw bits. It used to be divided by log₂(prefix length) to land in
[0, 1]; that normaliser puts sequence-length metadata into a number labelled
behaviour, and it saturates. The value grows with prefix length by
construction — read it against the position axis, not as a fraction.

IMPORTANT: this is the *Horizon instrument*, NOT the semantic meaning-cloud. The
meaning-cloud (candidate clustering) lives under Exposure ◇, where it belongs.

Backing data: ``profile.attention`` (prompt self-attention) — requires attention
capture (open HuggingFace models).

SUBJECT: prompt-only. The attention here is the *analysis encoder's*, not the
profiled model's (hif/analysis/attention.py never accesses the generating
model's attention weights), and this side reads the prompt. The value is a
function of prompt text and encoder weights alone, so it cannot vary with the
profiled model at all — chart it as a property of the prompt, never as evidence
about the model. The corresponding record value lives in `prompt_measurements`,
not `measurements`. See docs/MEASUREMENTS.md § Subject.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import NEEDS_ATTENTION, na_figure, save_fig, signal_title
from hif.viz._theme import INDIGO, dark_layout
from hif.viz.signals._attention import get_attention_map, row_entropy_trace
from hif.profile.schema import BehavioralRangeProfile

LABEL, GLYPH = "Input attention entropy", None


def available(profile: BehavioralRangeProfile) -> str | None:
    tokens, weights = get_attention_map(profile, "input")
    return None if weights else NEEDS_ATTENTION


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    tokens, weights = get_attention_map(profile, "input")
    if not weights:
        return save_fig(na_figure(LABEL, GLYPH, NEEDS_ATTENTION), output_path, formats)

    trace = row_entropy_trace(weights) or []
    x = list(range(len(trace)))
    toks = tokens or [""] * len(trace)
    mean = float(np.mean(trace)) if trace else 0.0

    hover = [f"Position {i} — {t!r}<br>Attention-row entropy: {v:.3f} bits"
             for i, t, v in zip(x, toks, trace)]

    fig = go.Figure(go.Scatter(x=x, y=trace, mode="lines+markers",
                               line=dict(color=INDIGO, width=2.2), marker=dict(size=6),
                               fill="tozeroy", fillcolor="rgba(99,102,241,0.10)",
                               hovertext=hover, hoverinfo="text"))
    fig.add_hline(y=mean, line_dash="dash", line_color=INDIGO, opacity=0.5,
                  annotation_text=f"mean {mean:.2f} bits", annotation_position="top left")
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, GLYPH, profile.model.name,
                           "Self-attention row entropy per prompt position, in bits — low = "
                           "attention concentrated on a few tokens, high = spread across "
                           "the causal prefix (grows with prefix length)"),
        xaxis=dict(title="Prompt token position"),
        yaxis=dict(title="Attention-row entropy (bits)", rangemode="tozero"),
        height=460, showlegend=False, margin=dict(t=90, b=60),
    ))
    return save_fig(fig, output_path, formats)
