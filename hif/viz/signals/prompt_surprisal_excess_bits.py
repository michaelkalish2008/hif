"""Prompt surprisal excess (per-position view) — surprisal over entropy.

Fidelity: excessᵢ = max(0, sᵢ − H(Pᵢ)), the per-position residual cost of the
actual token beyond the model's distributional entropy. This is the full-
resolution instrument behind prompt_surprisal_excess_bits (its mean).

Two-panel chart: the top panel overlays surprisal sᵢ against entropy H(Pᵢ) so
the reader sees where they diverge; the bottom panel bars the excess (the
excess value itself) directly, per position, so the delta doesn't rely solely on
the tooltip to read.

Backing data: ``input_side.positions`` — requires teacher forcing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hif.viz.base import NEEDS_TEACHER_FORCING, add_click_to_dim_js, na_figure, save_fig, signal_title
from hif.viz._theme import AMBER, INDIGO, RED, TEXT_SEC, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL = "Prompt surprisal excess (bits)"


def available(profile: BehavioralRangeProfile) -> str | None:
    positions = getattr(profile.input_side, "positions", None) or []
    return None if len(positions) > 0 else NEEDS_TEACHER_FORCING


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, reason), output_path, formats)

    positions = profile.input_side.positions
    idx = [p.position for p in positions]
    surp = [p.surprisal for p in positions]
    ent = [p.entropy for p in positions]
    excess = [max(0.0, s - h) for s, h in zip(surp, ent)]
    toks = [p.token_str for p in positions]
    x_labels = [f"{i}: {tok!r}" for i, tok in zip(idx, toks)]
    mean_excess = float(np.mean(excess)) if excess else 0.0

    hover = [f"Position {i} — {tok!r}<br>Surprisal sᵢ: {s:.2f} bits<br>"
             f"Entropy H(Pᵢ): {h:.2f} bits<br>Surprisal excess: {e:.2f} bits"
             for i, tok, s, h, e in zip(idx, toks, surp, ent, excess)]

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.16,
        shared_xaxes=True,
        # No top-panel subplot title — it's redundant with the main subtitle and
        # collided with the legend sitting just above the chart area.
        subplot_titles=["", "Surprisal excess per position — the delta itself, not just the gap"],
    )

    # Top: entropy floor + surprisal line; the gap above the floor is the excess.
    fig.add_trace(go.Scatter(x=x_labels, y=ent, mode="lines", name="Entropy H(Pᵢ)",
                             line=dict(color=INDIGO, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=surp, mode="lines+markers", name="Surprisal sᵢ",
                             line=dict(color=AMBER, width=2), marker=dict(size=5),
                             hovertext=hover, hoverinfo="text"), row=1, col=1)
    hi = [(x, s) for x, s, e in zip(x_labels, surp, excess) if e >= max(mean_excess * 2, 1.0)]
    if hi:
        fig.add_trace(go.Scatter(x=[x for x, _ in hi], y=[s for _, s in hi],
                                 mode="markers", name="High excess",
                                 marker=dict(color=RED, size=10, symbol="triangle-up"),
                                 hoverinfo="skip"), row=1, col=1)

    # Bottom: the excess itself, as a bar per position — the delta made explicit.
    bar_colors = [RED if e >= max(mean_excess * 2, 1.0) else AMBER for e in excess]
    fig.add_trace(go.Bar(x=x_labels, y=excess, marker_color=bar_colors, opacity=0.85,
                         hovertext=hover, hoverinfo="text", showlegend=False,
                         name="Surprisal excess"), row=2, col=1)
    fig.add_hline(y=mean_excess, line_dash="dash", line_color=TEXT_SEC, row=2, col=1,
                  annotation_text=f"mean = {mean_excess:.3f} bits", annotation_position="top left")

    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, profile.model.name,
                           f"Mean excess surprisal {mean_excess:.3f} bits · gap above the entropy "
                           "line in the top panel = the bar height in the bottom panel · click a bar to isolate it"),
        xaxis=dict(categoryorder="array", categoryarray=x_labels, showticklabels=False),
        xaxis2=dict(title="Prompt token position", categoryorder="array", categoryarray=x_labels,
                    tickangle=-55, tickfont=dict(size=9)),
        yaxis=dict(title="Bits", rangemode="tozero"),
        yaxis2=dict(title="Surprisal excess (bits)", rangemode="tozero"),
        height=720,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom"),
        margin=dict(t=170, b=110),
    ))
    result = save_fig(fig, output_path, formats, png_size=(1000, 700))
    add_click_to_dim_js(result["html"])
    return result
