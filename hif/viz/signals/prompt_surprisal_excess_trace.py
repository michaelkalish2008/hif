"""Prompt surprisal excess, per position — underdog selections against a committed field.

Fidelity: Surprise = mean over prompt positions of max(0, sᵢ − H(Pᵢ)), the
excess surprisal of the actual token beyond the model's distributional entropy.
The faithful chart is the per-position excess-surprisal bar series; the mean is
the aggregate score. (The wager chart draws this same quantity at full
per-position resolution — see wager.py.)

Backing data: ``input_side.positions`` — requires teacher forcing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import NEEDS_TEACHER_FORCING, add_click_to_dim_js, na_figure, save_fig, signal_title
from hif.viz._theme import AMBER, TEXT_SEC, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL = "Prompt surprisal excess (trace)"


def available(profile: BehavioralRangeProfile) -> str | None:
    positions = getattr(profile.input_side, "positions", None) or []
    return None if len(positions) > 0 else NEEDS_TEACHER_FORCING


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, reason), output_path, formats)

    positions = profile.input_side.positions
    idx = [p.position for p in positions]
    excess = [max(0.0, p.surprisal - p.entropy) for p in positions]
    toks = [p.token_str for p in positions]
    # Categorical labels carrying the actual token — a bare position number tells
    # the reader nothing about WHICH token was surprising.
    x_labels = [f"{i}: {tok!r}" for i, tok in zip(idx, toks)]
    mean = float(np.mean(excess)) if excess else 0.0

    hover = [
        f"Position {i} — {tok!r}<br>Surprisal: {p.surprisal:.2f} bits<br>"
        f"Entropy: {p.entropy:.2f} bits<br>Excess: {e:.2f} bits"
        for i, tok, p, e in zip(idx, toks, positions, excess)
    ]

    fig = go.Figure(go.Bar(
        x=x_labels, y=excess, marker_color=AMBER, opacity=0.85,
        hovertext=hover, hoverinfo="text", name="Excess surprisal",
    ))
    fig.add_hline(y=mean, line_dash="dash", line_color=TEXT_SEC,
                  annotation_text=f"Surprise = {mean:.3f}", annotation_position="top left")
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, profile.model.name,
                           "Per-position excess surprisal max(0, sᵢ − H) — tall bars mark tokens the model "
                           "committed against and still selected · shown in prompt order (left to right) so "
                           "spikes can be read against sentence structure · click a bar to isolate it"),
        xaxis=dict(title="Prompt token position", categoryorder="array", categoryarray=x_labels,
                   tickangle=-55, tickfont=dict(size=9)),
        yaxis=dict(title="Excess surprisal (bits)", rangemode="tozero"),
        height=550, showlegend=False, margin=dict(t=155, b=110),
    ))
    result = save_fig(fig, output_path, formats)
    add_click_to_dim_js(result["html"])
    return result
