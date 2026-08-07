"""Input entropy trace — per-position uncertainty while reading the prompt.

The chart is the per-position input-entropy trace in bits, with its mean and
±1 SD band. Both the trace and the band are in bits; nothing is normalised and
nothing is inverted.

Backing data: ``input_side.positions[].entropy`` — requires teacher forcing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import NEEDS_TEACHER_FORCING, na_figure, save_fig, signal_title
from hif.viz._theme import INDIGO, AMBER, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL = "Input entropy trace"


def available(profile: BehavioralRangeProfile) -> str | None:
    positions = getattr(profile.input_side, "positions", None) or []
    return None if len(positions) > 0 else NEEDS_TEACHER_FORCING


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, reason), output_path, formats)

    positions = profile.input_side.positions
    idx = [p.position for p in positions]
    ent = [p.entropy for p in positions]
    mean = float(np.mean(ent)) if ent else 0.0
    sd = float(np.std(ent)) if ent else 0.0
    shift = getattr(profile.metrics.stability, "input_entropy_shift_bits", None)
    score_txt = (
        f"input entropy shift under perturbation = {shift:.3g} bits"
        if shift is not None else "input entropy shift: absent this run"
    )

    fig = go.Figure()
    # ±1σ band around the mean
    fig.add_hrect(y0=mean - sd, y1=mean + sd, fillcolor="rgba(99,102,241,0.10)",
                  line_width=0, layer="below")
    fig.add_hline(y=mean, line_dash="dash", line_color=AMBER,
                  annotation_text=f"mean {mean:.2f} bits", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=idx, y=ent, mode="lines+markers",
        line=dict(color=INDIGO, width=2), marker=dict(size=6),
        hovertemplate="Position %{x}<br>Input entropy: %{y:.3f} bits<extra></extra>",
        name="Per-position input entropy",
    ))
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, profile.model.name,
                           f"Per-position uncertainty while reading the prompt · {score_txt} · "
                           "SD is in bits"),
        xaxis=dict(title="Prompt token position"),
        yaxis=dict(title="Input entropy (bits)", rangemode="tozero"),
        height=460, showlegend=False, margin=dict(t=90, b=60),
    ))
    return save_fig(fig, output_path, formats)
