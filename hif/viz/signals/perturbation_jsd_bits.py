"""Sensitivity (aggregate view) — output shift under meaning-preserving paraphrase.

Fidelity: Sensitivity = mean Jensen-Shannon divergence between the baseline and
paraphrase-variant output distributions, across perturbation generators. The
faithful chart is a per-generator JSD bar series (which paraphrase family moved
the output most), with the aggregate mean line.

Backing data: ``metrics.sensitivity[]`` — requires perturbation variants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import NEEDS_PERTURBATION, na_figure, save_fig, signal_title
from hif.viz._theme import VIOLET, AMBER, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL = "Perturbation JSD (bits)"


def available(profile: BehavioralRangeProfile) -> str | None:
    sens = getattr(profile.metrics, "sensitivity", None) or []
    return None if len(sens) > 0 else NEEDS_PERTURBATION


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, reason), output_path, formats)

    sens = profile.metrics.sensitivity
    gens = [s.perturbation_generator for s in sens]
    jsd = [s.mean_js_divergence for s in sens]
    mean = float(np.mean(jsd)) if jsd else 0.0

    hover = [
        f"{g}<br>Mean JSD: {j:.4f}<br>Mean KL: {s.mean_kl_divergence:.4f}<br>"
        f"Δ entropy: {s.output_entropy_delta:+.3f} bits"
        for g, j, s in zip(gens, jsd, sens)
    ]

    fig = go.Figure(go.Bar(
        x=gens, y=jsd, marker_color=VIOLET, opacity=0.85,
        hovertext=hover, hoverinfo="text", name="Mean JSD",
    ))
    fig.add_hline(y=mean, line_dash="dash", line_color=AMBER,
                  annotation_text=f"Sensitivity = {mean:.3f}", annotation_position="top right")
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, profile.model.name,
                           "Output shift (Jensen-Shannon divergence) under each paraphrase family · "
                           "low = robust, high = brittle to surface wording"),
        xaxis=dict(title="Perturbation generator", tickangle=-30,
                   tickfont=dict(size=10)),
        yaxis=dict(title="Mean JSD (bits)", rangemode="tozero"),
        height=480, showlegend=False, margin=dict(t=90, b=120),
    ))
    return save_fig(fig, output_path, formats)
