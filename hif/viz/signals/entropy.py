"""Entropy ● (per-step view) — output distribution entropy per step.

Fidelity: the per-step Shannon entropy H(Qⱼ) in bits — the full trace that the
Breadth aggregate compresses (Breadth = mean of ESS = 2^H). Shows whether a mean
conceals a flat plateau, a single spike, or alternating peaks/troughs. Both the
nucleus entropy (comparable across model types) and the raw top-K entropy
(truncation lower bound) are drawn.

Backing data: ``metrics.distribution[].entropy_bits`` / ``nucleus_entropy_bits``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import NEEDS_DISTRIBUTION, na_figure, save_fig, signal_title
from hif.viz._theme import EMERALD, TEXT_MUTED, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL, GLYPH = "Entropy", "●"


def available(profile: BehavioralRangeProfile) -> str | None:
    dist = getattr(profile.metrics, "distribution", None) or []
    return None if len(dist) > 0 else NEEDS_DISTRIBUTION


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, GLYPH, reason), output_path, formats)

    dist = profile.metrics.distribution
    steps = list(range(len(dist)))
    h_raw = [d.entropy_bits for d in dist]
    h_nuc = [d.nucleus_entropy_bits for d in dist]
    out_steps = profile.output_side.steps
    toks = [out_steps[i].selected_token_str if i < len(out_steps) else "" for i in steps]
    mean_nuc = float(np.mean(h_nuc)) if h_nuc else 0.0

    hover = [f"Step {i} → {tok!r}<br>Nucleus H: {n:.3f} bits<br>Raw top-K H: {r:.3f} bits"
             for i, tok, n, r in zip(steps, toks, h_nuc, h_raw)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=h_raw, mode="lines", name="Raw top-K entropy (≈ lower bound)",
                             line=dict(color=TEXT_MUTED, width=1.4, dash="dot"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=steps, y=h_nuc, mode="lines+markers", name="Nucleus entropy (95%)",
                             line=dict(color=EMERALD, width=2.4), marker=dict(size=6),
                             hovertext=hover, hoverinfo="text"))
    fig.add_hline(y=mean_nuc, line_dash="dash", line_color=EMERALD, opacity=0.5,
                  annotation_text=f"mean {mean_nuc:.2f} bits", annotation_position="top left")
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, GLYPH, profile.model.name,
                           "Output distribution entropy per step (bits) — peaks are genuine decision "
                           "moments, troughs are committed choices"),
        xaxis=dict(title="Generation step"),
        yaxis=dict(title="Entropy (bits)", rangemode="tozero"),
        height=460,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18, yanchor="top"),
        margin=dict(t=90, b=90),
    ))
    return save_fig(fig, output_path, formats)
