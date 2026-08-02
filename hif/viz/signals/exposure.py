"""Exposure ◇ (per-step view) — counterfactual semantic exposure.

Fidelity: at each generation step the top-K candidates form a meaning-cloud.
Exposure measures how often a *probabilistically accessible* alternative token
was semantically distant from the selected token in a *diffuse* candidate cloud —
i.e. how sensitive the response's meaning was to sampling chance. The scalar is
the fraction of such steps.

This is EXPLICITLY NOT a factuality judgment. It does not see the convergence
case (a confident, narrow model aimed wrong); a confident response can still be
wrong and this reading will not show it. Copy here must never imply "danger",
"hallucination", or a correctness claim — only counterfactual semantic exposure.

Backing data: ``profile.exposure`` (runtime type ExposureProfile) — top-K
probabilities + an embedding encoder.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hif.viz.base import NEEDS_EXPOSURE, na_figure, save_fig, signal_title
from hif.viz._theme import INDIGO, AMBER, EMERALD, TEXT_SEC, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL, GLYPH = "Exposure", "◇"

# Neutral divergence scale (NOT a risk scale): low → high semantic divergence.
_LOW = EMERALD
_MID = INDIGO
_HIGH = AMBER


def _coerce(profile):
    """profile.exposure is typed Any: an ExposureProfile when built in-memory,
    but a plain dict when the profile was loaded from JSON. Coerce to the typed
    model so both paths work; return None if absent/malformed. (Validation
    aliases on ExposureProfile accept pre-rename JSON keys.)"""
    exp = getattr(profile, "exposure", None)
    if exp is None:
        return None
    if isinstance(exp, dict):
        try:
            from hif.analysis.exposure import ExposureProfile
            return ExposureProfile.model_validate(exp)
        except Exception:  # noqa: BLE001
            return None
    return exp


def available(profile: BehavioralRangeProfile) -> str | None:
    exp = _coerce(profile)
    if exp is None or not getattr(exp, "candidates", None):
        return NEEDS_EXPOSURE
    return None


def _divergence_color(distance: float) -> str:
    if distance >= 0.4:
        return _HIGH
    if distance >= 0.25:
        return _MID
    return _LOW


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, GLYPH, reason), output_path, formats)

    exp = _coerce(profile)
    model_name = profile.model.name
    candidates = exp.candidates
    exposed_steps = set(exp.exposed_steps)  # high-exposure (diffusion + distant) steps

    x_labels = [f"{c.step}: {c.selected_token!r}" for c in candidates]
    distances = [c.semantic_distance for c in candidates]
    alt_probs = [c.divergent_prob for c in candidates]
    prob_gaps = [c.selected_prob - c.divergent_prob for c in candidates]
    sizes = [max(10, int(p * 70)) for p in alt_probs]

    hover = [
        f"<b>Step {c.step} — chose {c.selected_token!r}</b><br>"
        f"Most divergent accessible alternative: <b>{c.divergent_token!r}</b><br>"
        f"  alternative probability: {c.divergent_prob:.3f} (rank #{c.prob_rank} in top-K)<br>"
        f"  semantic distance: {c.semantic_distance:.3f} "
        f"({'high' if c.semantic_distance >= 0.4 else 'moderate' if c.semantic_distance >= 0.25 else 'low'} divergence)<br>"
        f"  candidate cloud: {c.cloud_phenomenon}<br>"
        f"chosen-token probability: {c.selected_prob:.3f} · gap {c.selected_prob - c.divergent_prob:.3f}"
        + ("<br><b>high-exposure step</b>" if c.step in exposed_steps else "")
        for c in candidates
    ]

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.58, 0.42], vertical_spacing=0.22,
        subplot_titles=[
            "How far could an accessible alternative have shifted the meaning?",
            "How much more probable was the chosen token than that alternative?",
        ],
    )

    # Panel 1 — semantic distance, one trace per divergence band (neutral, not risk).
    bands = [
        ("Low divergence", _LOW, lambda d: d < 0.25),
        ("Moderate divergence", _MID, lambda d: 0.25 <= d < 0.4),
        ("High divergence", _HIGH, lambda d: d >= 0.4),
    ]
    for name, color, pred in bands:
        idx = [i for i, d in enumerate(distances) if pred(d)]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x=[x_labels[i] for i in idx], y=[distances[i] for i in idx], mode="markers",
            name=name, marker=dict(color=color, size=[sizes[i] for i in idx], opacity=0.85,
                                   line=dict(color="white", width=1)),
            hovertext=[hover[i] for i in idx], hoverinfo="text",
        ), row=1, col=1)

    # Panel 2 — probability gap (smaller gap = meaning was more exposed to sampling chance).
    fig.add_trace(go.Bar(
        x=x_labels, y=prob_gaps, marker_color=INDIGO, opacity=0.8,
        hovertext=hover, hoverinfo="text", showlegend=False,
    ), row=2, col=1)

    exposure_scalar = getattr(exp, "exposure", 0.0)
    mean_dist = exp.mean_semantic_distance
    n_exposed = len(exp.exposed_steps)
    diff_pct = int(exp.diffusion_zone_ratio * 100)

    fig.update_layout(**dark_layout(
        title=signal_title(
            LABEL, GLYPH, model_name,
            f"Counterfactual semantic exposure = {exposure_scalar:.3f} · mean distance {mean_dist:.3f} · "
            f"{n_exposed} high-exposure step{'s' if n_exposed != 1 else ''} · {diff_pct}% steps diffuse · "
            "measures sensitivity to sampling chance, NOT factuality"),
        xaxis=dict(categoryorder="array", categoryarray=x_labels, tickangle=-60,
                   tickfont=dict(size=9, color=TEXT_SEC), automargin=True),
        xaxis2=dict(categoryorder="array", categoryarray=x_labels, tickangle=-60,
                    tickfont=dict(size=9, color=TEXT_SEC), automargin=True,
                    title=dict(text="Generation step (token chosen)")),
        yaxis=dict(title="Semantic distance", rangemode="tozero"),
        yaxis2=dict(title="Probability gap", rangemode="tozero"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14, yanchor="top"),
        height=800, margin=dict(t=120, b=120, l=80, r=20),
    ))
    return save_fig(fig, output_path, formats, png_size=(1000, 680))
