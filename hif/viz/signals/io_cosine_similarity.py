"""Similarity (aggregate view) — how anchored outputs stay to their inputs.

Fidelity: Similarity is a small family of cosine measures — input_sim (spread of
inputs), output_sim (spread of outputs), io_sim (mean cosine of each input to
its paired output), plus a trend (slope of per-step similarity). The faithful
chart shows the three anchoring cosines as bars with the trend called out.

Backing data: ``metrics.similarity`` — requires perturbation variant pairs.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from hif.viz.base import na_figure, save_fig, signal_title
from hif.viz._theme import INDIGO, EMERALD, AMBER, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL = "Input/output cosine similarity"
_NEEDS = "Requires perturbation variant pairs (no (input, output) pairs were recorded)."
_NO_OUTPUT = "The target generated no output on this run — there is no output side to anchor."


def available(profile: BehavioralRangeProfile) -> str | None:
    # Two separate refusals. The similarity stage may not have run at all, and
    # it may have run over a pair set whose baseline output is the empty
    # string — in which case io_sim is a mean over the perturbation variants'
    # continuations, and drawing it under this model's name would show the
    # paraphrases as the model's own anchoring. `measurements()` withholds the
    # key for the same reason (needs_generated_output), and this gate has to
    # match it or the chart becomes evidence for a withheld claim.
    if not (getattr(profile.output_side, "steps", None) or []):
        return _NO_OUTPUT
    return None if getattr(profile.metrics, "similarity", None) is not None else _NEEDS


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, reason), output_path, formats)

    sim = profile.metrics.similarity
    labels = ["Input spread<br>(input_sim)", "Output spread<br>(output_sim)", "Input→Output anchor<br>(io_sim)"]
    values = [sim.input_sim, sim.output_sim, sim.io_sim]
    colors = [INDIGO, EMERALD, AMBER]
    if sim.trend is None:
        trend_note = "per-step trend absent (fewer than two steps)"
    else:
        trend_word = "converging" if sim.trend >= 0 else "diverging"
        trend_note = f"per-step trend {sim.trend:+.4f} ({trend_word})"

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors, opacity=0.88,
        text=[f"{v:.3f}" for v in values], textposition="outside",
        hovertemplate="%{x}<br>cosine: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, profile.model.name,
                           f"Semantic anchoring (cosine) · io_ratio {sim.io_ratio:.2f} · "
                           f"{trend_note} · n={sim.n_pairs} pairs"),
        xaxis=dict(title=""),
        yaxis=dict(title="Cosine similarity", range=[0, 1.05]),
        height=460, showlegend=False, margin=dict(t=90, b=70),
    ))
    return save_fig(fig, output_path, formats)
