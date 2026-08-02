"""I/O Correlation (aggregate view) — does output uncertainty track prompt complexity?

Fidelity: Pearson r between the input-side entropy trace and the output-side
entropy trace, both interpolated to a common grid. The faithful chart overlays
the two normalized-position traces so the reader sees where they move together
or diverge; the reported r is the aggregate.

Backing data: ``input_side.positions[].entropy`` (teacher forcing) +
``metrics.distribution[].entropy_bits``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hif.viz.base import NEEDS_TEACHER_FORCING, na_figure, save_fig, signal_title
from hif.viz._theme import INDIGO, EMERALD, dark_layout
from hif.profile.schema import BehavioralRangeProfile

LABEL, GLYPH = "I/O Correlation", None


def available(profile: BehavioralRangeProfile) -> str | None:
    positions = getattr(profile.input_side, "positions", None) or []
    dist = getattr(profile.metrics, "distribution", None) or []
    if len(positions) == 0:
        return NEEDS_TEACHER_FORCING
    if len(dist) == 0:
        return "Requires per-step output distributions."
    return None


def _resample(y: list[float], n: int = 100) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return np.full(n, y[0] if len(y) else 0.0)
    xp = np.linspace(0.0, 1.0, len(y))
    return np.interp(np.linspace(0.0, 1.0, n), xp, y)


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, GLYPH, reason), output_path, formats)

    in_ent = [p.entropy for p in profile.input_side.positions]
    out_ent = [d.entropy_bits for d in profile.metrics.distribution]
    gx = np.linspace(0.0, 1.0, 100)
    in_r = _resample(in_ent)
    out_r = _resample(out_ent)
    if np.std(in_r) > 0 and np.std(out_r) > 0:
        r = float(np.corrcoef(in_r, out_r)[0, 1])
        r_txt = f"Pearson r = {r:.3f}"
    else:
        r_txt = "Pearson r n/a (degenerate trace)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gx, y=in_r, mode="lines", line=dict(color=INDIGO, width=2.4),
                             name="Input entropy (prompt)"))
    fig.add_trace(go.Scatter(x=gx, y=out_r, mode="lines", line=dict(color=EMERALD, width=2.4, dash="dot"),
                             name="Output entropy (generation)"))
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, GLYPH, profile.model.name,
                           f"Input vs output uncertainty over normalized position · {r_txt} · "
                           "tracking together = output complexity follows the prompt"),
        xaxis=dict(title="Normalized position (0 = start → 1 = end)"),
        yaxis=dict(title="Entropy (bits)", rangemode="tozero"),
        height=460,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18, yanchor="top"),
        margin=dict(t=90, b=90),
    ))
    return save_fig(fig, output_path, formats)
