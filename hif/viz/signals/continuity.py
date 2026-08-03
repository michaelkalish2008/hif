"""Continuity (aggregate view) — do independently sampled branches converge?

Fidelity: Continuity reads the trajectory rollout — B branches forked mid-
sequence, each rolled forward, clustered per step. The faithful chart is the
count of distinct semantic clusters across rollout steps (converging toward 1 =
the task has a preferred direction; staying high = sustained multi-directionality),
with the persistence / convergence / explosion scores summarized.

Backing data: ``trajectory.convergence_profile`` / ``trajectory.branches``.
Ported from the former ``plots/trajectory.py``.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hif.viz.base import NEEDS_TRAJECTORY, na_figure, save_fig, signal_title
from hif.viz._theme import (
    PLOT_BG, SURFACE, BORDER, TEXT_PRI, TEXT_SEC, TEXT_MUTED, INDIGO, dark_layout,
)
from hif.profile.schema import BehavioralRangeProfile

LABEL = "Branch pairwise cosine similarity"

GROUP_COLORS = ["#6366f1", "#22d3ee", "#f59e0b", "#10b981",
                "#f43f5e", "#a78bfa", "#fb923c", "#34d399"]


def available(profile: BehavioralRangeProfile) -> str | None:
    conv = getattr(profile.trajectory, "convergence_profile", None) or []
    return None if len(conv) > 0 else NEEDS_TRAJECTORY


def _hex_rgba(hex_color: str, alpha: float = 0.13) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _wrap(text: str, width: int = 45) -> str:
    words, lines, cur, ln = text.split(), [], [], 0
    for w in words:
        if ln + len(w) + 1 > width and cur:
            lines.append(" ".join(cur)); cur, ln = [w], len(w)
        else:
            cur.append(w); ln += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return "<br>".join(lines)


def generate(profile, output_path: Path, formats: list[str] = ["html"]) -> dict[str, Path]:
    reason = available(profile)
    if reason:
        return save_fig(na_figure(LABEL, reason), output_path, formats)

    traj = profile.trajectory
    convergence = traj.convergence_profile
    steps = [c.step for c in convergence]
    n_clusters = [c.n_remaining_clusters for c in convergence]
    initial = traj.initial_n_clusters
    model_name = profile.model.name
    branches = traj.branches
    has_table = len(branches) > 0

    # Plotly table traces render at their CONTENT height regardless of the
    # row_heights fraction they're given — a fixed split (e.g. 60/40) overflows
    # into whatever sits below once there are more than ~3 branches. Size the
    # table's row_heights fraction from its actual content height instead.
    CHART_PX, HEADER_PX, ROW_PX, PAD_PX = 420, 34, 46, 40
    table_px = HEADER_PX + len(branches) * ROW_PX + PAD_PX if has_table else 0
    total_px = CHART_PX + table_px if has_table else CHART_PX

    if has_table:
        fig = make_subplots(rows=2, cols=1, row_heights=[CHART_PX / total_px, table_px / total_px],
                            specs=[[{"type": "scatter"}], [{"type": "table"}]],
                            vertical_spacing=0.06)
    else:
        fig = make_subplots(rows=1, cols=1)

    unique_ids = sorted(set(b.cluster_id for b in branches)) if has_table else []
    id_to_color = {cid: GROUP_COLORS[i % len(GROUP_COLORS)] for i, cid in enumerate(unique_ids)}

    fig.add_trace(go.Scatter(
        x=steps, y=n_clusters, mode="lines+markers",
        name="Active semantic clusters",
        line=dict(color=INDIGO, width=2), marker=dict(size=8),
        hovertemplate="Step %{x}<br>Clusters remaining: %{y}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=initial, line_dash="dash", line_color=TEXT_MUTED,
                  annotation_text=f"Branches forked: {initial}",
                  annotation_position="top right", row=1, col=1)

    persist = traj.persistence_score
    conv = traj.convergence_score
    explode = traj.explosion_score

    if has_table:
        cluster_ids = [f"Group {b.cluster_id}" for b in branches]
        row_colors = [id_to_color.get(b.cluster_id, "#94a3b8") for b in branches]
        rep_tokens = [
            (", ".join(str(t) for t in b.representative_token_ids[:6]) or "(none)")
            for b in branches
        ]
        final_texts = [
            (_wrap(b.final_text[:120].replace("\n", " ")) or "<i>(no distinct continuation — "
             "this branch produced no output beyond the fork point)</i>")
            for b in branches
        ]
        fig.add_trace(go.Table(
            header=dict(values=["Semantic group",
                                "Anchor tokens<br><sub>shared vocabulary</sub>",
                                "Branch continuation<br><sub>what this fork generated</sub>"],
                        fill_color=SURFACE, font=dict(color=TEXT_PRI, size=11),
                        align="left", height=36),
            cells=dict(values=[cluster_ids, rep_tokens, final_texts],
                       fill_color=[[_hex_rgba(c) for c in row_colors], PLOT_BG, PLOT_BG],
                       font=dict(color=[row_colors, TEXT_SEC, TEXT_SEC], size=11),
                       line_color=BORDER, align="left", height=ROW_PX),
        ), row=2, col=1)

    n_steps_label = len(steps)
    # Score summary folded into the (auto-wrapping) subtitle rather than a
    # floating annotation below the chart — a fixed-position annotation
    # collided with the table once there were more than ~3 branches, since
    # Plotly tables render at content height regardless of their subplot domain.
    subtitle = (
        f"{initial} branches forked mid-sequence, {n_steps_label} rollout steps each · "
        "clusters converging toward 1 = branches produced similar outputs · "
        f"Persistence {persist:.2f} (stayed distinct) · Convergence {conv:.2f} (collapsed to 1) · "
        f"Explosion {explode:.2f} (split further)"
    )
    fig.update_layout(**dark_layout(
        title=signal_title(LABEL, model_name, subtitle),
        xaxis=dict(title="Rollout step"),
        yaxis=dict(title="Distinct semantic clusters", rangemode="tozero"),
        height=(total_px + 170 if has_table else CHART_PX + 130), showlegend=True,
        margin=dict(t=150, b=50),
    ))
    return save_fig(fig, output_path, formats, png_size=(900, 640))
