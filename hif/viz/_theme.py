"""Shared dark-theme constants and helpers for all Plotly plot files.

Import this module and call ``dark_layout(**overrides)`` inside
``fig.update_layout(...)`` to apply the site design system colours.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Colour constants — site design system
# ---------------------------------------------------------------------------

PAPER_BG   = "#020617"   # slate-950 — outer background
PLOT_BG    = "#0f172a"   # slate-900 — chart area
SURFACE    = "#1e293b"   # slate-800 — dropdowns, table headers, annotation boxes
BORDER     = "#334155"   # slate-700
TEXT_PRI   = "#f1f5f9"   # slate-100 — primary text
TEXT_SEC   = "#94a3b8"   # slate-400 — secondary / tick labels
TEXT_MUTED = "#64748b"   # slate-500

GRID_COLOR     = "rgba(148,163,184,0.10)"
ZEROLINE_COLOR = "rgba(148,163,184,0.18)"

# Accent colours
INDIGO  = "#6366f1"
VIOLET  = "#8b5cf6"
AMBER   = "#f59e0b"
EMERALD = "#10b981"
RED     = "#ef4444"
GOLD    = "#F4A922"   # chosen-token highlight — keep unchanged

# Phenomenon palette — dark-calibrated
PHENOMENON_COLORS: dict[str, str] = {
    "convergence": "#60a5fa",   # blue-400
    "clustering":  "#a78bfa",   # violet-400
    "divergence":  "#fb923c",   # orange-400
    "diffusion":   "#94a3b8",   # slate-400
}


# ---------------------------------------------------------------------------
# Helper: single-axis dark defaults
# ---------------------------------------------------------------------------

def dark_axis(**overrides) -> dict:
    """Return kwargs for a single axis (pass to xaxis= / yaxis= in update_layout)."""
    base = dict(
        gridcolor=GRID_COLOR,
        zerolinecolor=ZEROLINE_COLOR,
        tickfont=dict(color=TEXT_SEC),
        title_font=dict(color=TEXT_SEC),
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Helper: full-layout dark defaults
# ---------------------------------------------------------------------------

def dark_layout(**overrides) -> dict:
    """Return a dict of update_layout kwargs that apply the full dark theme.

    Pass the result as ``**dark_layout(...)`` inside ``fig.update_layout()``,
    or merge it with your own kwargs.
    """
    base = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_PRI),
        xaxis=dark_axis(),
        yaxis=dark_axis(),
        legend=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=TEXT_PRI),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER,
            font=dict(color=TEXT_PRI),
        ),
    )
    # Shallow-merge overrides (caller can still override individual keys).
    base.update(overrides)
    return base
