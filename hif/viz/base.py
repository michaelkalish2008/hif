"""Shared plumbing for the signal-faithful visualization engine.

Every signal generator is a thin function ``generate(profile, output_path,
formats) -> dict[str, Path]``.  One gate decides what it renders:

- **availability** — whether the signal's backing data exists in *this* profile.
  A signal whose data is absent (no teacher forcing, no attention capture, no
  perturbations…) renders a branded placeholder naming the requirement, never a
  fabricated or mislabeled chart.  This is the fidelity contract.

Helpers here keep each per-signal module small and consistent.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.offline as offline

from hif.viz._theme import (
    BORDER, SURFACE, TEXT_PRI, TEXT_SEC, TEXT_MUTED, dark_layout,
)

# Standard reasons a signal is unavailable for a given run. Kept as constants so
# every generator names the requirement identically.
NEEDS_TEACHER_FORCING = (
    "Requires teacher forcing — open-weight models only (this backend does not "
    "expose it)."
)
NEEDS_ATTENTION = (
    "Requires attention capture — open HuggingFace models only (GPT-2, Gemma). "
    "Not available for API or Ollama models."
)
NEEDS_PERTURBATION = "Requires perturbation variants (run with paraphrase generators enabled)."
NEEDS_TRAJECTORY = "Requires trajectory branching (no branch rollout was recorded for this run)."
NEEDS_DISTRIBUTION = "Requires per-step output distributions (no generation steps recorded)."
NEEDS_EXPOSURE = "Requires the Exposure analysis extension (top-K probabilities + an embedding encoder)."


def na_figure(label: str, glyph: str | None, reason: str) -> go.Figure:
    """A branded 'not available for this run' placeholder figure.

    Deliberately explicit: the reader must never mistake an absent signal for a
    flat or zero one. Names the signal and the exact data requirement.
    """
    head = f"{glyph}  {label}" if glyph else label
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.58, xref="paper", yref="paper",
        text=f"<b>{head}</b>",
        showarrow=False, font=dict(size=16, color=TEXT_SEC), align="center",
    )
    fig.add_annotation(
        x=0.5, y=0.44, xref="paper", yref="paper",
        text=f"<i>Not available for this run</i><br><br>{reason}",
        showarrow=False, font=dict(size=12.5, color=TEXT_MUTED), align="center",
    )
    fig.update_layout(**dark_layout(
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=40, b=40, l=40, r=40),
    ))
    return fig


def save_fig(
    fig: go.Figure,
    output_path: Path,
    formats: list[str],
    png_size: tuple[int, int] = (1000, 600),
) -> dict[str, Path]:
    """Write ``fig`` as standalone HTML (+ optional PNG) and return the paths."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = output_path.with_suffix(".html")
    # responsive=False: offline.plot() defaults to responsive:true, which makes
    # Plotly resize the chart to fit whatever container it's dropped into (the
    # generated div is `height:100%`) — silently overriding the explicit
    # layout.height every chart here sets deliberately (e.g. continuity.py sizes
    # its height to the branch table's actual row count). Without this, a chart
    # taller than the embedding viewport/iframe gets squeezed rather than
    # scrolling, corrupting the layout math instead of respecting it.
    offline.plot(fig, filename=str(html_path), auto_open=False, include_plotlyjs="cdn",
                config={"responsive": False})
    result: dict[str, Path] = {"html": html_path}
    if "png" in formats:
        png_path = output_path.with_suffix(".png")
        fig.write_image(str(png_path), width=png_size[0], height=png_size[1])
        result["png"] = png_path
    return result


def _wrap_words(text: str, max_line_chars: int) -> list[str]:
    """Hard-wrap a single long run of text by word boundary onto multiple lines."""
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = f"{current} {w}".strip()
        if current and len(candidate) > max_line_chars:
            lines.append(current)
            current = w
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _wrap_subtitle(subtitle: str, max_line_chars: int = 68) -> str:
    """Wrap a ' · '-joined subtitle onto multiple <br> lines.

    Plotly titles never auto-wrap — a single long line just overflows the
    iframe/container and gets visually clipped. Subtitles here are
    conventionally a run of "clause · clause · clause" fragments, so wrap by
    packing clauses onto lines up to ``max_line_chars``. A single clause that is
    ITSELF longer than one line (e.g. a caveat sentence) is further hard-wrapped
    by word boundary — packing clauses alone isn't enough if one clause exceeds
    the line budget on its own.
    """
    clauses = [c.strip() for c in subtitle.split(" · ") if c.strip()]
    if len(clauses) <= 1:
        wrapped = _wrap_words(subtitle, max_line_chars)
        return "<br>".join(wrapped)
    lines: list[str] = []
    current = clauses[0]
    for clause in clauses[1:]:
        candidate = f"{current} · {clause}"
        if len(candidate) > max_line_chars:
            lines.append(current)
            current = clause
        else:
            current = candidate
    lines.append(current)
    # Second pass: any packed line still over budget (one oversized clause) gets
    # hard-wrapped by word.
    final_lines: list[str] = []
    for line in lines:
        if len(line) > max_line_chars:
            final_lines.extend(_wrap_words(line, max_line_chars))
        else:
            final_lines.append(line)
    return "<br>".join(final_lines)


def signal_title(label: str, glyph: str | None, model_name: str, subtitle: str) -> dict:
    """Consistent chart title: `<glyph> Label — model` with a wrapped <sub> line."""
    head = f"{glyph}  {label}" if glyph else label
    return dict(
        text=f"{head} — {model_name}<br><sub>{_wrap_subtitle(subtitle)}</sub>",
        font=dict(size=14, color=TEXT_PRI),
    )


def add_click_to_dim_js(html_path: Path) -> None:
    """Post-process a saved chart HTML: clicking a bar/point dims all others.

    Lets a reader isolate one token/step without the tooltip being the only way
    to tell which bar is which. Idempotent — safe to call more than once.
    Mirrors the existing `_patch_horizon_html` post-processing pattern.
    """
    content = html_path.read_text()
    if "__clickToDim" in content:
        return
    import re as _re

    m = _re.search(r'Plotly\.newPlot\(\s*"([^"]+)"', content)
    if not m:
        return
    gid = m.group(1)
    script = f"""
<script>
(function() {{
    var __clickToDim = true;
    var gd = document.getElementById("{gid}");
    if (!gd) return;
    gd.on('plotly_click', function(evt) {{
        if (!evt || !evt.points || !evt.points.length) return;
        var pt = evt.points[0];
        var traceIdx = pt.curveNumber;
        var trace = gd.data[traceIdx];
        var n = (trace.x || trace.y || []).length;
        var already = trace.__dimmedAt === pt.pointNumber;
        var opacities = [];
        for (var i = 0; i < n; i++) {{
            opacities.push(already ? 0.88 : (i === pt.pointNumber ? 0.95 : 0.15));
        }}
        Plotly.restyle(gd, {{'marker.opacity': [opacities]}}, [traceIdx]);
        trace.__dimmedAt = already ? null : pt.pointNumber;
    }});
}})();
</script>
"""
    content = content.replace("</body>", script + "</body>") if "</body>" in content else content + script
    html_path.write_text(content)
