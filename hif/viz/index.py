"""Combined signal dashboard — one shareable page indexing every signal chart.

Grouped by kind (aggregate metrics / instrument readings). Entitled signals embed
their chart. Every chart is always rendered — there is no gating.
Each card shows an availability badge so the reader knows whether the chart is
live data or a "requires teacher forcing / attention capture" placeholder.

Footer states the boundary the strategy rests on (Grafana-style): a single
profile snapshot is not continuous drift monitoring.
"""

from __future__ import annotations

from pathlib import Path

from hif.viz._theme import (
    PAPER_BG, PLOT_BG, SURFACE, BORDER, TEXT_PRI, TEXT_SEC, TEXT_MUTED,
    INDIGO, AMBER, EMERALD,
)
from hif.viz.registry import SIGNALS
from hif.profile.schema import BehavioralRangeProfile

_CARD_HEIGHT = {"continuity": 700, "exposure": 840}  # taller charts; default below
_DEFAULT_HEIGHT = 500


def _height(sig_id: str) -> int:
    return _CARD_HEIGHT.get(sig_id, _DEFAULT_HEIGHT)


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def _card_embed(sig, avail_reason: str | None) -> str:
    glyph = f"{sig.glyph} " if sig.glyph else ""
    if avail_reason is None:
        avail_badge = _badge("live data", "live")
    else:
        avail_badge = _badge("not available this run", "na")
    return f"""
    <section class="card">
      <div class="card-head">
        <div class="badges">{avail_badge}
          <span class="kind">{sig.kind}</span></div>
        <h3>{glyph}{sig.label}</h3>
      </div>
      <iframe src="{sig.id}.html" height="{_height(sig.id)}" loading="lazy"
              title="{sig.label}"></iframe>
    </section>"""


def _section(title: str, subtitle: str, cards: list[str]) -> str:
    if not cards:
        return ""
    return f"""
    <div class="section">
      <div class="section-head"><h2>{title}</h2><p>{subtitle}</p></div>
      {''.join(cards)}
    </div>"""


def build_index(
    profile: BehavioralRangeProfile,
    output_dir: Path,
    signals: list,
    availability: dict[str, str | None],
) -> Path:
    """Write ``index.html`` — the combined, grouped signal dashboard."""
    output_dir = Path(output_dir)
    model_name = profile.model.name
    included = {s.id for s in signals}

    agg_cards: list[str] = []
    read_cards: list[str] = []
    for sig in SIGNALS:
        if sig.id not in included:
            continue
        card = _card_embed(sig, availability.get(sig.id))
        (agg_cards if sig.kind == "aggregate" else read_cards).append(card)

    sections = (
        _section("Aggregate views",
                 "How the run's perturbation-response and anchoring measurements look.",
                 agg_cards)
        + _section("Per-step views",
                   "The per-step traces behind the aggregates.", read_cards)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>HIF — Signal Profile — {model_name}</title>
  <style>
    :root {{
      --paper:{PAPER_BG}; --plot:{PLOT_BG}; --surface:{SURFACE}; --border:{BORDER};
      --text:{TEXT_PRI}; --text-sec:{TEXT_SEC}; --muted:{TEXT_MUTED};
      --indigo:{INDIGO}; --amber:{AMBER}; --emerald:{EMERALD};
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--text);
      font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
    header {{ max-width:1120px; margin:0 auto; padding:28px 24px 18px; border-bottom:1px solid var(--border); }}
    .brand {{ font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--text-sec); font-weight:600; }}
    header h1 {{ margin:8px 0 6px; font-size:26px; }}
    header .sub {{ color:var(--text-sec); font-size:14px; margin:0; }}
    .note-pill {{ display:inline-block; margin-top:12px; font-size:12px; font-weight:600;
      padding:3px 10px; border-radius:999px; border:1px solid var(--border); color:var(--text-sec); }}
    main {{ max-width:1120px; margin:0 auto; padding:8px 24px 32px; }}
    .section-head {{ margin:34px 0 6px; }}
    .section-head h2 {{ margin:0; font-size:20px; }}
    .section-head p {{ margin:2px 0 0; color:var(--text-sec); font-size:13px; }}
    .card {{ margin-top:20px; background:var(--plot); border:1px solid var(--border);
      border-radius:12px; overflow:hidden; }}
    .card-head {{ padding:16px 18px 8px; }}
    .card-head h3 {{ margin:8px 0 0; font-size:17px; }}
    .badges {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .kind {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }}
    .badge {{ font-size:10.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
      padding:2px 8px; border-radius:6px; }}
    .badge-live {{ background:rgba(99,102,241,.14); color:var(--indigo); border:1px solid rgba(99,102,241,.35); }}
    .badge-na {{ background:rgba(148,163,184,.12); color:var(--muted); border:1px solid var(--border); }}
    iframe {{ display:block; width:100%; border:0; background:var(--plot); border-top:1px solid var(--border); }}
    footer {{ max-width:1120px; margin:0 auto; padding:28px 24px 48px; border-top:1px solid var(--border);
      color:var(--muted); font-size:12.5px; }}
    footer strong {{ color:var(--text-sec); }}
    footer .cta {{ color:var(--amber); text-decoration:none; font-weight:600; }}
  </style>
</head>
<body>
  <header>
    <div class="brand">HIF</div>
    <h1>Signal Profile</h1>
    <p class="sub">{model_name} &middot; one generation run, measured in natural units</p>
    <span class="note-pill">descriptive only &mdash; no thresholds, levels, or verdicts</span>
  </header>
  <main>{sections}</main>
  <footer>
    <p><strong>This is a single behavioral snapshot</strong> — one model, one prompt, one moment.
       It describes what the model did. It does <strong>not</strong> detect drift, attacks, or
       quality, and it makes no claim that any value here is normal or abnormal.</p>
    <p>Where a chart draws a reported measurement, the registry names it
       (<code>measurement_key</code>); some charts instead draw a component
       series that is deliberately not a measurement. Charts marked
       <em>not available this run</em> need data this backend does not expose
       (teacher forcing, attention capture, or perturbation variants).</p>
  </footer>
</body>
</html>"""

    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path
