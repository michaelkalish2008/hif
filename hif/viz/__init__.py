"""Signal-faithful visualization engine.

One visualization per HIF signal, driven by the registry. Rendering is gated
on **availability** only: a signal renders real data when its backing data
exists in the profile; otherwise a labeled "requires teacher forcing /
attention capture / …" placeholder. Fidelity: never fabricate or mislabel a
signal.

Entry point: ``generate_signal_plots(profile, output_dir)``.
"""

from __future__ import annotations

from pathlib import Path

from hif.viz.index import build_index
from hif.viz.registry import (
    AGGREGATE_SIGNALS, NEAREST_CHART, READING_SIGNALS, SIGNALS, SIGNALS_BY_ID,
    resolve_signal,
)
from hif.profile.schema import BehavioralRangeProfile

__all__ = [
    "generate_signal_plots",
    "SIGNALS",
    "SIGNALS_BY_ID",
    "AGGREGATE_SIGNALS",
    "READING_SIGNALS",
]


def generate_signal_plots(
    profile: BehavioralRangeProfile,
    output_dir: Path,
    formats: list[str] = ["html"],
    only_signal: str | None = None,
) -> dict[str, dict[str, Path]]:
    """Render signal visualizations.

    - ``only_signal`` renders exactly that one chart and nothing else (no
      dashboard) — for a single ``--metric`` request. It accepts a signal id
      (``"effective_support_size"``) or a measurement key
      (``"output_entropy_bits"``): the
      CLI passes measurement keys, and the registry's ``measurement_key``
      bridge maps each to the chart that draws it. A measurement no chart
      draws raises with the nearest chart named — never "Unknown signal".
    - Otherwise renders one chart per signal + a combined index.

    Returns ``{requested_name: {"html": path, ...}}`` — keyed by the string
    the caller passed, whichever namespace it came from — plus an ``"index"``
    entry in the multi-signal case. Signals whose data is absent still render
    as labeled placeholders so the dashboard is complete and honest about what
    this backend exposes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single-signal mode: one chart, no dashboard — the user explicitly asked
    # for this metric, so we render exactly it.
    if only_signal is not None:
        sig = resolve_signal(only_signal)
        if sig is None:
            nearest = NEAREST_CHART.get(only_signal)
            if nearest is not None:
                near = SIGNALS_BY_ID[nearest]
                raise ValueError(
                    f"No chart draws the measurement {only_signal!r} directly. "
                    f"The nearest chart is {near.id!r} ({near.label} — it shows "
                    f"the series or companion quantity behind it); drop "
                    f"--charts to print the number alone, or use --charts "
                    f"without --metric for the full dashboard."
                )
            raise ValueError(f"Unknown signal {only_signal!r}")
        return {only_signal: sig.generate(profile, output_dir / sig.id, formats=formats)}

    results: dict[str, dict[str, Path]] = {}
    availability: dict[str, str | None] = {}

    for sig in SIGNALS:
        availability[sig.id] = sig.available(profile)
        results[sig.id] = sig.generate(profile, output_dir / sig.id, formats=formats)

    index_path = build_index(profile, output_dir, SIGNALS, availability)
    results["index"] = {"html": index_path}
    return results
