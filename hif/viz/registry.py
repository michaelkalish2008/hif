"""The signal-visualization registry.

Single source of truth for the viz engine: which signals exist, their identity
(id/label/kind/family), the generator that draws them, and the
availability predicate that decides whether *this* profile has the backing data.

Insertion order is stable and taxonomy-faithful:
  aggregates: stability, breadth, surprise, sensitivity, similarity
  readings:   entropy, wager

Charts for the measurements cut in hif-v4 were cut with them: a chart whose
measurement is gone recreates the "existed only as a chart" gap that hif-v3.1
was created to close.

FIDELITY CONTRACT: one visualization per signal, gated on data availability.
No signal is ever rendered from another signal's data, and no absent signal is
drawn as zero/flat.

Relation to the measurement registry (hif/profile/registry.py)
-------------------------------------------------------------
Signal ids are chart names, not measurement keys — the two namespaces are
deliberately different (a chart may draw a trace whose run-level summary is
the measurement). Ids are also filenames: generated charts are written under
them and consumers link to those paths, so an id is renamed only with a
corresponding migration of the artifacts. A `label` is display text and
carries no such cost.

Labels name the quantity drawn, in the terms it is computed in — the same
convention as `name` on a measurement row, and for the same reason
(SIGNAL_SET_VERSION history, hif-v3.3: a coined shorthand is free to drift
off the quantity while the key stays pinned to it). Three charts do not draw
a measurement directly and say so in their label: `stability` draws the
per-position input entropy trace behind the aggregate, `surprise` draws the
same per-position series `wager` summarises, and `breadth` draws effective
support size, which is not in the measurement set at all.

The bridge is explicit: `measurement_key` on each row names
the measurement the chart draws (None for a chart, like breadth, that draws a
component quantity which is deliberately NOT in the measurement set), and
`family` is copied from that measurement's `functional` so the two registries
share one vocabulary. `resolve_signal()` accepts either a signal id or a
measurement key, which is what lets `hif profile --metric <key> --charts`
render the chart that gives the printed number meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from hif.profile.registry import MEASUREMENT_BY_KEY
from hif.viz.signals import (
    breadth, entropy, sensitivity, similarity, stability, surprise, wager,
)


@dataclass(frozen=True)
class SignalViz:
    id: str
    label: str
    kind: str        # "aggregate" | "reading"
    family: str      # a FUNCTIONALS value: "information-theoretic" | "geometric"
    generate: Callable
    available: Callable  # (profile) -> str | None
    # The measurement this chart draws, as a MEASUREMENT_REGISTRY key — the
    # explicit bridge between chart names and measurement keys. None when the
    # chart draws a component quantity that is deliberately not a measurement.
    measurement_key: str | None = None


# Ordered as the v1 signal set. `module` supplies generate/available, and also
# LABEL — the display name is read off the module that draws the chart, not
# restated here. They used to be restated here, and the two
# copies drifted: this table said "Stability" while the module said "Input
# entropy trace", and the chart a reader actually saw was titled from the
# module.
#
# NOTE on `stability`: the chart draws the per-position input entropy TRACE,
# not the aggregate its measurement reduces it to (input_entropy_std_bits, the
# spread of per-variant entropy shifts). The aggregate is a single scalar with
# no informative direct chart; the trace is the series behind it.
_SPEC = [
    # id, kind, module, measurement_key
    ("stability",      "aggregate", stability,      "input_entropy_std_bits"),
    # Breadth draws per-step effective support size — deliberately NOT a
    # measurement (ESS is entropy in different units; docs/MEASUREMENTS.md
    # excludes it), so it maps to no key.
    ("breadth",        "aggregate", breadth,        None),
    # Surprise draws the same per-position excess-surprisal series as the
    # wager reading; wager is the designated chart for the measurement, so
    # only wager carries the key (one measurement must resolve to one chart).
    ("surprise",       "aggregate", surprise,       None),
    ("sensitivity",    "aggregate", sensitivity,    "perturbation_jsd_bits"),
    ("similarity",     "aggregate", similarity,     "io_cosine_similarity"),
    ("entropy",        "reading",   entropy,        "output_entropy_bits"),
    ("wager",          "reading",   wager,          "prompt_surprisal_excess_bits"),
]

SIGNALS: list[SignalViz] = [
    SignalViz(
        id=sid, label=mod.LABEL, kind=kind,
        # One vocabulary: the chart's family IS its measurement's functional.
        # Breadth (no measurement) draws support size off the output
        # distribution — information-theoretic like the entropy it re-scales.
        family=(
            MEASUREMENT_BY_KEY[mkey].functional
            if mkey is not None
            else "information-theoretic"
        ),
        generate=mod.generate, available=mod.available,
        measurement_key=mkey,
    )
    for (sid, kind, mod, mkey) in _SPEC
]

SIGNALS_BY_ID: dict[str, SignalViz] = {s.id: s for s in SIGNALS}

# measurement key -> the signal that draws it. Injective by construction
# (asserted in tests/unit/test_viz_measurement_bridge.py): each key appears on
# at most one row above.
SIGNALS_BY_MEASUREMENT: dict[str, SignalViz] = {
    s.measurement_key: s for s in SIGNALS if s.measurement_key is not None
}

# Measurements no chart draws directly, mapped to the nearest chart — the one
# that shows the series or the companion quantity a reader of that measurement
# most needs. Used only for the error message when a chart is requested for
# them, so the failure names where to look instead of "Unknown signal".
NEAREST_CHART: dict[str, str] = {
    # Mean of the same per-variant entropy-shift series whose spread the
    # stability chart's measurement summarises.
    "input_entropy_shift_bits": "stability",
}


def resolve_signal(name: str) -> SignalViz | None:
    """Resolve a signal id OR a measurement key to its chart, else None."""
    sig = SIGNALS_BY_ID.get(name)
    if sig is None:
        sig = SIGNALS_BY_MEASUREMENT.get(name)
    return sig


AGGREGATE_SIGNALS = [s.id for s in SIGNALS if s.kind == "aggregate"]
READING_SIGNALS = [s.id for s in SIGNALS if s.kind == "reading"]
