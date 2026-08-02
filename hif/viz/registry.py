"""The signal-visualization registry.

Single source of truth for the viz engine: which signals exist, their identity
(id/label/kind/glyph/family), the generator that draws them, and the
availability predicate that decides whether *this* profile has the backing data.

Insertion order is the v1 ordered set, so the engine renders signals in a
stable, taxonomy-faithful order:
  aggregates: stability, breadth, surprise, io_correlation, sensitivity,
              continuity, similarity
  readings:   entropy, shift, wager, spread, horizon, exposure

FIDELITY CONTRACT: one visualization per signal, gated on data availability.
No signal is ever rendered from another signal's data, and no absent signal is
drawn as zero/flat.

Relation to the measurement registry (hif/profile/signals.py)
-------------------------------------------------------------
Signal ids are chart names, not measurement keys — the two namespaces are
deliberately different (a chart may draw a trace whose run-level summary is
the measurement). The bridge is explicit: `measurement_key` on each row names
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

from hif.profile.signals import MEASUREMENT_BY_KEY
from hif.viz.signals import (
    breadth, continuity, entropy, exposure, horizon, io_correlation,
    sensitivity, shift, similarity, spread, stability, surprise, wager,
)


@dataclass(frozen=True)
class SignalViz:
    id: str
    label: str
    kind: str        # "aggregate" | "reading"
    family: str      # a FUNCTIONALS value: "information-theoretic" | "geometric"
    glyph: str | None
    generate: Callable
    available: Callable  # (profile) -> str | None
    # The measurement this chart draws, as a MEASUREMENT_REGISTRY key — the
    # explicit bridge between chart names and measurement keys. None when the
    # chart draws a component quantity that is deliberately not a measurement.
    measurement_key: str | None = None


# Ordered as the v1 signal set. `module` supplies generate/available.
# The last column is the measurement key the chart draws (or None).
_SPEC = [
    # id, label, kind, module, measurement_key
    # NOTE: the `stability` chart draws the per-position input entropy TRACE,
    # not the aggregate the measurement labelled "Stability" reduces it to
    # (input_entropy_std_bits, the spread of per-variant entropy shifts). The
    # aggregate is a single scalar with no informative direct chart; the trace
    # is the series behind it, which is why docs/ARCHITECTURE.md describes it
    # as "Stability (rendered as the input entropy trace)".
    ("stability",      "Stability",       "aggregate", stability,      "input_entropy_std_bits"),
    # Breadth draws per-step effective support size — deliberately NOT a
    # measurement (ESS is entropy in different units; docs/MEASUREMENTS.md
    # excludes it), so it maps to no key.
    ("breadth",        "Breadth",         "aggregate", breadth,        None),
    # Surprise draws the same per-position excess-surprisal series as the
    # Wager reading; wager is the designated chart for the measurement, so
    # only wager carries the key (one measurement must resolve to one chart).
    ("surprise",       "Surprise",        "aggregate", surprise,       None),
    ("io_correlation", "I/O Correlation", "aggregate", io_correlation, "io_correlation_r"),
    ("sensitivity",    "Sensitivity",     "aggregate", sensitivity,    "perturbation_jsd_bits"),
    ("continuity",     "Continuity",      "aggregate", continuity,     "branch_pairwise_cosine_similarity"),
    ("similarity",     "Similarity",      "aggregate", similarity,     "io_cosine_similarity"),
    ("entropy",        "Entropy",         "reading",   entropy,        "output_entropy_bits"),
    ("shift",          "Shift",           "reading",   shift,          "output_step_jsd_bits"),
    ("wager",          "Wager",           "reading",   wager,          "prompt_surprisal_excess_bits"),
    ("spread",         "Spread",          "reading",   spread,         "attention_entropy_output_bits"),
    ("horizon",        "Horizon",         "reading",   horizon,        "attention_entropy_input_bits"),
    ("exposure",       "Exposure",        "reading",   exposure,       "counterfactual_exposure_fraction"),
]

SIGNALS: list[SignalViz] = [
    SignalViz(
        id=sid, label=label, kind=kind,
        # One vocabulary: the chart's family IS its measurement's functional.
        # Breadth (no measurement) draws support size off the output
        # distribution — information-theoretic like the entropy it re-scales.
        family=(
            MEASUREMENT_BY_KEY[mkey].functional
            if mkey is not None
            else "information-theoretic"
        ),
        glyph=getattr(mod, "GLYPH", None),
        generate=mod.generate, available=mod.available,
        measurement_key=mkey,
    )
    for (sid, label, kind, mod, mkey) in _SPEC
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
    # The per-step entropy trace the deltas are differences of.
    "output_entropy_step_delta_bits": "entropy",
    # Companion measurement: the same consecutive-step transitions, and the
    # resolution limit the Shift chart must be read with.
    "output_step_topk_overlap_fraction": "shift",
    # The candidate cloud's width per step; the cluster-mass entropy has no
    # chart of its own.
    "candidate_cluster_entropy_bits": "breadth",
    # Veer is the geometric twin of Shift (same transitions, embedding space
    # instead of vocabulary space); no chart draws Veer itself.
    "semantic_centroid_veer_cosine": "shift",
}


def resolve_signal(name: str) -> SignalViz | None:
    """Resolve a signal id OR a measurement key to its chart, else None."""
    sig = SIGNALS_BY_ID.get(name)
    if sig is None:
        sig = SIGNALS_BY_MEASUREMENT.get(name)
    return sig


AGGREGATE_SIGNALS = [s.id for s in SIGNALS if s.kind == "aggregate"]
READING_SIGNALS = [s.id for s in SIGNALS if s.kind == "reading"]
