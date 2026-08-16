"""The signal-visualization registry.

Single source of truth for the viz engine: which signals exist, their identity
(id/label/kind/family), the generator that draws them, and the
availability predicate that decides whether *this* profile has the backing data.

Insertion order is stable and taxonomy-faithful:
  aggregates: input_entropy_trace, nucleus_effective_support_size,
              prompt_surprisal_excess_trace, perturbation_jsd_bits,
              io_cosine_similarity
  readings:   output_entropy_bits, prompt_surprisal_excess_bits

Charts for the measurements cut in hif-v4 were cut with them: a chart whose
measurement is gone recreates the "existed only as a chart" gap that hif-v3.1
was created to close.

FIDELITY CONTRACT: one visualization per signal, gated on data availability.
No signal is ever rendered from another signal's data, and no absent signal is
drawn as zero/flat.

Relation to the measurement registry (hif/profile/registry.py)
-------------------------------------------------------------
A chart that draws a measurement carries that measurement's key as its id.
The two namespaces used to be deliberately different, and the difference is
what let the retired shorthand survive here: `stability`, `breadth`,
`surprise`, `sensitivity`, `similarity`, `entropy`, `wager` were removed from
the measurement rows in `6a0dd45` — "Stability" on `input_entropy_std_bits`
inverts the reading of its own number — and this registry kept them for two
more versions, so `--charts` went on writing `stability.html`. One namespace
now, and no shorthand in it.

Ids are also filenames: generated charts are written under them and consumers
link to those paths, so an id is renamed only with a corresponding migration
of the artifacts (the site's sample charts and their generated catalog moved
with this one). A `label` is display text and carries no such cost.

Labels name the quantity drawn, in the terms it is computed in — the same
convention as `name` on a measurement row, and for the same reason. Three
charts do not draw a measurement directly, and their ids say so by not being
registry keys: `input_entropy_trace` draws the per-position series behind the
aggregate, `prompt_surprisal_excess_trace` draws the per-position series
`prompt_surprisal_excess_bits` summarises, and
`nucleus_effective_support_size` draws a quantity that is not in the
measurement set at all — the id names its basis because the basis decides the
ceiling (nucleus size, not vocabulary).

The bridge is explicit: `measurement_key` on each row names
the measurement the chart draws (None for a chart, like
`nucleus_effective_support_size`, that draws a component quantity which is
deliberately NOT in the measurement set), and
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
    input_entropy_trace,
    io_cosine_similarity,
    nucleus_effective_support_size,
    output_entropy_bits,
    perturbation_jsd_bits,
    prompt_surprisal_excess_bits,
    prompt_surprisal_excess_trace,
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
    # Does this chart plot the measurement's OWN series, or a related one it
    # merely resolves for? True (the common case) means the chart's data is
    # the basis the measurement reduces — so the chart must decline exactly
    # the runs the measurement declines, enforced in `_gated` below. False
    # means the key is carried for `resolve_signal` only, and the chart is
    # gated on its own data like an unkeyed chart. Only `input_entropy_trace`
    # is False, and its name says so: a chart that draws its measurement is
    # named FOR that measurement, so an id that is not a registry key is the
    # visible sign that this one draws the series instead.
    draws_measurement: bool = True


# Ordered as the v1 signal set. `module` supplies generate/available, and also
# LABEL — the display name is read off the module that draws the chart, not
# restated here. They used to be restated here, and the two
# copies drifted: this table said "Stability" while the module said "Input
# entropy trace", and the chart a reader actually saw was titled from the
# module.
#
# Chart ids are measurement keys, and the coined vocabulary that used to
# supply them — stability, breadth, surprise, sensitivity, similarity,
# entropy, wager — is gone. `6a0dd45` removed exactly those words from the
# measurement rows because a second naming layer drifts off the quantity while
# the key stays pinned to it, and it named the worst case: "Stability" sat on
# `input_entropy_std_bits`, a standard deviation, where a HIGHER number means
# LESS stable. That purge cleaned the measurement registry and left this one
# untouched, so `--charts` went on writing `stability.html` for two more
# versions. A chart that draws a measurement is now named for it; a chart that
# draws a series that is deliberately not a measurement is named for the
# series.
_SPEC = [
    # id, kind, module, measurement_key, draws_measurement
    #
    # input_entropy_trace is the one False: it plots the per-position input entropy of
    # the ORIGINAL prompt, while `input_entropy_std_bits` is the spread of
    # per-VARIANT entropy shifts. Real data either way, but not the same
    # series, so the measurement's absence is not this chart's absence. The
    # key is carried so `--metric input_entropy_std_bits --charts` resolves.
    ("input_entropy_trace", "aggregate", input_entropy_trace,
     "input_entropy_std_bits", False),
    # Draws per-step nucleus effective support size — deliberately NOT a
    # measurement (ESS is a bijection of entropy and fails the gate's
    # distinct-disclosure condition), so it maps to no key. The id names the
    # 0.95 nucleus basis rather than the backend's top-K, which is a different
    # count with a different ceiling.
    ("nucleus_effective_support_size", "aggregate", nucleus_effective_support_size, None, False),
    # The same per-position excess-surprisal series that
    # prompt_surprisal_excess_bits reduces. That chart is the designated one
    # for the measurement, so only it carries the key — one measurement must
    # resolve to one chart — and this one is named for the series it draws.
    ("prompt_surprisal_excess_trace", "aggregate", prompt_surprisal_excess_trace,
     None, False),
    ("perturbation_jsd_bits", "aggregate", perturbation_jsd_bits,
     "perturbation_jsd_bits", True),
    ("io_cosine_similarity", "aggregate", io_cosine_similarity,
     "io_cosine_similarity", True),
    ("output_entropy_bits", "reading", output_entropy_bits,
     "output_entropy_bits", True),
    ("prompt_surprisal_excess_bits", "reading", prompt_surprisal_excess_bits,
     "prompt_surprisal_excess_bits", True),
]

WITHHELD = (
    "The run did not publish {key} — the evidence for it does not exist here. "
    "Drawing this chart would show a trace for a quantity the record declines "
    "to claim. See `hif models` for what this backend supports."
)


def _gated(mod, mkey: str | None, draws: bool):
    """Wrap a chart's availability check AND its generator with its measurement's.

    Both, because each `generate()` re-asks its own module-level `available()`
    to decide whether to draw the not-available placeholder — so gating the
    predicate alone would leave the index page correctly marking a chart
    unavailable while the chart file beside it was drawn in full.

    A chart may only be drawn when the record publishes the measurement it
    draws. Each chart's `available()` reads whichever block it happens to
    plot and answers from that; the measurement's absence rules live in
    `hif/profile/measure.py` and are stricter, because a block being merely
    *present* is not the same as its contents being real. On a selected-only
    backend `metrics.distribution` is populated with point masses, so the
    entropy chart's own check passed and it rendered a full trace for
    `output_entropy_bits` — a number the record deliberately withheld. Same
    for sensitivity: divergences between point masses are a token-
    disagreement rate, not the quantity the key names.

    Derived rather than hand-written per chart, for the same reason the
    `needs_distribution_pair` sweep in `measure.py` is: hand-enforcement is
    what let those two diverge in the first place, and a chart added later
    inherits this for free. Pinned by
    `tests/unit/test_chart_measurement_gate.py`.
    """
    own, own_generate = mod.available, mod.generate
    if mkey is None or not draws:
        return own, own_generate

    def available(profile) -> str | None:
        reason = own(profile)
        if reason is not None:
            return reason
        # Imported here: `hif.profile.measure` is the authority on absence,
        # and a module-level import would make the viz package a dependency
        # of nothing it needs at import time.
        from hif.profile.measure import measurements, prompt_measurements

        # Both blocks, because the gate asks "did the run publish this value"
        # and `measurements()` alone answers a narrower question: "did the run
        # publish this value ABOUT THE TARGET". A row whose effective subject
        # is prompt-only under `--surrogate` leaves `measurements()` for
        # `prompt_measurements()` with its value intact — that is a change of
        # subject, not an absence. Reading only the first block would make
        # `prompt_surprisal_excess_bits` write "the evidence does not
        # exist here" onto every
        # surrogate run, beside a record carrying the number.
        published = mkey in measurements(profile) or (
            mkey in prompt_measurements(profile)
        )
        if not published:
            return WITHHELD.format(key=mkey)
        return None

    def generate(profile, output_path, formats=["html"]):
        reason = available(profile)
        if reason is not None:
            from hif.viz.base import na_figure, save_fig

            return save_fig(na_figure(mod.LABEL, reason), output_path, formats)
        return own_generate(profile, output_path, formats=formats)

    return available, generate


def _build(sid: str, kind: str, mod, mkey: str | None, draws: bool) -> SignalViz:
    available, generate = _gated(mod, mkey, draws)
    return SignalViz(
        id=sid, label=mod.LABEL, kind=kind,
        # One vocabulary: the chart's family IS its measurement's functional.
        # nucleus_effective_support_size (no measurement) draws support size off
        # the output distribution — information-theoretic like the entropy it
        # re-scales.
        family=(
            MEASUREMENT_BY_KEY[mkey].functional
            if mkey is not None
            else "information-theoretic"
        ),
        available=available, generate=generate,
        measurement_key=mkey, draws_measurement=draws,
    )


SIGNALS: list[SignalViz] = [
    _build(sid, kind, mod, mkey, draws) for (sid, kind, mod, mkey, draws) in _SPEC
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
    # input_entropy_trace chart's measurement summarises.
    "input_entropy_shift_bits": "input_entropy_trace",
    # No chart of its own, and deliberately so: it is the same per-step series
    # the entropy chart already draws, read over a fixed fraction of the mass
    # instead of over everything the backend exposed. A second near-identical
    # trace would invite reading the gap between the two lines as a finding,
    # when it is the definition of the two lines.
    "output_nucleus_entropy_bits": "output_entropy_bits",
}


def resolve_signal(name: str) -> SignalViz | None:
    """Resolve a signal id OR a measurement key to its chart, else None."""
    sig = SIGNALS_BY_ID.get(name)
    if sig is None:
        sig = SIGNALS_BY_MEASUREMENT.get(name)
    return sig


AGGREGATE_SIGNALS = [s.id for s in SIGNALS if s.kind == "aggregate"]
READING_SIGNALS = [s.id for s in SIGNALS if s.kind == "reading"]
