"""Shift ◆ — the step-to-step output divergence, and the rules that bound it.

Three properties are under test here, and each corresponds to a defect this
module was written to close:

1. Shift is a MEASUREMENT, not only a chart. It has a registry row, it is
   emitted by `measurements()`, and it reaches the machine record — so a reader
   who sees "Shift ◆" on the companion website can reproduce the number with
   the CLI.
2. The chart and the measurement are the same arithmetic. `hif/metrics/shift.py`
   owns it; `hif/viz/signals/shift.py` imports it. They cannot drift.
3. The saturation caveat is acted on, not just documented — the top-K overlap
   that bounds the divergence ships as its own measurement, and the divergence
   is ABSENT where it stops being a divergence between distributions at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from hif.metrics.shift import (
    NEEDS_DISTRIBUTIONS,
    NEEDS_TWO_STEPS,
    shift_jsd_bits,
    shift_summary,
    shift_trace,
    topk_overlap_fraction,
    unavailable_reason,
)
from hif.profile.signals import (
    MEASUREMENT_BY_KEY,
    MEASUREMENT_KEYS,
    measurements,
    signals_record,
)
from tests.unit.profile_helpers import _make_profile, _make_step

SHIFT_KEY = "output_step_jsd_bits"
OVERLAP_KEY = "output_step_topk_overlap_fraction"


# ---------------------------------------------------------------------------
# The computation, on inputs whose answer is checkable by hand
# ---------------------------------------------------------------------------


def test_identical_distributions_diverge_by_zero():
    p = {0: 0.5, 1: 0.5}
    assert shift_jsd_bits(p, dict(p)) == pytest.approx(0.0, abs=1e-12)
    assert topk_overlap_fraction(p, dict(p)) == 1.0


def test_disjoint_supports_pin_at_the_one_bit_ceiling():
    """The saturation the caveat is about: disjoint support ALONE gives 1 bit.

    Neither distribution is more "different" than the other case below; the
    supports simply do not meet, which is why the overlap fraction has to be
    reported next to the divergence.
    """
    p = {0: 0.5, 1: 0.5}
    q = {2: 0.5, 3: 0.5}
    assert shift_jsd_bits(p, q) == pytest.approx(1.0)
    assert topk_overlap_fraction(p, q) == 0.0


def test_half_overlapping_supports_give_a_hand_checkable_half_bit():
    # p = (.5, .5, 0), q = (0, .5, .5) over {0,1,2}; m = (.25, .5, .25).
    # KL(p‖m) = .5·log2(.5/.25) = .5, likewise KL(q‖m) ⇒ JSD = .5 bits.
    p = {0: 0.5, 1: 0.5}
    q = {1: 0.5, 2: 0.5}
    assert shift_jsd_bits(p, q) == pytest.approx(0.5)
    assert topk_overlap_fraction(p, q) == pytest.approx(1 / 3)


def test_stored_topk_mass_is_renormalised_before_the_divergence():
    """Top-K mass sums to < 1; the divergence must not read that as a shift."""
    p = {0: 0.02, 1: 0.02}          # 4% of the true mass
    q = {0: 0.4, 1: 0.4}            # 80% of it, same shape
    assert shift_jsd_bits(p, q) == pytest.approx(0.0, abs=1e-12)


def test_divergence_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = {int(k): float(v) for k, v in enumerate(rng.random(8))}
        q = {int(k) + 4: float(v) for k, v in enumerate(rng.random(8))}
        d = shift_jsd_bits(p, q)
        assert 0.0 <= d <= 1.0
        assert d == pytest.approx(shift_jsd_bits(q, p))


# ---------------------------------------------------------------------------
# Absence rules — None, never a fabricated 0.0
# ---------------------------------------------------------------------------


def _steps(dists: list[dict[int, float]]):
    return [_make_step(i, d) for i, d in enumerate(dists)]


def test_a_single_step_has_no_transition_to_measure():
    steps = _steps([{0: 0.5, 1: 0.5}])
    assert shift_summary(steps) is None
    assert unavailable_reason(steps) == NEEDS_TWO_STEPS


def test_point_masses_are_absent_not_a_token_disagreement_rate():
    """A selected-only backend's steps are point masses.

    JSD between two point masses is 0 when the tokens agree and exactly 1 bit
    when they differ — a token-disagreement indicator, not a divergence between
    distributions. Reporting it under this key would be a different quantity
    wearing the key's definition, so the quantity is absent.
    """
    steps = _steps([{0: 1.0}, {1: 1.0}, {2: 1.0}])
    assert shift_summary(steps) is None
    assert shift_trace(steps) == []
    assert unavailable_reason(steps) == NEEDS_DISTRIBUTIONS


def test_a_measured_zero_is_still_reported():
    """Absence is "no evidence"; two identical distributions are evidence."""
    d = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    summary = shift_summary(_steps([dict(d), dict(d), dict(d)]))
    assert summary is not None
    assert summary.mean_jsd_bits == pytest.approx(0.0, abs=1e-12)
    assert summary.mean_overlap_fraction == 1.0
    assert summary.n_transitions == 2


def test_summary_means_the_per_transition_trace():
    steps = _steps([{0: 0.5, 1: 0.5}, {1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}])
    trace = shift_trace(steps)
    summary = shift_summary(steps)
    assert [t.step for t in trace] == [1, 2]
    assert summary.mean_jsd_bits == pytest.approx(
        float(np.mean([t.jsd_bits for t in trace]))
    )
    assert summary.mean_overlap_fraction == pytest.approx(
        float(np.mean([t.topk_overlap_fraction for t in trace]))
    )


# ---------------------------------------------------------------------------
# The saturation caveat is acted on: the overlap ships as its own measurement
# ---------------------------------------------------------------------------


def test_the_overlap_companion_is_a_registered_measurement():
    """Not a flag on the Shift number — a measurement with its own row.

    A flag would be the thing the `subject` precedent rejects: an adornment the
    consumer has to know to look for. A registry row is a second fact, carried
    in `measurements` with its own unit and definition.
    """
    assert OVERLAP_KEY in MEASUREMENT_KEYS
    row = MEASUREMENT_BY_KEY[OVERLAP_KEY]
    assert row.unit == "fraction of shared top-K token ids"
    assert row.label is None
    # And the divergence's own definition points the reader at it, so the
    # bound cannot be read without the number that quantifies it.
    assert OVERLAP_KEY in MEASUREMENT_BY_KEY[SHIFT_KEY].definition


def test_a_saturated_run_reports_its_zero_overlap_alongside_the_ceiling():
    """The case the caveat describes, end to end."""
    steps = _steps([{0: 0.5, 1: 0.5}, {2: 0.5, 3: 0.5}, {4: 0.5, 5: 0.5}])
    summary = shift_summary(steps)
    assert summary.mean_jsd_bits == pytest.approx(1.0)
    assert summary.mean_overlap_fraction == 0.0


# ---------------------------------------------------------------------------
# Shift is reachable from the CLI: registered, emitted, recorded
# ---------------------------------------------------------------------------


def test_shift_is_registered_with_its_canonical_label():
    assert SHIFT_KEY in MEASUREMENT_KEYS
    row = MEASUREMENT_BY_KEY[SHIFT_KEY]
    assert row.label == "Shift ◆"
    assert row.unit == "bits"
    assert row.resolution == "per-step"
    assert row.functional == "information-theoretic"
    assert row.observable == "output distribution"
    assert row.subject == "target-distribution"


def test_the_entropy_delta_row_is_still_not_labelled_shift():
    """|H(i) − H(i−1)| is a different quantity and must not carry the name.

    Two steps can hold identical entropy over completely different token sets:
    the entropy delta would read 0 where Shift reads 1 bit.
    """
    assert MEASUREMENT_BY_KEY["output_entropy_step_delta_bits"].label is None
    labels = {m: MEASUREMENT_BY_KEY[m].label for m in MEASUREMENT_KEYS}
    assert [k for k, v in labels.items() if v == "Shift ◆"] == [SHIFT_KEY]


def test_a_profile_emits_shift_and_its_overlap():
    values = measurements(_make_profile())
    assert SHIFT_KEY in values
    assert OVERLAP_KEY in values
    # The synthetic profile repeats one uniform distribution across its steps.
    assert values[SHIFT_KEY] == pytest.approx(0.0, abs=1e-12)
    assert values[OVERLAP_KEY] == 1.0


def test_the_record_carries_shift():
    record = signals_record(
        _make_profile(), model_name="m", backend="hf", regime="r", seed=1,
        prompt="hi", include_units=True,
    )
    assert SHIFT_KEY in record["measurements"]
    assert OVERLAP_KEY in record["measurements"]
    assert record["units"][SHIFT_KEY].startswith("bits — ")


# ---------------------------------------------------------------------------
# One computation: the chart cannot drift from the number
# ---------------------------------------------------------------------------


def test_the_chart_imports_the_canonical_computation():
    """Identity, not equality — the viz module must not own a second copy."""
    import hif.metrics.shift as canonical
    import hif.viz.signals.shift as chart

    assert chart.shift_trace is canonical.shift_trace
    assert chart.shift_summary is canonical.shift_summary
    assert chart.LABEL == canonical.LABEL == "Shift"
    assert chart.GLYPH == canonical.GLYPH == "◆"
    # No private reimplementation left behind.
    assert not hasattr(chart, "_jsd")
    assert not hasattr(chart, "_overlap_frac")


def test_the_chart_declines_the_same_runs_the_measurement_declines():
    """`available()` and the absence rule are one decision, not two."""
    import hif.viz.signals.shift as chart

    p = _make_profile()
    assert chart.available(p) is None
    assert shift_summary(p.output_side.steps) is not None

    p.output_side.steps = _steps([{0: 1.0}, {1: 1.0}])
    assert chart.available(p) == NEEDS_DISTRIBUTIONS
    assert shift_summary(p.output_side.steps) is None
