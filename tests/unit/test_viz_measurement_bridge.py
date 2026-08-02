"""The chart registry's bridge to the measurement registry.

`hif profile --metric X --charts` passes a MEASUREMENT key where the viz
engine used to accept only its own signal ids — two disjoint namespaces, so
the combination *always* failed with "Pipeline error: Unknown signal", for
every one of the keys `--metric` itself validates against. The registry now
carries an explicit `measurement_key` bridge (and shares the measurement
registry's `functional` vocabulary for `family`); these tests pin the bridge
so the two registries cannot drift apart again.
"""

from __future__ import annotations

import pytest

from hif.profile.signals import FUNCTIONALS, MEASUREMENT_BY_KEY, MEASUREMENT_KEYS
from hif.viz.registry import (
    NEAREST_CHART,
    SIGNALS,
    SIGNALS_BY_ID,
    SIGNALS_BY_MEASUREMENT,
    resolve_signal,
)

from tests.unit.profile_helpers import _make_profile


def test_measurement_keys_on_charts_are_registry_keys():
    for sig in SIGNALS:
        if sig.measurement_key is not None:
            assert sig.measurement_key in MEASUREMENT_BY_KEY, sig.id


def test_measurement_to_chart_mapping_is_injective():
    """One measurement resolves to one chart; two charts must not both claim
    the same key (surprise/wager draw the same series — only wager carries it)."""
    keys = [s.measurement_key for s in SIGNALS if s.measurement_key is not None]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(SIGNALS_BY_MEASUREMENT)


def test_family_vocabulary_is_the_registry_functional():
    """The viz registry used to say family="information" against the
    measurement registry's "information-theoretic" — one concept, two
    vocabularies. Now every chart's family is a FUNCTIONALS value, and equals
    its measurement's functional where a measurement exists."""
    for sig in SIGNALS:
        assert sig.family in FUNCTIONALS, (sig.id, sig.family)
        if sig.measurement_key is not None:
            assert sig.family == MEASUREMENT_BY_KEY[sig.measurement_key].functional


def test_every_measurement_key_resolves_or_has_a_named_nearest_chart():
    for key in MEASUREMENT_KEYS:
        sig = resolve_signal(key)
        if sig is None:
            assert key in NEAREST_CHART, (
                f"{key!r} has no chart and no nearest-chart entry — "
                f"--metric {key} --charts would fail with 'Unknown signal'"
            )
            assert NEAREST_CHART[key] in SIGNALS_BY_ID
        else:
            assert sig.id in SIGNALS_BY_ID


def test_nearest_chart_never_shadows_a_real_chart():
    assert not set(NEAREST_CHART) & set(SIGNALS_BY_MEASUREMENT)


@pytest.mark.parametrize("key", sorted(MEASUREMENT_KEYS))
def test_charts_for_every_metric_render_or_fail_naming_the_nearest(key, tmp_path):
    """The CLI contract: --metric <any registry key> --charts either renders a
    chart (live data or an honest placeholder) or fails with a message naming
    the nearest chart. Never "Unknown signal"."""
    from hif.viz import generate_signal_plots

    profile = _make_profile()
    try:
        res = generate_signal_plots(profile, tmp_path, only_signal=key)
    except ValueError as exc:
        msg = str(exc)
        assert "Unknown signal" not in msg
        nearest = NEAREST_CHART[key]  # only unmapped keys may raise
        assert nearest in msg and SIGNALS_BY_ID[nearest].label in msg
    else:
        # Keyed by the requested name, so the CLI's res.get(metric) works.
        assert key in res
        assert res[key]["html"].exists()


def test_signal_ids_still_resolve_directly():
    for sig_id in SIGNALS_BY_ID:
        assert resolve_signal(sig_id) is SIGNALS_BY_ID[sig_id]
