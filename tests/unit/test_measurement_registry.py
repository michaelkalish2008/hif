"""Registry invariants: MEASUREMENT_REGISTRY is the single source of truth.

These tests assert properties, never counts — the registry is the count, and
`hif schema` reports it. A new measurement is covered automatically the moment
its row is added (see CONTRIBUTING.md step 5).
"""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from hif.cli import app
from hif.profile.signals import (
    FUNCTIONALS,
    MEASUREMENT_KEYS,
    MEASUREMENT_REGISTRY,
    MEASUREMENT_UNITS,
    MEASUREMENTS,
    RESOLUTIONS,
    measurements,
)
from tests.unit.profile_helpers import _make_profile

runner = CliRunner()


# ---------------------------------------------------------------------------
# Row completeness and validity
# ---------------------------------------------------------------------------


def test_keys_are_unique():
    keys = [m.key for m in MEASUREMENT_REGISTRY]
    assert len(keys) == len(set(keys)), "duplicate measurement keys"


def test_every_row_is_complete():
    """Every row declares key, name, unit, definition, and its full triple."""
    for m in MEASUREMENT_REGISTRY:
        for field in ("key", "name", "unit", "definition", "observable"):
            assert getattr(m, field), f"{m.key or m}: empty {field}"


def test_every_row_has_a_valid_triple():
    for m in MEASUREMENT_REGISTRY:
        assert m.resolution in RESOLUTIONS, (
            f"{m.key}: resolution {m.resolution!r} not in {RESOLUTIONS}"
        )
        assert m.functional in FUNCTIONALS, (
            f"{m.key}: functional {m.functional!r} not in {FUNCTIONALS}"
        )


def test_surrogate_group_is_valid():
    for m in MEASUREMENT_REGISTRY:
        assert m.surrogate_group in ("", "input", "output"), (
            f"{m.key}: surrogate_group {m.surrogate_group!r}"
        )


def test_keys_are_snake_case():
    """Keys are stable machine names: lowercase snake_case, no spaces."""
    for m in MEASUREMENT_REGISTRY:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", m.key), m.key


def test_labels_are_canonical_shorthands_or_none():
    """A label is either a non-empty shorthand or None — never ""."""
    for m in MEASUREMENT_REGISTRY:
        assert m.label is None or m.label.strip(), f"{m.key}: blank label"
    labels = [m.label for m in MEASUREMENT_REGISTRY if m.label is not None]
    assert len(labels) == len(set(labels)), "duplicate labels"


# ---------------------------------------------------------------------------
# Derived views project the registry faithfully
# ---------------------------------------------------------------------------


def test_derived_views_match_registry():
    assert MEASUREMENT_KEYS == tuple(m.key for m in MEASUREMENT_REGISTRY)
    assert MEASUREMENTS == [
        (m.key, m.name, m.surrogate_group) for m in MEASUREMENT_REGISTRY
    ]
    for m in MEASUREMENT_REGISTRY:
        assert MEASUREMENT_UNITS[m.key].startswith(m.unit + " — ")


# ---------------------------------------------------------------------------
# Emitted ⊆ registered — a profile can never produce an unregistered key
# ---------------------------------------------------------------------------


def test_every_emitted_measurement_is_registered():
    vals = measurements(_make_profile())
    assert vals, "synthetic profile produced no measurements"
    unregistered = set(vals) - set(MEASUREMENT_KEYS)
    assert not unregistered, (
        f"measurements() emitted unregistered keys: {sorted(unregistered)} — "
        "add a registry row (CONTRIBUTING.md step 4)"
    )


# ---------------------------------------------------------------------------
# `hif schema` emits the full row per measurement
# ---------------------------------------------------------------------------


def test_schema_emits_full_row_per_measurement():
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert set(doc["measurements"]) == set(MEASUREMENT_KEYS)
    for m in MEASUREMENT_REGISTRY:
        row = doc["measurements"][m.key]
        assert row["name"] == m.name
        assert row["label"] == m.label
        assert row["unit"] == m.unit
        assert row["definition"] == m.definition
        assert row["observable"] == m.observable
        assert row["functional"] == m.functional
        assert row["resolution"] == m.resolution
        assert row["surrogate_group"] == (m.surrogate_group or None)
    # The resolution legend covers exactly the declared enum.
    assert set(doc["resolutions"]) == set(RESOLUTIONS)
