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
    SUBJECT_LEGEND,
    SUBJECT_PROMPT_ONLY,
    SUBJECTS,
    effective_subject,
    measurements,
    prompt_measurement_block,
    prompt_measurements,
    run_subjects,
    signals_record,
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


def test_every_row_declares_a_subject():
    """A row that cannot say whose behaviour it describes does not belong.

    The subject is what the record could not previously express: a
    surrogate-derived number that never touched the target used to be emitted
    with a caveat flag, as though it were a caveated fact about the target.
    """
    for m in MEASUREMENT_REGISTRY:
        assert m.subject in SUBJECTS, (
            f"{m.key}: subject {m.subject!r} not in {SUBJECTS}"
        )


def test_subject_under_surrogate_is_a_subject_or_none():
    for m in MEASUREMENT_REGISTRY:
        assert m.subject_under_surrogate is None or (
            m.subject_under_surrogate in SUBJECTS
        ), f"{m.key}: subject_under_surrogate {m.subject_under_surrogate!r}"


def test_degrading_rows_declare_the_surrogate_that_degrades_them():
    """A declared degradation that no surrogate can trigger is dead prose.

    effective_subject() resolves the degradation against surrogate_group, so a
    row claiming one without naming a group would silently never degrade.
    """
    for m in MEASUREMENT_REGISTRY:
        if m.subject_under_surrogate is not None:
            assert m.surrogate_group in ("input", "output"), (
                f"{m.key}: declares subject_under_surrogate but no "
                f"surrogate_group to trigger it"
            )
            assert m.subject_under_surrogate != m.subject, (
                f"{m.key}: subject_under_surrogate repeats subject — declare "
                f"None instead"
            )


def test_subject_legend_covers_exactly_the_enum():
    assert set(SUBJECT_LEGEND) == set(SUBJECTS)
    for value, gloss in SUBJECT_LEGEND.items():
        assert gloss.strip(), f"{value}: empty legend line"


def test_effective_subject_resolves_against_the_run():
    """The declared subject holds on [F]; the degraded one under a surrogate."""
    for m in MEASUREMENT_REGISTRY:
        assert effective_subject(m) == m.subject
        degraded = effective_subject(
            m, input_surrogate=True, output_surrogate=True
        )
        expected = m.subject if m.subject_under_surrogate is None else (
            m.subject_under_surrogate
        )
        assert degraded == expected, m.key
        # A surrogate for the OTHER side never changes this row's subject.
        other = "output" if m.surrogate_group == "input" else "input"
        assert effective_subject(m, **{f"{other}_surrogate": True}) == m.subject


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
# Absent, not flagged — a prompt-only quantity is never a measurement of the
# target. This is the absent-not-pinned rule extended from "cannot measure" to
# "measured something else".
# ---------------------------------------------------------------------------


def _surrogate_profile():
    """A profile as a `--surrogate` run on a closed backend produces it.

    findings.surrogate_model_name is what build_profile sets when the target
    cannot teacher-force and a surrogate read the prompt in its place.
    """
    p = _make_profile()
    p.findings.surrogate_model_name = "unsloth/Llama-3.2-1B"
    return p


def test_measurements_and_prompt_measurements_partition_the_values():
    """No value is dropped by the split, and none is reported twice."""
    for p in (_make_profile(), _surrogate_profile()):
        target = measurements(p)
        prompt = prompt_measurements(p)
        assert not set(target) & set(prompt)
        subjects = run_subjects(p)
        assert all(subjects[k] != SUBJECT_PROMPT_ONLY for k in target)
        assert all(subjects[k] == SUBJECT_PROMPT_ONLY for k in prompt)


def test_surrogate_run_moves_prompt_only_quantities_out_of_measurements():
    p = _surrogate_profile()
    target = measurements(p)
    prompt = prompt_measurements(p)

    # The surrogate teacher-forced the PROMPT; the target contributed nothing
    # to these, so they are not measurements of the target at any caveat level.
    for key in ("input_entropy_shift_bits", "prompt_surprisal_excess_bits"):
        assert key not in target, f"{key} emitted as a measurement of the target"
    assert "input_entropy_shift_bits" in prompt
    # The target's own output response is unaffected by an input surrogate.
    assert "perturbation_jsd_bits" in target
    # Mixed, not prompt-only: the target's response is half the computation.
    assert "io_correlation_r" in target
    assert run_subjects(p)["io_correlation_r"] == "mixed"


def test_full_access_run_keeps_input_side_quantities_as_measurements():
    """Without a surrogate the same quantities ARE about the target."""
    p = _make_profile()  # findings.surrogate_model_name is None
    target = measurements(p)
    assert "input_entropy_shift_bits" in target
    assert "prompt_surprisal_excess_bits" in target
    assert not prompt_measurements(p)
    # Nothing to report about the prompt ⇒ no block at all, rather than an
    # empty one asserting the run looked and found nothing.
    assert prompt_measurement_block(p) is None


def test_attention_input_entropy_is_prompt_only_even_on_a_full_access_backend():
    """The one row no access tier can make target-side.

    hif/analysis/attention.py runs a bidirectional encoder over text as an
    object — "This is NOT the model under analysis". The output-side row reads
    the target's actual generated continuation, so it moves when the target
    does; the input-side row reads the PROMPT, so it is a function of prompt
    text and encoder weights alone and cannot see the target at all.
    """
    from tests.unit.test_attention import _make_attention_data

    p = _make_profile()  # no surrogate: a full-access run
    p.attention_capture = _make_attention_data()

    target = measurements(p)
    prompt = prompt_measurements(p)
    assert "attention_entropy_input_bits" in prompt
    assert "attention_entropy_input_bits" not in target
    assert "attention_entropy_output_bits" in target

    block = prompt_measurement_block(p)
    assert (
        block["reference_models"]["attention_entropy_input_bits"]
        == "distilbert-base-uncased"
    )


def _record(p):
    return signals_record(
        p, model_name="m", backend="b", regime="r", seed=1, prompt="hi"
    )


def test_record_omits_the_prompt_block_on_a_full_access_run():
    record = _record(_make_profile())
    assert "prompt_measurements" not in record


def test_record_carries_prompt_block_with_subject_and_reference_model():
    record = _record(_surrogate_profile())
    block = record["prompt_measurements"]
    assert block["subject"] == SUBJECT_PROMPT_ONLY
    assert set(block["values"]) == set(block["reference_models"])
    assert all(
        ref == "unsloth/Llama-3.2-1B" for ref in block["reference_models"].values()
    )
    # And the same keys are absent from the measurement set.
    assert not set(block["values"]) & set(record["measurements"])


def test_units_block_covers_the_prompt_values_too():
    record = signals_record(
        _surrogate_profile(), model_name="m", backend="b", regime="r",
        seed=1, prompt="hi", include_units=True,
    )
    for key in record["prompt_measurements"]["values"]:
        assert key in record["units"]


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
        assert row["subject"] == m.subject
        assert row["subject_under_surrogate"] == m.subject_under_surrogate
        assert row["surrogate_group"] == (m.surrogate_group or None)
    # The resolution legend covers exactly the declared enum.
    assert set(doc["resolutions"]) == set(RESOLUTIONS)
    # …and so does the subject legend, one line each.
    assert doc["subjects"] == dict(SUBJECT_LEGEND)


def test_schema_text_mode_shows_the_subject_and_its_legend():
    result = runner.invoke(app, ["schema", "--text"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "Subject" in flat
    for value in SUBJECTS:
        assert value in flat


def test_models_names_the_measurements_that_are_never_about_the_target():
    """`hif models` must say which quantities no backend can make target-side,
    and which degrade to prompt-only when a surrogate stands in."""
    result = runner.invoke(app, ["models", "--backend", "anthropic"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "never about the target" in flat
    assert "attention_entropy_input_bits" in flat
    assert "prompt-only under --surrogate" in flat
    assert "prompt_measurements" in flat


def test_models_does_not_promise_surrogate_degradation_on_a_teacher_forcing_backend():
    result = runner.invoke(app, ["models", "--backend", "hf"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "prompt-only under --surrogate" not in flat
    # The unconditional case still applies on [F].
    assert "never about the target" in flat
