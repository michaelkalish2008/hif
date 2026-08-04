"""The zero-variance canary: subject semantics, not subject spelling.

The empirical signature that exposed `attention_entropy_input_bits` was two
different models returning a bit-identical value — 1.6677721955190443 for both
gpt2 and gpt2-medium. A number that does not move when the model changes is not
a measurement of the model, whatever its registry row says.

These tests encode that discovery as a permanent check. The same prompt is run
through two mock models built to differ in every input a measurement can read,
and then:

* every measurement whose subject is target-derived must DIFFER between them,
* every measurement whose subject is `prompt-only` must be IDENTICAL.

Both directions matter. The first catches a row that claims to be about the
target while computing something the target cannot influence — the original
defect. The second catches the opposite mislabelling: a quantity declared
prompt-only that in fact varies with the target is not comparable across
targets, which is the entire reason the prompt-only block exists.

This tests what the subject field MEANS rather than how it is spelled, so a
contributor who declares the wrong value fails automatically without anyone
reading their diff against the code.

Offline: mock models, a hash-based embedder, and a stand-in for the attention
analyser. No weights are downloaded and no API key is read.
"""

from __future__ import annotations

import pytest

from hif.profile.builder import build_profile
from hif.profile.signals import (
    MEASUREMENT_KEYS,
    SUBJECT_PROMPT_ONLY,
    measurements,
    prompt_measurements,
    run_subjects,
)
from mock_backends import (
    TIER_SELECTED_ONLY,
    TextHashEmbedder,
    alpha_model,
    beta_model,
    contract_config,
    install_attention_analyzer,
    install_perturbation_generator,
    surrogate_model,
)

PROMPT = "Explain why the sky appears blue."


@pytest.fixture(autouse=True)
def _offline_stages(monkeypatch):
    install_perturbation_generator(monkeypatch)
    install_attention_analyzer(monkeypatch)


def _run(model, *, backend: str = "hf", surrogate=None):
    return build_profile(
        model=model,
        prompt=PROMPT,
        regime="test",
        config=contract_config(backend),
        embedder=TextHashEmbedder(),
        seed=42,
        surrogate_model=surrogate,
    )


def _values(profile) -> dict[str, float]:
    """Everything the run produced, target-side and prompt-side alike."""
    return {**measurements(profile), **prompt_measurements(profile)}


# ---------------------------------------------------------------------------
# The canary
# ---------------------------------------------------------------------------


def test_two_different_models_move_every_target_derived_measurement():
    """Same prompt, two different models: the target-side numbers must move."""
    a, b = _run(alpha_model()), _run(beta_model())
    va, vb = _values(a), _values(b)
    subjects = run_subjects(a)

    assert set(va) == set(vb), (
        "the two runs produced different measurement sets, so the comparison "
        "is not like-for-like — both mocks are full-access with every stage "
        f"enabled: {sorted(set(va) ^ set(vb))}"
    )

    unmoved = [
        key
        for key, value in va.items()
        if subjects[key] != SUBJECT_PROMPT_ONLY and value == vb[key]
    ]
    assert not unmoved, (
        f"{unmoved} are declared to be about the target, but two models built "
        "to differ in every observable returned identical values. Either the "
        "computation cannot see the target — in which case the subject is "
        "wrong — or the mocks no longer differ in the input that measurement "
        "reads (see tests/unit/mock_backends.py)."
    )


def test_no_prompt_only_measurements_on_a_teacher_forcing_backend():
    """On an [F] backend, every measurement is about the target.

    Before hif-v4 this test asserted that prompt-only quantities are identical
    across models, and its specimen was `attention_entropy_input_bits` — a row
    that was prompt-only on EVERY backend because the number came from a fixed
    encoder reading the prompt. That row is cut: bit-identical across all
    fifteen corpus models, it measured the prompt, and the set is for
    measurements of the model. What remains prompt-only arises only under
    --surrogate, where the input-side rows describe the prompt by construction.
    So the invariant on a teacher-forcing run is now stronger and simpler:
    nothing in the record is about anything except the target.
    """
    a = _run(alpha_model())
    subjects = run_subjects(a)
    prompt_only = {k for k, s in subjects.items() if s == SUBJECT_PROMPT_ONLY}
    assert not (prompt_only & set(_values(a))), (
        f"prompt-only quantities {prompt_only & set(_values(a))} produced on a "
        "teacher-forcing backend — either a subject is mislabelled or a "
        "prompt-only row has been reintroduced without a surrogate gate"
    )


def test_the_canary_exercises_every_registered_measurement():
    """A row no run produces is a row the canary cannot check.

    Asserted as set equality against the registry, so a new measurement is
    covered the moment its row is added — and a contributor who adds one the
    mock backends cannot produce is told to extend them rather than silently
    shipping an unchecked subject declaration.
    """
    produced = set(_values(_run(alpha_model())))
    assert produced == set(MEASUREMENT_KEYS), (
        "the canary does not exercise every registered measurement: "
        f"missing {sorted(set(MEASUREMENT_KEYS) - produced)}, unexpected "
        f"{sorted(produced - set(MEASUREMENT_KEYS))}. Extend "
        "tests/unit/mock_backends.py so the new row is produced and its "
        "subject is checked."
    )


def test_one_surrogate_over_two_targets_returns_the_same_prompt_readings():
    """The degraded case, in the form the record actually publishes.

    On a backend that cannot teacher-force, a `--surrogate` reads the PROMPT in
    the target's place. Whatever it reports is a fact about the prompt under
    the surrogate, so two different targets profiled with the same surrogate
    must get byte-identical values — and if any of them moves, it was never
    prompt-only.
    """
    surrogate = surrogate_model()
    a = _run(
        alpha_model(tier=TIER_SELECTED_ONLY),
        backend="anthropic",
        surrogate=surrogate,
    )
    b = _run(
        beta_model(tier=TIER_SELECTED_ONLY),
        backend="anthropic",
        surrogate=surrogate,
    )
    pa, pb = prompt_measurements(a), prompt_measurements(b)

    assert set(pa) == set(pb) and pa, "no prompt-only values to compare"
    # The input-side rows degraded to prompt-only, not just the attention row:
    # this is the surrogate path, not the static one.
    assert "prompt_surprisal_excess_bits" in pa
    for key in sorted(pa):
        assert pa[key] == pb[key], (
            f"{key} was produced by the same surrogate reading the same "
            f"prompt for both targets, yet differs ({pa[key]!r} vs "
            f"{pb[key]!r}) — it is not prompt-only"
        )

    # And the target's own output response still moves: a surrogate on the
    # input side does not make the whole record prompt-only.
    ma, mb = measurements(a), measurements(b)
    assert ma["io_cosine_similarity"] != mb["io_cosine_similarity"]


# ---------------------------------------------------------------------------
# A run that produced nothing must measure nothing
# ---------------------------------------------------------------------------


def test_a_run_with_no_output_steps_measures_no_distribution_quantities():
    """Zero generated steps is absence of evidence, not a measured zero.

    A gpt-5 run returned no output steps at all — the backend refuses logprobs
    — and the record carried `perturbation_jsd_bits = 0.0` and
    `io_correlation_r = 0.0`, which were published as measurements of the
    model. Both are 0.0 by construction over an empty series.

    The cause was a guard written to stop `all([]) == True` claiming
    degeneracy: `output_distribution_degenerate` opens with `bool(steps)`, so
    an empty run answered "not degenerate" and every downstream computation
    proceeded on no data. Absence of evidence is the stronger reason to
    withhold, not the weaker one.
    """
    from hif.hourglass.output_side import (
        output_distribution_degenerate,
        output_distributions_unusable,
    )

    # The two predicates answer different questions, and only one of them is
    # about computability.
    assert output_distribution_degenerate([]) is False
    assert output_distributions_unusable([]) is True


def test_every_row_needing_a_distribution_pair_is_absent_without_one():
    """The rule is derived from the rows, so a new row inherits it.

    It used to be hand-written into individual branches, which is why
    `io_correlation_r` — half a JSD series by construction — kept emitting.
    """
    from hif.profile.registry import MEASUREMENT_REGISTRY

    pair_rows = [m.key for m in MEASUREMENT_REGISTRY if m.needs_distribution_pair]
    assert pair_rows, "no row declares needs_distribution_pair — the guard has nothing to enforce"
    # io_correlation_r carried the flag until hif-v4 cut the row itself.
    assert "perturbation_jsd_bits" in pair_rows


def test_sentinel_logprobs_do_not_become_a_measured_point_mass():
    """-9999 filler entries are not a distribution.

    DeepSeek at temperature=0 fills every non-selected top_logprob with a
    ~-9999 sentinel. Softmaxed, that is prob 1.0 / 0.0 / … — entropy 0.0 by
    construction — and it walks past the selected-only guard because the topk
    LIST has twenty entries. Sixteen published profiles carried
    output_entropy_bits = 0.0 this way. After sentinel filtering the step
    keeps only the entries the provider actually scored.
    """
    raw = [("a", -0.01), ("b", -9999.0), ("c", -9999.0)]
    kept = [(t, lp) for t, lp in raw if lp > -9000.0]
    assert kept == [("a", -0.01)]  # selected-only → degeneracy machinery applies

