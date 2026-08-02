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


def test_prompt_only_measurements_are_identical_across_models():
    """A prompt-only quantity that moves with the target is mislabelled too."""
    a, b = _run(alpha_model()), _run(beta_model())
    va, vb = _values(a), _values(b)
    subjects = run_subjects(a)

    prompt_only = {k for k, s in subjects.items() if s == SUBJECT_PROMPT_ONLY}
    assert prompt_only & set(va), (
        "no prompt-only quantity was produced, so this assertion checked "
        "nothing — the attention stage must run for the input-side row to exist"
    )
    for key in sorted(prompt_only & set(va)):
        assert va[key] == vb[key], (
            f"{key} is declared prompt-only, but it changed when the target "
            f"model changed ({va[key]!r} vs {vb[key]!r}). A value that moves "
            "with the target is not comparable across targets, which is the "
            "only reason prompt-only values are reported at all."
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
