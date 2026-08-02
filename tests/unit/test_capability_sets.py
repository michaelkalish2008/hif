"""The capability sets are the registry, viewed — never a second list.

hif/models/capabilities.py used to hand-maintain frozensets of measurement
keys. They drifted twice: `input_entropy_std_bits` fell out of every set (so
`metric_support` promised it on backends that cannot produce it), and
`branch_pairwise_cosine_similarity` vanished from the `hif models` capability
matrix. The sets are now derived from MEASUREMENT_REGISTRY rows by
comprehension; these tests pin the properties that make the derivation safe,
so a registry change that breaks the mapping fails here instead of shipping a
wrong capability answer.
"""

from __future__ import annotations

from hif.models.capabilities import (
    ALL_GROUPED_METRICS,
    ATTENTION_METRICS,
    BACKENDS,
    INPUT_SIDE_METRICS,
    NEEDS_DISTRIBUTION,
    NEEDS_TWO_DISTRIBUTIONS,
    OUTPUT_SIDE_METRICS,
    TRAJECTORY_METRICS,
    metric_support,
    signals_available,
)
from hif.profile.signals import MEASUREMENT_BY_KEY, MEASUREMENT_KEYS

_GROUPS = {
    "INPUT_SIDE_METRICS": INPUT_SIDE_METRICS,
    "ATTENTION_METRICS": ATTENTION_METRICS,
    "OUTPUT_SIDE_METRICS": OUTPUT_SIDE_METRICS,
    "TRAJECTORY_METRICS": TRAJECTORY_METRICS,
}


def test_every_registry_key_is_in_exactly_one_group():
    """The partition that makes `metric_support` total over the registry.

    A key in no group would silently be reported available everywhere (the
    input_entropy_std_bits drift); a key in two groups would answer two
    different capability questions at once.
    """
    for key in MEASUREMENT_KEYS:
        homes = [name for name, group in _GROUPS.items() if key in group]
        assert len(homes) == 1, f"{key!r} is in {homes or 'no group'}"


def test_no_group_contains_a_key_the_registry_lacks():
    registry = set(MEASUREMENT_KEYS)
    for name, group in _GROUPS.items():
        stray = group - registry
        assert not stray, f"{name} carries non-registry keys: {sorted(stray)}"
    assert ALL_GROUPED_METRICS == registry


def test_each_group_is_the_registry_fact_it_claims_to_be():
    """The derivation predicates, restated independently of capabilities.py.

    If a row changes in a way that moves a key between groups, this names the
    row so the change is a decision rather than an accident.
    """
    for key in MEASUREMENT_KEYS:
        m = MEASUREMENT_BY_KEY[key]
        assert (key in INPUT_SIDE_METRICS) == (m.surrogate_group == "input")
        assert (key in ATTENTION_METRICS) == (m.observable == "attention row")
        assert (key in TRAJECTORY_METRICS) == (
            m.observable == "trajectory branch embeddings"
        )


def test_selected_only_sets_are_registry_facts_too():
    registry = set(MEASUREMENT_KEYS)
    assert NEEDS_DISTRIBUTION <= registry
    assert NEEDS_TWO_DISTRIBUTIONS <= registry
    # The two absences are different (one is surrogate-recoverable, the other
    # is not), so a key must not claim both.
    assert not (NEEDS_DISTRIBUTION & NEEDS_TWO_DISTRIBUTIONS)
    for key in MEASUREMENT_KEYS:
        m = MEASUREMENT_BY_KEY[key]
        assert (key in NEEDS_DISTRIBUTION) == (m.surrogate_group == "output")
        assert (key in NEEDS_TWO_DISTRIBUTIONS) == m.needs_distribution_pair


def test_every_registry_key_has_a_capability_answer_on_every_backend():
    """`signals_available` must answer for the whole registry, per backend.

    This is the invariant whose absence let `hif models` silently omit
    branch_pairwise_cosine_similarity from its capability matrix.
    """
    for backend in BACKENDS:
        answers = signals_available(backend)
        assert set(answers) == set(MEASUREMENT_KEYS)
        for key in MEASUREMENT_KEYS:
            # metric_support returns None (supported) or a reason string;
            # either way it must have an opinion, never raise.
            reason = metric_support(key, backend)
            assert reason is None or isinstance(reason, str)


def test_the_historical_drifts_stay_fixed():
    """The two live consequences of the hand-list era, pinned as regressions."""
    # Ollama cannot teacher-force; the std of the input-entropy series must be
    # refused there, not waved through as "supported".
    assert metric_support("input_entropy_std_bits", "ollama") is not None
    # And the trajectory row must appear in every backend's capability matrix
    # (refused where teacher forcing is absent, supported where it is).
    assert "branch_pairwise_cosine_similarity" in signals_available("anthropic")
    assert signals_available("anthropic")["branch_pairwise_cosine_similarity"] is False
    assert signals_available("hf")["branch_pairwise_cosine_similarity"] is True
