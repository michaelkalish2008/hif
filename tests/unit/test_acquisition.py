"""The acquisition axis: what a measurement costs in produced content.

The registry claims, per row, whether taking a measurement requires nothing
beyond the caller's own call (`observational`), authored prompt text
(`synthesized-input`), or model output that did not exist before
(`elicited-output`). That claim is only worth having if the pipeline honours it,
and the registry deliberately imports nothing from the pipeline — so nothing but
a test can hold the two together.

The invariant: a run capped at tier T reports exactly the rows whose acquisition
tier is at or below T. Not a superset (the cap leaked), not a subset (the row is
mislabelled and needs more than it claims).
"""

from __future__ import annotations

import pytest

from hif.config import (
    ClusterConfig,
    GenerationConfig,
    ModelConfig,
    PerturbationConfig,
    RunConfig,
    TrajectoryConfig,
)
from hif.profile.builder import build_profile
from hif.profile.record import signals_record
from hif.profile.registry import (
    ACQUISITION_ELICITED_OUTPUT,
    ACQUISITION_OBSERVATIONAL,
    ACQUISITION_SYNTHESIZED_INPUT,
    ACQUISITIONS,
    MEASUREMENT_REGISTRY,
)
from hif.perturbation.base import PerturbationGenerator, PerturbationResult

import hashlib

import numpy as np

from profile_helpers import FakeModel


class TextDeterministicEmbedder:
    """An encoder whose output depends on the TEXT and nothing else.

    `profile_helpers.FakeEmbeddingModel` draws from one RNG stream, so a text's
    vector depends on how many embeddings were taken before it. That is fine
    for tests that only need well-formed vectors, but it makes cross-run
    comparison impossible here: raising the acquisition ceiling adds embedding
    calls, which would shift every later vector and show up as geometric
    measurements "moving" when nothing about them changed. A real encoder
    returns the same vector for the same string, and so does this.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(digest, "big"))
            vec = rng.random(self._dim).astype(np.float32)
            rows.append(vec / (np.linalg.norm(vec) + 1e-8))
        return np.array(rows, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

# Ordered weakest to strongest — each tier permits everything below it.
TIER_ORDER = [
    ACQUISITION_OBSERVATIONAL,
    ACQUISITION_SYNTHESIZED_INPUT,
    ACQUISITION_ELICITED_OUTPUT,
]


class FakeSynonymGenerator(PerturbationGenerator):
    name = "synonym"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        # Honours n_variants, unlike the one-variant stub other builder tests
        # use: input_entropy_std_bits is a spread across variants and needs two
        # distinct ones to exist at all.
        return PerturbationResult(
            original=prompt,
            variants=[f"hi world {i}" for i in range(n_variants)],
            generator="synonym",
        )


@pytest.fixture(autouse=True)
def _fake_synonym_generator(monkeypatch):
    import hif.perturbation as perturbation_module

    monkeypatch.setitem(perturbation_module._RULE_TYPES, "synonym", FakeSynonymGenerator)


def _config_for(tier: str) -> RunConfig:
    """The same overrides `--acquisition` applies in hif/cli.py."""
    config = RunConfig(
        model=ModelConfig(name="mock-model", backend="hf"),
        generation=GenerationConfig(max_new_tokens=3, top_k=5),
        trajectory=TrajectoryConfig(n_branches=2, rollout_steps=3),
        # Two variants: input_entropy_std_bits is a sample standard
        # deviation and is absent below two, for reasons that have
        # nothing to do with the acquisition ceiling.
        perturbation=PerturbationConfig(n_variants=2, generators=["synonym"]),
        cluster=ClusterConfig(method="kmeans"),
    )
    if tier == ACQUISITION_OBSERVATIONAL:
        config.perturbation.generators = []
        config.perturbation.n_variants = 0
        config.trajectory.n_branches = 0
    elif tier == ACQUISITION_SYNTHESIZED_INPUT:
        config.perturbation.elicit_variant_outputs = False
        config.trajectory.n_branches = 0
    return config


def _reported_keys(tier: str) -> set[str]:
    profile = build_profile(
        model=FakeModel(vocab_size=100, context_length=512, name="mock-model"),
        prompt="hello world",
        regime="test",
        config=_config_for(tier),
        embedder=TextDeterministicEmbedder(dim=16),
        seed=42,
    )
    record = signals_record(
        profile,
        model_name="mock-model",
        prompt="hello world",
        backend="hf",
        regime="test",
        seed=42,
    )
    return set(record["measurements"])


def _registry_keys_up_to(tier: str) -> set[str]:
    permitted = set(TIER_ORDER[: TIER_ORDER.index(tier) + 1])
    return {m.key for m in MEASUREMENT_REGISTRY if m.acquisition in permitted}


class TestAxisIsWellFormed:
    def test_every_row_declares_a_known_tier(self):
        for m in MEASUREMENT_REGISTRY:
            assert m.acquisition in ACQUISITIONS, f"{m.key} has acquisition={m.acquisition!r}"

    def test_all_three_tiers_are_populated(self):
        """A partition with an empty class is a distinction that isn't being
        drawn."""
        for tier in TIER_ORDER:
            assert any(m.acquisition == tier for m in MEASUREMENT_REGISTRY), tier


class TestPipelineHonoursTheAxis:
    @pytest.mark.parametrize("tier", TIER_ORDER)
    def test_run_reports_nothing_above_its_ceiling(self, tier):
        """The cap holds: no measurement needing more than the tier permits."""
        above = {
            m.key
            for m in MEASUREMENT_REGISTRY
            if TIER_ORDER.index(m.acquisition) > TIER_ORDER.index(tier)
        }
        leaked = _reported_keys(tier) & above
        assert not leaked, f"{tier} run reported {sorted(leaked)}, which it may not acquire"

    def test_observational_needs_no_authored_or_elicited_text(self):
        reported = _reported_keys(ACQUISITION_OBSERVATIONAL)
        assert reported, "an observational run must still measure something"
        assert reported <= _registry_keys_up_to(ACQUISITION_OBSERVATIONAL)

    def test_synthesized_input_adds_the_input_side_pair_and_nothing_else(self):
        """The de-entangling this axis exists for: the input-side measurements
        never needed a variant continuation, and now do not wait on one."""
        gained = _reported_keys(ACQUISITION_SYNTHESIZED_INPUT) - _reported_keys(
            ACQUISITION_OBSERVATIONAL
        )
        assert gained == {"input_entropy_shift_bits", "input_entropy_std_bits"}

    def test_elicited_output_adds_only_elicited_rows(self):
        gained = _reported_keys(ACQUISITION_ELICITED_OUTPUT) - _reported_keys(
            ACQUISITION_SYNTHESIZED_INPUT
        )
        assert gained, "the top tier must add something, or the ceiling is meaningless"
        for key in gained:
            row = next(m for m in MEASUREMENT_REGISTRY if m.key == key)
            assert row.acquisition == ACQUISITION_ELICITED_OUTPUT, (
                f"{key} appeared only once elicitation was permitted, but the "
                f"registry calls it {row.acquisition!r}"
            )

    def test_tiers_are_nested(self):
        """Raising the ceiling only ever adds. If a measurement disappears when
        MORE is permitted, a stage is interfering with one below it."""
        observational = _reported_keys(ACQUISITION_OBSERVATIONAL)
        synthesized = _reported_keys(ACQUISITION_SYNTHESIZED_INPUT)
        elicited = _reported_keys(ACQUISITION_ELICITED_OUTPUT)
        assert observational <= synthesized <= elicited


class TestValuesDoNotMoveWithTheCeiling:
    def test_observational_values_survive_unchanged_at_every_tier(self):
        """A ceiling removes measurements; it must not change the ones that
        remain. Otherwise --acquisition is a different measurement, not a
        subset of the same one."""
        profiles = {tier: None for tier in TIER_ORDER}
        records = {}
        for tier in TIER_ORDER:
            profile = build_profile(
                model=FakeModel(vocab_size=100, context_length=512, name="mock-model"),
                prompt="hello world",
                regime="test",
                config=_config_for(tier),
                embedder=TextDeterministicEmbedder(dim=16),
                seed=42,
            )
            records[tier] = signals_record(
                profile,
                model_name="mock-model",
                prompt="hello world",
                backend="hf",
                regime="test",
                seed=42,
            )["measurements"]
            profiles[tier] = profile

        base = records[ACQUISITION_OBSERVATIONAL]
        for tier in (ACQUISITION_SYNTHESIZED_INPUT, ACQUISITION_ELICITED_OUTPUT):
            for key, value in base.items():
                assert records[tier][key] == value, f"{key} moved between observational and {tier}"
