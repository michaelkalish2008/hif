"""Unit tests for build_profile — wires all pipeline stages with a fake model."""

from __future__ import annotations

import numpy as np
import pytest

from hif.config import (
    ClusterConfig,
    GenerationConfig,
    ModelConfig,
    PerturbationConfig,
    RunConfig,
    TrajectoryConfig,
)
from hif.models.base import GenerationResult, Logits, StepRecord, TopKEntry
from hif.profile.builder import build_profile
from hif.profile.schema import BehavioralRangeProfile
from hif.perturbation.base import PerturbationGenerator, PerturbationResult

from profile_helpers import FakeEmbeddingModel, FakeModel


def _make_mock_model(vocab_size: int = 100) -> FakeModel:
    return FakeModel(vocab_size=vocab_size, context_length=512, name="mock-model")


def _make_mock_embedder(dim: int = 16) -> FakeEmbeddingModel:
    return FakeEmbeddingModel(dim=dim, seed=0)


class FakeSynonymGenerator(PerturbationGenerator):
    """Stands in for SynonymGenerator, which depends on real NLTK/WordNet data —
    an external dependency irrelevant to testing build_profile's orchestration."""

    name = "synonym"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        return PerturbationResult(original=prompt, variants=["hi world"], generator="synonym")


@pytest.fixture(autouse=True)
def _fake_synonym_generator(monkeypatch):
    """Inject a fake synonym generator into the registry build_profile resolves
    from — direct module-global injection, not mock.patch, matching this
    project's established pattern (see tests/README.md)."""
    import hif.perturbation as perturbation_module

    monkeypatch.setitem(perturbation_module._RULE_TYPES, "synonym", FakeSynonymGenerator)


def _make_config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(name="mock-model", backend="hf"),
        generation=GenerationConfig(max_new_tokens=3, top_k=5),
        trajectory=TrajectoryConfig(n_branches=2, rollout_steps=3),
        perturbation=PerturbationConfig(n_variants=1, generators=["synonym"]),
        cluster=ClusterConfig(method="kmeans", n_clusters=2),
    )


def _patched_build(**kwargs):
    """Call build_profile — synonym generation is faked via the autouse
    _fake_synonym_generator fixture, not here."""
    return build_profile(**kwargs)


class TestBuildProfileStructure:
    """Verify build_profile returns a valid BehavioralRangeProfile. No real model loaded."""

    def test_returns_behavioral_range_profile(self):
        profile = _patched_build(
            model=_make_mock_model(), prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
        )
        assert isinstance(profile, BehavioralRangeProfile)

    def test_profile_has_all_required_fields(self):
        profile = _patched_build(
            model=_make_mock_model(), prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
        )
        assert profile.model is not None
        assert profile.prompt is not None
        assert profile.input_side is not None
        assert profile.output_side is not None
        assert profile.center is not None
        assert profile.trajectory is not None
        assert profile.perturbations is not None
        assert profile.metrics is not None
        assert profile.findings is not None
        assert profile.config is not None

    def test_profile_model_identity_correct(self):
        profile = _patched_build(
            model=_make_mock_model(), prompt="hello world", regime="factual",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
        )
        assert profile.model.name == "mock-model"
        assert profile.model.backend == "hf"
        assert profile.prompt.text == "hello world"
        assert profile.prompt.regime == "factual"

    def test_profile_findings_carry_provenance_not_levels(self):
        """Findings is non-inferential: a float slope plus surrogate
        provenance. No level, no flag, no verdict, no summary sentence."""
        profile = _patched_build(
            model=_make_mock_model(), prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
        )
        f = profile.findings
        assert isinstance(f.similarity_trend_slope, float)
        assert f.surrogate_model_name is None
        assert f.output_distribution_surrogate_name is None
        assert set(f.model_dump()) == {
            "similarity_trend_slope",
            "surrogate_model_name",
            "output_distribution_surrogate_name",
        }

    def test_metric_bundle_has_distribution_and_semantic(self):
        profile = _patched_build(
            model=_make_mock_model(), prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
        )
        n_steps = len(profile.output_side.steps)
        assert len(profile.metrics.distribution) == n_steps
        assert len(profile.metrics.semantic) == n_steps

    def test_api_model_without_surrogate_zeroes_input_side(self):
        model = _make_mock_model()
        model.supports_teacher_forcing = False
        profile = _patched_build(
            model=model, prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
            surrogate_model=None,
        )
        assert profile.input_side.positions == []
        assert profile.input_side.mean_entropy == 0.0
        assert profile.input_side.mean_surprisal == 0.0

    def test_api_model_with_surrogate_computes_input_side(self):
        api_model = _make_mock_model()
        api_model.supports_teacher_forcing = False
        surrogate = _make_mock_model()
        surrogate.supports_teacher_forcing = True
        surrogate.name = "surrogate-model"

        profile = _patched_build(
            model=api_model, prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
            surrogate_model=surrogate,
        )
        assert len(profile.input_side.positions) > 0
        assert profile.input_side.mean_entropy > 0.0

    def test_surrogate_run_record_keeps_prompt_only_out_of_measurements(self):
        """End-to-end through the real builder on a mock closed backend.

        The surrogate teacher-forced the PROMPT, so the quantities it produced
        are not measurements of the target at any caveat level: the record puts
        them in `prompt_measurements`, naming the reference model, and leaves
        them out of `measurements`.
        """
        from hif.profile.signals import signals_record

        api_model = _make_mock_model()
        api_model.supports_teacher_forcing = False
        surrogate = _make_mock_model()
        surrogate.supports_teacher_forcing = True
        surrogate.name = "surrogate-model"

        profile = _patched_build(
            model=api_model, prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
            surrogate_model=surrogate,
        )
        record = signals_record(
            profile, model_name="closed-model", backend="anthropic",
            regime="test", seed=42, prompt="hello world",
        )
        block = record["prompt_measurements"]
        assert block["subject"] == "prompt-only"
        assert "prompt_surprisal_excess_bits" in block["values"]
        assert block["reference_models"]["prompt_surprisal_excess_bits"] == (
            "surrogate-model"
        )
        assert "prompt_surprisal_excess_bits" not in record["measurements"]
        assert not set(block["values"]) & set(record["measurements"])
        # The target's own output response is untouched by an input surrogate.
        assert "perturbation_jsd_bits" in record["measurements"]

    def test_full_access_run_record_has_no_prompt_block(self):
        """The [F] path is unchanged: nothing leaves `measurements`."""
        from hif.profile.signals import signals_record

        model = _make_mock_model()
        model.supports_teacher_forcing = True
        profile = _patched_build(
            model=model, prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
        )
        record = signals_record(
            profile, model_name="open-model", backend="hf", regime="test",
            seed=42, prompt="hello world",
        )
        assert "prompt_measurements" not in record
        assert "prompt_surprisal_excess_bits" in record["measurements"]

    def test_surrogate_not_used_when_model_supports_teacher_forcing(self):
        model = _make_mock_model()
        model.supports_teacher_forcing = True
        surrogate = _make_mock_model()
        surrogate.supports_teacher_forcing = True
        surrogate.name = "should-not-be-called"

        _patched_build(
            model=model, prompt="hello world", regime="test",
            config=_make_config(), embedder=_make_mock_embedder(), seed=42,
            surrogate_model=surrogate,
        )
        assert surrogate.forward_calls == 0
