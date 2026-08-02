"""Unit tests for the traceability opt-in (raw trace capture, schema 0.7.0).

Compute-and-discard is the default: profiles carry no raw perturbation-variant
or trajectory-branch traces unless config.traceability.enabled — the sanctioned
exception under which the artifact captures them so field descriptors,
JS-centroids, and branch fields are retroactively recomputable.
"""

from __future__ import annotations

import json

import pytest

from hif.config import (
    ClusterConfig,
    GenerationConfig,
    ModelConfig,
    PerturbationConfig,
    RunConfig,
    TraceabilityConfig,
    TrajectoryConfig,
)
from hif.profile.builder import build_profile
from hif.profile.schema import BehavioralRangeProfile, RawTraces, VariantRawTrace
from hif.perturbation.base import PerturbationGenerator, PerturbationResult

from profile_helpers import FakeEmbeddingModel, FakeModel


N_VARIANTS = 2


def _make_mock_model(vocab_size: int = 100) -> FakeModel:
    return FakeModel(vocab_size=vocab_size, context_length=512, name="mock-model")


def _make_mock_embedder(dim: int = 16) -> FakeEmbeddingModel:
    return FakeEmbeddingModel(dim=dim, seed=0)


class FakeSynonymGenerator(PerturbationGenerator):
    """Stands in for SynonymGenerator (real one needs NLTK/WordNet data).
    Honors n_variants so the count assertion is meaningful."""

    name = "synonym"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        return PerturbationResult(
            original=prompt,
            variants=[f"hi world {i}" for i in range(n_variants)],
            generator="synonym",
        )


class FakeToneGenerator(FakeSynonymGenerator):
    name = "tone"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        return PerturbationResult(
            original=prompt,
            variants=[f"greetings world {i}" for i in range(n_variants)],
            generator="tone",
        )


@pytest.fixture(autouse=True)
def _fake_generators(monkeypatch):
    """Inject fake generators into the registry build_profile resolves from —
    direct module-global injection, matching this project's established
    pattern (see tests/README.md)."""
    import hif.perturbation as perturbation_module

    monkeypatch.setitem(perturbation_module._RULE_TYPES, "synonym", FakeSynonymGenerator)
    monkeypatch.setitem(perturbation_module._RULE_TYPES, "tone", FakeToneGenerator)


def _make_config(traceability_enabled: bool = False) -> RunConfig:
    return RunConfig(
        model=ModelConfig(name="mock-model", backend="hf"),
        generation=GenerationConfig(max_new_tokens=3, top_k=5),
        trajectory=TrajectoryConfig(n_branches=2, rollout_steps=3),
        perturbation=PerturbationConfig(
            n_variants=N_VARIANTS, generators=["synonym", "tone"]
        ),
        cluster=ClusterConfig(method="kmeans", n_clusters=2),
        traceability=TraceabilityConfig(enabled=traceability_enabled),
    )


def _build(traceability_enabled: bool = False) -> BehavioralRangeProfile:
    return build_profile(
        model=_make_mock_model(),
        prompt="hello world",
        regime="test",
        config=_make_config(traceability_enabled),
        embedder=_make_mock_embedder(),
        seed=42,
    )


class TestTraceabilityDisabled:
    """Default: compute-and-discard — no raw traces on the artifact."""

    def test_default_config_is_disabled(self):
        assert RunConfig().traceability.enabled is False
        assert RunConfig().traceability.profiles_dir is None

    def test_profile_has_no_raw_traces(self):
        profile = _build(traceability_enabled=False)
        assert profile.raw_traces is None

    def test_json_round_trip(self):
        profile = _build(traceability_enabled=False)
        raw = profile.model_dump_json()
        reparsed = BehavioralRangeProfile.model_validate_json(raw)
        assert reparsed.raw_traces is None
        assert json.loads(raw)["raw_traces"] is None

    def test_no_behavior_change_vs_enabled(self):
        """Enabling traceability must not perturb any computed value — the
        two profiles differ only in raw_traces and the config flag."""
        disabled = _build(traceability_enabled=False)
        enabled = _build(traceability_enabled=True)
        d = disabled.model_dump(mode="json", exclude={"created_at"})
        e = enabled.model_dump(mode="json", exclude={"created_at"})
        d.pop("raw_traces")
        e.pop("raw_traces")
        d["config"].pop("traceability")
        e["config"].pop("traceability")
        assert d == e


class TestTraceabilityEnabled:
    def test_raw_traces_present(self):
        profile = _build(traceability_enabled=True)
        assert isinstance(profile.raw_traces, RawTraces)

    def test_variant_trace_count_matches_variants_times_generators(self):
        profile = _build(traceability_enabled=True)
        n_generators = len(profile.config.perturbation.generators)
        assert len(profile.raw_traces.variant_traces) == N_VARIANTS * n_generators

    def test_variant_traces_keyed_by_generator_and_index(self):
        profile = _build(traceability_enabled=True)
        keys = {
            (vt.generator, vt.variant_index)
            for vt in profile.raw_traces.variant_traces
        }
        expected = {
            (gen, i)
            for gen in ("synonym", "tone")
            for i in range(N_VARIANTS)
        }
        assert keys == expected

    def test_variant_traces_have_per_step_topk(self):
        profile = _build(traceability_enabled=True)
        for vt in profile.raw_traces.variant_traces:
            assert isinstance(vt, VariantRawTrace)
            assert len(vt.trace.steps) > 0
            for step in vt.trace.steps:
                assert len(step.topk) > 0
                for entry in step.topk:
                    assert isinstance(entry.token_id, int)
                    assert 0.0 <= entry.prob <= 1.0

    def test_branch_traces_present_when_trajectory_ran(self):
        profile = _build(traceability_enabled=True)
        # FakeModel supports teacher forcing, so the trajectory stage ran.
        assert profile.trajectory.n_branches > 0
        assert len(profile.raw_traces.branch_traces) == profile.trajectory.n_branches
        for branch in profile.raw_traces.branch_traces:
            assert len(branch.steps) > 0
            for step in branch.steps:
                assert len(step.topk) > 0

    def test_json_round_trip_preserves_raw_traces(self):
        profile = _build(traceability_enabled=True)
        raw = profile.model_dump_json()
        reparsed = BehavioralRangeProfile.model_validate_json(raw)
        assert reparsed.raw_traces is not None
        assert reparsed.raw_traces.model_dump() == profile.raw_traces.model_dump()


class TestSchemaBackwardCompat:
    def test_old_profile_json_without_raw_traces_validates(self):
        """A pre-0.7.0 profile on disk has neither raw_traces nor
        config.traceability — both must default cleanly."""
        profile = _build(traceability_enabled=False)
        data = profile.model_dump(mode="json")
        data["schema_version"] = "0.6.0"
        data.pop("raw_traces")
        data["config"].pop("traceability")

        loaded = BehavioralRangeProfile.model_validate(data)
        assert loaded.raw_traces is None
        assert loaded.config.traceability.enabled is False
