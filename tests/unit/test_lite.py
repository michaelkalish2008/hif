"""The `--lite` stage budget: what it skips, and what it must leave alone.

The whole value of the flag rests on one property — a lite run and a full run
report the SAME number for every measurement lite does not disable. If skipping
the perturbation, trajectory, and semantic stages moved the entropy-side
readings, the flag would be a different measurement wearing the same key, and
the record has no field that would say so.

The second property is absence-not-zero: the measurements those stages feed must
be omitted from the record, never reported as a measured 0.0.
"""

from __future__ import annotations

import pytest

from hif.config import (
    ClusterConfig,
    ExposureConfig,
    GenerationConfig,
    ModelConfig,
    PerturbationConfig,
    RunConfig,
    SemanticConfig,
    TrajectoryConfig,
)
from hif.profile.builder import build_profile
from hif.profile.record import signals_record
from hif.perturbation.base import PerturbationGenerator, PerturbationResult

from profile_helpers import FakeEmbeddingModel, FakeModel


class FakeSynonymGenerator(PerturbationGenerator):
    """The real one needs NLTK/WordNet data, which has nothing to do with
    whether the stage ran."""

    name = "synonym"

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        return PerturbationResult(original=prompt, variants=["hi world"], generator="synonym")


@pytest.fixture(autouse=True)
def _fake_synonym_generator(monkeypatch):
    import hif.perturbation as perturbation_module

    monkeypatch.setitem(perturbation_module._RULE_TYPES, "synonym", FakeSynonymGenerator)


def _full_config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(name="mock-model", backend="hf"),
        generation=GenerationConfig(max_new_tokens=3, top_k=5),
        trajectory=TrajectoryConfig(n_branches=2, rollout_steps=3),
        perturbation=PerturbationConfig(n_variants=1, generators=["synonym"]),
        cluster=ClusterConfig(method="kmeans", n_clusters=2),
    )


def _lite_config() -> RunConfig:
    """The same overrides `--lite` applies in hif/cli.py::_run_single_profile."""
    config = _full_config()
    config.perturbation.generators = []
    config.perturbation.n_variants = 0
    config.trajectory.n_branches = 0
    config.semantic = SemanticConfig(enabled=False)
    config.exposure = ExposureConfig(enabled=False)
    config.semantic_field.enabled = False
    config.attention.enabled = False
    return config


def _build(config: RunConfig):
    return build_profile(
        model=FakeModel(vocab_size=100, context_length=512, name="mock-model"),
        prompt="hello world",
        regime="test",
        config=config,
        embedder=FakeEmbeddingModel(dim=16, seed=0),
        seed=42,
    )


def _measurements(config: RunConfig) -> dict:
    profile = _build(config)
    return signals_record(
        profile,
        model_name="mock-model",
        prompt="hello world",
        backend="hf",
        regime="test",
        seed=42,
    )["measurements"]


# Fed by the stages lite disables. Each must be absent from a lite record.
STAGE_DEPENDENT = [
    "perturbation_jsd_bits",
    "input_entropy_shift_bits",
    "input_entropy_std_bits",
    "io_correlation_r",
    "io_cosine_similarity",
    "branch_pairwise_cosine_similarity",
    "candidate_cluster_entropy_bits",
    "counterfactual_exposure_fraction",
]


class TestLiteSkipsStages:
    def test_no_perturbation_records(self):
        assert _build(_lite_config()).perturbations == []

    def test_no_trajectory_branches(self):
        assert _build(_lite_config()).trajectory.branches == []

    def test_no_semantic_metrics(self):
        assert _build(_lite_config()).metrics.semantic == []

    def test_full_run_produces_all_three(self):
        """Guards the test itself: if the full run stopped producing these, the
        absence assertions above would pass for the wrong reason."""
        profile = _build(_full_config())
        assert profile.perturbations
        assert profile.trajectory.branches
        assert profile.metrics.semantic


class TestLiteRecord:
    def test_stage_dependent_measurements_are_absent_not_zero(self):
        lite = _measurements(_lite_config())
        for key in STAGE_DEPENDENT:
            assert key not in lite, f"{key} should be omitted from a lite record, not reported"

    def test_surviving_measurements_are_bit_identical_to_a_full_run(self):
        """The property the flag lives or dies on."""
        full = _measurements(_full_config())
        lite = _measurements(_lite_config())
        assert lite, "a lite record must still carry measurements"
        for key, value in lite.items():
            assert key in full, f"lite produced {key}, which a full run does not"
            assert value == full[key], f"{key} moved between full and lite"

    def test_entropy_side_survives(self):
        """Not just 'something survived' — the output-side entropy readings are
        the reason to run lite at all."""
        assert "output_entropy_bits" in _measurements(_lite_config())
