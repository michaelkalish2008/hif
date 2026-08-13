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


def _record(config: RunConfig) -> dict:
    return signals_record(
        _build(config),
        model_name="mock-model",
        prompt="hello world",
        backend="hf",
        regime="test",
        seed=42,
    )


def _measurements(config: RunConfig) -> dict:
    return _record(config)["measurements"]


# Fed by the stages lite disables. Each must be absent from a lite record.
#
# LIVE KEYS ONLY. This list carried five keys that hif-v4 had already cut, so
# five of its eight absence assertions passed for the wrong reason — a
# measurement that cannot be emitted at all is trivially absent from a lite
# record, and the assertion said nothing about lite. The guard below keeps
# that from recurring.
STAGE_DEPENDENT = [
    "perturbation_jsd_bits",
    "input_entropy_shift_bits",
    "input_entropy_std_bits",
    "io_cosine_similarity",
]


def test_stage_dependent_names_only_live_measurements():
    """Every key above must still be in the registry.

    Otherwise the absence assertions that read this list go vacuous silently:
    they would pass on a key no run can produce, which is not evidence that
    --lite skipped a stage.
    """
    from hif.profile.registry import MEASUREMENT_BY_KEY

    dead = [k for k in STAGE_DEPENDENT if k not in MEASUREMENT_BY_KEY]
    assert not dead, f"STAGE_DEPENDENT names retired measurements: {dead}"


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


class TestLiteHash:
    """The identifier has to separate the two runs the flag separates.

    The hash reads as a run id: it names the `--output-dir` artifacts
    (`profile_<hash>.json`, `profile_<hash>_technical.md`, `plots/<hash>/`),
    it names the `--trace` artifact, and it is the `hash` field of the record.
    While it covered only (model, prompt, seed), a lite run and a full run of
    the same prompt shared it — six measurements and two under one id, each
    overwriting the other's artifacts, and nothing in the id to say which one
    it was.
    """

    def test_lite_and_full_records_do_not_share_a_hash(self):
        full = _record(_full_config())
        lite = _record(_lite_config())
        assert full["hash"] != lite["hash"], (
            "a lite run and a full run of the same model and prompt share an "
            "identifier while carrying different measurement sets: "
            f"{sorted(full['measurements'])} vs {sorted(lite['measurements'])}"
        )

    def test_the_two_runs_really_do_differ(self):
        """Guards the test above: if the measurement sets ever converged, the
        hash assertion would be asking for a distinction that is not one."""
        full = _record(_full_config())
        lite = _record(_lite_config())
        assert set(lite["measurements"]) < set(full["measurements"])

    def test_same_budget_still_hashes_the_same(self):
        """The fix must not make the hash a nonce — a rerun of the same run is
        the same run, and hash-addressed artifacts depend on it."""
        assert _record(_full_config())["hash"] == _record(_full_config())["hash"]

    def test_hash_is_over_the_resolved_budget_not_the_flag(self):
        """`--lite` has no independent existence in the run: it is a ceiling
        applied to the config. A config file that switches the same stages off
        by hand IS a lite run and must hash as one — and, conversely, a full
        run must not collide with it."""
        from hif.profile.record import profile_hash

        by_hand = _full_config()
        by_hand.perturbation.generators = []
        by_hand.perturbation.n_variants = 0
        by_hand.trajectory.n_branches = 0
        by_hand.semantic.enabled = False
        by_hand.exposure.enabled = False
        by_hand.semantic_field.enabled = False
        by_hand.attention.enabled = False

        assert profile_hash("m", "p", 42, by_hand) == profile_hash(
            "m", "p", 42, _lite_config()
        )
        assert profile_hash("m", "p", 42, by_hand) != profile_hash(
            "m", "p", 42, _full_config()
        )


class TestStageBudgetCoverage:
    """What else the identifier has to separate.

    `--lite` is the loudest case, not the only one. Each knob below changes
    which measurements a run can emit, so two runs differing in it are two
    runs.
    """

    @staticmethod
    def _h(config: RunConfig) -> str:
        from hif.profile.record import profile_hash

        return profile_hash("m", "p", 42, config)

    def test_acquisition_observational_differs_from_full(self):
        """`--acquisition observational` (hif/cli/_run.py) drops the
        perturbation and trajectory stages but keeps the per-step ones, so it
        is also distinct from `--lite`, which drops both sets."""
        obs = _full_config()
        obs.perturbation.generators = []
        obs.perturbation.n_variants = 0
        obs.perturbation.variants_file = None
        obs.trajectory.n_branches = 0

        assert self._h(obs) != self._h(_full_config())
        assert self._h(obs) != self._h(_lite_config())

    def test_acquisition_synthesized_input_differs_from_full(self):
        """The variants are authored and teacher-forced but never generated
        from: two of the four perturbation measurements go absent."""
        synth = _full_config()
        synth.perturbation.elicit_variant_outputs = False
        synth.trajectory.n_branches = 0

        assert self._h(synth) != self._h(_full_config())

    def test_variant_count_and_generator_set_are_covered(self):
        more = _full_config()
        more.perturbation.n_variants = 5
        other = _full_config()
        other.perturbation.generators = ["tone"]

        assert self._h(more) != self._h(_full_config())
        assert self._h(other) != self._h(_full_config())

    def test_unknown_budget_is_not_the_default_budget(self):
        """A profile that carries no config hashes the legacy key. "I do not
        know what this run did" is a different claim from "it ran the
        defaults", and must not be recorded as the same run."""
        assert self._h(None) != self._h(RunConfig())
