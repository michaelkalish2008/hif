"""Unit tests for hif.analysis.exposure (the Exposure ◇ instrument).

Uses real schema objects (TopKEntry, StepRecord) and a Fake embedder — no
mock.patch, no MagicMock. A bug in the real schema's field names or the
analyzer's attribute access is caught here; MagicMock would silently accept
any attribute regardless of whether it exists on the real type.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hif.models.base import StepRecord, TopKEntry


# ---------------------------------------------------------------------------
# Helpers — build minimal fixtures
# ---------------------------------------------------------------------------


def _make_topk(tokens_probs: list[tuple[str, float]]) -> list[TopKEntry]:
    """Return real TopKEntry objects, deterministic and schema-checked."""
    return [
        TopKEntry(
            token_id=i,
            token_str=tok,
            logit=-1.0,
            logprob=float(np.log(prob + 1e-10)),
            prob=prob,
        )
        for i, (tok, prob) in enumerate(tokens_probs)
    ]


def _make_output_step(step: int, selected: str, topk_pairs: list[tuple[str, float]]) -> StepRecord:
    """Return a real StepRecord."""
    return StepRecord(
        step=step,
        selected_token_id=0,
        selected_token_str=selected,
        topk=_make_topk(topk_pairs),
    )


def _make_semantic_metrics(
    cluster_count: int = 3,
    mean_pairwise_distance: float = 0.5,
    intra_cluster_density: float = 0.4,
    embeddings_2d: list[list[float]] | None = None,
):
    """Return a real SemanticMetrics."""
    from hif.metrics.semantic import SemanticMetrics

    return SemanticMetrics(
        cluster_count=cluster_count,
        cluster_entropy=1.0,
        mean_pairwise_distance=mean_pairwise_distance,
        max_inter_cluster_distance=0.8,
        intra_cluster_density=intra_cluster_density,
        topic_variance=0.1,
        n_candidates=5,
        truncated=False,
        embeddings_2d=embeddings_2d or [],
        projection_method="pca",
    )


class FakeOutputTrace:
    """Implements the subset of OutputSideTrace's contract ExposureAnalyzer
    reads (.steps, .generated_ids) — a full OutputSideTrace needs many unrelated
    fields (model_name, seed, ...) irrelevant to these tests."""

    def __init__(self, steps: list[StepRecord]) -> None:
        self.steps = steps
        self.generated_ids = list(range(len(steps)))


def _make_output_trace(steps: list[StepRecord]) -> FakeOutputTrace:
    return FakeOutputTrace(steps)


class FakeEmbeddingModel:
    """Implements EmbeddingModel's real contract (embed(texts) -> np.ndarray)
    with simplified, deterministic logic — random unit vectors, seeded."""

    def __init__(self, n_dim: int = 8, seed: int = 0) -> None:
        self._n_dim = n_dim
        self._rng = np.random.default_rng(seed)

    def embed(self, strings: list[str]) -> np.ndarray:
        vecs = self._rng.standard_normal((len(strings), self._n_dim))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-10)


def _make_embedder(n_dim: int = 8) -> FakeEmbeddingModel:
    return FakeEmbeddingModel(n_dim=n_dim)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestExposureCandidateSchema:
    def test_fields_present(self):
        from hif.analysis.exposure import ExposureCandidate

        c = ExposureCandidate(
            step=3,
            selected_token=" the",
            selected_prob=0.4,
            divergent_token=" a",
            divergent_prob=0.1,
            prob_rank=2,
            semantic_distance=0.65,
            cloud_phenomenon="diffusion",
            cloud_position_2d=[0.3, -0.2],
        )
        assert c.step == 3
        assert c.selected_token == " the"
        assert c.divergent_token == " a"
        assert c.semantic_distance == pytest.approx(0.65)
        assert c.cloud_phenomenon == "diffusion"
        assert c.cloud_position_2d == [0.3, -0.2]

    def test_empty_cloud_position_is_valid(self):
        from hif.analysis.exposure import ExposureCandidate

        c = ExposureCandidate(
            step=0,
            selected_token="hello",
            selected_prob=0.5,
            divergent_token="hi",
            divergent_prob=0.05,
            prob_rank=3,
            semantic_distance=0.3,
            cloud_phenomenon="convergence",
            cloud_position_2d=[],
        )
        assert c.cloud_position_2d == []


class TestExposureProfileSchema:
    def test_fields_present(self):
        from hif.analysis.exposure import ExposureCandidate, ExposureProfile

        c = ExposureCandidate(
            step=0, selected_token="a", selected_prob=0.5,
            divergent_token="b", divergent_prob=0.1,
            prob_rank=1, semantic_distance=0.4,
            cloud_phenomenon="diffusion", cloud_position_2d=[],
        )
        hp = ExposureProfile(
            candidates=[c],
            exposed_steps=[0],
            mean_semantic_distance=0.4,
            diffusion_zone_ratio=1.0,
        )
        assert len(hp.candidates) == 1
        assert hp.exposed_steps == [0]
        assert hp.diffusion_zone_ratio == pytest.approx(1.0)

    def test_empty_profile_is_valid(self):
        from hif.analysis.exposure import ExposureProfile

        hp = ExposureProfile(
            candidates=[],
            exposed_steps=[],
            mean_semantic_distance=0.0,
            diffusion_zone_ratio=0.0,
        )
        assert hp.candidates == []


# ---------------------------------------------------------------------------
# Cloud phenomenon classifier
# ---------------------------------------------------------------------------


class TestCloudPhenomenon:
    def test_convergence(self):
        from hif.analysis.exposure import _cloud_phenomenon

        sem = _make_semantic_metrics(cluster_count=1, mean_pairwise_distance=0.2)
        assert _cloud_phenomenon(sem) == "convergence"

    def test_clustering(self):
        from hif.analysis.exposure import _cloud_phenomenon

        sem = _make_semantic_metrics(cluster_count=5, intra_cluster_density=0.8)
        assert _cloud_phenomenon(sem) == "clustering"

    def test_divergence(self):
        from hif.analysis.exposure import _cloud_phenomenon

        sem = _make_semantic_metrics(cluster_count=2, mean_pairwise_distance=0.6)
        assert _cloud_phenomenon(sem) == "divergence"

    def test_diffusion(self):
        from hif.analysis.exposure import _cloud_phenomenon

        sem = _make_semantic_metrics(cluster_count=3, mean_pairwise_distance=0.5)
        assert _cloud_phenomenon(sem) == "diffusion"

    def test_zero_cluster_count_is_diffusion(self):
        from hif.analysis.exposure import _cloud_phenomenon

        sem = _make_semantic_metrics(cluster_count=0)
        assert _cloud_phenomenon(sem) == "diffusion"


# ---------------------------------------------------------------------------
# ExposureAnalyzer
# ---------------------------------------------------------------------------


class TestExposureAnalyzer:
    def _run(self, steps, sem_metrics=None, min_prob=0.01, distance_threshold=0.3):
        from hif.analysis.exposure import ExposureAnalyzer

        if sem_metrics is None:
            sem_metrics = [_make_semantic_metrics() for _ in steps]

        trace = _make_output_trace(steps)
        embedder = _make_embedder()
        analyzer = ExposureAnalyzer(embedder=embedder, min_prob=min_prob)
        return analyzer.analyze(trace, sem_metrics, distance_threshold=distance_threshold)

    def test_returns_exposure_profile(self):
        from hif.analysis.exposure import ExposureProfile

        steps = [
            _make_output_step(0, "the", [("the", 0.5), ("a", 0.3), ("an", 0.1)]),
            _make_output_step(1, "cat", [("cat", 0.6), ("dog", 0.2), ("rat", 0.1)]),
        ]
        result = self._run(steps)
        assert isinstance(result, ExposureProfile)

    def test_empty_trace_returns_empty_profile(self):
        result = self._run([])
        assert result.candidates == []
        assert result.mean_semantic_distance == 0.0

    def test_single_token_topk_produces_no_candidates(self):
        # Only the selected token in topk — no alternatives
        steps = [_make_output_step(0, "the", [("the", 1.0)])]
        result = self._run(steps)
        assert result.candidates == []

    def test_candidates_count_at_most_one_per_step(self):
        steps = [
            _make_output_step(0, "hello", [("hello", 0.5), ("hi", 0.3), ("hey", 0.1)]),
            _make_output_step(1, "world", [("world", 0.6), ("earth", 0.2)]),
            _make_output_step(2, "foo", [("foo", 0.7), ("bar", 0.2)]),
        ]
        result = self._run(steps)
        assert len(result.candidates) <= len(steps)

    def test_candidate_steps_are_valid_indices(self):
        steps = [
            _make_output_step(i, f"tok{i}", [(f"tok{i}", 0.5), (f"alt{i}", 0.3)])
            for i in range(4)
        ]
        result = self._run(steps)
        for c in result.candidates:
            assert 0 <= c.step < len(steps)

    def test_semantic_distance_in_range(self):
        steps = [
            _make_output_step(0, "the", [("the", 0.5), ("a", 0.3), ("an", 0.1)]),
        ]
        result = self._run(steps)
        for c in result.candidates:
            assert 0.0 <= c.semantic_distance <= 2.0

    def test_exposed_steps_are_subset_of_candidate_steps(self):
        steps = [
            _make_output_step(i, f"t{i}", [(f"t{i}", 0.5), (f"h{i}", 0.3)])
            for i in range(6)
        ]
        result = self._run(steps)
        candidate_steps = {c.step for c in result.candidates}
        for s in result.exposed_steps:
            assert s in candidate_steps

    def test_diffusion_ratio_in_unit_interval(self):
        steps = [
            _make_output_step(i, f"t{i}", [(f"t{i}", 0.5), (f"h{i}", 0.3)])
            for i in range(4)
        ]
        result = self._run(steps)
        assert 0.0 <= result.diffusion_zone_ratio <= 1.0

    def test_min_prob_filter_excludes_low_prob_candidates(self):
        # Only one alternative, below the min_prob threshold
        steps = [
            _make_output_step(0, "hello", [("hello", 0.5), ("bye", 0.001)])
        ]
        result = self._run(steps, min_prob=0.01)
        # "bye" (prob=0.001) should be excluded
        assert all(c.divergent_token != "bye" for c in result.candidates)

    def test_cloud_position_from_embeddings_2d(self):
        embeddings_2d = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        sem = [_make_semantic_metrics(embeddings_2d=embeddings_2d)]
        steps = [
            _make_output_step(0, "hello", [("hello", 0.5), ("hi", 0.3), ("hey", 0.1)])
        ]
        result = self._run(steps, sem_metrics=sem)
        for c in result.candidates:
            # cloud_position_2d should come from embeddings_2d at the candidate's rank
            if c.cloud_position_2d:
                assert len(c.cloud_position_2d) == 2


# ---------------------------------------------------------------------------
# The Exposure chart
# ---------------------------------------------------------------------------


def _make_minimal_profile_for_exposure(exposure_data):
    """Build a minimal BehavioralRangeProfile with exposure data."""
    import numpy as np

    from hif.config import (
        GenerationConfig,
        ModelConfig,
        PerturbationConfig,
        RunConfig,
        TrajectoryConfig,
    )
    from hif.hourglass.center import CenterDiagnostics
    from hif.hourglass.input_side import InputSideAnalysis, PositionRecord
    from hif.hourglass.output_side import OutputSideTrace
    from hif.hourglass.trajectory import Branch, BranchConvergence, TrajectoryAnalysis
    from hif.metrics.distribution import DistributionMetrics
    from hif.metrics.semantic import SemanticMetrics
    from hif.metrics.sensitivity import SensitivityMetrics, StepSensitivity
    from hif.metrics.stability import PerturbationResponse
    from hif.models.base import StepRecord, TopKEntry
    from hif.profile.builder import generate_findings
    from hif.profile.schema import (
        BehavioralRangeProfile,
        MetricBundle,
        ModelIdentity,
        PerturbationRecord,
        PromptRecord,
    )

    def _topk(n=3):
        prob = 1.0 / n
        return [
            TopKEntry(token_id=i, token_str=f"tok{i}", logit=-float(i+1),
                      logprob=float(np.log(prob + 1e-12)), prob=prob)
            for i in range(n)
        ]

    def _step(s):
        return StepRecord(step=s, selected_token_id=0, selected_token_str="tok0", topk=_topk())

    pos = PositionRecord(
        position=1, token_id=1, token_str="hello", surprisal=3.0, entropy=5.0,
        top_k_alternatives=[{"token_id": 2, "token_str": "world", "prob": 0.1}],
    )
    input_analysis = InputSideAnalysis(
        positions=[pos], prompt_token_ids=[0, 1], prompt_text="hello world",
        mean_surprisal=3.0, mean_entropy=5.0, max_entropy=16.0,
    )
    output_trace = OutputSideTrace(
        steps=[_step(i) for i in range(3)],
        input_ids=[0, 1], generated_ids=[0, 1, 2],
        prompt_text="hello world", model_name="mock-model",
        top_k=3, max_new_tokens=3, seed=42, mean_step_entropy=3.0,
    )
    center = CenterDiagnostics(
        input_mean_entropy=5.0, output_mean_entropy=3.0,
        entropy_ratio=1.0, prompt_output_cosine_distance=0.2,
    )
    branch = Branch(
        cluster_id=0, representative_token_ids=[0],
        generated_ids=[0, 1, 2], steps=[_step(i) for i in range(3)],
        final_text="tok0 tok0 tok0",
    )
    trajectory = TrajectoryAnalysis(
        start_step=2, n_branches=1, rollout_steps=2,
        branches=[branch],
        convergence_profile=[BranchConvergence(step=0, n_remaining_clusters=1)],
        persistence_score=1.0, explosion_score=0.0, convergence_score=0.0, initial_n_clusters=1,
    )
    sens = SensitivityMetrics(
        perturbation_generator="synonym", perturbed_prompt="hi world", original_prompt="hello world",
        step_sensitivities=[StepSensitivity(step=0, js_divergence=0.05, kl_divergence=0.1, entropy_delta=0.0, nucleus_overlap_p90=1.0)],
        mean_js_divergence=0.05, mean_kl_divergence=0.1, mean_entropy_delta=0.0, output_entropy_delta=0.0,
        mean_nucleus_stability_p90=1.0,
    )
    dm = DistributionMetrics(
        entropy_bits=3.0, logit_margin=2.0, topk_cumulative_mass=0.9,
        nucleus_effective_support_size=8.0, tail_weight=0.05, truncated=True,
        nucleus_fraction={"p90": 0.002, "p95": 0.004},
        nucleus_entropy_bits=2.5,
    )
    sm = SemanticMetrics(
        cluster_count=3, cluster_entropy=1.0, mean_pairwise_distance=0.5,
        max_inter_cluster_distance=0.8, intra_cluster_density=0.4, topic_variance=0.1,
        n_candidates=3, truncated=False,
    )
    stability = PerturbationResponse(input_entropy_shift_bits=0.4, perturbation_jsd_bits=0.1, input_output_correlation=0.0, n_perturbations=1)
    metric_bundle = MetricBundle(distribution=[dm] * 3, semantic=[sm] * 3, sensitivity=[sens], stability=stability)
    findings = generate_findings(input_analysis, output_trace, center, metric_bundle)

    return BehavioralRangeProfile(
        model=ModelIdentity(name="mock", backend="hf", vocab_size=50257, context_length=1024),
        prompt=PromptRecord.from_text("hello world", "ordinary_conversation", 2),
        input_side=input_analysis,
        output_side=output_trace,
        center=center,
        trajectory=trajectory,
        perturbations=[PerturbationRecord(generator="synonym", variants=["hi world"], sensitivity=[sens])],
        metrics=metric_bundle,
        findings=findings,
        config=RunConfig(
            model=ModelConfig(name="mock", backend="hf"),
            generation=GenerationConfig(max_new_tokens=3, top_k=3),
            trajectory=TrajectoryConfig(n_branches=1, rollout_steps=2),
            perturbation=PerturbationConfig(n_variants=1, generators=["synonym"]),
        ),
        exposure=exposure_data,
    )


class TestOldVocabularyAliases:
    """Archived profile JSON (pre-0.10.0) carries the retired names
    (hallucinated_token/-_prob, high_risk_steps, config.hallucination).
    Validation aliases accept them on read; everything newly emitted carries
    only the exposure vocabulary. These tests are the contract that `hif
    render` keeps loading the archived corpus.
    """

    def test_exposure_profile_loads_old_json_keys(self):
        from hif.analysis.exposure import ExposureProfile

        old = {
            "candidates": [
                {
                    "step": 0,
                    "selected_token": " the",
                    "selected_prob": 0.4,
                    "hallucinated_token": " a",
                    "hallucinated_prob": 0.1,
                    "prob_rank": 2,
                    "semantic_distance": 0.65,
                    "cloud_phenomenon": "diffusion",
                    "cloud_position_2d": [],
                }
            ],
            "high_risk_steps": [0],
            "mean_semantic_distance": 0.65,
            "diffusion_zone_ratio": 1.0,
            "exposure": 1.0,
        }
        hp = ExposureProfile.model_validate(old)
        assert hp.exposed_steps == [0]
        assert hp.candidates[0].divergent_token == " a"
        assert hp.candidates[0].divergent_prob == pytest.approx(0.1)

    def test_new_dumps_carry_only_the_exposure_vocabulary(self):
        from hif.analysis.exposure import ExposureCandidate, ExposureProfile

        hp = ExposureProfile(
            candidates=[
                ExposureCandidate(
                    step=0, selected_token="a", selected_prob=0.5,
                    divergent_token="b", divergent_prob=0.1,
                    prob_rank=1, semantic_distance=0.4,
                    cloud_phenomenon="diffusion", cloud_position_2d=[],
                )
            ],
            exposed_steps=[0], mean_semantic_distance=0.4,
            diffusion_zone_ratio=1.0,
        )
        dumped = hp.model_dump_json()
        assert "hallucinated" not in dumped
        assert "high_risk" not in dumped
        assert "divergent_token" in dumped and "exposed_steps" in dumped

    def test_run_config_accepts_the_old_hallucination_table(self):
        from hif.config import RunConfig

        cfg = RunConfig.model_validate(
            {"hallucination": {"enabled": False, "min_prob": 0.05}}
        )
        assert cfg.exposure.enabled is False
        assert cfg.exposure.min_prob == pytest.approx(0.05)
        assert "hallucination" not in cfg.model_dump()

    def test_full_profile_json_roundtrip_from_old_vocabulary(self):
        """A profile dumped today, rewritten to the archived vocabulary, must
        validate — the minimal stand-in for the 2.6 GB archived corpus."""
        import json

        from hif.analysis.exposure import ExposureCandidate, ExposureProfile
        from hif.profile.schema import BehavioralRangeProfile

        hp = ExposureProfile(
            candidates=[
                ExposureCandidate(
                    step=0, selected_token="a", selected_prob=0.5,
                    divergent_token="b", divergent_prob=0.1,
                    prob_rank=1, semantic_distance=0.4,
                    cloud_phenomenon="diffusion", cloud_position_2d=[],
                )
            ],
            exposed_steps=[0], mean_semantic_distance=0.4,
            diffusion_zone_ratio=1.0, exposure=1.0,
        )
        profile = _make_minimal_profile_for_exposure(hp)
        data = json.loads(profile.model_dump_json())
        # Rewrite to the archived shape: old exposure keys, old config table.
        exp = data.pop("exposure")
        exp["high_risk_steps"] = exp.pop("exposed_steps")
        for c in exp["candidates"]:
            c["hallucinated_token"] = c.pop("divergent_token")
            c["hallucinated_prob"] = c.pop("divergent_prob")
        data["hallucination"] = exp
        data["config"]["hallucination"] = data["config"].pop("exposure")
        loaded = BehavioralRangeProfile.model_validate(data)
        assert loaded.config.exposure.enabled is True
        # profile.exposure is lazy-typed Any, so the old-vocabulary block
        # survives as a dict under its aliased keys. Since hif-v4 nothing in
        # hif consumes it — the chart and the measurement were cut — so the
        # contract this pins is validation alone: the archived corpus loads.
        assert loaded.exposure["candidates"][0]["hallucinated_token"] == "b"
