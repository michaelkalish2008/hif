"""Unit tests for center diagnostics, trajectory analysis, and schema."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from hif.config import ClusterConfig, TrajectoryConfig
from hif.hourglass.center import compute_center_diagnostics
from hif.hourglass.output_side import OutputSideTrace
from hif.hourglass.trajectory import TrajectoryAnalysis, analyze_trajectory
from hif.models.base import StepRecord, TopKEntry
from hif.profile.schema import PromptRecord

from profile_helpers import (
    FakeEmbeddingModel,
    FakeModel,
    _make_center,
    _make_input_analysis,
    _make_output_trace,
    _make_profile,
    _make_step,
)


class TestCenterDiagnostics:
    def _make_embedder(self, dim: int = 8):
        return FakeEmbeddingModel(dim=dim, seed=1)

    def test_output_mean_entropy_is_recomputed_in_bits(self):
        """Output mean entropy is re-derived from the recorded top-K probs, in
        bits — not copied from the trace's own field and not rescaled."""
        input_analysis = _make_input_analysis(mean_entropy=5.0, max_entropy=16.0)
        output_trace = _make_output_trace()  # uniform over 5 tokens
        embedder = self._make_embedder()
        result = compute_center_diagnostics(
            input_analysis, output_trace, embedder, max_entropy=16.0
        )
        assert result.output_mean_entropy == pytest.approx(np.log2(5), abs=1e-6)

    def test_peaked_output_has_near_zero_entropy(self):
        input_analysis = _make_input_analysis(mean_entropy=5.0, max_entropy=16.0)
        peaked_probs = {0: 0.999, 1: 0.001}
        steps = [_make_step(i, peaked_probs) for i in range(2)]
        output_trace = OutputSideTrace(
            steps=steps,
            input_ids=[0, 1],
            generated_ids=[0, 1],
            prompt_text="hello world",
            model_name="mock",
            top_k=2,
            max_new_tokens=2,
            seed=42,
            mean_step_entropy=0.01,
        )
        embedder = self._make_embedder()
        result = compute_center_diagnostics(
            input_analysis, output_trace, embedder, max_entropy=16.0
        )
        assert result.output_mean_entropy < 0.05
        # A near-deterministic output is reported as the small number it is,
        # never bucketed into a "rigid"/"collapsed" verdict.
        assert not hasattr(result, "equilibrium_flag")

    def test_entropy_ratio_computed(self):
        input_analysis = _make_input_analysis(mean_entropy=4.0, max_entropy=16.0)
        output_trace = _make_output_trace()
        embedder = self._make_embedder()
        result = compute_center_diagnostics(input_analysis, output_trace, embedder, max_entropy=16.0)
        assert result.entropy_ratio > 0.0
        assert result.input_mean_entropy == pytest.approx(4.0)

    def test_prompt_output_cosine_distance_in_range(self):
        """Renamed from `semantic_drift`: it is a cosine distance between two
        embeddings, bounded to [0, 2] by definition — not evidence of drift."""
        input_analysis = _make_input_analysis()
        output_trace = _make_output_trace()
        embedder = self._make_embedder()
        result = compute_center_diagnostics(input_analysis, output_trace, embedder, max_entropy=16.0)
        assert 0.0 <= result.prompt_output_cosine_distance <= 2.0


class TestTrajectoryAnalysis:
    def _make_mock_model(self, vocab_size: int = 50) -> FakeModel:
        return FakeModel(vocab_size=vocab_size, context_length=512)

    def _make_embedder(self, dim: int = 16) -> FakeEmbeddingModel:
        return FakeEmbeddingModel(dim=dim, seed=2)

    def test_returns_trajectory_analysis(self):
        result = analyze_trajectory(
            self._make_mock_model(), [1, 2, 3, 4, 5], self._make_embedder(),
            TrajectoryConfig(n_branches=2, rollout_steps=3),
            ClusterConfig(method="kmeans", n_clusters=2), seed=0,
        )
        assert isinstance(result, TrajectoryAnalysis)

    def test_n_branches_at_most_B(self):
        config = TrajectoryConfig(n_branches=3, rollout_steps=2)
        result = analyze_trajectory(
            self._make_mock_model(), [1, 2, 3], self._make_embedder(),
            config, ClusterConfig(method="kmeans", n_clusters=3),
        )
        assert result.n_branches <= config.n_branches

    def test_convergence_profile_length_equals_rollout_steps(self):
        config = TrajectoryConfig(n_branches=2, rollout_steps=4)
        result = analyze_trajectory(
            self._make_mock_model(), [1, 2, 3], self._make_embedder(),
            config, ClusterConfig(method="kmeans", n_clusters=2),
        )
        assert len(result.convergence_profile) == config.rollout_steps

    def test_scores_in_unit_interval(self):
        result = analyze_trajectory(
            self._make_mock_model(), [1, 2, 3], self._make_embedder(),
            TrajectoryConfig(n_branches=2, rollout_steps=3),
            ClusterConfig(method="kmeans", n_clusters=2),
        )
        assert 0.0 <= result.persistence_score <= 1.0
        assert 0.0 <= result.explosion_score <= 1.0
        assert 0.0 <= result.convergence_score <= 1.0

    def test_start_step_equals_context_length(self):
        context_ids = [10, 20, 30, 40]
        result = analyze_trajectory(
            self._make_mock_model(), context_ids, self._make_embedder(),
            TrajectoryConfig(n_branches=2, rollout_steps=3),
            ClusterConfig(method="kmeans", n_clusters=2),
        )
        assert result.start_step == len(context_ids)

    def test_branches_have_generated_ids(self):
        config = TrajectoryConfig(n_branches=2, rollout_steps=3)
        result = analyze_trajectory(
            self._make_mock_model(), [1, 2, 3], self._make_embedder(),
            config, ClusterConfig(method="kmeans", n_clusters=2),
        )
        for branch in result.branches:
            assert len(branch.generated_ids) == config.rollout_steps
            assert isinstance(branch.final_text, str)

    def test_branch_field_present_and_well_formed(self):
        result = analyze_trajectory(
            self._make_mock_model(), [1, 2, 3, 4, 5], self._make_embedder(),
            TrajectoryConfig(n_branches=3, rollout_steps=3),
            ClusterConfig(method="kmeans", n_clusters=3), seed=0,
        )
        bf = result.branch_field
        if result.n_branches >= 2:
            assert bf is not None
            assert bf.n_branches == result.n_branches
            assert bf.mean_radius >= 0.0
            assert bf.max_radius >= bf.mean_radius >= 0.0
            assert bf.radius_variance >= 0.0
            assert bf.cluster_count >= 1
            # field_dispersion == 1 - trajectory_continuity (both from the same
            # branch embeddings), when continuity is available.
            if result.trajectory_continuity is not None:
                assert bf.field_dispersion == pytest.approx(
                    1.0 - result.trajectory_continuity, abs=1e-6
                )
        else:
            assert bf is None


class TestSchema:
    def test_prompt_record_hash(self):
        import hashlib
        text = "hello world"
        rec = PromptRecord.from_text(text, "factual", 2)
        assert rec.prompt_hash == hashlib.sha256(text.encode()).hexdigest()

    def test_behavioral_range_profile_schema_version(self):
        assert _make_profile().schema_version == "0.14.0"

    def test_behavioral_range_profile_created_at_is_datetime(self):
        assert isinstance(_make_profile().created_at, datetime)

    def test_model_dump_json_produces_valid_json(self):
        data = json.loads(_make_profile().model_dump_json())
        assert "schema_version" in data
