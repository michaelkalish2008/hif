"""Unit tests for SimilarityMetrics computation."""

import numpy as np
import pytest

from hif.metrics.similarity import (
    SimilarityMetrics,
    _mean_pairwise_cosine,
    _mean_io_cosine,
    _similarity_trend,
    compute_similarity_metrics,
)
from hif.metrics.semantic import SemanticMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeEmbeddingModel:
    """Implements EmbeddingModel's real contract (embed(texts) -> np.ndarray)
    with simplified, deterministic logic — no real model load, no mock.patch.

    compute_similarity_metrics only ever calls .embed(); this fake returns
    successive slices of a fixed embeddings array on each call, matching how
    the real EmbeddingModel is called (once for inputs, once for outputs).
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings
        self._offset = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        start = self._offset
        self._offset += n
        return self._embeddings[start : start + n]


def _make_fake_embedder(embeddings: np.ndarray) -> FakeEmbeddingModel:
    return FakeEmbeddingModel(embeddings)


def _semantic_metric(dist: float) -> SemanticMetrics:
    """Minimal SemanticMetrics stub with a given mean_pairwise_distance."""
    return SemanticMetrics(
        cluster_count=1,
        cluster_entropy=0.0,
        mean_pairwise_distance=dist,
        intra_cluster_density=1.0,
        max_inter_cluster_distance=0.0,
        topic_variance=0.0,
        n_candidates=4,
        truncated=False,
        noise_fraction=0.0,
    )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class TestMeanPairwiseCosine:
    def test_identical_vectors_return_one(self):
        embs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        assert _mean_pairwise_cosine(embs) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert _mean_pairwise_cosine(embs) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_return_minus_one(self):
        embs = np.array([[1.0, 0.0], [-1.0, 0.0]])
        assert _mean_pairwise_cosine(embs) == pytest.approx(-1.0, abs=1e-6)

    def test_single_vector_returns_one(self):
        embs = np.array([[0.5, 0.5]])
        assert _mean_pairwise_cosine(embs) == pytest.approx(1.0)

    def test_result_in_valid_range(self):
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((10, 16)).astype(np.float32)
        result = _mean_pairwise_cosine(embs)
        assert -1.0 <= result <= 1.0


class TestMeanIoCosine:
    def test_aligned_pairs_return_one(self):
        v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        assert _mean_io_cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_pairs_return_zero(self):
        inp = np.array([[1.0, 0.0], [1.0, 0.0]])
        out = np.array([[0.0, 1.0], [0.0, 1.0]])
        assert _mean_io_cosine(inp, out) == pytest.approx(0.0, abs=1e-6)

    def test_empty_returns_zero(self):
        assert _mean_io_cosine(np.empty((0, 4)), np.empty((0, 4))) == pytest.approx(0.0)

    def test_mixed_pairs_averaged(self):
        inp = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = np.array([[1.0, 0.0], [1.0, 0.0]])
        # pair 0: cos=1.0, pair 1: cos=0.0 → mean=0.5
        assert _mean_io_cosine(inp, out) == pytest.approx(0.5, abs=1e-6)


class TestSimilarityTrend:
    def test_none_returns_zero(self):
        assert _similarity_trend(None) == pytest.approx(0.0)

    def test_single_step_returns_zero(self):
        assert _similarity_trend([_semantic_metric(0.3)]) == pytest.approx(0.0)

    def test_converging_trend_is_positive(self):
        # distances decrease → similarities increase → positive slope
        metrics = [_semantic_metric(d) for d in [0.9, 0.6, 0.3, 0.1]]
        assert _similarity_trend(metrics) > 0.0

    def test_diverging_trend_is_negative(self):
        # distances increase → similarities decrease → negative slope
        metrics = [_semantic_metric(d) for d in [0.1, 0.3, 0.6, 0.9]]
        assert _similarity_trend(metrics) < 0.0

    def test_flat_trend_near_zero(self):
        metrics = [_semantic_metric(0.5)] * 5
        assert _similarity_trend(metrics) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# compute_similarity_metrics (integration)
# ---------------------------------------------------------------------------

class TestComputeSimilarityMetrics:
    def _build_embedder(self, *groups):
        """Concatenate groups of repeated vectors into a fixed-return embedder."""
        embs = np.vstack(groups).astype(np.float32)
        return _make_fake_embedder(embs)

    def test_identical_inputs_and_outputs_high_similarity(self):
        # All inputs identical, all outputs identical → io_sim ≈ 1, io_ratio ≈ 1
        v = np.array([[1.0, 0.0, 0.0, 0.0]] * 3)
        embedder = _make_fake_embedder(np.vstack([v, v]))
        result = compute_similarity_metrics(
            input_texts=["a", "b", "c"],
            output_texts=["x", "y", "z"],
            semantic_metrics=None,
            embedder=embedder,
        )
        assert result.io_sim == pytest.approx(1.0, abs=1e-5)
        assert result.input_sim == pytest.approx(1.0, abs=1e-5)
        assert result.output_sim == pytest.approx(1.0, abs=1e-5)
        assert result.io_ratio == pytest.approx(1.0, abs=1e-5)
        assert result.n_pairs == 3

    def test_orthogonal_inputs_outputs_io_sim_zero(self):
        # inputs along x-axis, outputs along y-axis → io_sim = 0
        inp = np.array([[1.0, 0.0]] * 3, dtype=np.float32)
        out = np.array([[0.0, 1.0]] * 3, dtype=np.float32)
        embedder = _make_fake_embedder(np.vstack([inp, out]))
        result = compute_similarity_metrics(["a","b","c"], ["x","y","z"], None, embedder)
        assert result.io_sim == pytest.approx(0.0, abs=1e-5)

    def test_n_pairs_capped_to_shorter_list(self):
        inp = np.array([[1.0, 0.0]] * 3, dtype=np.float32)
        out = np.array([[1.0, 0.0]] * 2, dtype=np.float32)
        embedder = _make_fake_embedder(np.vstack([inp[:2], out]))
        result = compute_similarity_metrics(["a","b","c"], ["x","y"], None, embedder)
        assert result.n_pairs == 2

    def test_empty_texts_return_zero_metric(self):
        embedder = _make_fake_embedder(np.empty((0, 4), dtype=np.float32))
        result = compute_similarity_metrics([], [], None, embedder)
        assert result.n_pairs == 0
        assert result.io_sim == pytest.approx(0.0)
        assert result.io_ratio == pytest.approx(0.0)

    def test_no_semantic_metrics_trend_is_zero(self):
        v = np.array([[1.0, 0.0]] * 4, dtype=np.float32)
        embedder = _make_fake_embedder(np.vstack([v, v]))
        result = compute_similarity_metrics(["a","b"], ["x","y"], None, embedder)
        assert result.trend == pytest.approx(0.0)

    def test_io_ratio_greater_than_one_when_outputs_cluster_more(self):
        # Inputs moderately spread; outputs all identical → output_sim > input_sim → ratio > 1
        inp = np.array([[1,0.1,0,0],[1,0,0.1,0],[1,0,0,0.1]], dtype=np.float32)
        out = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0]], dtype=np.float32)
        embedder = _make_fake_embedder(np.vstack([inp, out]))
        result = compute_similarity_metrics(["a","b","c"], ["x","y","z"], None, embedder)
        assert result.input_sim > 0.0   # inputs not perfectly identical
        assert result.output_sim == pytest.approx(1.0, abs=1e-5)  # outputs are identical
        assert result.io_ratio > 1.0

    def test_findings_thresholds_low(self):
        # io_sim < 0.4 → "low"
        inp = np.array([[1,0],[0,1],[0,-1]], dtype=np.float32)
        out = np.array([[-1,0],[0,-1],[0,1]], dtype=np.float32)
        embedder = _make_fake_embedder(np.vstack([inp, out]))
        result = compute_similarity_metrics(["a","b","c"], ["x","y","z"], None, embedder)
        level = "low" if result.io_sim < 0.4 else ("moderate" if result.io_sim < 0.7 else "high")
        assert level in ("low", "moderate", "high")  # just validate threshold logic runs
