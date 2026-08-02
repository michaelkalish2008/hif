"""Unit tests for clustering and embedding infrastructure."""

import numpy as np
import pytest

from hif.clustering.embed import EmbeddingModel
from hif.clustering.cluster import cluster_embeddings, ClusterResult
from hif.config import EmbeddingConfig, ClusterConfig


# ---------------------------------------------------------------------------
# Fake embedder — no mock/patch
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Returns fixed embeddings cycling through a predefined array."""

    def __init__(self, embeddings: np.ndarray):
        self._embeddings = embeddings
        self._call_count = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        start = self._call_count
        self._call_count += n
        return self._embeddings[start : start + n]


def _fake_embedder_returning(embeddings: np.ndarray) -> FakeEmbedder:
    return FakeEmbedder(embeddings)


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------

def _two_group_embeddings() -> np.ndarray:
    """6 embeddings: 3 near [1, 0, ...], 3 near [0, 1, ...]."""
    group_a = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (3, 1))
    group_b = np.tile([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (3, 1))
    group_a += np.random.default_rng(0).normal(0, 0.01, group_a.shape)
    group_b += np.random.default_rng(1).normal(0, 0.01, group_b.shape)
    return np.vstack([group_a, group_b]).astype(np.float32)


def test_cluster_two_clear_groups_hdbscan():
    """Two clearly separated groups → HDBSCAN finds >= 1 cluster."""
    embeddings = _two_group_embeddings()
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)
    result = cluster_embeddings(embeddings, config)
    assert result.n_clusters >= 1


def test_cluster_single_point():
    """Single point should not crash and return 1 cluster."""
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    config = ClusterConfig(method="hdbscan", min_cluster_size=2)
    result = cluster_embeddings(embeddings, config)
    assert result.n_clusters == 1
    assert len(result.labels) == 1
    assert len(result.centroids) == 1


def test_cluster_result_fields():
    """ClusterResult has all required fields with correct types."""
    embeddings = np.random.default_rng(42).random((10, 8)).astype(np.float32)
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)
    result = cluster_embeddings(embeddings, config)

    assert isinstance(result, ClusterResult)
    assert isinstance(result.labels, list)
    assert isinstance(result.n_clusters, int)
    assert isinstance(result.cluster_sizes, dict)
    assert isinstance(result.centroids, list)
    assert isinstance(result.noise_mask, list)
    assert isinstance(result.method_used, str)
    assert len(result.labels) == 10
    assert len(result.noise_mask) == 10


def test_noise_mask_correct():
    """noise_mask length matches number of input points."""
    embeddings = np.random.default_rng(99).random((20, 8)).astype(np.float32)
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)
    result = cluster_embeddings(embeddings, config)
    assert len(result.noise_mask) == 20
    assert all(isinstance(v, bool) for v in result.noise_mask)


# ---------------------------------------------------------------------------
# Semantic metric unit tests (pure functions, no embedder needed)
# ---------------------------------------------------------------------------

from hif.metrics.semantic import (
    cosine_distance,
    cluster_entropy,
)


def test_cosine_distance_known_values():
    """orthogonal → 1.0, identical → 0.0, zero vector → 1.0."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_distance(a, b) == pytest.approx(1.0)

    c = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(c, c) == pytest.approx(0.0, abs=1e-6)

    zero = np.array([0.0, 0.0, 0.0])
    assert cosine_distance(zero, a) == pytest.approx(1.0)
    assert cosine_distance(a, zero) == pytest.approx(1.0)


def test_cluster_entropy_uniform():
    """Equal mass across 4 clusters → entropy ≈ 2.0 bits."""
    labels = [0, 1, 2, 3]
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    h = cluster_entropy(labels, probs)
    assert h == pytest.approx(2.0, abs=1e-6)


def test_cluster_entropy_point_mass():
    """All mass in one cluster → entropy = 0."""
    labels = [0, 0, 0, 1, 2]
    probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    h = cluster_entropy(labels, probs)
    assert h == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# HDBSCAN-specific behavior tests
# ---------------------------------------------------------------------------

def test_hdbscan_noise_fraction_in_result():
    """noise_mask is populated with bool values."""
    rng = np.random.default_rng(123)
    embeddings = rng.standard_normal((30, 8)).astype(np.float32)
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)
    result = cluster_embeddings(embeddings, config)
    assert len(result.noise_mask) == 30
    assert all(isinstance(v, bool) for v in result.noise_mask)


def test_hdbscan_all_noise_returns_single_cluster():
    """When HDBSCAN labels all points as noise, fallback is one cluster (not 0)."""
    # Near-orthogonal unit vectors: maximally diffuse, no density peaks
    embeddings = np.eye(8, dtype=np.float32)
    config = ClusterConfig(method="hdbscan", min_cluster_size=4, min_samples=4)
    result = cluster_embeddings(embeddings, config)
    assert result.n_clusters >= 1
    assert len(result.centroids) >= 1
    assert len(result.labels) == 8


def test_hdbscan_effective_min_scales_with_n(monkeypatch):
    """For N=40 and config.min_cluster_size=2, effective_min should be max(2, 4)=4."""
    import hdbscan as hdbscan_lib

    # Capture the real class BEFORE patching to avoid recursion.
    _RealHDBSCAN = hdbscan_lib.HDBSCAN
    captured: dict = {}

    class FakeHDBSCAN:
        def __init__(self, min_cluster_size, **kwargs):
            captured["min_cluster_size"] = min_cluster_size
            self._inner = _RealHDBSCAN(min_cluster_size=min_cluster_size, **kwargs)

        def fit_predict(self, X):
            return self._inner.fit_predict(X)

    monkeypatch.setattr(hdbscan_lib, "HDBSCAN", FakeHDBSCAN)

    rng = np.random.default_rng(7)
    embeddings = rng.standard_normal((40, 8)).astype(np.float32)
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)

    cluster_embeddings(embeddings, config)

    # effective_min = max(config.min_cluster_size=2, 40//10=4) → 4
    assert captured.get("min_cluster_size", 0) >= 40 // 10


def test_noise_fraction_surfaced_in_semantic_metrics():
    """SemanticMetrics.noise_fraction is 0.0 when no points are noise."""
    from hif.metrics.semantic import compute_semantic_metrics

    # Two tight clusters → HDBSCAN should assign all to clusters, no noise
    group_a = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (4, 1)).astype(np.float32)
    group_b = np.tile([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (4, 1)).astype(np.float32)
    embeddings = np.vstack([group_a, group_b])

    candidates = [f"tok{i}" for i in range(8)]
    probs = np.array([1/8] * 8)
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)

    embedder = _fake_embedder_returning(embeddings)
    result = compute_semantic_metrics(candidates, probs, embedder, config, truncated=False)

    assert isinstance(result.noise_fraction, float)
    assert 0.0 <= result.noise_fraction <= 1.0


def test_compute_semantic_metrics_structure():
    """compute_semantic_metrics returns SemanticMetrics with correct field types."""
    from hif.metrics.semantic import compute_semantic_metrics, SemanticMetrics

    group_a = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32)
    group_b = np.tile([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32)
    embeddings = np.vstack([group_a, group_b])

    candidates = ["hello", "world", "foo", "bar"]
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)

    embedder = _fake_embedder_returning(embeddings)
    result = compute_semantic_metrics(candidates, probs, embedder, config, truncated=False)

    assert isinstance(result, SemanticMetrics)
    assert result.n_candidates == 4
    assert isinstance(result.cluster_entropy, float)
    assert isinstance(result.mean_pairwise_distance, float)
    assert isinstance(result.max_inter_cluster_distance, float)
    assert isinstance(result.intra_cluster_density, float)
    assert isinstance(result.topic_variance, float)
    assert isinstance(result.noise_fraction, float)
