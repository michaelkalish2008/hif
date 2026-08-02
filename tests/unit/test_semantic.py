"""End-to-end tests for compute_semantic_metrics with a fake embedder."""

import numpy as np
import pytest

from hif.config import ClusterConfig
from hif.metrics.semantic import compute_semantic_metrics, SemanticMetrics


# ---------------------------------------------------------------------------
# Fake embedder — no mock/patch
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Returns rows from a fixed embedding matrix on successive embed() calls."""

    def __init__(self, embeddings: np.ndarray):
        self._embeddings = embeddings
        self._pos = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        rows = self._embeddings[self._pos : self._pos + n]
        self._pos += n
        return rows


def _two_cluster_embeddings(n_per_cluster: int = 2) -> np.ndarray:
    """Return embeddings for n_per_cluster points near [1,0,...] and [0,1,...] alternating."""
    a = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (n_per_cluster, 1))
    b = np.tile([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (n_per_cluster, 1))
    return np.vstack([a, b]).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_semantic_metrics_hdbscan_two_clusters():
    """4 candidates in 2 tight groups, uniform probs → all fields present and in range."""
    candidates = ["apple", "banana", "car", "dog"]
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)

    embedder = FakeEmbedder(_two_cluster_embeddings(n_per_cluster=2))
    result = compute_semantic_metrics(candidates, probs, embedder, config, truncated=False)

    assert isinstance(result, SemanticMetrics)
    assert result.n_candidates == 4
    assert result.cluster_count >= 1
    assert 0.0 <= result.mean_pairwise_distance <= 1.0
    assert 0.0 <= result.intra_cluster_density <= 1.0
    assert result.truncated is False


def test_truncated_flag_propagates():
    """truncated=True must be reflected in the result."""
    candidates = ["x", "y", "z", "w"]
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    config = ClusterConfig(method="hdbscan", min_cluster_size=2, min_samples=1)

    embedder = FakeEmbedder(_two_cluster_embeddings(n_per_cluster=2))
    result = compute_semantic_metrics(candidates, probs, embedder, config, truncated=True)
    assert result.truncated is True
