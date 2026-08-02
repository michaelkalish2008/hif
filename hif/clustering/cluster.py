"""Cluster assignment using HDBSCAN (default) or k-means over output embeddings."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from hif.config import ClusterConfig
from hif.utils.logging import get_logger

logger = get_logger(__name__)


class ClusterResult(BaseModel):
    labels: list[int]              # cluster label per item; -1 = noise (HDBSCAN only)
    n_clusters: int                # number of clusters (excluding noise)
    cluster_sizes: dict[int, int]  # {cluster_id: count}
    centroids: list[list[float]]   # shape (n_clusters, embedding_dim); noise excluded
    noise_mask: list[bool]         # True where label == -1
    method_used: str               # always "hdbscan"


def _single_cluster(embeddings: np.ndarray) -> ClusterResult:
    """Return a trivial single-cluster result."""
    n = len(embeddings)
    centroid = embeddings.mean(axis=0).tolist()
    return ClusterResult(
        labels=[0] * n,
        n_clusters=1,
        cluster_sizes={0: n},
        centroids=[centroid],
        noise_mask=[False] * n,
        method_used="kmeans",
    )


def _compute_centroids(
    embeddings: np.ndarray, labels: np.ndarray, cluster_ids: list[int]
) -> list[list[float]]:
    centroids = []
    for cid in cluster_ids:
        mask = labels == cid
        centroids.append(embeddings[mask].mean(axis=0).tolist())
    return centroids


def cluster_embeddings(
    embeddings: np.ndarray,
    config: ClusterConfig,
) -> ClusterResult:
    """Assign cluster labels to an (N, D) embedding matrix using HDBSCAN.

    Edge cases: N==1 or N < min_cluster_size → single cluster.
    All-noise result → single diffuse cluster fallback.
    """
    n = len(embeddings)

    if n == 1 or n < config.min_cluster_size:
        logger.debug("Too few points (%d) for clustering; returning single cluster.", n)
        return _single_cluster(embeddings)

    return _run_hdbscan(embeddings, config)


def _run_hdbscan(embeddings: np.ndarray, config: ClusterConfig) -> ClusterResult:
    import hdbscan as hdbscan_lib

    # Scale min_cluster_size with N so clusters represent meaningful groups,
    # not token pairs. Floor at config value; scale at N//10 for N≥20.
    n = len(embeddings)
    effective_min = max(config.min_cluster_size, n // 10)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=effective_min,
        min_samples=config.min_samples,
    )
    labels: np.ndarray = clusterer.fit_predict(embeddings)

    cluster_ids = sorted(set(labels.tolist()) - {-1})
    n_clusters = len(cluster_ids)

    if n_clusters == 0:
        # All-noise is a real signal: the distribution is diffuse with no density peaks.
        logger.debug("HDBSCAN produced 0 clusters (all noise); treating as single diffuse cluster.")
        return _single_cluster(embeddings)

    noise_mask = (labels == -1).tolist()
    cluster_sizes = {cid: int((labels == cid).sum()) for cid in cluster_ids}
    centroids = _compute_centroids(embeddings, labels, cluster_ids)

    logger.debug("HDBSCAN: %d clusters, %d noise points.", n_clusters, sum(noise_mask))
    return ClusterResult(
        labels=labels.tolist(),
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        centroids=centroids,
        noise_mask=noise_mask,
        method_used="hdbscan",
    )


