"""Semantic metrics: embedding-space dispersion and centroid distance of sampled outputs."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from hif.clustering.cluster import ClusterResult, cluster_embeddings
from hif.clustering.embed import EmbeddingModel
from hif.config import ClusterConfig
from hif.utils.logging import get_logger

logger = get_logger(__name__)


class SemanticMetrics(BaseModel):
    cluster_count: int
    cluster_entropy: float           # Shannon entropy of probability mass across clusters (bits)
    mean_pairwise_distance: float    # mean cosine distance between candidates, weighted by prob
    max_inter_cluster_distance: float
    intra_cluster_density: float     # mean within-cluster cosine similarity, weighted by prob
    topic_variance: float            # variance of cluster centroids weighted by cluster mass
    n_candidates: int
    noise_fraction: float = 0.0     # fraction of candidates HDBSCAN labelled as noise
    truncated: bool                  # True if input was top-K truncated (not full vocab)
    cluster_labels: list[int] = []  # cluster label per top-K candidate; -1 = noise
    embeddings_2d: list[list[float]] = []  # shape (n_candidates, 2) — 2D projection
    projection_method: str = "pca"  # "pca" or "umap"


# ---------------------------------------------------------------------------
# Low-level metric helpers
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity(a, b). Handles zero vectors (returns 1.0)."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp for numerical safety.
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def cluster_entropy(
    cluster_labels: list[int],
    probs: np.ndarray,
) -> float:
    """Shannon entropy (bits) of probability mass distributed across clusters.

    Noise points (label == -1) are excluded.
    """
    labels = np.array(cluster_labels)
    probs = np.array(probs, dtype=np.float64)

    # Exclude noise.
    valid = labels != -1
    labels = labels[valid]
    probs = probs[valid]

    cluster_ids = np.unique(labels)
    if len(cluster_ids) == 0:
        return 0.0

    cluster_masses = np.array([probs[labels == cid].sum() for cid in cluster_ids])
    total = cluster_masses.sum()
    if total == 0.0:
        return 0.0

    p = cluster_masses / total
    # Avoid log(0).
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mean_pairwise_distance(
    embeddings: np.ndarray,   # (N, D)
    probs: np.ndarray,        # (N,) — weights
) -> float:
    """Probability-weighted mean cosine distance between all pairs."""
    n = len(embeddings)
    if n < 2:
        return 0.0

    probs = np.array(probs, dtype=np.float64)
    total_weight = 0.0
    weighted_sum = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            w = probs[i] * probs[j]
            d = cosine_distance(embeddings[i], embeddings[j])
            weighted_sum += w * d
            total_weight += w

    if total_weight == 0.0:
        return 0.0
    return float(weighted_sum / total_weight)


def max_inter_cluster_distance(
    centroids: np.ndarray,   # (K, D)
) -> float:
    """Maximum cosine distance between any two cluster centroids."""
    k = len(centroids)
    if k < 2:
        return 0.0

    max_dist = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            d = cosine_distance(centroids[i], centroids[j])
            if d > max_dist:
                max_dist = d
    return float(max_dist)


def intra_cluster_density(
    embeddings: np.ndarray,  # (N, D)
    labels: list[int],
    probs: np.ndarray,       # (N,)
) -> float:
    """Probability-weighted mean cosine similarity within each cluster."""
    label_arr = np.array(labels)
    probs = np.array(probs, dtype=np.float64)

    cluster_ids = sorted(set(labels) - {-1})
    if not cluster_ids:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for cid in cluster_ids:
        mask = label_arr == cid
        cluster_embs = embeddings[mask]
        cluster_probs = probs[mask]

        m = len(cluster_embs)
        if m < 2:
            # Single-point cluster: similarity with itself = 1.
            w = float(cluster_probs.sum())
            weighted_sum += w * 1.0
            total_weight += w
            continue

        for i in range(m):
            for j in range(i + 1, m):
                w = cluster_probs[i] * cluster_probs[j]
                # cosine similarity = 1 - cosine_distance
                sim = 1.0 - cosine_distance(cluster_embs[i], cluster_embs[j])
                weighted_sum += w * sim
                total_weight += w

    if total_weight == 0.0:
        return 0.0
    return float(weighted_sum / total_weight)


def topic_variance(
    centroids: np.ndarray,       # (K, D)
    cluster_masses: np.ndarray,  # (K,) — probability mass per cluster
) -> float:
    """Weighted variance of cluster centroids.

    Weighted mean centroid, then weighted mean squared L2 distance from it.
    """
    k = len(centroids)
    if k == 0:
        return 0.0

    cluster_masses = np.array(cluster_masses, dtype=np.float64)
    total = cluster_masses.sum()
    if total == 0.0:
        return 0.0

    weights = cluster_masses / total
    # Weighted mean centroid.
    mean_centroid = np.average(centroids, axis=0, weights=weights)

    # Weighted mean squared L2 distance.
    squared_dists = np.array(
        [np.sum((c - mean_centroid) ** 2) for c in centroids],
        dtype=np.float64,
    )
    return float(np.dot(weights, squared_dists))


# ---------------------------------------------------------------------------
# 2D projection helper
# ---------------------------------------------------------------------------

def _project_2d(embeddings: np.ndarray) -> tuple[list[list[float]], str]:
    """Project embeddings to 2D for visualization.

    Tries UMAP first (if installed), falls back to PCA.
    Returns (coords_2d as list[list[float]], method_name).
    """
    n = len(embeddings)
    if n < 2:
        coords = [[0.0, 0.0]] * n
        return coords, "pca"

    # Try UMAP first.
    try:
        import umap  # type: ignore

        n_neighbors = min(5, n - 1)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        coords_2d = reducer.fit_transform(embeddings.astype(np.float32))
        return coords_2d.tolist(), "umap"
    except (ImportError, Exception):  # noqa: BLE001
        pass

    # Fall back to PCA.
    from sklearn.decomposition import PCA  # type: ignore

    n_components = min(2, n, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords_2d = pca.fit_transform(embeddings.astype(np.float64))
    # Pad to 2 columns if embedding dim < 2.
    if coords_2d.shape[1] == 1:
        coords_2d = np.hstack([coords_2d, np.zeros((n, 1))])
    return coords_2d.tolist(), "pca"


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------

def compute_semantic_metrics(
    candidate_strings: list[str],
    probs: np.ndarray,           # (N,) — probabilities of each candidate
    embedder: EmbeddingModel,
    cluster_config: ClusterConfig,
    truncated: bool = False,
) -> SemanticMetrics:
    """Embed candidates, cluster, compute all semantic metrics."""
    probs = np.array(probs, dtype=np.float64)
    n = len(candidate_strings)

    if n == 0:
        return SemanticMetrics(
            cluster_count=0,
            cluster_entropy=0.0,
            mean_pairwise_distance=0.0,
            max_inter_cluster_distance=0.0,
            intra_cluster_density=0.0,
            topic_variance=0.0,
            n_candidates=0,
            truncated=truncated,
            cluster_labels=[],
            embeddings_2d=[],
            projection_method="pca",
        )

    # Embed candidates using the embedding model (not the model under analysis).
    embeddings = embedder.embed(candidate_strings)  # (N, D)

    # Cluster embeddings.
    cluster_result: ClusterResult = cluster_embeddings(embeddings, cluster_config)

    # Collect per-cluster probability masses (for non-noise clusters).
    label_arr = np.array(cluster_result.labels)
    cluster_ids = sorted(cluster_result.cluster_sizes.keys())
    cluster_mass_arr = np.array(
        [probs[label_arr == cid].sum() for cid in cluster_ids],
        dtype=np.float64,
    )

    centroids_arr = np.array(cluster_result.centroids, dtype=np.float64)

    h = cluster_entropy(cluster_result.labels, probs)
    mpd = mean_pairwise_distance(embeddings, probs)
    micd = max_inter_cluster_distance(centroids_arr) if len(centroids_arr) >= 2 else 0.0
    icd = intra_cluster_density(embeddings, cluster_result.labels, probs)
    tv = topic_variance(centroids_arr, cluster_mass_arr) if len(centroids_arr) >= 1 else 0.0

    noise_fraction = float(sum(cluster_result.noise_mask) / n) if n > 0 else 0.0

    # Compute 2D projection for visualization.
    embeddings_2d, projection_method = _project_2d(embeddings)

    logger.debug(
        "SemanticMetrics: clusters=%d noise=%.0f%% entropy=%.3f mpd=%.3f projection=%s",
        cluster_result.n_clusters,
        noise_fraction * 100,
        h,
        mpd,
        projection_method,
    )

    return SemanticMetrics(
        cluster_count=cluster_result.n_clusters,
        cluster_entropy=h,
        mean_pairwise_distance=mpd,
        max_inter_cluster_distance=micd,
        intra_cluster_density=icd,
        topic_variance=tv,
        n_candidates=n,
        noise_fraction=noise_fraction,
        truncated=truncated,
        cluster_labels=list(cluster_result.labels),
        embeddings_2d=embeddings_2d,
        projection_method=projection_method,
    )
