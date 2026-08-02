"""Similarity metrics: semantic consistency across inputs, outputs, and input-output pairs.

Three complementary lenses:
  input_sim   — mean pairwise cosine across all input texts (baseline + perturbation variants).
                Tells you how varied the inputs actually were.
  output_sim  — mean pairwise cosine across all corresponding output texts.
                Only meaningful relative to input_sim.
  io_sim      — mean cosine(input_i, output_i) for each paired input and output.
                Measures whether the model stays semantically anchored to its prompt.
  io_ratio    — output_sim / input_sim.  > 1 means the model suppresses input variation
                (outputs converge more than inputs did); < 1 means it amplifies it.
  trend       — linear slope of per-step mean_pairwise_similarity across the output
                token sequence.  Positive = candidates converging as generation proceeds
                (semantic focus building); negative = diverging (coherence evaporating).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from hif.clustering.embed import EmbeddingModel
from hif.metrics.semantic import SemanticMetrics, cosine_distance
from hif.utils.logging import get_logger

logger = get_logger(__name__)


class SimilarityMetrics(BaseModel):
    input_sim: float    # mean pairwise cosine across all input embeddings
    output_sim: float   # mean pairwise cosine across all output embeddings
    io_sim: float       # mean cosine(input_i, output_i) per pair
    io_ratio: float     # output_sim / input_sim; None collapsed to 0.0 when input_sim == 0
    trend: float        # linear slope of per-step similarity over output sequence
    n_pairs: int        # number of (input, output) pairs used


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine similarity over all unique pairs of N embeddings."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    i_idx, j_idx = np.triu_indices(n, k=1)
    return float(sim_matrix[i_idx, j_idx].mean())


def _mean_io_cosine(
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
) -> float:
    """Mean cosine similarity between each input_i and its paired output_i."""
    n = len(input_embeddings)
    if n == 0:
        return 0.0
    sims = [
        1.0 - cosine_distance(input_embeddings[i], output_embeddings[i])
        for i in range(n)
    ]
    return float(np.mean(sims))


def _similarity_trend(semantic_metrics: list[SemanticMetrics]) -> float:
    """Linear slope of per-step mean pairwise cosine similarity over output steps.

    Converts existing mean_pairwise_distance → similarity (1 - distance), then
    fits a degree-1 polynomial.  Positive slope = converging, negative = diverging.
    Returns 0.0 when fewer than 2 steps are available.
    """
    if not semantic_metrics or len(semantic_metrics) < 2:
        return 0.0
    per_step = np.array(
        [1.0 - sm.mean_pairwise_distance for sm in semantic_metrics],
        dtype=np.float64,
    )
    steps = np.arange(len(per_step), dtype=np.float64)
    slope = float(np.polyfit(steps, per_step, 1)[0])
    return slope


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_similarity_metrics(
    input_texts: list[str],
    output_texts: list[str],
    semantic_metrics: list[SemanticMetrics] | None,
    embedder: EmbeddingModel,
) -> SimilarityMetrics:
    """Compute all similarity metrics for a set of (input, output) text pairs.

    Parameters
    ----------
    input_texts:
        All input prompts in order: [baseline_prompt, variant_1, variant_2, ...].
    output_texts:
        Corresponding generated outputs in the same order.
    semantic_metrics:
        Per-step SemanticMetrics from the baseline output trace, used for trend.
    embedder:
        Shared EmbeddingModel instance.
    """
    n = min(len(input_texts), len(output_texts))
    if n == 0:
        return SimilarityMetrics(
            input_sim=0.0,
            output_sim=0.0,
            io_sim=0.0,
            io_ratio=0.0,
            trend=0.0,
            n_pairs=0,
        )

    input_texts = input_texts[:n]
    output_texts = output_texts[:n]

    input_embs = embedder.embed(input_texts)    # (n, D)
    output_embs = embedder.embed(output_texts)  # (n, D)

    input_sim = _mean_pairwise_cosine(input_embs)
    output_sim = _mean_pairwise_cosine(output_embs)
    io_sim = _mean_io_cosine(input_embs, output_embs)
    io_ratio = (output_sim / input_sim) if input_sim > 1e-6 else 0.0
    trend = _similarity_trend(semantic_metrics) if semantic_metrics else 0.0

    logger.debug(
        "SimilarityMetrics: input_sim=%.3f output_sim=%.3f io_sim=%.3f "
        "io_ratio=%.3f trend=%.5f n_pairs=%d",
        input_sim,
        output_sim,
        io_sim,
        io_ratio,
        trend,
        n,
    )

    return SimilarityMetrics(
        input_sim=input_sim,
        output_sim=output_sim,
        io_sim=io_sim,
        io_ratio=io_ratio,
        trend=trend,
        n_pairs=n,
    )
