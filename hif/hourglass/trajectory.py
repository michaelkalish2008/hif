"""Multi-branch output trajectory sampling for characterizing generation dynamics.

Implements the "double helix" view: B semantic clusters → one representative per
cluster → roll forward R steps. Total expansion is B branches, not K^B or K^R.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from hif.clustering.cluster import ClusterConfig, cluster_embeddings
from hif.clustering.embed import EmbeddingModel
from hif.config import TrajectoryConfig
from hif.models.base import Model, StepRecord
from hif.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Branch(BaseModel):
    cluster_id: int
    representative_token_ids: list[int]   # the seed token IDs that started this branch
    generated_ids: list[int]              # R tokens generated from this branch
    steps: list[StepRecord]               # step records for the R generation steps
    final_text: str                       # detokenized generated_ids


class BranchConvergence(BaseModel):
    step: int
    n_remaining_clusters: int             # how many semantic clusters remain at this step


class BranchField(BaseModel):
    """Geometric field descriptors of the trajectory-branch cloud in embedding
    space — the sampling-perturbation twin of the perturbation field
    (DRIFT_FIELD_MODEL.md § trajectory branches). Where Continuity collapses the
    branch cloud to one mean-pairwise-cosine scalar, this restores its shape:
    the centroid (expected trajectory), the radii around it, and — crucially —
    cluster_count, which detects MULTI-MODALITY (branches splitting into distinct
    attractors) that a mean cannot see.

    Radii/dispersion are cosine distances in [0, 2]. Descriptor names align with
    PerturbationField so compute_field_deformation / platform parseField consume
    both; cluster_count is the extra (multi-modality) descriptor. Derived scalars
    only — no embeddings persisted."""

    n_branches: int
    mean_radius: float        # mean branch→centroid cosine distance
    radius_variance: float    # variance of branch radii (isotropy of the cloud)
    max_radius: float         # most divergent branch (the worst-case future)
    field_dispersion: float   # mean pairwise cosine distance (1 − trajectory_continuity)
    cluster_count: int        # HDBSCAN clusters among branches — multi-modality


class TrajectoryAnalysis(BaseModel):
    start_step: int                       # generation step at which branching begins
    n_branches: int                       # B
    rollout_steps: int                    # R
    branches: list[Branch]
    convergence_profile: list[BranchConvergence]   # one per rollout step
    persistence_score: float              # fraction of rollout steps with > 1 cluster remaining
    explosion_score: float                # fraction of rollout steps where n_clusters > initial
    convergence_score: float              # fraction of rollout steps where n_clusters < initial
    initial_n_clusters: int
    trajectory_continuity: float | None = None  # mean pairwise cosine similarity between branch embeddings; high = branches converge semantically
    # Branch field (geometry of the branch cloud). None when < 2 branches aligned
    # or on skipped/degenerate trajectory paths. Defaults to None so profiles
    # written before it still validate.
    branch_field: "BranchField | None" = None


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity across N embeddings. Returns 1.0 for N < 2."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = embeddings / norms
    sim = normed @ normed.T
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean([sim[i, j] for i, j in pairs]))


def _compute_branch_field(
    embeddings: np.ndarray, cluster_config: ClusterConfig
) -> "BranchField | None":
    """Branch-field descriptors from the branch final-text embeddings.

    Geometric field: centroid = normalised mean embedding; per-branch radius =
    cosine distance to it. cluster_count comes from clustering the branch
    embeddings — the multi-modality signal. Returns None for < 2 branches.
    """
    n = len(embeddings)
    if n < 2:
        return None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = embeddings / norms
    centroid = normed.mean(axis=0)
    cn = np.linalg.norm(centroid)
    centroid = centroid / (cn if cn > 0 else 1e-8)
    radii = 1.0 - (normed @ centroid)  # cosine distance to centroid, per branch
    field_dispersion = 1.0 - _mean_pairwise_cosine(embeddings)
    try:
        cluster_count = cluster_embeddings(embeddings, cluster_config).n_clusters
    except Exception:
        cluster_count = 1
    return BranchField(
        n_branches=n,
        mean_radius=float(np.mean(radii)),
        radius_variance=float(np.var(radii)),
        max_radius=float(np.max(radii)),
        field_dispersion=float(field_dispersion),
        cluster_count=int(cluster_count),
    )


def _cluster_texts(
    texts: list[str],
    embedder: EmbeddingModel,
    cluster_config: ClusterConfig,
    n_clusters_target: int,
) -> tuple[list[int], int]:
    """Embed texts and cluster them. Returns (labels, n_clusters).

    Uses HDBSCAN; if fewer than 2 clusters emerge, returns the single-cluster result as-is.
    """
    embeddings = embedder.embed(texts)
    result = cluster_embeddings(embeddings, cluster_config)
    return result.labels, result.n_clusters


def analyze_trajectory(
    model: Model,
    context_ids: list[int],              # the full prompt + generated tokens up to start_step
    embedder: EmbeddingModel,
    config: TrajectoryConfig,
    cluster_config: ClusterConfig,
    seed: int = 42,
) -> TrajectoryAnalysis:
    """Analyze multi-branch trajectory from context_ids.

    Steps:
      1. Forward pass to get logits at position -1.
      2. Take top-K tokens, embed, cluster into B semantic clusters.
      3. Pick highest-probability representative from each cluster.
      4. For each branch: extend context with representative and generate R tokens.
      5. At each rollout step, cluster branch texts and record convergence.
      6. Compute summary scores and return TrajectoryAnalysis.
    """
    B = config.n_branches
    R = config.rollout_steps
    top_k = 50  # fixed candidate pool for branching

    start_step = len(context_ids)

    # ------------------------------------------------------------------
    # Step 1: forward pass → logits at last position
    # ------------------------------------------------------------------
    logits_result = model.forward(context_ids)
    logits_np = logits_result.to_numpy()  # (seq_len, vocab_size)
    last_logits = logits_np[-1]           # (vocab_size,)

    # Softmax
    shifted = last_logits - last_logits.max()
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum()

    vocab_size = logits_result.vocab_size
    k = min(top_k, vocab_size)

    # Top-K by probability
    top_indices = np.argpartition(probs, -k)[-k:]
    top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]
    top_probs = probs[top_indices]

    # ------------------------------------------------------------------
    # Step 2: embed top-K token strings and cluster
    # ------------------------------------------------------------------
    top_token_strings = [model.detokenize([int(tid)]) for tid in top_indices]
    labels, n_clusters = _cluster_texts(top_token_strings, embedder, cluster_config, B)

    initial_n_clusters = n_clusters
    logger.debug("Trajectory: %d initial clusters from top-%d tokens", n_clusters, k)

    # ------------------------------------------------------------------
    # Step 3: pick representative (highest-prob) token per cluster
    # ------------------------------------------------------------------
    label_arr = np.array(labels)
    cluster_ids = sorted(set(labels) - {-1})

    # If all noise (shouldn't happen after fallback), treat all as one cluster
    if not cluster_ids:
        cluster_ids = [0]
        label_arr = np.zeros(k, dtype=int)

    representatives: list[int] = []   # token_id of representative per cluster
    for cid in cluster_ids:
        cluster_mask = label_arr == cid
        cluster_indices = np.where(cluster_mask)[0]
        # Highest probability in this cluster
        best_local = int(np.argmax(top_probs[cluster_indices]))
        best_global = int(top_indices[cluster_indices[best_local]])
        representatives.append(best_global)

    # Limit to B branches
    representatives = representatives[:B]

    # ------------------------------------------------------------------
    # Step 4: generate R steps for each branch
    # ------------------------------------------------------------------
    branches: list[Branch] = []
    for i, rep_token_id in enumerate(representatives):
        branch_context = context_ids + [rep_token_id]
        gen_result = model.generate(
            input_ids=branch_context,
            max_new_tokens=R,
            top_k=50,
            seed=seed + i,
        )
        final_text = model.detokenize(gen_result.generated_ids)
        branch = Branch(
            cluster_id=cluster_ids[i] if i < len(cluster_ids) else i,
            representative_token_ids=[rep_token_id],
            generated_ids=gen_result.generated_ids,
            steps=gen_result.steps,
            final_text=final_text,
        )
        branches.append(branch)

    # ------------------------------------------------------------------
    # Step 5: convergence profile — at each rollout step, cluster branch texts
    # ------------------------------------------------------------------
    convergence_profile: list[BranchConvergence] = []

    for step_i in range(R):
        # Collect generated text for each branch up to step_i (inclusive)
        branch_texts = []
        for branch in branches:
            partial_ids = branch.generated_ids[: step_i + 1]
            if partial_ids:
                branch_texts.append(model.detokenize(partial_ids))
            else:
                branch_texts.append("")

        # Need at least 2 texts to cluster meaningfully
        if len(branch_texts) < 2:
            n_rem = 1
        else:
            try:
                _, n_rem = _cluster_texts(branch_texts, embedder, cluster_config, B)
            except Exception:
                n_rem = 1

        convergence_profile.append(BranchConvergence(step=step_i, n_remaining_clusters=n_rem))

    # ------------------------------------------------------------------
    # Step 6: compute summary scores
    # ------------------------------------------------------------------
    n_steps = len(convergence_profile)
    if n_steps == 0:
        persistence_score = 0.0
        explosion_score = 0.0
        convergence_score = 0.0
    else:
        n_rem_values = [bc.n_remaining_clusters for bc in convergence_profile]
        persistence_score = float(sum(1 for n in n_rem_values if n > 1) / n_steps)
        explosion_score = float(sum(1 for n in n_rem_values if n > initial_n_clusters) / n_steps)
        convergence_score = float(sum(1 for n in n_rem_values if n < initial_n_clusters) / n_steps)

    # Trajectory continuity: mean pairwise cosine similarity between branch final_text embeddings
    # High = branches converge semantically; Low = branches scatter (discontinuity)
    branch_field: "BranchField | None" = None
    try:
        branch_texts = [b.final_text for b in branches if b.final_text]
        if len(branch_texts) >= 2:
            branch_embeddings = embedder.embed(branch_texts)
            trajectory_continuity = _mean_pairwise_cosine(branch_embeddings)
            branch_field = _compute_branch_field(branch_embeddings, cluster_config)
        else:
            trajectory_continuity = None
    except Exception:
        trajectory_continuity = None

    return TrajectoryAnalysis(
        start_step=start_step,
        n_branches=len(branches),
        rollout_steps=R,
        branches=branches,
        convergence_profile=convergence_profile,
        persistence_score=persistence_score,
        explosion_score=explosion_score,
        convergence_score=convergence_score,
        initial_n_clusters=initial_n_clusters,
        trajectory_continuity=trajectory_continuity,
        branch_field=branch_field,
    )
