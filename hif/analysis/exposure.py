"""Hallucination analysis — distributional adjacency and semantic divergence.

Methodological position
-----------------------
Within the horizon of possibility, hallucination is not random: it is a
structurally intelligible move within the model's own distributional space.
At each generation step the model's top-K candidates form a meaning-cloud.
The *selected* token is one point in that cloud; a *hallucinated* token is a
different point that is simultaneously:

1. **Distributionally adjacent** — it has nonzero probability, meaning it sits
   within the model's horizon at that step.
2. **Semantically divergent** — it is far from the selected token in embedding
   space, meaning the two tokens pull the continuation in different directions.

The combination of high distributional adjacency (the hallucination was
probabilistically accessible) and high semantic distance (it would have changed
meaning significantly) is what makes a step "high-risk" for hallucination.

This analysis reuses the already-computed semantic embeddings from the
per-step output trace — no new model inference is required.  The embedding
model's cache ensures that candidate context strings embedded during semantic
metric computation are retrieved without re-encoding.

The cloud phenomenon at each step (convergence / clustering / divergence /
diffusion) tells you *what kind* of hallucination risk is present:

- **Diffusion zone**: the model was already in a high-entropy state — the
  hallucination was probabilistically cheap and semantically varied.
- **Convergence zone**: the model was confident but aimed wrong — the
  hallucination is the road not taken at a narrow fork.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from hif.utils.logging import get_logger

if TYPE_CHECKING:
    from hif.clustering.embed import EmbeddingModel
    from hif.metrics.semantic import SemanticMetrics
    from hif.profile.schema import OutputSideTrace

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Analysis constants (documented defaults)
#
# IMPORTANT: all distance-based values here are EMBEDDER-SPACE-DEPENDENT.
# Cosine distances (and therefore the high-risk flagging and any scalar
# derived from it) are only comparable between profiles computed with the
# same embedding model. ExposureProfile records the `embedder` name so
# consumers can refuse cross-encoder comparisons.
# ---------------------------------------------------------------------------

#: Minimum candidate probability for an alternative to count as
#: "probabilistically accessible" at a step. Below this, the alternative is
#: treated as outside the model's practical horizon.
DEFAULT_MIN_PROB = 0.01

#: Cosine distance (embedder-space-dependent) above which the most divergent
#: accessible alternative marks a step as high-risk.
DEFAULT_DISTANCE_THRESHOLD = 0.3

# Cloud-phenomenon classifier cutoffs (see _cloud_phenomenon). These bound
# the four named regimes on (cluster_count, mean_pairwise_distance,
# intra_cluster_density). The distance cutoff is embedder-space-dependent.
CONVERGENCE_MAX_CLUSTERS = 2       # ≤ this many clusters → convergence/divergence side
CLUSTERING_MIN_CLUSTERS = 4        # ≥ this many clusters → clustering candidate
CLOUD_DISTANCE_SPLIT = 0.4         # mean pairwise distance splitting convergence vs divergence
CLUSTERING_MIN_DENSITY = 0.6       # intra-cluster density required for "clustering"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ExposureCandidate(BaseModel):
    """The most semantically divergent high-probability alternative at one step.

    Represents the token that, had it been selected, would have pulled the
    continuation furthest from the actual path — while remaining within the
    model's distributional horizon at that step.
    """

    step: int
    selected_token: str          # token the model actually produced
    selected_prob: float         # probability of the selected token
    hallucinated_token: str      # most semantically distant top-K alternative
    hallucinated_prob: float     # probability of the hallucinated token
    prob_rank: int               # position in top-K (0 = highest prob)
    semantic_distance: float     # cosine distance from selected token in embedding space [0, 2]
    cloud_phenomenon: str        # convergence | clustering | divergence | diffusion
    cloud_position_2d: list[float]  # [x, y] in the step's 2D semantic cloud; [] if unavailable


class ExposureProfile(BaseModel):
    """Per-step hallucination risk profile.

    Maps the distributional possibility space at each generation step onto a
    risk landscape: where is the model's horizon wide, and where would a
    semantically divergent alternative have been easiest to slip in?
    """

    candidates: list[ExposureCandidate]
    high_risk_steps: list[int]       # diffusion-zone steps with distance > threshold
    mean_semantic_distance: float    # average best-candidate distance across steps
    diffusion_zone_ratio: float      # fraction of steps classified as diffusion
    # Scalar reading value ("Exposure"): fraction of analyzed steps flagged
    # high-risk — len(high_risk_steps) / n analyzed steps, in [0, 1].
    # Chosen over a weighted composite because it is directly localizable (each
    # counted step is inspectable) and its threshold is explicit
    # (DEFAULT_DISTANCE_THRESHOLD + diffusion-zone membership). It measures
    # counterfactual semantic exposure, NOT factuality.
    exposure: float = 0.0
    # Name of the embedding model used to compute semantic distances. Scores
    # are NOT comparable across encoders. None on profiles written before this
    # field existed.
    embedder: str | None = None


# ---------------------------------------------------------------------------
# Cloud phenomenon classifier (local copy — avoids importing from plots/)
# ---------------------------------------------------------------------------


def _cloud_phenomenon(sem: SemanticMetrics) -> str:
    """Classify the semantic cloud at a step as one of four named phenomena."""
    n = sem.cluster_count or 0
    if n == 0:
        return "diffusion"  # no clusters found = maximally diffuse state
    mpd = sem.mean_pairwise_distance or 0.0
    density = sem.intra_cluster_density or 0.0
    if n <= CONVERGENCE_MAX_CLUSTERS and mpd < CLOUD_DISTANCE_SPLIT:
        return "convergence"
    if n >= CLUSTERING_MIN_CLUSTERS and density > CLUSTERING_MIN_DENSITY:
        return "clustering"
    if n <= CONVERGENCE_MAX_CLUSTERS and mpd >= CLOUD_DISTANCE_SPLIT:
        return "divergence"
    return "diffusion"


# ---------------------------------------------------------------------------
# ExposureAnalyzer
# ---------------------------------------------------------------------------


class ExposureAnalyzer:
    """Identifies the most semantically divergent accessible alternative at each step.

    Uses the embedding model (already loaded and cached in the builder) to
    compute cosine distances between the selected token's context string and
    each top-K alternative's context string.  Cache hits from the semantic
    metrics computation make this nearly free.
    """

    def __init__(self, embedder: EmbeddingModel, min_prob: float = DEFAULT_MIN_PROB) -> None:
        self._embedder = embedder
        self._min_prob = min_prob

    def analyze(
        self,
        output_trace: OutputSideTrace,
        semantic_metrics: list[SemanticMetrics],
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    ) -> ExposureProfile:
        """Run hallucination analysis on an output trace.

        For each step finds the top-K candidate that is maximally semantically
        distant from the selected token while satisfying a minimum probability
        threshold.  Records the cloud phenomenon and 2D position at each step.

        Parameters
        ----------
        output_trace:
            The output side trace from the profile builder.
        semantic_metrics:
            Pre-computed semantic metrics, one per output step.
        distance_threshold:
            Cosine distance above which a candidate is flagged as high-risk.

        Returns
        -------
        ExposureProfile
        """
        steps = output_trace.steps
        candidates: list[ExposureCandidate] = []

        for i, (step, sem) in enumerate(zip(steps, semantic_metrics)):
            # Full generated prefix as shared context. Exposure asks how much an
            # accessible alternative would shift the RESPONSE's meaning, so the
            # comparison holds the entire response-so-far fixed (a shared "mask")
            # and varies only the final token. A short fixed window (the old
            # min(5, i)) let a single token dominate a short embedding and
            # inflated distances for later steps — verified mean |Δ| ~0.08 vs
            # full prefix, up to 0.57 on individual steps. Long generations are
            # bounded by the embedder's own max-sequence truncation.
            context_prefix = "".join(s.selected_token_str for s in steps[:i])

            selected_ctx = context_prefix + step.selected_token_str
            try:
                selected_emb = self._embedder.embed([selected_ctx])[0]
            except Exception:
                continue

            sel_norm = np.linalg.norm(selected_emb)
            if sel_norm < 1e-8:
                continue
            selected_unit = selected_emb / sel_norm

            best_dist = -1.0
            best_entry = None
            best_rank = 0

            for rank, entry in enumerate(step.topk):
                if entry.token_str == step.selected_token_str:
                    continue
                if entry.prob < self._min_prob:
                    break  # topk is sorted by prob descending

                cand_ctx = context_prefix + entry.token_str
                try:
                    cand_emb = self._embedder.embed([cand_ctx])[0]
                except Exception:
                    continue

                c_norm = np.linalg.norm(cand_emb)
                if c_norm < 1e-8:
                    continue
                cand_unit = cand_emb / c_norm

                dist = float(1.0 - np.dot(selected_unit, cand_unit))
                if dist > best_dist:
                    best_dist = dist
                    best_entry = entry
                    best_rank = rank

            if best_entry is None:
                continue

            phenomenon = _cloud_phenomenon(sem)

            # Cloud position of the hallucinated token (index into pre-projected 2D)
            cloud_pos: list[float] = []
            if sem.embeddings_2d and best_rank < len(sem.embeddings_2d):
                cloud_pos = list(sem.embeddings_2d[best_rank])

            candidates.append(ExposureCandidate(
                step=i,
                selected_token=step.selected_token_str,
                selected_prob=step.topk[0].prob if step.topk else 0.0,
                hallucinated_token=best_entry.token_str,
                hallucinated_prob=best_entry.prob,
                prob_rank=best_rank,
                semantic_distance=best_dist,
                cloud_phenomenon=phenomenon,
                cloud_position_2d=cloud_pos,
            ))

        embedder_name = getattr(self._embedder, "model_name", None) or None

        if not candidates:
            return ExposureProfile(
                candidates=[],
                high_risk_steps=[],
                mean_semantic_distance=0.0,
                diffusion_zone_ratio=0.0,
                exposure=0.0,
                embedder=embedder_name,
            )

        high_risk = [
            c.step for c in candidates
            if c.cloud_phenomenon == "diffusion" and c.semantic_distance >= distance_threshold
        ]
        mean_dist = float(np.mean([c.semantic_distance for c in candidates]))
        n_diffusion = sum(1 for c in candidates if c.cloud_phenomenon == "diffusion")
        diffusion_ratio = n_diffusion / len(candidates)

        logger.info(
            "Hallucination check: %d of %d steps flagged high-risk (mean divergence %.2f)",
            len(high_risk),
            len(candidates),
            mean_dist,
        )
        logger.debug(
            "Hallucination detail: diffusion zone=%.0f%% of steps",
            diffusion_ratio * 100,
        )

        return ExposureProfile(
            candidates=candidates,
            high_risk_steps=high_risk,
            mean_semantic_distance=mean_dist,
            diffusion_zone_ratio=diffusion_ratio,
            exposure=len(high_risk) / len(candidates),
            embedder=embedder_name,
        )
