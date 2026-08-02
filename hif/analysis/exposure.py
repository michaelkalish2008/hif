"""Counterfactual exposure analysis — distributional adjacency and semantic
distance of accessible alternatives.

What is computed
----------------
At each generation step the model's top-K candidates form a cloud of
accessible continuations. The *selected* token is one point in that cloud.
For each step, this analysis finds the alternative candidate that is
simultaneously:

1. **Distributionally adjacent** — its probability clears a floor
   (min_prob), meaning it was practically accessible at that step, and
2. **Semantically distant** — its context string embeds far from the selected
   token's in the analysis encoder's space, meaning selecting it would have
   pulled the continuation toward a different meaning.

A step is counted as *exposed* when its most divergent accessible alternative
is semantically distant (distance ≥ threshold) inside a *diffuse* candidate
cloud. The scalar reading (`exposure`, surfaced as
`counterfactual_exposure_fraction`) is the fraction of analysed steps so
counted: how often the response's meaning was exposed to sampling chance.

This is a description of the run's own distributional possibility space —
what the model COULD accessibly have said and how far away in meaning that
was. It is not an inference about what the model was doing, and it is
explicitly NOT a factuality or correctness judgment: only diffusion-zone
steps are counted, so the convergence case (a model that is confident and
narrow but aimed wrong) is excluded by construction, and a confident response
can still be wrong. See docs/MEASUREMENTS.md § counterfactual_exposure_fraction.

This analysis reuses the already-computed semantic embeddings from the
per-step output trace — no new model inference is required.  The embedding
model's cache ensures that candidate context strings embedded during semantic
metric computation are retrieved without re-encoding.

The cloud phenomenon at each step (convergence / clustering / divergence /
diffusion) says what kind of possibility space the alternative sat in:

- **Diffusion zone**: a high-entropy step — semantically varied alternatives
  were probabilistically cheap.
- **Convergence zone**: a narrow fork — the alternative was the road not
  taken from a confident state (not counted toward exposure; see above).

Vocabulary note: the fields here were renamed from a "hallucination"/"risk"
vocabulary (hallucinated_token/-_prob, high_risk_steps) that framed the
computation as detecting a model failure it does not detect. Validation
aliases accept the old names so archived profile JSON keeps loading; new
artifacts carry only the exposure vocabulary (profile schema 0.10.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

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
# Cosine distances (and therefore the exposed-step counting and any scalar
# derived from it) are only comparable between profiles computed with the
# same embedding model. ExposureProfile records the `embedder` name so
# consumers can refuse cross-encoder comparisons.
# ---------------------------------------------------------------------------

#: Minimum candidate probability for an alternative to count as
#: "probabilistically accessible" at a step. Below this, the alternative is
#: treated as outside the model's practical horizon.
DEFAULT_MIN_PROB = 0.01

#: Cosine distance (embedder-space-dependent) above which the most divergent
#: accessible alternative marks a step as exposed.
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

    populate_by_name lets the analyzer construct by field name while the
    validation aliases keep pre-rename profile JSON (hallucinated_token /
    hallucinated_prob) loadable.
    """

    model_config = ConfigDict(populate_by_name=True)

    step: int
    selected_token: str          # token the model actually produced
    selected_prob: float         # probability of the selected token
    # Most semantically distant accessible top-K alternative, and its
    # probability. (Aliases: renamed from hallucinated_token/-_prob — the old
    # names claimed the alternative was a hallucination, which nothing here
    # measures.)
    divergent_token: str = Field(
        validation_alias=AliasChoices("divergent_token", "hallucinated_token"),
    )
    divergent_prob: float = Field(
        validation_alias=AliasChoices("divergent_prob", "hallucinated_prob"),
    )
    prob_rank: int               # position in top-K (0 = highest prob)
    semantic_distance: float     # cosine distance from selected token in embedding space [0, 2]
    cloud_phenomenon: str        # convergence | clustering | divergence | diffusion
    cloud_position_2d: list[float]  # [x, y] in the step's 2D semantic cloud; [] if unavailable


class ExposureProfile(BaseModel):
    """Per-step counterfactual exposure profile.

    Maps the distributional possibility space at each generation step: where
    the candidate cloud was wide, and how far in meaning an accessible
    alternative sat from the token actually selected.
    """

    model_config = ConfigDict(populate_by_name=True)

    candidates: list[ExposureCandidate]
    # Diffusion-zone steps whose most divergent accessible alternative cleared
    # the distance threshold. (Alias: renamed from high_risk_steps — "risk"
    # asserted a hazard judgment this analysis does not make.)
    exposed_steps: list[int] = Field(
        validation_alias=AliasChoices("exposed_steps", "high_risk_steps"),
    )
    mean_semantic_distance: float    # average best-candidate distance across steps
    diffusion_zone_ratio: float      # fraction of steps classified as diffusion
    # Scalar reading value ("Exposure"): fraction of analyzed steps counted
    # exposed — len(exposed_steps) / n analyzed steps, in [0, 1].
    # Chosen over a weighted composite because it is directly localizable (each
    # counted step is inspectable) and its threshold is explicit
    # (DEFAULT_DISTANCE_THRESHOLD + diffusion-zone membership). It measures
    # counterfactual semantic exposure, NOT factuality.
    exposure: float = 0.0
    # Name of the embedding model used to compute semantic distances. Distances
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
        """Run counterfactual exposure analysis on an output trace.

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
            Cosine distance at or above which a diffusion-zone step is
            counted as exposed.

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

            # Cloud position of the divergent alternative (index into pre-projected 2D)
            cloud_pos: list[float] = []
            if sem.embeddings_2d and best_rank < len(sem.embeddings_2d):
                cloud_pos = list(sem.embeddings_2d[best_rank])

            candidates.append(ExposureCandidate(
                step=i,
                selected_token=step.selected_token_str,
                selected_prob=step.topk[0].prob if step.topk else 0.0,
                divergent_token=best_entry.token_str,
                divergent_prob=best_entry.prob,
                prob_rank=best_rank,
                semantic_distance=best_dist,
                cloud_phenomenon=phenomenon,
                cloud_position_2d=cloud_pos,
            ))

        embedder_name = getattr(self._embedder, "model_name", None) or None

        if not candidates:
            return ExposureProfile(
                candidates=[],
                exposed_steps=[],
                mean_semantic_distance=0.0,
                diffusion_zone_ratio=0.0,
                exposure=0.0,
                embedder=embedder_name,
            )

        exposed = [
            c.step for c in candidates
            if c.cloud_phenomenon == "diffusion" and c.semantic_distance >= distance_threshold
        ]
        mean_dist = float(np.mean([c.semantic_distance for c in candidates]))
        n_diffusion = sum(1 for c in candidates if c.cloud_phenomenon == "diffusion")
        diffusion_ratio = n_diffusion / len(candidates)

        logger.info(
            "Exposure: %d of %d steps had a semantically divergent accessible "
            "alternative in a diffuse cloud (mean distance %.2f)",
            len(exposed),
            len(candidates),
            mean_dist,
        )
        logger.debug(
            "Exposure detail: diffusion zone=%.0f%% of steps",
            diffusion_ratio * 100,
        )

        return ExposureProfile(
            candidates=candidates,
            exposed_steps=exposed,
            mean_semantic_distance=mean_dist,
            diffusion_zone_ratio=diffusion_ratio,
            exposure=len(exposed) / len(candidates),
            embedder=embedder_name,
        )
