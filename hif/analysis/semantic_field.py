"""Within-generation semantic field instrument — "Veer" (◈).

DRIFT_FIELD_MODEL.md § 11.3. The six admitted instruments read one generation
event; this adds a seventh that reads the *trajectory* of the output's semantic
possibility field WITHIN a single generation. At each step the top-K candidate
cloud has a probability-weighted centroid in embedding space. As generation
proceeds that centroid traces a path:

- **Veer** (translation) — the step-to-step displacement of the centroid: how far
  the semantic center of the possibility field moved between consecutive steps.
  Small, steady Veer = coherent semantic development; large jumps = topic pivots.
- **Deformation** — the step-to-step change in the field's spread (probability-
  weighted mean distance of candidates from the centroid): the field widening or
  narrowing / fragmenting.

Veer is the GEOMETRIC twin of Shift (◆): Shift reads step-to-step change in the
distribution's *spread* in vocabulary space (information-theoretic); Veer reads
step-to-step change in the distribution's *semantic location* in embedding space
(geometric). `center.semantic_drift` gives only the prompt→output endpoint; Veer
is the per-step trace.

Privacy invariant: compute-and-discard. Candidate embeddings and the per-step
centroids live only in this call's stack frame; only the scalar per-step traces
(cosine distances) are returned. No embedding, centroid, or token identity is
persisted. Sample-only, scoring-time-only; do not import from a request path.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from hif.clustering.embed import EmbeddingModel
from hif.hourglass.output_side import OutputSideTrace
from hif.metrics.semantic import cosine_distance


class SemanticFieldReading(BaseModel):
    """Per-step within-generation semantic-field traces (scalars only)."""

    veer: list[float]          # step-to-step centroid displacement (cosine dist), length G-1
    deformation: list[float]   # step-to-step field-spread change, length G-1
    mean_veer: float
    max_veer: float
    mean_deformation: float
    n_steps: int               # number of generation steps with a defined centroid


def _weighted_centroid(embeddings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    total = probs.sum()
    w = probs / total if total > 0 else np.full(len(probs), 1.0 / max(1, len(probs)))
    return w @ embeddings


def _dispersion(embeddings: np.ndarray, probs: np.ndarray, centroid: np.ndarray) -> float:
    """Probability-weighted mean cosine distance of candidates from the centroid."""
    total = probs.sum()
    if total <= 0:
        return 0.0
    w = probs / total
    return float(sum(wi * cosine_distance(embeddings[i], centroid) for i, wi in enumerate(w)))


class SemanticFieldAnalyzer:
    """Computes the Veer reading from an output trace (compute-and-discard)."""

    def __init__(self, embedder: EmbeddingModel, context_window: int = 5):
        self.embedder = embedder
        self.context_window = context_window

    def analyze(self, output_trace: OutputSideTrace) -> SemanticFieldReading | None:
        steps = output_trace.steps
        if len(steps) < 2:
            return None

        centroids: list[np.ndarray] = []
        dispersions: list[float] = []
        for i, step in enumerate(steps):
            if not step.topk:
                centroids.append(None)  # type: ignore[arg-type]
                dispersions.append(0.0)
                continue
            # Same candidate-context construction as the semantic-metrics pipeline:
            # prepend up to `context_window` already-generated tokens so the
            # embedding reflects the candidate in context, not the bare token.
            cw = min(self.context_window, i)
            prefix = "".join(s.selected_token_str for s in steps[i - cw:i]) if cw > 0 else ""
            candidate_strings = [prefix + e.token_str for e in step.topk]
            probs = np.array([e.prob for e in step.topk], dtype=np.float64)
            embeddings = np.asarray(self.embedder.embed(candidate_strings), dtype=np.float64)
            centroid = _weighted_centroid(embeddings, probs)
            centroids.append(centroid)
            dispersions.append(_dispersion(embeddings, probs, centroid))
            # embeddings fall out of scope here — never persisted.

        veer: list[float] = []
        deformation: list[float] = []
        for j in range(1, len(centroids)):
            a, b = centroids[j - 1], centroids[j]
            if a is None or b is None:
                continue
            veer.append(cosine_distance(a, b))
            deformation.append(abs(dispersions[j] - dispersions[j - 1]))

        if not veer:
            return None
        return SemanticFieldReading(
            veer=veer,
            deformation=deformation,
            mean_veer=float(np.mean(veer)),
            max_veer=float(np.max(veer)),
            mean_deformation=float(np.mean(deformation)) if deformation else 0.0,
            n_steps=sum(1 for c in centroids if c is not None),
        )
