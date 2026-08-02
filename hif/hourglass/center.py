"""Horizon structure — center zone: relational diagnostics between input-side and output-side analyses.

The center is the model's computation, treated as a black box. This module computes
relational diagnostics — no activation inspection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel

from hif.clustering.embed import EmbeddingModel
from hif.hourglass.input_side import InputSideAnalysis
from hif.hourglass.output_side import OutputSideTrace
from hif.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CenterDiagnostics(BaseModel):
    input_mean_entropy: float        # from InputSideAnalysis
    output_mean_entropy: float       # from OutputSideTrace (re-computed from top-K probs)
    entropy_ratio: Optional[float] = None  # output / input (both bits); None for API models without input-side data
    # Cosine distance between the prompt embedding and the generated-text
    # embedding. Bounded to [0, 2] by definition. Named for what it measures:
    # it is a distance between two embeddings, not evidence of drift.
    prompt_output_cosine_distance: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity(a, b). Returns 1.0 for zero vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def _output_mean_entropy(output_trace: OutputSideTrace) -> float:
    """Re-compute mean entropy from normalized top-K probs in each StepRecord."""
    step_entropies: list[float] = []
    for step in output_trace.steps:
        probs = np.array([entry.prob for entry in step.topk], dtype=np.float64)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        probs = np.clip(probs, 1e-10, 1.0)
        h = float(-np.sum(probs * np.log2(probs)))
        step_entropies.append(h)
    return float(np.mean(step_entropies)) if step_entropies else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_center_diagnostics(
    input_analysis: InputSideAnalysis,
    output_trace: OutputSideTrace,
    embedder: EmbeddingModel,
    max_entropy: float,           # log2(vocab_size)
) -> CenterDiagnostics:
    """Compute relational diagnostics between input-side and output-side analyses.

    Parameters
    ----------
    input_analysis:
        Result of analyze_input_side().
    output_trace:
        Result of collect_output_trace().
    embedder:
        EmbeddingModel used to embed prompt text and generated text.
    max_entropy:
        Theoretical maximum entropy = log2(vocab_size). Recorded for reference only;
        nothing here is normalised by it.
    """
    # 1. Input entropy (from InputSideAnalysis)
    input_mean_entropy = input_analysis.mean_entropy

    # 2. Output entropy (re-computed from top-K normalized distributions)
    output_mean_entropy = _output_mean_entropy(output_trace)

    # 3. Entropy ratio
    if input_mean_entropy > 0:
        entropy_ratio = output_mean_entropy / input_mean_entropy
    else:
        entropy_ratio = float("inf") if output_mean_entropy > 0 else 1.0

    # 4. Cosine distance between prompt embedding and generated text embedding.
    # There is deliberately no "equilibrium" classification here any more: it
    # thresholded output entropy against 0.1/0.9 x log2(vocab_size), i.e. it
    # bucketed behaviour by a property of the tokenizer.
    prompt_text = input_analysis.prompt_text

    # Build generated text from all generated tokens
    if output_trace.generated_ids:
        # We use the output trace's steps to reconstruct the generated text string
        generated_tokens = [step.selected_token_str for step in output_trace.steps]
        generated_text = "".join(generated_tokens)
    else:
        generated_text = ""

    if generated_text:
        prompt_emb = embedder.embed_single(prompt_text)
        output_emb = embedder.embed_single(generated_text)
        prompt_output_cosine_distance = _cosine_distance(prompt_emb, output_emb)
    else:
        prompt_output_cosine_distance = 0.0

    logger.debug(
        "CenterDiagnostics: in_entropy=%.3f out_entropy=%.3f ratio=%.3f cos_dist=%.3f",
        input_mean_entropy,
        output_mean_entropy,
        entropy_ratio,
        prompt_output_cosine_distance,
    )

    return CenterDiagnostics(
        input_mean_entropy=input_mean_entropy,
        output_mean_entropy=output_mean_entropy,
        entropy_ratio=entropy_ratio,
        prompt_output_cosine_distance=prompt_output_cosine_distance,
    )
