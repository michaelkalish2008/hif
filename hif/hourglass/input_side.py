"""Input-side analysis: teacher-forced surprisal and entropy over prompt tokens."""

import numpy as np
from pydantic import BaseModel

from hif.models.base import Model
from hif.utils.logging import get_logger

logger = get_logger(__name__)


class PositionRecord(BaseModel):
    position: int          # token index in the prompt (1-indexed; position 0 has no predecessor)
    token_id: int
    token_str: str
    surprisal: float       # -log2(p(token_id | context))
    entropy: float         # Shannon entropy of full vocab distribution at this position (bits)
    top_k_alternatives: list[dict]  # list of {token_id, token_str, prob} for top K alternatives


class InputSideAnalysis(BaseModel):
    positions: list[PositionRecord]  # one per token position (positions 1..N-1)
    prompt_token_ids: list[int]
    prompt_text: str
    mean_surprisal: float
    mean_entropy: float
    max_entropy: float      # log2(vocab_size) — theoretical maximum for this tokenizer
    # DEPRECATED, retained only so older profile JSON still validates. This is
    # mean_entropy / log2(vocab_size): a behavioural number divided by a piece
    # of tokenizer metadata. Nothing in hif computes from it any more — read
    # mean_entropy (bits) instead.
    volatility_score: float


def mean_surprisal_excess(positions: list["PositionRecord"]) -> float | None:
    """Mean max(0, surprisal - entropy) over positions (bits).

    The residual
    cost of the actual token beyond what the distribution's own entropy
    already expected. Unlike raw mean_surprisal, it isn't inflated by a
    model's baseline uncertainty. None if there are no teacher-forced
    positions to measure.
    """
    if not positions:
        return None
    return float(np.mean([max(0.0, p.surprisal - p.entropy) for p in positions]))


def analyze_input_side(
    model: Model,
    prompt: str,
    top_k: int = 50,
) -> InputSideAnalysis:
    """Run teacher-forced input analysis on a prompt.

    For each token position i from 1 to N-1, uses the logits at position i-1
    (which predict position i) to compute surprisal and entropy.

    Args:
        model: A Model instance with supports_teacher_forcing == True.
        prompt: The text prompt to analyze.
        top_k: Number of top-K alternative tokens to record per position.

    Returns:
        InputSideAnalysis with per-position records and aggregate metrics.

    Raises:
        NotImplementedError: If model.supports_teacher_forcing is False.
    """
    if not model.supports_teacher_forcing:
        raise NotImplementedError(
            f"Model '{model.name}' does not support teacher forcing. "
            "Input-side analysis requires access to full-vocabulary logits at every "
            "position. Use an HFModel or TLensModel backend instead."
        )

    token_ids = model.tokenize(prompt)
    n = len(token_ids)

    # Need at least 2 tokens to have one predictable position
    logits_result = model.forward(token_ids)
    logits_np = logits_result.to_numpy()  # shape (n, vocab_size)

    vocab_size = logits_result.vocab_size
    max_entropy = float(np.log2(vocab_size))

    positions: list[PositionRecord] = []

    for i in range(1, n):
        # Logits at position i-1 predict the token at position i
        logits_i = logits_np[i - 1]  # shape (vocab_size,)

        # Numerically stable softmax
        logits_shifted = logits_i - logits_i.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()

        # log2 probabilities (stable)
        # Use log-sum-exp for logprobs
        log_sum_exp = np.log(exp_logits.sum()) + logits_i.max() - logits_i.max()
        # simpler: logprobs = logits_shifted - log(sum(exp(logits_shifted)))
        log_sum = np.log(exp_logits.sum())
        logprobs2 = (logits_shifted - log_sum) / np.log(2)  # log2 probs

        # Surprisal for the observed token
        surprisal = float(-logprobs2[token_ids[i]])

        # Shannon entropy in bits
        p_clip = np.clip(probs, 1e-10, 1.0)
        entropy = float(-np.sum(p_clip * np.log2(p_clip)))

        # Top-K alternatives by probability
        k = min(top_k, vocab_size)
        top_indices = np.argpartition(probs, -k)[-k:]
        top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

        top_k_alternatives = [
            {
                "token_id": int(idx),
                "token_str": model.detokenize([int(idx)]),
                "prob": float(probs[idx]),
            }
            for idx in top_indices
        ]

        positions.append(
            PositionRecord(
                position=i,
                token_id=token_ids[i],
                token_str=model.detokenize([token_ids[i]]),
                surprisal=surprisal,
                entropy=entropy,
                top_k_alternatives=top_k_alternatives,
            )
        )

    if positions:
        mean_surprisal = float(np.mean([p.surprisal for p in positions]))
        mean_entropy = float(np.mean([p.entropy for p in positions]))
    else:
        mean_surprisal = 0.0
        mean_entropy = 0.0

    volatility_score = mean_entropy / max_entropy if max_entropy > 0 else 0.0

    return InputSideAnalysis(
        positions=positions,
        prompt_token_ids=token_ids,
        prompt_text=prompt,
        mean_surprisal=mean_surprisal,
        mean_entropy=mean_entropy,
        max_entropy=max_entropy,
        volatility_score=volatility_score,
    )


def analyze_input_side_mm(
    model,
    prepared,
    prompt_text: str,
    top_k: int = 50,
) -> InputSideAnalysis:
    """Teacher-forced input analysis over a prepared multimodal sequence.

    Entropy/surprisal are computed ONLY over positions in
    prepared.part_map.text_positions() — patch/placeholder positions have no
    meaningful vocab distribution and are excluded from mean_entropy/
    volatility (MULTIMODAL.md Design §2 and Risk rule 3).

    Args:
        model: A MultimodalModel with supports_teacher_forcing == True.
        prepared: PreparedInput from model.prepare().
        prompt_text: MultimodalInput.text_concat (recorded verbatim).
        top_k: Number of top-K alternative tokens to record per position.
    """
    if not model.supports_teacher_forcing:
        raise NotImplementedError(
            f"Model '{model.name}' does not support teacher forcing. "
            "Multimodal input-side analysis requires forward_prepared logits."
        )

    token_ids = list(prepared.input_ids)
    text_positions = set(prepared.part_map.text_positions())

    logits_result = model.forward_prepared(prepared)
    logits_np = logits_result.to_numpy()

    vocab_size = logits_result.vocab_size
    max_entropy = float(np.log2(vocab_size))

    positions: list[PositionRecord] = []

    for i in range(1, len(token_ids)):
        # Only positions whose predicted token belongs to a text-kind span.
        if i not in text_positions:
            continue

        logits_i = logits_np[i - 1]

        logits_shifted = logits_i - logits_i.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()
        log_sum = np.log(exp_logits.sum())
        logprobs2 = (logits_shifted - log_sum) / np.log(2)

        surprisal = float(-logprobs2[token_ids[i]])

        p_clip = np.clip(probs, 1e-10, 1.0)
        entropy = float(-np.sum(p_clip * np.log2(p_clip)))

        k = min(top_k, vocab_size)
        top_indices = np.argpartition(probs, -k)[-k:]
        top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

        top_k_alternatives = [
            {
                "token_id": int(idx),
                "token_str": model.detokenize([int(idx)]),
                "prob": float(probs[idx]),
            }
            for idx in top_indices
        ]

        positions.append(
            PositionRecord(
                position=i,
                token_id=token_ids[i],
                token_str=model.detokenize([token_ids[i]]),
                surprisal=surprisal,
                entropy=entropy,
                top_k_alternatives=top_k_alternatives,
            )
        )

    if positions:
        mean_surprisal = float(np.mean([p.surprisal for p in positions]))
        mean_entropy = float(np.mean([p.entropy for p in positions]))
    else:
        mean_surprisal = 0.0
        mean_entropy = 0.0

    volatility_score = mean_entropy / max_entropy if max_entropy > 0 else 0.0

    return InputSideAnalysis(
        positions=positions,
        prompt_token_ids=token_ids,
        prompt_text=prompt_text,
        mean_surprisal=mean_surprisal,
        mean_entropy=mean_entropy,
        max_entropy=max_entropy,
        volatility_score=volatility_score,
    )
