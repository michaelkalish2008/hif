"""Output-side analysis: collects and characterizes the model's generation trace."""

import numpy as np
from pydantic import BaseModel

from hif.models.base import Model, StepRecord, TopKEntry
from hif.utils.logging import get_logger

logger = get_logger(__name__)

# Dedup key set for the "top_k exceeds max_top_k" note: (model_name, requested_top_k, max_top_k).
# Multiple perturbation variants within a single profile run otherwise re-trigger this
# once per generation call; we only want to log once per run per model/top_k.
#
# The clamp itself is silent by default: it fires on 100% of runs against
# backends with a hard logprob cap (max_top_k=20), so it is an expected
# adjustment, not an anomaly. The EFFECTIVE top_k is recorded in the artifact
# (OutputSideTrace.top_k / GenerationResult.top_k; the REQUESTED value stays in
# config.generation.top_k) and surfaced as one line in `--verbose` stats; the
# log line below is visible under --verbose only.
_warned_top_k_combos: set[tuple[str, int, int]] = set()


def _warn_top_k_once(model: Model, top_k: int) -> None:
    key = (getattr(model, "name", repr(model)), top_k, model.max_top_k)
    if key in _warned_top_k_combos:
        return
    _warned_top_k_combos.add(key)
    logger.debug(
        f"Requested top_k={top_k} exceeds model's max_top_k={model.max_top_k}. "
        f"Reducing top_k to {model.max_top_k}."
    )


class OutputSideTrace(BaseModel):
    steps: list[StepRecord]      # one per generated token
    input_ids: list[int]
    generated_ids: list[int]
    prompt_text: str
    model_name: str
    top_k: int
    max_new_tokens: int
    seed: int
    mean_step_entropy: float     # mean Shannon entropy of the normalized top-K distribution across steps
    # Note: entropy is computed over the truncated top-K distribution (truncated=True downstream)


def collect_output_trace(
    model: Model,
    prompt: str,
    max_new_tokens: int = 64,
    top_k: int = 50,
    seed: int = 42,
) -> OutputSideTrace:
    """Generate tokens from the model and return a full output trace.

    Args:
        model: A Model instance to generate from.
        prompt: The text prompt to condition on.
        max_new_tokens: Maximum number of tokens to generate.
        top_k: Number of top-K candidates to record at each step.
        seed: Random seed for reproducible generation.

    Returns:
        OutputSideTrace with per-step records and aggregate entropy metric.
    """
    # Enforce top_k <= model.max_top_k if the model has a hard cap
    if model.max_top_k is not None and top_k > model.max_top_k:
        _warn_top_k_once(model, top_k)
        top_k = model.max_top_k

    input_ids = model.tokenize(prompt)

    result = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
    )

    # Compute mean_step_entropy using nucleus entropy (95% mass, renormalized).
    # Nucleus entropy is comparable across model types regardless of top-k size.
    from hif.metrics.distribution import nucleus_entropy_bits
    step_entropies: list[float] = []
    for step in result.steps:
        probs = np.array([entry.prob for entry in step.topk], dtype=np.float64)
        step_entropies.append(nucleus_entropy_bits(probs, p=0.95))

    mean_step_entropy = float(np.mean(step_entropies)) if step_entropies else 0.0

    return OutputSideTrace(
        steps=result.steps,
        input_ids=result.input_ids,
        generated_ids=result.generated_ids,
        prompt_text=prompt,
        model_name=result.model_name,
        top_k=result.top_k,
        max_new_tokens=max_new_tokens,
        seed=result.seed,
        mean_step_entropy=mean_step_entropy,
    )


def output_distribution_degenerate(steps: list[StepRecord]) -> bool:
    """True when every recorded output step carries only the selected token
    (topk length <= 1) — i.e. the backend returned no real logprobs at
    generation time (e.g. Anthropic), so any entropy computed directly from
    it is 0.0 by construction rather than measured.
    """
    return bool(steps) and all(len(s.topk) <= 1 for s in steps)


def output_steps_via_surrogate(
    surrogate_model: Model,
    prompt: str,
    continuation_text: str,
    top_k: int = 50,
) -> list[StepRecord]:
    """Teacher-force `surrogate_model` over prompt+continuation and return one
    StepRecord (selected token + real top-k alternatives) per continuation
    position, in the surrogate's own tokenization.

    Used when the target model exposes no per-step distribution at all (e.g.
    Anthropic's API): the surrogate never produced this output, but reading
    the same text still recovers *a* plausible next-token distribution and a
    set of real alternative tokens at each position — the same proxy
    technique already used for the input-side Stability/Surprise/Wager
    readings, applied to the continuation instead of the prompt. Because the
    surrogate is teacher-forced over the ACTUAL generated text, its own
    "selected" token at each position is a faithful (if differently
    segmented) decomposition of what was really said — not fabricated.

    These StepRecords are a drop-in replacement for the degenerate
    OutputSideTrace.steps: everything downstream that reads step.topk
    (distribution metrics, semantic clustering, exposure analysis) works
    unmodified once given these instead of the target's one-entry steps.

    Empty list if the continuation retokenizes to no new positions under the
    surrogate.
    """
    if not continuation_text:
        return []
    prompt_ids = surrogate_model.tokenize(prompt)
    full_ids = surrogate_model.tokenize(prompt + continuation_text)
    split = len(prompt_ids)
    if len(full_ids) <= split:
        return []

    logits_result = surrogate_model.forward(full_ids)
    logits_np = logits_result.to_numpy()  # shape (n, vocab_size)

    steps: list[StepRecord] = []
    for step_idx, i in enumerate(range(split, len(full_ids))):
        logits_i = logits_np[i - 1]  # predicts token at position i
        shifted = logits_i - logits_i.max()
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum()
        # log-softmax via the same shift, numerically stable
        logprobs = shifted - np.log(exp_logits.sum())
        order = np.argsort(probs)[::-1][: min(top_k, len(probs))]
        topk_entries = [
            TopKEntry(
                token_id=int(idx),
                token_str=surrogate_model.detokenize([int(idx)]),
                logit=float(logits_i[idx]),
                logprob=float(logprobs[idx]),
                prob=float(probs[idx]),
            )
            for idx in order
        ]
        selected_token_id = int(full_ids[i])
        steps.append(StepRecord(
            step=step_idx,
            selected_token_id=selected_token_id,
            selected_token_str=surrogate_model.detokenize([selected_token_id]),
            topk=topk_entries,
        ))
    return steps


def collect_output_trace_mm(
    model,
    prepared,
    prompt_text: str,
    max_new_tokens: int = 64,
    top_k: int = 50,
    seed: int = 42,
) -> OutputSideTrace:
    """Multimodal counterpart of collect_output_trace.

    Uses MultimodalModel.generate_prepared on an already-prepared input
    (tokenize is never called with media — Design §1-2,
    docs/ARCHITECTURE.md § Multimodal notes).
    input_ids in the trace are the full processed sequence ids (placeholder
    ids for patches included).

    Args:
        model: A MultimodalModel instance.
        prepared: PreparedInput from model.prepare().
        prompt_text: MultimodalInput.text_concat (recorded verbatim).
    """
    if model.max_top_k is not None and top_k > model.max_top_k:
        _warn_top_k_once(model, top_k)
        top_k = model.max_top_k

    result = model.generate_prepared(
        prepared,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
    )

    from hif.metrics.distribution import nucleus_entropy_bits
    step_entropies: list[float] = []
    for step in result.steps:
        probs = np.array([entry.prob for entry in step.topk], dtype=np.float64)
        step_entropies.append(nucleus_entropy_bits(probs, p=0.95))

    mean_step_entropy = float(np.mean(step_entropies)) if step_entropies else 0.0

    return OutputSideTrace(
        steps=result.steps,
        input_ids=result.input_ids,
        generated_ids=result.generated_ids,
        prompt_text=prompt_text,
        model_name=result.model_name,
        top_k=result.top_k,
        max_new_tokens=max_new_tokens,
        seed=result.seed,
        mean_step_entropy=mean_step_entropy,
    )
