"""Anthropic API backend for closed-model profiling.

Supports Claude 3, 3.5, 3.7, and 4 families.

Constraints:
- supports_teacher_forcing = False   (API only returns generated text)
- max_top_k = None                   (Anthropic does not expose logprobs)
- tokenize() via tiktoken cl100k_base (approximate — not Claude's true tokenizer)
- distribution metrics will be degenerate: only the selected token is recorded
  at prob=1.0 per step; entropy, nucleus fraction etc. will reflect this.

This backend is useful for:
- Comparing generation outputs and sensitivity across prompt types
- Running perturbation analysis (JSD on the text-level behaviour)
- Attention analysis via the surrogate model pipeline
"""

from __future__ import annotations

import os
from typing import Optional

from hif.config import ModelConfig
from hif.models.base import GenerationResult, Logits, Model, StepRecord, TopKEntry
from hif.utils.logging import get_logger

logger = get_logger(__name__)

_CONTEXT_MAP: dict[str, int] = {
    # Claude 4 family
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    # Claude 3.x family
    "claude-3-7-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
}

# Approximate — Claude uses its own BPE; cl100k_base is the closest public tokenizer
_APPROX_VOCAB_SIZE = 100_000
_APPROX_CONTEXT = 200_000


class AnthropicModel(Model):
    """Anthropic Messages API backend.

    NOTE: Anthropic does not expose token-level logprobs. All distribution
    metrics (entropy, nucleus fraction, effective support) will be computed
    from a degenerate distribution where prob=1.0 for the selected token.
    Use OpenAI or Gemini backends for richer distribution analysis.
    """

    def __init__(self, config: ModelConfig) -> None:
        try:
            import tiktoken
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend requires 'anthropic' and 'tiktoken'. "
                "Run: pip install anthropic tiktoken"
            ) from exc

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY in your environment or .env file."
            )

        self._config = config
        self._client = Anthropic(api_key=api_key)
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._context_length_val = next(
            (v for k, v in _CONTEXT_MAP.items() if config.name.startswith(k)),
            _APPROX_CONTEXT,
        )
        self._last_prompt_text: str = ""

        logger.warning(
            "AnthropicModel: logprobs are not available via the Anthropic API. "
            "Distribution metrics will be degenerate (prob=1.0 per selected token). "
            "Sensitivity and perturbation metrics based on output text will still work."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def vocab_size(self) -> int:
        return _APPROX_VOCAB_SIZE

    @property
    def context_length(self) -> int:
        return self._context_length_val

    @property
    def max_top_k(self) -> Optional[int]:
        return 1  # only the selected token is available

    @property
    def supports_teacher_forcing(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Tokenization (approximate)
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        self._last_prompt_text = text
        return list(self._enc.encode(text))

    def detokenize(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    # ------------------------------------------------------------------
    # Forward pass — not supported
    # ------------------------------------------------------------------

    def forward(self, input_ids: list[int]) -> Logits:
        raise NotImplementedError(
            "AnthropicModel does not support teacher-forced input analysis."
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        top_k: int,
        seed: int,
    ) -> GenerationResult:
        prompt_text = self._last_prompt_text or self.detokenize(input_ids)

        logger.info("Anthropic generate: model=%s max_tokens=%d", self.name, max_new_tokens)

        create_kwargs: dict = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_new_tokens,
        }
        # Newer reasoning-capable models (e.g. claude-opus-4-8, claude-fable-5)
        # reject an explicit temperature param outright ("temperature is
        # deprecated for this model") — only send it when the study config
        # asks for a specific value; let the API default apply otherwise.
        if self._config.temperature is not None:
            create_kwargs["temperature"] = self._config.temperature
        response = self._client.messages.create(**create_kwargs)

        # Extended-thinking models (claude-opus-4-8, claude-fable-5) can put a
        # ThinkingBlock before the TextBlock in response.content — content[0]
        # is not reliably the text block, so scan for the first block that
        # actually has one instead of assuming position 0.
        generated_text: str = next(
            (block.text for block in (response.content or [])
             if hasattr(block, "text")),
            "",
        )

        # Tokenize the output to build per-step records.
        # Without logprobs we create degenerate StepRecords (prob=1.0 for chosen token).
        output_tokens = self._enc.encode(generated_text)
        steps: list[StepRecord] = []
        for step_idx, tok_id in enumerate(output_tokens):
            tok_str = self._enc.decode([tok_id])
            steps.append(StepRecord(
                step=step_idx,
                selected_token_id=tok_id,
                selected_token_str=tok_str,
                topk=[TopKEntry(
                    token_id=tok_id,
                    token_str=tok_str,
                    logit=0.0,
                    logprob=0.0,
                    prob=1.0,
                )],
            ))

        # Why it stopped, as the API says it ("end_turn", "max_tokens",
        # "stop_sequence", "refusal"). Recorded on every call, and load-bearing
        # on the empty one: the scan above returns "" when the response carries
        # no TextBlock at all, which an extended-thinking model can produce by
        # spending max_tokens inside a ThinkingBlock. That yields zero steps,
        # and zero steps with no reason is an absence a reader has to guess at.
        stop_reason = getattr(response, "stop_reason", None)
        if not steps:
            logger.warning(
                "%s returned no text (stop_reason=%s). Every output-side "
                "measurement is absent for this run.",
                self.name, stop_reason or "not reported",
            )

        return GenerationResult(
            input_ids=input_ids,
            generated_ids=output_tokens,
            steps=steps,
            model_name=self.name,
            top_k=1,
            seed=seed,
            stop_reason=stop_reason,
        )
