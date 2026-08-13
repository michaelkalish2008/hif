"""OpenAI API backend for closed-model profiling.

Supports GPT-4.1, GPT-4.1-mini, GPT-4o, GPT-4o-mini, GPT-5, and OpenAI-compatible
APIs (Mistral, DeepSeek) via ModelConfig.base_url.

Constraints:
- supports_teacher_forcing = False  (API only returns generated tokens)
- max_top_k = 20                    (OpenAI hard cap on top_logprobs)
- tokenize() via tiktoken           (accurate for GPT families; cl100k fallback for others)
- generate() via chat completions   (temperature=0 for determinism)
"""

from __future__ import annotations

import math
import os
from typing import Optional

from hif.config import ModelConfig
from hif.models.base import GenerationResult, Logits, Model, StepRecord, TopKEntry
from hif.utils.logging import get_logger

logger = get_logger(__name__)

# tiktoken encoding per model family (prefix-matched)
_ENCODING_MAP: dict[str, str] = {
    "gpt-5": "o200k_base",          # gpt-5, gpt-5-mini
    "gpt-4.1": "o200k_base",        # gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
    "gpt-4o": "o200k_base",         # gpt-4o, gpt-4o-mini
    "o1": "o200k_base",             # o1, o1-mini, o1-preview
    "o3": "o200k_base",             # o3, o3-mini
    "o4": "o200k_base",             # o4-mini
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    # Non-OpenAI models via compatible APIs — tiktoken is an approximation only
    "mistral": "cl100k_base",
    "open-mistral": "cl100k_base",
    "codestral": "cl100k_base",
    "deepseek": "cl100k_base",
}

_CONTEXT_MAP: dict[str, int] = {
    "gpt-5": 1_047_576,             # gpt-5 family
    "gpt-4.1": 1_047_576,           # gpt-4.1 family
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    "o4": 200_000,
    "mistral": 128_000,
    "open-mistral": 128_000,
    "deepseek": 64_000,
}

_VOCAB_MAP: dict[str, int] = {
    "o200k_base": 200_019,
    "cl100k_base": 100_277,
}


def _enc_name(model_name: str) -> str:
    for prefix, enc in _ENCODING_MAP.items():
        if model_name.startswith(prefix):
            return enc
    return "cl100k_base"  # safe default


class OpenAIModel(Model):
    """OpenAI chat-completion backend."""

    def __init__(self, config: ModelConfig) -> None:
        # Set the first time the API refuses a logprob request, so the rest of
        # the run skips straight to the no-logprobs path. Per instance, not per
        # class: whether a model exposes logprobs is a fact about that model,
        # and the same process may profile several.
        self._refuses_logprobs = False
        try:
            import tiktoken
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI backend requires 'openai' and 'tiktoken'. "
                "Run: pip install openai tiktoken"
            ) from exc

        # Resolve API key — backends keyed by base_url use their own env var names
        api_key = config.api_key
        if not api_key:
            if config.base_url and "mistral" in config.base_url:
                api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
            elif config.base_url and "deepseek" in config.base_url:
                api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
            elif config.base_url and "x.ai" in config.base_url:
                api_key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. For OpenAI: OPENAI_API_KEY. "
                "For Mistral: MISTRAL_API_KEY. For DeepSeek: DEEPSEEK_API_KEY. "
                "For Grok: XAI_API_KEY."
            )

        self._config = config
        client_kwargs: dict = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = OpenAI(**client_kwargs)
        enc_name = _enc_name(config.name)
        self._enc = tiktoken.get_encoding(enc_name)
        self._vocab_size_val = _VOCAB_MAP.get(enc_name, 100_277)
        self._context_length_val = next(
            (v for k, v in _CONTEXT_MAP.items() if config.name.startswith(k)), 8_192
        )
        self._last_prompt_text: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_val

    @property
    def context_length(self) -> int:
        return self._context_length_val

    @property
    def max_top_k(self) -> Optional[int]:
        return 20  # OpenAI hard cap

    @property
    def supports_teacher_forcing(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Tokenization
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
            "OpenAIModel does not support teacher-forced input analysis."
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
        return self._generate_from_messages(
            messages=[{"role": "user", "content": prompt_text}],
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            seed=seed,
        )

    def _generate_from_messages(
        self,
        messages: list[dict],
        input_ids: list[int],
        max_new_tokens: int,
        top_k: int,
        seed: int,
    ) -> GenerationResult:
        """Chat-completions call with logprobs, mapped to GenerationResult steps.

        Shared by the text path (generate) and the vision adapter
        — the messages payload is the only
        difference between the two.
        """
        effective_k = min(top_k, self.max_top_k)

        # Ask once. A model that refuses logprobs refuses every time, and the
        # refusal costs a full round trip plus a retry — on a profile run that
        # is one wasted request per generation, sixteen per prompt with the
        # perturbation variants, and it was the difference between a ~30s
        # gpt-4.1-mini run and a 626–1056s gpt-5 one.
        if self._refuses_logprobs:
            return self._generate_no_logprobs(input_ids, max_new_tokens, seed, messages=messages)

        logger.info("OpenAI generate: model=%s tokens=%d top_logprobs=%d", self.name, max_new_tokens, effective_k)

        # Per-model override wins; otherwise the default depends on the provider.
        # OpenAI gets 0 for determinism. DeepSeek gets 1: at temperature=0 it
        # returns -9999 sentinel logprobs for every non-selected candidate, so
        # the "distribution" is a point mass wearing twenty entries. The old
        # comment here told the caller to set temperature=1 in ModelConfig —
        # advice at the exact place with the power to just do it. The regen
        # pipeline never read the comment, profiled two DeepSeek models at
        # temperature=0, and published output_entropy_bits = 0.0 for all
        # sixteen profiles.
        temperature = self._config.temperature
        if temperature is None:
            is_deepseek = bool(self._config.base_url and "deepseek" in self._config.base_url)
            temperature = 1.0 if is_deepseek else 0.0

        extra = self._config.extra_body or None
        try:
            response = self._client.chat.completions.create(
                model=self.name,
                messages=messages,
                max_tokens=max_new_tokens,
                logprobs=True,
                top_logprobs=effective_k,
                seed=seed,
                temperature=temperature,
                **({"extra_body": extra} if extra else {}),
            )
        except Exception as exc:
            # Reasoning models (GPT-5, o-series) and some compatible APIs reject logprob requests
            # or don't support temperature/seed. Fall back to degenerate mode.
            exc_str = str(exc).lower()
            is_capability_limit = (
                "logprob" in exc_str
                or "not allowed" in exc_str
                or "unsupported parameter" in exc_str
                or "temperature" in exc_str
                or "403" in str(exc)
            )
            if is_capability_limit:
                # Remember it, so the rest of the run stops asking. Logged once
                # at warning; repeats would be the same fact many times over.
                if not self._refuses_logprobs:
                    logger.warning(
                        "Model %s refuses logprobs — degenerate mode for the rest of this run: %s",
                        self.name, str(exc)[:120],
                    )
                self._refuses_logprobs = True
                return self._generate_no_logprobs(input_ids, max_new_tokens, seed, messages=messages)
            raise

        content_logprobs = (
            response.choices[0].logprobs.content
            if response.choices[0].logprobs
            else []
        ) or []

        steps: list[StepRecord] = []
        for step_idx, token_lp in enumerate(content_logprobs):
            selected_str = token_lp.token
            selected_lp = token_lp.logprob
            top_entries = token_lp.top_logprobs or []

            # Normalise to probabilities (top_logprobs may not sum to 1.0)
            raw = [(e.token, e.logprob) for e in top_entries]
            # Sentinel filter. Some providers fill the non-selected entries
            # with logprob ≈ -9999 instead of refusing the request. Softmaxed,
            # that is a point mass with a straight face: entropy 0.0, JSD 0.0,
            # all "measured" — and it walks straight past the selected-only
            # guard because the topk LIST has twenty entries. Dropping the
            # sentinels leaves the entries the provider actually scored; if
            # only the selected token survives, the step is selected-only and
            # the existing degeneracy machinery reports everything absent.
            raw = [(t, lp) for t, lp in raw if lp > -9000.0]
            # Ensure selected token is present
            if not any(t == selected_str for t, _ in raw):
                raw.insert(0, (selected_str, selected_lp))
            raw = raw[:effective_k]

            max_lp = max(lp for _, lp in raw)
            exp_sum = sum(math.exp(lp - max_lp) for _, lp in raw)

            topk: list[TopKEntry] = []
            selected_id = 0
            for rank, (tok_str, lp) in enumerate(raw):
                prob = math.exp(lp - max_lp) / exp_sum
                is_selected = tok_str == selected_str
                if is_selected:
                    selected_id = rank
                topk.append(TopKEntry(
                    token_id=rank,  # synthetic — API doesn't expose integer IDs
                    token_str=tok_str,
                    logit=lp,
                    logprob=lp,
                    prob=prob,
                ))

            steps.append(StepRecord(
                step=step_idx,
                selected_token_id=selected_id,
                selected_token_str=selected_str,
                topk=topk,
            ))

        generated_ids = [s.selected_token_id for s in steps]

        return GenerationResult(
            input_ids=input_ids,
            generated_ids=generated_ids,
            steps=steps,
            model_name=self.name,
            top_k=effective_k,
            seed=seed,
            stop_reason=self._stop_reason(
                response.choices[0], response, empty=not steps
            ),
        )

    def _generate_no_logprobs(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        seed: int,
        messages: "list[dict] | None" = None,
    ) -> GenerationResult:
        """Fallback for models that disallow logprob requests (GPT-5, o-series, etc.).

        Reasoning models need a large token budget because reasoning tokens consume
        completion quota before the visible output. We scale to 20× max_new_tokens
        (capped at 2000) so the model can finish its chain-of-thought.
        Temperature and seed are omitted — not supported by reasoning models.
        Produces degenerate StepRecords with prob=1.0 per visible token.
        """
        if messages is None:
            prompt_text = self._last_prompt_text or self.detokenize(input_ids)
            messages = [{"role": "user", "content": prompt_text}]
        # Reasoning models spend hidden tokens on chain-of-thought before producing
        # any visible output. max_completion_tokens covers both, so we need a large
        # minimum (2000) even when max_new_tokens=1 — otherwise GPT-5 / o-series
        # exhausts its quota on thinking and returns empty visible content.
        budget = max(min(max_new_tokens * 20, 16000), 2000)

        # Try max_completion_tokens first (required by reasoning models); fall back to max_tokens
        try:
            response = self._client.chat.completions.create(
                model=self.name,
                messages=messages,
                max_completion_tokens=budget,
            )
        except Exception:
            response = self._client.chat.completions.create(
                model=self.name,
                messages=messages,
                max_tokens=budget,
            )

        choice = response.choices[0]
        content = choice.message.content or ""
        token_strs = content.split()  # word-level approximation
        steps: list[StepRecord] = []
        for step_idx, tok_str in enumerate(token_strs):
            steps.append(StepRecord(
                step=step_idx,
                selected_token_id=0,
                selected_token_str=tok_str,
                topk=[TopKEntry(token_id=0, token_str=tok_str, logit=0.0, logprob=0.0, prob=1.0)],
            ))

        return GenerationResult(
            input_ids=input_ids,
            generated_ids=[0] * len(steps),
            steps=steps,
            model_name=self.name,
            top_k=1,  # signals degenerate to compute_instrument_summary
            seed=seed,
            stop_reason=self._stop_reason(choice, response, empty=not steps),
        )

    def _stop_reason(self, choice, response, *, empty: bool) -> "str | None":
        """Why this generation ended, in terms the record can state.

        The empty case is the one this exists for. `message.content` was read
        as `content or ""`, so a response that carried no visible text became
        zero steps with nothing recorded about why — and two gpt-5 profiles in
        the published corpus have `output_side.steps = []` with no reason
        anywhere in the artifact. Three different facts produce that shape:

          refusal          the model declined; the API says so in
                           `message.refusal`, a field we never read.
          content_filter   the provider blocked it; `finish_reason` says so.
          length           the completion budget ran out. On a reasoning model
                           this is the trap the budget below was sized to
                           avoid and does not always avoid: reasoning tokens
                           are billed against `max_completion_tokens`, so a
                           model can spend the entire allowance thinking and
                           return `finish_reason="length"` with empty content.
                           The reasoning-token count is reported in
                           `usage.completion_tokens_details`, which makes this
                           case distinguishable from a truncated answer rather
                           than a guess — so we record it.

        Best-effort throughout: a provider that omits any of these fields
        yields None for that part, and None means "not reported", never
        "nothing happened".
        """
        finish = getattr(choice, "finish_reason", None)
        refusal = getattr(getattr(choice, "message", None), "refusal", None)
        if refusal:
            return f"refusal: {refusal}"

        reasoning_tokens = None
        try:
            details = response.usage.completion_tokens_details
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
        except Exception:  # noqa: BLE001 — usage accounting is optional
            reasoning_tokens = None

        if empty and finish == "length" and reasoning_tokens:
            logger.warning(
                "%s returned no visible content: the completion budget was "
                "exhausted by %d reasoning tokens. Raise max_new_tokens (the "
                "budget is 20x it, min 2000) or lower the model's reasoning "
                "effort.",
                self.name, reasoning_tokens,
            )
            return (
                f"length: completion budget exhausted by {reasoning_tokens} "
                "reasoning tokens before any visible output"
            )
        if empty:
            logger.warning(
                "%s returned no visible content (finish_reason=%s).",
                self.name, finish or "not reported",
            )
        return finish
