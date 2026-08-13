"""Google Gemini API backend for closed-model profiling.

Supports Gemini models via two auth modes (auto-detected):

1. Vertex AI (preferred — full logprob support):
   Set up Application Default Credentials via:
     gcloud auth application-default login
   Then set GOOGLE_CLOUD_PROJECT (or pass gcp_project in ModelConfig).
   Logprobs work on gemini-2.0-flash-001, gemini-1.5-flash-001, etc.

2. Developer API (fallback — logprobs NOT supported by current models):
   Set GEMINI_API_KEY. The adapter will try logprobs and silently
   fall back to generation-only (degenerate distribution) if refused.

Constraints:
- supports_teacher_forcing = False  (API only returns generated tokens)
- max_top_k = 20                    (Vertex AI supports up to 20 logprob candidates)
- tokenize() via tiktoken           (approximate — Gemini uses SentencePiece internally)
- generate() via google-genai SDK
"""

from __future__ import annotations

import math
import os
from typing import Optional

from hif.config import ModelConfig
from hif.models.base import GenerationResult, Logits, Model, StepRecord, TopKEntry
from hif.utils.logging import get_logger

logger = get_logger(__name__)


def _enum_name(value) -> "str | None":
    """A google-genai enum as a plain string, or None.

    These fields come back as enum members (`FinishReason.MAX_TOKENS`) on some
    client versions and bare strings on others. The record wants one shape.
    """
    if value is None:
        return None
    return str(getattr(value, "name", value))


def _finish_reason(candidate) -> "str | None":
    """Why this candidate stopped, as the API reports it.

    None means the field was absent, never "it stopped cleanly" — the same
    None-is-not-asked rule the rest of the provenance block follows.
    """
    return _enum_name(getattr(candidate, "finish_reason", None))


def _block_reason(response) -> "str | None":
    """Why the request produced no candidate at all.

    The two early returns below build a GenerationResult with zero steps, and
    zero steps with no reason is exactly the absence this pass exists to
    remove: the reader cannot tell a safety block from an empty body.
    """
    feedback = getattr(response, "prompt_feedback", None)
    reason = _enum_name(getattr(feedback, "block_reason", None))
    return f"blocked: {reason}" if reason else "no candidate returned"


_CONTEXT_MAP: dict[str, int] = {
    "gemini-3.5-flash": 1_048_576,
    "gemini-3.1-flash": 1_048_576,
    "gemini-3.1-pro": 1_048_576,
    "gemini-3-flash": 1_048_576,
    "gemini-3-pro": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.0-pro": 32_768,
    "gemini-pro": 32_768,
}

# Gemini uses SentencePiece ~32k vocab; we approximate with tiktoken
_APPROX_VOCAB_SIZE = 32_000
# Vertex AI supports up to 20; Developer API cap is 5 but logprobs are broken there.
_GEMINI_MAX_TOP_K = 20

# Default Vertex AI location
_DEFAULT_LOCATION = "us-central1"


class GeminiModel(Model):
    """Google Gemini generative API backend.

    Auth priority:
    1. Vertex AI — if GOOGLE_CLOUD_PROJECT is set (or config.gcp_project),
       uses Application Default Credentials. Supports real logprobs.
    2. Developer API — if GEMINI_API_KEY is set. Logprobs currently
       unavailable; adapter falls back to generation-only mode.
    """

    def __init__(self, config: ModelConfig) -> None:
        try:
            import tiktoken
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ImportError(
                "Gemini backend requires 'google-genai' and 'tiktoken'. "
                "Run: pip install google-genai tiktoken"
            ) from exc

        self._config = config
        self._genai = genai
        self._genai_types = genai_types

        # --- Auth: Vertex AI preferred; fallback to Developer API key ---
        gcp_project = (
            getattr(config, "gcp_project", None)
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )
        api_key = config.api_key or os.environ.get("GEMINI_API_KEY")

        if gcp_project:
            location = (
                getattr(config, "gcp_location", None)
                or os.environ.get("GOOGLE_CLOUD_LOCATION")
                or _DEFAULT_LOCATION
            )
            logger.info(
                "GeminiModel: using Vertex AI (project=%s, location=%s)", gcp_project, location
            )
            self._client = genai.Client(
                vertexai=True,
                project=gcp_project,
                location=location,
            )
            self._using_vertex = True
            # Keep a fallback API-key client in case Vertex AI credentials expire at runtime
            self._fallback_client = genai.Client(api_key=api_key) if api_key else None
        elif api_key:
            logger.info("GeminiModel: using Developer API (logprobs may be unavailable)")
            self._client = genai.Client(api_key=api_key)
            self._using_vertex = False
            self._fallback_client = None
        else:
            raise ValueError(
                "No credentials found. Either:\n"
                "  • Set GOOGLE_CLOUD_PROJECT and run: gcloud auth application-default login\n"
                "  • Set GEMINI_API_KEY in your environment or .env file."
            )

        self._enc = tiktoken.get_encoding("cl100k_base")
        self._context_length_val = next(
            (v for k, v in _CONTEXT_MAP.items() if config.name.startswith(k)), 32_768
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
        return _APPROX_VOCAB_SIZE

    @property
    def context_length(self) -> int:
        return self._context_length_val

    @property
    def max_top_k(self) -> Optional[int]:
        return _GEMINI_MAX_TOP_K

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
            "GeminiModel does not support teacher-forced input analysis."
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
        effective_k = min(top_k, _GEMINI_MAX_TOP_K)
        prompt_text = self._last_prompt_text or self.detokenize(input_ids)

        logger.info(
            "Gemini generate: model=%s max_tokens=%d logprobs=%d via=%s",
            self.name, max_new_tokens, effective_k,
            "vertex" if self._using_vertex else "developer-api",
        )

        # Determine whether this is a thinking model that can have thinking disabled.
        # gemini-2.5-flash and similar: thinking is optional → set thinking_budget=0.
        # gemini-2.5-pro: thinking is mandatory, cannot be set to 0 → omit.
        is_thinking_model = "2.5" in self.name or self.name.startswith("gemini-3")
        # "pro" models require thinking; "flash" models allow it to be disabled.
        can_disable_thinking = is_thinking_model and "flash" in self.name.lower()

        # --- attempt 1: generation with logprobs ---
        # For flash thinking models, set thinking_budget=0 so logprob_result
        # covers the actual output tokens (not thinking tokens).
        # For pro thinking models, leave thinking enabled — logprobs should still
        # cover the text output portion.
        temperature = self._config.temperature if self._config.temperature is not None else 0.0
        cfg_kwargs: dict = dict(
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            response_logprobs=True,
            logprobs=effective_k,
        )
        if can_disable_thinking:
            cfg_kwargs["thinking_config"] = self._genai_types.ThinkingConfig(
                thinking_budget=0
            )
        elif is_thinking_model:
            # Pro: mandatory thinking consumes max_output_tokens before visible output.
            # Give enough budget for thinking (capped at 512) + visible tokens.
            cfg_kwargs["max_output_tokens"] = max(max_new_tokens * 20, 2000)
            cfg_kwargs["thinking_config"] = self._genai_types.ThinkingConfig(
                thinking_budget=512
            )

        config = self._genai_types.GenerateContentConfig(**cfg_kwargs)
        try:
            response = self._client.models.generate_content(
                model=self.name,
                contents=prompt_text,
                config=config,
            )
            return self._parse_logprob_response(response, input_ids, effective_k, seed)

        except Exception as exc:
            err_str = str(exc)
            is_logprob_err = "INVALID_ARGUMENT" in err_str and (
                "ogprob" in err_str or "logprob" in err_str.lower()
            )
            is_auth_err = "reauthentication" in err_str.lower() or "refresh" in err_str.lower()

            if is_auth_err and self._fallback_client is not None:
                # Vertex AI credentials have expired — switch to Developer API key permanently
                logger.warning(
                    "Vertex AI credentials expired; switching to Developer API (GEMINI_API_KEY). "
                    "Run `gcloud auth application-default login` to restore Vertex AI access."
                )
                self._client = self._fallback_client
                self._fallback_client = None
                self._using_vertex = False
                # Retry with logprobs; if Developer API doesn't support them, fall through to attempt 2
                try:
                    response = self._client.models.generate_content(
                        model=self.name,
                        contents=prompt_text,
                        config=config,
                    )
                    return self._parse_logprob_response(response, input_ids, effective_k, seed)
                except Exception:
                    pass  # Developer API doesn't support logprobs — fall through to text-only
            elif is_logprob_err:
                logger.warning(
                    "Model %s does not support logprobs (%s). "
                    "Falling back to generation-only mode — distribution metrics will be degenerate.",
                    self.name, err_str[:120],
                )
            else:
                raise

        # --- attempt 2: generation without logprobs (degenerate fallback) ---
        # For thinking models (Pro) without thinking disabled, max_output_tokens
        # covers both thinking and visible tokens — use a large minimum budget so
        # the model can finish its chain-of-thought and produce visible output.
        if can_disable_thinking:
            output_budget = max_new_tokens
            thinking_cfg = self._genai_types.ThinkingConfig(thinking_budget=0)
        else:
            # Pro-tier thinking models cannot disable thinking. Cap the thinking
            # budget at 512 tokens so the model reasons briefly before outputting.
            # max_output_tokens covers both thinking + visible output, so 2000
            # gives 512 thinking + ~1488 visible — enough for a single letter answer.
            output_budget = max(max_new_tokens * 20, 2000)
            thinking_cfg = self._genai_types.ThinkingConfig(thinking_budget=512)
        cfg_no_lp_kwargs: dict = dict(max_output_tokens=output_budget, temperature=temperature)
        if True:  # always set thinking config — Flash disables, Pro caps
            cfg_no_lp_kwargs["thinking_config"] = thinking_cfg
        config_no_lp = self._genai_types.GenerateContentConfig(**cfg_no_lp_kwargs)
        response = self._client.models.generate_content(
            model=self.name,
            contents=prompt_text,
            config=config_no_lp,
        )
        return self._parse_text_only_response(response, input_ids, seed)

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    def _parse_logprob_response(
        self,
        response,
        input_ids: list[int],
        effective_k: int,
        seed: int,
    ) -> GenerationResult:
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None:
            # No candidate at all: blocked before generation, or an empty
            # body. `prompt_feedback.block_reason` is where the API says which.
            return GenerationResult(
                input_ids=input_ids, generated_ids=[], steps=[],
                model_name=self.name, top_k=effective_k, seed=seed,
                stop_reason=_block_reason(response),
            )

        logprobs_result = getattr(candidate, "logprobs_result", None)
        chosen = getattr(logprobs_result, "chosen_candidates", []) or []
        top_candidates = getattr(logprobs_result, "top_candidates", []) or []

        steps: list[StepRecord] = []
        for step_idx, chosen_entry in enumerate(chosen):
            selected_str = chosen_entry.token
            selected_lp = chosen_entry.log_probability

            step_top = top_candidates[step_idx].candidates if step_idx < len(top_candidates) else []
            raw: list[tuple[str, float]] = [(c.token, c.log_probability) for c in step_top]

            if not any(t == selected_str for t, _ in raw):
                raw.insert(0, (selected_str, selected_lp))
            raw = raw[:effective_k]

            max_lp = max(lp for _, lp in raw)
            exp_sum = sum(math.exp(lp - max_lp) for _, lp in raw)

            topk: list[TopKEntry] = []
            selected_id = 0
            for rank, (tok_str, lp) in enumerate(raw):
                prob = math.exp(lp - max_lp) / exp_sum
                if tok_str == selected_str:
                    selected_id = rank
                topk.append(TopKEntry(
                    token_id=rank,
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
            stop_reason=_finish_reason(candidate),
        )

    def _parse_text_only_response(
        self,
        response,
        input_ids: list[int],
        seed: int,
    ) -> GenerationResult:
        """Build degenerate StepRecords when logprobs are unavailable."""
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None:
            return GenerationResult(
                input_ids=input_ids, generated_ids=[], steps=[],
                model_name=self.name, top_k=1, seed=seed,
                stop_reason=_block_reason(response),
            )

        # Extract text — parts list or .text convenience attribute
        text = ""
        if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
            text = "".join(p.text for p in candidate.content.parts if hasattr(p, "text"))
        elif hasattr(response, "text"):
            text = response.text or ""

        # Approximate tokenise the output text
        token_ids = list(self._enc.encode(text))
        token_strs = [self._enc.decode([t]) for t in token_ids]

        steps: list[StepRecord] = []
        for step_idx, (tok_id, tok_str) in enumerate(zip(token_ids, token_strs)):
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

        return GenerationResult(
            input_ids=input_ids,
            generated_ids=token_ids,
            steps=steps,
            model_name=self.name,
            top_k=1,
            seed=seed,
            stop_reason=_finish_reason(candidate),
        )
