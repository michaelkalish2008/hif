"""Ollama HTTP backend for local model inference via the Ollama REST API."""

import time
from typing import Optional

import httpx

from hif.config import ModelConfig
from hif.models.base import GenerationResult, Logits, Model, StepRecord, TopKEntry
from hif.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds

# Model-family → HuggingFace tokenizer mapping used when the Ollama server does
# not serve /api/tokenize (removed/never shipped in current Ollama releases).
# Keys are matched as prefixes of the Ollama model name (before any ":tag").
_HF_TOKENIZER_MAP: dict[str, str] = {
    "llama3.2": "meta-llama/Llama-3.2-1B",
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "gemma3": "google/gemma-3-1b-it",
    "gemma2": "google/gemma-2-2b-it",
    "qwen3": "Qwen/Qwen3-1.7B",
    "qwen2.5": "Qwen/Qwen2.5-1.5B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

# Generic last-resort tokenizer (ungated, small). Token boundaries are only
# approximate for models outside its family.
_DEFAULT_HF_TOKENIZER = "gpt2"


class _OllamaHTTPError(RuntimeError):
    """HTTP error from the Ollama API, carrying the status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class OllamaModel(Model):
    """Ollama backend using httpx to call a local Ollama server.

    Constraints:
    - max_top_k = 20 (hard Ollama API cap)
    - supports_teacher_forcing = False — forward() raises NotImplementedError
    - tokenize/detokenize via /api/tokenize (may raise NotImplementedError if unavailable)
    - Model metadata from /api/show; falls back to config values
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._host = config.ollama_host.rstrip("/")
        self._timeout = config.ollama_timeout
        self._client = httpx.Client(timeout=self._timeout)

        # Attempt to load model metadata from /api/show
        self._vocab_size_val: Optional[int] = None
        self._context_length_val: Optional[int] = None
        self._load_metadata()

        # Tokenization fallback state.  None = /api/tokenize not yet probed;
        # True/False = cached result of the first probe (a 404 is cached so we
        # never re-hit the missing endpoint per call).
        self._api_tokenize_available: Optional[bool] = None
        self._fallback_tokenizer = None  # lazily loaded HF tokenizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_with_retry(self, url: str, payload: dict) -> dict:
        """POST JSON to url with up to _MAX_RETRIES retries on connection errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError as exc:
                last_exc = exc
                logger.warning(
                    "Connection error on attempt %d/%d to %s: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    url,
                    exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
            except httpx.HTTPStatusError as exc:
                raise _OllamaHTTPError(
                    exc.response.status_code,
                    f"Ollama API returned {exc.response.status_code}: {exc.response.text}",
                ) from exc
        raise RuntimeError(
            f"Failed to connect to Ollama after {_MAX_RETRIES} attempts: {last_exc}"
        )

    def _load_metadata(self) -> None:
        """Attempt to fetch vocab_size and context_length from /api/show."""
        try:
            url = f"{self._host}/api/show"
            data = self._post_with_retry(url, {"name": self._config.name})
            # model_info may contain architecture details
            model_info = data.get("model_info", {})
            # Try common keys for context length
            for key in ("llama.context_length", "context_length"):
                if key in model_info:
                    self._context_length_val = int(model_info[key])
                    break
            # Try common keys for vocab size
            for key in ("llama.vocab_size", "tokenizer.ggml.tokens", "vocab_size"):
                if key in model_info:
                    v = model_info[key]
                    if isinstance(v, int):
                        self._vocab_size_val = v
                    elif isinstance(v, list):
                        self._vocab_size_val = len(v)
                    break
            logger.info(
                "Loaded Ollama model metadata: context_length=%s vocab_size=%s",
                self._context_length_val,
                self._vocab_size_val,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load Ollama model metadata from /api/show: %s. "
                "Using config defaults.",
                exc,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_val if self._vocab_size_val is not None else 0

    @property
    def context_length(self) -> int:
        return self._context_length_val if self._context_length_val is not None else 0

    @property
    def max_top_k(self) -> Optional[int]:
        return 20  # hard Ollama API cap

    @property
    def supports_teacher_forcing(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """Tokenize via /api/tokenize, falling back to a local HF tokenizer on 404.

        Current Ollama releases do not serve /api/tokenize.  We try the endpoint
        once; a 404 is cached and all subsequent calls go straight to a local
        HuggingFace tokenizer matched to the model family.
        """
        if self._api_tokenize_available is not False:
            url = f"{self._host}/api/tokenize"
            try:
                data = self._post_with_retry(
                    url, {"model": self._config.name, "content": text}
                )
                tokens = data.get("tokens")
                if tokens is None:
                    raise NotImplementedError("OllamaModel tokenization not supported")
                self._api_tokenize_available = True
                return [int(t) for t in tokens]
            except _OllamaHTTPError as exc:
                if exc.status_code == 404:
                    # Expected on stock Ollama servers (no /api/tokenize) —
                    # internal routing detail, visible under --verbose only.
                    logger.info(
                        "Ollama server does not serve /api/tokenize (404); "
                        "falling back to a local HuggingFace tokenizer."
                    )
                    self._api_tokenize_available = False  # cache — don't re-probe
                else:
                    raise NotImplementedError(
                        f"OllamaModel tokenization not supported: {exc}"
                    ) from exc
            except RuntimeError as exc:
                raise NotImplementedError(
                    f"OllamaModel tokenization not supported: {exc}"
                ) from exc

        tokenizer = self._get_fallback_tokenizer()
        return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]

    def detokenize(self, ids: list[int]) -> str:
        """Detokenize via the local HF fallback tokenizer when the Ollama API lacks it."""
        if self._api_tokenize_available is False or self._fallback_tokenizer is not None:
            tokenizer = self._get_fallback_tokenizer()
            return tokenizer.decode(ids, skip_special_tokens=False)
        raise NotImplementedError("OllamaModel tokenization not supported")

    # ------------------------------------------------------------------
    # HF tokenizer fallback
    # ------------------------------------------------------------------

    def _get_fallback_tokenizer(self):
        """Load (once) a local HuggingFace tokenizer matched to the model family."""
        if self._fallback_tokenizer is not None:
            return self._fallback_tokenizer

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise NotImplementedError(
                "OllamaModel tokenization fallback requires the 'transformers' "
                "package: pip install transformers for Ollama tokenization fallback"
            ) from exc

        # Match the Ollama model name (before any ":tag") against known families.
        base_name = self._config.name.split(":")[0].lower()
        hf_name: Optional[str] = None
        for prefix in sorted(_HF_TOKENIZER_MAP, key=len, reverse=True):
            if base_name.startswith(prefix):
                hf_name = _HF_TOKENIZER_MAP[prefix]
                break

        candidates = [hf_name] if hf_name else []
        candidates.append(_DEFAULT_HF_TOKENIZER)

        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                if candidate == _DEFAULT_HF_TOKENIZER and hf_name != candidate:
                    logger.warning(
                        "No HF tokenizer mapping matched Ollama model '%s' (or the "
                        "mapped tokenizer failed to load); using generic '%s'. "
                        "Token boundaries are approximate.",
                        self._config.name,
                        _DEFAULT_HF_TOKENIZER,
                    )
                else:
                    logger.info(
                        "Using local HF tokenizer '%s' for Ollama model '%s'.",
                        candidate,
                        self._config.name,
                    )
                self._fallback_tokenizer = tokenizer
                return tokenizer
            except Exception as exc:  # noqa: BLE001 — gated/offline models etc.
                last_exc = exc
                logger.warning(
                    "Could not load HF tokenizer '%s': %s", candidate, exc
                )

        raise NotImplementedError(
            f"OllamaModel tokenization fallback failed: could not load any HF "
            f"tokenizer ({last_exc}). pip install transformers and ensure a "
            f"tokenizer is available locally."
        )

    # ------------------------------------------------------------------
    # Forward pass — not supported
    # ------------------------------------------------------------------

    def forward(self, input_ids: list[int]) -> Logits:
        raise NotImplementedError(
            "OllamaModel does not support teacher-forced input analysis. "
            "Run input-side analysis using HFModel or TLensModel, even if "
            "generation uses Ollama."
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
        """Generate via /api/generate (stream=false) and parse logprobs."""
        # Respect the hard Ollama cap
        effective_k = min(top_k, self.max_top_k)

        # Reconstruct prompt text from token IDs if possible, else fall back to empty
        # (caller should pass tokenized prompt; we pass raw token string back as prompt)
        # We convert ids back to a prompt string heuristically — use detokenize-by-join
        # when we have strings; in practice the caller usually passes the raw prompt text
        # via collect_output_trace which calls model.tokenize(prompt) first.
        # Since Ollama works with text, we need the prompt as a string.  We reconstruct
        # by calling /api/generate with the numeric ids as context using the raw endpoint.
        # However, the cleanest approach is to store the original prompt text.
        # The Model.generate() API only provides input_ids, not text — so we attempt
        # to re-decode using a best-effort approach.
        prompt_text = self._ids_to_text(input_ids)

        url = f"{self._host}/api/generate"
        payload = {
            "model": self._config.name,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_k": effective_k,
                "num_predict": max_new_tokens,
                "seed": seed,
            },
            # logprobs is a top-level bool in the current Ollama API (>= 0.12),
            # with top_logprobs (top-level int) controlling candidates per step.
            "logprobs": True,
            "top_logprobs": effective_k,
        }

        data = self._post_with_retry(url, payload)

        # --- Parse generated text ---
        generated_text: str = data.get("response", "")

        # --- Parse logprobs ---
        # Ollama may return prompt_eval_logprobs and eval_logprobs (generation logprobs)
        eval_logprobs = data.get("eval_logprobs") or data.get("logprobs")
        if eval_logprobs is None:
            raise RuntimeError(
                "Ollama response did not include logprobs. "
                "Ensure your Ollama version supports logprobs and the model is compatible."
            )

        # eval_logprobs is a list of per-step entries.
        # Each entry may be:
        #   - current format (Ollama >= ~0.12): a dict
        #       {token: str, logprob: float, bytes: [...],
        #        top_logprobs: [{token, logprob, bytes}, ...]}
        #   - older format: a list of {token: str, logprob: float} objects,
        #     first element = selected token
        # We flatten into per-step TopKEntry lists.
        import math

        steps: list[StepRecord] = []
        skipped_steps = 0
        for step_idx, step_entry in enumerate(eval_logprobs):
            if step_idx >= max_new_tokens:
                break

            if isinstance(step_entry, dict):
                # Current format: selected token at top level, candidates under
                # "top_logprobs". Ensure the selected token is rank 0.
                candidates = list(step_entry.get("top_logprobs") or [])
                selected = {
                    "token": step_entry.get("token", ""),
                    "logprob": step_entry.get("logprob", -float("inf")),
                }
                if not candidates or candidates[0].get("token") != selected["token"]:
                    candidates.insert(0, selected)
                step_candidates = candidates
            elif isinstance(step_entry, list):
                # Older format: list of candidate dicts, first = selected.
                step_candidates = step_entry
            else:
                skipped_steps += 1
                logger.warning(
                    "Unexpected logprobs format at step %d; skipping step.", step_idx
                )
                continue

            topk_entries: list[TopKEntry] = []
            selected_token_id: Optional[int] = None
            selected_token_str: str = ""

            for rank, entry in enumerate(step_candidates[:effective_k]):
                token_str = str(entry.get("token", ""))
                logprob = float(entry.get("logprob", -float("inf")))
                prob = math.exp(logprob) if logprob > -1e9 else 0.0
                token_id = entry.get("token_id", rank)  # fallback: use rank as synthetic id

                topk_entries.append(
                    TopKEntry(
                        token_id=int(token_id),
                        token_str=token_str,
                        logit=logprob,  # Ollama doesn't give raw logits; use logprob
                        logprob=logprob,
                        prob=prob,
                    )
                )

                # The first candidate (rank 0) is the selected token
                if rank == 0:
                    selected_token_id = int(token_id)
                    selected_token_str = token_str

            if not topk_entries:
                skipped_steps += 1
                logger.warning(
                    "No usable logprob candidates at step %d; skipping step.", step_idx
                )
                continue

            if selected_token_id is None:
                selected_token_id = topk_entries[0].token_id
                selected_token_str = topk_entries[0].token_str

            steps.append(
                StepRecord(
                    step=step_idx,
                    selected_token_id=selected_token_id,
                    selected_token_str=selected_token_str,
                    topk=topk_entries,
                )
            )

        # Guard: metrics computed from zero steps would silently produce
        # wrong values (e.g. sensitivity = 0.0000). Fail loudly instead.
        if not steps:
            raise RuntimeError(
                "Ollama returned no usable logprobs — cannot compute distribution "
                "metrics. Check that your Ollama version supports logprobs "
                "(>= 0.12) and that the model is compatible."
            )
        if skipped_steps:
            logger.warning(
                "Skipped %d of %d logprob steps due to unexpected format; "
                "metrics computed from the remaining steps.",
                skipped_steps,
                len(eval_logprobs),
            )

        # Collect generated_ids from steps
        generated_ids = [s.selected_token_id for s in steps]

        return GenerationResult(
            input_ids=input_ids,
            generated_ids=generated_ids,
            steps=steps,
            model_name=self.name,
            top_k=effective_k,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ids_to_text(self, ids: list[int]) -> str:
        """Best-effort reconstruction of prompt text from token IDs.

        OllamaModel does not natively support detokenization; this returns
        a placeholder that callers of generate() should not rely on for
        semantic accuracy — the round-trip is lossy.
        """
        # If we tokenized with the local HF fallback, decode with the same
        # tokenizer so the round-trip is consistent.
        if self._api_tokenize_available is False or self._fallback_tokenizer is not None:
            try:
                tokenizer = self._get_fallback_tokenizer()
                return tokenizer.decode(ids, skip_special_tokens=True)
            except NotImplementedError:
                pass
        # Try /api/detokenize if it exists (not standard in most Ollama versions)
        try:
            url = f"{self._host}/api/detokenize"
            data = self._post_with_retry(
                url, {"model": self._config.name, "tokens": ids}
            )
            text = data.get("content") or data.get("text")
            if text:
                return str(text)
        except Exception:  # noqa: BLE001
            pass
        # Fallback: join ids as space-separated strings (lossy but functional)
        return " ".join(str(i) for i in ids)
