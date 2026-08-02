"""LLM-backed paraphrase generator, via any OpenAI-compatible chat endpoint.

Replaces the WordNet / rule-based generators with an instruction-tuned model
that produces genuinely meaning-preserving rewrites. Three variant types are
supported:

  synonym   — substitute 1–2 content words with genuine synonyms
  tone      — rewrite in a different register (formal / casual / direct / hedged)
  reorder   — restructure sentence order without changing meaning

Deliberately generic — one client (the `openai` Python SDK pointed at a
configurable `base_url`), not a per-provider adapter, so any OpenAI-compatible
endpoint works without code changes: local Ollama (has its own OpenAI-compat
endpoint at /v1), OpenAI directly, or a gateway like OpenRouter that exposes
non-OpenAI models (e.g. Claude) through the same schema.

Cost ownership (see hif/perturbation/__init__.py's module docstring for the
full rationale):
  - Local Ollama (this module's default) — self-hosted, no metered API cost.
  - A paid hosted endpoint (e.g. an OpenRouter-routed Claude Haiku) works by
    pointing base_url/api_key/model at it. get_generator() requires that to
    be passed explicitly and never defaults to it, so no run silently spends
    money.

Each type issues one batched request and parses a numbered list. Falls back
to the original prompt on parse failure rather than silently returning
garbage.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from hif.perturbation.base import PerturbationGenerator, PerturbationResult

logger = logging.getLogger(__name__)

VariantType = Literal["synonym", "tone", "reorder"]

# Default: local Ollama via its OpenAI-compatible endpoint. Ollama ignores
# the API key but the openai SDK requires a non-empty string.
_OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434/v1"
_OLLAMA_API_KEY_DEFAULT = "ollama"
_OLLAMA_MODEL_DEFAULT = "gemma3:4b-it-qat"

# Hosted alternative: any OpenAI-compatible gateway works via base_url —
# e.g. OpenRouter (https://openrouter.ai/api/v1) exposes Claude models through
# the same schema, so this module needs no Anthropic-specific adapter. Verify
# the model slug against the gateway's current catalog; nothing here pins one.

_SYSTEM_PROMPTS: dict[VariantType, str] = {
    "synonym": (
        "You are a paraphrase generator. Rewrite the given sentence by substituting "
        "1–2 content words (nouns, verbs, or adjectives) with genuine synonyms that "
        "preserve the exact meaning. Do not change word order or sentence structure. "
        "Produce natural, idiomatic English. "
        "Return ONLY a numbered list with one rewrite per line — no explanations, no extra text."
    ),
    "tone": (
        "You are a tone-shift paraphrase generator. Rewrite the given sentence in a "
        "different register while preserving its exact meaning. Rotate through these "
        "registers in order across the numbered variants: formal, casual, direct, hedged. "
        "Produce natural, idiomatic English. "
        "Return ONLY a numbered list with one rewrite per line — no explanations, no extra text."
    ),
    "reorder": (
        "You are a paraphrase generator. Rewrite the given sentence by restructuring "
        "its grammatical word order — changing clause order, active/passive voice, or "
        "fronting different constituents — while preserving its exact meaning. "
        "Each variant must be structurally distinct from the others. "
        "Produce natural, idiomatic English. "
        "Return ONLY a numbered list with one rewrite per line — no explanations, no extra text."
    ),
}


def _build_user_prompt(prompt: str, n: int) -> str:
    return (
        f'Rewrite this sentence {n} times, one per numbered line:\n'
        f'"{prompt}"\n\n'
        + "\n".join(f"{i + 1}." for i in range(n))
    )


def _parse_numbered_list(text: str, n: int, fallback: str) -> list[str]:
    """Extract up to n lines from a numbered-list response.

    Accepts lines like: '1. text', '1) text', or just 'text' as fallback.
    Returns exactly n items, padding with `fallback` if the model under-produces.
    """
    # Strip markdown code fences if the model wrapped the output
    text = re.sub(r"```[^\n]*\n?", "", text).strip()

    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1.", "1)", "- " etc.
        cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^[-*]\s*", "", cleaned).strip()
        # Strip surrounding quotes the model sometimes adds
        cleaned = cleaned.strip('"').strip("'").strip()
        if cleaned:
            lines.append(cleaned)

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for l in lines:
        if l.lower() not in seen:
            seen.add(l.lower())
            unique.append(l)

    # Pad if short; truncate if long
    while len(unique) < n:
        unique.append(fallback)
    return unique[:n]


def _openai_compatible_chat(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.8,
    timeout: float = 60.0,
    seed: int | None = None,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint and return the
    assistant response text. Works against Ollama's OpenAI-compat endpoint,
    OpenAI itself, or any gateway (e.g. OpenRouter) exposing the same schema
    — the caller decides which provider by choice of base_url/api_key/model.

    Passing `seed` makes sampling reproducible for a given (model, messages,
    temperature) where the provider honors it — without it, most providers
    draw from their own unseeded RNG and repeated calls with "the same seed"
    at the caller level are not actually deterministic. Not all
    OpenAI-compatible providers honor `seed` (support varies); this is a
    best-effort request, not a guarantee.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=512,
        seed=seed,
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""


class LLMParaphraseGenerator(PerturbationGenerator):
    """Meaning-preserving paraphrase generator backed by any OpenAI-compatible
    chat endpoint (local Ollama by default — see module docstring)."""

    def __init__(
        self,
        variant_type: VariantType = "synonym",
        model: str = _OLLAMA_MODEL_DEFAULT,
        base_url: str = _OLLAMA_BASE_URL_DEFAULT,
        api_key: str = _OLLAMA_API_KEY_DEFAULT,
        temperature: float = 0.8,
    ) -> None:
        self.variant_type = variant_type
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.variant_type

    def generate(self, prompt: str, n_variants: int = 2, seed: int = 42) -> PerturbationResult:
        system = _SYSTEM_PROMPTS[self.variant_type]
        user = _build_user_prompt(prompt, n_variants)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            raw = _openai_compatible_chat(
                messages, self.model, self.base_url, self.api_key, self.temperature, seed=seed,
            )
            variants = _parse_numbered_list(raw, n_variants, prompt)
        except Exception as exc:
            logger.warning(
                "LLMParaphraseGenerator (%s) failed for prompt %r: %s — using original.",
                self.variant_type,
                prompt[:60],
                exc,
            )
            variants = [prompt] * n_variants

        return PerturbationResult(
            original=prompt,
            variants=variants,
            generator=self.variant_type,
        )
