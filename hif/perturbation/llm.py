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
        # Register is how something is said, not what is claimed. Raising the
        # formality of "high blood pressure" to "hypertension" swaps a lay
        # observation for a clinical diagnosis, which is a different statement
        # in a healthcare prompt — and the kind of drift no embedding screen
        # catches, because it is one word.
        "Do NOT exchange a lay term for its technical or clinical equivalent, or the "
        "reverse: keep every domain term exactly as given. Do not add or remove a claim, "
        "and keep each sentence the same kind of act it was — a statement stays a "
        "statement, a question stays a question, an instruction stays an instruction. "
        "Produce natural, idiomatic English. "
        "Return ONLY a numbered list with one rewrite per line — no explanations, no extra text."
    ),
    "reorder": (
        "You are a paraphrase generator. Rewrite the given sentence by restructuring "
        "its grammatical word order — changing clause order, active/passive voice, or "
        "fronting different constituents — while preserving its exact meaning. "
        # Two sentences folded into one conditional loses an assertion: "My
        # doctor mentioned X. What does that mean?" states that the doctor
        # said it; "What does it mean if my doctor mentioned X?" no longer
        # does. That is a change of claim wearing a change of word order.
        "Keep the SAME NUMBER of sentences, and keep each one the same kind of act — "
        "do not merge a statement and a question into one conditional, and do not turn "
        "an assertion into a hypothetical. "
        "Each variant must be structurally distinct from the others. "
        "Produce natural, idiomatic English. "
        "Return ONLY a numbered list with one rewrite per line — no explanations, no extra text."
    ),
}


# A stimulus that does not end in terminal punctuation is an open
# continuation — the model is meant to continue it, not answer it. A drafter
# left to itself closes them: on the built-in suite, 28% of variants for the
# seven fragment prompts came back as complete sentences, several of them
# broken ("You must first accept that to understand the ocean."). A model
# responding differently to a mangled fragment is behaving correctly, so those
# variants inflate every perturbation measurement for the literary_continuation
# and poetic_metaphorical regimes.
_FRAGMENT_RULE = (
    " The text is an INCOMPLETE sentence that trails off mid-thought. Every "
    "rewrite must also be incomplete, must stop at the same point in the "
    "thought, and must NOT be completed, resolved, or given closing "
    "punctuation."
)


def _is_fragment(prompt: str) -> bool:
    return not re.search(r"[.?!]['\")\]]*\s*$", prompt.strip())


def _build_user_prompt(prompt: str, n: int) -> str:
    noun = "fragment" if _is_fragment(prompt) else "sentence"
    rule = _FRAGMENT_RULE if _is_fragment(prompt) else ""
    return (
        f'Rewrite this {noun} {n} times, one per numbered line.{rule}\n'
        f'"{prompt}"\n\n'
        + "\n".join(f"{i + 1}." for i in range(n))
    )


def parse_variants(text: str, prompt: str, n: int) -> list[str]:
    """Numbered-list lines, deduplicated, with no-ops removed.

    Returns AT MOST n, and fewer when the model under-produces. It used to pad
    the shortfall with the original prompt, which is why that mattered: a
    variant identical to the baseline contributes a divergence of exactly zero
    to every measurement computed from it, so padding manufactured agreement
    the model never showed. Against a live Ollama this was not hypothetical —
    a 2-variant request came back as two byte-identical copies of the prompt,
    indistinguishable in the record from a model that genuinely did not move.

    It is the same defect the rule-based `tone` generator had, where 50% of
    variants on the built-in stimulus set were the prompt unchanged. Returning
    fewer variants is the honest result; the aggregate is over what was
    actually produced, and a generator that produced none is dropped from the
    plan by the builder rather than entered as an empty set.
    """
    text = re.sub(r"```[^\n]*\n?", "", text)
    # Reasoning models emit a think block before the answer; drop it.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    out: list[str] = []
    seen: set[str] = set()
    norm = " ".join(prompt.split()).strip().lower().rstrip(".")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line).strip()
        line = line.strip('"').strip("'").strip()
        if not line:
            continue
        key = " ".join(line.split()).lower().rstrip(".")
        if key == norm:          # a no-op is not a perturbation
            continue
        if key in seen:          # a duplicate adds no evidence
            continue
        seen.add(key)
        out.append(line)

    return out[:n]


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
            variants = parse_variants(raw, prompt, n_variants)
        except Exception as exc:
            # Produce nothing rather than n copies of the prompt. Each copy
            # would contribute a divergence of exactly zero, so a failed
            # endpoint call would have been recorded as a model that did not
            # move — the strongest possible stability claim, published about a
            # request that never completed.
            logger.warning(
                "LLMParaphraseGenerator (%s) failed for prompt %r: %s — "
                "producing no variants; its measurements will be absent, not zero.",
                self.variant_type,
                prompt[:60],
                exc,
            )
            variants = []

        return PerturbationResult(
            original=prompt,
            variants=variants,
            generator=self.variant_type,
        )
