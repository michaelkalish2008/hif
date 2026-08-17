"""Prompt perturbation generators.

Two implementations exist for synonym/tone/reorder-style perturbation:

  Rule-based (SynonymGenerator, ToneGenerator, WordOrderGenerator) — the
  DEFAULT. Deterministic, zero compute cost, no external dependency.

  LLM-backed (LLMParaphraseGenerator, via any OpenAI-compatible chat
  endpoint) — higher-quality paraphrasing, but costs real inference compute.
  OPT-IN ONLY: pass use_llm=True to get_generator() with an explicit
  base_url/api_key. The default stays rule-based so a run never bills anyone
  by surprise; pointing base_url at a local Ollama instance keeps the compute
  local and free. get_generator() never silently defaults to a paid hosted
  endpoint — the caller must supply base_url/api_key explicitly.
"""

from hif.perturbation.ambiguity import AmbiguityGenerator
from hif.perturbation.base import (
    PerturbationGenerator,
    PerturbationResult,
)
from hif.perturbation.llm import LLMParaphraseGenerator
from hif.perturbation.local_llm import LocalParaphraseGenerator
from hif.perturbation.substitution import SubstitutionGenerator
from hif.perturbation.synonym import SynonymGenerator
from hif.perturbation.tone import ToneGenerator
from hif.perturbation.word_order import WordOrderGenerator

__all__ = [
    "AmbiguityGenerator",
    "PerturbationGenerator",
    "PerturbationResult",
    "LLMParaphraseGenerator",
    "LocalParaphraseGenerator",
    "SubstitutionGenerator",
    "SynonymGenerator",
    "ToneGenerator",
    "WordOrderGenerator",
    "get_generator",
    "ImageGridMaskFamily",
    "ImageBrightnessFamily",
]

# LLM-backed variant types, keyed by the same name used for their rule-based
# counterpart below (LLMParaphraseGenerator.variant_type must be one of these).
# `substitution` is deliberately NOT here. It briefly had a drafted variant,
# and on the built-in stimulus set that variant was indistinguishable from
# `synonym`: a substitution rewrite sat exactly as far from a synonym rewrite
# (0.143 cosine) as two synonym rewrites sat from each other (0.143) — a ratio
# of 1.00, and 27% of its output was dropped as a literal duplicate. Both are
# lexical-substitution operations, so an embedder is the right instrument for
# that comparison and the verdict stands.
#
# The RULE-based generator keeps its distinct job: a fixed table of general
# words (person, thing, way, system) swapped for specific ones. That is not
# what synonym does. The drafted prompt lost the distinction the table has,
# which is an argument against the prompt, not against the family.
_LLM_TYPES = {"synonym", "tone", "reorder"}

# Rule-based generators — the default. "reorder" (the LLM-side name) maps to
# WordOrderGenerator, whose own .name is "word_order"; both keys resolve to
# the same class so callers can use either name.
_RULE_TYPES: dict[str, type[PerturbationGenerator]] = {
    "synonym": SynonymGenerator,
    "tone": ToneGenerator,
    "word_order": WordOrderGenerator,
    "reorder": WordOrderGenerator,
    "substitution": SubstitutionGenerator,
    "ambiguity": AmbiguityGenerator,
}


def get_generator(
    name: str,
    *,
    use_llm: bool = False,
    use_local: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> PerturbationGenerator:
    """Resolve a perturbation generator by name.

    Defaults to the deterministic rule-based implementation (zero compute
    cost). Pass use_llm=True with an explicit base_url/api_key to get the
    LLM-backed variant instead, via any OpenAI-compatible chat endpoint (a
    local Ollama instance, or a hosted provider/gateway) — see module
    docstring for the cost-ownership rationale. use_llm=True without
    base_url/api_key is a configuration error, not a silent fallback to some
    default endpoint. `model` is optional; omit it to use the generator's own
    default (see hif/perturbation/llm.py).
    """
    if use_local:
        # Locally cached instruct model — no server, no key, no network once
        # the weights are present. Unlike use_llm this needs no endpoint, so
        # there is nothing to require of the caller and no metered cost to
        # guard against; the cost is wall-clock on this machine.
        if name not in _LLM_TYPES:
            raise ValueError(
                f"{name!r} has no paraphrase variant. Available: {sorted(_LLM_TYPES)}"
            )
        kwargs: dict = {"variant_type": name}
        if model:
            kwargs["model"] = model
        return LocalParaphraseGenerator(**kwargs)
    if use_llm:
        if name not in _LLM_TYPES:
            raise ValueError(f"{name!r} has no LLM-backed variant. Available: {sorted(_LLM_TYPES)}")
        if not base_url or not api_key:
            raise ValueError(
                "use_llm=True requires an explicit base_url and api_key (the caller's own "
                "OpenAI-compatible endpoint — e.g. a local Ollama instance — or HIF's "
                "own hosted endpoint only for an opted-in Premium request) — get_generator() "
                "will not silently default to a compute-cost-bearing endpoint."
            )
        kwargs = {"variant_type": name, "base_url": base_url, "api_key": api_key}
        if model:
            kwargs["model"] = model
        return LLMParaphraseGenerator(**kwargs)  # type: ignore[arg-type]
    if name in _RULE_TYPES:
        return _RULE_TYPES[name]()
    all_names = sorted(_LLM_TYPES | _RULE_TYPES.keys())
    raise ValueError(f"Unknown generator: {name!r}. Available: {all_names}")


