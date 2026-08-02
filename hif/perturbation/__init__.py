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
    MultimodalVariant,
    PerturbationFamily,
    PerturbationGenerator,
    PerturbationResult,
    PerturbationTrace,
)
from hif.perturbation.image_grid import ImageBrightnessFamily, ImageGridMaskFamily
from hif.perturbation.llm import LLMParaphraseGenerator
from hif.perturbation.substitution import SubstitutionGenerator
from hif.perturbation.synonym import SynonymGenerator
from hif.perturbation.tone import ToneGenerator
from hif.perturbation.word_order import WordOrderGenerator

__all__ = [
    "AmbiguityGenerator",
    "PerturbationGenerator",
    "PerturbationResult",
    "LLMParaphraseGenerator",
    "SubstitutionGenerator",
    "SynonymGenerator",
    "ToneGenerator",
    "WordOrderGenerator",
    "get_generator",
    "PerturbationFamily",
    "PerturbationTrace",
    "MultimodalVariant",
    "ImageGridMaskFamily",
    "ImageBrightnessFamily",
    "get_family",
]

# LLM-backed variant types, keyed by the same name used for their rule-based
# counterpart below (LLMParaphraseGenerator.variant_type must be one of these).
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


# ---------------------------------------------------------------------------
# Media-side family registry (Design §6, docs/ARCHITECTURE.md § Multimodal
# notes) — a SEPARATE
# namespace from get_generator(); text and media names never mix.
# ---------------------------------------------------------------------------

_FAMILY_TYPES: dict[str, type[PerturbationFamily]] = {
    ImageGridMaskFamily.name: ImageGridMaskFamily,
    ImageBrightnessFamily.name: ImageBrightnessFamily,
}


def get_family(name: str, **kwargs) -> PerturbationFamily:
    """Resolve a media perturbation family by name.

    kwargs are forwarded to the family constructor (e.g. grid_rows/grid_cols
    for image_grid_mask, delta for image_brightness).
    """
    if name in _FAMILY_TYPES:
        return _FAMILY_TYPES[name](**kwargs)
    raise ValueError(
        f"Unknown perturbation family: {name!r}. Available: {sorted(_FAMILY_TYPES)}"
    )
