"""Paraphrase generator backed by a locally cached instruction-tuned model.

Reads weights from the HuggingFace cache. No server, no API key, no network
once the weights are present — the same acquisition story as the `hf` backend,
applied to the input side.

WHY THIS EXISTS

The rule-based generators do not produce meaning-preserving paraphrases, which
is what `perturbation_jsd_bits` and the two input-entropy rows are defined
against (README § "Four of those numbers are comparisons"). Two failures, both
reproducible on the published stimulus set:

* `synonym` pools lemmas from EVERY WordNet synset of a word, with no sense
  disambiguation, then filters out rare and multi-word lemmas. Those filters
  delete the correct sense and keep the wrong one, because precise terms are
  rarer and more often compound. For "headache" the surviving pool is exactly
  `concern, worry, vexation` — every one of them from `concern.n.04`,
  "something or someone that causes anxiety" — because `head_ache` has an
  underscore and `cephalalgia` is rare. There is no correct substitution
  available. For "week" the only survivor is `workweek`, which is a different
  number of days.
* `tone` returns the prompt unchanged for 50% of the built-in stimulus set at
  the default budget, and `reorder` duplicates 34% of its own output. Nothing
  filtered either, so those variants entered the aggregate as JSD = 0 exactly.

`LLMParaphraseGenerator` already solves the first problem, but only through an
OpenAI-compatible endpoint — in practice a running Ollama daemon. This module
removes that requirement.

WHAT IT DOES NOT FIX

A local model produces natural English and respects word sense; it does not
guarantee meaning preservation. In testing, `google/gemma-3-4b-it` rewrote
"headaches" as "migraines" (a narrowing) and dropped "every morning" to
"regularly" in two of three variants. That is a large improvement on a pool
with no correct option in it, and it is not a solution. The construct remains
the caller's to check.

DETERMINISM

Decoding is greedy, so a given (model, prompt, n) is reproducible on the same
weights and dtype. `seed` is accepted for interface compatibility and does not
change the output. The model that wrote the variants is part of the
instrument, not a detail: `model_id` is exposed so the run can record it.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from hif.perturbation.base import PerturbationGenerator, PerturbationResult
# One source for the instruction text — this module changes where inference
# happens, not what is asked for.
from hif.perturbation.llm import (
    _SYSTEM_PROMPTS,
    _build_user_prompt,
    parse_variants,
    VariantType,
)

logger = logging.getLogger(__name__)

# Chosen by testing every instruction-tuned model in the local cache against
# the same prompt: SmolLM2-135M-Instruct echoed the input unchanged,
# gemma-3-1b-it produced telegraphic fragments that dropped the subject, and
# Qwen3-1.7B spent its budget on <think> tokens. This is also the family
# llm.py already defaults to on Ollama (gemma3:4b-it-qat), so the two paths
# agree on the paraphraser rather than quietly differing.
DEFAULT_LOCAL_MODEL = "google/gemma-3-4b-it"

# Loading a 4B checkpoint per variant type per prompt would dominate the run.
# Keyed by (model_id, dtype, device) so a caller switching any of them gets the
# right one rather than a silently mismatched cache hit.
_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}


def _pick_device() -> str:
    """Accelerator if there is one. On CPU a 4B paraphraser is not viable.

    Measured on this machine: three 2-variant batches for a single prompt did
    not finish inside ten minutes on CPU. The same work on the Metal backend is
    the difference between the generator being usable per-prompt and not being
    usable at all, so device selection is not a tuning knob here — it decides
    whether the default model can be the default.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_dtype(device: str) -> str:
    """bfloat16 on any accelerator, float32 on CPU.

    float16 is NOT a safe fallback here even where it is faster to reach for.
    Gemma is trained in bfloat16 and overflows in float16: on Metal the same
    prompt that yields clean paraphrases in bfloat16 (3.6s) returns an empty
    string in float16 (47.5s), because generation runs to its token budget
    producing nothing decodable. An empty completion parses to zero variants,
    which this module correctly reports as absent — so the failure would have
    looked like "the paraphraser had nothing to say" rather than a dtype bug.
    """
    if device in ("cuda", "mps"):
        return "bfloat16"
    return "float32"


def _resolve_model_class(config: Any) -> Any:
    """Pick the concrete class named by the checkpoint's own config.

    `AutoModelForCausalLM` is wrong for multimodal checkpoints: gemma-3-4b-it
    declares `Gemma3ForConditionalGeneration`. Reading `architectures` off the
    config handles causal-LM and multimodal checkpoints on one path.
    """
    import importlib

    import transformers

    arch = (getattr(config, "architectures", None) or [None])[0]
    if not arch:
        return transformers.AutoModelForCausalLM
    cls = getattr(transformers, arch, None)
    if cls is not None:
        return cls
    # Named but not exported at top level — reach into its own module.
    model_type = getattr(config, "model_type", None)
    if model_type:
        mod = importlib.import_module(f"transformers.models.{model_type}")
        cls = getattr(mod, arch, None)
        if cls is not None:
            return cls
    raise ImportError(
        f"checkpoint declares architecture {arch!r}, which could not be imported "
        f"from transformers (model_type={model_type!r})"
    )


def _load(model_id: str, dtype_name: str | None, device: str | None) -> tuple[Any, Any]:
    device = device or _pick_device()
    dtype_name = dtype_name or _default_dtype(device)
    key = (model_id, dtype_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    from transformers import AutoConfig, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    config = AutoConfig.from_pretrained(model_id)
    cls = _resolve_model_class(config)

    logger.info("loading local paraphraser %s (%s, %s)", model_id, dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = cls.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


class LocalParaphraseGenerator(PerturbationGenerator):
    """Meaning-preserving paraphrases from a locally cached instruct model."""

    def __init__(
        self,
        variant_type: VariantType = "synonym",
        model: str = DEFAULT_LOCAL_MODEL,
        dtype: Literal["bfloat16", "float16", "float32"] | None = None,
        device: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.variant_type = variant_type
        self.name = variant_type
        self.model_id = model
        self.dtype = dtype
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _chat(self, prompt: str, n: int) -> str:
        import torch

        tokenizer, model = _load(self.model_id, self.dtype, self.device)
        # The instruction goes in the user turn rather than a system turn:
        # gemma-3 declares no system role, and a template that silently drops
        # it would leave the model with no instruction at all.
        content = _SYSTEM_PROMPTS[self.variant_type] + "\n\n" + _build_user_prompt(prompt, n)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        budget = self.max_new_tokens or (64 + 48 * n)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,  # greedy — reproducible on the same weights
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(self, prompt: str, n_variants: int = 5, seed: int = 42) -> PerturbationResult:
        try:
            raw = self._chat(prompt, n_variants)
        except Exception as exc:  # noqa: BLE001 — surface the cause, produce nothing
            logger.warning(
                "local paraphraser %s failed on %r: %s",
                self.model_id, self.variant_type, exc, exc_info=True,
            )
            return PerturbationResult(original=prompt, variants=[], generator=self.name)

        variants = parse_variants(raw, prompt, n_variants)
        if len(variants) < n_variants:
            # Absent, not padded. The caller sees how many it actually got.
            logger.info(
                "local paraphraser produced %d/%d usable %s variants",
                len(variants), n_variants, self.variant_type,
            )
        return PerturbationResult(original=prompt, variants=variants, generator=self.name)
