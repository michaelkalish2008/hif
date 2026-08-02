"""Backend capabilities & model discovery — single source of truth.

Different backends expose different amounts of the model's internals, which
directly determines which measurements can be computed:

- **Teacher forcing** (running the model forward over the prompt to get per-
  position logits) is required for the input-side measurements:
  input_entropy_shift_bits, prompt_surprisal_excess_bits, io_correlation_r.
  Only local open-weight backends (hf, tlens, hf-vlm) can do it. Hosted APIs
  and Ollama cannot.
- **Attention capture** is required for attention_entropy_input_bits and
  attention_entropy_output_bits. Only open HuggingFace models expose
  attention, and only when attention analysis is enabled.
- **Top-K logprobs** are required for the output-side measurements
  (output_entropy_bits, output_entropy_step_delta_bits, perturbation_jsd_bits,
  candidate_cluster_entropy_bits, io_cosine_similarity,
  counterfactual_exposure_fraction, semantic_centroid_veer_cosine). Most
  backends provide them; Anthropic returns only the selected token, so its
  distributions degenerate.

This module powers three things: the early metric/backend guard in `profile`,
the `doctor` preflight command, and the `models` discovery command — so a user
learns what they can run *before* a long pipeline zeroes their requested metric.

Availability is not the same question as SUBJECT
------------------------------------------------
What follows says whether a backend can produce a number. Whether that number
is *about the target model* is a separate question, answered by the `subject`
field on each `MEASUREMENT_REGISTRY` row (hif/profile/signals.py). The two come
apart in exactly one place: `--surrogate`. A surrogate makes the input-side
quantities computable on a backend that cannot teacher-force, but it computes
them by reading the PROMPT — the target contributes nothing, so on that backend
their subject is `prompt-only` and they are reported in `prompt_measurements`
rather than in `measurements`. `hif models` prints both facts per backend.

One row is prompt-only on every backend, including `[F]`:
`attention_entropy_input_bits`. Attention here is not the target's — it comes
from a bidirectional analysis encoder (hif/analysis/attention.py) reading text
as an object, and the input-side row reads the prompt. Its availability gate
below is therefore about whether the *stage runs*, not about whether the target
exposes anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Measurement groupings by what data they require.
INPUT_SIDE_METRICS = frozenset({
    "input_entropy_shift_bits", "prompt_surprisal_excess_bits", "io_correlation_r",
})
ATTENTION_METRICS = frozenset({
    "attention_entropy_input_bits", "attention_entropy_output_bits",
})
OUTPUT_SIDE_METRICS = frozenset({
    "candidate_cluster_entropy_bits", "output_entropy_bits",
    "output_entropy_step_delta_bits", "perturbation_jsd_bits",
    "output_step_jsd_bits", "output_step_topk_overlap_fraction",
    "io_cosine_similarity", "semantic_centroid_veer_cosine",
    "counterfactual_exposure_fraction",
})

TEACHER_FORCING_BACKENDS = frozenset({"hf", "tlens", "hf-vlm"})
ATTENTION_BACKENDS = frozenset({"hf", "tlens", "hf-vlm"})
# Backends whose logprobs degenerate to the selected token only.
DEGENERATE_LOGPROB_BACKENDS = frozenset({"anthropic"})


@dataclass(frozen=True)
class BackendInfo:
    name: str
    kind: str                 # "local-open" | "hosted-api" | "local-service"
    deps: str                 # pip extras needed
    setup: str                # services / credentials the user must provide
    teacher_forcing: bool
    attention: bool
    logprobs: str             # "full" | "top-k" | "selected-only"
    multimodal: bool = False
    example_models: list[str] = field(default_factory=list)
    notes: str = ""


# Ordered best-first for a user who wants full-fidelity signals.
BACKENDS: dict[str, BackendInfo] = {
    "hf": BackendInfo(
        name="hf", kind="local-open",
        deps="torch, transformers (base install)",
        setup="none (HF_TOKEN only for gated repos); weights auto-download",
        teacher_forcing=True, attention=True, logprobs="full",
        example_models=["gpt2", "distilgpt2", "gpt2-medium",
                        "EleutherAI/pythia-160m", "EleutherAI/gpt-neo-125M"],
        notes="Full fidelity — every measurement. Best for a complete profile.",
    ),
    "tlens": BackendInfo(
        name="tlens", kind="local-open",
        deps="transformer_lens  (pip install 'hif[tlens]')",
        setup="none (HF_TOKEN for gated); GPU recommended",
        teacher_forcing=True, attention=True, logprobs="full",
        example_models=["gpt2", "gpt2-medium", "EleutherAI/pythia-160m"],
        notes="Full fidelity via TransformerLens.",
    ),
    "hf-vlm": BackendInfo(
        name="hf-vlm", kind="local-open",
        deps="torch, transformers, Pillow (base install)",
        setup="none (HF_TOKEN for gated); weights auto-download",
        teacher_forcing=True, attention=True, logprobs="full", multimodal=True,
        example_models=["HuggingFaceTB/SmolVLM-256M-Instruct"],
        notes="Multimodal (image+text). Full fidelity on the text parts.",
    ),
    "ollama": BackendInfo(
        name="ollama", kind="local-service",
        deps="httpx  (pip install 'hif[ollama]')",
        setup="run `ollama serve`, then `ollama pull <model>` FIRST",
        teacher_forcing=False, attention=False, logprobs="top-k",
        example_models=["llama3.2", "llama3.1", "gemma3", "gemma2",
                        "qwen2.5", "mistral", "phi3"],
        notes="Output-side signals only (top-20). No input-side or attention "
              "signals. The model MUST be pulled locally before profiling.",
    ),
    "openai": BackendInfo(
        name="openai", kind="hosted-api",
        deps="openai, tiktoken  (pip install 'hif[openai]')",
        setup="OPENAI_API_KEY env var (billed per token)",
        teacher_forcing=False, attention=False, logprobs="top-k",
        example_models=["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
        notes="Output-side signals only (top-20 logprobs).",
    ),
    "anthropic": BackendInfo(
        name="anthropic", kind="hosted-api",
        deps="anthropic, tiktoken  (pip install 'hif[anthropic]')",
        setup="ANTHROPIC_API_KEY env var (billed per token)",
        teacher_forcing=False, attention=False, logprobs="selected-only",
        example_models=["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
        notes="No token-level logprobs — distribution signals (breadth, entropy) "
              "degenerate. Best for similarity/sensitivity/exposure only.",
    ),
    "gemini": BackendInfo(
        name="gemini", kind="hosted-api",
        deps="google-genai, tiktoken  (pip install 'hif[gemini]')",
        setup="Vertex AI: GOOGLE_CLOUD_PROJECT + `gcloud auth application-default "
              "login` (needed for logprobs) · or GEMINI_API_KEY (no logprobs)",
        teacher_forcing=False, attention=False, logprobs="top-k",
        example_models=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        notes="Top-20 logprobs on Vertex AI only; the developer API degenerates.",
    ),
    "openai-vlm": BackendInfo(
        name="openai-vlm", kind="hosted-api",
        deps="openai, tiktoken  (pip install 'hif[openai]')",
        setup="OPENAI_API_KEY env var (billed per token)",
        teacher_forcing=False, attention=False, logprobs="top-k", multimodal=True,
        example_models=["gpt-4o", "gpt-4o-mini"],
        notes="Multimodal (image+text). Output-side signals only.",
    ),
}


# Recommended --surrogate-model choices: small, ungated, teacher-forcing-capable
# HF causal LMs suitable for recovering input-side signals on backends that
# can't teacher-force themselves (see `--surrogate` in `hif profile`).
# `unsloth/Llama-3.2-1B` is the default (see _load_surrogate in cli.py).
SURROGATE_CANDIDATES: list[str] = [
    "unsloth/Llama-3.2-1B",
    "gpt2",
    "distilgpt2",
    "gpt2-medium",
    "EleutherAI/pythia-160m",
    "Qwen/Qwen2.5-0.5B",
]


def metric_support(metric: str, backend: str) -> str | None:
    """Return None if `backend` can produce `metric`, else a reason + the fix.

    This is the guard that catches e.g. `--metric input_entropy_shift_bits
    --backend ollama`: that measurement is input-side, ollama has no teacher
    forcing.
    """
    info = BACKENDS.get(backend)
    if info is None:
        return f"Unknown backend {backend!r}."

    if metric in INPUT_SIDE_METRICS and not info.teacher_forcing:
        return (
            f"'{metric}' is an input-side measurement — it requires teacher "
            f"forcing, which the '{backend}' backend cannot do (hosted APIs and "
            f"Ollama never expose per-token input logits).\n"
            f"  Fix: use an open-weight model, e.g. `--backend hf` with `gpt2`; "
            f"pass --surrogate to teacher-force a small local proxy instead; or "
            f"pick an output-side measurement."
        )
    if metric in ATTENTION_METRICS and not info.attention:
        return (
            f"'{metric}' requires attention capture, available only on open "
            f"HuggingFace models (`--backend hf`/`tlens`), not '{backend}'.\n"
            f"  Fix: use `--backend hf` with an open model such as `gpt2`."
        )
    if metric in OUTPUT_SIDE_METRICS and info.logprobs == "selected-only" and metric in (
        "candidate_cluster_entropy_bits", "output_entropy_bits",
        "output_entropy_step_delta_bits",
    ):
        return (
            f"'{metric}' needs a token distribution, but the '{backend}' backend "
            f"returns only the selected token (no logprobs), so it degenerates.\n"
            f"  Fix: use a backend with logprobs (hf, openai, ollama), pass "
            f"--surrogate, or pick a measurement that does not need one."
        )
    return None


def signals_available(backend: str) -> dict[str, bool]:
    """Map each measurement key → whether `backend` can produce it."""
    all_metrics = INPUT_SIDE_METRICS | ATTENTION_METRICS | OUTPUT_SIDE_METRICS
    return {m: metric_support(m, backend) is None for m in sorted(all_metrics)}
