"""Backend capabilities & model discovery — single source of truth.

Different backends expose different amounts of the model's internals, which
directly determines which measurements can be computed:

- **Teacher forcing** (running the model forward over the prompt to get per-
  position logits) is required for the input-side measurements:
  input_entropy_shift_bits, prompt_surprisal_excess_bits, io_correlation_r.
  Only local open-weight backends (hf, tlens, hf-vlm) can do it. Hosted APIs
  and Ollama cannot.
- **The attention-analysis stage** is required for attention_entropy_input_bits
  and attention_entropy_output_bits. It is NOT a backend capability: neither
  measurement reads the target model's attention, and no backend has ever been
  asked to expose any. `hif/analysis/attention.py` loads its own bidirectional
  encoder (`distilbert-base-uncased`, `AttentionConfig.model_name`) and reads
  *text* — the prompt for the input-side row, the target's generated
  continuation for the output-side row. Both are therefore computable on every
  backend that returns text, which is all of them; the only requirement is that
  the stage runs (`--diagnostics`, or `[attention] enabled` in a config file).
  The gate below enforces that requirement and no other.
- **Top-K logprobs** are required for the output-side measurements
  (output_entropy_bits, output_entropy_step_delta_bits, perturbation_jsd_bits,
  output_step_jsd_bits, output_step_topk_overlap_fraction,
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
exposes anything. Note that this is a statement about SUBJECT, not
availability: the value is computable everywhere and is reported everywhere the
stage runs — in `prompt_measurements`, because it is a fact about the prompt.

History: this module used to claim, two paragraphs apart, both that "only open
HuggingFace models expose attention" and that the attention is "not the
target's". The first was false — verifiable in one command, since profiling
gpt2 and gpt2-medium on the same prompt returns a bit-identical
attention_entropy_input_bits (1.6677721955190443), which is what a number that
cannot see the target looks like. A previous pass left the gate closed anyway,
reasoning that refusing was safer than over-claiming. It was not: refusing told
users their backend could not produce a measurement it produces perfectly well,
which is a false statement about their backend rather than a cautious one.
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

# Output-side measurements that a selected-only backend cannot produce AT ALL,
# because their input is a real per-step distribution and a point mass is not
# one. Split into two reasons, because the two absences are different:
#
#   _NEEDS_DISTRIBUTION   entropy-shaped quantities computed over one step's
#                         candidates. A point mass gives 0.0 by construction.
#   _NEEDS_TWO_DISTRIBUTIONS
#                         divergences between two distributions. Between two
#                         point masses the JSD is 0 when the tokens agree and
#                         exactly 1 bit when they differ — a token-agreement
#                         rate, a different quantity under the same key, so it
#                         is reported absent rather than emitted. Unlike the
#                         first group, no --surrogate rescues these: the
#                         surrogate recovery in builder.py step 6b rebuilds
#                         `semantic_steps`, which these never read.
_NEEDS_DISTRIBUTION = frozenset({
    "candidate_cluster_entropy_bits", "output_entropy_bits",
    "output_entropy_step_delta_bits",
})
_NEEDS_TWO_DISTRIBUTIONS = frozenset({
    "perturbation_jsd_bits", "output_step_jsd_bits",
    "output_step_topk_overlap_fraction",
})

TEACHER_FORCING_BACKENDS = frozenset({"hf", "tlens", "hf-vlm"})
# Backends whose logprobs degenerate to the selected token only.
DEGENERATE_LOGPROB_BACKENDS = frozenset({"anthropic"})


@dataclass(frozen=True)
class BackendInfo:
    name: str
    kind: str                 # "local-open" | "hosted-api" | "local-service"
    deps: str                 # pip extras needed
    setup: str                # services / credentials the user must provide
    teacher_forcing: bool
    logprobs: str             # "full" | "top-k" | "selected-only"
    # There is deliberately no `attention` field. It used to record whether the
    # backend "exposes attention", which nothing in the pipeline has ever asked
    # for: the attention-row measurements come from a separate analysis encoder
    # reading text (see the module docstring). A per-backend column implied a
    # backend-dependence that does not exist, so the column is gone rather than
    # set to True everywhere.
    multimodal: bool = False
    example_models: list[str] = field(default_factory=list)
    notes: str = ""


# Ordered best-first for a user who wants full-fidelity signals.
BACKENDS: dict[str, BackendInfo] = {
    "hf": BackendInfo(
        name="hf", kind="local-open",
        deps="torch, transformers (base install)",
        setup="none (HF_TOKEN only for gated repos); weights auto-download",
        teacher_forcing=True, logprobs="full",
        example_models=["gpt2", "distilgpt2", "gpt2-medium",
                        "EleutherAI/pythia-160m", "EleutherAI/gpt-neo-125M"],
        notes="Full fidelity — every measurement. Best for a complete profile.",
    ),
    "tlens": BackendInfo(
        name="tlens", kind="local-open",
        deps="transformer_lens  (pip install 'hif[tlens]')",
        setup="none (HF_TOKEN for gated); GPU recommended",
        teacher_forcing=True, logprobs="full",
        example_models=["gpt2", "gpt2-medium", "EleutherAI/pythia-160m"],
        notes="Full fidelity via TransformerLens.",
    ),
    "hf-vlm": BackendInfo(
        name="hf-vlm", kind="local-open",
        deps="torch, transformers, Pillow (base install)",
        setup="none (HF_TOKEN for gated); weights auto-download",
        teacher_forcing=True, logprobs="full", multimodal=True,
        example_models=["HuggingFaceTB/SmolVLM-256M-Instruct"],
        notes="Multimodal (image+text). Full fidelity on the text parts.",
    ),
    "ollama": BackendInfo(
        name="ollama", kind="local-service",
        deps="httpx  (pip install 'hif[ollama]')",
        setup="run `ollama serve`, then `ollama pull <model>` FIRST",
        teacher_forcing=False, logprobs="top-k",
        example_models=["llama3.2", "llama3.1", "gemma3", "gemma2",
                        "qwen2.5", "mistral", "phi3"],
        notes="Output-side signals only (top-20). No input-side or attention "
              "signals. The model MUST be pulled locally before profiling.",
    ),
    "openai": BackendInfo(
        name="openai", kind="hosted-api",
        deps="openai, tiktoken  (pip install 'hif[openai]')",
        setup="OPENAI_API_KEY env var (billed per token)",
        teacher_forcing=False, logprobs="top-k",
        example_models=["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
        notes="Output-side signals only (top-20 logprobs).",
    ),
    "anthropic": BackendInfo(
        name="anthropic", kind="hosted-api",
        deps="anthropic, tiktoken  (pip install 'hif[anthropic]')",
        setup="ANTHROPIC_API_KEY env var (billed per token)",
        teacher_forcing=False, logprobs="selected-only",
        example_models=["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
        notes="No token-level logprobs. Entropy-shaped signals degenerate, and "
              "the distribution divergences are reported absent rather than as "
              "the token-agreement rate two point masses actually produce. "
              "Best for similarity/exposure and the attention rows.",
    ),
    "gemini": BackendInfo(
        name="gemini", kind="hosted-api",
        deps="google-genai, tiktoken  (pip install 'hif[gemini]')",
        setup="Vertex AI: GOOGLE_CLOUD_PROJECT + `gcloud auth application-default "
              "login` (needed for logprobs) · or GEMINI_API_KEY (no logprobs)",
        teacher_forcing=False, logprobs="top-k",
        example_models=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        notes="Top-20 logprobs on Vertex AI only; the developer API degenerates.",
    ),
    "openai-vlm": BackendInfo(
        name="openai-vlm", kind="hosted-api",
        deps="openai, tiktoken  (pip install 'hif[openai]')",
        setup="OPENAI_API_KEY env var (billed per token)",
        teacher_forcing=False, logprobs="top-k", multimodal=True,
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


def metric_support(
    metric: str,
    backend: str,
    *,
    attention_enabled: bool | None = None,
) -> str | None:
    """Return None if `metric` can be produced here, else a reason + the fix.

    This is the guard that catches e.g. `--metric input_entropy_shift_bits
    --backend ollama`: that measurement is input-side, ollama has no teacher
    forcing.

    `attention_enabled` is the one requirement that is not a property of the
    backend. Pass the run's effective `config.attention.enabled` to have the
    attention rows gated on the stage that actually produces them; leave it
    `None` (the default, used by the static `hif models` table) to answer the
    backend question alone, which for those two rows is always "yes".
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
    if metric in ATTENTION_METRICS and attention_enabled is False:
        # NOT a backend refusal. The attention rows come from the analysis
        # encoder reading text, so every backend can produce them; what they
        # need is the optional stage, which is off by default.
        return (
            f"'{metric}' is produced by the attention-analysis stage, which is "
            f"off by default. It does not need anything from the '{backend}' "
            f"backend — the entropy is an analysis encoder's own attention over "
            f"text (the prompt, or the model's generated continuation), not the "
            f"target model's.\n"
            f"  Fix: pass --diagnostics (or set `[attention] enabled = true` in "
            f"a --config file) to run the stage."
        )
    if info.logprobs == "selected-only" and metric in _NEEDS_DISTRIBUTION:
        return (
            f"'{metric}' needs a token distribution, but the '{backend}' backend "
            f"returns only the selected token (no logprobs), so it degenerates.\n"
            f"  Fix: use a backend with logprobs (hf, openai, ollama), pass "
            f"--surrogate, or pick a measurement that does not need one."
        )
    if info.logprobs == "selected-only" and metric in _NEEDS_TWO_DISTRIBUTIONS:
        return (
            f"'{metric}' is a divergence between two token distributions, but "
            f"the '{backend}' backend returns only the selected token. Between "
            f"two point masses the divergence is 0 when the tokens agree and "
            f"exactly 1 bit when they differ — a token-agreement rate, not the "
            f"quantity this key names, so it is reported ABSENT rather than "
            f"emitted under a definition it no longer satisfies.\n"
            f"  Fix: use a backend with logprobs (hf, openai, ollama). "
            f"--surrogate does NOT recover this one: it rebuilds the per-step "
            f"basis the entropy measurements read, which this measurement does "
            f"not use."
        )
    return None


def signals_available(
    backend: str, *, attention_enabled: bool | None = None
) -> dict[str, bool]:
    """Map each measurement key → whether `backend` can produce it."""
    all_metrics = INPUT_SIDE_METRICS | ATTENTION_METRICS | OUTPUT_SIDE_METRICS
    return {
        m: metric_support(m, backend, attention_enabled=attention_enabled) is None
        for m in sorted(all_metrics)
    }
