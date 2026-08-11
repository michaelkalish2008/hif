"""Backend capabilities & model discovery — single source of truth.

Different backends expose different amounts of the model's internals, which
directly determines which measurements can be computed:

- **Teacher forcing** (running the model forward over the prompt to get per-
  position logits) is required for the input-side measurements. Only local
  open-weight backends (hf, tlens) can do it. Hosted APIs and Ollama
  cannot. A `--surrogate` recovers the input-side rows by reading the prompt,
  which makes their subject prompt-only.
- **Top-K logprobs** are required for the output-side measurements
  (output_entropy_bits, perturbation_jsd_bits, io_cosine_similarity). Most
  backends provide them; Anthropic returns only the selected token, so its
  distributions degenerate.

hif-v4 cut ten rows, including both attention rows and every row read off the
trajectory, cluster and exposure stages. Those stages still run and still
record their blocks under `--diagnostics` — they are evidence, not claims — so
nothing below gates on them. ATTENTION_METRICS and TRAJECTORY_METRICS are
consequently empty; see the note on the derived groups below.

This module powers three things: the early metric/backend guard in `profile`,
the `doctor` preflight command, and the `models` discovery command — so a user
learns what they can run *before* a long pipeline zeroes their requested metric.

Availability is not the same question as SUBJECT
------------------------------------------------
What follows says whether a backend can produce a number. Whether that number
is *about the target model* is a separate question, answered by the `subject`
field on each `MEASUREMENT_REGISTRY` row (hif/profile/registry.py). The two come
apart in exactly one place: `--surrogate`. A surrogate makes the input-side
quantities computable on a backend that cannot teacher-force, but it computes
them by reading the PROMPT — the target contributes nothing, so on that backend
their subject is `prompt-only` and they are reported in `prompt_measurements`
rather than in `measurements`. `hif models` prints both facts per backend.

No row is statically prompt-only in hif-v4: every surviving row is about the
target on a backend that can support it, and `prompt-only` is now reached only
dynamically, under `--surrogate`. `hif models` prints both facts per backend.

History, kept because it is the argument that eventually cut the row: this
module once claimed, two paragraphs apart, both that "only open HuggingFace
models expose attention" and that the attention was "not the target's". The
first was false, verifiable in one command — profiling gpt2 and gpt2-medium on
the same prompt returned a bit-identical attention_entropy_input_bits
(1.6677721955190443), which is what a number that cannot see the target looks
like. That fix opened the gate. hif-v4 asked the sharper question the same
evidence supports — a quantity identical across two different targets is not a
measurement OF either — and removed the row instead. Condition 3 of the
Significance Gate (docs/MEASUREMENTS.md) is that argument generalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Measurement groupings by what data they require — DERIVED from the registry
# rows (hif/profile/registry.py), never hand-listed. This module used to keep
# its own frozensets of measurement keys; they drifted twice
# (`input_entropy_std_bits` was silently promised on Anthropic,
# `branch_pairwise_cosine_similarity` vanished from `hif models`), because a
# second list of registry facts is a second place for the facts to be wrong.
# Every fact the groups encode is on the row:
#
#   INPUT_SIDE_METRICS   surrogate_group == "input": the row inherits the
#                        input-side teacher-forcing caveat, which is the same
#                        fact as "requires teacher forcing over the prompt".
#   ATTENTION_METRICS    observable == "attention row": produced by the
#                        attention-analysis stage's encoder, not the backend.
#   TRAJECTORY_METRICS   observable == "trajectory branch embeddings":
#                        rollouts the TARGET generates from its own context.
#                        builder.py step 5 runs the stage only when the target
#                        can teacher-force and returns an empty branch list
#                        otherwise, so these are absent rather than proxied on
#                        a backend that cannot. No surrogate recovers them: a
#                        proxy's rollouts would be the proxy's behaviour, not
#                        a reading of the target's.
#   OUTPUT_SIDE_METRICS  everything else — quantities read off the run's own
#                        generation, with no backend requirement beyond what
#                        the selected-only rules below add. A new registry row
#                        lands here unless its row says otherwise, so a new
#                        measurement cannot fall out of the capability matrix.
#
# tests/unit/test_capability_sets.py asserts the partition (every registry key
# in exactly one group, no group key outside the registry), so the derivation
# cannot silently diverge from the registry it reads.
from hif.profile.registry import MEASUREMENT_REGISTRY

INPUT_SIDE_METRICS = frozenset(
    m.key for m in MEASUREMENT_REGISTRY if m.surrogate_group == "input"
)
# BOTH OF THESE ARE EMPTY IN hif-v4, and that is not a bug: the cut removed
# every row observing an attention row or a trajectory branch. They are kept
# derived rather than deleted because deleting them would delete the guards
# that read them, and a future row with either observable would then arrive
# ungated — the exact failure the "never hand-listed" note above describes.
# While empty they make their consumers dormant, not wrong: the two absence
# contracts in provenance.py that intersect them can never fire, the
# `attention_enabled` gate is inert, and the corresponding halves of
# tests/unit/test_capability_sets.py pass vacuously. Add a row with either
# observable and all of it re-arms with no further edit.
ATTENTION_METRICS = frozenset(
    m.key for m in MEASUREMENT_REGISTRY if m.observable == "attention row"
)
TRAJECTORY_METRICS = frozenset(
    m.key for m in MEASUREMENT_REGISTRY
    if m.observable == "trajectory branch embeddings"
)
OUTPUT_SIDE_METRICS = frozenset(
    m.key for m in MEASUREMENT_REGISTRY
    if m.key not in INPUT_SIDE_METRICS
    and m.key not in ATTENTION_METRICS
    and m.key not in TRAJECTORY_METRICS
)

# Output-side measurements that a selected-only backend cannot produce AT ALL,
# because their input is a real per-step distribution and a point mass is not
# one. Two different absences, both read off the row:
#
#   NEEDS_DISTRIBUTION    quantities computed over one step's candidate cloud.
#                         A point mass is a cloud of one: the entropies are 0.0
#                         by construction, the candidate-cloud centroid is just
#                         the selected token's embedding, and counterfactual
#                         exposure has no alternative to find. A --surrogate
#                         DOES rescue these — step 6b rebuilds `semantic_steps`
#                         by teacher-forcing the proxy over the target's actual
#                         continuation, which is what they all read. That is
#                         why the predicate is surrogate_group == "output":
#                         "recoverable by the output surrogate" and "reads the
#                         per-step candidate cloud" are the same fact about
#                         the computation, declared once on the row.
#   NEEDS_TWO_DISTRIBUTIONS
#                         divergences between two distributions
#                         (needs_distribution_pair on the row). Between two
#                         point masses the JSD is 0 when the tokens agree and
#                         exactly 1 bit when they differ — a token-agreement
#                         rate, a different quantity under the same key, so it
#                         is reported absent rather than emitted. Unlike the
#                         first group, no --surrogate rescues these: the
#                         surrogate recovery in builder.py step 6b rebuilds
#                         `semantic_steps`, which these never read. See the
#                         per-row comments for why each row carries the flag.
NEEDS_DISTRIBUTION = frozenset(
    m.key for m in MEASUREMENT_REGISTRY if m.surrogate_group == "output"
)
NEEDS_TWO_DISTRIBUTIONS = frozenset(
    m.key for m in MEASUREMENT_REGISTRY if m.needs_distribution_pair
)


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
    example_models: list[str] = field(default_factory=list)
    notes: str = ""


# Ordered best-first for a user who wants full-fidelity signals.
BACKENDS: dict[str, BackendInfo] = {
    "hf": BackendInfo(
        name="hf", kind="local-open",
        deps="torch, transformers (base install)",
        setup="none (HF_TOKEN only for gated repos); weights auto-download",
        teacher_forcing=True, logprobs="full",
        example_models=["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen2.5-0.5B",
                        "HuggingFaceTB/SmolLM2-135M", "gpt2",
                        "EleutherAI/pythia-160m"],
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
              "Best for io_cosine_similarity, the one measurement it can fully support.",
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
}


# Recommended --surrogate-model choices: small, ungated, teacher-forcing-capable
# HF causal LMs suitable for recovering input-side signals on backends that
# can't teacher-force themselves (see `--surrogate` in `hif profile`).
# `unsloth/Llama-3.2-1B` is the default (see _load_surrogate in cli.py).
SURROGATE_CANDIDATES: list[str] = [
    "unsloth/Llama-3.2-1B",
    "Qwen/Qwen3-0.6B-Base",
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
    surrogate: bool = False,
) -> str | None:
    """Return None if `metric` can be produced here, else a reason + the fix.

    This is the guard that catches e.g. `--metric input_entropy_shift_bits
    --backend ollama`: that measurement is input-side, ollama has no teacher
    forcing.

    Two requirements here are not properties of the backend:

    `attention_enabled` — pass the run's effective `config.attention.enabled`
    to have the attention rows gated on the stage that actually produces them;
    leave it `None` (the default, used by the static `hif models` table) to
    answer the backend question alone, which for those two rows is always
    "yes".

    `surrogate` — whether the run has a teacher-forcing proxy. A surrogate
    changes what a restricted backend can PRODUCE, not what the numbers are
    about: it recovers the input-side rows by reading the prompt (so they are
    produced, and reported in `prompt_measurements` because their subject
    becomes prompt-only), and it recovers the candidate-cloud rows by
    teacher-forcing over the target's actual continuation. It recovers neither
    the distribution divergences (which read the raw trace) nor the trajectory
    rows (which need the target to roll out its own branches). Availability and
    subject are answered separately — see the module docstring.
    """
    info = BACKENDS.get(backend)
    if info is None:
        return f"Unknown backend {backend!r}."

    if metric in TRAJECTORY_METRICS and not info.teacher_forcing:
        return (
            f"'{metric}' is read off trajectory branches — rollouts the target "
            f"generates from its own context, which requires teacher forcing "
            f"the '{backend}' backend cannot do.\n"
            f"  Fix: use an open-weight model, e.g. `--backend hf` with `Qwen/Qwen3-0.6B-Base`. "
            f"--surrogate does NOT recover this one: a proxy's rollouts would "
            f"be the proxy's behaviour, not a reading of the target's."
        )
    if (
        metric in INPUT_SIDE_METRICS
        and surrogate
        and not info.teacher_forcing
        # …unless the metric ALSO reads a divergence between two output
        # distributions, which no surrogate recovers. Checked before the
        # shortcut so a selected-only backend still refuses those rows —
        # today that is perturbation_jsd_bits.
        and not (
            info.logprobs == "selected-only" and metric in NEEDS_TWO_DISTRIBUTIONS
        )
    ):
        # Produced, by teacher-forcing the proxy over the prompt. The target
        # contributes nothing, so the value lands in `prompt_measurements` —
        # available, and not a measurement of the target.
        return None
    if metric in INPUT_SIDE_METRICS and not info.teacher_forcing:
        return (
            f"'{metric}' is an input-side measurement — it requires teacher "
            f"forcing, which the '{backend}' backend cannot do (hosted APIs and "
            f"Ollama never expose per-token input logits).\n"
            f"  Fix: use an open-weight model, e.g. `--backend hf` with `Qwen/Qwen3-0.6B-Base`; "
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
    if (
        info.logprobs == "selected-only"
        and metric in NEEDS_DISTRIBUTION
        and not surrogate
    ):
        return (
            f"'{metric}' needs a token distribution, but the '{backend}' backend "
            f"returns only the selected token (no logprobs), so it degenerates.\n"
            f"  Fix: use a backend with logprobs (hf, openai, ollama), pass "
            f"--surrogate, or pick a measurement that does not need one."
        )
    if info.logprobs == "selected-only" and metric in NEEDS_TWO_DISTRIBUTIONS:
        return (
            f"'{metric}' is computed from a divergence between two token "
            f"distributions, but "
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


ALL_GROUPED_METRICS = (
    INPUT_SIDE_METRICS | ATTENTION_METRICS | OUTPUT_SIDE_METRICS
    | TRAJECTORY_METRICS
)


def signals_available(
    backend: str,
    *,
    attention_enabled: bool | None = None,
    surrogate: bool = False,
) -> dict[str, bool]:
    """Map each measurement key → whether `backend` can produce it."""
    return {
        m: metric_support(
            m, backend, attention_enabled=attention_enabled, surrogate=surrogate
        ) is None
        for m in sorted(ALL_GROUPED_METRICS)
    }
