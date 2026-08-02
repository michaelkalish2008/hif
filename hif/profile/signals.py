"""Measurement extraction from a BehavioralRangeProfile.

Single source of truth for turning a profile object into the scalar
measurements hif reports. The CLI presentation layer (hif/cli.py) and the
SessionEngine record path (hif/engine.py) both build on these functions, so a
number shown in a terminal table and a number emitted in a machine record can
never diverge.

Natural units only
------------------
Every measurement is reported in the unit it is measured in — bits for
entropies and surprisals, dimensionless for correlations, cosine distance for
embedding displacement, a fraction for a count of steps. Key names carry the
unit.

Nothing here normalises an unbounded quantity into [0, 1], and nothing reports
``1 - x``. Three things were removed, and this docstring records why so they
do not come back:

* ``normalized`` block — unbounded quantities were divided by
  ``log2(vocab_size)``. That normaliser then surfaced as the strongest
  apparent "behavioural" feature in the study corpus (r = 0.980, constant
  within a model): tokenizer metadata masquerading as behaviour. Bounded
  scales also saturate, and bits are self-interpreting (4.9 bits is about the
  uncertainty of a uniform choice among ~30 tokens; "0.0178" is not
  checkable).
* ``levels`` (low/medium/high) — assigning a level is an inference requiring a
  null this project never established. The decision rule built on them
  measured a ~43% false-positive rate on pairs of runs known to be identical.
* Duplicate names — ``continuity`` was ``1 - sensitivity`` computed from the
  same JS divergences, and the ``wager`` reading was byte-for-byte the same
  computation as the ``surprise`` aggregate. Reporting one measurement twice
  under two names inflates the apparent dimensionality of the signal set.
  Each quantity now appears exactly once.

Privacy note: everything returned here is a derived scalar. Nothing in this
module reads or emits raw token distributions — records built from these
values are safe under the compute-and-discard default. Raw-artifact
persistence is a separate, explicit opt-in (TraceabilityConfig / --trace)
handled by the engine, never here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

# Version of the machine-record shape emitted by signals_record(). Bump on
# any breaking change to record structure (field renames/removals). Additive
# keys do not require a bump.
#
# record-v2: the `signals`/`readings` split, the `normalized` and `levels`
# blocks, and `findings_levels` are gone; a single flat `measurements` block
# in natural units replaces them.
# record-v3: the per-record `units` block is opt-in (`--units`)
# rather than always present — it is identical for every record of a given
# schema_version and `hif schema` prints it on demand. The field-descriptor
# blocks are renamed to the names docs/MEASUREMENTS.md Part 4 gives them:
# `field` -> `perturbation_field`, `branch_field` -> `trajectory_branch_field`.
# record-v4: every measurement declares a SUBJECT (whose behaviour
# the number describes), and quantities whose subject on the active backend is
# `prompt-only` are no longer emitted inside `measurements` with a surrogate
# flag — they move to a separate top-level `prompt_measurements` block naming
# the reference model that produced them. See the "Subject" section below.
# record-v5 (current): a `provenance` block carries which model actually filled
# each role in the run (teacher forcing, output distributions, attention
# analysis) plus the degradation flags, so a published profile carries the
# evidence behind its subject declarations rather than only the claim. The
# record path cross-checks every emitted measurement against it and refuses to
# emit a record that contradicts it (hif/profile/provenance.py). Absent — like
# any other absent block — on a profile built before the block existed.
RECORD_SCHEMA_VERSION = "record-v5"

# Version of the measurement set this package computes. Minor bumps within a
# major family are additive supersets (see cli._signal_set_family), so
# artifacts across them still compare over the intersection.
#
# hif-v2.1: added input_entropy_std_bits and
# branch_pairwise_cosine_similarity — the natural-unit forms of the Stability
# and Continuity aggregates, which were computed but never surfaced.
# hif-v3 (current): `measurements` no longer contains prompt-only quantities.
# This is a REMOVAL from the measurement set, not an additive superset — a
# hif-v2 artifact carries numbers under keys a hif-v3 artifact deliberately
# does not, so the two are not intersectable without silently comparing a
# fact about the target against a fact about a reference model.
# hif-v3.1 (current): added output_step_jsd_bits (Shift ◆ — the step-to-step
# output divergence that existed only as a chart, so a reader could see it on
# the companion website and not reproduce it with the CLI) and its companion
# output_step_topk_overlap_fraction. Purely additive within the hif-v3 family:
# no key was removed, so `hif compare` still intersects across v3 and v3.1.
# The same release made `perturbation_jsd_bits` ABSENT on selected-only
# backends, which is an absence rule on an already-optional key rather than a
# change of set membership — a hif-v3 artifact from such a backend carries a
# number a hif-v3.1 run declines to produce, and declining is the correction.
# hif-v3.2 (current): the same absence rule now covers the candidate-cloud
# quantities on a selected-only backend with no surrogate recovery
# (output_entropy_bits, output_entropy_step_delta_bits,
# candidate_cluster_entropy_bits, and the two analyses built on the same cloud).
# hif/models/capabilities.py already declared them unproducible there and the
# CLI already refused `--metric` for them; `measurements()` emitted 0.0 anyway,
# because a point mass has exactly one candidate and the entropy of a cloud of
# one is zero by construction. That is a fabricated measurement claim under the
# absent-not-pinned rule. No key was removed from the set — the rule is an
# absence condition on already-optional keys, so `hif compare` still intersects
# across the v3 family.
SIGNAL_SET_VERSION = "hif-v3.2"


def profile_hash(model_name: str, prompt: str, seed: int) -> str:
    key = f"{model_name}|{prompt}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# The measurement registry
# ---------------------------------------------------------------------------

# Every measurement is a triple — observable × functional × resolution — plus
# the contract details a consumer needs (key, unit, definition) and the
# surrogate caveat. This registry is the single source of truth: the CLI
# table, `hif schema`, the Markdown renderers, and the record path all derive
# from it. Adding a measurement means adding one row here (see
# CONTRIBUTING.md).

# Resolution — the granularity of the underlying series a scalar summarises
# within one run. The record always carries the run-level scalar; resolution
# says what that scalar is a summary OF, and therefore whether a token-level
# trace exists behind the number (docs/MEASUREMENTS.md Part 2):
#
#   "per-position"  one sample per prompt/context position — the trace behind
#                   the scalar is indexed by token position.
#   "per-step"      one sample per generation step — the trace is indexed by
#                   output step.
#   "aggregate"     the quantity exists only at whole-run level (across
#                   perturbation variants, trajectory branches, or the run's
#                   endpoints); there is no per-token trace to restore.
RESOLUTIONS: tuple[str, ...] = ("aggregate", "per-step", "per-position")

# Functional — which family of functional produces the number.
# "information-theoretic" reads the shape of a distribution (entropy,
# surprisal, JSD, trace correlation); "geometric" reads where the mass sits
# in embedding space (distance, cluster structure).
FUNCTIONALS: tuple[str, ...] = ("information-theoretic", "geometric")

# Subject — WHOSE behaviour the number describes. The triple says what was
# measured and at what granularity; the subject says who it is about, which
# the record previously could not express.
#
# The distinction that matters is not "was a proxy involved" but "whose
# behaviour is standing in for whose". A fixed instrument (a teacher-forcing
# surrogate, a local encoder) applied to data the target model actually
# produced is a reading instrument on real data: the number still moves when
# the target's behaviour moves, so it is a fact about the target, read
# indirectly. An instrument applied only to the prompt is a different thing
# entirely: nothing the target did enters the computation, the number is
# deterministic in prompt text + instrument weights + seed, and it cannot see
# the target at all. Emitting the second kind inside `measurements` with a
# caveat flag says "a caveated fact about this model"; it is not a fact about
# this model.
#
#   "target-distribution"  computed from the target model's own probability
#                          distributions (its forward pass over its input, or
#                          over its own generation).
#   "target-output-text"   computed by a local instrument (embedder, analysis
#                          encoder) reading text the target actually
#                          generated. The instrument is fixed; the data is the
#                          target's.
#   "mixed"                couples a target-derived series with a series
#                          derived from something other than the target. The
#                          target participates but does not solely determine
#                          the number.
#   "prompt-only"          computed from the prompt text alone under a fixed
#                          reference model or encoder. No data the target
#                          produced enters. NOT a measurement of the target,
#                          and therefore never emitted inside `measurements`.
SUBJECT_TARGET_DISTRIBUTION = "target-distribution"
SUBJECT_TARGET_OUTPUT_TEXT = "target-output-text"
SUBJECT_MIXED = "mixed"
SUBJECT_PROMPT_ONLY = "prompt-only"

SUBJECTS: tuple[str, ...] = (
    SUBJECT_TARGET_DISTRIBUTION,
    SUBJECT_TARGET_OUTPUT_TEXT,
    SUBJECT_MIXED,
    SUBJECT_PROMPT_ONLY,
)

# One line per value — the legend `hif schema` prints.
SUBJECT_LEGEND: dict[str, str] = {
    SUBJECT_TARGET_DISTRIBUTION: (
        "the target model's own probability distributions — its forward pass "
        "over its input or its own generation"
    ),
    SUBJECT_TARGET_OUTPUT_TEXT: (
        "a fixed local instrument (embedder or analysis encoder) reading text "
        "the target model actually generated"
    ),
    SUBJECT_MIXED: (
        "a target-derived series coupled with a series derived from something "
        "other than the target; the target participates but does not solely "
        "determine the number"
    ),
    SUBJECT_PROMPT_ONLY: (
        "the prompt text alone under a fixed reference model — no data the "
        "target produced enters, so it is not a measurement of the target and "
        "is reported in `prompt_measurements`, never in `measurements`"
    ),
}


@dataclass(frozen=True)
class Measurement:
    """One registry row: a measurement's triple and its record contract.

    key             record/measurement key — descriptive and unit-suffixed;
                    this is the stable machine name and never changes.
    name            human-readable quantity name, shown in CLI and report
                    tables.
    unit            the natural unit ("bits", "dimensionless", "cosine
                    distance", "fraction of analysed generation steps").
    definition      what the quantity is, in one or two sentences, including
                    its bound (or that it is unbounded) and any absence rule.
    observable      what the quantity is computed from — what the forward
                    pass (or the encoder over its text) exposes.
    functional      one of FUNCTIONALS.
    resolution      one of RESOLUTIONS (see the comment above).
    subject         one of SUBJECTS — whose behaviour the number describes when
                    the target's own machinery produced it (the `[F]` case).
                    Required: a row that cannot say who it is about should not
                    be in the set.
    label           optional canonical shorthand from the docs ("Wager ▲",
                    "Continuity"). None when no established shorthand exists —
                    a made-up name would be worse than none.
    surrogate_group which surrogate caveat applies when the target model could
                    not produce the quantity itself: "input" measurements
                    inherit findings.surrogate_model_name's proxy caveat,
                    "output" measurements inherit
                    output_distribution_surrogate_name's, "" inherits neither.
                    This is a claim about the computation, verifiable against
                    hif/profile/builder.py — see the per-row comments.
    subject_under_surrogate
                    what `subject` degrades to when the surrogate named by
                    surrogate_group actually stood in. None when the subject
                    does not change (no surrogate can apply to this row).
                    Subject is therefore backend-dependent by construction
                    rather than by a static value that is wrong half the time:
                    `effective_subject()` resolves it against the surrogates a
                    given run actually used.
    """

    key: str
    name: str
    unit: str
    definition: str
    observable: str
    functional: str
    resolution: str
    subject: str
    label: Optional[str] = None
    surrogate_group: str = ""
    subject_under_surrogate: Optional[str] = None


MEASUREMENT_REGISTRY: tuple[Measurement, ...] = (
    Measurement(
        key="input_entropy_shift_bits",
        name="Input entropy shift (bits)",
        unit="bits",
        definition=(
            "mean |mean input-token entropy(variant) − mean input-token "
            "entropy(baseline)| over perturbation variants. Unbounded above."
        ),
        observable="input distribution",
        functional="information-theoretic",
        resolution="aggregate",
        # builder.py step 6: `tf_model = model if model.supports_teacher_forcing
        # else surrogate_model`, and stability.py differences ONLY the resulting
        # per-variant mean_entropy values. On [F] those distributions are the
        # target's; under --surrogate they are the surrogate's over prompt text
        # the target never saw a token of.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_PROMPT_ONLY,
        label=None,  # the historical "Input Stability" named the removed
        # inverted score (1 − x), not this quantity — carrying that name
        # forward would re-attach the score reading.
        surrogate_group="input",
    ),
    Measurement(
        key="input_entropy_std_bits",
        name="Input entropy shift spread (bits)",
        unit="bits",
        definition=(
            "standard deviation (ddof=1) of the per-variant input entropy "
            "shifts. The spread of the model's entropy response across "
            "perturbations. Unbounded above; absent when fewer than two "
            "variants exist."
        ),
        observable="input distribution",
        functional="information-theoretic",
        resolution="aggregate",
        # Same series as input_entropy_shift_bits (its standard deviation
        # rather than its mean), so the same subject and the same degradation.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_PROMPT_ONLY,
        label="Stability",  # the natural-unit form of the Stability
        # aggregate (see SIGNAL_SET_VERSION history above).
        surrogate_group="input",
    ),
    Measurement(
        key="perturbation_jsd_bits",
        name="Perturbation JSD (bits)",
        unit="bits",
        definition=(
            "mean Jensen-Shannon divergence between the baseline output "
            "distribution and each perturbed variant's. Bounded to [0, 1] by "
            "definition in log base 2. Absent on a backend that returns only "
            "the selected token: two point masses have no distributional "
            "overlap to diverge over, so the computation would report a "
            "token-disagreement rate under a key that promises a divergence "
            "between distributions."
        ),
        observable="output distribution",
        functional="information-theoretic",
        resolution="aggregate",
        # Always the target's own distributions. builder.py step 6 calls
        # compute_sensitivity_metrics(output_trace, variant_trace, ...) on the
        # RAW traces, before the step-6b surrogate recovery; nothing downstream
        # re-derives the JSDs from `semantic_steps`.
        #
        # ABSENCE, NOT A COMMENT. On a selected-only backend those raw traces
        # are point masses: JSD between two point masses is 0 when the selected
        # tokens agree and exactly 1 bit when they differ, so the mean is a
        # token-disagreement rate — a different quantity, and the surrogate
        # recovery that rescues the other output-side rows never reaches this
        # one. This used to be documented in a comment here and emitted anyway,
        # which is the failure the `subject` field was introduced to stop: a
        # measurement that stopped describing what its definition says, still
        # reported under the same key. `_all_measured_values` now omits it, and
        # the token-agreement rate is NOT re-admitted under another key — it
        # would have to pass the Significance Gate on its own, and no run has
        # been shown to need it.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        label="Sensitivity",  # the quantity the historical `sensitivity`
        # score was computed from (mean JS divergence per variant).
        # CORRECTED (was "output"): the surrogate recovery in builder.py step 6b
        # substitutes `semantic_steps` for the distribution/semantic/exposure
        # metrics and rebuilds the perturbation FIELD basis, but never touches
        # `all_sensitivity_metrics`, which is what this key reduces. Flagging it
        # as surrogate-derived claimed a proxy that never ran.
        surrogate_group="",
    ),
    Measurement(
        key="io_correlation_r",
        name="Input/output correlation (r)",
        unit="dimensionless",
        definition=(
            "Pearson r between per-variant input entropy shift and "
            "per-variant JSD. Bounded to [-1, 1] by definition; reported "
            "signed."
        ),
        observable="input × output distributions",
        functional="information-theoretic",
        resolution="aggregate",
        # The genuine mixed case, and the reason "mixed" exists as a value.
        # stability.py: pearsonr(entropy_shifts, js_divergences). The second
        # series is always the target's (see perturbation_jsd_bits); the first
        # is the target's on [F] and the surrogate's under --surrogate. So
        # under a surrogate this is neither a fact about the target alone nor
        # prompt-only: the target's output response is half the computation,
        # and a correlation cannot be attributed to one of its two series. It
        # stays in `measurements` — the target's data does enter — but it is
        # declared `mixed` rather than lumped in with the target-side rows,
        # because r says how the surrogate's reading of the prompt tracks the
        # target's reaction, which is a claim about the pair.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_MIXED,
        label=None,  # no established shorthand — the docs name it only by
        # its zone ("Center").
        surrogate_group="input",
    ),
    Measurement(
        key="io_cosine_similarity",
        name="Input/output cosine similarity",
        unit="dimensionless",
        definition=(
            "cosine similarity between the input embedding and the output "
            "embedding. Bounded to [-1, 1] by definition."
        ),
        observable="input/output text embeddings",
        functional="geometric",
        resolution="aggregate",
        # similarity.py::_mean_io_cosine over (prompt, output) pairs collected
        # in builder.py step 6. The embedder is a fixed local instrument; the
        # output texts are the target's actual generations. Not "mixed": the
        # other member of each pair is the prompt itself — real input data, not
        # another model's behaviour standing in for the target's. No surrogate
        # path touches it.
        subject=SUBJECT_TARGET_OUTPUT_TEXT,
        label=None,  # no established shorthand (`io_sim` is a field name,
        # not a doc name).
    ),
    Measurement(
        key="prompt_surprisal_excess_bits",
        name="Prompt surprisal excess (bits)",
        unit="bits",
        definition=(
            "mean max(0, surprisal(token) − H(distribution)) over "
            "teacher-forced prompt positions. Unbounded above."
        ),
        observable="input distribution",
        functional="information-theoretic",
        resolution="per-position",
        # builder.py step 2: input_analysis comes from the target when it
        # teacher-forces, else from analyze_input_side(surrogate_model, prompt).
        # In the surrogate case every position record — surprisal AND entropy —
        # is the surrogate's over the prompt. The target contributes nothing.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_PROMPT_ONLY,
        label="Wager ▲",
        surrogate_group="input",
    ),
    Measurement(
        key="candidate_cluster_entropy_bits",
        name="Candidate cluster entropy (bits)",
        unit="bits",
        definition=(
            "mean Shannon entropy of the semantic-cluster mass distribution "
            "over each generation step's top-K candidate cloud. Unbounded "
            "above (bounded in practice by log2 of the cluster count)."
        ),
        observable="output distribution",
        functional="geometric",
        resolution="per-step",
        # builder.py step 8 iterates `semantic_steps`, which step 6b replaces
        # with output_steps_via_surrogate(surrogate, prompt, continuation_text)
        # on a selected-only backend. The surrogate is then teacher-forced over
        # the target's ACTUAL generated continuation — a reading instrument on
        # the target's real output, not a stand-in for the target's behaviour.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_TARGET_OUTPUT_TEXT,
        label=None,  # "Cluster Entropy" in the docs names the per-step
        # component, not a canonical instrument shorthand.
        surrogate_group="output",
    ),
    Measurement(
        key="output_entropy_bits",
        name="Output entropy (bits)",
        unit="bits",
        definition=(
            "mean Shannon entropy of the per-step top-K output distribution. "
            "A lower bound on full-vocabulary entropy when the distribution "
            "is truncated."
        ),
        observable="output distribution",
        functional="information-theoretic",
        resolution="per-step",
        # builder.py step 7 iterates the same `semantic_steps` basis — see
        # candidate_cluster_entropy_bits.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_TARGET_OUTPUT_TEXT,
        label="Entropy ●",
        surrogate_group="output",
    ),
    Measurement(
        key="output_entropy_step_delta_bits",
        name="Output entropy step delta (bits)",
        unit="bits",
        definition=(
            "mean |H(step i) − H(step i−1)| over the nucleus (95% mass, "
            "renormalised) entropy trace. Unbounded above."
        ),
        observable="output distribution",
        functional="information-theoretic",
        resolution="per-step",
        # Nucleus entropies of the same `semantic_steps` basis — see
        # candidate_cluster_entropy_bits.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_TARGET_OUTPUT_TEXT,
        label=None,  # deliberately NOT "Shift ◆": Shift is the step-to-step
        # JSD (where the mass sits); this is the step-to-step change in the
        # amount of uncertainty. Two steps can carry identical entropy over
        # completely different token sets, so the two quantities are not
        # substitutes. Shift is `output_step_jsd_bits`, the next row but one.
        surrogate_group="output",
    ),
    Measurement(
        key="output_step_jsd_bits",
        name="Output step-to-step JSD (bits)",
        unit="bits",
        definition=(
            "mean Jensen-Shannon divergence between CONSECUTIVE generation "
            "steps' output distributions, JSD(Qⱼ₋₁, Qⱼ). Bounded to [0, 1] by "
            "definition in log base 2. Computed over the stored top-K supports "
            "only: two consecutive steps whose top-K sets are disjoint give "
            "exactly 1 bit however similar their true full-vocabulary "
            "distributions are, so this number must be read together with its "
            "companion output_step_topk_overlap_fraction, which reports how "
            "much support the same transitions actually shared. Absent when "
            "the run has fewer than two generation steps, and on a backend "
            "that returns only the selected token, where consecutive steps are "
            "point masses and the divergence is a token-disagreement indicator "
            "rather than a divergence between distributions."
        ),
        observable="output distribution",
        functional="information-theoretic",
        resolution="per-step",
        # The target's own distributions, and deliberately the RAW ones:
        # hif/metrics/shift.py reads `profile.output_side.steps`, which
        # builder.py step 13 sets from the unrecovered `output_trace`. That is
        # the same series the Shift chart draws, which is the point — one
        # computation feeds both, so the instrument on the website and the key
        # in the record cannot drift. It also means no surrogate can stand in
        # here: on a selected-only backend the quantity is ABSENT rather than
        # proxied, hence no surrogate_group and no subject_under_surrogate.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        label="Shift ◆",
        surrogate_group="",
    ),
    Measurement(
        key="output_step_topk_overlap_fraction",
        name="Output step-to-step top-K overlap (fraction)",
        unit="fraction of shared top-K token ids",
        definition=(
            "mean Jaccard overlap |Aⱼ₋₁ ∩ Aⱼ| / |Aⱼ₋₁ ∪ Aⱼ| between "
            "consecutive generation steps' top-K candidate token-id sets. "
            "Bounded to [0, 1] by construction. It is the resolution limit on "
            "output_step_jsd_bits: at 0 the two steps share no candidate and "
            "the divergence is pinned at its 1-bit ceiling by the truncation "
            "alone, so a high Shift over low overlap is weaker evidence of a "
            "vocabulary pivot than the same Shift over high overlap. Absent "
            "under the same conditions as output_step_jsd_bits."
        ),
        observable="output distribution",
        functional="information-theoretic",
        resolution="per-step",
        # Same series, same raw basis, same absence rules as output_step_jsd_bits.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        label=None,  # no established shorthand — it is new with Shift's
        # admission to the measurement set, and inventing one would imply a
        # doc vocabulary that does not exist.
        surrogate_group="",
    ),
    Measurement(
        key="semantic_centroid_veer_cosine",
        name="Semantic centroid veer (cosine distance)",
        unit="cosine distance",
        definition=(
            "mean step-to-step displacement of the candidate cloud's "
            "semantic centroid in embedding space. Bounded to [0, 2] by "
            "definition."
        ),
        observable="output distribution",
        functional="geometric",
        resolution="per-step",
        # builder.py step 11d embeds each step's candidate cloud from
        # `semantic_steps` — the same basis as the distribution metrics.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_TARGET_OUTPUT_TEXT,
        label="Veer ◈",
        # CORRECTED (was ""): step 11d passes `sf_trace`, which IS the
        # surrogate-recovered basis when step 6b fired. The row was silent
        # about a proxy it actually uses, so on a [P] backend the CLI table
        # showed a surrogate-derived number with no attribution at all.
        surrogate_group="output",
    ),
    Measurement(
        key="attention_entropy_output_bits",
        name="Output attention-row entropy (bits)",
        unit="bits",
        definition=(
            "mean Shannon entropy of the causal-prefix attention row at each "
            "output position. Grows with prefix length; not divided by "
            "log2(prefix length)."
        ),
        observable="attention row",
        functional="information-theoretic",
        resolution="per-position",
        # NOT the target's attention. hif/analysis/attention.py::AttentionAnalyzer
        # is a bidirectional encoder (DistilBERT by default) applied to text as
        # an object — "This is NOT the model under analysis ... The generation
        # mechanism of the model under analysis is never accessed." This row
        # reads `continuation_attention`, the encoder's self-attention over the
        # target's ACTUAL generated continuation, so it is a fixed instrument on
        # the target's real output: it moves when the target's output moves.
        subject=SUBJECT_TARGET_OUTPUT_TEXT,
        label="Spread ■",
    ),
    Measurement(
        key="attention_entropy_input_bits",
        name="Input attention-row entropy (bits)",
        unit="bits",
        definition=(
            "mean Shannon entropy of the causal-prefix attention row at each "
            "input position. Grows with prefix length; not divided by "
            "log2(prefix length)."
        ),
        observable="attention row",
        functional="information-theoretic",
        resolution="per-position",
        # PROMPT-ONLY ON EVERY BACKEND, including [F]. Same encoder as
        # attention_entropy_output_bits (see that row), but this one reads
        # `input_analysis.attention_map` — the encoder's self-attention over the
        # PROMPT. Nothing the target produced enters: the value is a function of
        # prompt text and encoder weights alone, so it is deterministic in the
        # prompt and cannot vary with any model-side change. That is precisely
        # the zero-variance signature the predecessor audit found. It is a real
        # measurement of the prompt under a fixed reference encoder; it is not a
        # measurement of the target, and no backend can make it one.
        subject=SUBJECT_PROMPT_ONLY,
        label="Horizon",  # no glyph: the ▼ symbol is not used in the code.
    ),
    Measurement(
        key="counterfactual_exposure_fraction",
        name="Counterfactual exposure (fraction of steps)",
        unit="fraction of analysed generation steps",
        definition=(
            "steps where a probabilistically accessible alternative token "
            "would have pulled the response toward a different meaning. A "
            "proportion, bounded to [0, 1] by construction."
        ),
        observable="output distribution",
        functional="geometric",
        resolution="per-step",
        # builder.py step 11b passes `exposure_trace` — the same
        # surrogate-recovered basis as the distribution metrics when step 6b
        # fired — to ExposureAnalyzer, which reads each step's topk candidates.
        subject=SUBJECT_TARGET_DISTRIBUTION,
        subject_under_surrogate=SUBJECT_TARGET_OUTPUT_TEXT,
        label="Exposure ◇",
        # CORRECTED (was ""): same omission as semantic_centroid_veer_cosine —
        # the row consumed the proxy basis without declaring it.
        surrogate_group="output",
    ),
    Measurement(
        key="branch_pairwise_cosine_similarity",
        name="Branch pairwise cosine similarity",
        unit="dimensionless",
        definition=(
            "mean pairwise cosine similarity between trajectory branch "
            "embeddings. High = branches converge semantically; low = they "
            "scatter. Bounded to [-1, 1] by definition. This is the "
            "natural-unit form of the Continuity aggregate, reported "
            "directly rather than as a derived score."
        ),
        observable="trajectory branch embeddings",
        functional="geometric",
        resolution="aggregate",
        # trajectory.py embeds each branch's `final_text`. Branches are rollouts
        # the TARGET generated: builder.py step 5 runs analyze_trajectory only
        # when model.supports_teacher_forcing, and returns an empty branch list
        # otherwise, so this quantity is absent rather than proxied on any
        # backend that cannot generate them itself. No surrogate path exists.
        subject=SUBJECT_TARGET_OUTPUT_TEXT,
        label="Continuity",
    ),
)


# ---------------------------------------------------------------------------
# Derived views — thin projections of the registry, kept for callers that
# only need one facet. The registry above is the source of truth.
# ---------------------------------------------------------------------------

# (key, name, surrogate group) — the historical tuple shape.
MEASUREMENTS: list[tuple[str, str, str]] = [
    (m.key, m.name, m.surrogate_group) for m in MEASUREMENT_REGISTRY
]

MEASUREMENT_KEYS: tuple[str, ...] = tuple(m.key for m in MEASUREMENT_REGISTRY)

# "unit — definition" per key, the historical one-line form.
MEASUREMENT_UNITS: dict[str, str] = {
    m.key: f"{m.unit} — {m.definition}" for m in MEASUREMENT_REGISTRY
}

MEASUREMENT_BY_KEY: dict[str, Measurement] = {
    m.key: m for m in MEASUREMENT_REGISTRY
}


# ---------------------------------------------------------------------------
# Subject resolution
# ---------------------------------------------------------------------------


def effective_subject(
    m: Measurement,
    *,
    input_surrogate: bool = False,
    output_surrogate: bool = False,
) -> str:
    """The subject of `m` on a run that used the given surrogates.

    A row's declared `subject` describes the case where the target's own
    machinery produced the quantity. When the surrogate named by the row's
    surrogate_group actually stood in, the subject becomes
    `subject_under_surrogate` — which is the honest answer to "who is this
    number about?" on that backend, and is not always the same answer.
    """
    if m.subject_under_surrogate is None:
        return m.subject
    stood_in = (
        (m.surrogate_group == "input" and input_surrogate)
        or (m.surrogate_group == "output" and output_surrogate)
    )
    return m.subject_under_surrogate if stood_in else m.subject


def run_subjects(p) -> dict[str, str]:
    """key -> effective subject for every registered measurement on this run.

    Reads which surrogates actually stood in from the profile's findings, so
    the answer is the run's, not the registry's default.
    """
    f = getattr(p, "findings", None)
    input_surrogate = bool(getattr(f, "surrogate_model_name", None))
    output_surrogate = bool(
        getattr(f, "output_distribution_surrogate_name", None)
    )
    return {
        m.key: effective_subject(
            m,
            input_surrogate=input_surrogate,
            output_surrogate=output_surrogate,
        )
        for m in MEASUREMENT_REGISTRY
    }


def _all_measured_values(p) -> dict[str, float]:
    """Every value this profile produced, before the subject split.

    Internal: callers want `measurements()` (about the target) or
    `prompt_measurements()` (about the prompt under a reference model). This
    function does not distinguish them, which is exactly the confusion the
    subject field exists to prevent.
    """
    from hif.hourglass.input_side import mean_surprisal_excess
    from hif.hourglass.output_side import output_distribution_degenerate

    out: dict[str, float] = {}
    m = p.metrics
    st = m.stability

    # Did this backend return real distributions at generation time, or only
    # the selected token? The perturbation JSDs were computed from the RAW
    # baseline and variant traces (builder.py step 6, before any surrogate
    # recovery), and every one of those traces came off the same backend as
    # `output_side` — so this one check answers it for all of them.
    selected_only = output_distribution_degenerate(
        getattr(p.output_side, "steps", None) or []
    )
    # Did the step-6b recovery rebuild a real candidate cloud from the target's
    # actual continuation? Without it, every "cloud" on a selected-only backend
    # is a single token, and every quantity read off the cloud answers a
    # question about a set of one rather than the question its key names.
    _f = getattr(p, "findings", None)
    point_mass_cloud = selected_only and not getattr(
        _f, "output_distribution_surrogate_name", None
    )

    # --- perturbation response (input side, output side, and their coupling)
    if st.input_entropy_shift_bits is not None:
        out["input_entropy_shift_bits"] = st.input_entropy_shift_bits
    if getattr(st, "input_entropy_std_bits", None) is not None:
        out["input_entropy_std_bits"] = st.input_entropy_std_bits
    # Absent, not degenerate: between two point masses the "divergence" is 0
    # when the selected tokens agree and 1 bit when they differ, which is a
    # token-disagreement rate and not what `perturbation_jsd_bits` promises.
    # See the registry row.
    if not selected_only:
        if st.perturbation_jsd_bits is not None:
            out["perturbation_jsd_bits"] = st.perturbation_jsd_bits
        elif m.sensitivity:
            # Fall back to the per-perturbation records when the aggregate is
            # absent — same quantity, same unit.
            js = [s.mean_js_divergence for s in m.sensitivity]
            out["perturbation_jsd_bits"] = sum(js) / len(js)
    if st.input_output_correlation is not None:
        out["io_correlation_r"] = st.input_output_correlation

    # --- input/output anchoring
    if m.similarity is not None:
        out["io_cosine_similarity"] = m.similarity.io_sim

    # --- prompt-side surprisal excess (the input-side instrument)
    mean_excess = mean_surprisal_excess(getattr(p.input_side, "positions", None) or [])
    if mean_excess is not None:
        out["prompt_surprisal_excess_bits"] = mean_excess

    # --- candidate-cloud semantics
    if m.semantic and not point_mass_cloud:
        ces = [s.cluster_entropy for s in m.semantic]
        out["candidate_cluster_entropy_bits"] = sum(ces) / len(ces)

    # --- output distribution
    if m.distribution and not point_mass_cloud:
        ents = [d.entropy_bits for d in m.distribution]
        out["output_entropy_bits"] = sum(ents) / len(ents)
    if m.distribution and len(m.distribution) >= 2 and not point_mass_cloud:
        # Nucleus entropy (95% mass, renormalised) so the trace is comparable
        # across backends regardless of how many logprobs each exposes.
        nents = [d.nucleus_entropy_bits for d in m.distribution]
        deltas = [abs(nents[i] - nents[i - 1]) for i in range(1, len(nents))]
        out["output_entropy_step_delta_bits"] = sum(deltas) / len(deltas)

    # --- Shift ◆: step-to-step divergence of the output distribution, plus the
    # top-K overlap that bounds how much of it the truncation could have
    # manufactured. Both come from hif/metrics/shift.py, the same function the
    # Shift chart draws — one computation, so instrument and record agree.
    # shift_summary() returns None (never 0.0) on < 2 steps or a selected-only
    # backend, so both keys are simply omitted there.
    from hif.metrics.shift import shift_summary

    _shift = shift_summary(getattr(p.output_side, "steps", None) or [])
    if _shift is not None:
        out["output_step_jsd_bits"] = _shift.mean_jsd_bits
        out["output_step_topk_overlap_fraction"] = _shift.mean_overlap_fraction

    # --- semantic field (Veer): present only when semantic_field analysis ran.
    # Absent on a point-mass cloud: the "candidate cloud's centroid" would be
    # the selected token's own embedding, so the number would be the selected
    # token's path through embedding space — a different quantity under this
    # key's definition.
    sf = getattr(p, "semantic_field", None)
    if (
        sf is not None
        and getattr(sf, "mean_veer", None) is not None
        and not point_mass_cloud
    ):
        out["semantic_centroid_veer_cosine"] = sf.mean_veer

    # --- counterfactual exposure: present only when the divergence analysis
    # ran and found accessible alternatives. A point mass has none by
    # construction, which is absence of evidence, not a measured zero.
    exp = getattr(p, "exposure", None) or getattr(p, "hallucination", None)
    if exp is not None and getattr(exp, "candidates", None) and not point_mass_cloud:
        out["counterfactual_exposure_fraction"] = exp.exposure

    # --- trajectory continuity, in its natural unit (mean pairwise cosine
    # between branch embeddings). Computed by the trajectory analysis; was
    # previously reduced to a score and never surfaced directly.
    traj = getattr(p, "trajectory", None)
    tc = getattr(traj, "trajectory_continuity", None) if traj is not None else None
    if tc is not None:
        out["branch_pairwise_cosine_similarity"] = float(tc)

    # --- attention-row entropy: present only when attention analysis ran.
    # Both sides in raw bits. The historical "Horizon" reading divided the
    # input-side row entropy by log2(prefix_len); that normaliser leaks
    # sequence-length metadata into a behavioural number, so it is gone.
    from hif.viz.signals._attention import get_attention_map, row_entropy_trace

    _, w_out = get_attention_map(p, "output")
    if w_out:
        tr = row_entropy_trace(w_out)
        if tr:
            out["attention_entropy_output_bits"] = sum(tr) / len(tr)
    _, w_in = get_attention_map(p, "input")
    if w_in:
        tr = row_entropy_trace(w_in)
        if tr:
            out["attention_entropy_input_bits"] = sum(tr) / len(tr)

    return out


def measurements(p) -> dict[str, float]:
    """The measurements OF THE TARGET MODEL this profile supports.

    Absent measurements are OMITTED, never pinned to 0.0 or 1.0: absent means
    the run produced no evidence for that quantity (a backend that cannot
    teacher-force, an analysis stage that did not run), which is a different
    statement from a measured zero.

    Absence extends to "measured something else". A quantity whose effective
    subject on this run is `prompt-only` never touched the target's data, so it
    is omitted here — not emitted with a surrogate flag — and reported in
    `prompt_measurements()` instead. A flag would say "a caveated number about
    this model"; only "this model produced no number" is true.
    """
    subjects = run_subjects(p)
    return {
        k: v
        for k, v in _all_measured_values(p).items()
        if subjects.get(k) != SUBJECT_PROMPT_ONLY
    }


def prompt_measurements(p) -> dict[str, float]:
    """The values whose subject on this run is the prompt, not the target.

    Useful and comparable in their own right — "how surprising is this prompt
    under a fixed reference model" is a real question, and the answer is
    comparable across targets precisely BECAUSE the target does not enter it.
    They are simply not measurements of the target, so they are reported
    separately rather than inside `measurements()`.
    """
    subjects = run_subjects(p)
    return {
        k: v
        for k, v in _all_measured_values(p).items()
        if subjects.get(k) == SUBJECT_PROMPT_ONLY
    }


def _text_analysis_encoder(p) -> Optional[str]:
    """Name of the encoder that read the prompt as text, if one ran.

    The analysis encoder is recorded on the attention map it produced, so this
    reports the model that actually ran rather than the configured default.
    """
    att = getattr(p, "attention_capture", None) or getattr(p, "attention", None)
    if att is None:
        return None
    try:
        if isinstance(att, dict):
            return (
                att.get("input_analysis", {})
                .get("attention_map", {})
                .get("analysis_model")
            )
        return att.input_analysis.attention_map.analysis_model
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return None


def _prompt_reference_model(key: str, p) -> Optional[str]:
    """Whose behaviour a prompt-only value describes."""
    m = MEASUREMENT_BY_KEY[key]
    if m.subject_under_surrogate == SUBJECT_PROMPT_ONLY:
        # Degraded to prompt-only because a teacher-forcing surrogate read the
        # prompt in the target's place.
        return getattr(getattr(p, "findings", None), "surrogate_model_name", None)
    # Statically prompt-only: a local encoder read the prompt text directly.
    return _text_analysis_encoder(p)


def prompt_measurement_block(p) -> Optional[dict]:
    """The record's `prompt_measurements` block, or None when it would be empty.

    Omitted rather than emitted empty, for the same reason an unmeasurable
    quantity is omitted from `measurements`: an empty block would assert that
    the run considered these quantities and found nothing, when in fact none
    was in play.
    """
    values = prompt_measurements(p)
    if not values:
        return None
    return {
        "subject": SUBJECT_PROMPT_ONLY,
        "about": (
            "These describe the PROMPT under the reference model named for "
            "each key, not the model named in this record. No data the target "
            "produced enters their computation, so they cannot vary with the "
            "target's behaviour. They are comparable across targets for "
            "exactly that reason."
        ),
        "reference_models": {
            k: _prompt_reference_model(k, p) for k in values
        },
        "values": values,
    }


# Historical names kept as thin aliases so external callers and tests that
# import them keep working. Both return the same flat measurement dict.
def profile_scores(profile) -> dict[str, float]:
    return measurements(profile)


def extended_signal_values(p) -> dict[str, float]:
    return measurements(p)


# ---------------------------------------------------------------------------
# Field / branch / semantic-field descriptor scalars
# ---------------------------------------------------------------------------


def field_scalars(profile) -> Optional[dict]:
    """Per-profile perturbation-field descriptors, or None when absent.

    Derived scalars only — the raw distributions the field was computed from
    were discarded inside build_profile (unless the run opted into
    traceability, in which case the raw traces live on the profile artifact,
    not here)."""
    f = getattr(profile.metrics, "field", None)
    return f.model_dump() if f is not None else None


def branch_field_scalars(profile) -> Optional[dict]:
    """Per-profile trajectory branch-field descriptors, or None when absent
    (< 2 branches, or a skipped/degenerate trajectory path)."""
    bf = getattr(profile.trajectory, "branch_field", None)
    return bf.model_dump() if bf is not None else None


def semantic_field_scalars(profile) -> Optional[dict]:
    """Per-profile within-generation semantic-field (Veer) summary scalars,
    or None when absent (< 2 generation steps, or the instrument disabled)."""
    sf = getattr(profile, "semantic_field", None)
    if sf is None:
        return None
    return {"mean_veer": sf.mean_veer, "max_veer": sf.max_veer,
            "mean_deformation": sf.mean_deformation, "n_steps": sf.n_steps}


# ---------------------------------------------------------------------------
# The canonical machine record
# ---------------------------------------------------------------------------


def signals_record(
    profile,
    *,
    model_name: str,
    backend: str,
    regime: str,
    seed: int,
    prompt: str,
    latency: Optional[dict] = None,
    trace_path: Optional[str] = None,
    extras: Optional[dict] = None,
    include_units: bool = False,
) -> dict:
    """Build the canonical measurement record for one profiled prompt.

    This is what `--json` prints for a single profile run and what each JSONL
    line of `hif batch` contains. Derived values only — no raw distributions,
    no token alternatives. The model's generated output TEXT is included (it's
    the response the caller already has); reconstructable distribution data is
    not, unless the run traced (then trace_path points at the artifact).

    Round-trip rule: every value here is the same number the terminal table
    displays, sourced from the same function.

    Raises
    ------
    ProvenanceMismatch
        When a measurement's declared subject contradicts what the run
        actually did. See hif/profile/provenance.py::check_provenance for why
        this ends the run rather than warning.
    """
    from hif.profile.provenance import ProvenanceMismatch, check_provenance

    f = profile.findings

    output_text = "".join(s.selected_token_str for s in profile.output_side.steps)

    # The contract check: every emitted measurement's declared subject against
    # what the run actually did. A mismatch means this record would attribute a
    # number to the wrong model, so no record is produced at all.
    violations = check_provenance(profile)
    if violations:
        raise ProvenanceMismatch(
            "measurement subjects contradict what the run recorded:\n  - "
            + "\n  - ".join(violations)
        )
    provenance = getattr(profile, "provenance", None)

    # Quantities whose subject on this backend is the prompt rather than the
    # target are reported in their own block, never inside `measurements`.
    # The block is omitted when nothing falls into it, so a record from a
    # backend where every quantity is target-side is byte-identical in shape
    # to one with no such block at all.
    prompt_block = prompt_measurement_block(profile)

    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "signal_set_version": SIGNAL_SET_VERSION,
        "hash": profile_hash(model_name, prompt, seed),
        "model": model_name,
        "backend": backend,
        "regime": regime,
        "seed": seed,
        "modality": getattr(profile.prompt, "modality", "text") or "text",
        # Every measurement OF THIS MODEL in its natural unit. Absent
        # measurements are omitted. See MEASUREMENT_REGISTRY for what each
        # quantity and unit means, and which subject it has.
        "measurements": measurements(profile),
        **({"prompt_measurements": prompt_block} if prompt_block else {}),

        # Part 4 of docs/MEASUREMENTS.md — behaviour as a region rather than a
        # point. Named as the docs name them.
        "perturbation_field": field_scalars(profile),
        "trajectory_branch_field": branch_field_scalars(profile),
        "semantic_field": semantic_field_scalars(profile),
        "surrogate": {
            "input_side": f.surrogate_model_name,
            "output_distribution": getattr(
                f, "output_distribution_surrogate_name", None
            ),
        },
        # What actually ran, per role — the evidence behind every subject
        # above. Omitted (never emitted empty or guessed) on a profile built
        # before the block existed: an unchecked record must not look like a
        # checked one.
        **(
            {"provenance": provenance.model_dump()}
            if provenance is not None
            else {}
        ),
        "output_text": output_text,
        "output_tokens": len(profile.output_side.generated_ids),
        "input_tokens": len(profile.input_side.prompt_token_ids),
    }
    # Multimodal provenance — InputPartRecord is hash + dims ONLY (never
    # pixels/base64; see schema.py), so it is safe in a derived record.
    # Units are constant per signal_set_version and identical on every record,
    # so they are opt-in (`--units`) rather than repeated on every JSONL line.
    # `hif schema` prints them for every measurement without running a model.
    if include_units:
        keyed = list(record["measurements"]) + list(
            (prompt_block or {}).get("values", {})
        )
        record["units"] = {
            k: MEASUREMENT_UNITS[k] for k in keyed if k in MEASUREMENT_UNITS
        }
    input_parts = getattr(profile.prompt, "input_parts", None)
    if input_parts:
        record["input_parts"] = [part.model_dump() for part in input_parts]
    rs = getattr(profile, "region_sensitivity", None)
    if rs is not None:
        # Derived per-cell JSD grid (multimodal runs) — scalars only.
        record["region_sensitivity"] = rs.model_dump()
    if latency:
        record["latency"] = {k: round(v, 6) for k, v in latency.items()}
    if trace_path:
        record["trace_path"] = trace_path
    if extras:
        record.update(extras)
    return record
