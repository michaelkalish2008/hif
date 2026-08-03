"""The measurement registry — what measurements exist, and whose behaviour each describes.

This module answers one question: *what is a measurement?* It carries the
ontology (resolution, functional, subject), the ``Measurement`` row type, the
registry rows themselves, the thin derived views over them, and the rule that
resolves a row's subject against the surrogates a given run actually used.

It is the single extension point for the measurement set. Adding a measurement
means adding one row here (see CONTRIBUTING.md); how the value is *taken* lives
in hif/profile/measure.py, and how it is *reported* lives in
hif/profile/record.py.

Nothing in this module reads a profile's numbers, and it imports nothing from
the rest of hif — so a reader can check a row's claims against the pipeline
without the pipeline being able to change what a row says.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Version of the measurement set this package computes. Minor bumps within a
# major family are additive supersets (see cli._signal_set_family), so
# artifacts across them still compare over the intersection.
#
# hif-v2.1: added input_entropy_std_bits and
# branch_pairwise_cosine_similarity — the natural-unit forms of the Stability
# and Continuity aggregates, which were computed but never surfaced.
# hif-v3: `measurements` no longer contains prompt-only quantities.
# This is a REMOVAL from the measurement set, not an additive superset — a
# hif-v2 artifact carries numbers under keys a hif-v3 artifact deliberately
# does not, so the two are not intersectable without silently comparing a
# fact about the target against a fact about a reference model.
# hif-v3.1: added output_step_jsd_bits (the step-to-step
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
# hif-v3.3 (current): removed the `label` field. Rows carried an optional
# shorthand from this project's own vocabulary — Stability, Sensitivity, Wager,
# Continuity, Horizon, Exposure, Veer, Spread, Entropy, Shift — alongside the
# descriptive `name`. Two names for one
# quantity is one name too many, and the shorthand was the one that went wrong:
# "Stability" ended up on `input_entropy_std_bits`, a standard deviation, where
# a higher number means LESS stable. A name that inverts the reading direction
# of its own number is worse than no name. `name` remains, and says what the
# quantity is in the terms it is computed in. No key, unit, definition, or
# absence rule changed, so `hif compare` still intersects across the v3 family;
# a consumer reading `label` off `hif schema` reads `name` instead.
# hif-v3.4 (current): added the `acquisition` field. Every row now says what
# taking the measurement had to bring into existence — nothing (observational),
# authored prompt text (synthesized-input), or model output that did not exist
# before (elicited-output). The fact was always true of the pipeline and was
# recoverable only by reading builder.py and tracing which stage fed which key,
# so `hif schema` could not distinguish an observation from an elicitation and
# the two were reported side by side under one heading. Purely additive
# metadata: no key, unit, definition, or absence rule changed, so `hif compare`
# still intersects across the v3 family. A consumer that does not read
# `acquisition` sees exactly what it saw before.
SIGNAL_SET_VERSION = "hif-v3.4"


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

# Acquisition — what taking the measurement COSTS in produced content.
#
# `subject` says whose behaviour a number describes. It does not say what had
# to be brought into existence to get it, and those are different questions
# with different consequences. Some measurements read the prompt and the one
# continuation the caller already has; others require the tool to author new
# prompt text, or to make the model generate continuations that did not exist
# before and that the caller never asked for.
#
# The distinction is not a performance note. It decides:
#   * whether profiling a hosted model sends text the user never wrote,
#   * whether the run produces model output nobody reviewed,
#   * whether a second run is reproducible from the same inputs (elicited
#     content is resampled; observational readings are not),
#   * and what a provider's terms and a billing line actually cover.
#
# It was previously implicit — recoverable only by reading builder.py and
# tracing which stage fed which key — so a reader of `hif schema` could not
# tell an observation from an elicitation. Now every row says.
#
#   "observational"      computed from the prompt as given and the single
#                        continuation the run already produced. No text is sent
#                        to the model beyond the one call the caller asked for,
#                        and no model output exists afterwards that did not
#                        exist before. Local instruments (embedder, analysis
#                        encoder) may construct strings — exposure embeds
#                        `prefix + candidate` counterfactuals — but nothing
#                        constructed leaves the process or reaches a provider.
#   "synthesized-input"  the tool AUTHORS new prompt text (paraphrase variants)
#                        and puts it through the model's forward pass. The
#                        model does not generate: this is teacher forcing over
#                        text the tool wrote.
#   "elicited-output"    the tool causes the model to GENERATE text that did
#                        not exist before — variant continuations, trajectory
#                        branch rollouts. This is the tier that costs tokens,
#                        multiplies API calls, and produces unreviewed model
#                        output.
ACQUISITION_OBSERVATIONAL = "observational"
ACQUISITION_SYNTHESIZED_INPUT = "synthesized-input"
ACQUISITION_ELICITED_OUTPUT = "elicited-output"

ACQUISITIONS: tuple[str, ...] = (
    ACQUISITION_OBSERVATIONAL,
    ACQUISITION_SYNTHESIZED_INPUT,
    ACQUISITION_ELICITED_OUTPUT,
)

# One line per value — the legend `hif schema` prints.
ACQUISITION_LEGEND: dict[str, str] = {
    ACQUISITION_OBSERVATIONAL: (
        "computed from the prompt as given and the one continuation the run "
        "already produced; nothing is sent to the model beyond the call the "
        "caller asked for, and no new model output exists afterwards"
    ),
    ACQUISITION_SYNTHESIZED_INPUT: (
        "the tool authors new prompt text (paraphrase variants) and teacher-"
        "forces the model over it; the model does not generate"
    ),
    ACQUISITION_ELICITED_OUTPUT: (
        "the tool makes the model generate text that did not exist before "
        "(variant continuations, trajectory branches); costs tokens and "
        "produces unreviewed model output"
    ),
}

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
                    tables. It names the quantity in the terms the quantity is
                    computed in — "Input entropy shift spread (bits)", not a
                    coined shorthand — so `key` and `name` say the same thing
                    at two registers and a reader needs no glossary to move
                    between them.

                    There is deliberately no second naming layer. Rows used to
                    carry an optional `label` holding a shorthand from this
                    project's own vocabulary ("Stability", " prompt surprisal excess",
                    "Continuity"), and the shorthand outlived the quantity it
                    was coined for: "Stability" ended up on a standard
                    deviation, where a HIGHER number means LESS stable, so the
                    name inverted the reading direction of the number it
                    named. The quantities here have accepted names already —
                    Shannon entropy, Jensen-Shannon divergence, Pearson r,
                    cosine similarity, surprisal — and `name` uses them. Chart
                    glyphs live in hif/viz/registry.py, which never needed the
                    measurement row to carry them.
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
    acquisition     one of ACQUISITIONS — what taking the measurement had to
                    bring into existence. `subject` says whose behaviour the
                    number is about; `acquisition` says whether getting it
                    required authoring prompt text or eliciting model output
                    the caller never asked for. Required, and verifiable
                    against hif/profile/builder.py: an "observational" row must
                    read only the baseline trace and the prompt as given.
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
    needs_distribution_pair
                    True when the quantity is computed from (or from a series
                    of) divergences BETWEEN two per-step token distributions.
                    On a selected-only backend both sides are point masses and
                    the divergence collapses to a token-agreement indicator —
                    a different quantity under the same key — so the value is
                    reported ABSENT there, and no surrogate recovers it: the
                    step-6b recovery rebuilds the single-step candidate cloud
                    (`semantic_steps`), which these never read. This is the
                    one capability fact not implied by the other fields;
                    hif/models/capabilities.py derives its gate from it.
    """

    key: str
    name: str
    unit: str
    definition: str
    observable: str
    functional: str
    resolution: str
    subject: str
    acquisition: str = ACQUISITION_OBSERVATIONAL
    surrogate_group: str = ""
    subject_under_surrogate: Optional[str] = None
    needs_distribution_pair: bool = False


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
        acquisition=ACQUISITION_SYNTHESIZED_INPUT,
        subject_under_surrogate=SUBJECT_PROMPT_ONLY,
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
        acquisition=ACQUISITION_SYNTHESIZED_INPUT,
        subject_under_surrogate=SUBJECT_PROMPT_ONLY,
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
        acquisition=ACQUISITION_ELICITED_OUTPUT,
        # CORRECTED (was "output"): the surrogate recovery in builder.py step 6b
        # substitutes `semantic_steps` for the distribution/semantic/exposure
        # metrics and rebuilds the perturbation FIELD basis, but never touches
        # `all_sensitivity_metrics`, which is what this key reduces. Flagging it
        # as surrogate-derived claimed a proxy that never ran.
        surrogate_group="",
        needs_distribution_pair=True,  # JSD(baseline, variant) — see the
        # ABSENCE note above.
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
        acquisition=ACQUISITION_ELICITED_OUTPUT,
        subject_under_surrogate=SUBJECT_MIXED,
        surrogate_group="input",
        needs_distribution_pair=True,  # because of what it correlates: one of
        # its two series IS the per-variant perturbation JSD. On a selected-only
        # backend that series is a token-disagreement rate, so the correlation
        # would couple entropy shifts with token disagreement — a different
        # quantity under a key whose definition names the JSD. No surrogate
        # rescues it either: the recovery rebuilds `semantic_steps`, and the
        # sensitivity series is computed from the raw traces before it.
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
        acquisition=ACQUISITION_ELICITED_OUTPUT,
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
        # Not a substitute for `output_step_jsd_bits`, the next row but one:
        # that one measures where the mass sits, this one the change in how
        # much uncertainty there is. Two steps can carry identical entropy
        # over completely different token sets, so the entropy delta reads 0
        # where the step JSD reads a full bit.
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
        surrogate_group="",
        needs_distribution_pair=True,  # JSD(Qⱼ₋₁, Qⱼ) — consecutive steps.
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
        surrogate_group="",
        needs_distribution_pair=True,  # same series, same absence rules as
        # output_step_jsd_bits.
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
        # prompt and cannot vary with any model-side change — profile two
        # different models on one prompt and this value is bit-identical. It is a
        # real measurement of the prompt under a fixed reference encoder; not a
        # measurement of the target, and no backend can make it one.
        subject=SUBJECT_PROMPT_ONLY,
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
        acquisition=ACQUISITION_ELICITED_OUTPUT,
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
