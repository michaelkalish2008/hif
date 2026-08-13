"""Profile builder: orchestrates the full BRI pipeline and assembles a BehavioralRangeProfile."""

from __future__ import annotations

import numpy as np

from hif.clustering.embed import EmbeddingModel
from hif.config import RunConfig
from hif.hourglass.center import compute_center_diagnostics
from hif.hourglass.input_side import (
    InputSideAnalysis,
    analyze_input_side,
)
from hif.hourglass.output_side import (
    OutputSideTrace,
    collect_output_trace,
    output_distribution_degenerate,
    output_distributions_unusable,
    output_steps_via_surrogate,
)
from hif.hourglass.trajectory import TrajectoryAnalysis, analyze_trajectory
from hif.metrics.distribution import DistributionMetrics, compute_distribution_metrics
from hif.metrics.semantic import SemanticMetrics, compute_semantic_metrics
from hif.metrics.sensitivity import SensitivityMetrics, compute_sensitivity_metrics
from hif.metrics.similarity import SimilarityMetrics, compute_similarity_metrics
from hif.metrics.stability import StabilityMetrics, compute_stability_metrics
from hif.models.base import Model
from hif.perturbation import get_generator
from hif.profile.provenance import RunProvenance
from hif.profile.schema import (
    BehavioralRangeProfile,
    Findings,
    MetricBundle,
    ModelIdentity,
    PerturbationRecord,
    PromptRecord,
    RawTraces,
    VariantRawTrace,
)
from hif.utils.logging import get_logger
from hif.utils.seeding import seed_everything

logger = get_logger(__name__)

# Facts about the RUN, not about a case, and not anomalies.
#
# WARNING is for something a reader should act on. Neither of the two notes
# below qualifies: both describe a backend behaving exactly as its capability
# row says it will, on a default the user never chose. Raising them to WARNING
# taught readers that this tool warns during normal operation, which is how a
# real warning gets scrolled past.
#
# The information is not lost. An absent measurement is reported as absent in
# the measurement table (ABSENT_TEXT in cli_render.py), omitted from the record
# rather than zeroed, and explained by the subject/provenance blocks. That is
# where a reader looks for it; a log line during a validation sweep that never
# reads the input side is not.
#
# DEBUG, so `--verbose` restores them. Still deduped, because `--verbose` on a
# hundred-case batch should not print one sentence a hundred times either.
# Same pattern as _warned_top_k_combos in hif/hourglass/output_side.py.
_noted_once: set[tuple] = set()


def _note_once(key: tuple, message: str, *args) -> None:
    if key in _noted_once:
        return
    _noted_once.add(key)
    logger.debug(message, *args)


# ---------------------------------------------------------------------------
# Findings generator
# ---------------------------------------------------------------------------


def generate_findings(
    input_analysis: InputSideAnalysis,
    output_trace: OutputSideTrace,
    center,
    metric_bundle: MetricBundle,
    surrogate_model_name: str | None = None,
    output_distribution_surrogate_name: str | None = None,
) -> Findings:
    """Collect the run's non-inferential provenance.

    This used to bucket six measurements into low/medium/high and emit a
    one-sentence verdict. It no longer does either. See the Findings docstring
    in hif/profile/schema.py for why, and hif.profile.measure.measurements()
    for the numbers themselves.
    """
    # Absent when the similarity stage did not run at all, and absent when it
    # ran but had fewer than two steps to fit a line through — the stage
    # already reports the second case as None, so this only has to not
    # manufacture a value for the first.
    trend = getattr(metric_bundle.similarity, "trend", None)
    similarity_trend_slope = None if trend is None else float(trend)
    return Findings(
        similarity_trend_slope=similarity_trend_slope,
        surrogate_model_name=surrogate_model_name,
        output_distribution_surrogate_name=output_distribution_surrogate_name,
    )


# ---------------------------------------------------------------------------
# Shared pipeline stages
# ---------------------------------------------------------------------------
#
# Stages that are the same computation in build_profile (text) and
# _build_profile_mm (image+text) live here once, so what stays inline in each
# orchestrator is only what actually differs. Where a stage differs, the
# difference is an ARGUMENT, visible at the call site. The shape is kept from
# when there were two builder paths — one of them an image path removed in
# hif-v4 — because it is what stops a stage from growing a second, silently
# different implementation.
#
# A stage whose log line differs between callers keeps that line at the call
# site: moving it in would change what one path emits, and these change nothing.


def _zeroed_input_analysis(
    model: Model, *, prompt_token_ids: list[int], prompt_text: str
) -> InputSideAnalysis:
    """Input-side analysis when nothing could read the prompt.

    Neither the target (no teacher forcing) nor a surrogate (none supplied) can
    produce per-position distributions, so there are none. The scalars are zero
    because there is nothing to average over; the measurements that depend on
    them are omitted downstream rather than reported as zero. `max_entropy`
    survives because it is a property of the tokenizer, not of the run.
    """
    import math

    _note_once(
        ("no-input-side", model.name),
        "%s cannot teacher-force and no surrogate was given, so nothing read "
        "the prompt: the input-side measurements are ABSENT from this run's "
        "records, not zero. Pass surrogate_model= to have a small local model "
        "read the prompt instead — those numbers describe the prompt rather "
        "than %s (docs/MEASUREMENTS.md § Subject).",
        model.name, model.name,
    )
    max_entropy = math.log2(model.vocab_size) if model.vocab_size > 0 else 16.0
    return InputSideAnalysis(
        positions=[],
        prompt_token_ids=prompt_token_ids,
        prompt_text=prompt_text,
        mean_surprisal=0.0,
        mean_entropy=0.0,
        max_entropy=max_entropy,
    )


def _skipped_trajectory(*, start_step: int, rollout_steps: int) -> TrajectoryAnalysis:
    """An empty trajectory: no branches were generated.

    Reached when the backend cannot teacher-force. `branches=[]`, which is
    what `trajectory_analysis_ran` reads.
    """
    return TrajectoryAnalysis(
        start_step=start_step,
        n_branches=0,
        rollout_steps=rollout_steps,
        branches=[],
        convergence_profile=[],
        persistence_score=0.0,
        explosion_score=0.0,
        convergence_score=0.0,
        initial_n_clusters=0,
    )


def _distribution_metrics_for(
    steps: list, *, vocab_size: int, entropy_percentile: float | None = None
) -> list[DistributionMetrics]:
    """One DistributionMetrics per step of the caller's chosen basis.

    `vocab_size` must belong to whichever model produced `steps`: it is the
    uniform-tail upper bound's denominator, so the target's vocabulary over a
    surrogate's distributions would silently mis-bound every step.
    """
    logger.debug("Computing distribution metrics...")
    out: list[DistributionMetrics] = []
    for step in steps:
        probs_arr = np.array([e.prob for e in step.topk], dtype=np.float64)
        logits_arr = np.array([e.logit for e in step.topk], dtype=np.float64)
        # Pass raw (unnormalized) probs so entropy_bits gives the correct lower bound.
        # uniform_tail_entropy uses the tail mass (1 - sum) for the upper bound.
        out.append(
            compute_distribution_metrics(
                probs=probs_arr,
                logits=logits_arr,
                top_k_for_mass=min(10, len(probs_arr)),
                truncated=True,
                vocab_size=vocab_size,
                entropy_percentile=entropy_percentile,
            )
        )
    return out


def _semantic_metrics_for(
    steps: list, *, embedder: EmbeddingModel, cluster_config, enabled: bool = True
) -> list[SemanticMetrics]:
    """One SemanticMetrics per step of the caller's chosen basis.

    Each step's candidates are embedded in the context of the few tokens that
    preceded them, so the same token at two points in a generation is not the
    same point in embedding space.

    Returns [] when the stage is disabled (`--lite`). Every consumer already
    handles a short or empty list — the alternative, per-step zeros, would
    publish a measured value where none was taken.
    """
    if not enabled:
        logger.debug("Skipping semantic metrics — stage disabled.")
        return []
    logger.debug("Computing semantic metrics...")
    out: list[SemanticMetrics] = []
    for i, step in enumerate(steps):
        # Build context prefix from the last min(5, i) already-generated token strings.
        # GPT-2-style tokenizers include leading spaces in token_str, so plain
        # concatenation (no separator) reconstructs natural text.
        context_window = min(5, i)
        if context_window > 0:
            context_prefix = "".join(
                s.selected_token_str for s in steps[i - context_window : i]
            )
            candidate_strings = [context_prefix + e.token_str for e in step.topk]
        else:
            # Step 0: no context yet — use the bare token string
            candidate_strings = [e.token_str for e in step.topk]
        probs_arr = np.array([e.prob for e in step.topk], dtype=np.float64)
        total = probs_arr.sum()
        if total > 0:
            probs_arr = probs_arr / total
        out.append(
            compute_semantic_metrics(
                candidate_strings=candidate_strings,
                probs=probs_arr,
                embedder=embedder,
                cluster_config=cluster_config,
                truncated=True,
            )
        )
    return out


def _exposure_reading(
    config: RunConfig,
    *,
    trace: OutputSideTrace,
    semantic_metrics: list[SemanticMetrics],
    embedder: EmbeddingModel,
):
    """Counterfactual exposure, or None when the stage is disabled.

    Reads `trace.steps[*].topk` for accessible alternatives, so `trace` must
    carry the same basis the distribution and semantic metrics used — otherwise
    it silently finds none on a selected-only backend while the other readings
    used the recovered cloud.
    """
    if not config.exposure.enabled:
        return None
    from hif.analysis.exposure import ExposureAnalyzer

    logger.debug("Running exposure analysis...")
    exposure_analyzer = ExposureAnalyzer(
        embedder=embedder,
        min_prob=config.exposure.min_prob,
    )
    return exposure_analyzer.analyze(
        output_trace=trace,
        semantic_metrics=semantic_metrics,
        distance_threshold=config.exposure.distance_threshold,
    )


def _attention_reading(
    config: RunConfig,
    *,
    model: Model,
    prompt_text: str,
    output_trace: OutputSideTrace,
    variants: list[str],
):
    """Attention-row analysis over (prompt, continuation), plus any variants.

    The analyser is a separate bidirectional encoder reading text as an object;
    the model under analysis contributes only the text. `variants` carries the
    perturbed prompts.

    The caller owns the enabled check and the log line.
    """
    from hif.analysis.attention import AttentionAnalyzer

    analyzer = AttentionAnalyzer(config.attention)
    continuation = model.detokenize(output_trace.generated_ids)
    continuation_token_strs = [s.selected_token_str for s in output_trace.steps]
    return analyzer.analyze(
        prompt_text,
        continuation,
        variants,
        continuation_token_strs=continuation_token_strs,
    )


def _model_identity(model: Model, config: RunConfig) -> ModelIdentity:
    """Who generated. Identical on both paths."""
    return ModelIdentity(
        name=model.name,
        backend=config.model.backend,
        vocab_size=model.vocab_size,
        context_length=model.context_length,
        parameter_count=None,
    )


def _raw_traces(
    config: RunConfig,
    *,
    variant_traces: list[VariantRawTrace],
    trajectory: TrajectoryAnalysis,
) -> "RawTraces | None":
    """Opt-in raw-trace capture (schema 0.7.0), or None — the default.

    None is absence, not an empty capture: with traceability off the transient
    variant traces fall out of scope and there is nothing left to persist.
    Branch traces reuse the trajectory's own Branch records.
    """
    if not config.traceability.enabled:
        return None
    return RawTraces(
        variant_traces=variant_traces,
        branch_traces=list(trajectory.branches),
    )


def _run_provenance(
    *,
    model: Model,
    input_teacher_forcing_model: str | None,
    output_distribution_model: str,
    attention_analysis,
    output_trace: OutputSideTrace,
    trajectory: TrajectoryAnalysis,
) -> RunProvenance:
    """Which model filled each role, and what degraded.

    Every field is an observation made while the pipeline ran, never an
    inference from the result — the evidence each declared `subject` is checked
    against. `output_distribution_model` names the surrogate when step 6b
    recovered a cloud, and the target otherwise.

    `chat_template_present` is read off the target rather than the run: it is
    what the checkpoint declares, and what this run did with it is fixed (hif
    applies no template anywhere). Backends with no tokenizer of the
    checkpoint's own answer None, and None is carried through as None — an
    unasked question is not a "no".
    """
    return RunProvenance(
        generation_model=model.name,
        input_teacher_forcing_model=input_teacher_forcing_model,
        output_distribution_model=output_distribution_model,
        attention_analysis_model=_attention_analysis_model(attention_analysis),
        output_distribution_selected_only=output_distribution_degenerate(
            output_trace.steps
        ),
        target_generated_no_output=not output_trace.steps,
        generation_stop_reason=output_trace.stop_reason,
        trajectory_analysis_ran=bool(trajectory.branches),
        chat_template_present=getattr(model, "chat_template_present", None),
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_profile(
    model: Model,
    prompt: str,
    regime: str,
    config: RunConfig,
    embedder: EmbeddingModel,
    seed: int = 42,
    surrogate_model: "Model | None" = None,
    authored_variants: "list[str] | None" = None,
    variant_output_sink: "dict[str, str] | None" = None,
) -> BehavioralRangeProfile:
    """Orchestrate the full BRI pipeline and return a BehavioralRangeProfile.

    Parameters
    ----------
    model:
        The model under analysis.
    prompt:
        The text prompt to analyze.
    regime:
        A label for the prompt category/regime (e.g., "factual", "creative").
    config:
        Full RunConfig controlling all sub-pipeline parameters.
    embedder:
        EmbeddingModel instance for semantic analysis.
    seed:
        Global random seed.
    surrogate_model:
        Optional HF model used for input-side teacher-forcing when the target
        model does not support it (e.g. all API backends). If None and the
        target model lacks teacher forcing, input-side metrics are zeroed.
    """

    # 1. Seed everything
    seed_everything(seed)

    # 2. Input-side analysis. Whichever branch runs, record WHICH model read
    # the prompt — that identity is what every input-side row's declared
    # subject is a claim about (hif/profile/provenance.py).
    input_teacher_forcing_model: str | None = None
    if model.supports_teacher_forcing:
        logger.debug("Running input-side analysis...")
        input_teacher_forcing_model = model.name
        input_analysis = analyze_input_side(
            model, prompt, top_k=config.generation.top_k
        )
    elif surrogate_model is not None:
        input_teacher_forcing_model = surrogate_model.name
        logger.debug(
            "Running input-side analysis via surrogate (%s) for %s...",
            surrogate_model.name, model.name,
        )
        input_analysis = analyze_input_side(
            surrogate_model, prompt, top_k=config.generation.top_k
        )
    else:
        input_analysis = _zeroed_input_analysis(
            model,
            prompt_token_ids=model.tokenize(prompt),
            prompt_text=prompt,
        )

    # 3. Output trace
    logger.debug("Collecting output trace...")
    output_trace = collect_output_trace(
        model,
        prompt,
        max_new_tokens=config.generation.max_new_tokens,
        top_k=config.generation.top_k,
        seed=seed,
    )

    # 4. Center diagnostics
    logger.debug("Computing center diagnostics...")
    center = compute_center_diagnostics(
        input_analysis,
        output_trace,
        embedder,
        max_entropy=input_analysis.max_entropy,
    )

    # 5. Trajectory analysis (skipped for API models without teacher forcing,
    #    and when the branch budget is zero — `--lite` sets it there). Both
    #    routes produce the same empty TrajectoryAnalysis, so a lite run is
    #    indistinguishable from an API run downstream: absent, not zero.
    context_ids = model.tokenize(prompt) + output_trace.generated_ids
    if model.supports_teacher_forcing and config.trajectory.n_branches > 0:
        logger.debug("Analyzing trajectory...")
        trajectory = analyze_trajectory(
            model,
            context_ids,
            embedder,
            config.trajectory,
            config.cluster,
            seed,
        )
    else:
        reason = (
            "branch budget is zero"
            if config.trajectory.n_branches <= 0
            else f"{model.name} does not support teacher forcing"
        )
        logger.debug("Skipping trajectory analysis — %s.", reason)
        trajectory = _skipped_trajectory(
            start_step=len(context_ids),
            rollout_steps=config.trajectory.rollout_steps,
        )

    # 6. Perturbation analysis
    logger.debug("Running perturbation analysis...")
    perturbation_records: list[PerturbationRecord] = []
    all_sensitivity_metrics: list[SensitivityMetrics] = []
    perturbed_input_analyses: list[InputSideAnalysis] = []
    # Transient perturbation-field members: (generator, variant output trace).
    # Held only long enough to derive the field descriptors below, then
    # discarded — the raw variant distributions reach the artifact only under
    # the traceability opt-in (field.py: measurements and raw data stay apart).
    field_variant_traces: list[tuple[str, OutputSideTrace]] = []
    # Opt-in raw-trace capture (config.traceability.enabled): retain references
    # to the SAME transient variant traces (no recomputation) so the artifact
    # can persist them. Stays empty — and nothing is persisted — by default.
    raw_variant_traces: list[VariantRawTrace] = []
    # Collect (input, output) text pairs for similarity metrics.
    # Baseline pair is index 0; each variant appends one more pair.
    baseline_output_text = "".join(s.selected_token_str for s in output_trace.steps)
    similarity_inputs: list[str] = [prompt]
    similarity_outputs: list[str] = [baseline_output_text]

    # The variant plan: which perturbation texts this run compares against,
    # and under what generator name. Two mutually exclusive sources —
    # researcher-authored variants (passed in by the caller, who resolved
    # them from a workload row's `variants` or the [perturbation]
    # variants_file; the builder does no file I/O) or the generator pipeline.
    # Resolved up front so the measurement loop below is identical for both:
    # nothing downstream knows or cares who authored the text, which is the
    # point — authorship changes provenance, not procedure.
    variant_plan: list[tuple[str, list[str]]] = []
    if authored_variants:
        variant_plan.append(("authored", list(authored_variants)))
    else:
        for gen_name in config.perturbation.generators:
            try:
                generator = get_generator(
                    gen_name,
                    use_llm=config.perturbation.use_llm_perturbation,
                    base_url=config.perturbation.llm_base_url,
                    api_key=config.perturbation.llm_api_key,
                    model=config.perturbation.llm_model,
                )
                pert_result = generator.generate(
                    prompt, config.perturbation.n_variants, seed
                )
            except Exception as exc:
                logger.warning("Perturbation generator %r failed: %s", gen_name, exc)
                continue
            variant_plan.append((gen_name, pert_result.variants))

    for gen_name, gen_variants in variant_plan:
        per_variant_sensitivity: list[SensitivityMetrics] = []
        for variant_index, variant_text in enumerate(gen_variants):
            # Output trace for this variant — the elicitation half of the stage.
            # Skipped under `acquisition = synthesized-input`: the variants are
            # still authored and teacher-forced below (input side), but the
            # model is never asked to generate from them. The four measurements
            # that read a variant continuation go absent, which is the correct
            # reading of "this run was not permitted to elicit that."
            if config.perturbation.elicit_variant_outputs:
                try:
                    variant_trace = collect_output_trace(
                        model,
                        variant_text,
                        max_new_tokens=config.generation.max_new_tokens,
                        top_k=config.generation.top_k,
                        seed=seed,
                    )
                    sens = compute_sensitivity_metrics(
                        output_trace, variant_trace, variant_text, gen_name
                    )
                    per_variant_sensitivity.append(sens)
                    all_sensitivity_metrics.append(sens)
                    # Transient field member — discarded after field computation.
                    field_variant_traces.append((gen_name, variant_trace))
                    if config.traceability.enabled:
                        # Sanctioned exception to compute-and-discard: retain the
                        # same trace reference for the artifact (schema 0.7.0).
                        raw_variant_traces.append(
                            VariantRawTrace(
                                generator=gen_name,
                                variant_index=variant_index,
                                trace=variant_trace,
                            )
                        )
                    # Capture for similarity metrics while the trace is available.
                    variant_output_text = "".join(
                        s.selected_token_str for s in variant_trace.steps
                    )
                    similarity_inputs.append(variant_text)
                    similarity_outputs.append(variant_output_text)
                    if variant_output_sink is not None:
                        # Caller-owned capture (--write); see engine.profile_one.
                        variant_output_sink[variant_text] = variant_output_text
                except Exception as exc:
                    logger.warning(
                        "Sensitivity computation failed for variant %r: %s", variant_text, exc
                    )

            # Input-side analysis for perturbation stability. Use the target's
            # teacher forcing when available, else the surrogate (proxy) that
            # read the base prompt — otherwise Stability and I/O Correlation are
            # absent for API arms even though a proxy is present.
            tf_model = model if model.supports_teacher_forcing else surrogate_model
            if tf_model is not None:
                try:
                    p_input = analyze_input_side(
                        tf_model, variant_text, top_k=config.generation.top_k
                    )
                    perturbed_input_analyses.append(p_input)
                except Exception as exc:
                    logger.warning(
                        "Input-side analysis failed for variant %r: %s", variant_text, exc
                    )

        perturbation_records.append(
            PerturbationRecord(
                generator=gen_name,
                variants=gen_variants,
                sensitivity=per_variant_sensitivity,
            )
        )

    # 6b. Recover real per-step alternatives when the target backend gives
    # none. Anthropic (and other selected-only backends) never returns real
    # logprobs at generation time, so output_trace.steps[*].topk has length 1
    # everywhere: no alternatives to compute entropy, semantic breadth, or
    # exposure from — Breadth/Entropy/Shift compute a trivial 0.0 and Exposure
    # has no candidate to find (every step's only topk entry IS the selected
    # token). When a --surrogate model is available, teacher-force it over
    # prompt+continuation instead, recovering real alternatives at each
    # position — the same proxy technique already used for the input-side
    # Stability/Surprise/Wager readings. `semantic_steps` feeds distribution
    # metrics, semantic/breadth metrics, AND exposure analysis below so all
    # three stay consistent with each other rather than only patching one.
    semantic_steps = output_trace.steps
    output_distribution_surrogate_name: str | None = None
    if output_distributions_unusable(output_trace.steps) and surrogate_model is not None:
        continuation_text = "".join(s.selected_token_str for s in output_trace.steps)
        try:
            surrogate_steps = output_steps_via_surrogate(
                surrogate_model, prompt, continuation_text, top_k=config.generation.top_k
            )
        except Exception as exc:
            logger.warning(
                "Surrogate output-distribution recovery failed for %s: %s", model.name, exc
            )
            surrogate_steps = []
        if surrogate_steps:
            output_distribution_surrogate_name = surrogate_model.name
            semantic_steps = surrogate_steps
    dist_vocab_size = (
        surrogate_model.vocab_size if output_distribution_surrogate_name else model.vocab_size
    )

    # 7. Distribution metrics — one DistributionMetrics per output step, over
    #    the (possibly surrogate-recovered) basis chosen in 6b.
    distribution_metrics = _distribution_metrics_for(
        semantic_steps,
        vocab_size=dist_vocab_size,
        entropy_percentile=config.generation.entropy_percentile,
    )

    # 8. Semantic metrics — one SemanticMetrics per output step, same basis.
    semantic_metrics = _semantic_metrics_for(
        semantic_steps,
        embedder=embedder,
        cluster_config=config.cluster,
        enabled=config.semantic.enabled,
    )

    # 9. Stability metrics
    logger.debug("Computing stability metrics...")
    stability = compute_stability_metrics(
        baseline_input=input_analysis,
        perturbed_inputs=perturbed_input_analyses,
        sensitivity_results=all_sensitivity_metrics,
    )

    # 9a. Perturbation field — derived geometry of the {baseline + variants}
    # cloud around its Jensen-Shannon centroid. Compute-and-discard: the
    # transient variant traces fall out of scope after this call; only the
    # derived scalars persist (docs/ARCHITECTURE.md § Field-model notes).
    # None when < 2 members aligned (e.g. n_variants=0).
    #
    # Distribution basis MUST match the other output-side metrics (the
    # basis-consistency rule, docs/ARCHITECTURE.md § Field-model notes): for a
    # degenerate selected-only backend (e.g. Anthropic — topk length 1)
    # the raw traces are point masses, so the field would collapse to a crude
    # token-agreement signal. When a surrogate is available we proxy-recover each
    # member's distribution the same way `semantic_steps` recovers the baseline,
    # giving closed models a real field at the [P] proxy tier. Truncated backends
    # (gpt-4o top-20) and open backends are non-degenerate → raw distributions are
    # used unchanged (the model's own, more faithful than a proxy).
    logger.debug("Computing perturbation field...")
    from hif.metrics.field import compute_perturbation_field

    def _field_basis_trace(trace: OutputSideTrace) -> OutputSideTrace:
        if output_distributions_unusable(trace.steps) and surrogate_model is not None:
            continuation = "".join(s.selected_token_str for s in trace.steps)
            try:
                recovered = output_steps_via_surrogate(
                    surrogate_model, trace.prompt_text, continuation,
                    top_k=config.generation.top_k,
                )
            except Exception as exc:
                logger.warning(
                    "Field surrogate recovery failed for %r: %s", trace.prompt_text, exc
                )
                recovered = []
            if recovered:
                return trace.model_copy(update={"steps": recovered})
        return trace

    # Baseline reuses the already-recovered semantic_steps (no redundant proxy pass);
    # each variant is recovered on demand when degenerate.
    field_baseline = output_trace.model_copy(update={"steps": semantic_steps})
    field_members = [(gen, _field_basis_trace(t)) for gen, t in field_variant_traces]
    perturbation_field = compute_perturbation_field(field_baseline, field_members)

    # 9b. Similarity metrics (requires at least baseline + one variant)
    logger.debug("Computing similarity metrics...")
    similarity: SimilarityMetrics | None = None
    if len(similarity_inputs) >= 2:
        similarity = compute_similarity_metrics(
            input_texts=similarity_inputs,
            output_texts=similarity_outputs,
            semantic_metrics=semantic_metrics,
            embedder=embedder,
        )
    else:
        logger.debug("Skipping similarity metrics — no perturbation variants available.")

    # 10. Assemble MetricBundle
    metric_bundle = MetricBundle(
        distribution=distribution_metrics,
        semantic=semantic_metrics,
        sensitivity=all_sensitivity_metrics,
        stability=stability,
        similarity=similarity,
        field=perturbation_field,
    )

    # 11. Generate Findings
    used_surrogate = not model.supports_teacher_forcing and surrogate_model is not None
    findings = generate_findings(
        input_analysis, output_trace, center, metric_bundle,
        surrogate_model_name=surrogate_model.name if used_surrogate else None,
        output_distribution_surrogate_name=output_distribution_surrogate_name,
    )

    # 11a. The basis every cloud-reading stage below shares: 6b's recovery
    # when it fired, the target's own trace otherwise. `is` rather than `==`
    # because 6b either rebinds semantic_steps or leaves it pointing at
    # output_trace.steps, so identity is the exact question.
    recovered_trace = (
        output_trace if semantic_steps is output_trace.steps
        else output_trace.model_copy(update={"steps": semantic_steps})
    )

    # 11b. Optional counterfactual exposure analysis (uses cached embeddings
    # — cheap), over the recovered basis.
    exposure_profile = _exposure_reading(
        config,
        trace=recovered_trace,
        semantic_metrics=semantic_metrics,
        embedder=embedder,
    )

    # 11d. Within-generation semantic field (centroid veer) — per-step semantic-centroid
    # trajectory, same recovered basis. Compute-and-discard.
    semantic_field_reading = None
    if config.semantic_field.enabled:
        from hif.analysis.semantic_field import SemanticFieldAnalyzer

        logger.debug("Running within-generation semantic field (Veer)...")
        semantic_field_reading = SemanticFieldAnalyzer(
            embedder, context_window=config.semantic_field.context_window
        ).analyze(recovered_trace)

    # 11c. Optional attention analysis. Up to 5 perturbed prompts go in
    # alongside the baseline — the text path is the one that has them.
    attention_analysis = None
    if config.attention.enabled:
        logger.debug("Running attention analysis...")
        all_variants: list[str] = []
        for pr in perturbation_records:
            all_variants.extend(pr.variants[:2])
        attention_analysis = _attention_reading(
            config,
            model=model,
            prompt_text=prompt,
            output_trace=output_trace,
            variants=all_variants[:5],
        )

    # 12. Build ModelIdentity and PromptRecord
    model_identity = _model_identity(model, config)

    prompt_record = PromptRecord.from_text(
        text=prompt,
        regime=regime,
        token_count=len(input_analysis.prompt_token_ids),
    )

    # 12b. Opt-in raw-trace capture (schema 0.7.0). None (absent) by default.
    raw_traces = _raw_traces(
        config, variant_traces=raw_variant_traces, trajectory=trajectory
    )

    # 12c. Run provenance. The output-distribution role names the surrogate
    # when 6b recovered a cloud — the one field the two paths fill differently.
    provenance = _run_provenance(
        model=model,
        input_teacher_forcing_model=input_teacher_forcing_model,
        output_distribution_model=(
            output_distribution_surrogate_name or model.name
        ),
        attention_analysis=attention_analysis,
        output_trace=output_trace,
        trajectory=trajectory,
    )

    # 13. Return full profile (persist the embedder that actually ran — a
    # fallback changes what the similarity/exposure numbers mean)
    config = _record_effective_embedder(config, embedder)
    return BehavioralRangeProfile(
        model=model_identity,
        prompt=prompt_record,
        input_side=input_analysis,
        output_side=output_trace,
        center=center,
        trajectory=trajectory,
        perturbations=perturbation_records,
        metrics=metric_bundle,
        findings=findings,
        config=config,
        attention_capture=attention_analysis,
        exposure=exposure_profile,
        semantic_field=semantic_field_reading,
        raw_traces=raw_traces,
        provenance=provenance,
    )


def _attention_analysis_model(attention_analysis) -> str | None:
    """The encoder the attention stage actually loaded, or None if it skipped.

    Read off the map the stage produced rather than off the config, so a
    fallback or an override is recorded as what ran. This is never the target
    model: the analyser is a separate bidirectional encoder reading text as an
    object, which is exactly why the attention rows are available on every
    backend and why one of them can never be about the target.
    """
    if attention_analysis is None:
        return None
    try:
        return attention_analysis.input_analysis.attention_map.analysis_model
    except AttributeError:
        return None


def _record_effective_embedder(config: RunConfig, embedder: EmbeddingModel) -> RunConfig:
    """Return a config whose embedding.model_name is the embedder that actually
    loaded (primary or fallback), so the artifact records the encoder behind
    the similarity/exposure values. No-op when they already match."""
    effective = getattr(embedder, "model_name", "") or ""
    if effective and effective != config.embedding.model_name:
        config = config.model_copy(deep=True)
        config.embedding.model_name = effective
    return config
