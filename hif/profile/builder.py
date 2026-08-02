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
    output_steps_via_surrogate,
)
from hif.hourglass.trajectory import TrajectoryAnalysis, analyze_trajectory
from hif.metrics.distribution import compute_distribution_metrics
from hif.metrics.semantic import SemanticMetrics, compute_semantic_metrics
from hif.metrics.sensitivity import SensitivityMetrics, compute_sensitivity_metrics
from hif.metrics.similarity import SimilarityMetrics, compute_similarity_metrics
from hif.metrics.stability import StabilityMetrics, compute_stability_metrics
from hif.models.base import Model
from hif.models.mm import MultimodalInput, MultimodalModel
from hif.perturbation import get_generator
from hif.profile.schema import (
    BehavioralRangeProfile,
    Findings,
    InputPartRecord,
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
    in hif/profile/schema.py for why, and hif.profile.signals.measurements()
    for the numbers themselves.
    """
    similarity_trend_slope = (
        float(metric_bundle.similarity.trend)
        if metric_bundle.similarity is not None
        else 0.0
    )
    return Findings(
        similarity_trend_slope=similarity_trend_slope,
        surrogate_model_name=surrogate_model_name,
        output_distribution_surrogate_name=output_distribution_surrogate_name,
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_profile(
    model: Model,
    prompt: "str | MultimodalInput",
    regime: str,
    config: RunConfig,
    embedder: EmbeddingModel,
    seed: int = 42,
    surrogate_model: "Model | None" = None,
) -> BehavioralRangeProfile:
    """Orchestrate the full BRI pipeline and return a BehavioralRangeProfile.

    Parameters
    ----------
    model:
        The model under analysis.
    prompt:
        The text prompt to analyze, or a MultimodalInput. A plain str (or a
        MultimodalInput with no media parts) takes the existing text path
        verbatim. Media parts require a MultimodalModel — a ValueError is
        raised before any inference otherwise (MULTIMODAL.md § Builder
        entry point).
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
    # 0. Route by input type/modality (MULTIMODAL.md § Builder entry point).
    #    Text-only inputs — plain str or MultimodalInput without media —
    #    take the existing text path verbatim (byte-identical profiles).
    if isinstance(prompt, MultimodalInput):
        has_media = any(p.kind != "text" for p in prompt.parts)
        if has_media:
            if not model.supports_multimodal_input:
                # Before any inference.
                raise ValueError(
                    f"Input has media parts (modality "
                    f"'{prompt.modality}') but model '{model.name}' does not "
                    "support multimodal input. Use a MultimodalModel backend "
                    "(e.g. HFVLMModel)."
                )
            return _build_profile_mm(
                model, prompt, regime, config, embedder, seed,
                surrogate_model=surrogate_model,
            )
        prompt = prompt.text_concat

    # 1. Seed everything
    seed_everything(seed)

    # 2. Input-side analysis
    if model.supports_teacher_forcing:
        logger.debug("Running input-side analysis...")
        input_analysis = analyze_input_side(
            model, prompt, top_k=config.generation.top_k
        )
    elif surrogate_model is not None:
        logger.debug(
            "Running input-side analysis via surrogate (%s) for %s...",
            surrogate_model.name, model.name,
        )
        input_analysis = analyze_input_side(
            surrogate_model, prompt, top_k=config.generation.top_k
        )
    else:
        import math
        logger.warning(
            "No teacher-forcing and no surrogate — input-side metrics zeroed for %s. "
            "Pass surrogate_model= to compute hermeneutic input-side analysis.",
            model.name,
        )
        max_entropy = math.log2(model.vocab_size) if model.vocab_size > 0 else 16.0
        input_analysis = InputSideAnalysis(
            positions=[],
            prompt_token_ids=model.tokenize(prompt),
            prompt_text=prompt,
            mean_surprisal=0.0,
            mean_entropy=0.0,
            max_entropy=max_entropy,
            volatility_score=0.0,
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

    # 5. Trajectory analysis (skipped for API models without teacher forcing)
    context_ids = model.tokenize(prompt) + output_trace.generated_ids
    if model.supports_teacher_forcing:
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
        logger.debug("Skipping trajectory analysis — %s does not support teacher forcing.", model.name)
        trajectory = TrajectoryAnalysis(
            start_step=len(context_ids),
            n_branches=0,
            rollout_steps=config.trajectory.rollout_steps,
            branches=[],
            convergence_profile=[],
            persistence_score=0.0,
            explosion_score=0.0,
            convergence_score=0.0,
            initial_n_clusters=0,
        )

    # 6. Perturbation analysis
    logger.debug("Running perturbation analysis...")
    perturbation_records: list[PerturbationRecord] = []
    all_sensitivity_metrics: list[SensitivityMetrics] = []
    perturbed_input_analyses: list[InputSideAnalysis] = []
    # Transient perturbation-field members: (generator, variant output trace).
    # Held only long enough to derive the field descriptors below, then discarded
    # — the raw variant distributions are never persisted (field.py privacy
    # invariant: top-k with token identity is reconstructable content).
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

        per_variant_sensitivity: list[SensitivityMetrics] = []
        for variant_index, variant_text in enumerate(pert_result.variants):
            # Output trace for this variant
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
                variants=pert_result.variants,
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
    if output_distribution_degenerate(output_trace.steps) and surrogate_model is not None:
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

    # 7. Distribution metrics — one DistributionMetrics per output step
    logger.debug("Computing distribution metrics...")
    from hif.metrics.distribution import DistributionMetrics
    distribution_metrics: list[DistributionMetrics] = []
    for step in semantic_steps:
        probs_arr = np.array([e.prob for e in step.topk], dtype=np.float64)
        logits_arr = np.array([e.logit for e in step.topk], dtype=np.float64)
        # Pass raw (unnormalized) probs so entropy_bits gives the correct lower bound.
        # uniform_tail_entropy uses the tail mass (1 - sum) for the upper bound.
        dm = compute_distribution_metrics(
            probs=probs_arr,
            logits=logits_arr,
            top_k_for_mass=min(10, len(probs_arr)),
            truncated=True,
            vocab_size=dist_vocab_size,
        )
        distribution_metrics.append(dm)

    # 8. Semantic metrics — one SemanticMetrics per output step
    logger.debug("Computing semantic metrics...")
    semantic_metrics: list[SemanticMetrics] = []
    for i, step in enumerate(semantic_steps):
        # Build context prefix from the last min(5, i) already-generated token strings.
        # GPT-2-style tokenizers include leading spaces in token_str, so plain
        # concatenation (no separator) reconstructs natural text.
        context_window = min(5, i)
        if context_window > 0:
            context_prefix = "".join(
                s.selected_token_str
                for s in semantic_steps[i - context_window : i]
            )
            candidate_strings = [context_prefix + e.token_str for e in step.topk]
        else:
            # Step 0: no context yet — use the bare token string
            candidate_strings = [e.token_str for e in step.topk]
        probs_arr = np.array([e.prob for e in step.topk], dtype=np.float64)
        total = probs_arr.sum()
        if total > 0:
            probs_arr = probs_arr / total
        sm = compute_semantic_metrics(
            candidate_strings=candidate_strings,
            probs=probs_arr,
            embedder=embedder,
            cluster_config=config.cluster,
            truncated=True,
        )
        semantic_metrics.append(sm)

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
    # derived scalars persist (DRIFT_FIELD_MODEL.md). None when < 2 members
    # aligned (e.g. n_variants=0).
    #
    # Distribution basis MUST match the other output-side metrics (DRIFT_FIELD_MODEL
    # §9): for a degenerate selected-only backend (e.g. Anthropic — topk length 1)
    # the raw traces are point masses, so the field would collapse to a crude
    # token-agreement signal. When a surrogate is available we proxy-recover each
    # member's distribution the same way `semantic_steps` recovers the baseline,
    # giving closed models a real field at the [P] proxy tier. Truncated backends
    # (gpt-4o top-20) and open backends are non-degenerate → raw distributions are
    # used unchanged (the model's own, more faithful than a proxy).
    logger.debug("Computing perturbation field...")
    from hif.metrics.field import compute_perturbation_field

    def _field_basis_trace(trace: OutputSideTrace) -> OutputSideTrace:
        if output_distribution_degenerate(trace.steps) and surrogate_model is not None:
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

    # 11b. Optional hallucination analysis (uses cached embeddings — cheap).
    # Reads step.topk for its candidate alternatives, so it needs the same
    # semantic_steps substitution as distribution/semantic metrics above —
    # otherwise it silently finds zero candidates on degenerate backends.
    hallucination_profile = None
    if config.hallucination.enabled:
        from hif.analysis.exposure import ExposureAnalyzer

        logger.debug("Running hallucination analysis...")
        hall_analyzer = ExposureAnalyzer(
            embedder=embedder,
            min_prob=config.hallucination.min_prob,
        )
        exposure_trace = (
            output_trace if semantic_steps is output_trace.steps
            else output_trace.model_copy(update={"steps": semantic_steps})
        )
        hallucination_profile = hall_analyzer.analyze(
            output_trace=exposure_trace,
            semantic_metrics=semantic_metrics,
            distance_threshold=config.hallucination.distance_threshold,
        )

    # 11d. Within-generation semantic field (Veer ◈) — per-step semantic-centroid
    # trajectory. Uses the same surrogate-recovered basis (semantic_steps) as the
    # other output-side readings on degenerate backends. Compute-and-discard.
    semantic_field_reading = None
    if config.semantic_field.enabled:
        from hif.analysis.semantic_field import SemanticFieldAnalyzer

        logger.debug("Running within-generation semantic field (Veer)...")
        sf_trace = (
            output_trace if semantic_steps is output_trace.steps
            else output_trace.model_copy(update={"steps": semantic_steps})
        )
        semantic_field_reading = SemanticFieldAnalyzer(
            embedder, context_window=config.semantic_field.context_window
        ).analyze(sf_trace)

    # 11c. Optional attention analysis
    attention_analysis = None
    if config.attention.enabled:
        from hif.analysis.attention import AttentionAnalyzer

        logger.debug("Running attention analysis...")
        analyzer = AttentionAnalyzer(config.attention)
        # Collect up to 5 perturbed variants across all generators
        all_variants: list[str] = []
        for pr in perturbation_records:
            all_variants.extend(pr.variants[:2])
        # Detokenize the generated continuation
        continuation = model.detokenize(output_trace.generated_ids)
        continuation_token_strs = [s.selected_token_str for s in output_trace.steps]
        attention_analysis = analyzer.analyze(
            prompt,
            continuation,
            all_variants[:5],
            continuation_token_strs=continuation_token_strs,
        )

    # 12. Build ModelIdentity and PromptRecord
    model_identity = ModelIdentity(
        name=model.name,
        backend=config.model.backend,
        vocab_size=model.vocab_size,
        context_length=model.context_length,
        parameter_count=None,
    )

    prompt_record = PromptRecord.from_text(
        text=prompt,
        regime=regime,
        token_count=len(input_analysis.prompt_token_ids),
    )

    # 12b. Opt-in raw-trace capture (schema 0.7.0). None (absent) by default —
    # disabled behavior is unchanged and the transient traces above simply
    # fall out of scope. Branch traces reuse the trajectory's Branch records
    # (empty list when the trajectory stage was skipped).
    raw_traces: RawTraces | None = None
    if config.traceability.enabled:
        raw_traces = RawTraces(
            variant_traces=raw_variant_traces,
            branch_traces=list(trajectory.branches),
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
        exposure=hallucination_profile,
        semantic_field=semantic_field_reading,
        raw_traces=raw_traces,
    )


# ---------------------------------------------------------------------------
# Multimodal builder path (M1: image+text → text)
# ---------------------------------------------------------------------------


def _build_profile_mm(
    model: Model,
    mm_input: MultimodalInput,
    regime: str,
    config: RunConfig,
    embedder: EmbeddingModel,
    seed: int = 42,
    surrogate_model: "Model | None" = None,
) -> BehavioralRangeProfile:
    """Multimodal profile path per MULTIMODAL.md § Design.

    Key differences from the text path (each spec'd, none improvised):
    - prepare() is run exactly once; tokenize() is never called with media.
    - Input-side entropy/surprisal are computed only over
      part_map.text_positions() (Risk rule 3).
    - Trajectory analysis is skipped/zeroed in M1 (Risk rule 6 — rollout
      re-forwarding with pixel state is deferred).
    - Text-part perturbation of a multimodal input is out of scope for M1:
      EXPLICITLY configured text generators are a config error, raised before
      inference. Default (unset) text generators are ignored with a warning
      when a media family is configured — default config on multimodal input
      Just Works with the image_grid_mask family (spec sign-off 2026-07-03).
    - Media perturbation runs via PerturbationFamily (image_grid_mask by
      default); per-variant SensitivityMetrics mirror the text path, and the
      grid-mask traces are assembled into the region_sensitivity artifact.
    - prompt_hash = sha256 over the concatenated part content_hashes (in
      part order); PromptRecord.text holds MultimodalInput.text_concat.
    """
    import hashlib

    from hif.hourglass.input_side import analyze_input_side_mm
    from hif.hourglass.output_side import collect_output_trace_mm
    from hif.perturbation import get_family
    from hif.perturbation.base import PerturbationTrace

    assert isinstance(model, MultimodalModel)

    # Config errors before any inference. Text generators explicitly set by
    # the user are a hard error; the untouched default list is ignored with a
    # warning so the default config works on multimodal input.
    if config.perturbation.generators:
        explicitly_set = "generators" in config.perturbation.model_fields_set
        if explicitly_set:
            raise ValueError(
                "Text-part perturbation of a multimodal input is out of scope "
                "in M1 (MULTIMODAL.md § Builder entry point). Set "
                "perturbation.generators=[] for multimodal profiles."
            )
        logger.warning(
            "Ignoring default text perturbation generators %s on multimodal "
            "input — media perturbation families %s will run instead.",
            config.perturbation.generators,
            config.perturbation.media_families,
        )

    # 1. Seed everything
    seed_everything(seed)

    # 2. Prepare once (processor owns all media/tokenization logic)
    logger.debug("Preparing multimodal input (%s)...", mm_input.modality)
    prepared = model.prepare(mm_input)
    text_concat = mm_input.text_concat

    # 3. Input-side analysis — text positions only (Risk rule 3)
    if model.supports_teacher_forcing:
        logger.debug("Running input-side analysis over text positions...")
        input_analysis = analyze_input_side_mm(
            model, prepared, text_concat, top_k=config.generation.top_k
        )
    elif surrogate_model is not None:
        # Same proxy technique as the text path: teacher-force the surrogate
        # over the concatenated TEXT parts. Risk rule 3 already restricts
        # input-side analysis to text positions on full-access mm backends,
        # so a text-only surrogate reading text_concat is the exact proxy
        # analogue — it recovers the input-side readings (Surprise/Wager and
        # the Stability baseline) that were previously zeroed on closed
        # (API) multimodal arms.
        logger.debug(
            "Running input-side analysis via surrogate (%s) over text parts "
            "for %s...", surrogate_model.name, model.name,
        )
        input_analysis = analyze_input_side(
            surrogate_model, text_concat, top_k=config.generation.top_k
        )
    else:
        import math
        logger.warning(
            "No teacher-forcing and no surrogate — input-side metrics zeroed "
            "for %s. Pass surrogate_model= to compute hermeneutic input-side "
            "analysis.",
            model.name,
        )
        max_entropy = math.log2(model.vocab_size) if model.vocab_size > 0 else 16.0
        input_analysis = InputSideAnalysis(
            positions=[],
            prompt_token_ids=list(prepared.input_ids),
            prompt_text=text_concat,
            mean_surprisal=0.0,
            mean_entropy=0.0,
            max_entropy=max_entropy,
            volatility_score=0.0,
        )

    # 4. Output trace via generate_prepared
    logger.debug("Collecting multimodal output trace...")
    output_trace = collect_output_trace_mm(
        model,
        prepared,
        text_concat,
        max_new_tokens=config.generation.max_new_tokens,
        top_k=config.generation.top_k,
        seed=seed,
    )

    # 5. Center diagnostics
    logger.debug("Computing center diagnostics...")
    center = compute_center_diagnostics(
        input_analysis,
        output_trace,
        embedder,
        max_entropy=input_analysis.max_entropy,
    )

    # 6. Trajectory analysis skipped for multimodal in M1 (Risk rule 6):
    #    context re-forwarding with input_ids alone would drop pixel state.
    logger.debug(
        "Skipping trajectory analysis — multimodal trajectory is deferred in M1."
    )
    context_len = len(prepared.input_ids) + len(output_trace.generated_ids)
    trajectory = TrajectoryAnalysis(
        start_step=context_len,
        n_branches=0,
        rollout_steps=config.trajectory.rollout_steps,
        branches=[],
        convergence_profile=[],
        persistence_score=0.0,
        explosion_score=0.0,
        convergence_score=0.0,
        initial_n_clusters=0,
    )

    # 7. Media perturbation analysis (PerturbationFamily protocol). Each
    #    variant is a NEW MultimodalInput (original never mutated); perturbed
    #    pixels live only in memory as image_bytes parts. SensitivityMetrics
    #    are computed per variant exactly like the text path.
    logger.debug("Running media perturbation analysis...")
    perturbation_records: list[PerturbationRecord] = []
    all_sensitivity_metrics: list[SensitivityMetrics] = []
    perturbed_input_analyses: list[InputSideAnalysis] = []
    # Transient field members (see text path) — discarded after field compute.
    field_variant_traces: list[tuple[str, OutputSideTrace]] = []
    # Opt-in raw-trace capture (see text path §6) — empty unless
    # config.traceability.enabled.
    raw_variant_traces: list[VariantRawTrace] = []
    trace_sensitivity_pairs: list[tuple[PerturbationTrace, SensitivityMetrics]] = []
    # (input, output) text pairs for similarity metrics. Media perturbation
    # never changes the text parts, so every input text is text_concat; the
    # outputs vary with the perturbed pixels.
    baseline_output_text = "".join(s.selected_token_str for s in output_trace.steps)
    similarity_inputs: list[str] = [text_concat]
    similarity_outputs: list[str] = [baseline_output_text]

    for family_name in config.perturbation.media_families:
        try:
            family_kwargs = {}
            if family_name == "image_grid_mask":
                family_kwargs = {
                    "grid_rows": config.perturbation.image_grid_rows,
                    "grid_cols": config.perturbation.image_grid_cols,
                }
            family = get_family(family_name, **family_kwargs)
            mm_variants = family.perturb(
                mm_input, config.perturbation.n_variants, seed
            )
        except Exception as exc:
            logger.warning("Perturbation family %r failed: %s", family_name, exc)
            continue

        per_variant_sensitivity: list[SensitivityMetrics] = []
        variant_descriptors: list[str] = []
        variant_traces: list[PerturbationTrace] = []
        for variant_index, variant in enumerate(mm_variants):
            descriptor = (
                f"{variant.trace.family}[part={variant.trace.part_index}, "
                f"regions={variant.trace.regions}, params={variant.trace.params}]"
            )
            try:
                prepared_variant = model.prepare(variant.input)
                variant_trace = collect_output_trace_mm(
                    model,
                    prepared_variant,
                    text_concat,
                    max_new_tokens=config.generation.max_new_tokens,
                    top_k=config.generation.top_k,
                    seed=seed,
                )
                sens = compute_sensitivity_metrics(
                    output_trace, variant_trace, descriptor, family.name
                )
                per_variant_sensitivity.append(sens)
                all_sensitivity_metrics.append(sens)
                field_variant_traces.append((family.name, variant_trace))
                if config.traceability.enabled:
                    # Sanctioned exception to compute-and-discard (see text
                    # path): retain the reference for the artifact (0.7.0).
                    raw_variant_traces.append(
                        VariantRawTrace(
                            generator=family.name,
                            variant_index=variant_index,
                            trace=variant_trace,
                        )
                    )
                variant_descriptors.append(descriptor)
                variant_traces.append(variant.trace)
                trace_sensitivity_pairs.append((variant.trace, sens))
                similarity_inputs.append(text_concat)
                similarity_outputs.append(
                    "".join(s.selected_token_str for s in variant_trace.steps)
                )
                # Input-side analysis of the perturbed input. Full access:
                # perturbed pixels shift the teacher-forced distributions at
                # TEXT positions, so this is a real, varying input-side
                # series — it feeds input_stability and
                # input_output_correlation exactly like the text path.
                # Patch positions stay excluded (Risk rule 3) inside
                # analyze_input_side_mm.
                #
                # Closed (API) backends fall back to the surrogate proxy,
                # mirroring the text path's tf_model rule so Stability and
                # I/O Correlation are produced (not absent) on API mm arms.
                # KNOWN PROXY LIMIT: media families perturb pixels only, and
                # a text-only surrogate reads the (unchanged) text parts, so
                # its perturbed series equals its baseline — input_stability
                # reads ~1.0 and io_correlation degenerates to a measured
                # 0.0 at the proxy tier. That is the proxy-tier statement
                # "text-position input distributions did not move", the
                # blind approximation of the full-access measurement (e.g.
                # gemma_mm reads 0.998); the record carries the surrogate
                # provenance (findings.surrogate_model_name) so consumers
                # can see the [P] tier.
                tf_model = (
                    model if model.supports_teacher_forcing else surrogate_model
                )
                if tf_model is not None:
                    try:
                        if model.supports_teacher_forcing:
                            p_input = analyze_input_side_mm(
                                model,
                                prepared_variant,
                                text_concat,
                                top_k=config.generation.top_k,
                            )
                        else:
                            p_input = analyze_input_side(
                                tf_model,
                                variant.input.text_concat,
                                top_k=config.generation.top_k,
                            )
                        perturbed_input_analyses.append(p_input)
                    except Exception as exc:
                        logger.warning(
                            "Input-side analysis failed for media variant %s: %s",
                            descriptor, exc,
                        )
            except Exception as exc:
                logger.warning(
                    "Sensitivity computation failed for media variant %s: %s",
                    descriptor, exc,
                )

        perturbation_records.append(
            PerturbationRecord(
                generator=family.name,
                variants=variant_descriptors,
                sensitivity=per_variant_sensitivity,
                traces=variant_traces,
            )
        )

    # 7b. Region-sensitivity artifact (perturbation-JSD per grid cell; never
    #     generation-model attention — Risk rule 7).
    from hif.analysis.region_sensitivity import assemble_region_sensitivity

    region_sensitivity = assemble_region_sensitivity(trace_sensitivity_pairs)

    # 8. Distribution metrics — one per output step (same math as text path)
    logger.debug("Computing distribution metrics...")
    from hif.metrics.distribution import DistributionMetrics
    distribution_metrics: list[DistributionMetrics] = []
    for step in output_trace.steps:
        probs_arr = np.array([e.prob for e in step.topk], dtype=np.float64)
        logits_arr = np.array([e.logit for e in step.topk], dtype=np.float64)
        dm = compute_distribution_metrics(
            probs=probs_arr,
            logits=logits_arr,
            top_k_for_mass=min(10, len(probs_arr)),
            truncated=True,
            vocab_size=model.vocab_size,
        )
        distribution_metrics.append(dm)

    # 9. Semantic metrics — one per output step (same math as text path)
    logger.debug("Computing semantic metrics...")
    semantic_metrics: list[SemanticMetrics] = []
    for i, step in enumerate(output_trace.steps):
        context_window = min(5, i)
        if context_window > 0:
            context_prefix = "".join(
                s.selected_token_str
                for s in output_trace.steps[i - context_window : i]
            )
            candidate_strings = [context_prefix + e.token_str for e in step.topk]
        else:
            candidate_strings = [e.token_str for e in step.topk]
        probs_arr = np.array([e.prob for e in step.topk], dtype=np.float64)
        total = probs_arr.sum()
        if total > 0:
            probs_arr = probs_arr / total
        sm = compute_semantic_metrics(
            candidate_strings=candidate_strings,
            probs=probs_arr,
            embedder=embedder,
            cluster_config=config.cluster,
            truncated=True,
        )
        semantic_metrics.append(sm)

    # 10. Stability metrics. Full-access models feed real per-variant
    #     input-side analyses (computed over text positions in the loop
    #     above); partial-access models feed surrogate-proxy analyses when a
    #     surrogate is available (see the proxy-limit note in the loop), and
    #     only when NEITHER is available are stability's input-side
    #     components ABSENT (None), never pinned.
    stability = compute_stability_metrics(
        baseline_input=input_analysis,
        perturbed_inputs=perturbed_input_analyses,
        sensitivity_results=all_sensitivity_metrics,
    )

    # 10a. Perturbation field (compute-and-discard; see text path §9a).
    from hif.metrics.field import compute_perturbation_field
    perturbation_field = compute_perturbation_field(
        output_trace, field_variant_traces
    )

    # 10b. Similarity metrics from the media-variant generations (baseline +
    #      at least one variant output; input text identical across pairs).
    similarity: SimilarityMetrics | None = None
    if len(similarity_inputs) >= 2:
        logger.info("Computing similarity metrics...")
        similarity = compute_similarity_metrics(
            input_texts=similarity_inputs,
            output_texts=similarity_outputs,
            semantic_metrics=semantic_metrics,
            embedder=embedder,
        )
    else:
        logger.info("Skipping similarity metrics — no perturbation variants available.")

    metric_bundle = MetricBundle(
        distribution=distribution_metrics,
        semantic=semantic_metrics,
        sensitivity=all_sensitivity_metrics,
        stability=stability,
        similarity=similarity,
        field=perturbation_field,
    )

    # Surrogate provenance mirrors the text path: input-side readings on a
    # non-teacher-forcing mm backend came from the proxy, and the record's
    # surrogate.input_side field must say so.
    used_surrogate = not model.supports_teacher_forcing and surrogate_model is not None
    findings = generate_findings(
        input_analysis, output_trace, center, metric_bundle,
        surrogate_model_name=surrogate_model.name if used_surrogate else None,
    )

    # Optional hallucination analysis (text-side outputs only)
    hallucination_profile = None
    if config.hallucination.enabled:
        from hif.analysis.exposure import ExposureAnalyzer

        logger.debug("Running hallucination analysis...")
        hall_analyzer = ExposureAnalyzer(
            embedder=embedder,
            min_prob=config.hallucination.min_prob,
        )
        hallucination_profile = hall_analyzer.analyze(
            output_trace=output_trace,
            semantic_metrics=semantic_metrics,
            distance_threshold=config.hallucination.distance_threshold,
        )

    # Within-generation semantic field (Veer ◈) — mm path uses the raw output
    # trace (no surrogate output-recovery on the mm path). Compute-and-discard.
    semantic_field_reading = None
    if config.semantic_field.enabled:
        from hif.analysis.semantic_field import SemanticFieldAnalyzer

        semantic_field_reading = SemanticFieldAnalyzer(
            embedder, context_window=config.semantic_field.context_window
        ).analyze(output_trace)

    # Optional attention analysis — separate analysis model, text parts only
    # (never generation-model internals; MULTIMODAL.md § Design §3/§7).
    attention_analysis = None
    if config.attention.enabled:
        from hif.analysis.attention import AttentionAnalyzer

        logger.debug("Running attention analysis on text parts...")
        analyzer = AttentionAnalyzer(config.attention)
        continuation = model.detokenize(output_trace.generated_ids)
        continuation_token_strs = [s.selected_token_str for s in output_trace.steps]
        attention_analysis = analyzer.analyze(
            text_concat,
            continuation,
            [],
            continuation_token_strs=continuation_token_strs,
        )

    model_identity = ModelIdentity(
        name=model.name,
        backend=config.model.backend,
        vocab_size=model.vocab_size,
        context_length=model.context_length,
        parameter_count=None,
    )

    # Multimodal prompt_hash: sha256 over concatenated part content_hashes,
    # in part order (spec § Profile schema impact).
    prompt_hash = hashlib.sha256(
        "".join(p.content_hash for p in mm_input.parts).encode()
    ).hexdigest()

    input_parts: list[InputPartRecord] = []
    for p in mm_input.parts:
        if p.kind == "text":
            byte_len = len((p.text or "").encode())
        elif p.image_bytes is not None:
            byte_len = len(p.image_bytes)
        elif p.image_path is not None:
            import os
            byte_len = os.path.getsize(p.image_path)
        else:
            byte_len = None
        input_parts.append(
            InputPartRecord(
                kind=p.kind,
                content_hash=p.content_hash,
                width=p.width,
                height=p.height,
                byte_len=byte_len,
            )
        )

    prompt_record = PromptRecord(
        text=text_concat,
        regime=regime,
        token_count=len(prepared.input_ids),
        prompt_hash=prompt_hash,
        modality=mm_input.modality,
        input_parts=input_parts,
    )

    # Opt-in raw-trace capture (schema 0.7.0; see text path §12b). None by
    # default — disabled behavior is unchanged.
    raw_traces: RawTraces | None = None
    if config.traceability.enabled:
        raw_traces = RawTraces(
            variant_traces=raw_variant_traces,
            branch_traces=list(trajectory.branches),
        )

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
        exposure=hallucination_profile,
        semantic_field=semantic_field_reading,
        input_part_map=prepared.part_map,
        region_sensitivity=region_sensitivity,
        raw_traces=raw_traces,
    )


def _record_effective_embedder(config: RunConfig, embedder: EmbeddingModel) -> RunConfig:
    """Return a config whose embedding.model_name is the embedder that actually
    loaded (primary or fallback), so the artifact records the encoder behind
    the similarity/exposure values. No-op when they already match."""
    effective = getattr(embedder, "model_name", "") or ""
    if effective and effective != config.embedding.model_name:
        config = config.model_copy(deep=True)
        config.embedding.model_name = effective
    return config
