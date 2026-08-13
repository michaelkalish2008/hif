"""Pydantic v2 schema for a hif behavioral range profile artifact."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hif.config import RunConfig
from hif.hourglass.center import CenterDiagnostics
from hif.hourglass.input_side import InputSideAnalysis
from hif.hourglass.output_side import OutputSideTrace
from hif.hourglass.trajectory import Branch, TrajectoryAnalysis
from hif.metrics.distribution import DistributionMetrics
from hif.metrics.semantic import SemanticMetrics
from hif.metrics.sensitivity import SensitivityMetrics
from hif.metrics.field import PerturbationField
from hif.metrics.similarity import SimilarityMetrics
from hif.metrics.stability import StabilityMetrics
from hif.profile.provenance import RunProvenance

if TYPE_CHECKING:
    from hif.analysis.attention import TextAttentionAnalysis


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class ModelIdentity(BaseModel):
    name: str
    backend: str                   # "hf" | "tlens" | "ollama"
    vocab_size: int
    context_length: int
    parameter_count: Optional[int] = None  # None if unknown


class PromptRecord(BaseModel):
    text: str
    regime: str
    token_count: int
    prompt_hash: str               # sha256 of text

    @classmethod
    def from_text(cls, text: str, regime: str, token_count: int) -> PromptRecord:
        prompt_hash = hashlib.sha256(text.encode()).hexdigest()
        return cls(
            text=text,
            regime=regime,
            token_count=token_count,
            prompt_hash=prompt_hash,
        )


class MetricBundle(BaseModel):
    distribution: list[DistributionMetrics]   # one per output step
    semantic: list[SemanticMetrics]           # one per output step
    sensitivity: list[SensitivityMetrics]     # one per perturbation
    stability: StabilityMetrics
    similarity: SimilarityMetrics | None = None  # None when no perturbation variants exist
    # Derived perturbation-field geometry (centroid dispersion + radii + per-class
    # sub-fields). Derived scalars only — never a distribution
    # (docs/ARCHITECTURE.md § Field-model notes).
    # None when < 2 field members aligned (e.g. n_variants=0). Defaults to None so
    # profiles written before schema 0.4.0 still validate.
    field: PerturbationField | None = None


class PerturbationRecord(BaseModel):
    generator: str
    variants: list[str]
    sensitivity: list[SensitivityMetrics]     # one per variant


class VariantRawTrace(BaseModel):
    """One perturbation variant's raw output trace, labeled by its origin.

    ``trace`` reuses OutputSideTrace verbatim (per-step top-K {token_id, prob}
    records) — never a parallel representation. ``variant_index`` is the
    0-based position within the generator's variant list, so
    (generator, variant_index) uniquely keys the member.
    """

    generator: str
    variant_index: int
    trace: OutputSideTrace


class RawTraces(BaseModel):
    """Opt-in raw-trace capture (config.traceability.enabled ONLY).

    Compute-and-discard remains the default: this container exists so that
    field descriptors, JS-centroids, translation, and branch fields are
    retroactively recomputable from the artifact without re-running models.
    It must never be populated unless TraceabilityConfig.enabled was
    explicitly set — a run that did not ask to keep the raw traces should not
    be handed an artifact whose size says it did.
    """

    # One entry per successfully-traced perturbation variant, across all
    # generators (text path) / media families (mm path).
    variant_traces: list[VariantRawTrace] = Field(default_factory=list)
    # Raw trajectory-branch traces (one Branch per rollout branch, reusing the
    # trajectory model — per-step StepRecords included). Empty when the
    # trajectory stage was skipped (e.g. no teacher forcing).
    branch_traces: list[Branch] = Field(default_factory=list)


class Findings(BaseModel):
    """Non-inferential run provenance.

    This model deliberately carries no low/medium/high level, no pass/fail
    flag, and no verdict. Assigning a level is an inference that requires a
    null distribution this project never established, and the decision rule
    built on the previous levels measured a ~43% false-positive rate on pairs
    of runs known to be identical. What the run measured lives in
    hif.profile.measure.measurements(), in natural units; what it means is the
    reader's call.
    """

    # Ordinary-least-squares slope of the per-step mean pairwise cosine
    # similarity of the candidate cloud, across the generation. Signed and
    # unrounded: positive means the step's candidates grew more alike as
    # generation went on. Not thresholded.
    #
    # (It is not "input/output" similarity — that is `io_cosine_similarity`,
    # a different quantity. The series fitted here is
    # `SemanticMetrics.mean_pairwise_distance`, inverted; docs/MEASUREMENTS.md
    # § io_cosine_similarity has always described it correctly.)
    #
    # None when the run had fewer than two steps to fit a line through, which
    # includes generating nothing at all. The default was 0.0, so two gpt-5
    # profiles with `output_side.steps = []` published a flat trend for a
    # generation that never happened.
    similarity_trend_slope: Optional[float] = None
    # Set when the input-side measurements (input entropy shift, prompt
    # surprisal excess, I/O correlation) were computed by teacher-forcing a
    # --surrogate proxy model over the prompt instead of the target model
    # itself (the target cannot teacher-force, e.g. hosted APIs / Ollama).
    # None when they came from the target model directly.
    surrogate_model_name: str | None = None
    # Set when the target backend's OWN per-step distribution is degenerate
    # (only the selected token recorded, e.g. Anthropic — no logprobs at all)
    # and a --surrogate proxy was teacher-forced over prompt+continuation to
    # recover a real output entropy / step-delta reading instead of a trivial
    # 0.0 computed over a one-entry "distribution". Independent of
    # surrogate_model_name above.
    output_distribution_surrogate_name: str | None = None


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------


class BehavioralRangeProfile(BaseModel):
    # populate_by_name lets the builder construct by field name (exposure=,
    # attention_capture=) even though those fields carry validation_alias for
    # backward-compatible loading of pre-rename JSON.
    model_config = ConfigDict(populate_by_name=True)

    # Bump this whenever a field is added, removed, or its meaning changes —
    # profiles are validated strictly (Pydantic rejects unknown-shape data), so
    # an unbumped version number is not a reliable compatibility signal for
    # older profile JSON on disk or uploaded via `hif push`. (The hosted
    # platform's import endpoint keeps its own allow-list of supported schema
    # versions in the platform repo, so a bump here needs a matching entry
    # there before hosted imports of new artifacts succeed.)
    #
    # 0.2.0: added metrics.distribution[].nucleus_entropy_bits,
    #   findings.similarity_level, findings.similarity_trend. Profiles written
    #   under 0.1.0 do not have these fields and will fail validation.
    # 0.3.0: the image path — added prompt.modality (default "text"),
    #   prompt.input_parts (default []), input_part_map (default None),
    #   region_sensitivity (default None), perturbations[].traces (default []).
    #   All new fields default, so 0.2.0 profile JSON still validates
    #   unchanged.
    # 0.4.0: added metrics.field (PerturbationField, default None) —
    #   derived perturbation-field geometry for the drift-field model. Derived
    #   scalars only; no raw distributions. Defaults to None, so 0.3.0 profile
    #   JSON still validates unchanged.
    # 0.5.0: added trajectory.branch_field (BranchField, default None) —
    #   geometry of the sampling-perturbation (trajectory branch) cloud, the twin
    #   of the perturbation field. Derived scalars only. Defaults to None, so
    #   0.4.0 profile JSON still validates unchanged.
    # 0.6.0: added semantic_field (SemanticFieldReading, default None) —
    #   the within-generation semantic field instrument (Veer): per-step semantic-
    #   centroid displacement + field-spread change. Derived scalars only. Defaults
    #   to None, so 0.5.0 profile JSON still validates unchanged.
    # 0.7.0: added raw_traces (RawTraces, default None) and
    #   config.traceability (TraceabilityConfig, default disabled) — opt-in raw
    #   trace capture: per-perturbation-variant OutputSideTraces and per-branch
    #   trajectory traces, persisted ONLY when config.traceability.enabled so
    #   field descriptors/JS-centroids/translation/branch fields are
    #   retroactively recomputable. Both default (None / disabled), so 0.6.0
    #   profile JSON still validates unchanged and disabled-mode artifacts are
    #   unchanged apart from the two defaulted fields.
    # 0.8.0: added provenance (RunProvenance, default None) — which
    #   model actually filled each role in the run (teacher forcing, output
    #   distributions, attention analysis) plus the degradation flags. It turns
    #   every registry row's `subject` declaration into a claim the record path
    #   checks rather than repeats; see hif/profile/provenance.py. Defaults to
    #   None, so 0.7.0 profile JSON still validates unchanged — and a profile
    #   without it is simply unchecked, never assumed compliant.
    # 0.9.0: REMOVED input_side.volatility_score
    #   (mean_entropy / log2(vocab_size)) — dividing entropy by vocabulary size
    #   puts tokenizer metadata into a number labelled behaviour, and it
    #   saturates. Nothing in hif read it. Read input_side.mean_entropy (bits)
    #   with
    #   input_side.max_entropy as labelled context instead. Older profile JSON
    #   still carries the key and loads unchanged: pydantic ignores unknown
    #   fields on validation, so removal is read-compatible in the direction
    #   that matters (old artifact → new code).
    # 0.10.0: the exposure vocabulary replaces the last hallucination
    #   remnants in persisted keys. ExposureCandidate.hallucinated_token/-_prob
    #   → divergent_token/-_prob, ExposureProfile.high_risk_steps →
    #   exposed_steps, and RunConfig.hallucination → RunConfig.exposure
    #   (ExposureConfig). All four carry validation aliases accepting the old
    #   names, so archived profile JSON and old TOML config files load
    #   unchanged; newly written artifacts emit only the new names. The
    #   analysis was renamed because it measures the semantic distance of
    #   accessible alternatives — not hallucination (docs/MEASUREMENTS.md
    #   § Retired in hif-v4: "This is not a factuality
    #   judgment").
    #   The same release removed fields nothing ever populated or read:
    #   metrics.stability.temperature_robustness / prompt_order_robustness
    #   (documented-dead since docs/MEASUREMENTS.md first said so), and from
    #   the embedded config, output.save_plots / output.plot_format,
    #   attention.top_pairs, and traceability.profiles_dir. Old JSON carrying
    #   any of them still validates — unknown fields are ignored on load.
    # 0.11.0: REMOVED the image path. prompt.modality,
    #   prompt.input_parts, input_part_map, region_sensitivity and
    #   perturbations[].traces are gone with the VLM backends that populated
    #   them. No measurement ever read any of them — the image quantities were
    #   never argued through the Significance Gate and were never covered by
    #   SIGNAL_SET_VERSION, so records from that path were unversioned claims.
    #   A MAJOR removal for image profiles and a no-op for text profiles,
    #   which never carried the fields.
    # 0.12.0: added provenance.chat_template_present
    #   (Optional[bool], default None) — whether the target checkpoint's
    #   tokenizer declares a chat template, which hif does not apply on any
    #   backend. An instruct-tuned checkpoint therefore continues the prompt
    #   instead of answering it, and the record now carries the fact rather
    #   than leaving a reader to infer it from the output text. Defaults to
    #   None ("not asked", never "no template"), so 0.11.0 profile JSON still
    #   validates unchanged. See hif/models/chat_template.py for why the field
    #   is the literal declaration and not a test for "instruct-tuned".
    # 0.13.0 (current): the empty-generation pass — what a run reports when the
    #   target returned no output at all. Three fields become nullable and
    #   three are added, all for the same reason: a run with zero output steps
    #   was publishing measured-looking zeros.
    #   - output_side.mean_step_entropy: float -> Optional[float]. Was 0.0 over
    #     no steps, which reads as "certain at every step" about a run that
    #     took none.
    #   - metrics.sensitivity[].output_entropy_delta: float -> Optional[float],
    #     following mean_step_entropy — a difference of two absent means.
    #   - center.output_mean_entropy and center.entropy_ratio: float ->
    #     Optional[float], same reason.
    #   - center.prompt_output_cosine_distance: float -> Optional[float]. Was
    #     0.0 — the MINIMUM of its [0, 2] range, which on a distance reads as
    #     "output identical to the prompt". It reported perfect anchoring for
    #     a model that returned nothing.
    #   - findings.similarity_trend_slope: float (default 0.0) ->
    #     Optional[float] (default None). A line fitted through fewer than two
    #     points is undefined, not flat.
    #   - output_side.stop_reason (Optional[str], default None) — why
    #     generation ended, as the backend reported it.
    #   - provenance.target_generated_no_output (bool, default False) and
    #     provenance.generation_stop_reason (Optional[str], default None) —
    #     so an empty output side is a stated fact with a stated reason rather
    #     than an absence a reader has to notice.
    #   A MAJOR change: 0.12.0 JSON still validates (widening a type and adding
    #   defaulted fields both accept old data), but a consumer that read
    #   `mean_step_entropy` or `similarity_trend_slope` as a float will now
    #   meet None on exactly the runs where the old value was fiction.
    # 0.14.0 (current): the same pass applied one layer down, to the
    #   per-variant sensitivity records the perturbation response is built
    #   from. `stability.py` promised "Never a fake 0.0 and never a fake 1.0"
    #   and kept that promise at its own level while being handed constants
    #   from below.
    #   - metrics.sensitivity[].{mean_js_divergence, mean_kl_divergence,
    #     mean_entropy_delta, mean_nucleus_stability_p90}: float -> Optional,
    #     None when the variant aligned no steps against the baseline. Were
    #     0.0/0.0/0.0/1.0.
    #   - metrics.sensitivity[].step_sensitivities[].kl_divergence: float ->
    #     Optional. Was clamped to a 1e9 sentinel, which `math.isfinite`
    #     accepts — so the filter written to drop undefined steps dropped
    #     none, and 833 records across half the corpus carry a mean near
    #     9.65e8 that the report rendered as 965517241.3793.
    #   - metrics.sensitivity[].step_sensitivities[].nucleus_overlap_p90:
    #     float -> Optional. Was 1.0 on an empty nucleus.
    #   - metrics.sensitivity[].n_steps_aligned (int, default 0) — the
    #     shared-prefix length each row's means were taken over. Variants are
    #     weighted equally in the aggregate regardless of it, which is the
    #     measurement's own definition; this makes the difference visible.
    #   - metrics.sensitivity[].n_undefined_kl_steps (int, default 0).
    #   - metrics.stability.n_perturbations_aligned (int, default 0) — how
    #     many variants actually contributed to perturbation_jsd_bits, so
    #     dropping an absent one is not a silent narrowing of n.
    #   A MAJOR change on the same terms as 0.13.0: old JSON still validates,
    #   but a consumer reading any of these as a float now meets None where
    #   the old value was invented.
    schema_version: str = "0.14.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: ModelIdentity
    prompt: PromptRecord
    input_side: InputSideAnalysis
    output_side: OutputSideTrace
    center: CenterDiagnostics
    trajectory: TrajectoryAnalysis
    perturbations: list[PerturbationRecord]
    metrics: MetricBundle
    plots: dict[str, str] = Field(default_factory=dict)   # relative paths as strings
    findings: Findings
    config: RunConfig
    notes: str = ""
    # Optional analysis extensions — typed as Any because their heavy dependencies
    # (transformers, torch) are loaded lazily and cannot appear in the module namespace
    # at class-creation time.
    #
    # Field names track the HIF taxonomy, not implementation concepts: neither
    # "attention" nor "hallucination" is a signal. `attention_capture` is the raw
    # attention substrate the output-side and input-side attention-row readings derive from;
    # `exposure` is the counterfactual exposure instrument (runtime type ExposureProfile, from
    # hif.analysis.exposure — both renamed from the historical "hallucination").
    # validation_alias keeps older profile JSON (with the pre-rename keys) loadable.
    #
    # Runtime type is TextAttentionAnalysis | None.
    attention_capture: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("attention_capture", "attention"),
    )
    # Runtime type is ExposureProfile | None.
    exposure: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("exposure", "hallucination"),
    )
    # Within-generation semantic field instrument (Veer). Runtime type is
    # SemanticFieldReading | None (hif.analysis.semantic_field). Derived scalars
    # only; None unless config.semantic_field.enabled. Lazy-typed like `exposure`.
    semantic_field: Optional[Any] = None
    # Opt-in raw trace capture (0.7.0). None unless config.traceability.enabled
    # — the sanctioned exception to compute-and-discard; see RawTraces docstring.
    raw_traces: Optional[RawTraces] = None
    # Which model filled each role in this run, recorded as the pipeline ran
    # (0.8.0). The evidence behind every measurement's declared subject; the
    # record path refuses to emit a record whose declarations contradict it.
    provenance: Optional[RunProvenance] = None
