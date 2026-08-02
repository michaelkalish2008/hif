"""Pydantic v2 schema for a HI behavioral range profile artifact."""

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
from hif.models.mm import InputPartMap
from hif.perturbation.base import PerturbationTrace
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


class InputPartRecord(BaseModel):
    """Persisted record of one multimodal input part: hash + dims ONLY.

    Never pixels/base64 — raw media must not reach the profile JSON or any
    API payload (§ Storage & privacy and Risk rule 2, docs/ARCHITECTURE.md
    § Multimodal notes; the hosted platform enforces the same guard on its
    side at import).
    """

    kind: str                      # "text" | "image" (M2/M3 add more)
    content_hash: str              # sha256 of text-utf8 or media bytes
    width: Optional[int] = None
    height: Optional[int] = None
    byte_len: Optional[int] = None


class PromptRecord(BaseModel):
    text: str                      # multimodal: MultimodalInput.text_concat
    regime: str
    token_count: int
    # sha256 of text; for multimodal profiles: sha256 over the concatenated
    # part content_hashes (in part order) — see builder._build_profile_mm.
    prompt_hash: str
    modality: str = "text"         # "text" | "image+text" (closed enum, M1)
    input_parts: list[InputPartRecord] = Field(default_factory=list)

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
    # Media-family traces (one per variant when a PerturbationFamily produced
    # the variants; empty for text generators). Geometry + params only — never
    # pixels. Default keeps existing text profiles valid under schema 0.3.0.
    traces: list[PerturbationTrace] = Field(default_factory=list)


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
    A top-k distribution with token identity is reconstructable content —
    this block must never be populated unless TraceabilityConfig.enabled was
    explicitly set (see hif/metrics/field.py's privacy invariant).
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
    hif.profile.signals.measurements(), in natural units; what it means is the
    reader's call.
    """

    # Ordinary-least-squares slope of per-step input/output cosine similarity
    # across the generation. Signed and unrounded: positive means the output
    # grew more similar to the input as it went on. Not thresholded.
    similarity_trend_slope: float = 0.0
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
    # 0.3.0: multimodal M1 — added prompt.modality (default "text"),
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
    # 0.9.0 (current): REMOVED input_side.volatility_score
    #   (mean_entropy / log2(vocab_size)) — the normaliser the measurement set
    #   banned, still computed and persisted under a *_score name. Nothing in
    #   hif read it. read input_side.mean_entropy (bits) with
    #   input_side.max_entropy as labelled context instead. Older profile JSON
    #   still carries the key and loads unchanged: pydantic ignores unknown
    #   fields on validation, so removal is read-compatible in the direction
    #   that matters (old artifact → new code).
    # 0.10.0 (current): the exposure vocabulary replaces the last hallucination
    #   remnants in persisted keys. ExposureCandidate.hallucinated_token/-_prob
    #   → divergent_token/-_prob, ExposureProfile.high_risk_steps →
    #   exposed_steps, and RunConfig.hallucination → RunConfig.exposure
    #   (ExposureConfig). All four carry validation aliases accepting the old
    #   names, so archived profile JSON and old TOML config files load
    #   unchanged; newly written artifacts emit only the new names. The
    #   analysis was renamed because it measures the semantic distance of
    #   accessible alternatives — not hallucination (docs/MEASUREMENTS.md
    #   § counterfactual_exposure_fraction: "This is not a factuality
    #   judgment").
    #   The same release removed fields nothing ever populated or read:
    #   metrics.stability.temperature_robustness / prompt_order_robustness
    #   (documented-dead since docs/MEASUREMENTS.md first said so), and from
    #   the embedded config, output.save_plots / output.plot_format,
    #   attention.top_pairs, and traceability.profiles_dir. Old JSON carrying
    #   any of them still validates — unknown fields are ignored on load.
    schema_version: str = "0.10.0"
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
    # attention substrate the Spread ■ and Horizon ▼ readings derive from;
    # `exposure` is the Exposure ◇ instrument (runtime type ExposureProfile, from
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
    # Multimodal (0.3.0): position → part/patch-grid geometry for the prepared
    # sequence (geometry only, never pixels). None for text-only profiles.
    input_part_map: Optional[InputPartMap] = None
    # Runtime type is RegionSensitivityResult | None (perturbation-JSD per
    # grid cell; M1 session 2+). Lazy-typed like `attention`.
    region_sensitivity: Optional[Any] = None
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
