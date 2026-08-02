"""Pydantic v2 configuration models for HI runs."""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    name: str = "gpt2"
    backend: str = "hf"  # "hf" | "tlens" | "ollama" | "openai" | "anthropic" | "gemini"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "float32"
    revision: Optional[str] = None  # pin a HF Hub commit SHA; None = float "main"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 120.0
    api_key: Optional[str] = None  # overrides env var for API backends
    base_url: Optional[str] = None  # custom endpoint for OpenAI-compatible APIs (Mistral, DeepSeek, etc.)
    temperature: Optional[float] = None  # per-model override; None = use backend default (0 for OpenAI, 1 for DeepSeek)


class EmbeddingConfig(BaseModel):
    # MiniLM is the primary default. The old default ("google/embedding-gemma-300m")
    # was a wrong repo id (404s always; the real id is "google/embeddinggemma-300m"),
    # so every artifact ever produced actually used the MiniLM fallback — making
    # MiniLM the configured primary is a zero-behavior-change correction.
    # "google/embeddinggemma-300m" remains available as an opt-in override
    # (set matryoshka_dim=256 with it); note that switching encoders changes
    # similarity/exposure comparability — the effective embedder is recorded in
    # each profile's config.embedding.model_name.
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Matryoshka truncation dim (e.g. 256 for EmbeddingGemma). None = model's
    # native dimension (384 for MiniLM) — matches what all prior runs produced.
    matryoshka_dim: Optional[int] = None
    cache_dir: Path = Path.home() / ".cache" / "hif" / "embeddings"


class ClusterConfig(BaseModel):
    method: str = "hdbscan"
    min_cluster_size: int = 2
    min_samples: int = 1


class GenerationConfig(BaseModel):
    max_new_tokens: int = 64
    top_k: int = 50
    temperature: float = 1.0
    seed: int = 42


class TrajectoryConfig(BaseModel):
    n_branches: int = 5   # B in the spec
    rollout_steps: int = 10  # R in the spec


class PerturbationConfig(BaseModel):
    n_variants: int = 2  # variants per generator (3 generators × 2 = 6 total)
    generators: list[str] = ["synonym", "tone", "reorder"]

    # Rule-based generators (zero compute cost) are the default for every
    # generator name above. Set use_llm_perturbation=True with an explicit
    # llm_base_url/llm_api_key to opt into LLM-backed paraphrasing instead,
    # via any OpenAI-compatible endpoint (bring your own key/endpoint, e.g.
    # a local Ollama server). llm_model is optional — omit to use the
    # generator's own default.
    use_llm_perturbation: bool = False
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None

    # Media-side perturbation families (multimodal profiles only; ignored on
    # the text path). Separate namespace from `generators` — resolved via
    # hif.perturbation.get_family(). Default makes multimodal input Just
    # Work with grid masking. n_variants <= 0 means exhaustive cell sweep
    # (audit mode).
    media_families: list[str] = ["image_grid_mask"]
    image_grid_rows: int = 4
    image_grid_cols: int = 4


class OutputConfig(BaseModel):
    output_dir: Path = Path("outputs")


class ExposureConfig(BaseModel):
    """Counterfactual exposure analysis (hif.analysis.exposure).

    Renamed from HallucinationConfig — the analysis measures the semantic
    distance of accessible alternatives, not hallucination; see the module
    docstring of hif/analysis/exposure.py. RunConfig accepts the old
    `hallucination` key on read so archived profile JSON and old TOML config
    files keep loading."""

    enabled: bool = True
    min_prob: float = 0.01          # minimum candidate probability to consider
    distance_threshold: float = 0.3  # cosine distance at/above which a diffusion-zone step counts as exposed


class SemanticFieldConfig(BaseModel):
    """Within-generation semantic field instrument (Veer). Off by default — it
    re-embeds each step's candidate cloud to trace the semantic centroid; enable
    it where the per-step semantic trajectory is wanted (e.g. the study)."""
    enabled: bool = False
    context_window: int = 5   # generated tokens of left-context prepended to each candidate


class AttentionConfig(BaseModel):
    enabled: bool = False
    model_name: str = "distilbert-base-uncased"
    aggregate_method: str = "mean_all_layers"  # "mean_all_layers" | "last_layer" | "mean_upper_half"
    max_seq_length: int = 512  # DistilBERT limit
    trajectory_interval: int = 4  # checkpoint every N continuation tokens


class TraceabilityConfig(BaseModel):
    """Opt-in raw-trace capture for retroactive recomputation.

    Compute-and-discard is the DEFAULT: perturbation-variant output traces and
    trajectory-branch traces are held only transiently while sensitivity and
    field descriptors are derived, then dropped (see hif/metrics/field.py's
    privacy invariant). Setting ``enabled=True`` is the sanctioned exception:
    the profile artifact additionally captures those raw traces (per-step top-K
    with token identity — reconstructable content) so field descriptors,
    JS-centroids, translation, and branch fields can be recomputed from the
    artifact without re-running models. Only enable where the artifact's
    storage location is trusted with prompt/continuation-level content.
    """
    enabled: bool = False


class RunConfig(BaseModel):
    # populate_by_name lets callers construct by field name (exposure=) while
    # the validation alias keeps the pre-rename `hallucination` key loadable
    # from archived profile JSON and old TOML config files.
    model_config = ConfigDict(populate_by_name=True)

    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    perturbation: PerturbationConfig = Field(default_factory=PerturbationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    exposure: ExposureConfig = Field(
        default_factory=ExposureConfig,
        validation_alias=AliasChoices("exposure", "hallucination"),
    )
    semantic_field: SemanticFieldConfig = Field(default_factory=SemanticFieldConfig)
    # Defaults (disabled), so RunConfig JSON embedded in pre-0.7.0 profiles
    # still validates unchanged.
    traceability: TraceabilityConfig = Field(default_factory=TraceabilityConfig)
