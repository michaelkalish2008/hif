"""Pydantic v2 configuration models for hif runs."""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen3-0.6B-Base"
    backend: str = "hf"  # "hf" | "tlens" | "ollama" | "openai" | "anthropic" | "gemini"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "float32"
    revision: Optional[str] = None  # pin a HF Hub commit SHA; None = float "main"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 120.0
    api_key: Optional[str] = None  # overrides env var for API backends
    base_url: Optional[str] = None  # custom endpoint for OpenAI-compatible APIs (Mistral, DeepSeek, etc.)
    temperature: Optional[float] = None  # per-model override; None = use backend default (0 for OpenAI, 1 for DeepSeek)
    # Extra JSON passed straight through on each OpenAI-compatible request.
    # Needed for provider options outside the OpenAI schema — DeepSeek's
    # `{"thinking": {"type": "disabled"}}` is the case this exists for. Left
    # on, that model spends the token budget reasoning: at max_new_tokens=64 it
    # returned 48 reasoning tokens and only 15 content steps, so the measured
    # generation would be a quarter the length of every other model's and the
    # comparison would be between different amounts of text. Disabled, it
    # returns 64 content steps with full top-5 logprobs.
    extra_body: Optional[dict] = None


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
    # Mass threshold for output_nucleus_entropy_bits, as a fraction (0.95 for
    # p95). None — the default — means the measurement is not taken at all, so
    # output_entropy_bits keeps the full-vocabulary basis every published
    # profile carries. Raising this does not retune nucleus_effective_support_size
    # or the charts; those are defined at a fixed 0.95 (see
    # metrics/distribution.py) and redefining them from here would change
    # three numbers to answer a question about one.
    entropy_percentile: Optional[float] = None


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

    # Researcher-authored variants, replacing the generator pipeline.
    #
    # A workload-format JSONL file (hif/batch.py — the same rows `hif batch`
    # profiles), whose rows carry a `variants` list:
    #
    #   {"query_id": "q1", "text": "<prompt>", "variants": ["<paraphrase>", ...]}
    #
    # One row format for all case data, on purpose. Rows whose `text` exactly
    # matches the run's prompt supply the variants; one file can therefore
    # serve a whole suite or batch (`hif batch` reads the SAME rows directly
    # and needs no pointer here). When variants apply, `generators` and
    # `n_variants` are ignored: the researcher's file IS the perturbation
    # set, every variant is a string a person wrote and can point to, and the
    # profile's perturbation entries carry generator="authored". A prompt
    # with no usable rows is a hard error, not an empty perturbation set —
    # silence here would report an un-perturbed run under a perturbed run's
    # config.
    #
    # This is the strongest form of perturbation control: the tool authors
    # nothing. The file is resolved at the CLI layer (hif/perturbation/
    # authored.py) — the builder is handed texts and does no file I/O.
    variants_file: Optional[Path] = None

    # Whether the variants are GENERATED from, or only teacher-forced over.
    #
    # The perturbation stage does two separable things that were previously
    # welded together: it authors paraphrased prompts and teacher-forces the
    # model over them (input-side, no new model output), and it generates a
    # continuation for each variant (output-side, new model output that did not
    # exist before). The input-side measurements never needed the second half —
    # `input_entropy_shift_bits` and `input_entropy_std_bits` difference
    # teacher-forced entropies and read no variant continuation at all.
    #
    # Set False to author and teacher-force the variants without generating
    # from them. The two input-side measurements survive; the two that read a
    # variant continuation (perturbation_jsd_bits and io_cosine_similarity),
    # along with the perturbation field, become absent. This is
    # the `acquisition = synthesized-input` ceiling; see the `acquisition` axis
    # in hif/profile/registry.py.
    elicit_variant_outputs: bool = True



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


class SemanticConfig(BaseModel):
    """Per-step semantic metrics — embedding and clustering each step's
    candidate cloud.

    On by default; this is the switch `--lite` throws. No measurement is
    derived from candidate geometry — `candidate_cluster_entropy_bits` is not
    in the set — so disabling it costs only the diagnostic blocks that read the
    cloud (cluster, exposure, semantic field) and leaves every published
    measurement on the entropy side untouched. It is the single most expensive per-step stage on a run
    with no perturbation variants, which is why it is separable at all."""
    enabled: bool = True


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
    field descriptors are derived, then dropped (see hif/metrics/field.py).
    Setting ``enabled=True`` keeps them: the profile artifact additionally
    captures those raw traces, so field descriptors, JS-centroids,
    translation, and branch fields can be recomputed from the artifact without
    re-running models. The cost is size — the variant traces scale with the
    variant count — not exposure; the baseline trace is in the artifact
    regardless.
    """
    enabled: bool = False


def public_config_dict(config: "RunConfig") -> dict:
    """The resolved config as a plain dict, safe to print or persist.

    One serialization for both consumers — `hif config show` and the record's
    `run_config` block — so what the researcher confirmed before the run is
    byte-identical in shape to what the record attests afterwards.

    Secrets are redacted, not omitted: a key that was set becomes
    "<redacted>", a key that was not stays None. The difference matters —
    "this run authenticated" is provenance; the credential itself must never
    reach a record that gets shared. Paths become strings (JSON/TOML have no
    Path type).
    """
    data = config.model_dump(mode="json")
    for table, key in (("model", "api_key"), ("perturbation", "llm_api_key")):
        if data.get(table, {}).get(key) is not None:
            data[table][key] = "<redacted>"
    return data


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
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    semantic_field: SemanticFieldConfig = Field(default_factory=SemanticFieldConfig)
    # Defaults (disabled), so RunConfig JSON embedded in pre-0.7.0 profiles
    # still validates unchanged.
    traceability: TraceabilityConfig = Field(default_factory=TraceabilityConfig)
