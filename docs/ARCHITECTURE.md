# Architecture — Horizonal Interpretability Framework (HIF)

Implementation reference for contributors and researchers. Covers the model roles, hermeneutic attention as a module, the module map, backend capabilities, and the end-to-end data flow.

This document is an implementation reference only. It describes what the system computes and how; it does not interpret those measurements or draw conclusions from them.

---

## Model Roles

A HIF run involves up to four model roles. Conflating them is the most common conceptual error.

**1. Model under analysis**

The causal language model whose behavioral range is being characterized. hif extracts only what the model exposes externally: tokenized text, full-vocabulary logits (for HF/TLens), and top-K logprobs at generation time.

Backends are constructed by `hif/models/factory.py` from a `ModelConfig`; `hif/models/capabilities.py` is the single source of truth for what each one exposes (and backs `hif models`, `hif doctor`, and the early `--metric` guard in `hif profile`):

| Backend | Class | Kind | Teacher forcing | Attention | Logprobs |
|---------|-------|------|-----------------|-----------|----------|
| `hf` | `HFModel` | local-open | Yes | Yes | full |
| `tlens` | `TLensModel` | local-open | Yes | Yes | full |
| `ollama` | `OllamaModel` | local-service | No | No | top-k |
| `openai` | `OpenAIModel` | hosted-api | No | No | top-k |
| `gemini` | `GeminiModel` | hosted-api | No | No | top-k |
| `anthropic` | `AnthropicModel` | hosted-api | No | No | selected-only |

Example model ids per backend live in `BACKENDS` in `hif/models/capabilities.py` rather than here, so `hif models` and this document cannot drift apart.

**2. Embedding model**

A sentence-transformer used internally to embed candidate token strings and generated text for semantic clustering. Default and fallback: `sentence-transformers/all-MiniLM-L6-v2` at its native 384 dimensions. `google/embeddinggemma-300m` is available as an opt-in override (pair it with `matryoshka_dim=256`).

The embedding model is measurement infrastructure. It does not analyze the model under analysis — it helps group outputs by meaning. Its quality affects semantic clustering granularity but does not affect distribution metrics, sensitivity metrics, or raw top-K data. Switching encoders changes what `io_cosine_similarity` means — and what the semantic-field and exposure diagnostic blocks mean — so the *effective* embedder — the one that actually loaded — is written back into `config.embedding.model_name` on every profile.

**3. Analysis transformer**

A bidirectional encoder (DistilBERT by default, configurable via `AttentionConfig.model_name`) used as a text-analysis instrument in `hif/analysis/attention.py`. Present only when `AttentionConfig.enabled=True` (which `hif profile --diagnostics` sets).

Applied to two observable texts in independent readings: the input prompt and the generated continuation. No concatenation, no joint forward pass. **Does not access the generating model's internal attention weights.** Attention is aggregated across heads and layers (`aggregate_method`, default `mean_all_layers`) before it is stored.

**4. Teacher-forcing surrogate (`--surrogate`)**

An open-weight HF causal LM — default `unsloth/Llama-3.2-1B`, see `SURROGATE_CANDIDATES` in `hif/models/capabilities.py` — loaded alongside the target model when the target cannot teacher-force. It serves two independent purposes in `build_profile()`:

- **Input-side recovery.** Teacher-forced over the prompt to produce `InputSideAnalysis` when `model.supports_teacher_forcing` is False. Recorded as `findings.surrogate_model_name`.
- **Output-distribution recovery.** Teacher-forced over prompt + continuation when the target backend's own per-step distribution is degenerate (Anthropic returns the selected token only). Recorded as `findings.output_distribution_surrogate_name`.

The two are independent, are reported separately, and — crucially — do **not** have the same standing. Output-distribution recovery teacher-forces the surrogate over text the target actually generated: a reading instrument on the target's real output, whose value still moves when the target's output moves. Input-side recovery teacher-forces the surrogate over the *prompt*, which the target never touched: nothing the target did enters the result.

That difference is declared per registry row as the measurement's **subject** (`hif/profile/registry.py`), and enforced in the record. Output-recovered quantities stay in `measurements` with subject `target-output-text` and are starred in the CLI table. Input-recovered quantities have subject `prompt-only` on that backend and leave `measurements` entirely for a top-level `prompt_measurements` block naming the reference model — a flag would say "a caveated number about this model", and only "this model produced no number" is true. A `mixed` subject exists in the vocabulary for a quantity that couples the surrogate's prompt reading with the target's own output response; hif-v4 has no such row (`io_correlation_r` was the only one, and it was cut). `signals_record()` still emits both surrogate names under `surrogate`. See docs/MEASUREMENTS.md § Subject.

---

## Measurements and Formulas

A run reports a set of scalar **measurements**, each in its natural unit (bits, cosine distance, Pearson *r*, a fraction). They are defined once, in `MEASUREMENT_REGISTRY` in `hif/profile/registry.py` — run `hif schema` for the current set — and derived from the low-level distribution, semantic, sensitivity, and perturbation-response metrics computed at each generation step. `hif schema` prints every registry row in full; `signals_record()` emits the values under `measurements`, with the matching unit strings under `units` on request. `measurements` carries measurements of the model named in the record and nothing else — quantities whose subject on the active backend is `prompt-only` go under `prompt_measurements` instead (§ Teacher-forcing surrogate above).

Absent measurements are omitted from the record, never pinned to a default: a backend that cannot teacher-force produces no `input_entropy_shift_bits`, and that is a different statement from a measured zero.

Nothing is normalised into `[0, 1]`, inverted into a score, or bucketed into a level. Mathematical definitions, formulas, and ranges: [docs/MEASUREMENTS.md](MEASUREMENTS.md).

---

## Signal Visualizations

`hif/viz/registry.py` is the single source of truth for the chart set: one visualization per signal, ordered aggregates-then-readings. Each entry supplies a generator and an *availability predicate*, so a signal whose backing data is missing renders an explicit "requires teacher forcing / attention capture / …" placeholder rather than a flat or zero chart.

**Aggregates** (`kind="aggregate"`) — by registry `id`, with the chart label and the measurement each joins to:

| Registry `id` | Chart label | `measurement_key` |
|---|---|---|
| `stability` | Input entropy trace | `input_entropy_std_bits` *(resolves only)* |
| `breadth` | Effective support size | — (chart only) |
| `surprise` | Prompt surprisal excess (trace) | — (chart only) |
| `sensitivity` | Perturbation JSD (bits) | `perturbation_jsd_bits` |
| `similarity` | Input/output cosine similarity | `io_cosine_similarity` |

Note `stability` → `input_entropy_std_bits`: the id is a leftover shorthand and the measurement is a standard deviation, where a *higher* value means *less* stable. The label and the key say so; the id does not. This is the mismatch that got the shorthand vocabulary retired — see below.

*(resolves only)* marks the one chart that carries a `measurement_key` without drawing that measurement's own series: `stability` plots the per-position entropy of the original prompt, while `input_entropy_std_bits` is the spread of per-*variant* entropy shifts. The key is carried so `--metric input_entropy_std_bits --charts` finds a chart. Every other keyed chart draws its measurement's basis, and is therefore required to decline exactly the runs the measurement declines — enforced in `hif/viz/registry.py` and pinned by `tests/unit/test_chart_measurement_gate.py`.

**Readings** (`kind="reading"`) — the per-token/per-step traces:

Each row's `id` is an internal, stable identifier; the `label` is what a chart
is titled with, and it names the quantity in the terms the quantity is computed
in. Read the `measurement_key` column as the join to `hif schema`.

| Registry `id` | Chart label | `measurement_key` | Source signal |
|---|---|---|---|
| `entropy` | Output entropy (bits) | `output_entropy_bits` | Per-output-step Shannon entropy, in bits (nucleus and raw top-K both drawn) |
| `wager` | Prompt surprisal excess (bits) | `prompt_surprisal_excess_bits` | Per-prompt-position surprisal excess over entropy — where the model committed and the actual token overrode that commitment |

Six charts were removed in hif-v4 with the measurements they drew (`continuity`, `io_correlation`, `shift`, `spread`, `horizon`, `exposure`). A chart whose measurement is gone recreates the "existed only as a chart" gap hif-v3.1 was created to close, so the charts went with the rows. See `docs/MEASUREMENTS.md` § *Retired in hif-v4* for what each measured and what to read instead.

**There is no glyph column, and no glyph.** Charts were once headed with one —
● ◆ ▲ ■ ◇ — and a symbol set does not extend: adding a measurement means either
picking a mark nobody has taken or subscripting one that is (this project
reached ▼p and ▼g), and ◆ against ◈ was two near-identical marks on two
quantities a reader is specifically likely to confuse. A colour swatch indexes a
legend without pretending to be a mnemonic; a name says what the thing is. See
`hif/viz/base.py::signal_title`.

The same applies to the names. Registry rows used to carry a shorthand from this
project's own vocabulary — Entropy, Shift, Wager, Spread, Horizon, Exposure,
Veer, Stability — beside the descriptive name. That was removed in `hif-v3.3`:
two names for one quantity is one name too many, and the shorthand was the one
that went wrong, with "Stability" landing on `input_entropy_std_bits`, a
standard deviation where a *higher* number means *less* stable. The shorthands
survive only as `id`s, which are internal and never displayed.

Attention rows are captured and persisted under `attention_capture`, but hif-v4 publishes no measurement from them and draws no chart: the two rows that read them were cut, and the module that traced them went with the charts. The block ships under `--diagnostics` as evidence. Its historical `log₂(seq_len)` normaliser is gone either way — the denominator is sequence length, so it leaked position metadata into a number presented as behaviour.

The per-step semantic-centroid displacement from `hif/analysis/semantic_field.py` was a measurement (`semantic_centroid_veer_cosine`) through hif-v3 and is now a diagnostic block only. It remains a persisted per-step trace (`profile.semantic_field`), but has no entry in the viz registry, so it has no chart label.

### Hermeneutic Attention (DistilBERT)

`hif/analysis/attention.py` applies a bidirectional encoder (DistilBERT) as a reading instrument to input and output texts independently, then compares their structural resonance.

Four readings:
1. **Input reading** — the prompt analyzed on its own terms, plus per-token importance deltas under up to five perturbed variants
2. **Output reading** — the generated continuation analyzed independently
3. **Resonance comparison** — which continuation tokens echo the load-bearing structure of the input
4. **Joint trajectory trace** — DistilBERT run on `[prompt + continuation[:k]]` at `trajectory_interval` checkpoints, tracking which prompt tokens hold or release cross-attention as the continuation grows

No concatenation in readings 1–3, no joint forward pass, and the generation process is never observed. The stored result (`BehavioralRangeProfile.attention_capture`, a `TextAttentionAnalysis`) once backed the `spread` and `horizon` readings; hif-v4 cut both, precisely because they described an independent reader's attention over the texts rather than the generating model's own. The block itself still ships under `--diagnostics`.

---

## Module Architecture

```
hif/
  __init__.py              # version
  config.py                # Pydantic v2 RunConfig and sub-configs
  cli/                     # the command surface. ONE MODULE PER COMMAND,
                           #   named for the command; shared infrastructure
                           #   prefixed with `_`. That is the whole rule.
    __init__.py            #   builds `app`, registers commands by importing
                           #   them, and is the `hif.cli:app` entry point
    __main__.py            #   `python -m hif.cli`
    profile.py             #   `hif profile`
    models.py              #   `hif models`
    doctor.py              #   `hif doctor`
    compare.py             #   `hif compare`
    render.py              #   `hif render`
    schema.py              #   `hif schema`
    batch.py               #   `hif batch`
    config.py              #   `hif config show|init`
    _app.py                #   the Typer app, consoles, .env discovery
    _run.py                #   the shared pipeline call (profile and batch)
    _load.py               #   model / embedder / surrogate loading
    _config.py             #   RunConfig assembly from flags and files
    _output.py             #   terminal rendering helpers
    _compat.py             #   signal-set family checks across versions
  engine.py                # SessionEngine — load model/embedder/surrogate once, profile many
  batch.py                 # `hif batch` workload runner; streams one JSON record per row

  models/
    base.py                # Abstract Model interface; Logits, TopKEntry, StepRecord, GenerationResult
    factory.py             # load_model() — backend → class dispatch, KNOWN_BACKENDS
    capabilities.py        # BACKENDS table, metric_support(), signals_available(), SURROGATE_CANDIDATES
    hf.py                  # HFModel (AutoModelForCausalLM)
    tlens.py               # TLensModel (TransformerLens)
    openai_model.py        # OpenAIModel (and OpenAI-compatible endpoints)
    anthropic_model.py     # AnthropicModel (selected-token-only logprobs)
    gemini_model.py        # GeminiModel (top-k logprobs on Vertex AI)
    ollama.py              # OllamaModel (local Ollama server)

  hourglass/
    input_side.py          # analyze_input_side() → InputSideAnalysis; mean_surprisal_excess()
    output_side.py         # collect_output_trace() → OutputSideTrace
    center.py              # compute_center_diagnostics() → CenterDiagnostics
    trajectory.py          # analyze_trajectory() → TrajectoryAnalysis; BranchField

  metrics/
    distribution.py        # compute_distribution_metrics() → DistributionMetrics
    semantic.py            # compute_semantic_metrics() → SemanticMetrics
    sensitivity.py         # compute_sensitivity_metrics() → SensitivityMetrics;
                           #   js_centroid(), generalized_js_divergence()
    stability.py           # compute_stability_metrics() → PerturbationResponse
    similarity.py          # compute_similarity_metrics() → SimilarityMetrics
    field.py               # compute_perturbation_field() → PerturbationField;
                           #   compute_field_deformation() → FieldDeformation

  perturbation/
    base.py                # PerturbationGenerator / PerturbationFamily, PerturbationResult, PerturbationTrace
    synonym.py             # SynonymGenerator (WordNet, NLTK POS tagging)
    substitution.py        # SubstitutionGenerator (word-level replacements)
    word_order.py          # WordOrderGenerator (clause reordering)
    ambiguity.py           # AmbiguityGenerator (hedging and qualification variants)
    tone.py                # ToneGenerator (formal, casual, direct, hedged)
    llm.py                 # LLMParaphraseGenerator (opt-in; explicit base_url/api_key)
    __init__.py            # get_generator() / get_family() registries

  clustering/
    embed.py               # EmbeddingModel (sentence-transformers wrapper)
    cluster.py             # cluster_embeddings() → ClusterResult (HDBSCAN / KMeans)

  profile/
    schema.py              # BehavioralRangeProfile and all sub-schemas; Findings
    builder.py             # build_profile() — orchestrates the full pipeline; generate_findings()
    signals.py             # MEASUREMENT_REGISTRY (triple + subject), measurements(),
                           # prompt_measurements(), signals_record()
    render_json.py         # render_json() → profile.json
    render_markdown.py     # render_technical(), render_public() → Markdown

  prompts/
    regimes.py             # REGIMES list, Regime dataclass, the regime definitions
    suite.py               # get_regime(), suite accessors

  archetypes/
    __init__.py            # flat-YAML archetype registry (--application)
    *.yaml                 # rag-qa, summarization, classification, extraction,
                           #   coding-assistant, support-chatbot, agent-tool-use,
                           #   document-understanding
    suites/                # bundled prompt suites (JSONL)

  viz/
    __init__.py            # generate_signal_plots() entry point
    registry.py            # SIGNALS — id/label/kind/family + generate/available, gated on the measurement
    base.py                # na_figure(), save_fig(), signal_title(), NEEDS_* reasons
    index.py               # build_index() — combined signal dashboard page
    _theme.py              # shared dark-theme colors and layout helpers
    signals/               # one module per chart; hif-v4 removed six with their rows
      stability.py         # input entropy trace
      breadth.py           # per-step effective support size
      surprise.py          # per-position excess surprisal
      sensitivity.py       # per-generator JSD bars
      similarity.py        # input/output/io cosine bars
      entropy.py           # per-step output entropy
      wager.py             # surprisal vs. entropy, two-panel

  analysis/
    __init__.py            # module docstring explaining the text-instrument role
    attention.py           # AttentionAnalyzer, TextAttentionAnalysis, and Pydantic schemas
    exposure.py            # ExposureAnalyzer → ExposureProfile (per-step counterfactual exposure)
    semantic_field.py      # SemanticFieldAnalyzer → SemanticFieldReading


  utils/
    logging.py             # get_logger()
    seeding.py             # seed_everything()
    io.py                  # File I/O helpers
```

---

## Backend Notes

| Backend | Extra | Teacher forcing | Notes |
|---------|-------|-----------------|-------|
| `hf` | (base) | Yes | Primary. Full-vocabulary logits and attention — every measurement |
| `tlens` | `[tlens]` | Yes | Replicability with MI literature; same interface as HF |
| `ollama` | `[ollama]` | No | Local Ollama server. Output-side only (top-20). Model must be pulled first |
| `openai` | `[openai]` | No | Output-side only (top-20 logprobs) |
| `gemini` | `[gemini]` | No | Top-20 logprobs on Vertex AI only; the developer API degenerates |
| `anthropic` | `[anthropic]` | No | Selected token only — distribution measurements degenerate without `--surrogate` |

The `notes`, `setup`, and `deps` strings shown by `hif doctor` and `hif models` come from the same `BACKENDS` table, so this summary and the CLI cannot disagree about capability.

Device handling for HFModel: if `device="auto"`, the backend checks for CUDA, then MPS (Apple Silicon), then falls back to CPU.

---

## Data Flow

The full pipeline, as orchestrated by `build_profile()` in `hif/profile/builder.py`:

1. **Seed** — `seed_everything(seed)` sets Python, NumPy, and PyTorch seeds for reproducibility.
2. **Input-side analysis** — teacher-forced forward pass, per-position surprisal and entropy, top-K alternatives → `InputSideAnalysis`. Falls back to the `--surrogate` model when the target cannot teacher-force; with neither, an empty `InputSideAnalysis` is constructed and the input-side measurements are simply absent.
3. **Output trace** — generates text while recording `StepRecord` at each step → `OutputSideTrace`.
4. **Center diagnostics** — input/output mean entropy, entropy ratio, `prompt_output_cosine_distance` → `CenterDiagnostics`. There is no equilibrium classification: it thresholded output entropy against `0.1`/`0.9 × log₂(vocab_size)`, i.e. bucketed behaviour by a property of the tokenizer.
5. **Trajectory analysis** — branches from the end of the generated context, B×R generation steps, convergence profile and `BranchField` → `TrajectoryAnalysis`. Skipped (zero branches) for models without teacher forcing.
6. **Perturbation analysis** — for each configured generator (default `["synonym", "tone", "reorder"]`, `n_variants=2` each; `substitution` and `ambiguity` are available but off by default, and LLM-backed paraphrasing is opt-in with an explicit endpoint), generates prompt variants, runs output traces, computes `SensitivityMetrics` against baseline. Variant traces are held transiently and discarded unless `config.traceability.enabled`.
   - **6b.** When the backend's own distribution is degenerate and a surrogate is loaded, per-step alternatives are recovered by teacher-forcing the surrogate over prompt + continuation.
7. **Distribution metrics** — once per output step, over the top-K distribution → list of `DistributionMetrics` (raw entropy, nucleus entropy, ESS, logit margin, tail weight, nucleus fractions).
8. **Semantic metrics** — once per output step, embedding top-K candidates and clustering → list of `SemanticMetrics`.
9. **Perturbation response** — aggregates across all perturbation variants → `PerturbationResponse` (`input_entropy_shift_bits`, `perturbation_jsd_bits`, `input_output_correlation`; each `None` when its evidence does not exist).
   - **9a.** `compute_perturbation_field()` derives the geometry of the {baseline + variants} cloud around its Jensen-Shannon centroid → `PerturbationField`. `None` with fewer than two aligned members.
   - **9b.** `compute_similarity_metrics()` over the (input, output) text pairs → `SimilarityMetrics`. Skipped when there are no variants.
10. **MetricBundle assembly** — distribution, semantic, sensitivity, stability, similarity, field.
11. **Findings** — `generate_findings()` collects run provenance only: `similarity_trend_slope` plus the two surrogate model names. No levels, no verdict, no summary sentence.
    - **11b.** Exposure analysis (`config.exposure.enabled`, on by default) → `ExposureProfile`.
    - **11c.** Attention analysis (`config.attention.enabled`, off by default; set by `--diagnostics`) → `TextAttentionAnalysis`.
    - **11d.** Within-generation semantic field (`config.semantic_field.enabled`, off by default; set by `--diagnostics`) → `SemanticFieldReading`.
12. **Profile assembly** — `BehavioralRangeProfile` constructed from all of the above plus `ModelIdentity` and `PromptRecord` metadata, and the effective embedder recorded into the persisted config. Raw traces are attached only under the traceability opt-in.
13. **Measurement extraction** — `hif/profile/measure.py::measurements()` reduces the profile to the flat measurement dict, splitting off the prompt-only quantities by subject; `signals_record()` wraps both with provenance for `--json`, `suite`, and `batch`.
14. **Rendering (optional)** — `render_json()` writes the full profile as JSON; `render_technical()` and `render_public()` write Markdown reports. Nothing is written unless an output directory is requested — the privacy-first default writes nothing.
15. **Charts (optional)** — `generate_signal_plots()` renders the registry's signal charts plus a combined dashboard index.

`SessionEngine` (`hif/engine.py`) wraps steps 1–13 for the load-once/profile-many callers (`hif profile`, `hif batch`); it never writes an artifact implicitly.

## Field-model notes

The perturbation field, trajectory branch field, and within-generation
semantic field were built against a second private spec
(`DRIFT_FIELD_MODEL.md`), also not part of this repository. As above, this
section restates the rules the code verifiably implements and is the in-repo
authority for them.

- **Derived scalars only.** Field blocks persist descriptors — dispersion,
  radii, cluster counts, per-step cosine displacements — never a
  distribution, an embedding, or token identities. The raw variant/branch
  traces they are computed from are compute-and-discard
  (`hif/metrics/field.py`'s privacy invariant); persisting them is the
  explicit `traceability` opt-in.
- **Basis consistency.** The distribution basis used for field members must
  match the basis of the other output-side metrics. On a selected-only
  backend the raw traces are point masses and the field would collapse to a
  token-agreement signal, so when a surrogate is available each degenerate
  member is proxy-recovered the same way `semantic_steps` recovers the
  baseline (`builder.py` step 9a).
- **No deformation from one sample.** A per-generator sub-field needs ≥ 2
  members; with a single sample the within-class radius variance is
  undefined, and the sub-field is omitted rather than estimated.
- **Trajectory branch field.** The geometric twin of the perturbation field
  over sampling (branch) variation: where the retired
  `branch_pairwise_cosine_similarity` collapsed the branch cloud to one
  mean-pairwise-cosine scalar, `BranchField` restores its shape
  — centroid radii plus `cluster_count`, which detects multi-modality a mean
  cannot see (`hif/hourglass/trajectory.py`).
- **Within-generation semantic field.** The admitted measurements read one
  generation event; the semantic field reads the trajectory of the output's
  possibility field *within* a generation — per-step displacement of the
  probability-weighted candidate-cloud centroid (translation) and the
  step-to-step change in its spread (deformation)
  (`hif/analysis/semantic_field.py`). It published
  `semantic_centroid_veer_cosine` through hif-v3; that row was retired and
  the block remains as evidence.
