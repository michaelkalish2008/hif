# Configuration — Horizonal Interpretability Framework (HIF)

`hif/config.py` is the authoritative schema. This document explains what each
key does to the numbers, which is the part the source cannot tell you.

Four of the six measurements are not readings of one forward pass — they are
comparisons against runs the tool constructs. `perturbation_jsd_bits` compares
the baseline distribution against paraphrases; `input_entropy_shift_bits` and
`input_entropy_std_bits` read the mean and spread of the same paraphrase
sweep; `io_cosine_similarity` compares the prompt's embedding against the
generated output's. **For all four the configuration is part of the
measurement.** Two runs that differ only in `[perturbation] generators`
produce records that are identical in shape and different in value, with nothing
in the record to say why.

That is the reason this file exists, and the reason the run config belongs in
version control next to whatever you do with the output.

---

## One control surface, every scale

`profile` (one case) and `batch` (many cases, model loaded once) are two
scales of one operation. Both take `--config-file`, `--mode`, `--acquisition`,
`--lite`, and `--variant-io`, resolved through the same code path, so a
ceiling means the same thing at either scale and a corpus is comparable with
the single runs it aggregates.

**The built-in prompt suite is a row source, not a third command.**
`hif batch --sample-set all <model>` feeds `batch` the fixed 8 x 5 stimulus
set; `--sample-set <regime>` narrows it. Because it is a row source it
inherits every control above — which the old standalone `hif suite` did not,
having drifted to where it accepted no config file and no ceilings at all.

Its prompts are fixed and that is the entire point: a cross-model comparison
is only a comparison when the stimulus was identical. It is not a benchmark —
the prompts are unlabeled, nothing is scored — and it is not where your own
research question lives. For that, fork it:

```bash
hif batch --sample-set all --export-workload suite.jsonl   # 40 rows, no model
$EDITOR suite.jsonl                                        # add prompts, add `variants`
hif batch suite.jsonl gpt2
```

## How a config is assembled

A run's `RunConfig` is built from up to four sources. Later beats earlier:

1. **Schema defaults** — everything in `hif/config.py`.
2. **`--config-file run.toml`** — the baseline you author. Tables mirror
   `RunConfig` fields.
3. **`--mode`, `--diagnostics`** — presets that touch specific stages.
4. **Explicit CLI flags** — `--max-new-tokens`, `--top-k`, `--seed`, and the
   model identity. "Explicit" means you typed it; a flag left at its default
   does not override the file.

`--lite` is applied last of all and overrides every source for the stages it
disables (see below). Model **name and backend** always come from the CLI
arguments and are never read from the file — but `[model] base_url`, `api_key`,
`dtype`, `revision`, and `temperature` do survive from the file, which is the
only way to point an `openai`-backend arm at an OpenAI-compatible endpoint.

```bash
hif profile gpt2 "Explain why the sky appears blue." --config-file run.toml --json
```

### The authoring loop

```bash
hif config init                                # write run.toml, every key at its default
$EDITOR run.toml                               # your diff from this file IS the condition
hif config show --config-file run.toml --diff  # confirm what will actually run
hif profile gpt2 "..." --config-file run.toml --json
```

Three guarantees hold across that loop:

**A mistyped key anywhere in the file exits 3.** Both an unknown table
(`[perturbaton]`) and an unknown key inside a valid table
(`generatorz = [...]`) are rejected with the valid alternatives named. A typo
can no longer silently measure with the defaults. The one exemption is
`[model] extra_body`, whose keys are the provider's vocabulary, not hif's.

**`hif config show` resolves without running.** It goes through the *same*
resolution path `hif profile` executes — file, `--mode`, `--acquisition`,
`--lite`, explicit flags, all of it — so what it prints is what a run would do,
and the two cannot drift. `--diff` limits output to departures from the schema
defaults; the TOML output is valid `--config-file` input, so
`hif config show ... > run.toml` round-trips. Secrets print as `<redacted>`.

**The record carries the resolved config.** `record-v6` embeds a `run_config`
block — the same dict `config show` prints, from the same serializer — so a
record is reproducible and comparable on its own. Two records that differ in
`distance_threshold` now say so. Secrets are redacted, not omitted:
`"<redacted>"` vs `null` preserves whether the run authenticated without
carrying the credential.

---

## `[perturbation]` — what `perturbation_jsd_bits` compares against

Feeds `perturbation_jsd_bits`, `input_entropy_shift_bits`,
`input_entropy_std_bits` — three of the six measurements, which is why this
section is the one most worth reading before comparing two records.

| key | default | effect |
| --- | --- | --- |
| `generators` | `["synonym", "tone", "reorder"]` | Which paraphrase families are applied to the prompt |
| `n_variants` | `2` | Variants per generator. Total runs = `len(generators) × n_variants`, each costing one generation pass |
| `use_llm_perturbation` | `false` | Rule-based (deterministic, free) vs. LLM-authored paraphrases |
| `llm_base_url` | `null` | OpenAI-compatible endpoint. Required when `use_llm_perturbation = true` |
| `llm_api_key` | `null` | Key for that endpoint |
| `llm_model` | `null` | Model at that endpoint; omit for the generator's own default |
| `elicit_variant_outputs` | `true` | Whether the model **generates** from each variant. `false` authors and teacher-forces them only — the input-side pair survives, the four output-side ones go absent. See [`--acquisition`](#--acquisition--what-the-run-may-bring-into-existence) |
| `variants_file` | `null` | Workload JSONL of researcher-authored variants — the tool authors nothing. See below |
| `media_families` | `["image_grid_mask"]` | Multimodal only; ignored on the text path |
| `image_grid_rows` / `image_grid_cols` | `4` / `4` | Mask granularity for multimodal runs |

### `variants_file` — author the perturbations yourself

The generators above are rule-based text manipulations; you control *which
families* run but not *what they write*. Authored variants invert that: every
variant is a string a person wrote.

There is **one row format for all case data**: the workload JSONL that
`hif batch` profiles (`{"query_id", "text", ...}`), extended with a
`variants` list. No second file format, no join to maintain — a variants file
IS a workload file:

```jsonl
{"query_id": "sky_1", "text": "Explain why the sky appears blue.", "variants": ["Explain why the sky looks blue.", "Why does the sky appear blue? Explain."]}
```

Two ways to use the same file:

```bash
# batch: rows are profiled directly, each row's variants replace the generators
hif batch variants.jsonl gpt2

# single run: the config points at the file; rows are matched to the prompt
# by EXACT string equality on `text`
hif profile gpt2 "Explain why the sky appears blue." --config-file run.toml
```

```toml
[perturbation]
variants_file = "variants.jsonl"
```

- When variants apply, `generators` and `n_variants` are **ignored** — the
  file is the perturbation set, and the profile's perturbation entries carry
  `generator="authored"`.
- A prompt with no usable rows is a **hard error**, not an empty perturbation
  set: silence there would report an un-perturbed run under a perturbed
  config. In a workload row, omitting `variants` means "use the generators";
  an explicit empty list is rejected at load — the two must not be spelled
  the same way.
- Downstream, nothing changes: same measurements, same units. Authorship
  changes provenance, not procedure.
- JSONL also carries paraphrase text better than any delimited format:
  prompts contain commas and newlines, and JSON escaping is defined where CSV
  quoting is a negotiation.

**`--variant-io` closes the review loop.** Inputs stay immutable and outputs
live in records — so instead of writing anything back into your file, the
flag adds a `variant_io` block to the `--json` record: each variant's input
text and the continuation it elicited, `null` where none was (a
`synthesized-input` run teacher-forces without generating, and `null` there
is the truthful value — inventing one would be a fabricated elicitation).
This answers the elicited-output tier's "unreviewed model output" objection:
every elicited continuation is in the record, next to the paraphrase that
produced it and the `run_config` that governed it. Opt-in, because it adds
model-generated content to every record.

Commit the JSONL next to `run.toml` — it is as much a part of the measurement
as the thresholds are.

**Available generators.** `synonym`, `tone`, `reorder`, `substitution`,
`ambiguity`. The first three have both rule-based and LLM implementations;
`substitution` and `ambiguity` are rule-based only.

The choice is not cosmetic. `synonym` and `substitution` perturb lexical choice;
`reorder` perturbs syntax; `tone` perturbs register; `ambiguity` perturbs
determinacy. A model can be flat under one family and volatile under another,
and `perturbation_jsd_bits` averages over whichever set you chose — so the
default's `0.6` and a `reorder`-only `0.6` are not the same finding.

`n_variants = 0` disables the stage: no variants, and the five measurements
above become absent rather than zero.

```toml
[perturbation]
generators = ["synonym", "substitution"]   # lexical only
n_variants = 4                             # 8 variants, 8 extra generation passes
```

---

## `[trajectory]` — the branch stage

hif-v4 publishes no measurement from this stage: `branch_pairwise_cosine_similarity`
was cut. The stage still runs and still records `trajectory.branch_field` as
evidence under `--diagnostics`, and these keys still shape it.

| key | default | effect |
| --- | --- | --- |
| `n_branches` | `5` | Continuations resampled from the branch point. `0` disables the stage |
| `rollout_steps` | `10` | Tokens generated per branch |

The measurement is the mean pairwise cosine similarity between branch
embeddings. Both keys move it: more branches change which pairs are averaged,
and longer rollouts give the branches more room to diverge, so similarity
generally falls as `rollout_steps` rises. Comparing this number across runs with
different `rollout_steps` is not a comparison.

Requires a backend that can teacher-force. On API backends the stage is skipped
regardless of configuration, and `trajectory_analysis_ran` in the record's
provenance block says so.

```toml
[trajectory]
n_branches = 8
rollout_steps = 16
```

---

## `[exposure]` — the counterfactual-exposure stage

hif-v4 publishes no measurement from this stage: `counterfactual_exposure_fraction`
was cut, in part because the two thresholds below were embedded in the number
it reported — a configured quantity presented as a measured one. The stage
still runs and records `profile.exposure` as evidence under `--diagnostics`.

| key | default | effect |
| --- | --- | --- |
| `enabled` | `true` | Run the stage at all |
| `min_prob` | `0.01` | Minimum probability for an alternative token to be considered accessible |
| `distance_threshold` | `0.3` | Cosine distance at or above which a diffusion-zone step counts as exposed |

```
E = |{ t : diffusion(t) ∧ d_t ≥ τ }| / G
```

Both are choices, not constants. Lowering `min_prob` admits fainter
alternatives; lowering `distance_threshold` counts nearer ones as divergent.
Either raises the fraction. A reported exposure of `0.08` means nothing without
the pair that produced it.

The distances are embedding-model-dependent — values are comparable only under
the same `[embedding] model_name`.

The old table name `[hallucination]` still loads, for archived configs.

---

## `[semantic]` — per-step candidate geometry

| key | default | effect |
| --- | --- | --- |
| `enabled` | `true` | Embed and cluster each step's candidate cloud |

Fed `candidate_cluster_entropy_bits`, which hif-v4 cut; exposure and the
semantic field still read it, and it remains the most expensive per-step stage
on a run with no perturbation variants — this is the switch `--lite` throws.

## `[cluster]` — how those candidates are grouped

| key | default | effect |
| --- | --- | --- |
| `method` | `"hdbscan"` | Clustering algorithm |
| `min_cluster_size` | `2` | Minimum members for a cluster |
| `min_samples` | `1` | HDBSCAN density parameter |

These keys change the cluster assignment directly — coarser clustering means
fewer clusters — and so change every diagnostic block read off it. The
measurement that reported cluster entropy, `candidate_cluster_entropy_bits`,
was cut in hif-v4: it moved with these settings as much as with the model.

---

## `[generation]` — the run everything else is derived from

| key | default | effect |
| --- | --- | --- |
| `max_new_tokens` | `64` | Generation length. Every per-step series is this long |
| `top_k` | `50` | Candidates retained per step |
| `temperature` | `1.0` | Sampling temperature |
| `seed` | `42` | Random seed |

`max_new_tokens` is the quietest comparability trap in the whole config. Every
step-series measurement is an average over the steps that exist, so a 16-token
run and a 64-token run of the same model on the same prompt are not comparable —
even though both report `output_entropy_bits` with no annotation.

`top_k` bounds what any distributional measurement can see. Backends impose
their own ceilings; when the effective K differs from the requested one,
`--verbose` says so.

**Temperature has two homes.** Sampling adapters read `[model] temperature`, not
`[generation] temperature`. A `[generation] temperature` you set explicitly is
mirrored onto the model config; an explicit `[model] temperature` wins. Left
unset, each backend uses its own default (`0` for OpenAI, unchanged sampling for
HF) — which is why the mirror only fires when the file actually set it.

---

## `[model]` — identity comes from the CLI, everything else from here

| key | default | effect |
| --- | --- | --- |
| `name` / `backend` | `"gpt2"` / `"hf"` | **Ignored from the file** — always the CLI arguments |
| `base_url` | `null` | OpenAI-compatible endpoint (Mistral, DeepSeek, Grok, vLLM) |
| `api_key` | `null` | Overrides the environment variable |
| `device` | `"auto"` | `auto` / `cpu` / `cuda` / `mps` |
| `dtype` | `"float32"` | Load precision |
| `revision` | `null` | Pin a HF Hub commit SHA. `null` floats on `main` |
| `temperature` | `null` | See above |
| `extra_body` | `null` | JSON merged into each OpenAI-compatible request |
| `ollama_host` / `ollama_timeout` | `http://localhost:11434` / `120.0` | Ollama transport |

`revision` is the difference between a reproducible profile and one that silently
re-runs against different weights next month.

`extra_body` exists for provider options outside the OpenAI schema. DeepSeek's
reasoning mode is the motivating case: left on at `max_new_tokens = 64` it spends
most of the budget reasoning and returns a fraction of the content steps, so the
measured generation is a different length from every other model's.

```toml
[model]
base_url = "https://api.deepseek.com/v1"
extra_body = { thinking = { type = "disabled" } }
```

---

## `[embedding]` — the basis for every geometric measurement

| key | default | effect |
| --- | --- | --- |
| `model_name` | `"sentence-transformers/all-MiniLM-L6-v2"` | Encoder |
| `fallback_model_name` | same | Used when the primary fails to load |
| `matryoshka_dim` | `null` | Truncation dim (e.g. `256` for EmbeddingGemma) |
| `cache_dir` | `~/.cache/hif/embeddings` | Embedding cache |

Changing the encoder changes `io_cosine_similarity` — the one measurement
that reads it — along with the trajectory, cluster and exposure diagnostic
blocks. Cosine values are comparable only within a single encoder. The
effective encoder is recorded in each profile's `config.embedding.model_name`
and printed by `--verbose`.

---

## `[attention]` and `[semantic_field]` — instrument readings, off by default

| table | key | default |
| --- | --- | --- |
| `[attention]` | `enabled` | `false` |
| | `model_name` | `"distilbert-base-uncased"` |
| | `aggregate_method` | `"mean_all_layers"` (or `last_layer`, `mean_upper_half`) |
| | `max_seq_length` | `512` — longer texts are truncated |
| | `trajectory_interval` | `4` |
| `[semantic_field]` | `enabled` | `false` |
| | `context_window` | `5` |

`--diagnostics` turns both on. It only ever turns analyzers **on** — it never
disables one a config file enabled.

`aggregate_method` decides which tokens appear load-bearing, and no choice is
canonical. Texts beyond `max_seq_length` are silently truncated, so structural
readings on long prompts describe only the portion that fit.

---

## `[traceability]` and `[output]`

| table | key | default | effect |
| --- | --- | --- | --- |
| `[traceability]` | `enabled` | `false` | Persist raw per-step top-K traces into the artifact |
| `[output]` | `output_dir` | `outputs` | Only consulted when `--output-dir` asks for reports |

Compute-and-discard is the default: variant and branch traces are held only long
enough to derive field descriptors, then dropped. `traceability.enabled = true`
is the sanctioned exception — it captures per-step top-K **with token identity**,
which is reconstructable content. Enable it only where the artifact's storage is
trusted with prompt- and continuation-level text.

Nothing is written to disk unless you ask. `--output-dir` opts into reports and
charts; `--trace` opts into the artifact.

---

## `--acquisition` — what the run may bring into existence

A separate axis from everything above, and the one to reach for first. It caps
what the run is *permitted to produce*, not how much work it does.

| tier | permits | adds |
| --- | --- | --- |
| `observational` | The prompt as given and the one continuation the run produces. Nothing else is sent to the model; no new model output exists afterwards. | `output_entropy_bits`, `prompt_surprisal_excess_bits` |
| `synthesized-input` | Additionally authors paraphrased prompts and teacher-forces the model over them. The model does not generate. | `input_entropy_shift_bits`, `input_entropy_std_bits` |
| `elicited-output` *(default)* | Additionally lets the model generate variant continuations and trajectory branches. | `perturbation_jsd_bits`, `io_cosine_similarity` |

```bash
hif profile gpt2 "Explain why the sky appears blue." --acquisition observational --json
```

The tiers are strictly nested, and the surviving values are **identical** across
them — raising the ceiling only ever adds keys. Every row in `hif schema`
carries its `acquisition`, so the partition is machine-readable rather than
something you have to reconstruct from the pipeline.

The corresponding config key is `[perturbation] elicit_variant_outputs`
(default `true`). Setting it `false` authors and teacher-forces the paraphrases
without generating from them — which is what `synthesized-input` does. The
input-side pair never needed a variant continuation; before this key existed
they were computed in the same loop that generated one, so taking them cost an
elicitation they did not require.

## `--lite`

Applied after every other source, overriding all of them:

```
[perturbation] generators = [], n_variants = 0
[trajectory]   n_branches = 0
[semantic]     enabled = false
[exposure]     enabled = false
[semantic_field] enabled = false
[attention]    enabled = false
```

Surviving measurements are identical to a full run — `--lite` removes stages, it
does not approximate them. The rest are absent from the record, never `0.0`.
See the README for the full survives/omitted table.

---

## Verifying what actually ran

In descending order of directness:

1. **`hif config show --config-file run.toml --diff`** — before the run. The
   same resolution path the run executes; a key you set that does not appear
   here did not apply (and a typo'd key exits 3 before you get this far).
2. **The record's `run_config` block** — after the run. The resolved config
   the run executed, embedded in the `--json` output itself.
3. **`--verbose`** prints the paraphrase variants the run actually used, the
   effective top-K, and the embedder.
4. **Provenance in `--json`** reports `trajectory_analysis_ran`, the
   teacher-forcing model, and the output-distribution model.
5. **`--trace`** persists the full artifact for retroactive recomputation.

---

## A worked file

```toml
# run.toml — lexical perturbation only, longer branches, stricter exposure
[generation]
max_new_tokens = 128
top_k = 50
seed = 7

[perturbation]
generators = ["synonym", "substitution"]
n_variants = 4

[trajectory]
n_branches = 8
rollout_steps = 16

[exposure]
min_prob = 0.02
distance_threshold = 0.25

[embedding]
model_name = "sentence-transformers/all-MiniLM-L6-v2"

[model]
revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"   # pin the weights
```

```bash
hif profile gpt2 "Explain why the sky appears blue." \
  --config-file run.toml --json --verbose
```

Note what this file does **not** do: it does not set `max_new_tokens` for one
model and leave it default for another, and it does not change the encoder
between runs you intend to compare. Both are the same mistake — a comparison
between two different measurements wearing one name.

---

## See also

- [`MEASUREMENTS.md`](MEASUREMENTS.md) — what each measurement is, in its unit
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — where in the pipeline each stage runs
- [`PROMPT_SUITE.md`](PROMPT_SUITE.md) — the regimes, and why they are unlabeled
