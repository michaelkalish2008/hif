# Flags — Horizonal Interpretability Framework (HIF)

Every command, argument and flag, generated from the CLI by `tools/gen_flags_doc.py`. Do not edit by hand: regenerate it, so this file cannot claim a flag `hif` does not have.

Run `hif <command> --help` for the same text in the terminal.

---

## `hif batch`

Profile many prompts against one loaded model.

| argument | meaning |
| --- | --- |
| `workload` | Workload JSONL file: one {"query_id", "text"[, "regime", "variants"]} row per line. Omit it when using --sample-set. |
| `model_name` | Model name (e.g. gpt2) |

| flag | meaning |
| --- | --- |
| `--sample-set` | Profile the built-in prompt suite instead of a workload file: `all` (8 regimes x 5 prompts) or one regime name. A fixed stimulus set, identical for every model — not a benchmark, and nothing is scored. |
| `--limit` | Profile only the first N rows. |
| `--export-workload` | Write the resolved rows to a workload JSONL and exit; no model loads. This is how you fork --sample-set: edit the rows, add per-row `variants`, run it back. |
| `--regime` | Regime for rows with no "regime" key of their own. A free-form label recorded with the run — any string, compared against nothing, changing no measurement. Name it whatever your work calls it; `hif batch --sample-set` names its own. *(default: `batch`)* |
| `--backend` | Model backend: `hf`, `tlens`, `ollama`, `openai`, `anthropic`, `gemini`. Run `hif models` for what each one can measure. See [Backends](#backends). *(default: `hf`)* |
| `--max-new-tokens` | Maximum new tokens to generate, per row. *(default: `64`)* |
| `--top-k` | How many candidates to record at each step. *(default: `50`)* |
| `--seed` | Random seed, recorded with every record. *(default: `42`)* |
| `--lite` | Speed: skip every stage that costs an extra generation pass or an embedding sweep, on every row. Their measurements come back absent, not zero. |
| `--mode` | Perturbation budget per row: fast = 2 paraphrase variants, audit = 5. *(default: `fast`)* |
| `--acquisition` | Ceiling on what each row may bring into existence — provenance, not speed: observational \| synthesized-input \| elicited-output. Same meaning as `hif profile --acquisition`; `hif schema` gives each measurement's tier. *(default: `elicited-output`)* |
| `--config-file` | TOML run config; its tables mirror RunConfig. Flags you pass explicitly win. Confirm with `hif config show`. |
| `--units` | Add a per-measurement units block to each record. Constant per signal_set_version, so off by default; `hif schema` prints the same information without running a model. |
| `--variant-io` | Add each perturbation variant's input text and the continuation it elicited to every record. |
| `--entropy-percentile` | Also report output_nucleus_entropy_bits: the entropy of the smallest per-step prefix carrying this percent of the output distribution's mass (e.g. 95), renormalized. Needs a full-logprob backend (`hif models`). |
| `--output-dir` | Also mirror the stdout record stream to <output-dir>/records.jsonl. |
| `--trace` | Persist each row's full profile artifact — raw per-step top-K distributions, reconstructable content — for later recomputation. |
| `--trace-dir` | Where --trace artifacts are written (default: <output-dir>/traces, or ./traces when no --output-dir). Passing this implies --trace. |
| `--surrogate` | Recover the input-side measurements on backends that cannot teacher-force by teacher-forcing a small local proxy model instead, so those numbers describe the proxy, not your model (see `hif profile --surrogate`). |
| `--surrogate-model` | Open-weight HF model id to use as that proxy (default: Llama 3.2 1B, ungated mirror). Passing it implies --surrogate; `hif models --surrogates` lists candidates. |

**Examples**

```bash
# the built-in suite: 8 regimes x 5 prompts, one record per row on stdout
hif batch --sample-set all gpt2

# your own rows; records stream to stdout and mirror to out/records.jsonl
hif batch workload.jsonl gpt2 --output-dir out

# write the suite's rows as a file to edit and run back — no model is loaded
hif batch --sample-set all --export-workload suite.jsonl gpt2

# a quick shape-check of a new workload before committing to the full run
hif batch workload.jsonl gpt2 --lite --limit 5
```

## `hif compare`

Report the per-measurement difference between two profiles.

| argument | meaning |
| --- | --- |
| `profile_a` | Path to the first profile JSON |
| `profile_b` | Path to the second profile JSON |

| flag | meaning |
| --- | --- |
| `--output` | Optional output Markdown file |
| `--json` | Output machine-readable JSON |

**Examples**

```bash
# first make the artifacts: compare reads --trace profiles, NOT --json records
hif profile gpt2 "..." --trace --trace-dir tr

# per-measurement difference between the two, as a table
hif compare tr/profile_<a>.json tr/profile_<b>.json

# the same comparison as a record, for a script
hif compare tr/profile_<a>.json tr/profile_<b>.json --json
```

## `hif config init`

Write a run.toml of pure schema defaults to edit from.

| flag | meaning |
| --- | --- |
| `--output`, `-o` | Where to write the template. *(default: `run.toml`)* |
| `--force` | Overwrite an existing file. |

## `hif config show`

Print the fully resolved run config — without loading a model or running.

| argument | meaning |
| --- | --- |
| `model_name` | Model name (affects [model] only) |

| flag | meaning |
| --- | --- |
| `--backend` | Model backend — see [Backends](#backends) *(default: `hf`)* |
| `--config-file` | TOML run config to resolve (same file `hif profile` takes). |
| `--mode` | fast \| audit (perturbation budget) *(default: `fast`)* |
| `--lite` | Apply the --lite stage budget |
| `--acquisition` | observational \| synthesized-input \| elicited-output *(default: `elicited-output`)* |
| `--max-new-tokens` | Maximum new tokens *(default: `64`)* |
| `--top-k` | Top-K candidates per step *(default: `50`)* |
| `--seed` | Random seed *(default: `42`)* |
| `--diagnostics` | Apply --diagnostics |
| `--diff` | Show only departures from the schema defaults. |
| `--json` | Emit JSON instead of TOML. |

## `hif doctor`

Preflight check: dependencies, running services, credentials, and per-backend readiness.

Takes no arguments or flags.

## `hif models`

List the backends you can profile, example models, and which signals each supports.

| flag | meaning |
| --- | --- |
| `--backend` | Show only this backend (hf, tlens, ollama, openai, anthropic, gemini). See [Backends](#backends). |
| `--list` | Query each backend's actual model catalog right now (needs the provider's API key, or a running Ollama server) instead of showing static examples — use this when an example model from the docs turns out to be retired/unavailable. |
| `--surrogates` | List recommended --surrogate-model choices (small open-weight models for recovering input-side signals on closed/Ollama backends via --surrogate) and check each is currently reachable and ungated on the Hugging Face Hub. |
| `--json` | Emit the catalogue as a single JSON document on stdout instead of the human table, so the model list can be piped, scripted, or fed to a picker. Composes with --backend, --list and --surrogates. |

**Examples**

```bash
# every backend, with example models and the signals each one supports
hif models

# just one backend's row
hif models --backend hf

# small open-weight models usable with --surrogate, checked for reachability
hif models --surrogates
```

## `hif profile`

Run the full hif pipeline on a single (model, prompt) pair.

| argument | meaning |
| --- | --- |
| `model_name` | Model name (e.g. gpt2) |
| `prompt` | Prompt text |

| flag | meaning |
| --- | --- |
| `--backend` | Model backend: `hf`, `tlens`, `ollama`, `openai`, `anthropic`, `gemini`. Run `hif models` for what each one can measure. See [Backends](#backends). *(default: `hf`)* |
| `--max-new-tokens` | Maximum new tokens to generate. *(default: `64`)* |
| `--top-k` | How many candidates to record at each step. *(default: `50`)* |
| `--seed` | Random seed, recorded with the run. *(default: `42`)* |
| `--truncate` | Cut the prompt to its first N whitespace-split tokens before the run. Results then reflect truncated context only. |
| `--lite` | Speed: skip every stage that costs an extra generation pass or an embedding sweep. Their measurements come back absent, not zero. |
| `--mode` | Perturbation budget: fast = 2 paraphrase variants, audit = 5. The prompt itself is always passed in full. *(default: `fast`)* |
| `--acquisition` | Ceiling on what the run may bring into existence — provenance, not speed. observational: only the one call you asked for. synthesized-input: + authored prompts, teacher-forced. elicited-output: + model-generated variants and branches. Above the ceiling, measurements are absent; `hif schema` gives each one's tier. *(default: `elicited-output`)* |
| `--diagnostics` | Also run attention capture and the semantic field. Neither produces a measurement; their blocks ship in the --trace artifact. |
| `--config-file` | TOML run config; its tables mirror RunConfig ([generation], [perturbation], [trajectory], [attention], [semantic_field], ...). Flags you pass explicitly win. Confirm with `hif config show`. |
| `--json` | Print the record as JSON: derived measurements only, never the raw per-step distributions. |
| `--metric` | Print ONE measurement in its natural unit, then exit. Names and units: `hif schema`. |
| `--units` | Add a per-measurement units block to each record. Constant per signal_set_version, so off by default; `hif schema` prints the same information without running a model. |
| `--variant-io` | Add each perturbation variant's input text and the continuation it elicited to the --json record (null where none was elicited). |
| `--entropy-percentile` | Also report output_nucleus_entropy_bits: the entropy of the smallest per-step prefix carrying this percent of the output distribution's mass (e.g. 95), renormalized. Needs a full-logprob backend (`hif models`). |
| `--verbose`, `-v` | Also show model input/output text, perturbation variants, full numeric stats, and internal logging. |
| `--output-dir` | Write the technical and public Markdown reports here, and the --charts plots. |
| `--charts` | One interactive Plotly HTML per signal, plus an index.html dashboard. Needs --output-dir. |
| `--trace` | Persist the full profile artifact — raw per-step top-K distributions, reconstructable content — so measurements can be recomputed later without re-running the model. |
| `--trace-dir` | Where --trace artifacts are written (default: <output-dir>/traces, or ./traces when no --output-dir). Passing this implies --trace. |
| `--regime` | A free-form label recorded with the run — any string, compared against nothing, changing no measurement. Name it whatever your work calls it; `hif batch --sample-set` names its own. *(default: `ordinary_conversation`)* |
| `--application` | A free-form label for what this run is for, recorded with the run — any string, changing no measurement. |
| `--surrogate` | Recover the input-side measurements on backends that cannot teacher-force — score text they did not generate (ollama, openai, anthropic, gemini; see `hif models`). A small local proxy model is teacher-forced instead, so those numbers describe the proxy, not your model. Ignored on hf/tlens. |
| `--surrogate-model` | Open-weight HF model id to use as that proxy (default: Llama 3.2 1B, ungated mirror). Passing it implies --surrogate; `hif models --surrogates` lists candidates. |

**Examples**

```bash
# measure one prompt; prints to the terminal and writes nothing
hif profile gpt2 "Why is the sky blue?"

# same run, plus Markdown reports and one Plotly chart per signal under out/
hif profile gpt2 "Why is the sky blue?" --output-dir out --charts

# print one number and exit — the form to use inside a script
hif profile gpt2 "Why is the sky blue?" --metric output_entropy_bits

# the fast subset, as a JSON record; skipped stages come back absent, not zero
hif profile gpt2 "Why is the sky blue?" --lite --json

# add output_nucleus_entropy_bits; the wide --top-k is what it needs, not the --lite
hif profile gpt2 "Why is the sky blue?" --entropy-percentile 95 --top-k 2000 --lite
```

## `hif render`

Load an existing profile from JSON and re-render Markdown.

| argument | meaning |
| --- | --- |
| `profile_json` | Path to profile JSON |

| flag | meaning |
| --- | --- |
| `--public` | Produce public-facing summary instead of technical |
| `--output` | Output path (default: alongside JSON) |

## `hif schema`

Print the measurement registry: every key with its full row.

| flag | meaning |
| --- | --- |
| `--json` | Emit the machine-readable schema document (default) or a human table. *(default: `True`)* |

**Examples**

```bash
# the measurement registry as JSON: every key, unit, subject and definition
hif schema

# just the measurement names — the valid values for --metric
hif schema | jq -r '.measurements | keys[]'
```

## What `hif doctor` checks

`doctor` takes no flags. It reports, in order:

| check | what it tells you |
| --- | --- |
| core (numpy, plotly) | the two hard dependencies |
| embedder (sentence-transformers) | needed by every geometric measurement |
| charts (--charts) | HTML always; PNG only with kaleido installed |
| ollama server | reachability and which models are pulled |
| per-backend readiness | optional deps and credentials, one row per backend |

## Backends

What every `--backend` accepts, and what each one lets you measure. Generated from `hif/models/capabilities.py`, the same registry `hif models` and `hif doctor` read.

| backend | access | input-side signals | output logprobs | notes |
| --- | --- | --- | --- | --- |
| `hf` (default) | local, open weights | yes | full | Full fidelity — every measurement. Best for a complete profile. |
| `tlens` | local, open weights | yes | full | Full fidelity via TransformerLens. |
| `gemini` | hosted API | no | top-k | Top-20 logprobs on Vertex AI only; the developer API degenerates. |
| `ollama` | local service | no | top-k | Output-side signals only (top-20). No input-side or attention signals. The model MUST be pulled locally before profiling. |
| `openai` | hosted API | no | top-k | Output-side signals only (top-20 logprobs). |
| `anthropic` | hosted API | no | selected-only | No token-level logprobs. Entropy-shaped signals degenerate, and the distribution divergences are reported absent rather than as the token-agreement rate two point masses actually produce. Best for io_cosine_similarity, the one measurement it can fully support. |
