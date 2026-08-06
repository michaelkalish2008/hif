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
| `--backend` | Model backend: hf \| tlens \| ollama \| openai \| anthropic \| gemini *(default: `hf`)* |
| `--regime` | Default prompt regime (a per-row "regime" key overrides it). *(default: `batch`)* |
| `--seed` | Random seed *(default: `42`)* |
| `--max-new-tokens` | Maximum new tokens to generate *(default: `64`)* |
| `--top-k` | Top-K candidates per step *(default: `50`)* |
| `--config-file` | TOML run config (tables mirror RunConfig). CLI flags you pass explicitly override the file. |
| `--mode` | fast: fewer perturbation variants. audit: full perturbation set. *(default: `fast`)* |
| `--acquisition` | Ceiling on what this run may bring into existence, applied to every row. observational \| synthesized-input \| elicited-output. Same meaning as `hif profile --acquisition`; run `hif schema` for each measurement's tier. *(default: `elicited-output`)* |
| `--lite` | Skip perturbation variants, trajectory branches, and per-step candidate geometry on every row (see `hif profile --lite`). |
| `--variant-io` | Include a `variant_io` block in each record: every perturbation variant's input text and the continuation it elicited. |
| `--surrogate` | Recover input-side signals on backends that cannot teacher-force by teacher-forcing a small local proxy model (see `hif profile --surrogate`). Implied by --surrogate-model. |
| `--surrogate-model` | Open-weight HF model id used for --surrogate (default: Llama 3.2 1B, ungated mirror). Passing this implies --surrogate. |
| `--trace` | Opt-in traceability: persist each row's full profile artifact (raw per-step top-K distributions). Default off: compute-and-discard. |
| `--trace-dir` | Where --trace artifacts are written (default: <output-dir>/traces, or ./traces when no --output-dir). Passing this implies --trace. |
| `--sample-set` | Use the built-in prompt suite instead of a workload file: `all` (8 regimes x 5 prompts) or a single regime name. A FIXED stimulus set — identical prompts for every model, which is the condition for a cross-model comparison being a comparison. It is not a benchmark: the prompts are unlabeled and nothing is scored. Pair with --export-workload to fork it. |
| `--export-workload` | Write the resolved rows as a workload JSONL and exit — no model is loaded. With --sample-set, this is how you fork the built-in suite: edit the rows, add per-row `variants`, then run it back. |
| `--limit` | Profile only the first N workload rows. |
| `--output-dir` | Also mirror the stdout record stream to <output-dir>/records.jsonl. Default: records stream to stdout only. |
| `--units` | Include a per-measurement units block in each record. Constant per signal_set_version and identical on every record, so off by default; `hif schema` prints the same information without running a model. |

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
| `--backend` | Model backend *(default: `hf`)* |
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
| `--backend` | Show only this backend (hf, tlens, ollama, openai, anthropic, gemini). |
| `--list` | Query each backend's actual model catalog right now (needs the provider's API key, or a running Ollama server) instead of showing static examples — use this when an example model from the docs turns out to be retired/unavailable. |
| `--surrogates` | List recommended --surrogate-model choices (small open-weight models for recovering input-side signals on closed/Ollama backends via --surrogate) and check each is currently reachable and ungated on the Hugging Face Hub. |
| `--json` | Emit the catalogue as a single JSON document on stdout instead of the human table, so the model list can be piped, scripted, or fed to a picker. Composes with --backend, --list and --surrogates. |

## `hif profile`

Run the full hif pipeline on a single (model, prompt) pair.

| argument | meaning |
| --- | --- |
| `model_name` | Model name (e.g. gpt2) |
| `prompt` | Prompt text |

| flag | meaning |
| --- | --- |
| `--regime` | Prompt regime *(default: `ordinary_conversation`)* |
| `--backend` | Model backend: hf \| tlens \| ollama \| openai \| anthropic \| gemini *(default: `hf`)* |
| `--seed` | Random seed *(default: `42`)* |
| `--output-dir` | Write derived reports (technical + public markdown, --charts plots) here. Default: nothing is written to disk — results print to the terminal only (privacy-first compute-and-discard). |
| `--max-new-tokens` | Maximum new tokens to generate *(default: `64`)* |
| `--top-k` | Top-K candidates per step *(default: `50`)* |
| `--config-file` | TOML run config (tables mirror RunConfig: [generation], [perturbation], [trajectory], [attention], [semantic_field], ...). CLI flags you pass explicitly override the file. |
| `--trace` | Opt-in traceability: persist the full profile artifact (raw per-step top-K distributions — reconstructable content) so signals can be recomputed or audited later without re-running the model. Default off: compute-and-discard. |
| `--trace-dir` | Where --trace artifacts are written (default: <output-dir>/traces, or ./traces when no --output-dir). Passing this implies --trace. |
| `--charts` | Generate plots + the combined dashboard locally (off by default). |
| `--diagnostics` | Also run the two optional analysis stages — attention capture and the semantic field. Neither produces a measurement in hif-v4; their blocks ship in the --trace artifact as evidence. Off by default because both cost extra compute. |
| `--application` | Application archetype (support-chatbot, rag-qa, coding-assistant, summarization, extraction, classification, agent-tool-use, document-understanding). Labels the run and supplies the default --analysis-window; both are recorded in the JSON record. It does not change how anything is measured. |
| `--mode` | fast: fewer perturbation variants. audit: full perturbation set. Input is always passed in full regardless of mode. *(default: `fast`)* |
| `--variant-io` | Include a `variant_io` block in the --json record: each perturbation variant's input text and the continuation it elicited (null where none was — synthesized-input tier, or a failure). Opt-in because it adds model-generated content to every record; outputs live in records, inputs stay immutable. |
| `--acquisition` | Ceiling on what this run may bring into existence. observational: read the prompt as given and the one continuation the run produces — nothing else is sent to the model and no new model output exists afterwards. synthesized-input: additionally author paraphrased prompts and teacher-force over them (the model still does not generate). elicited-output (default): additionally let the model generate variant continuations and trajectory branches. Measurements above the ceiling are absent, not zero. Run `hif schema` to see each measurement's acquisition tier. *(default: `elicited-output`)* |
| `--lite` | Skip every stage that costs an extra generation pass or an embedding sweep: perturbation variants, trajectory branches, and per-step candidate geometry. The entropy-side measurements are unchanged; the ones those stages feed are omitted, not zeroed. Overrides --mode and --config-file for the stages it disables. |
| `--analysis-window` | Maximum output tokens to analyze (does not truncate inference). Integer or 'adaptive' (default: adaptive = analyze all output). |
| `--metric` | Print ONE measurement, in its natural unit, and exit. Run `hif schema` for the full list with unit definitions. |
| `--verbose`, `-v` | Show model input/output text, perturbation variants, full numeric stats, effective-config notes, and full internal logging (pipeline + HTTP chatter) |
| `--json` | Output machine-readable JSON profile |
| `--units` | Include a per-measurement units block in each record. Constant per signal_set_version and identical on every record, so off by default; `hif schema` prints the same information without running a model. |
| `--truncate` | Truncate input to N tokens before analysis. Results reflect truncated context only. |
| `--surrogate` | Recover the input-side measurements (input_entropy_shift_bits, input_entropy_std_bits, prompt_surprisal_excess_bits) on backends that cannot teacher-force (ollama, openai, gemini, anthropic) by teacher-forcing a small local proxy model over the prompt+output — the same technique the study harness uses. Ignored when the target backend already teacher-forces (hf/tlens). Implied by --surrogate-model, so passing that alone is enough. |
| `--surrogate-model` | Open-weight HF model id used for --surrogate (default: Llama 3.2 1B, ungated mirror). Passing this flag implies --surrogate — you don't need both. |

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

## What `hif doctor` checks

`doctor` takes no flags. It reports, in order:

| check | what it tells you |
| --- | --- |
| core (numpy, plotly) | the two hard dependencies |
| embedder (sentence-transformers) | needed by every geometric measurement |
| charts (--charts) | HTML always; PNG only with kaleido installed |
| ollama server | reachability and which models are pulled |
| per-backend readiness | optional deps and credentials, one row per backend |
