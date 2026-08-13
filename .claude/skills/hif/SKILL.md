---
name: hif
description: Run and configure the hif CLI for distributional measurement of language models — author and verify run.toml config files, propose perturbation and trajectory options for the researcher to confirm, manage researcher-authored variant workload files, pick an acquisition tier, and read the resulting records. Use when the user mentions hif, distributional measurement, perturbation variants, trajectory branches, counterfactual exposure, entropy traces, output distributions, "profile a model", or a hif run.toml.
---

# hif

A CLI for distributional measurement of language models. Every measurement is
reported in its natural unit — bits, cosine distance, Pearson *r*, a fraction of
steps. Nothing is normalised to `[0,1]`, scored, or compared against a threshold.

## Your role: propose and confirm — never decide

The researcher must be able to scrutinize every measurement choice without
trusting you. So the division of labor is fixed:

- **You** translate intent into candidate configurations and explain tradeoffs.
- **The researcher** chooses.
- **The CLI** is the only authority on what will run and what ran. Your
  confirmations quote CLI output (`hif config show`, the record's `run_config`
  block) — never your own paraphrase of what you wrote.

Never run a profile with a config the researcher has not seen resolved. Never
present your summary of a config as equivalent to `hif config show` output.

## The workflow

**1. Elicit intent in measurement terms.** Before writing any config, establish:

- What question is the run asking? (sensitivity to rephrasing? output
  stability? exposure to divergent alternatives?)
- Which perturbation dimension matters — lexis (`synonym`, `substitution`),
  syntax (`reorder`), register (`tone`), determinacy (`ambiguity`) — **or does
  the researcher want to author the variants themselves** (a workload JSONL
  with `variants` — always offer this; it is the strongest form of control)?
- May the run send authored text to the model, or elicit output nobody asked
  for? (→ acquisition tier)
- What generation length? (runs of different `max_new_tokens` are not
  comparable)

**2. Propose 2–3 candidate configs**, each as complete TOML with a one-line
tradeoff. Start from the canonical template (`hif config init`), never from
memory. Example shape:

> **A — lexical only, tool-authored:** `generators = ["synonym",
> "substitution"]`, `n_variants = 4`. Cheap, deterministic; measures lexical
> sensitivity only.
> **B — researcher-authored:** a workload JSONL with `variants` — you write
> every paraphrase; nothing the tool invents touches the measurement. I can
> draft candidate rows, but each one is yours to edit before any run.
> **C — full default sweep:** all five families. Broadest, least
> interpretable per-family.

**3. Confirm with the CLI, not with prose.** After writing the chosen file:

```bash
hif config show --config-file run.toml --diff
```

Show the researcher this output — the departures from defaults ARE the
experimental condition. A typo'd key exits 3; a key you set that does not
appear did not apply. Only proceed on their confirmation.

**4. Run, then close the loop with the record.**

```bash
hif profile <model> "<prompt>" --config-file run.toml --json
```

The record embeds a `run_config` block (`record-v7`) — the same dict `config
show` printed. When reporting results, cite it: "the record's `run_config`
confirms `distance_threshold = 0.25`". Offer `--variant-io` when perturbation
is in play: it adds each variant's input and elicited continuation to the
record, so the elicited content is reviewable. Inputs stay immutable —
outputs live in records, never written back into an input file.

## Two rules that govern all reporting

1. **Absence is not zero.** A measurement missing from a record was not taken.
   Never report it as `0`, never substitute, never infer.
2. **Configuration is part of the measurement.** Numbers from different
   `max_new_tokens`, encoders, generator sets, thresholds, or access tiers are
   different measurements wearing the same key — do not compare them. Commit
   `run.toml` (and any variants JSONL) next to the outputs.

## Orientation commands

```bash
hif doctor     # what is installed, reachable, missing
hif models     # backends, and which measurements each can produce
hif schema     # every measurement: unit, definition, subject, acquisition
hif config init                    # canonical template, every key at its default
hif config show [...] --diff       # resolved config, without running
```

`hif schema --json` is the machine-readable contract — read it rather than
guessing key names or units.

## The two control axes

**Acquisition — what the run may bring into existence** (`--acquisition`):

| tier | permits |
| --- | --- |
| `observational` | The prompt as given, the one continuation the run produces. Nothing else reaches the model. |
| `synthesized-input` | + paraphrased prompts, teacher-forced only — the model does not generate |
| `elicited-output` *(default)* | + model-generated variant continuations and trajectory branches |

Propose `observational` when profiling hosted models where sending authored
text or generating unreviewed output raises cost, privacy, or terms questions.
Tiers are nested; surviving values are identical; capped measurements are
absent, not zero. Each measurement's tier is in `hif schema`.

**Stage budget — how much work within that permission**: `--lite` skips
perturbation, trajectory, semantic geometry, exposure (speed knob, composes
with `--acquisition`). `--mode fast|audit` changes only the variant budget.

## Scale

`profile` (one case) and `batch` (many, model loaded once) take the same
`--config-file`, `--mode`, `--acquisition`, `--lite`, and `--variant-io`. Past
one prompt, propose `batch` — a workload JSONL is also where authored
`variants` live.

The built-in prompt suite is a row source, not a command:
`hif batch --sample-set all <model>`, or `--sample-set <regime>` for one. It is
fixed on purpose: identical stimuli are the condition for a cross-model
comparison being a comparison. It is not a benchmark and not where the
researcher's own question lives. When they want their own prompts, offer:

```bash
hif batch --sample-set all --export-workload suite.jsonl   # 40 rows, no model
```

Then they edit it — add prompts, add per-row `variants` — and run `hif batch`.

## Which knob moves which measurement

- `[perturbation] generators` / `variants_file` → `perturbation_jsd_bits` and
  the input-side pair. A default-set 0.6 and a `reorder`-only 0.6 are
  different findings.
- `[generation] max_new_tokens` → every step-series average. The quietest
  comparability trap.
- `[embedding] model_name` → `io_cosine_similarity`. Cosine values compare
  only within one encoder.
- `[trajectory]` and `[exposure]` → no measurement. Both stages still run and
  still record their blocks — trajectory on any backend that can teacher-force,
  exposure by default and off only under `--lite`. Neither is behind
  `--diagnostics`, which sets exactly `[attention] enabled` and
  `[semantic_field] enabled`. The rows that reported them
  (`branch_pairwise_cosine_similarity`, `counterfactual_exposure_fraction`)
  were cut in hif-v4, the latter because its two thresholds were embedded in
  the number it reported.

Full reference: `docs/CONFIG.md`. Do not restate it — point to it.

## Access tiers

- `[F]` open weights — everything computable.
- `[T-k]` closed API with top-k logprobs — approximate where mass falls
  outside the window.
- `[P]` distributions withheld — `--surrogate` reads the output text through a
  disclosed local proxy. The distributions belong to the **proxy**; say so
  whenever reporting them.

Check `hif models --json` before promising a number on a given backend.

## What not to do

- No verdicts: never call a value good, bad, safe, aligned, or better than
  another model's. There are no thresholds, by design.
- No cross-configuration comparisons presented as model comparisons.
- No filling absent measurements.
- No claims that any measurement predicts anything — nothing here is validated
  against an outcome.
- Never author a variants file and run it without the researcher reviewing the
  rows — drafting is help; running unreviewed drafts is substituting your
  judgment for theirs.
- Never write model output back into an input file. Outputs are records.
