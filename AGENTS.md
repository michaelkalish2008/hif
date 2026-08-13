# AGENTS.md

Instructions for coding agents working in or with this repository. Claude Code
users get the same material as a skill at `.claude/skills/hif/SKILL.md`.

`hif` is a CLI for distributional measurement of language models. Every
measurement is reported in its natural unit — bits, cosine distance, Pearson
*r*, a fraction of steps. Nothing is normalised to `[0,1]`, scored, or compared
against a threshold.

## Two rules that govern everything

1. **Absence is not zero.** A measurement missing from a record was not taken.
   Never report it as `0`, never substitute an average, and never infer it from
   another measurement.
2. **Configuration is part of the measurement.** Three measurements are
   comparisons against runs the tool constructs, so their numbers mean nothing
   without the settings. The record carries them (`record-v7`'s `run_config`
   block) — cite that, not your memory. Numbers taken under different
   `max_new_tokens`, encoders, generator sets, thresholds, or access tiers are
   different measurements wearing the same key; do not compare them.

## Orientation

```bash
hif doctor     # what is installed, reachable, missing
hif models     # backends, and which measurements each can produce
hif schema     # every measurement: unit, definition, subject, acquisition
```

## Scale

`profile` (one case) and `batch` (many, model loaded once) take the same
`--config-file`, `--mode`, `--acquisition`, `--lite`, and `--variant-io`.
Prefer `batch` for anything past one prompt.

The built-in prompt suite is a row source, not a command:
`hif batch --sample-set all <model>` (or `--sample-set <regime>`). It is fixed
on purpose — identical stimuli are what make a cross-model comparison a
comparison — but it is not a benchmark and not where a researcher's own
question lives. Offer
`hif batch --sample-set all --export-workload suite.jsonl` as the starting
point for their own workload: they edit it, add per-row `variants`, run it
back.

`hif schema --json` is the machine-readable contract. Read it instead of
guessing key names or units.

## Acquisition — what a run may bring into existence

The distinction between measurements that read the I/O you already have and
measurements that make the model produce new content. Cap it with
`--acquisition`:

- `observational` — the prompt as given and the one continuation the run
  produces. Nothing else reaches the model; no new model output exists
  afterwards.
- `synthesized-input` — additionally authors paraphrased prompts and
  teacher-forces over them. The model does not generate.
- `elicited-output` *(default)* — additionally lets the model generate variant
  continuations and trajectory branches.

Tiers are strictly nested; surviving values are identical across them.
Measurements above the ceiling are absent, not zero. Each row's tier is in
`hif schema`.

`--lite` is a separate axis: a speed budget, not a content policy. They compose.

## Config files

Tables mirror the run config; full reference in [`docs/CONFIG.md`](docs/CONFIG.md).

```bash
hif profile Qwen/Qwen3-0.6B-Base "Explain why the sky appears blue." --config-file run.toml --json
```

Key knobs: `[perturbation] generators` and `n_variants` (behind three of the
six measurements — `perturbation_jsd_bits` and the input-side pair),
`[generation] max_new_tokens` (changes every step-series average), and
`[embedding] model_name` (behind `io_cosine_similarity`). `[trajectory]` and
`[exposure]` shape diagnostic blocks only; hif-v4 publishes no measurement
from either stage.

## Verify with the CLI, not with prose

The researcher must never have to trust your paraphrase of a config. The
authoring loop:

```bash
hif config init                                # canonical template
hif config show --config-file run.toml --diff  # resolved config, before running
hif profile <model> "<prompt>" --config-file run.toml --json
```

A mistyped key anywhere in the file — table or inner key — exits 3 with the
valid alternatives named. `config show` resolves through the same path
`profile` executes, so what it prints is what runs. The `--json` record
(`record-v7`) embeds a `run_config` block — the resolved config, secrets
redacted — so cite the record, not your memory, when reporting what ran.

Propose 2–3 candidate configs with tradeoffs and let the researcher choose;
show them the `config show --diff` output before any run.

## Researcher-authored perturbations

Always offer authored variants: the researcher writes every paraphrase and the
tool authors nothing.

One row format for all case data — the workload JSONL `hif batch` profiles,
with a `variants` list:

```jsonl
{"query_id": "q1", "text": "<prompt>", "variants": ["<paraphrase>", ...]}
```

`hif batch` reads those rows directly; a single `hif profile` run reaches the
same file via `[perturbation] variants_file`. Rows match the prompt by exact
`text` equality; where variants apply they replace the generators entirely.
No usable rows is a hard error. Omitting `variants` means "use the
generators"; an explicit `[]` is rejected.

`--variant-io` adds a `variant_io` block to the record (each variant's input
and elicited continuation, `null` where none was elicited). Inputs stay
immutable; outputs live in records — never write model output back into an
input file.

You may draft candidate rows, but never run a variants file the researcher has
not reviewed.

## Reporting rules

- No verdicts. Never call a value good, bad, safe, aligned, or better than
  another model's. There are no thresholds, by design.
- No cross-run comparisons across different `max_new_tokens`, encoders,
  perturbation generator sets, or access tiers.
- `prompt_measurements`, when present, is computed from the prompt alone under a
  reference model. It is not a measurement of the target.
- Under `--surrogate` (`[P]` tier) the distributions belong to the proxy, not
  the target. Say so.
- Nothing here has been validated against an outcome. Do not claim a measurement
  predicts anything.

## Working on the code

- The measurement registry (`hif/profile/registry.py`) is the single extension
  point. Adding a measurement means adding one row there, plus how it is taken
  in `hif/profile/measure.py` and reported in `hif/profile/record.py`.
- The registry imports nothing from the pipeline, on purpose — a reader can
  check a row's claims against the code without the code being able to change
  what the row says. Keep it that way; hold the two together with tests.
- Run `python -m pytest tests/unit -q` before proposing a change.
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the significance gate a new
  measurement has to clear.
