# hif

Measure what a language model *does* — its output distributions, how they move
under perturbation, and how much of the possibility space stays live — and report
every measurement in its natural unit.

Nothing is normalised to `[0,1]`, inverted into a score, or compared against a
threshold. A number is in bits, or cosine distance, or Pearson *r*, or a fraction
of steps, and it says which.

```bash
hif profile gpt2 "Explain why the sky appears blue." --json
```

```json
{
  "schema_version": "record-v6",
  "model": "gpt2",
  "backend": "hf",
  "regime": "ordinary_conversation",
  "measurements": {
    "input_entropy_shift_bits": 0.5453,
    "input_entropy_std_bits": 0.4759,
    "perturbation_jsd_bits": 0.5999,
    "io_cosine_similarity": 0.2054,
    "prompt_surprisal_excess_bits": 0.6556,
    "output_entropy_bits": 2.526
  }
}
```

Real output, rounded for width — `gpt2` on CPU, no API key, no network.

## Three of those numbers are procedures, and you control the procedure

Most measurements above read one forward pass. Three do not — they compare the
baseline run against runs the tool constructs, so the number means nothing until
you know what was constructed. All three are configured through a TOML file
passed to `--config-file`; its tables mirror the run config, and any CLI flag you
type explicitly beats the file.

`perturbation_jsd_bits` is the mean Jensen-Shannon divergence between the
baseline output distribution and each **paraphrase** of the prompt. The default
is three rule-based generators — `synonym`, `tone`, `reorder` — at two variants
each, so six paraphrases. `substitution` and `ambiguity` are implemented and
selectable. Rule-based generators are the default because they cost nothing and
are deterministic; LLM-backed paraphrasing is opt-in and needs an endpoint you
supply.

`branch_pairwise_cosine_similarity` re-samples the generation from a branch point
— **five branches, ten rollout steps** by default — embeds each branch's text,
and averages the cosine similarity over every pair.

`counterfactual_exposure_fraction` is the fraction of analysed steps that were
both in the diffusion zone and had an alternative token **probable enough**
(`min_prob`, default `0.01`) and **semantically far enough**
(`distance_threshold`, default `0.3` cosine) to count as exposure. Both
thresholds are choices, and the fraction moves when you change them.

```toml
# run.toml
[perturbation]
generators = ["synonym", "tone", "substitution", "ambiguity"]
n_variants = 4
# use_llm_perturbation = true    # then set llm_base_url / llm_api_key / llm_model

[trajectory]
n_branches = 8
rollout_steps = 16

[exposure]
min_prob = 0.02
distance_threshold = 0.25
```

```bash
hif profile gpt2 "Explain why the sky appears blue." --config-file run.toml --verbose
```

`--mode audit` raises the perturbation variant count without a file;
`--mode fast` (the default) lowers it. `--verbose` prints the paraphrase variants
the run actually used, which is the only way to see the six strings the default
compared against.

Because the configuration is part of the measurement, the record carries it:
every `--json` record (`record-v6`) embeds a `run_config` block — the resolved
configuration the run executed, secrets redacted. Two runs that differ only in
`distance_threshold` now say so in the records themselves.

The authoring loop is three commands:

```bash
hif config init                                # run.toml, every key at its default
hif config show --config-file run.toml --diff  # what will actually run, before it runs
hif profile gpt2 "..." --config-file run.toml --json
```

`config show` resolves through the same path `profile` executes, so the two
cannot drift, and a mistyped key anywhere in the file exits 3 rather than
silently measuring with defaults.

You can also author the perturbations yourself, so the tool writes no text at
all. Case data travels in one format — the workload JSONL `hif batch` already
profiles — with a `variants` list added:

```jsonl
{"query_id": "sky_1", "text": "Explain why the sky appears blue.", "variants": ["Explain why the sky looks blue."]}
```

`hif batch` profiles those rows directly; a single `hif profile` run reaches
the same file through `[perturbation] variants_file`. Where variants apply
they replace the generators entirely. Add `--variant-io` to put each
variant's input and elicited continuation in the record — inputs stay
immutable, outputs live in records. See [`docs/CONFIG.md`](docs/CONFIG.md).

### `--acquisition`, when it matters what the run produces

Those stages differ in a way the record used to hide: some measurements read the
prompt and the one continuation you asked for, and others make the model
generate text nobody asked for and nobody will read. `--acquisition` caps what a
run is permitted to bring into existence.

| tier | permits | adds |
| --- | --- | --- |
| `observational` | The prompt as given, the one continuation the run produces. Nothing else reaches the model. | the five entropy-side measurements, plus `candidate_cluster_entropy_bits` and `counterfactual_exposure_fraction` |
| `synthesized-input` | Additionally authors paraphrased prompts and teacher-forces over them. The model does not generate. | `input_entropy_shift_bits`, `input_entropy_std_bits` |
| `elicited-output` *(default)* | Additionally lets the model generate variant continuations and trajectory branches. | `perturbation_jsd_bits`, `io_correlation_r`, `io_cosine_similarity`, `branch_pairwise_cosine_similarity` |

```bash
hif profile gpt2 "Explain why the sky appears blue." --acquisition observational --json
```

Use `observational` when profiling a hosted model and sending authored
paraphrases — or generating output nobody reviews — is a cost, privacy, or terms
question. The tiers are strictly nested, surviving values are identical across
them, and every row in `hif schema` carries its `acquisition`, so the partition
is machine-readable rather than something to reconstruct from the source.

`--acquisition` is a content policy; `--lite` is a speed budget. They compose.

### `--lite`, when you only want the entropy side

Those three stages are also where the time goes: the paraphrases cost one
generation pass each, the branches cost five more, and the semantic stage embeds
every candidate at every step. `--lite` skips all of it.

```bash
hif profile gpt2 "Explain why the sky appears blue." --json --lite
```

```
              full     --lite
pipeline     11.4s       1.3s     (gpt2, 16 new tokens, CPU)
```

What comes back is the single baseline pass plus input-side teacher forcing:

| survives `--lite` | omitted under `--lite` |
| --- | --- |
| `output_entropy_bits` | `perturbation_jsd_bits` |
| `output_entropy_step_delta_bits` | `input_entropy_shift_bits` |
| `output_step_jsd_bits` | `input_entropy_std_bits` |
| `output_step_topk_overlap_fraction` | `io_correlation_r` |
| `prompt_surprisal_excess_bits` | `io_cosine_similarity` |
| | `candidate_cluster_entropy_bits` |
| | `counterfactual_exposure_fraction` |
| | `branch_pairwise_cosine_similarity` |

The surviving values are **identical** to what the same run reports without the
flag — `--lite` removes stages, it does not approximate them. The omitted ones
are absent from the record rather than reported as `0.0`, which is the same
convention every other unavailable measurement follows: absence means not
measured, and `0` always means measured zero.

`--lite` overrides `--mode` and `--config-file` for the stages it disables, so a
run asking for less never silently does more.

To print a single measurement rather than the whole record, `--metric` selects
what is *shown*; `--lite` selects what is *computed*. They compose:

```bash
hif profile gpt2 "Explain why the sky appears blue." --lite --metric output_entropy_bits
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python 3.10+. `pip install -e '.[hf]'` adds torch and transformers for local
weights; `[openai]`, `[anthropic]`, `[gemini]`, `[ollama]`, `[tlens]` add the
other backends. Bring your own weights or your own API key — there is no account,
no tier, and no hosted component.

```bash
hif doctor     # what is installed, what is reachable, what is missing
hif models     # backends, example models, and which signals each supports
hif schema     # every measurement, its unit, and its definition
```

`hif doctor` is the one to run first — it answers "will this work here?"
before a pipeline has a chance to fail halfway:

```
HIF doctor — environment & backend readiness

  core (numpy, plotly): ok
  embedder (sentence-transformers): ok
  charts (--charts): ok — HTML (PNG needs kaleido: pip install kaleido)
  ollama server (http://localhost:11434): not reachable — run `ollama serve`
```

## Credentials

Open-weight backends (`hf`, `tlens`, `ollama`) need no credentials at all. The
hosted ones read a key from the environment: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, plus `HF_TOKEN` for gated weights.

Put them in a dotenv and hif will find it. The nearest `.env` at or above the
working directory is read first, then `~/.config/hif/.env` — so a project keeps
its own keys, and an install with no project directory still has one place to
put them:

```bash
mkdir -p ~/.config/hif && printf 'OPENAI_API_KEY=sk-...\n' >> ~/.config/hif/.env
```

`--env-file path/to/.env` names one explicitly, ahead of anything discovered.
**A variable already exported always wins** — a dotenv only ever fills a gap, so
`OPENAI_API_KEY=… hif …` and a CI environment are never silently overridden.

`hif doctor` prints where each credential came from — the source, never the
value:

```
credentials
  OPENAI_API_KEY: set (.env)
  ANTHROPIC_API_KEY: set (.env)
  GEMINI_API_KEY: set (.env)
```

If it says `unset` for a key
you believe you loaded, the usual cause is `source .env`: on bare `KEY=value`
lines that creates *shell* variables, and a child process inherits only the
*environment*, so hif never sees them. `set -a; source .env; set +a` exports them
— or just let hif read the file itself.

## Commands

| command | what it does |
|---|---|
| `hif profile <model> <prompt>` | full pipeline on one (model, prompt) pair |
| `hif batch <workload.jsonl> <model>` | every row of a workload, model loaded once |
| `hif batch --sample-set all <model>` | the same, over the built-in fixed stimulus set |
| `hif config init` / `hif config show` | author a run.toml; see what will actually run |
| `hif compare <a.json> <b.json>` | per-measurement difference between two profiles |
| `hif validate-model <model>` | region-sensitivity check against a known-answer suite |
| `hif render <profile.json>` | re-render Markdown from an existing profile |

Both scales take the **same** `--config-file`, `--mode`, `--acquisition`,
`--lite`, and `--variant-io`, resolved through one code path — so a ceiling
means the same thing whether you run one prompt or forty, and a corpus is
comparable with the single runs it aggregates.

The built-in prompt suite is a **row source**, not a separate command:
`--sample-set all` (or a single regime name) feeds `batch` the same 8 x 5
fixed stimulus set, and inherits every control above. It is fixed on purpose —
a cross-model comparison is only a comparison when the stimulus was identical
— but it is not a benchmark (unlabeled prompts, nothing scored), and it is not
where your own question lives. Fork it:

```bash
hif batch --sample-set all --export-workload suite.jsonl   # 40 rows, no model
$EDITOR suite.jsonl                                        # add prompts, add `variants`
hif batch suite.jsonl gpt2
```

A workload run streams one record per row. Two prompts, `--lite`, trimmed to
the measurements for width:

```bash
hif batch workload.jsonl gpt2 --lite --json 2>/dev/null | jq -c
```

```json
{"query_id": "sky", "model": "gpt2", "measurements": {"prompt_surprisal_excess_bits": 0.6556, "output_entropy_bits": 2.3932, "output_entropy_step_delta_bits": 1.9093, "output_step_jsd_bits": 0.8475, "output_step_topk_overlap_fraction": 0.1023}}
{"query_id": "greet", "model": "gpt2", "measurements": {"prompt_surprisal_excess_bits": 0.631, "output_entropy_bits": 2.5541, "output_entropy_step_delta_bits": 2.013, "output_step_jsd_bits": 0.903, "output_step_topk_overlap_fraction": 0.0977}}
```

Note what is *absent*: `--lite` skipped the perturbation, trajectory and
geometric stages, so those keys are not in the record at all rather than
present and zero.

Before a configured run, `hif config show --diff` prints what will actually
happen — the departures from the defaults are the experimental condition:

```
# resolved run config — departures from defaults only
# gpt2 (hf) · mode=fast · --config-file run.toml

[perturbation]
n_variants = 4
generators = ["synonym", "substitution"]

[exposure]
distance_threshold = 0.25
```

**stdout carries JSON and nothing else.** `profile --json`, `compare --json`,
`models --json` and `schema` emit a single document; `suite` and `batch` emit
JSONL, one record per prompt. Progress, warnings, and errors go to stderr, so
this works:

```bash
hif batch workload.jsonl gpt2 2>/dev/null | jq '.measurements.output_entropy_bits'
```

A failed row is still a record — it carries an `error` key instead of
`measurements`, so one bad prompt does not lose the run.

To see every model you can pass to `profile`, with the backend each one needs:

```bash
hif models --json 2>/dev/null | jq -r '.backends[] | .name as $b | .models[] | "\($b)\t\(.)"'
```

Those are worked examples, not a catalogue — any HuggingFace repo id is eligible
on `hf`/`tlens`. Add `--list` to query each provider's live catalogue instead
(needs that provider's key, or a running Ollama server); each backend then
reports `models_source` as `live` or `examples`, so a provider that could not be
reached is visible rather than silently thin.

Units are **not** in the record by default. They are constant per
`signal_set_version` and would repeat verbatim on every JSONL line, so pass
`--units` to `profile` or `batch` when you want records that describe
themselves, or run `hif schema` to print every measurement with its unit and
definition without touching a model.

## Charts

Nothing is written to disk unless you ask. `--charts` (with `--output-dir`)
renders one interactive Plotly HTML per signal plus an `index.html` dashboard
that embeds them, grouped into **Aggregate views** and **Per-step views**:

```bash
hif profile gpt2 "Explain why the sky appears blue." \
  --charts --output-dir out --diagnostics
```

Thirteen charts, one per entry in `hif/viz/registry.py` — the same registry the
measurement table joins to, so a chart and its number are one arithmetic rather
than two:

| Aggregate views | Per-step views |
|---|---|
| `stability`, `sensitivity`, `continuity` | `entropy`, `shift`, `wager` |
| `io_correlation`, `similarity` | `spread`, `horizon`, `exposure` |
| `breadth`, `surprise` | |

A signal whose backing data is missing renders an explicit *"requires teacher
forcing / attention capture / …"* placeholder rather than a flat or zero chart —
the same absence-is-not-zero rule the records follow. `spread` and `horizon`
need `--diagnostics` (they read the DistilBERT attention capture); without it
they render as unavailable rather than empty.

HTML needs only plotly, which is a core dependency. PNG output additionally
needs `kaleido`, and plotly imports it inside `write_image()` — so without it a
PNG run fails *after* the whole pipeline has completed. `hif doctor` reports
both, up front:

```
  charts (--charts): ok — HTML (PNG needs kaleido: pip install kaleido)
```

## What you can measure depends on the backend

| access | backends | what you get |
|---|---|---|
| `[F]` full | `hf`, `tlens` | full-vocabulary distributions, teacher forcing, attention — every measurement |
| `[T-k]` truncated | `openai`, `gemini` (flash), `deepseek` | top-k logprobs only; entropy is a lower bound, no teacher forcing |
| `[P]` proxy | `anthropic`, `gemini` (pro), text-only APIs | output text only; distributional measurements unavailable |

Run `hif models` for the authoritative per-backend list — it names, per
backend, exactly which measurements it can and cannot produce:

```
hf  (local-open)  teacher-forcing: yes  ·  logprobs: full
  deps:  torch, transformers (base install)
  setup: none (HF_TOKEN only for gated repos); weights auto-download
  models: gpt2, distilgpt2, gpt2-medium, EleutherAI/pythia-160m, …
  Full fidelity — every measurement. Best for a complete profile.
  ✓ signals: attention_entropy_input_bits, branch_pairwise_cosine_similarity,
    counterfactual_exposure_fraction, io_correlation_r, output_entropy_bits, …
```

Measurements a backend cannot support are **absent from the record with a
stated reason** — never zero, never a default, never silently borrowed from
elsewhere.

On closed backends, `--surrogate` recovers some input-side quantities by
teacher-forcing a small local model over the *prompt* — which means those
numbers describe the prompt under a reference model, not the model you asked
about, and nothing the target did enters them. They are reported in a separate
`prompt_measurements` block naming that reference model, never inside
`measurements`, because a caveat flag would still read as a fact about your
model. `hif schema` gives every measurement's subject; docs/MEASUREMENTS.md
§ Subject gives the rule.

A fair objection: behavioural measurement of a closed model is of limited
value. Conceded — but on a closed model the API response is the entire
observable surface (no weights, no full logits, no attention, no teacher
forcing), so reading that surface is not the preferred method there, it is the
only one that exists, and the `[T-k]` and `[P]` tiers are honest inventories of
how little it exposes. This is a stopgap by design: every measurement becomes
exact on open weights, and if providers expose more, measurements should
migrate up the tiers and the proxy tier should shrink toward empty. The
limitation lives in provider opacity, not in the method.

## Scope and honesty

This instrument **describes** behaviour.

Ask it for a single number and it hands back the number, its unit, and whose
behaviour it is about — and stops there:

```bash
hif profile gpt2 "Explain why the sky appears blue." --metric output_entropy_bits
```

```
output_entropy_bits = 2.47248
bits — mean Shannon entropy of the per-step top-K output distribution. A
lower bound on full-vocabulary entropy when the distribution is truncated.
subject: target-distribution
```

There is no grade attached, because there is no scale to grade against: nothing
is normalised to `[0,1]`, inverted into a score, or compared against a
threshold. 2.47 bits is roughly the uncertainty of a uniform choice among five
or six tokens, which is checkable; `0.31` on some index would not be.

Interpretation is the researcher's, and it belongs in the work that cites this
tool rather than in the tool. That division is the point: a number you can check
is worth more than a verdict you have to trust.

Known limitations, stated plainly:

- **The signal set is smaller than it looks.** Effective dimensionality across the
  measurements is roughly 3. Several are correlated; treating them as independent
  evidence will mislead you.
- **On `[T-k]` and `[P]` backends, some quantities describe a local surrogate
  rather than the model you asked about.** They are computed from prompt text and
  a local reader, and cannot observe a hosted model's internals — they return
  the same value whichever model you profiled. They are reported separately from
  the measurement set on those backends, with their reference model named. Check
  `hif models` before drawing conclusions about an API model.
- **`output_entropy_bits` is a lower bound** whenever the distribution is
  truncated to top-k, and is not comparable across backends with different k.
- **No thresholds, no levels, no verdicts.** Deciding what a value *means*
  requires a baseline you establish yourself, on your own models and prompts —
  including measuring what the number does when nothing has changed. That
  judgement is deliberately out of scope here — it depends on your models, your
  prompts, and what you are asking, none of which this tool can know.

## Documentation

- [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) — every measurement as observable × functional × resolution: the run-level scalars, the token-level traces, the components, and the field descriptors
- [`docs/CONFIG.md`](docs/CONFIG.md) — every config key and what it moves; how a run config is assembled and how to verify it applied
- [`docs/PROMPT_SUITE.md`](docs/PROMPT_SUITE.md) — the prompt regimes; an unlabeled dataset, not a benchmark
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — model roles, module layout, data flow
- [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) — why read behaviour distributionally at all

Profiles generated with this tool are published and explorable at
[ai-interpretability.com](https://ai-interpretability.com).

### Driving hif from a coding agent

`.claude/skills/hif/SKILL.md` is a skill for Claude Code: how to author and
**verify** a `run.toml`, which knob moves which measurement, how to choose an
acquisition tier, and the reporting rules (absence is not zero; no thresholds,
no verdicts). It loads automatically when you work inside this repo.

To use it from another project, copy or symlink it:

```bash
mkdir -p ~/.claude/skills
ln -s "$PWD/.claude/skills/hif" ~/.claude/skills/hif
```

Codex and other agents that read `AGENTS.md` get the same rules from
[`AGENTS.md`](AGENTS.md) at the repository root. Both point at `docs/CONFIG.md`
as the reference; neither restates it.

## Contributing

The one contribution path is **adding a measurement**, and it is deliberately
small: compute the quantity in natural units, declare its triple, check it
passes the Significance Gate, add one row to the registry in
`hif/profile/registry.py`, add a test. [`CONTRIBUTING.md`](CONTRIBUTING.md)
walks through all five steps with a worked example.

## License

MIT (relicensed from the source project by its owner).
