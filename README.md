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
  "schema_version": "record-v3",
  "model": "gpt2",
  "backend": "hf",
  "regime": "ordinary_conversation",
  "measurements": {
    "input_entropy_shift_bits": 0.5453,
    "input_entropy_std_bits": 0.4759,
    "perturbation_jsd_bits": 0.5999,
    "io_correlation_r": 0.6318,
    "io_cosine_similarity": 0.2054,
    "prompt_surprisal_excess_bits": 0.6556,
    "candidate_cluster_entropy_bits": 0.4605,
    "output_entropy_bits": 2.526,
    "output_entropy_step_delta_bits": 1.403,
    "counterfactual_exposure_fraction": 0.0833,
    "branch_pairwise_cosine_similarity": 0.099
  }
}
```

Real output, rounded for width — `gpt2` on CPU, no API key, no network.

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

## Commands

| command | what it does |
|---|---|
| `hif profile <model> <prompt>` | full pipeline on one (model, prompt) pair |
| `hif suite <model>` | the same across every prompt regime |
| `hif batch <workload.jsonl> <model>` | every row of a workload, model loaded once |
| `hif compare <a.json> <b.json>` | per-measurement difference between two profiles |
| `hif validate-model <model>` | region-sensitivity check against a known-answer suite |
| `hif render <profile.json>` | re-render Markdown from an existing profile |

**stdout carries JSON and nothing else.** `profile --json` and `compare --json`
emit a single document; `suite` and `batch` emit JSONL, one record per prompt.
Progress, warnings, and errors go to stderr, so this works:

```bash
hif batch workload.jsonl gpt2 2>/dev/null | jq '.measurements.output_entropy_bits'
```

A failed row is still a record — it carries an `error` key instead of
`measurements`, so one bad prompt does not lose the run.

Units are **not** in the record by default. They are constant per
`signal_set_version` and would repeat verbatim on every JSONL line, so pass
`--units` to `profile`, `suite`, or `batch` when you want records that describe
themselves, or run `hif schema` to print every measurement with its unit and
definition without touching a model.

## What you can measure depends on the backend

| access | backends | what you get |
|---|---|---|
| `[F]` full | `hf`, `tlens` | full-vocabulary distributions, teacher forcing, attention — every measurement |
| `[T-k]` truncated | `openai`, `gemini` (flash), `deepseek` | top-k logprobs only; entropy is a lower bound, no teacher forcing |
| `[P]` proxy | `anthropic`, `gemini` (pro), text-only APIs | output text only; distributional measurements unavailable |

Run `hif models` for the authoritative per-backend list. Measurements a backend
cannot support are **absent from the record with a stated reason** — never zero,
never a default, never silently borrowed from elsewhere.

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

This instrument **describes** behaviour. It does not detect drift, identify
attacks, or judge quality, and it should not be used as though it does.

An earlier version of this project made inferential claims on top of these
measurements — that shifts indicated model drift, that signatures identified
adversarial inputs. Those claims were withdrawn after an internal audit found
they did not survive scrutiny. The measurement layer survived; the interpretation
built on top of it did not. What remains is deliberately narrow.

Known limitations, stated plainly:

- **The signal set is smaller than it looks.** Effective dimensionality across the
  measurements is roughly 3. Several are correlated; treating them as independent
  evidence will mislead you.
- **On `[T-k]` and `[P]` backends, some measurements describe a local surrogate
  rather than the model you asked about.** They are computed from prompt text and
  a local reader, and cannot observe a hosted model's internals. Check
  `hif models` before drawing conclusions about an API model.
- **`output_entropy_bits` is a lower bound** whenever the distribution is
  truncated to top-k, and is not comparable across backends with different k.
- **No thresholds, no levels, no verdicts.** Deciding what a value *means*
  requires a baseline you establish yourself, on your own models and prompts —
  including measuring what the number does when nothing has changed. That
  judgement is out of scope here, deliberately, because getting it wrong is how
  the earlier version of this project went wrong.

## Documentation

- [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) — every measurement as observable × functional × resolution: the run-level scalars, the token-level traces, the components, and the field descriptors
- [`docs/PROMPT_SUITE.md`](docs/PROMPT_SUITE.md) — the prompt regimes; an unlabeled dataset, not a benchmark
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — model roles, module layout, data flow
- [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) — why read behaviour distributionally at all

Profiles generated with this tool are published and explorable at
[ai-interpretability.com](https://ai-interpretability.com).

## Contributing

The one contribution path is **adding a measurement**, and it is deliberately
small: compute the quantity in natural units, declare its triple, check it
passes the Significance Gate, add one row to the registry in
`hif/profile/signals.py`, add a test. [`CONTRIBUTING.md`](CONTRIBUTING.md)
walks through all five steps with a worked example.

## License

MIT (relicensed from the source project by its owner).
