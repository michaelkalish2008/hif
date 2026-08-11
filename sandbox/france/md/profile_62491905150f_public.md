# Behavioural measurements — gpt2

**Prompt:** What is the capital of France?
**Regime:** ordinary_conversation

---

## What was measured

| Measurement | Value | Unit |
|---|---|---|
| Input entropy shift (bits) | absent | bits |
| Input entropy shift spread (bits) | absent | bits |
| Perturbation JSD (bits) | absent | bits |
| Input/output cosine similarity | absent | dimensionless |
| Prompt surprisal excess (bits) | 0.4354 | bits |
| Output entropy (bits) | 2.75101 | bits |
| Output nucleus entropy (bits) | absent | bits |

Absent means this run produced no evidence for that quantity — the
backend could not teacher-force, or an optional analysis stage did not
run. It does not mean zero.

## What this is not

These numbers describe what the model did on one prompt at one moment.
They are not a drift detection, an attack detection, or a quality score,
and none of them carries a threshold above or below which something is
wrong.
