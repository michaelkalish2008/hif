# Measurements — Horizonal Interpretability Framework (HIF)

## Derivation Scheme

Every HIF measurement is a **triple**: **observable × functional × resolution**.

- **Observable** — what the forward pass exposes: output distribution; input distribution (teacher forcing); attention row; logits. Gated by what the backend exposes (see Backend Access below).
- **Functional** — **information-theoretic** (entropy, surprisal, JSD, trace correlation) or **geometric** (embedding distance, cluster structure, silhouette). Two co-equal families. Entropy reads the *shape* of the distribution (concentration across the simplex), blind to identity. Geometry reads *where* the mass sits — which tokens the model seriously weighs and how far apart they are. Both are required.
- **Resolution** — the granularity of the underlying series a run-level scalar summarises. A declared field on every registry row, with three values:
  - **`aggregate`** — the quantity exists only at whole-run level (across perturbation variants, trajectory branches, or the run's endpoints); there is no per-token trace behind it.
  - **`per-step`** — one sample per generation step; the scalar summarises a per-step trace.
  - **`per-position`** — one sample per prompt/context position; the scalar summarises a per-position trace.

  The record always carries the run-level scalar (Part 1); measurements with `per-step` or `per-position` resolution additionally have a token-level trace that restores what the scalar compresses (Part 2).

There is one concept — a **measurement** — and resolution is a field on it, not a category of its own. Splitting aggregate and token-level quantities into separate vocabularies would be the resolution coordinate wearing two names, and two names for one axis is how a set starts double-counting itself.

## Acquisition — what taking the measurement brings into existence

The triple says what was measured. **Subject** says whose behaviour the number describes. Neither says what had to be *created* to get it, and that is a separate question with separate consequences.

Some measurements read the prompt as given and the one continuation the run already produced. Others require the tool to author prompt text the user never wrote, or to make the model generate continuations nobody asked for and nobody will read. Reporting both under one heading — which this document previously did — hides the difference between an observation and an elicitation.

Every registry row now declares an `acquisition`, and `hif schema` prints it:

- **`observational`** — computed from the prompt as given and the single continuation the run produced. Nothing is sent to the model beyond the one call the caller asked for, and no model output exists afterwards that did not exist before. Local instruments may construct strings — counterfactual exposure embeds `prefix + candidate` alternatives — but nothing constructed leaves the process or reaches a provider.
- **`synthesized-input`** — the tool **authors** new prompt text (paraphrase variants) and teacher-forces the model over it. The model does not generate.
- **`elicited-output`** — the tool makes the model **generate** text that did not exist before: variant continuations, trajectory branch rollouts. This is the tier that costs tokens, multiplies API calls, and produces unreviewed model output.

| acquisition | measurements |
| --- | --- |
| `observational` | `prompt_surprisal_excess_bits`, `output_entropy_bits` |
| `synthesized-input` | `input_entropy_shift_bits`, `input_entropy_std_bits` |
| `elicited-output` | `perturbation_jsd_bits`, `io_cosine_similarity` |

`--acquisition observational|synthesized-input|elicited-output` caps a run at one of these tiers; measurements above the ceiling are **absent, not zero**. The tiers are strictly nested and the surviving values are identical across them, so raising the ceiling only ever adds keys. See [CONFIG.md](CONFIG.md).

**The input-side pair was entangled with elicitation and no longer is.** `input_entropy_shift_bits` and `input_entropy_std_bits` difference teacher-forced entropies over paraphrased prompts; they read no variant continuation. The pipeline nonetheless generated one for every variant, because authoring and generating happened in the same loop. `[perturbation] elicit_variant_outputs = false` separates them, which is what the `synthesized-input` tier sets.

## Significance Gate — the bar for admitting a measurement

A computable triple is not automatically a measurement. **This gate is the acceptance criterion for contributing a new measurement** (see [CONTRIBUTING.md](../CONTRIBUTING.md)). Six conditions, all required — 1–2 admit a quantity at all, 3–6 are what removed ten rows in hif-v4:

1. **Derivability** — computable from the distributional observable alone, no inference to hidden structure.
2. **Distinct disclosure** — discloses a facet no admitted measurement already captures. Three tests, all required; see [How condition 2 is tested](#how-condition-2-is-tested) below for why "move independently somewhere" was not enough on its own.
   - **(a) Outside the span of the admitted set.** A quantity that is a deterministic function of admitted measurements — a difference, ratio, sum, or any fixed transform of them — discloses nothing by construction. It is rejected at the definition stage, before any measurement, because a consumer computes it from the record in one line.
   - **(b) Independence tested at the resolution the row reports.** A row reports a run-level scalar, so that is where the test binds. A per-step trace can show independent movement that the run-level mean does not.
   - **(c) Redundancy judged on reliability-corrected correlation.** Raw correlation understates redundancy when either series is noisy. A residual that is itself mostly measurement noise is not a disclosure.
3. **About the target** — the number must move when the target model changes. A quantity produced by a fixed reference instrument reading the prompt is bit-identical across targets and fails this by construction.
4. **Powered at the run's own n** — a statistic whose default sample size cannot distinguish its typical values from zero publishes noise. (`io_correlation_r` failed here: 69 of the 96 published corpus values sat below the significance floor of its own n=15.)
5. **No embedded thresholds** — a fraction of threshold-crossings is a verdict wearing a unit. Report the underlying quantity in its natural unit instead.
6. **Present where it claims to be** — a row absent from most of the corpus it was designed for is not carrying its weight; either the requirement is declared honestly or the row does not enter.

The *distinct disclosure* condition is why several plausible quantities are *not* in the set: `continuity` was `1 − sensitivity` computed from the same JS divergences, the historical `wager` aggregate was byte-for-byte the `surprise` aggregate, and ESS is a bijection of entropy (2^H). Each quantity appears exactly once.

### How condition 2 is tested

The condition previously read "must move independently **somewhere** across contexts." That wording is exploitable, and a candidate row exposed all three ways it fails. The candidate was the **nucleus entropy gap**, `entropy_bits − nucleus_entropy_bits`, proposed on the reasoning that two distributions with equal entropy can differ in how much of it sits outside the nucleus.

**(a) caught it with no experiment.** At run level the gap is *exactly* `output_entropy_bits − output_nucleus_entropy_bits` — the two run-level means agree with their difference to 4×10⁻¹⁶, because the mean is linear. It is a subtraction of two admitted rows. Everything below was measured only because this test did not exist yet; with clause (a) in the gate, none of it was necessary.

**(b) is why "somewhere" had to go.** Pooled over four open-weight models × eight prompts × 32 full-vocabulary steps, the gap's redundancy with entropy reads very differently at the two resolutions:

| resolution | R² with entropy |
|---|---|
| per-step trace | 0.26 |
| **run-level scalar (what the row would report)** | **0.66** |

The step-level number would have admitted the row under the old wording. It is also inflated by a mechanical branch: on 15.6% of steps the top token carries ≥ 0.95 of the mass, the nucleus collapses to one token, and the gap equals entropy *by construction*. Off that branch the step-level R² is 0.060 — the trace looks independent largely because it mixes an identity with noise.

**(c) is the correction to the naive reading.** The tempting conclusion from (b) — "averaging destroys independent variation" — is wrong, and the opposite of what happened. Disattenuated for reliability, run-level R² is 0.84–0.96, and the non-entropy residual has split-half reliability 0.35 with no prompt structure (F = 1.21, p = 0.33). The independence visible per-step was mostly noise, and averaging correctly removed it. Redundancy therefore has to be judged after correcting for reliability, not on raw correlation.

**A fourth observation, belonging to conditions 3 and 6.** On truncated backends the gap is not even the quantity it claims: `entropy_bits` reads the raw top-K slice while `nucleus_entropy_bits` renormalizes, so their difference is dominated by the missing mass. Across all 118 published corpus profiles — every one of them truncated — the gap is near-deterministic in fields already published (joint R² = 0.995 on mass, entropy and their interactions) and is *negative* in 41.5% of them, which the tail-structure story cannot explain. This is the same failure class as the `normalized` block below: an artifact of how much the backend returned, wearing a behavioural name.

## Natural units

Every measurement is reported in the unit it is measured in — bits for entropies and surprisals, dimensionless for correlations and cosine similarities, cosine distance for embedding displacement, a fraction for a count of steps. Key names carry the unit. Nothing here is normalised into `[0, 1]`, divided by a vocabulary size, or inverted into a `1 − x` score.

Three families of quantity were removed, and this document records why so they do not come back:

- **The `normalized` block.** Unbounded quantities were divided by `log₂(vocab_size)`. That normaliser then surfaced as the strongest apparent "behavioural" feature in the study corpus (r = 0.980, constant within a model) — tokenizer metadata masquerading as behaviour. Bounded scales also saturate, and bits are self-interpreting: 4.9 bits is about the uncertainty of a uniform choice among ~30 tokens, whereas "0.0178" is not checkable against anything.
- **The `levels` block (low/medium/high) and the verdict/equilibrium flags.** Assigning a level is an inference requiring a null distribution this project never established. The decision rule built on the previous levels measured a ~43% false-positive rate on pairs of runs known to be identical.
- **Duplicate names.** `continuity` was `1 − sensitivity` computed from the same JS divergences, and the `wager` aggregate was byte-for-byte the same computation as `surprise`. Reporting one measurement twice under two names inflates the apparent dimensionality of the signal set. Each quantity now appears exactly once in the measurement set.

Source of truth: `MEASUREMENT_REGISTRY` in `hif/profile/registry.py` — one row per measurement carrying its key, name, unit, definition, triple, and subject — printed in full by `hif schema`.

## Subject — whose behaviour the number describes

The triple says *what* was measured and at what granularity. It does not say *who it is about*, and for a while the record could not express that at all. Every row now declares a **subject**, and the distinction it draws is the one that decides whether a number belongs in the measurement set.

**The principle.** A proxy applied to the target model's actual input or output is a reading instrument on real data — legitimate. A proxy whose own behaviour stands in for the target's is a different subject wearing the target's name — not legitimate. The test is not "was a proxy involved" but "whose behaviour moves the number". A surrogate teacher-forced over text the target actually generated still moves when the target's output moves: it is a fact about the target, read indirectly. A surrogate teacher-forced over the prompt does not: nothing the target did enters it.

**The enum**, one line each (`hif schema` prints the same legend):

| Subject | Meaning |
|---|---|
| `target-distribution` | the target model's own probability distributions — its forward pass over its input or over its own generation |
| `target-output-text` | a fixed local instrument (embedder or analysis encoder) reading text the target actually generated |
| `mixed` | a target-derived series coupled with a series derived from something other than the target; the target participates but does not solely determine the number |
| `prompt-only` | the prompt text alone under a fixed reference model — no data the target produced enters |

**Subject is backend-dependent, and is modelled as such.** A row declares the subject it has when the target's own machinery produced the quantity, plus `subject_under_surrogate` — what that becomes once the surrogate named by its `surrogate_group` stands in. `effective_subject()` resolves the pair against the surrogates a given run actually used, so the answer is the run's rather than a static value that would be wrong on half the backends. On an `[F]` backend `prompt_surprisal_excess_bits` is `target-distribution`; on `[P]` with `--surrogate` the same key is `prompt-only`.

**The consequence: absent, not flagged.** When a measurement's computation never touches the target's data on the active backend, it is **omitted from `measurements`** and reported in a separate top-level `prompt_measurements` block carrying its own subject declaration and the reference model that produced each value. A flag would say "here is a caveated number about this model"; absence says "this model produced no number". Only the second is true. This is the absent-not-pinned rule extended from *cannot measure* to *measured something else*.

The prompt-only quantities are not worthless — "how surprising is this prompt under a fixed reference model" is a real question, and its answer is comparable across targets *precisely because* the target does not enter it. That is why they are reported rather than dropped, and why they are reported somewhere other than the model's measurement set.

**How to see it for yourself.** Profile any model twice — once on an open-weight backend, once on a hosted one with `--surrogate` — and compare `input_entropy_std_bits`:

| | subject | value |
|---|---|---|
| `Qwen/Qwen3-0.6B-Base` on `hf` | `target-distribution` | moves with the model |
| `gpt-4.1` with `--surrogate` | `prompt-only` | identical to every other hosted model on the same prompt |

Under the surrogate the number is the reference model reading the prompt, so it is bit-identical across targets — which is exactly what a quantity that cannot see the target looks like, and why it is reported in `prompt_measurements` rather than `measurements`.

That is the distinction `subject` records: not which tool computed a number, but whose data it was computed from. A quantity that returns the same value whichever model you profiled is measuring something other than that model. `tests/unit/test_zero_variance_canary.py` asserts this for every row, so a mislabelled subject fails without anyone reviewing the label.

**No row is prompt-only on every backend.** One was — `attention_entropy_input_bits`, a fixed encoder reading the prompt, bit-identical across all fifteen corpus models — and hif-v4 cut it on exactly that ground: a measurement set for models should not carry a quantity that is never about one. The canary now asserts the class stays empty.

This section is the companion to [Why measure behaviour at all on closed models](#why-measure-behaviour-at-all-on-closed-models) below. That one concedes how little a closed surface exposes and commits to reporting absence rather than approximation when the surface cannot support a quantity; this one draws the line the concession implies — a number produced by something other than the target is not a degraded reading of the target, and the record must not be able to say it is.

## Backend Access

Access is a property of what the backend exposes, not of the model. `hif/models/capabilities.py` is the enforcing authority; `hif models` prints the current table.

| Access | Backends | What is available |
|--------|----------|------------------|
| `[F]` full | `hf`, `tlens` | Full-vocabulary distributions and teacher forcing — every measurement |
| `[T-k]` truncated | `openai`, `gemini`, `ollama` | Top-k logprobs only; output entropy is a lower bound; no teacher forcing |
| `[P]` proxy | `anthropic` | Selected token only. The entropy-shaped measurements degenerate unless a `--surrogate` reads the output text under teacher forcing; the distribution **divergence** `perturbation_jsd_bits` is absent outright, and no surrogate recovers it |

The input-side measurements (`input_entropy_shift_bits`, `input_entropy_std_bits`, `prompt_surprisal_excess_bits`) require teacher forcing. On a backend that cannot teacher-force they are either **absent from the record entirely**, or — with `--surrogate` — computed by a small local open-weight model reading the same prompt. In the second case they describe the prompt under that reference model, not the target: their subject is `prompt-only`, so they leave `measurements` for the `prompt_measurements` block (see [Subject](#subject--whose-behaviour-the-number-describes)). `hif models` prints, per backend, which measurements degrade this way.

Absent is never zero. A measurement the run produced no evidence for is omitted from `measurements`, because "no evidence" and "measured zero" are different statements. Absent also covers *measured something else*: a quantity whose subject on this backend is the prompt is omitted rather than emitted with a caveat.

### The empty output side

Access is not the only way an output-side measurement loses its evidence. A run can simply come back with **nothing generated** — `output_side.steps == []` — and that case has broken the absence rule more than once, because an empty series is the one input on which every aggregate still returns a number. `mean([])` guarded to `0.0`, a divergence over no steps, a slope through no points, a cosine against the embedding of `""`: each is arithmetic that completes, and none of them is a measurement.

`io_correlation_r` was cut in hif-v4 for exactly this — on a run with no output steps it published a measured `0.0` correlation against a fabricated series. `output_distributions_unusable()` was then written so that "no steps" counts as unusable, which withholds every *distribution* row. It does not cover `io_cosine_similarity`, which reads output **text** and is therefore correctly present on a selected-only backend that returns real words and no logprobs — so on a run that returned no words either, nothing was watching. Two gpt-5 profiles shipped with the distribution rows correctly absent and `io_cosine_similarity` present, computed from the perturbation variants' continuations.

The rule now has its own declaration. A registry row sets `needs_generated_output` when its value is read off the continuation the target produced, `measurements()` sweeps those rows out whenever the run has no steps, and the matching chart gate declines alongside it. `tests/unit/test_empty_generation.py` is the guard, at the gate and end-to-end through a backend that returns nothing.

**An empty output side is stated, not merely implied.** `provenance.target_generated_no_output` records the fact and `provenance.generation_stop_reason` records why — a refusal, a content filter, or (the observed gpt-5 case) a reasoning model that spent its entire completion budget on hidden tokens and returned no visible content. Without the reason, a reader comparing eight regimes sees a sparser row rather than a run that never happened. `provenance.output_distribution_selected_only` stays `False` on these runs and that is correct: a run that returned nothing did not return point masses either.

### Why measure behaviour at all on closed models

A reasonable objection: behavioural measurement of a closed model is of limited value. The substance of that objection is conceded. On a closed model there are no weights, no full logits, no attention, and no teacher forcing — the API response is the entire observable surface, so reading that surface is not the preferred method there, it is the only one that exists, and the `[T-k]` and `[P]` tiers are inventories of how little it exposes. This arrangement is a stopgap, and treated as one: every measurement becomes exact on open weights, and if providers expose more (full logprobs, attention, stable version identifiers), measurements should migrate up the access tiers and the proxy tier should shrink toward empty. The limitation lives in provider opacity, not in the method; nothing in the proxy tier substitutes for direct access, which is why a quantity the surface cannot support is reported absent rather than approximated.

---

This document provides precise definitions for every quantity computed by HIF, in four parts:

1. **The measurement set** — the run-level scalars a run reports, in natural units
2. **The token-level traces** — the same quantities at `per-step` / `per-position` resolution, restoring what a scalar compresses
3. **Low-level component metrics** — the per-step distribution, semantic, sensitivity, and perturbation-response measurements the set is derived from
4. **Perturbation-field descriptors** — the run's behaviour as a *region* rather than a point

All distribution and semantic metrics are computed once per generation step over the top-K probability distribution. The `truncated=True` flag indicates the distribution covers only top-K tokens. On a `[P]` backend with a surrogate, the distributions are the surrogate's over the target model's output text.

---

## Part 1 — The Measurement Set

The measurement set is defined once, in `MEASUREMENT_REGISTRY` in `hif/profile/registry.py`, and extracted by `measurements(profile)` (`hif/profile/measure.py`) — run `hif schema` for the current set. The same function feeds the CLI table and the machine record, so a number shown in a terminal and a number in a JSONL line can never diverge. Each is a triple: observable × functional × resolution; the record carries the run-level scalar, and rows with `per-step` / `per-position` resolution also have a trace in Part 2.

| Key | Name | Unit | Resolution | Subject | Requires |
|-----|------|------|------------|---------|----------|
| `input_entropy_shift_bits` | Input entropy shift (bits) | bits | aggregate | `target-distribution` → `prompt-only` | teacher forcing (or `--surrogate`) + perturbation variants |
| `input_entropy_std_bits` | Input entropy shift spread (bits) | bits | aggregate | `target-distribution` → `prompt-only` | teacher forcing (or `--surrogate`) + ≥ 2 perturbation variants |
| `perturbation_jsd_bits` | Perturbation JSD (bits) | bits | aggregate | `target-distribution` | perturbation variants + top-k logprobs |
| `io_cosine_similarity` | Input/output cosine similarity | dimensionless | aggregate | `target-output-text` | ≥ 1 perturbation variant + an embedding encoder |
| `prompt_surprisal_excess_bits` | Prompt surprisal excess (bits) | bits | per-position | `target-distribution` → `prompt-only` | teacher forcing (or `--surrogate`) |
| `output_entropy_bits` | Output entropy (bits) | bits | per-step | `target-distribution` → `target-output-text` | top-k logprobs |
| `output_nucleus_entropy_bits` | Output nucleus entropy (bits) | bits | per-step | `target-distribution` → `target-output-text` | full logprobs + `--entropy-percentile` |

**hif-v4.1 adds `output_nucleus_entropy_bits`** — an addition, so a minor
bump: a v4 artifact and a v4.1 one stay in the same family and `hif compare`
still intersects over the six rows both carry. It is off unless
`--entropy-percentile` is passed, so a run that does not ask for it is
byte-identical to a v4 run.

**The set is six rows**, each admitted
against the project's own 120-profile corpus — the evidence is recorded row by
row in the `SIGNAL_SET_VERSION` history (`hif/profile/registry.py`) and the
criteria are now conditions 3–6 of the Significance Gate above. The pipeline
stages behind several cut rows still run under `--diagnostics` and their blocks
still ship in the artifact as evidence; the set is the claims.

Each row has exactly one name, and it names the quantity in the terms the quantity is computed in. Rows carried a second, coined name until `hif-v3.3` — "Stability", "Sensitivity", "Wager ▲", "Entropy ●", "Shift ◆", "Veer ◈", "Spread ■", "Horizon", "Exposure ◇", "Continuity" — and the coined one is the one that went wrong: "Stability" sat on `input_entropy_std_bits`, a standard deviation, where a *higher* number means *less* stable. A name that inverts the reading direction of its own number is worse than no name. The quantities here have accepted names already — Shannon entropy, Jensen-Shannon divergence, Pearson r, cosine similarity, surprisal — so the key and the name now say the same thing at two registers, and no glossary sits between a reader and a number. Chart glyphs are a display concern and live in `hif/viz/registry.py`.

The Subject column reads `declared` → `under surrogate`: the first value holds when the target's own machinery produced the quantity, the second when the surrogate named by the row's `surrogate_group` stood in. A single value means the subject does not change. See [Subject](#subject--whose-behaviour-the-number-describes).

### `input_entropy_shift_bits`

**Zone.** Input side, under perturbation.

**Definition.** Mean absolute difference, over perturbation variants, between the variant's mean input-token entropy and the baseline's.

```
input_entropy_shift_bits = mean_v | μ[H(Pᵢ)]_variant_v − μ[H(Pᵢ)]_baseline |

where H(Pᵢ) = −∑ₜ pₜ log₂ pₜ  (teacher-forced full-vocabulary entropy at prompt position i)
```

**Unit and range.** Bits. Unbounded above. `0` means the perturbations moved input-side entropy not at all.

**Not inverted, not normalised.** This replaces the former "Input Stability" (`1 − mean|Δ stability_score|`), which saturated at exactly 1.0 in the regime that mattered and divided by `log₂(vocab_size)`. The historical name named the inverted score, not this quantity, which is one of the reasons no coined name survives on any row (see Part 1).

**Absent when** there are no perturbed input-side analyses — the backend cannot teacher-force and no surrogate was supplied. Computed in `hif/metrics/stability.py::compute_stability_metrics`.

---

### `input_entropy_std_bits`

**Zone.** Input side, under perturbation.

**Definition.** Sample standard deviation of the per-variant input entropy shifts — the spread of the model's entropy response across perturbations, where `input_entropy_shift_bits` is its mean.

```
input_entropy_std_bits = std_v( | μ[H(Pᵢ)]_variant_v − μ[H(Pᵢ)]_baseline | ),  ddof = 1
```

**Unit and range.** Bits. Unbounded above. `0` means every variant moved input-side entropy by exactly the same amount.

**History.** Added in hif-v2.1 — the natural-unit form of the Stability aggregate, which was computed but never surfaced while the inverted `1 − x` stability scores existed.

**Absent when** fewer than two perturbation variants exist (a single shift has no spread), or the backend cannot teacher-force and no surrogate was supplied. Computed in `hif/metrics/stability.py::compute_stability_metrics`.

---

### `perturbation_jsd_bits`

**Zone.** Perturbation.

**Definition.** Mean Jensen-Shannon divergence between the baseline output distribution and each paraphrase variant's, averaged across generators and generation steps.

```
perturbation_jsd_bits = mean_v [ mean_j [ JSD(P_baseline,j ‖ P_variant,j) ] ]

JSD(P ‖ Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),  M = ½(P + Q),  logs base 2
```

**Unit and range.** Bits. Genuinely bounded to `[0, 1]` by definition in log base 2 — that bound is a property of JSD, not a rescaling. It is reported as measured.

**Perturbation generators.** Default `["synonym", "tone", "reorder"]` with `n_variants = 2` each. `substitution` and `ambiguity` are implemented and selectable; LLM-backed paraphrasing is opt-in and requires an explicit endpoint.

**Fallback.** When the aggregate is absent but per-perturbation `SensitivityMetrics` exist, `measurements()` averages `mean_js_divergence` over them — same quantity, same unit.

**Absent when** the backend returns only the selected token (the `[P]` tier). This is the point-mass rule, and it is an absence rather than a caveat for the same reason a prompt-only quantity leaves `measurements`: *the computation stops being the one the key names.* The JSDs are taken over the RAW baseline and variant traces (`build_profile` step 6, before any surrogate recovery), so on a selected-only backend both sides are point masses. The divergence between two point masses is `0` when the selected tokens agree and exactly `1` bit when they differ — a **token-disagreement rate**, not a divergence between distributions. The rate is not re-admitted under another key either: it would have to pass the Significance Gate on its own, and nothing has shown a run that needs it. `--surrogate` does not rescue this measurement — the step-6b recovery rebuilds `semantic_steps`, which the sensitivity path never reads.

---

### `io_cosine_similarity`

**Zone.** Perturbation (cross-pair).

**Definition.** Mean cosine similarity between each input embedding and its paired output embedding — the `io_sim` member of `SimilarityMetrics`.

```
io_sim = mean_i cos( embed(input_i), embed(output_i) )
```

over all `(input, output)` pairs: baseline plus one per perturbation variant.

**Unit and range.** Dimensionless, bounded to `[−1, 1]` by definition. High: output stays in the semantic neighbourhood of its input. Low: output sits far from the prompt's representational space.

**Companions (persisted on `metrics.similarity`, not in the measurement set).** `input_sim` and `output_sim` are the mean pairwise cosines *within* the input set and *within* the output set; `io_ratio = output_sim / input_sim` captures amplification vs. suppression (`> 1` = outputs converge more than inputs did; `< 1` = the model amplifies input variation). `trend` is the linear slope of per-step mean pairwise similarity across the output sequence, derived from `SemanticMetrics.mean_pairwise_distance`; it is surfaced separately as `findings.similarity_trend_slope`, signed and unrounded, and **`None` when the run has fewer than two steps to fit a line through** (it was `0.0`, which reported a flat trend for a generation that never happened).

**Absent when** there are no perturbation variants — at least one variant alongside the baseline is required — **or when the target generated nothing at all.**

The second clause was missing, and the gap is in the published corpus. The pair set is `{baseline} ∪ {variants}`, and only the baseline half is the run's own output. When the target returns no tokens the baseline pair is `(prompt, "")` — the empty string embeds to a real vector, so the cosine is a real number rather than a zero — and the remaining pairs are the *paraphrases'* continuations. gpt-5 answered two of eight prompt regimes with zero tokens and both profiles published `io_cosine_similarity` (0.17 and 0.10): correct arithmetic over sixteen pairs, fifteen of which described text the record is not about.

This row is deliberately exempt from the distribution gate that withholds `output_entropy_bits` and `perturbation_jsd_bits` — it reads output *text*, so it survives a selected-only backend by design — which is precisely why nothing above it caught the empty case. It is now gated on its own declaration (`needs_generated_output` in `MEASUREMENT_REGISTRY`), enforced in `measurements()` and guarded by `tests/unit/test_empty_generation.py`.

**Encoder-dependent.** Comparable only between profiles computed with the same embedding model, which is recorded in `config.embedding.model_name`.

---

### `prompt_surprisal_excess_bits`

**Zone.** Selection, input side. The per-position trace is in Part 2.

**Definition.** Mean excess surprisal over per-position entropy across teacher-forced prompt positions.

```
prompt_surprisal_excess_bits = (1/T) ∑ᵢ max(0, sᵢ − H(Pᵢ))

where sᵢ = −log₂ p(tokenᵢ | ctx<ᵢ)  (surprisal of the actual token)
and H(Pᵢ) = −∑ₜ pₜ log₂ pₜ         (Shannon entropy of the full distribution at position i)
```

The surprisal `sᵢ` is unweighted by probability — this is deliberate. Entropy is the probability-weighted average of surprisal: `∑ p(x)·(−log p(x))`. That weighting compresses the logarithmic spread: the product `p·(−log p)` is arch-shaped, peaking near `p = 1/e ≈ 0.37` and falling to zero at both extremes. High-probability tokens have small surprisals but large weights; low-probability tokens have large surprisals but small weights; the two effects pull toward each other. This compression makes entropy a stable average — but it dulls exactly the signal this measurement is after. The unweighted surprisal lets the logarithmic spread do what it naturally does: make rare events stand out. A token at 1% probability produces a surprisal of ~6.6 bits; weighted by 0.01 it contributes only 0.066 — nearly silent, precisely when it should be loudest.

When `sᵢ > H(Pᵢ)` the actual token was more surprising than the distribution's own average uncertainty — an "underdog" against a concentrated distribution.

**Worked example.** At position `i`: `H(Pᵢ) = 2.90 bits`. The actual token sits at rank 5 with probability 2.7%, so `sᵢ = −log₂(0.027) ≈ 5.2 bits`. Excess = 5.2 − 2.90 = 2.3 bits. A position whose actual token was the model's top-1 contributes 0.

**Unit and range.** Bits, `[0, ∞)`. Unbounded above, and reported unscaled.

**Absent when** no teacher-forced positions exist. Computed by `hif/hourglass/input_side.py::mean_surprisal_excess`.

---

### `output_entropy_bits`

**Zone.** Output side. The per-step trace is in Part 2.

**Definition.** Mean over generation steps of the Shannon entropy of the per-step top-K output distribution (`DistributionMetrics.entropy_bits`).

**Unit and range.** Bits. **A lower bound** on full-vocabulary entropy whenever the distribution is truncated to top-k, and **not comparable across backends with different k**. `DistributionMetrics.entropy_bits_upper` carries the uniform-tail upper bound when the vocabulary size is known.

---

## Part 2 — The Token-Level Traces

A run's scalar measurements (Part 1) compress a full generation into one number per quantity. The token-level traces restore what compression hides: which specific tokens drove a value, where entropy and attention converge or diverge step-by-step, and whether a flat mean reflects genuine uniformity or cancellation between extremes. These are the same measurements at their native resolution — a row whose `resolution` is `per-step` or `per-position` has its trace here.

Both traces here have a chart in the viz registry (`hif/viz/registry.py`, `kind="reading"`) and an aggregate in the measurement set, and the chart and the key read the same series, so they cannot drift apart. That invariant is why hif-v3.1 admitted a chart-only quantity to the set rather than leaving it visible on the companion website and unreachable from the CLI — and why hif-v4, cutting that same quantity, cut its chart with it.

| Trace | What it shows, per token | Run-level key | Requires |
|-------|--------------------------|---------------|----------|
| Prompt surprisal excess | Surprisal excess over entropy, per prompt position | `prompt_surprisal_excess_bits` | Teacher forcing (open-weight, or `--surrogate`) |
| Output entropy | Output distribution entropy, per generation step | `output_entropy_bits` | All models with top-k logprobs |


---

### Prompt surprisal excess — the per-position trace

**Formula.** `Wagerᵢ = max(0, sᵢ − H(Pᵢ))`

where `sᵢ = −log₂ p(tokᵢ | ctx<ᵢ)` is the surprisal of prompt token `i` and `H(Pᵢ)` is the Shannon entropy of the full-vocabulary distribution the model predicted at that position, both from the teacher-forced pass over the prompt.

**What it shows.** Surprisal excess per prompt position. Each bar shows how many bits the chosen token's surprisal exceeded the model's distributional entropy at that position — the residual cost of the actual token beyond general uncertainty. A tall bar at position `i` means the model had committed to a narrow distribution but the actual token was a long shot against that commitment.

The surprisal `sᵢ` is unweighted by probability — this is deliberate. Entropy is the probability-weighted average of surprisal: `∑ p(x)·(−log p(x))`. That weighting compresses the logarithmic spread: the product `p·(−log p)` is arch-shaped, peaking near `p = 1/e ≈ 0.37` and falling to zero at both extremes. High-probability tokens have small surprisals but large weights; low-probability tokens have large surprisals but small weights; the two effects pull toward each other. This compression makes entropy a stable average — but it dulls exactly the signal Wager is after. The unweighted surprisal lets the logarithmic spread do what it naturally does: make rare events stand out. A token selected with 1% probability produces a surprisal of ~6.6 bits; weighted by 0.01, it contributes only 0.066 — nearly silent, precisely when it should be loudest.

The interesting case is when `sᵢ` and `H(Pᵢ)` diverge: low `H(Pᵢ)` means the distribution is narrow and confident, but high `sᵢ` means the actual token is not what the model was confident about. `sᵢ − H(Pᵢ) > 0` is the excess — the model committed, and the token overrode that commitment. When the excess is zero, the model either selected its most likely token or was already broadly uncertain.

**Relation to `prompt_surprisal_excess_bits`.** The measurement runs `max(0, sᵢ − H(Pᵢ))` over all positions and averages to a single number; the Wager trace is that same quantity at full per-position resolution. The Surprise chart in the registry draws the same underlying series with the mean called out. They are one quantity at two resolutions, which is why the measurement set carries it only once — the historical `wager` aggregate was byte-for-byte identical to the `surprise` aggregate, and reporting one number twice under two names inflated the apparent dimensionality of the signal set.

**Expected range.** `[0, ∞)` bits. Most positions contribute zero. Large values at specific positions identify structurally surprising tokens — places where the model had committed and the actual token overrode that commitment.

**Access.** Requires teacher forcing: an open-weight backend (`hf`, `tlens`), or a `--surrogate` proxy teacher-forced over the same prompt. A surrogate reading describes the surrogate, and is flagged as such via `findings.surrogate_model_name`.

---

### Output entropy — the per-step trace

**Formula.** `Entropyⱼ = H(Qⱼ) = −∑ᵥ Qⱼ(v) log₂ Qⱼ(v)`

Sum over the top-K candidates the backend returned at each generation step `j = 1…G`.

**What it shows.** Output distribution entropy per generation step, in bits. Peaks mark genuine decision moments where many tokens were competitive; troughs mark committed choices where the model narrowed sharply.

The chart draws two series: the **nucleus entropy** (95% mass, renormalised — comparable across backends regardless of how many logprobs each exposes) and the **raw top-K entropy** (the truncation lower bound). The trace is the full shape that a single mean compresses: it shows whether that mean conceals a flat plateau, a single tall spike, or alternating peaks and troughs — patterns that carry distinct interpretive weight.

**Relation to Breadth / ESS.** Output entropy and Effective Support Size are the same underlying signal in different units: `ESS = 2^H`, an equivalent token count. The Breadth chart draws the nucleus ESS trace with its mean. The measurement set reports `output_entropy_bits` and not ESS, because a quantity appears once. See Effective Support Size below for why the transform is 2^H and why the basis has to travel with the number.

**Truncation.** Whenever the distribution is top-k truncated, the reported entropy is a **lower bound** on true full-vocabulary entropy, and values are not comparable across backends with different k. `DistributionMetrics.entropy_bits_upper` carries the uniform-tail upper bound where the vocabulary size is known.

**Expected range.** `[0, log₂ K]` bits — about 5.64 bits for K=50. The full-vocabulary ceiling `log₂|V|` (≈ 15.6 bits for a 50,257-token vocabulary) is not reachable from a truncated distribution.

**Access.** All models with output logprob data. On a selected-token-only backend the distribution degenerates unless a `--surrogate` recovers it.

---

### `output_nucleus_entropy_bits` — entropy over a fixed share of the mass

**Formula.** `H(Q̃ⱼ)` where `Q̃ⱼ` is the smallest prefix of `Qⱼ`, sorted by
descending probability, whose cumulative mass reaches `p`, renormalised to sum
to 1. Reported as the mean over generation steps `j = 1…G`.

**Why a separate key.** `output_entropy_bits` is the entropy of whatever the
backend exposed; this is the entropy of a *fixed fraction* of the mass. They
answer different questions, so they are two rows rather than one row with a
flag deciding its own meaning. Reported under one key, every published profile
would have to be read alongside the flag to know what its number was.

**When it is absent.** Whenever the captured top-K did not reach `p` at every
step. This is the condition that makes the number comparable at all: a nucleus
computed from a slice that does not contain the nucleus is the entropy of the
slice, not of the nucleus, and two such values from different backends compare
two different definitions. `nucleus_entropy_bits` in `DistributionMetrics`
deliberately degrades instead — it must draw a chart on every backend — which
is why the measurement does not read it.

**Cost.** The requirement is strict in practice, not just in principle. On
GPT-2, `--top-k 50` captures roughly half the mass at a typical step; reaching
95% takes a top-K in the low thousands, and the pipeline cost scales with it.
Lower percentiles are reachable at ordinary budgets: p50 needs only the handful
of candidates carrying half the mass.

**Expected range.** `[0, log₂ n_p]` where `n_p` is the nucleus size. Monotone
in `p` — a smaller share of the mass can never spread over more candidates —
so p50 ≤ p80 ≤ p95 on the same distribution.

**Access.** Backends exposing full logprobs (`hf`, `tlens`). The CLI refuses
`--entropy-percentile` elsewhere rather than reporting a number computed from a
slice that cannot contain the nucleus.

---

### Reading the Traces Together

The two traces expose different sides of the same computational event, and the
one combination they support is the sharpest one the larger set had:

**Output entropy + prompt surprisal excess.** Where the output entropy is low
(a committed distribution) but the prompt's surprisal excess is high (the
actual token was a long shot against that commitment), the model was confident
and the input overrode it. These are the positions where the model's
distributional commitments were most systematically violated — per position on
the prompt side, per step on the output side.

Combinations involving the retired traces (attention spread, step JSD, centroid
veer, exposure) went with their measurements — see the hif-v4 history in
`hif/profile/registry.py`. The blocks those stages produce still ship in the
artifact under `--diagnostics` for anyone who wants to read them as evidence.

## Part 3 — Low-Level Component Metrics

These are the per-step measurements that feed into the aggregate measurements above.

---

## Distribution Metrics

Computed by `compute_distribution_metrics()` in `hif/metrics/distribution.py`. One `DistributionMetrics` object is produced per output generation step.

---

### Shannon Entropy

**Definition.** The expected information content of the top-K distribution at a single generation step, measured in bits.

```
H(p) = -∑ p_i · log₂(p_i)    for p_i > 0
```

The sum is over the K candidates as the model returned them — deliberately **not** renormalized, because the tail mass `1 − ∑p_i` is what the upper-bound correction below needs.

**Expected range.** [0, log₂(K)]. For K=50: [0, ~5.64 bits]. A single certain token gives H=0; a uniform distribution over all 50 candidates gives H=log₂(50) ≈ 5.64 bits.

**Interpretation.** Low entropy: the model is highly confident about what comes next, with most probability mass on one or two tokens. High entropy: the model is genuinely uncertain among many alternatives. Because this is computed over the truncated top-K distribution (not the full vocabulary), it is a **lower bound** on true full-vocabulary entropy, and absolute values are not comparable across runs with different K.

**Companions.** `entropy_bits_upper` is the uniform-tail upper bound — the entropy obtained by spreading the unobserved tail mass uniformly over the remaining vocabulary. It is `None` when the vocabulary size is unknown or the distribution was not truncated. `nucleus_entropy_bits` is the entropy of the smallest prefix carrying 95% of the mass, renormalized to a proper distribution; it is what the step-delta measurement and Effective Support Size are computed from, because it always works with the same fraction of probability mass and is therefore comparable across backends exposing different numbers of logprobs.

---

### Logit Margin

**Definition.** The difference between the rank-1 and rank-2 logit values at a single generation step.

```
M = logit[rank-1] - logit[rank-2]
```

**Expected range.** ℝ. Typical values for GPT-2: roughly 0–10 logit units.

**Interpretation.** Large positive margin: the model strongly prefers the top candidate over the second-best alternative. Near-zero margin: the top two candidates are nearly indistinguishable — a small change in context could flip the preference.

---

### Effective Support Size

**Definition.** The effective number of equally-likely tokens a distribution behaves like — 1 at a point mass, |support| at the uniform.

```
ESS(p) = 2^H(p)
```

**Why 2^H and not another effective-size formula.** The measures satisfying the natural requirements form a one-parameter family, `S(p, α) = (Σ pᵢ^α)^(1/(1−α))` — the exponential of Rényi's α-entropy (Grendár, *Entropy and Effective Support Size*, Entropy 2006, 8[3], 169–174, [doi:10.3390/e8030169](https://doi.org/10.3390/e8030169)). Every α in it is continuous and symmetric, is bounded by `1 ≤ S ≤ m`, is unchanged by appending an impossible outcome, and is multiplicative over independent variables. Only **α = 1** additionally satisfies `S(X)·S(Y) ≥ S(X,Y)` with equality iff X and Y are independent; for α ≠ 1 that inequality can reverse. α = 1 is the limit case — the closed form is 0/0 there — and its value is `exp(H)`, which is `2^H` when H is in bits. So the common alternative α = 2 (inverse Simpson / participation ratio) is excluded by an axiom, not by taste.

**Whose entropy — the basis decides the ceiling.** 2^H says nothing about *which* distribution H came from, and hif carries two, over different bases:

| field | basis | ceiling |
|---|---|---|
| `nucleus_effective_support_size` | the renormalized 95% nucleus | nucleus size |
| `full_effective_support_size_upper` | full vocabulary, uniform-tail corrected | `vocab_size` |

These are **not** a bracket on one quantity and must not be read as one. The nucleus field is what the chart draws — it is comparable across backends precisely because it always works with the same fraction of the mass, whereas raw top-K entropy grows with K. (They were previously `effective_support_size` and `effective_support_size_upper`, which named them as one quantity and its bound.)

**Expected range.** `[1, ceiling]` per the table above.

**ESS is not a measurement.** It is a bijection of entropy: it moves with entropy exactly and discloses nothing entropy does not, so it fails the Significance Gate's *distinct disclosure* condition as squarely as `continuity = 1 − sensitivity` did. It never enters `measurements`, and the CLI does not print it.

**Why bits are the stored unit.** Entropy *differences* are meaningful and are themselves measurements here — `input_entropy_shift_bits` **is** a difference of entropies, and `perturbation_jsd_bits` is a divergence in bits. ESS does not subtract (`2^H₁ − 2^H₂` is not a quantity); it only ratios, `ESS₁/ESS₂ = 2^(H₁−H₂)`. The record's arithmetic lives in bits, so the record stores bits.

Explaining a number in the other unit is a job for prose, and this document does it above: 4.9 bits is about a uniform choice among ~30 tokens. That is a gloss on a unit, not a second measurement, and it needs no machinery — only the basis named, which the field names now carry.

---

### Top-K Cumulative Mass

**Definition.** The sum of the probabilities of the top-k tokens (using the configured `top_k_for_mass`, default 10) after normalizing the full top-K distribution.

```
CK(p, k) = ∑_{i=1}^{k} p_(i)    (sorted descending)
```

**Expected range.** [0, 1].

---

### Tail Weight

**Definition.** The sum of probability mass assigned to tokens with individual probability below a threshold (default θ=0.01).

```
TW(p, θ) = ∑_{p_i < θ} p_i
```

**Expected range.** [0, 1].

**Interpretation.** High tail weight: many low-probability tokens collectively carry meaningful mass. This can matter when the "right" answer is in the tail — a model that looks confident by entropy but has substantial tail weight is placing significant mass on rare tokens.

---

## Semantic Metrics

Computed by `compute_semantic_metrics()` in `hif/metrics/semantic.py`. One `SemanticMetrics` object per output generation step. Requires embedding the top-K candidate token strings and clustering with HDBSCAN (KMeans as fallback).

---

### Cluster Count

**Definition.** Number of HDBSCAN semantic clusters among the top-K candidates. Noise points (label −1) excluded.

**Interpretation.** Cluster count = 1: all top-K candidates are semantically similar. Cluster count = 5: the model is considering five meaningfully distinct directions.

---

### Cluster Entropy

**Definition.** Shannon entropy of the probability mass distribution across semantic clusters.

```
H_c = -∑_k m_k · log₂(m_k)

where m_k = ∑_{i in cluster k} p_i  (normalized)
```

**Interpretation.** Separates "how many clusters" from "how evenly mass is distributed across them." High cluster entropy means genuinely competing semantic directions; low cluster entropy means one direction dominates even if alternatives exist.

---

### Mean Pairwise Distance

**Definition.** Probability-weighted mean cosine distance between all pairs of top-K candidate embeddings.

```
D̄ = ∑_{i≠j} (p_i · p_j / Z) · d(e_i, e_j)
```

**Expected range.** [0, 1].

---

### Max Inter-Cluster Distance

**Definition.** Maximum cosine distance between any two cluster centroids.

**Interpretation.** Measures the span of the semantic space covered by top-K candidates at this step.

---

### Intra-Cluster Density

**Definition.** Probability-weighted mean cosine similarity between all pairs within the same cluster, averaged across clusters.

**Expected range.** [0, 1].

---

### Topic Variance

**Definition.** Probability-mass-weighted variance of cluster centroids around the weighted mean centroid.

```
TV = ∑_k w_k · ||c_k - c̄||²
```

**Interpretation.** High topic variance at an early step: the model is actively considering completions in semantically distant directions.

---

### Clustering Parameters and Noise Handling

HDBSCAN is the default clustering method (`ClusterConfig.method = "hdbscan"`).

**`min_cluster_size`** scales dynamically with N (number of candidates): `max(config.min_cluster_size, N // 10)`. With the default `config.min_cluster_size = 2` and top-K = 50, the effective minimum is 5 — requiring clusters to contain at least 5 candidates. This prevents token pairs from fragmenting what should be a single semantic region.

**`min_samples = 1`** — maximally permissive core-point threshold. Every point can be a core point, so density thresholding is driven entirely by `min_cluster_size`.

**Noise handling.** HDBSCAN assigns label `-1` to points it cannot assign to any cluster. Noise points are excluded from all weighted metric computations (cluster entropy, intra-cluster density, max inter-cluster distance, topic variance). The fraction of discarded candidates is reported as `noise_fraction` in `SemanticMetrics` so downstream consumers can assess metric reliability.

**All-noise fallback.** When HDBSCAN produces zero clusters (all candidates are noise), the result is treated as one undifferentiated single cluster rather than forcing a KMeans split. An all-noise result is a real signal — the distribution is diffuse with no density peaks — and reporting it as one cluster is more honest than fabricating structure with an arbitrary two-way split.

**KMeans fallback.** Used only when `config.method = "kmeans"` or `config.n_clusters` is explicitly set. In the default pipeline, KMeans is not used as a fallback for HDBSCAN failures.

---

## Sensitivity Metrics

Computed by `compute_sensitivity_metrics()` in `hif/metrics/sensitivity.py`. One `SensitivityMetrics` object per perturbation variant.

All divergence computations operate on the union of token ID sets from the baseline and perturbed top-K lists. Tokens absent from one step receive probability 0 before normalization. This is an approximation: tail tokens outside *both* top-K sets are invisible to it.

---

### Jensen-Shannon Divergence (per step)

**Definition.** Symmetric, bounded divergence between baseline and perturbed top-K distributions at a single step.

```
JSD(p ‖ q) = ½ KL(p ‖ m) + ½ KL(q ‖ m)

where m = ½(p + q)
and KL(a ‖ b) = ∑ a_i · log₂(a_i / b_i)
```

**Expected range.** [0, 1]. JSD=0: identical distributions. JSD=1: fully disjoint support. Typical synonym substitutions on well-calibrated prompts: 0.05–0.15. Values above 0.3 indicate materially changed output distribution.

---

### KL Divergence (per step)

**Definition.** Asymmetric divergence from perturbed to baseline.

```
KL(p ‖ q) = ∑ p_i · log₂(p_i / q_i)
```

**Expected range.** [0, ∞). Undefined (infinite) when the perturbed distribution places mass outside the baseline's support. **That case is `null`**, and `mean_kl_divergence` averages the steps where KL is defined, reporting how many it left out as `n_undefined_kl_steps`. Use JSD as the primary quantity; KL provides directionality information.

It was clamped to a `1e9` sentinel "so the value round-trips through JSON" — `null` round-trips perfectly well, and the clamp disarmed the guard immediately below it: the aggregate filtered its inputs with `math.isfinite`, and `1e9` is finite, so the filter written to drop undefined steps dropped none of them. 833 records across half the published corpus carried a mean near `9.65e8`, which the technical report rendered as `965517241.3793`. On a selected-only backend the real reading is starker and more useful: 56 of 58 steps undefined, and a mean of `0.0` over the 2 where the selected tokens agreed.

---

### Entropy Delta (per step)

**Definition.** Difference in per-step Shannon entropy between perturbed and baseline.

```
ΔH = H(p_perturbed) - H(p_baseline)
```

**Interpretation.** Positive: perturbation made the model more uncertain. Negative: more confident. Systematic direction across steps reveals whether a perturbation class consistently pushes toward or away from high-entropy regions.

---

### Nucleus Overlap (per step)

**Definition.** A set-based complement to JSD: the fraction of the baseline's 90% nucleus token set that survives in the perturbed step's 90% nucleus.

```
nucleus_overlap_p90 = |baseline_nucleus_p90 ∩ perturbed_nucleus_p90| / |baseline_nucleus_p90|
```

**Expected range.** [0, 1]; `1.0` means identical nuclei. An empty baseline nucleus is **`null`** — there is no set to take a fraction of. It was defined as stable (`1.0`), which is the top of the range and reads as perfect agreement rather than as no comparison having been made. Aggregated per variant as `mean_nucleus_stability_p90`, which is likewise `null` when no step could answer.

**Why both.** JSD measures mass shift; nucleus overlap measures whether the *viable token set* changed. A small mass shuffle near the threshold can flip nucleus membership without moving JSD much, and the reverse also happens.

---

### Alignment — how much of the run each variant actually covers

A variant is compared against the baseline over their **shared prefix**, `min(len(baseline), len(variant))` steps, recorded per variant as `n_steps_aligned`. It is routinely shorter than the baseline: 215 variants in the published corpus aligned over fewer steps than their baseline ran, the worst covering 6 of 64.

**Every variant is weighted equally in `perturbation_jsd_bits` regardless of its coverage**, and that is the measurement's definition rather than an oversight — `mean_v [ mean_j [ JSD ] ]` takes the *variant* as the unit of observation, because the quantity is about how much each paraphrase moved the model. Weighting by steps instead would move 5 of the corpus's 96 published values, the largest by 0.064 bits. `n_steps_aligned` exists so a reader can see the difference the definition deliberately ignores.

**A variant that aligned zero steps contributes nothing, not zero.** Its four means are `null` and it is excluded from the aggregate, with `metrics.stability.n_perturbations_aligned` recording how many variants actually contributed so the exclusion is not a silent reduction of *n*. This used to be `0.0` — indistinguishable from a paraphrase the model answered identically — and six variants of one corpus run aligned zero steps against a 603-step baseline. See [The empty output side](#the-empty-output-side).

---

## Perturbation Response

Computed by `compute_stability_metrics()` in `hif/metrics/stability.py`. One `PerturbationResponse` object per run, aggregating across perturbation variants. (`StabilityMetrics` remains as a backwards-compatible alias for the class name.)

Four fields, each computed independently from whatever evidence exists and set to `None` when that evidence does not exist — never a fake `0.0` and never a fake `1.0`:

| Field | Unit | Definition |
|-------|------|-----------|
| `input_entropy_shift_bits` | bits | `mean(\|perturbed.mean_entropy − baseline.mean_entropy\|)` |
| `input_entropy_std_bits` | bits | `std(\|perturbed.mean_entropy − baseline.mean_entropy\|, ddof=1)`; needs ≥ 2 variants |
| `perturbation_jsd_bits` | bits | `mean(mean_js_divergence)` over variants |
| `input_output_correlation` | dimensionless | Pearson `r` between the shift and JSD per-variant series |

These are the quantities that surface as `input_entropy_shift_bits`, `input_entropy_std_bits`, and `perturbation_jsd_bits` in Part 1 — see there for ranges and absent-vs-zero semantics.

---

## Center Diagnostics

Computed by `compute_center_diagnostics()` in `hif/hourglass/center.py`. One `CenterDiagnostics` object per run: `input_mean_entropy`, `output_mean_entropy`, `entropy_ratio`, `prompt_output_cosine_distance`.

`max_entropy = log₂(vocab_size)` is passed in and recorded for reference only. **Nothing here is normalised by it.**

---

### Entropy Ratio

**Definition.** Ratio of mean output entropy to mean input entropy, both in bits.

```
entropy_ratio = output_mean_entropy / input_mean_entropy
```

`output_mean_entropy` is re-computed from each step's normalized top-K probabilities. Ratio < 1: generation is more constrained than input processing. Ratio > 1: generation opens up more uncertainty than the input contained. `None` for API models with no input-side data; `inf` in the degenerate case where input entropy is 0 and output entropy is not.

---

### Prompt/Output Cosine Distance

**Definition.** Cosine distance between the embedding of the prompt and the embedding of the generated text.

```
prompt_output_cosine_distance = 1 - cosine_similarity(embed(prompt), embed(generated_text))
```

Zero vectors return `1.0`. **An empty generation returns `None`** — it returned `0.0`, which is the *minimum* of the range below and therefore reads, on a distance, as "the output is semantically identical to the prompt". That is the strongest possible anchoring claim, published about a model that produced no output. `output_mean_entropy` and `entropy_ratio` in the same block are `None` on the same runs, for the same reason.

(This paragraph previously documented the `0.0`, which is also why the published `io_cosine_similarity = 0.17` did not look like a known empty-generation path: it is a different quantity. `io_cosine_similarity` embeds each *pair* including the variants; `prompt_output_cosine_distance` embeds this run's prompt against this run's generated text, and only the latter had an empty-generation branch at all.)

**Expected range.** [0, 2] by definition. Low: the output stays in the semantic neighbourhood of the prompt. High: the output sits far from the prompt's representational space.

**Named for what it measures.** It is a distance between two embeddings from a single run, not evidence that a model drifted over time.

---

## Part 4 — Perturbation-Field Descriptors

Where Part 1 compresses a run to its run-level scalars and Part 3 exposes the per-step components, Part 4 characterizes the model's behavior as a **region** rather than a point. All are derived scalars, computed from distributions/embeddings held only transiently (compute-and-discard): a descriptor is a claim, and the raw distribution it was read from does not belong in the same block. `config.traceability.enabled` persists the raw member traces on the profile — alongside the descriptors, not inside them — so these can be recomputed without re-running models.

### Perturbation Field

`perturbation_jsd_bits` averages the pairwise Jensen-Shannon divergence of each perturbation variant to the baseline. The **perturbation field** treats the whole family — baseline plus all variants — as a *set* of distributions and characterizes its geometry. Computed by `compute_perturbation_field()` in `hif/metrics/field.py`; `None` when fewer than two members align or no output steps overlap.

**Jensen-Shannon centroid** — the probability-weighted mixture mean of the family's per-step output distributions on their union support, `M = Σᵢ wᵢ Pᵢ`. It is the family's behavioral center, adopted as the reference for the input in place of any single wording.

**Field dispersion (generalized JSD)** — a baseline-free measure of the family's internal dispersion:

```
GJSD = H( Σᵢ wᵢ Pᵢ ) − Σᵢ wᵢ H(Pᵢ)  =  Σᵢ wᵢ · KL(Pᵢ ‖ M)
```

Range `[0, log₂ n]`; `0` iff all members identical. Distinct from `perturbation_jsd_bits`, which is pairwise-to-baseline (how far the model moves from one canonical phrasing); field dispersion is baseline-free (how internally dispersed the neighbourhood is). They diverge when the baseline is itself an outlier.

**Field descriptors** — around the centroid: **mean_radius** (mean member→centroid JSD), **radius_variance** (isotropy of the cloud), **max_radius** (worst-case member), plus **n_members** and **n_steps_aligned** (the shared-prefix length the field was measured over). Per-generator **subfields** carry the same descriptors restricted to one perturbation class — the field's anisotropy. A class contributing only one member is skipped, because its within-class radius variance is undefined; with the default `n_variants = 2` each class does qualify. All values are bits (JSD, log base 2) except counts. Persisted as `metrics.field`.

**Distribution basis.** For a degenerate selected-only backend the raw traces are point masses, so the field would collapse to a crude token-agreement signal. When a surrogate is available each member's distribution is proxy-recovered the same way the baseline's is, keeping the field on the same basis as the other output-side measurements. Truncated and open backends use the model's own distributions unchanged.

### Trajectory Branch Field

The **branch field** restores the shape of the trajectory's stochastic branch cloud that a single continuity scalar reduces to one number — the *sampling-perturbation* twin of the perturbation field, in embedding (geometric) space. From the branch embeddings: `n_branches`, per-branch **mean_radius / radius_variance / max_radius**, **field_dispersion** = `1 − trajectory_continuity` (mean pairwise cosine distance), and a **cluster_count**. cluster_count is the **multi-modality** signal — ≥ 2 means the branches split into distinct semantic modes (genuinely divergent futures a mean similarity hides). Radii and dispersion are cosine distances in `[0, 2]`. Persisted as `trajectory.branch_field`; `None` with fewer than two branches or a skipped trajectory stage.

Descriptor names align with `PerturbationField` so `compute_field_deformation` consumes both; the branch cloud has no per-generator classes, so its `per_class` list comes back empty.

### Deformation (field shape change between two runs)

`compute_field_deformation(before, after)` in `hif/metrics/field.py` compares two derived fields — e.g. a baseline-phase and an event-phase field for the same prompt — and reports how the field's *shape* changed:

```
FieldDeformation.deformation = RMS( d_mean_radius, d_radius_variance, d_max_radius, d_field_dispersion )
where each component is a bounded relative change |Δx| / max(|before|, |after|, ε) in [0, 1]
```

plus a `per_class` list of per-generator dispersion changes, sorted largest-first.

**Translation is deliberately absent.** The behavioural centre *moving* and the field's *shape changing* are different events, and the second is often the more consequential one — a field that widens or fractures into modes at a stationary centre is invisible to a flat vector of per-metric deltas. But distribution-space translation is not recoverable from a persisted profile: the Jensen-Shannon centroid is discarded under the compute-and-discard rule and never reaches an artifact. This function therefore reports deformation only, and does not estimate translation from what it has.

There is no covariance-aware drift, no Mahalanobis translation, and no cross-run drift decomposition in this package. Deciding whether two runs differ meaningfully requires a null distribution established on your own models and prompts, and that judgement is out of scope here.

---

## Findings — provenance, not inference

`generate_findings()` in `hif/profile/builder.py` returns the run's non-inferential provenance. It carries no level, no flag, no verdict, and no summary sentence:

| Field | Meaning |
|-------|---------|
| `similarity_trend_slope` | OLS slope of per-step input/output cosine similarity across the generation. Signed and unrounded; positive means the output grew more similar to the input as it went on. Not thresholded. |
| `surrogate_model_name` | Set when the input-side measurements were computed by teacher-forcing a `--surrogate` proxy over the prompt instead of the target model. `None` when they came from the target directly. |
| `output_distribution_surrogate_name` | Set when the target backend's own per-step distribution was degenerate (selected token only) and a surrogate was teacher-forced over prompt + continuation to recover a real output entropy reading instead of a trivial `0.0` over a one-entry "distribution". Independent of the field above. |

The two surrogate names are reported rather than hidden because a measurement computed through a proxy describes the proxy reading the target's text — not the target's own computation. The CLI stars such measurements in its table, and `signals_record()` emits both names under `surrogate`.
