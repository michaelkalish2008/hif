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

**History (vocabulary).** Earlier versions of these docs split this one concept across two words and two files: aggregate-resolution quantities were "metrics" (a METRICS.md) and token-level ones were "instrument readings" (an INSTRUMENTS.md). The split was nothing but the resolution coordinate of the triple wearing two names, and maintaining two vocabularies for one axis invited exactly the kind of double-counting the dedup notes below warn against. There is one concept — a **measurement** — and resolution is a field, not a document boundary.

## Significance Gate — the bar for admitting a measurement

A computable triple is not automatically a measurement. **This gate is the acceptance criterion for contributing a new measurement** (see [CONTRIBUTING.md](../CONTRIBUTING.md)). Two conditions, both required:

1. **Derivability** — computable from the distributional observable alone, no inference to hidden structure.
2. **Distinct disclosure** — discloses a facet no admitted measurement already captures; must move independently somewhere across contexts.

The second condition is why several plausible quantities are *not* in the set: `continuity` was `1 − sensitivity` computed from the same JS divergences, the historical `wager` aggregate was byte-for-byte the `surprise` aggregate, and ESS is entropy in different units. Each quantity appears exactly once.

## Natural units

Every measurement is reported in the unit it is measured in — bits for entropies and surprisals, dimensionless for correlations and cosine similarities, cosine distance for embedding displacement, a fraction for a count of steps. Key names carry the unit. Nothing here is normalised into `[0, 1]`, divided by a vocabulary size, or inverted into a `1 − x` score.

Three families of quantity were removed, and this document records why so they do not come back:

- **The `normalized` block.** Unbounded quantities were divided by `log₂(vocab_size)`. That normaliser then surfaced as the strongest apparent "behavioural" feature in the study corpus (r = 0.980, constant within a model) — tokenizer metadata masquerading as behaviour. Bounded scales also saturate, and bits are self-interpreting: 4.9 bits is about the uncertainty of a uniform choice among ~30 tokens, whereas "0.0178" is not checkable against anything.
- **The `levels` block (low/medium/high) and the verdict/equilibrium flags.** Assigning a level is an inference requiring a null distribution this project never established. The decision rule built on the previous levels measured a ~43% false-positive rate on pairs of runs known to be identical.
- **Duplicate names.** `continuity` was `1 − sensitivity` computed from the same JS divergences, and the `wager` aggregate was byte-for-byte the same computation as `surprise`. Reporting one measurement twice under two names inflates the apparent dimensionality of the signal set. Each quantity now appears exactly once in the measurement set.

Source of truth: `MEASUREMENT_REGISTRY` in `hif/profile/signals.py` — one row per measurement carrying its key, name, label, unit, definition, triple, and subject — printed in full by `hif schema`.

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

**The empirical case.** In the predecessor project's audit, the prompt-only measurements showed **exactly zero variance across every model-side change tested** — they were deterministic in prompt text, surrogate weights, and seed. They could not see the model. The same signature reproduces here: profiling `gpt2` and `gpt2-medium` on the same prompt with `--diagnostics` moves every target-side number and leaves `attention_entropy_input_bits` bit-identical (`1.6677721955190443` in both).

**`io_correlation_r` is the genuine mixed case**, and is classified as such rather than lumped either way. Under a surrogate it is the Pearson r between a surrogate-read per-variant input entropy shift and the *target's own* per-variant JSD. The target's data does enter, so the quantity stays in `measurements`; but a correlation cannot be attributed to one of its two series, so its subject degrades to `mixed` and the CLI marks the row. On `[F]` both series are the target's and the subject is `target-distribution`.

**One row is prompt-only on every backend, `[F]` included:** `attention_entropy_input_bits` (Horizon). Attention here is not the target's — `hif/analysis/attention.py` runs a bidirectional analysis encoder over text as an object, and never accesses the generation mechanism of the model under analysis. The output-side row reads the target's actual generated continuation and is therefore `target-output-text`; the input-side row reads the prompt, so it is a function of prompt text and encoder weights alone. No access tier can make it a measurement of the target.

This section is the companion to [Why measure behaviour at all on closed models](#why-measure-behaviour-at-all-on-closed-models) below. That one concedes how little a closed surface exposes and commits to reporting absence rather than approximation when the surface cannot support a quantity; this one draws the line the concession implies — a number produced by something other than the target is not a degraded reading of the target, and the record must not be able to say it is.

## Backend Access

Access is a property of what the backend exposes, not of the model. `hif/models/capabilities.py` is the enforcing authority; `hif models` prints the current table.

| Access | Backends | What is available |
|--------|----------|------------------|
| `[F]` full | `hf`, `tlens`, `hf-vlm` | Full-vocabulary distributions and teacher forcing — every measurement |
| `[T-k]` truncated | `openai`, `openai-vlm`, `gemini`, `ollama` | Top-k logprobs only; output entropy is a lower bound; no teacher forcing |
| `[P]` proxy | `anthropic` | Selected token only. The entropy-shaped measurements degenerate unless a `--surrogate` reads the output text under teacher forcing; the distribution **divergences** (`perturbation_jsd_bits`, `output_step_jsd_bits`, `output_step_topk_overlap_fraction`) are absent outright, and no surrogate recovers them |

The attention-row measurements are **not** in this table, and that is the point: they read an analysis encoder's attention over text (the prompt, or the target's generated continuation), never the target's own attention, so no backend has ever been asked to expose anything for them. They are available on every backend and gated only on the optional stage that produces them (`--diagnostics`). `hif/models/capabilities.py` once claimed the opposite and enforced the claim, which told users their backend could not produce a measurement it produces perfectly well.

The input-side measurements (`input_entropy_shift_bits`, `input_entropy_std_bits`, `prompt_surprisal_excess_bits`) require teacher forcing. On a backend that cannot teacher-force they are either **absent from the record entirely**, or — with `--surrogate` — computed by a small local open-weight model reading the same prompt. In the second case they describe the prompt under that reference model, not the target: their subject is `prompt-only`, so they leave `measurements` for the `prompt_measurements` block (see [Subject](#subject--whose-behaviour-the-number-describes)). `io_correlation_r` needs the same teacher forcing but keeps the target's output response as half its computation, so it stays in `measurements` with subject `mixed`. `hif models` prints, per backend, which measurements degrade this way.

Absent is never zero. A measurement the run produced no evidence for is omitted from `measurements`, because "no evidence" and "measured zero" are different statements. Absent also covers *measured something else*: a quantity whose subject on this backend is the prompt is omitted rather than emitted with a caveat.

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

The measurement set is defined once, in `MEASUREMENT_REGISTRY` in `hif/profile/signals.py`, and extracted by `measurements(profile)` — run `hif schema` for the current set. The same function feeds the CLI table and the machine record, so a number shown in a terminal and a number in a JSONL line can never diverge. Each is a triple: observable × functional × resolution; the record carries the run-level scalar, and rows with `per-step` / `per-position` resolution also have a trace in Part 2.

| Key | Label | Unit | Resolution | Subject | Requires |
|-----|-------|------|------------|---------|----------|
| `input_entropy_shift_bits` | — | bits | aggregate | `target-distribution` → `prompt-only` | teacher forcing (or `--surrogate`) + perturbation variants |
| `input_entropy_std_bits` | Stability | bits | aggregate | `target-distribution` → `prompt-only` | teacher forcing (or `--surrogate`) + ≥ 2 perturbation variants |
| `perturbation_jsd_bits` | Sensitivity | bits | aggregate | `target-distribution` | perturbation variants + top-k logprobs |
| `io_correlation_r` | — | dimensionless | aggregate | `target-distribution` → `mixed` | ≥ 2 perturbation variants with both sides measured |
| `io_cosine_similarity` | — | dimensionless | aggregate | `target-output-text` | ≥ 1 perturbation variant + an embedding encoder |
| `prompt_surprisal_excess_bits` | Wager ▲ | bits | per-position | `target-distribution` → `prompt-only` | teacher forcing (or `--surrogate`) |
| `candidate_cluster_entropy_bits` | — | bits | per-step | `target-distribution` → `target-output-text` | top-k logprobs + an embedding encoder |
| `output_entropy_bits` | Entropy ● | bits | per-step | `target-distribution` → `target-output-text` | top-k logprobs |
| `output_entropy_step_delta_bits` | — | bits | per-step | `target-distribution` → `target-output-text` | top-k logprobs, ≥ 2 steps |
| `output_step_jsd_bits` | Shift ◆ | bits | per-step | `target-distribution` | top-k logprobs (real distributions, not the selected token alone), ≥ 2 steps |
| `output_step_topk_overlap_fraction` | — | fraction of shared top-K token ids | per-step | `target-distribution` | same as `output_step_jsd_bits` |
| `semantic_centroid_veer_cosine` | Veer ◈ | cosine distance | per-step | `target-distribution` → `target-output-text` | `semantic_field.enabled` + an embedding encoder |
| `attention_entropy_output_bits` | Spread ■ | bits | per-position | `target-output-text` | the attention-analysis stage (`--diagnostics`); any backend |
| `attention_entropy_input_bits` | Horizon | bits | per-position | `prompt-only` (every backend) | the attention-analysis stage (`--diagnostics`); any backend |
| `counterfactual_exposure_fraction` | Exposure ◇ | fraction of steps | per-step | `target-distribution` → `target-output-text` | top-k logprobs + an embedding encoder |
| `branch_pairwise_cosine_similarity` | Continuity | dimensionless | aggregate | `target-output-text` | trajectory analysis with ≥ 2 branches + an embedding encoder |

A label is a canonical shorthand carried over from the framework's instrument vocabulary; a measurement without one has no established shorthand, and none was invented for it.

The Subject column reads `declared` → `under surrogate`: the first value holds when the target's own machinery produced the quantity, the second when the surrogate named by the row's `surrogate_group` stood in. A single value means the subject does not change. See [Subject](#subject--whose-behaviour-the-number-describes).

---

### `input_entropy_shift_bits`

**Zone.** Input side, under perturbation.

**Definition.** Mean absolute difference, over perturbation variants, between the variant's mean input-token entropy and the baseline's.

```
input_entropy_shift_bits = mean_v | μ[H(Pᵢ)]_variant_v − μ[H(Pᵢ)]_baseline |

where H(Pᵢ) = −∑ₜ pₜ log₂ pₜ  (teacher-forced full-vocabulary entropy at prompt position i)
```

**Unit and range.** Bits. Unbounded above. `0` means the perturbations moved input-side entropy not at all.

**Not inverted, not normalised.** This replaces the former "Input Stability" (`1 − mean|Δ stability_score|`), which saturated at exactly 1.0 in the regime that mattered and divided by `log₂(vocab_size)`. It carries no label for the same reason: the historical name named the inverted score, not this quantity.

**Absent when** there are no perturbed input-side analyses — the backend cannot teacher-force and no surrogate was supplied. Computed in `hif/metrics/stability.py::compute_stability_metrics`.

---

### `input_entropy_std_bits`

**Zone.** Input side, under perturbation. Label: **Stability**.

**Definition.** Sample standard deviation of the per-variant input entropy shifts — the spread of the model's entropy response across perturbations, where `input_entropy_shift_bits` is its mean.

```
input_entropy_std_bits = std_v( | μ[H(Pᵢ)]_variant_v − μ[H(Pᵢ)]_baseline | ),  ddof = 1
```

**Unit and range.** Bits. Unbounded above. `0` means every variant moved input-side entropy by exactly the same amount.

**History.** Added in hif-v2.1 — the natural-unit form of the Stability aggregate, which was computed but never surfaced while the inverted `1 − x` stability scores existed.

**Absent when** fewer than two perturbation variants exist (a single shift has no spread), or the backend cannot teacher-force and no surrogate was supplied. Computed in `hif/metrics/stability.py::compute_stability_metrics`.

---

### `perturbation_jsd_bits`

**Zone.** Perturbation. Label: **Sensitivity**.

**Definition.** Mean Jensen-Shannon divergence between the baseline output distribution and each paraphrase variant's, averaged across generators and generation steps.

```
perturbation_jsd_bits = mean_v [ mean_j [ JSD(P_baseline,j ‖ P_variant,j) ] ]

JSD(P ‖ Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),  M = ½(P + Q),  logs base 2
```

**Unit and range.** Bits. Genuinely bounded to `[0, 1]` by definition in log base 2 — that bound is a property of JSD, not a rescaling. It is reported as measured, no longer inverted into an "Output Stability" score.

**Perturbation generators.** Default `["synonym", "tone", "reorder"]` with `n_variants = 2` each. `substitution` and `ambiguity` are implemented and selectable; LLM-backed paraphrasing is opt-in and requires an explicit endpoint.

**Fallback.** When the aggregate is absent but per-perturbation `SensitivityMetrics` exist, `measurements()` averages `mean_js_divergence` over them — same quantity, same unit.

**Absent when** the backend returns only the selected token (the `[P]` tier). This is the point-mass rule, and it is an absence rather than a caveat for the same reason a prompt-only quantity leaves `measurements`: *the computation stops being the one the key names.* The JSDs are taken over the RAW baseline and variant traces (`build_profile` step 6, before any surrogate recovery), so on a selected-only backend both sides are point masses. The divergence between two point masses is `0` when the selected tokens agree and exactly `1` bit when they differ — a **token-disagreement rate**, not a divergence between distributions. The rate is not re-admitted under another key either: it would have to pass the Significance Gate on its own, and nothing has shown a run that needs it. `--surrogate` does not rescue this measurement — the step-6b recovery rebuilds `semantic_steps`, which the sensitivity path never reads.

---

### `io_correlation_r`

**Zone.** Center — the coupling between the two sides.

**Definition.** Pearson correlation between the per-variant input entropy shift and the per-variant output JSD.

```
io_correlation_r = Pearson( [ |Δ mean input entropy|_v ]ᵥ , [ mean_j JSD_v,j ]ᵥ )
```

One point per perturbation variant, not per token position.

**Unit and range.** Dimensionless, bounded to `[−1, 1]` by definition. Reported **signed and un-clamped** — the sign is the interesting part. Positive: perturbations that shift the input distribution also shift the output (coherent sensitivity). Near zero or negative: output sensitivity is independent of input-distribution fit.

**Absent vs. zero.** `None` (omitted) when there are fewer than two aligned points — a single perturbation variant has no correlation to report. Exactly `0.0` only when the correlation is computable but degenerate (a constant series makes `r` undefined). Reporting the first case as a measured `0.0` would misrepresent "no evidence" as "measured zero correlation".

> The **I/O Correlation chart** in `hif/viz/signals/io_correlation.py` plots a *different* quantity under a similar name: the Pearson `r` between the input entropy trace and the output entropy trace, each resampled to a shared 100-point grid. It is a per-position trace comparison, not the per-variant coupling measured here. Do not read one as the other.

---

### `io_cosine_similarity`

**Zone.** Perturbation (cross-pair).

**Definition.** Mean cosine similarity between each input embedding and its paired output embedding — the `io_sim` member of `SimilarityMetrics`.

```
io_sim = mean_i cos( embed(input_i), embed(output_i) )
```

over all `(input, output)` pairs: baseline plus one per perturbation variant.

**Unit and range.** Dimensionless, bounded to `[−1, 1]` by definition. High: output stays in the semantic neighbourhood of its input. Low: output sits far from the prompt's representational space.

**Companions (persisted on `metrics.similarity`, not in the measurement set).** `input_sim` and `output_sim` are the mean pairwise cosines *within* the input set and *within* the output set; `io_ratio = output_sim / input_sim` captures amplification vs. suppression (`> 1` = outputs converge more than inputs did; `< 1` = the model amplifies input variation). `trend` is the linear slope of per-step mean pairwise similarity across the output sequence, derived from `SemanticMetrics.mean_pairwise_distance`; it is surfaced separately as `findings.similarity_trend_slope`, signed and unrounded.

**Absent when** there are no perturbation variants — at least one variant alongside the baseline is required.

**Encoder-dependent.** Comparable only between profiles computed with the same embedding model, which is recorded in `config.embedding.model_name`.

---

### `prompt_surprisal_excess_bits`

**Zone.** Selection, input side. Label: **Wager ▲** — the per-position trace is in Part 2.

**Definition.** Mean excess surprisal over per-position entropy across teacher-forced prompt positions.

```
prompt_surprisal_excess_bits = (1/T) ∑ᵢ max(0, sᵢ − H(Pᵢ))

where sᵢ = −log₂ p(tokenᵢ | ctx<ᵢ)  (surprisal of the actual token)
and H(Pᵢ) = −∑ₜ pₜ log₂ pₜ         (Shannon entropy of the full distribution at position i)
```

The surprisal `sᵢ` is unweighted by probability — this is deliberate. Entropy is the probability-weighted average of surprisal: `∑ p(x)·(−log p(x))`. That weighting compresses the logarithmic spread: the product `p·(−log p)` is arch-shaped, peaking near `p = 1/e ≈ 0.37` and falling to zero at both extremes. High-probability tokens have small surprisals but large weights; low-probability tokens have large surprisals but small weights; the two effects pull toward each other. This compression makes entropy a stable average — but it dulls exactly the signal this measurement is after. The unweighted surprisal lets the logarithmic spread do what it naturally does: make rare events stand out. A token at 1% probability produces a surprisal of ~6.6 bits; weighted by 0.01 it contributes only 0.066 — nearly silent, precisely when it should be loudest.

When `sᵢ > H(Pᵢ)` the actual token was more surprising than the distribution's own average uncertainty — an "underdog" against a concentrated distribution.

**Worked example.** At position `i`: `H(Pᵢ) = 2.90 bits`. The actual token sits at rank 5 with probability 2.7%, so `sᵢ = −log₂(0.027) ≈ 5.2 bits`. Excess = 5.2 − 2.90 = 2.3 bits. A position whose actual token was the model's top-1 contributes 0.

**Unit and range.** Bits, `[0, ∞)`. Unbounded above. There is no `/ 5.0` ceiling and no normalised variant: the linear clamp that used to produce `normalized_surprise = min(Surprise / 5.0, 1.0)` existed only to make this number sit alongside six other `[0, 1]` scores, and those scores are gone.

**Absent when** no teacher-forced positions exist. Computed by `hif/hourglass/input_side.py::mean_surprisal_excess`.

---

### `candidate_cluster_entropy_bits`

**Zone.** Output side, geometric.

**Definition.** Mean over generation steps of the Shannon entropy of the probability mass distributed across the step's semantic clusters (see *Cluster Entropy* in Part 3).

**Unit and range.** Bits. Unbounded above in principle, bounded in practice by `log₂` of the cluster count. High: genuinely competing semantic directions. Low: one direction dominates even where alternatives exist.

**Absent when** no per-step semantic metrics were computed.

---

### `output_entropy_bits`

**Zone.** Output side. Label: **Entropy ●** — the per-step trace is in Part 2.

**Definition.** Mean over generation steps of the Shannon entropy of the per-step top-K output distribution (`DistributionMetrics.entropy_bits`).

**Unit and range.** Bits. **A lower bound** on full-vocabulary entropy whenever the distribution is truncated to top-k, and **not comparable across backends with different k**. `DistributionMetrics.entropy_bits_upper` carries the uniform-tail upper bound when the vocabulary size is known.

---

### `output_entropy_step_delta_bits`

**Zone.** Output side, step-local.

**Definition.** Mean absolute step-to-step change in *nucleus* entropy.

```
output_entropy_step_delta_bits = mean_{i≥2} | H_nucleus(step i) − H_nucleus(step i−1) |
```

Nucleus entropy (95% mass, renormalised) is used rather than raw top-K entropy so the trace is comparable across backends regardless of how many logprobs each exposes.

**Not Shift.** This is deliberately *not* labelled Shift (◆). Shift is `output_step_jsd_bits`, the next section — the change in *where the mass sits*; this is the step-to-step change in the *amount* of uncertainty. Two consecutive steps can carry identical entropy over completely disjoint token sets, in which case this quantity reads `0` where Shift reads its `1`-bit ceiling. They are not substitutes and never share a label.

**Unit and range.** Bits. Unbounded above.

**Absent when** fewer than two generation steps were recorded.

---

### `output_step_jsd_bits`

**Zone.** Output side, step-local. Label: **Shift ◆** — the per-step trace is in Part 2.

**Definition.** Mean over transitions of the Jensen-Shannon divergence between *consecutive* generation steps' output distributions. The target's own distributions throughout; no surrogate is involved.

```
output_step_jsd_bits = mean_{j≥2} [ JSD(Q_{j−1} ‖ Q_j) ]
```

The two steps' top-K distributions are aligned over the union of their token ids and renormalised before the divergence is taken — the same computation the Shift chart draws, because both call `hif/metrics/shift.py`. That module is the single source of truth: the number in a record and the bars on a chart are one arithmetic and cannot drift.

**Unit and range.** Bits, bounded to `[0, 1]` by definition in log base 2 — a property of JSD, not a rescaling.

**Read it with `output_step_topk_overlap_fraction`.** The divergence is computed over the stored top-K supports, not the full vocabulary, and two consecutive steps whose top-K sets are **disjoint** give exactly `1` bit however similar their true full-vocabulary distributions are. The ceiling can therefore be reached by truncation alone. The companion measurement below reports how much support the same transitions actually shared, which is how much of the divergence is evidence rather than artifact. See the caveat discussion under Part 2 → ◆ Shift for why the resolution chosen was a companion measurement rather than absence below an overlap floor.

**Absent when** fewer than two generation steps were recorded, or the backend returns only the selected token. The second is the same point-mass rule as `perturbation_jsd_bits`: between two point masses this is a token-disagreement indicator, not a divergence between distributions. Note that this measurement deliberately reads the *raw* `output_side.steps` — the series the chart reads — so no surrogate stands in for it; a `[P]` backend gets absence, not a proxy.

**History.** Admitted to the measurement set in hif-v3.1. Shift had existed since the framework's beginning as a chart and a Part 2 trace with no key in the record, so a reader could see "Shift ◆" on the companion website and had no way to reproduce it with the CLI — precisely the gap the project exists to close.

---

### `output_step_topk_overlap_fraction`

**Zone.** Output side, step-local. No label — it has no established shorthand in the framework's instrument vocabulary, and none was invented.

**Definition.** Mean over the same transitions of the Jaccard overlap between consecutive steps' top-K candidate token-id sets.

```
output_step_topk_overlap_fraction = mean_{j≥2} [ |A_{j−1} ∩ A_j| / |A_{j−1} ∪ A_j| ]
```

**Unit and range.** A fraction of shared top-K token ids, bounded to `[0, 1]` by construction.

**What it is for.** It is the **resolution limit** on `output_step_jsd_bits`. At `0` the two steps share no candidate at all and the divergence is pinned at its ceiling by the truncation; a high Shift over low overlap is weaker evidence of a vocabulary pivot than the same Shift over high overlap. It also stands on its own: how much of a model's viable-token set survives from one step to the next is a fact about the model, and no other admitted measurement discloses it — the entropy measurements read the *shape* of each step's distribution and are blind to whether the tokens are the same ones.

**Absent when** `output_step_jsd_bits` is absent, under the same rules.

---

### `semantic_centroid_veer_cosine`

**Zone.** Output side, geometric. Label: **Veer ◈** — the per-step trace is in Part 2.

**Definition.** Mean step-to-step displacement of the candidate cloud's probability-weighted semantic centroid in embedding space.

```
cⱼ = Σᵥ pⱼ(v)·e(v) / Σᵥ pⱼ(v)        (semantic centroid of step j's top-K cloud)
Veerⱼ = 1 − cos(cⱼ, cⱼ₋₁)
semantic_centroid_veer_cosine = mean_j Veerⱼ
```

Each candidate is embedded with up to `context_window` (default 5) already-generated tokens of left context, so the reading reflects the candidate in context rather than the bare token.

**Unit and range.** Cosine distance, bounded to `[0, 2]` by definition; in practice `[0, 1]` for a sentence-embedding encoder.

**Absent when** `config.semantic_field.enabled` is False (the default) or fewer than two steps have a defined centroid. `hif profile --diagnostics` enables it. Computed by `hif/analysis/semantic_field.py`, compute-and-discard: only the scalar traces survive the call.

---

### `attention_entropy_output_bits` / `attention_entropy_input_bits`

**Zone.** Attention (an independent reader's, not the generating model's). Labels: **Spread ■** (output side) and **Horizon** (input side) — the per-position traces are in Part 2.

**Definition.** Mean Shannon entropy of the causal-prefix attention row, at each output position and at each input position respectively.

```
row i is restricted to columns 0..i, renormalised to a distribution, then
Hᵢ = −∑ⱼ≤ᵢ āᵢⱼ log₂ āᵢⱼ
```

**Unit and range.** Bits. **Grows with prefix length by construction** — read against the position axis, not as a fraction. It is deliberately *not* divided by `log₂(prefix length)`: that denominator is the sequence length, so it puts position metadata into a number labelled behaviour, the same mistake as the removed vocabulary-size normaliser.

**Source.** `profile.attention_capture`, a `TextAttentionAnalysis` produced by the DistilBERT reader in `hif/analysis/attention.py`, already aggregated across heads and layers. Extracted by `hif/viz/signals/_attention.py::row_entropy_trace`.

**Subject — the two sides differ, and it matters.** The reader is the same for both, but the text it reads is not. `attention_entropy_output_bits` reads the target's *actual generated continuation*: a fixed instrument on the target's real output, so its subject is `target-output-text` and it moves when the target's output moves. `attention_entropy_input_bits` reads the *prompt*, so it is a function of prompt text and encoder weights alone — subject `prompt-only`, on every backend including `[F]`, and reported in the `prompt_measurements` block rather than in `measurements`. It is a real measurement of the prompt under a reference encoder; it is not a measurement of the target, and no access tier can make it one. See [Subject](#subject--whose-behaviour-the-number-describes).

**Absent when** attention analysis did not run (`config.attention.enabled` is False by default).

---

### `counterfactual_exposure_fraction`

**Zone.** Output side, geometric. Label: **Exposure ◇** — the per-step reading is in Part 2.

**Definition.** The fraction of analysed generation steps at which a probabilistically accessible alternative token would have pulled the response toward a different meaning.

```
E = |{ t : diffusion(t) ∧ d_t ≥ τ }| / G

where d_t = max_v dist(e(prefix + tok_t), e(prefix + v)) over candidates v with p_t(v) ≥ p_min
```

Defaults: `p_min = 0.01`, `τ = 0.3`. The shared context is the *full* generated prefix, so the comparison holds the whole response-so-far fixed and varies only the final token. `diffusion(t)` restricts the count to steps whose candidate cloud is in the diffusion zone.

**Unit and range.** A proportion, bounded to `[0, 1]` by construction.

**What it does not see.** Only diffusion-zone steps are counted. The convergence case — a model that is confident and narrow but aimed wrong — is excluded by construction. **This is not a factuality judgment.** A confident response can still be wrong, and this measurement does not see that case.

**Encoder-dependent.** Distances are embedding-space-dependent; values are comparable only between profiles computed with the same encoder.

---

### `branch_pairwise_cosine_similarity`

**Zone.** Trajectory — the stochastic branch cloud. Label: **Continuity**.

**Definition.** Mean pairwise cosine similarity between the trajectory branch embeddings: the trajectory stage re-samples the generation from a branch point several times, embeds each branch's text, and averages the cosine similarity over all pairs.

```
branch_pairwise_cosine_similarity = mean_{a<b} cos( e(branch_a), e(branch_b) )
```

**Unit and range.** Dimensionless, bounded to `[−1, 1]` by definition. High: independently sampled branches converge semantically; low: they scatter.

**History.** Added in hif-v2.1 — the natural-unit form of the Continuity aggregate, which was computed but previously reduced to a derived score and never surfaced directly. The branch field's `field_dispersion` (Part 4) is `1 − this` — the same quantity as a distance; the measurement set carries the similarity form once.

**Absent when** the trajectory stage was skipped or degenerate, or fewer than two branches were embedded.

**Encoder-dependent.** Comparable only between profiles computed with the same embedding model, which is recorded in `config.embedding.model_name`.

---

## Part 2 — The Token-Level Traces

A run's scalar measurements (Part 1) compress a full generation into one number per quantity. The token-level traces restore what compression hides: which specific tokens drove a value, where entropy and attention converge or diverge step-by-step, and whether a flat mean reflects genuine uniformity or cancellation between extremes. These are the same measurements at their native resolution — a row whose `resolution` is `per-step` or `per-position` has its trace here.

The traces with a chart live in the viz registry (`hif/viz/registry.py`, `kind="reading"`); Veer has no chart, but its per-step trace is persisted on `profile.semantic_field` and its mean is reported as `semantic_centroid_veer_cosine`. Every trace here now has an aggregate in the measurement set. Shift was the exception until hif-v3.1 — it existed as a chart and nothing else, so the instrument was visible on the companion website and unreachable from the CLI. That is fixed: its run-level mean is `output_step_jsd_bits`, and the chart and the key call one function (`hif/metrics/shift.py`) so they cannot drift apart.

| Symbol | Name | Space measured | Measurement (Part 1) | Access |
|--------|------|----------------|----------------------|--------|
| ▲ | Wager | Surprisal excess over entropy, per prompt position | `prompt_surprisal_excess_bits` | Teacher forcing (open-weight, or `--surrogate`) |
| ● | Entropy | Output distribution entropy, per generation step | `output_entropy_bits` | All models with logprobs |
| ■ | Spread | Attention-row entropy over context positions, per generated token | `attention_entropy_output_bits` | The attention-analysis stage (`--diagnostics`) — any backend |
| ◆ | Shift | Step-to-step JSD between consecutive output distributions | `output_step_jsd_bits` | All models with real top-k distributions, ≥ 2 steps |
| ◈ | Veer | Step-to-step displacement of the candidate cloud's semantic centroid | `semantic_centroid_veer_cosine` | Top-k probs + an embedding encoder (`--diagnostics`) |
| — | Horizon | Attention-row entropy per prompt position | `attention_entropy_input_bits` | The attention-analysis stage (`--diagnostics`) — any backend |
| ◇ | Exposure | Fraction of steps where an accessible alternative diverged in meaning | `counterfactual_exposure_fraction` | Top-k probs + an embedding encoder |

Horizon carries no glyph in the registry and is labelled *Input attention entropy* there; the ▼ symbol it once used is not in the code and is not used below.

Reading these traces simultaneously — across vocabulary space, embedding (semantic) space, and context-position space — is the multi-register practice the framework is designed to support. Shift (◆) and Veer (◈) are twins at the same resolution: Shift reads the step-to-step change in the distribution's *spread* (vocabulary space); Veer reads the step-to-step change in its *semantic location* (embedding space).

---

### ▲ Wager

**Formula.** `Wagerᵢ = max(0, sᵢ − H(Pᵢ))`

where `sᵢ = −log₂ p(tokᵢ | ctx<ᵢ)` is the surprisal of prompt token `i` and `H(Pᵢ)` is the Shannon entropy of the full-vocabulary distribution the model predicted at that position, both from the teacher-forced pass over the prompt.

**What it shows.** Surprisal excess per prompt position. Each bar shows how many bits the chosen token's surprisal exceeded the model's distributional entropy at that position — the residual cost of the actual token beyond general uncertainty. A tall bar at position `i` means the model had committed to a narrow distribution but the actual token was a long shot against that commitment.

The surprisal `sᵢ` is unweighted by probability — this is deliberate. Entropy is the probability-weighted average of surprisal: `∑ p(x)·(−log p(x))`. That weighting compresses the logarithmic spread: the product `p·(−log p)` is arch-shaped, peaking near `p = 1/e ≈ 0.37` and falling to zero at both extremes. High-probability tokens have small surprisals but large weights; low-probability tokens have large surprisals but small weights; the two effects pull toward each other. This compression makes entropy a stable average — but it dulls exactly the signal Wager is after. The unweighted surprisal lets the logarithmic spread do what it naturally does: make rare events stand out. A token selected with 1% probability produces a surprisal of ~6.6 bits; weighted by 0.01, it contributes only 0.066 — nearly silent, precisely when it should be loudest.

The interesting case is when `sᵢ` and `H(Pᵢ)` diverge: low `H(Pᵢ)` means the distribution is narrow and confident, but high `sᵢ` means the actual token is not what the model was confident about. `sᵢ − H(Pᵢ) > 0` is the excess — the model committed, and the token overrode that commitment. When the excess is zero, the model either selected its most likely token or was already broadly uncertain.

**Relation to `prompt_surprisal_excess_bits`.** The measurement runs `max(0, sᵢ − H(Pᵢ))` over all positions and averages to a single number; the Wager trace is that same quantity at full per-position resolution. The Surprise chart in the registry draws the same underlying series with the mean called out. They are one quantity at two resolutions, which is why the measurement set carries it only once — the historical `wager` aggregate was byte-for-byte identical to the `surprise` aggregate, and reporting one number twice under two names inflated the apparent dimensionality of the signal set.

**Expected range.** `[0, ∞)` bits. Most positions contribute zero. Large values at specific positions identify structurally surprising tokens — places where the model had committed and the actual token overrode that commitment.

**Access.** Requires teacher forcing: an open-weight backend (`hf`, `tlens`, `hf-vlm`), or a `--surrogate` proxy teacher-forced over the same prompt. A surrogate reading describes the surrogate, and is flagged as such via `findings.surrogate_model_name`.

---

### ● Entropy

**Formula.** `Entropyⱼ = H(Qⱼ) = −∑ᵥ Qⱼ(v) log₂ Qⱼ(v)`

Sum over the top-K candidates the backend returned at each generation step `j = 1…G`.

**What it shows.** Output distribution entropy per generation step, in bits. Peaks mark genuine decision moments where many tokens were competitive; troughs mark committed choices where the model narrowed sharply.

The chart draws two series: the **nucleus entropy** (95% mass, renormalised — comparable across backends regardless of how many logprobs each exposes) and the **raw top-K entropy** (the truncation lower bound). The trace is the full shape that a single mean compresses: it shows whether that mean conceals a flat plateau, a single tall spike, or alternating peaks and troughs — patterns that carry distinct interpretive weight.

**Relation to Breadth / ESS.** Entropy (●) and Effective Support Size are the same underlying signal in different units: `ESS = 2^H_nucleus`, an equivalent token count. The Breadth chart draws the ESS trace with its mean; the measurement set reports `output_entropy_bits` and not ESS, because a quantity appears once.

**Truncation.** Whenever the distribution is top-k truncated, the reported entropy is a **lower bound** on true full-vocabulary entropy, and values are not comparable across backends with different k. `DistributionMetrics.entropy_bits_upper` carries the uniform-tail upper bound where the vocabulary size is known.

**Expected range.** `[0, log₂ K]` bits — about 5.64 bits for K=50. The full-vocabulary ceiling `log₂|V|` (≈ 15.6 bits for a 50,257-token vocabulary) is not reachable from a truncated distribution.

**Access.** All models with output logprob data. On a selected-token-only backend the distribution degenerates unless a `--surrogate` recovers it.

---

### ■ Spread

**Formula.** `Spreadᵢ = H(āᵢ,₀:ᵢ)`

Row `i` of the stored continuation attention map is restricted to columns `0..i` (its causal prefix), renormalised to a probability distribution, and its Shannon entropy taken in bits (`hif/viz/signals/_attention.py::row_entropy_trace`).

**What it shows.** Attention spread over context positions. How evenly attention was distributed across prior context at each token. A value of `k` bits means approximately `2ᵏ` context positions received meaningful weight. High Spread: attention is diffuse across many positions. Low Spread: attention is concentrated on a few.

**Whose attention.** Not the generating model's. The attention map is produced by the bidirectional reader in `hif/analysis/attention.py` (DistilBERT by default) reading the generated continuation as a text, already aggregated across heads and layers by `AttentionConfig.aggregate_method` (default `mean_all_layers`; `last_layer` and `mean_upper_half` are the alternatives). There is no per-layer selection and no middle-layer isolation. This is a reading of the text's structure, not an inspection of how the text was produced.

Spread is measured in context-position space; Entropy (●) is measured in vocabulary space. These are orthogonal dimensions. A model can attend narrowly (low Spread) while remaining uncertain about which token to select next (high Entropy), or attend broadly while being highly confident. The two readings can and do move in opposite directions.

**Expected range.** `[0, log₂(i+1)]` bits — the ceiling grows with position as more prefix becomes available. The value is reported in raw bits and is deliberately **not** divided by `log₂(prefix length)`; read it against the position axis, not as a fraction.

**Access.** Requires the attention-analysis stage, which runs only when `AttentionConfig.enabled` is set — `hif profile --diagnostics` does so. That is the *only* requirement. The reader is an analysis encoder over the target's generated text, so `attention_entropy_output_bits` is available on every backend; `hif/models/capabilities.py` used to gate it to `hf`/`tlens`/`hf-vlm`, which was a false claim about the backend and has been removed.

---

### ◆ Shift

**Formula.** `Shiftⱼ = JSD(Qⱼ₋₁, Qⱼ)` — step-to-step Jensen-Shannon divergence of the output distribution, `j = 2…G`. The two steps' top-K distributions are aligned over the union of their token ids and renormalised before the divergence is taken.

**What it shows.** Step-to-step divergence within a single forward pass. Tall bars mark abrupt vocabulary pivots — the field of viable tokens reorganized sharply between steps `j−1` and `j`. Low bars mark smooth continuation, the distribution changing little as the model extends an established direction.

**Measurement caveat (real, not cosmetic).** JSD is computed only over the stored top-K candidates, not the full vocabulary. When two consecutive steps' top-K sets share little or no overlap, JSD saturates at exactly 1 bit regardless of how similar the true full-vocabulary distributions are — disjoint support alone is enough to hit the ceiling. A chart where most bars sit near 1 more often reflects narrow top-K supports failing to overlap than genuine maximal divergence. The chart therefore surfaces the top-K overlap fraction in the hover, and shows a banner when overlap is low, so this is not silently mistaken for "everything is maximally different".

**How the caveat is carried into the measurement.** A caveat that lives only in a chart's hover text is not available to anyone reading a record, so `output_step_topk_overlap_fraction` (Part 1) reports the same overlap as a measurement in its own right, computed from the same trace as the divergence. Two resolutions were considered and one was rejected:

- **Companion measurement (chosen).** The overlap ships beside the divergence, with each of the two definitions naming the other. This is the treatment `output_entropy_bits` already gets — a quantity that is truncation-limited in a stated direction is reported *with* its bound stated, not withheld.
- **Absence below an overlap floor (rejected).** Measured on a real `gpt2` run (`--top-k 50`, 38 steps), the median consecutive top-K overlap is about **0.08** while the per-step divergence ranges from **0.10 to 1.00**, and only about a sixth of transitions sit at the ceiling. Any floor that run would fail — and it would fail every plausible one — deletes a number that demonstrably still discriminates between transitions. Suppressing a working measurement is not a more conservative choice than reporting it with its resolution limit attached; it is a different error.

The line between the two, and the reason this is not the discarded "caveat flag" pattern: a flag is an adornment on a number that a consumer must know to look for, whereas the overlap is a second **fact**, with its own registry row, unit, definition and absence rule. Absence is reserved for the case where the computation stops being the quantity at all — which is exactly what happens on a selected-only backend, below.

**Absent when** the run has fewer than two generation steps, or the backend returns only the selected token. In the second case consecutive steps are point masses: the "divergence" is `0` when the two selected tokens agree and exactly `1` bit when they differ, which is a token-disagreement indicator and not a divergence between distributions. The same rule governs `perturbation_jsd_bits` (Part 1). The chart says so too, on its unavailable panel, from the same reason string.

**Distinction from other quantities.** Unlike the input entropy trace (prompt-side, before generation), and unlike Continuity / Trajectory (which compares independently sampled branches), Shift operates entirely within one forward pass. It is a within-run, step-local measure of distributional change. It is also distinct from `output_entropy_step_delta_bits`, which is the step-to-step change in the *amount* of uncertainty; Shift is the step-to-step change in *where the mass sits*.

**Expected range.** `[0, 1]` bits (JSD in log base 2 is bounded by definition).

**Access.** All models with real top-k distributions and at least two generation steps; absent on the `[P]` tier per the rule above. There is no attention-domain variant of Shift in this package.

**Measurement.** `output_step_jsd_bits` (Part 1), with `output_step_topk_overlap_fraction` as its companion. The chart and the measurement import the same functions from `hif/metrics/shift.py`.

---

### ◈ Veer

**Formula.** `Veerⱼ = 1 − cos( cⱼ , cⱼ₋₁ )`

where `cⱼ = Σᵥ pⱼ(v)·e(v) / Σᵥ pⱼ(v)` is the probability-weighted mean embedding — the *semantic centroid* — of the top-K candidate tokens at generation step `j`, `e(·)` is the embedding encoder, and `cos` is cosine similarity. Each candidate is embedded within a short window of left-context so the reading reflects the candidate in context, not the bare token.

**What it shows.** How far the *semantic center* of the model's candidate cloud moved between consecutive steps — the step-to-step velocity of the output's possibility field through embedding space. Low, steady Veer means coherent development around a stable topic; a tall Veer marks a semantic pivot, where the field of what the model is about to say relocates to a different region of meaning. A companion **deformation** trace, `|dispersionⱼ − dispersionⱼ₋₁|`, reads whether the field is widening or fragmenting between steps — its change in *shape*, separately from where its center moved.

**Distinction from Shift (◆).** Veer is the geometric twin of Shift. Shift measures the step-to-step change in the *spread* of the output distribution in vocabulary space (information-theoretic); Veer measures the step-to-step change in the *semantic location* of the output distribution in embedding space (geometric). They are independent: a model can hold a steady spread (low Shift) while the meaning of its candidates drifts (high Veer), or hold its meaning (low Veer) while the spread reshapes (high Shift).

**Distinction from `prompt_output_cosine_distance`.** That center diagnostic (formerly "Semantic Drift") reports a single cosine distance between the prompt embedding and the generated-text embedding — one endpoint number for the whole generation. Veer is the per-step trace of the same idea: it shows *where along the generation* the semantic center moved. Neither is evidence that a model drifted; both are distances between embeddings from a single run.

**Expected range.** `[0, 2]` (cosine distance); in practice `[0, 1]` for a sentence-embedding encoder. Most steps are small; spikes identify the steps at which the output's meaning pivoted.

**Access.** All models exposing top-K candidate probabilities, plus an embedding encoder — from the model's own top-K for open and truncated backends, and via a `--surrogate` for selected-only backends. Off by default (`SemanticFieldConfig.enabled = False`) because it re-embeds every step's candidate cloud; `hif profile --diagnostics` turns it on. Encoder-dependent: comparable only between profiles computed with the same encoder, which is recorded in `config.embedding.model_name`.

**Privacy.** Compute-and-discard. Candidate embeddings and the per-step centroids live only in the analyzer's stack frame; only the scalar traces (cosine distances) are returned and persisted.

---

### ◇ Exposure

**Formula.** `E = |{t : diffusion(t) ∧ d_t ≥ τ}| / G`

where `d_t = max_v dist(e(tok_t), e(v))` over candidates `v` in the top-K with `p_t(v) ≥ p_min` (default 0.01), `e(·)` is the embedding encoder, `τ` is the distance threshold (default 0.3), `diffusion(t)` marks steps whose candidate cloud is in the diffusion zone, and `G` is the number of generation steps analyzed.

**What it shows.** How often high-probability alternatives at a step would have pulled the response toward a different meaning — the response's sensitivity to sampling chance. A high-exposure step is one where the model could cheaply (probabilistically) have said something semantically different. This is a measure of exposure to alternative meanings, **not a factuality judgment about any output**.

**What it does not see.** Exposure is computed only over diffusion-zone steps. The convergence case — a model that is confident and narrow but aimed wrong — is excluded by construction. A confident response can still be wrong, and this reading does not see that case.

**Expected range.** `[0, 1]`.

**Access.** All models exposing top-K probabilities, plus an embedding encoder. Distances are embedding-space-dependent: values are comparable only between profiles computed with the same encoder (the encoder is recorded in the profile).

**Since.** HIF Signal Set v1.1. Implementation: `hif/analysis/exposure.py::ExposureAnalyzer`. Exposure is a measure of a response's exposure to a semantically divergent alternative under sampling chance — explicitly not a factuality judgment about any output.

---

### ▼ Horizon

**Formula.** `Horizon_i = H(ā_{i,0:i}) / log₂(seq_len)`

where `ā_{i,0:i}` is the mean-head, mean-layer attention row at prompt position `i` over its causal prefix, normalized to a probability distribution, and the entropy is normalized by `log₂(seq_len)` to bound the result to `[0, 1]`.

**What it shows.** Self-attention diffuseness per prompt position. Low Horizon: the position's attention is concentrated on a few prior tokens. High Horizon: attention is spread broadly across the prefix.

**Correction.** This section previously described Horizon as "an internal measure — it reads the generating model's own attention". It does not, and never did in this codebase: `AttentionAnalyzer` is a bidirectional analysis encoder applied to text as an object, and the input side reads the prompt. Its subject is `prompt-only` — see Part 1 and [Subject](#subject--whose-behaviour-the-number-describes). The normalised `/ log₂(seq_len)` form above is also historical; the registry reports raw bits.

**Expected range.** `[0, 1]`.

**Access.** Requires the attention-analysis stage (`--diagnostics`) and nothing else. Available on every backend, API models included — the encoder reads the prompt text, which does not come from the target at all.

**Cross-reader extension (▼ₓ, opt-in).** A second, independent Horizon reading is available via `run_cross_reader: true` on the `/analyze` request: `Horizon_cross_k = JSD(ê_k, â_{k,0:k})`, where `ê` is Llama's entropy landscape (normalized to a distribution over positions) and `â_{k,0:k}` is DistilBERT's own attention row at position `k`, restricted to its causal prefix. Unlike the default Horizon above — which reads the *same* model that produced the entropy trace, so it can't diverge from it in the cross-reader sense — DistilBERT is bidirectional and has no knowledge of how Llama generated the text. This makes it a genuine second opinion: low JSD means an independent reader's attention pattern lines up with where Llama found the text difficult; high JSD means they diverge.

Default reading (loads a second model); disable via `run_cross_reader: false` for a cheaper Llama-only configuration. Implementation: `hif/server.py::_run_horizon_cross_reader_analysis`, reusing the existing `AttentionAnalyzer` (`hif/analysis/attention.py`).

**Alignment caveat.** Llama (BPE) and DistilBERT (WordPiece) tokenize differently, so there's no exact position-for-position correspondence between the two traces. The cross-reader resamples Llama's entropy landscape onto DistilBERT's token count via linear interpolation over normalized position (`resample_to_length`) — an approximation, not an exact alignment. The resulting trace is indexed by DistilBERT tokens, not Llama tokens, and differs in length from the other instruments' traces.

---

### Reading the Traces Together

The instruments expose different dimensions of the same computational event. Useful combinations:

**Entropy (●) + Wager (▲).** Where Entropy is low (committed distribution) but Wager is high (the actual token was a long shot against that committed distribution), the model was confident and the selection overrode it. These are the positions where the model's commitments were most systematically violated.

**Entropy (●) + Spread (■).** These move in different spaces. Divergence between them — high Entropy with low Spread, or low Entropy with high Spread — indicates that vocabulary uncertainty and contextual attention are operating on different regions of the text. A model attending narrowly while remaining broadly uncertain about output is in a structurally interesting position.

**Shift (◆) + Entropy (●).** Large Shift at a step where Entropy is already low indicates an abrupt pivot in a committed distribution — the model changed direction sharply while staying confident. Large Shift at high-Entropy steps is more expected (uncertain distributions reorganize more freely).

**Horizon (▼) + Spread (■).** Both come from the same analysis encoder, but they do not have the same subject, so this is not a within-model comparison: Horizon reads the prompt (subject `prompt-only`) and Spread reads the target's generated continuation (subject `target-output-text`). Read together they say how the encoder's attention structure differs between what was asked and what came back — a property of the pair of texts, not a property of the model's internals. Any reading that treats a Horizon/Spread divergence as evidence about the model's layers is unsupported.

---

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

**Definition.** The number of equally-likely tokens that would produce the same Shannon entropy — computed from the **nucleus** entropy, not the raw top-K entropy.

```
ESS(p) = 2^H_nucleus(p)
```

**Expected range.** [1, K]. ESS=1 means one token has all the mass; ESS=K means uniform across the K candidates.

**Interpretation.** ESS is the per-step unit the Breadth chart draws. "The model is effectively choosing among ~8 equally-likely candidates" is more accessible than "the entropy is 3.0 bits." A legacy `effective_support_size_upper = 2^entropy_bits_upper` is also recorded when the vocabulary size is known.

Note that ESS is *not* itself a reported measurement — `output_entropy_bits` is. ESS and entropy are the same signal in different units, and the measurement set carries each quantity exactly once.

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

**Expected range.** [0, ∞). Undefined (infinite) when the perturbed distribution places mass outside the baseline's support; the implementation clamps that case to a conventional `1e9` sentinel so the value round-trips through JSON, and the aggregate `mean_kl_divergence` averages only over finite steps. Use JSD as the primary quantity; KL provides directionality information.

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

**Expected range.** [0, 1]; `1.0` means identical nuclei, and an empty baseline nucleus is defined as stable (`1.0`). Aggregated per variant as `mean_nucleus_stability_p90`.

**Why both.** JSD measures mass shift; nucleus overlap measures whether the *viable token set* changed. A small mass shuffle near the threshold can flip nucleus membership without moving JSD much, and the reverse also happens.

---

## Perturbation Response

Computed by `compute_stability_metrics()` in `hif/metrics/stability.py`. One `PerturbationResponse` object per run, aggregating across perturbation variants. (`StabilityMetrics` remains as a backwards-compatible alias for the class name.)

Four fields, each computed independently from whatever evidence exists and set to `None` when that evidence does not exist — never a fake `0.0` and never a fake `1.0`:

| Field | Unit | Definition |
|-------|------|-----------|
| `input_entropy_shift_bits` | bits | `mean(|perturbed.mean_entropy − baseline.mean_entropy|)` |
| `input_entropy_std_bits` | bits | `std(|perturbed.mean_entropy − baseline.mean_entropy|, ddof=1)`; needs ≥ 2 variants |
| `perturbation_jsd_bits` | bits | `mean(mean_js_divergence)` over variants |
| `input_output_correlation` | dimensionless | Pearson `r` between the shift and JSD per-variant series |

`n_perturbations` records how many variants contributed. (Two documented-dead optional fields, `temperature_robustness` and `prompt_order_robustness`, were removed in profile schema 0.10.0 — nothing in the pipeline ever populated them.)

These are the quantities that surface as `input_entropy_shift_bits`, `input_entropy_std_bits`, `perturbation_jsd_bits`, and `io_correlation_r` in Part 1 — see there for ranges and absent-vs-zero semantics.

**History (kept deliberately visible).** This module used to report `input_stability = 1 − mean|Δ volatility|` and `output_stability = 1 − mean JSD`. Both were wrong in the same three ways: they saturated (pinning at exactly 1.0 and destroying resolution in the regime that mattered), the input one divided by `log₂(vocab_size)` so tokenizer metadata leaked into a number presented as behaviour, and `1 − x` hid the measurement behind a score. They are replaced by the measured quantities themselves.

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

Zero vectors return `1.0`; an empty generation returns `0.0`.

**Expected range.** [0, 2] by definition. Low: the output stays in the semantic neighbourhood of the prompt. High: the output sits far from the prompt's representational space.

**Named for what it measures.** This was called "Semantic Drift". It is a distance between two embeddings from a single run — it is not evidence that a model drifted, and the rename is deliberate.

**No equilibrium flag.** `CenterDiagnostics` used to carry a `"rigid" | "balanced" | "unstable"` classification of output entropy against `0.1` and `0.9 × log₂(vocab_size)`. That bucketed behaviour by a property of the tokenizer: the thresholds derive from vocabulary size, output entropy was computed over a top-K distribution bounded by `log₂(K)`, "unstable" was unreachable under normal generation, and every reference run returned `"balanced"`. It is gone, along with the rest of the level machinery.

---

## Part 4 — Perturbation-Field Descriptors

Where Part 1 compresses a run to its run-level scalars and Part 3 exposes the per-step components, Part 4 characterizes the model's behavior as a **region** rather than a point. All are derived scalars, computed from distributions/embeddings held only transiently (compute-and-discard): a top-k distribution *with token identity* is reconstructable content, so by default it never reaches an artifact. The one sanctioned exception is `config.traceability.enabled`, which persists the raw member traces on the profile so these descriptors can be recomputed without re-running models.

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

**Why the levels are gone.** `generate_findings()` used to bucket measurements into low/medium/high against a threshold table and emit a one-sentence verdict. Assigning a level is an inference that requires a null distribution this project never established, and the decision rule built on those levels measured a **~43% false-positive rate on pairs of runs known to be identical**. What a run measured lives in `hif.profile.signals.measurements()`, in natural units; what it means is the reader's call.

The two surrogate names are reported rather than hidden because a measurement computed through a proxy describes the proxy reading the target's text — not the target's own computation. The CLI stars such measurements in its table, and `signals_record()` emits both names under `surrogate`.
