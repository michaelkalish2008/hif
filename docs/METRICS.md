# Metrics — Horizonal Interpretability Framework (HIF)

## Derivation Scheme

Every HIF instrument is a **triple**: **observable × functional × resolution**.

- **Observable** — what the forward pass exposes: output distribution; input distribution (teacher forcing); attention row; logits. Gated by what the backend exposes (see Backend Access below).
- **Functional** — **information-theoretic** (entropy, surprisal, JSD, trace correlation) or **geometric** (embedding distance, cluster structure, silhouette). Two co-equal families. Entropy reads the *shape* of the distribution (concentration across the simplex), blind to identity. Geometry reads *where* the mass sits — which tokens the model seriously weighs and how far apart they are. Both are required.
- **Resolution** — **aggregate** → a scalar measurement (Part 1 below); **token-level** → a reading (see [INSTRUMENTS.md](INSTRUMENTS.md)).

## Significance Gate

A computable triple is not automatically an instrument. Two conditions, both required:
1. **Derivability** — computable from the distributional observable alone, no inference to hidden structure.
2. **Distinct disclosure** — discloses a facet no admitted instrument already captures; must move independently somewhere across contexts.

## Natural units

Every measurement is reported in the unit it is measured in — bits for entropies and surprisals, dimensionless for correlations and cosine similarities, cosine distance for embedding displacement, a fraction for a count of steps. Key names carry the unit. Nothing here is normalised into `[0, 1]`, divided by a vocabulary size, or inverted into a `1 − x` score.

Three families of quantity were removed, and this document records why so they do not come back:

- **The `normalized` block.** Unbounded quantities were divided by `log₂(vocab_size)`. That normaliser then surfaced as the strongest apparent "behavioural" feature in the study corpus (r = 0.980, constant within a model) — tokenizer metadata masquerading as behaviour. Bounded scales also saturate, and bits are self-interpreting: 4.9 bits is about the uncertainty of a uniform choice among ~30 tokens, whereas "0.0178" is not checkable against anything.
- **The `levels` block (low/medium/high) and the verdict/equilibrium flags.** Assigning a level is an inference requiring a null distribution this project never established. The decision rule built on the previous levels measured a ~43% false-positive rate on pairs of runs known to be identical.
- **Duplicate names.** `continuity` was `1 − sensitivity` computed from the same JS divergences, and the `wager` aggregate was byte-for-byte the same computation as `surprise`. Reporting one measurement twice under two names inflates the apparent dimensionality of the signal set. Each quantity now appears exactly once in the measurement set.

Source of truth: `MEASUREMENTS` and `MEASUREMENT_UNITS` in `hif/profile/signals.py`, printed by `hif schema`.

## Backend Access

Access is a property of what the backend exposes, not of the model. `hif/models/capabilities.py` is the enforcing authority; `hif models` prints the current table.

| Access | Backends | What is available |
|--------|----------|------------------|
| `[F]` full | `hf`, `tlens`, `hf-vlm` | Full-vocabulary distributions, teacher forcing, attention capture — every measurement |
| `[T-k]` truncated | `openai`, `openai-vlm`, `gemini`, `ollama` | Top-k logprobs only; output entropy is a lower bound; no teacher forcing, no attention |
| `[P]` proxy | `anthropic` | Selected token only; distribution measurements degenerate unless a `--surrogate` reads the output text under teacher forcing |

The input-side measurements (`input_entropy_shift_bits`, `prompt_surprisal_excess_bits`, `io_correlation_r`) require teacher forcing. On a backend that cannot teacher-force they are either **absent from the record entirely**, or — with `--surrogate` — computed by a small local open-weight model reading the same prompt, and flagged as such via `findings.surrogate_model_name`. A surrogate reading describes the surrogate, not the target model.

Absent is never zero. A measurement the run produced no evidence for is omitted from `measurements`, because "no evidence" and "measured zero" are different statements.

---

This document provides precise definitions for every quantity computed by HIF, in three parts:

1. **The measurement set** — the twelve scalars a run reports, in natural units
2. **Low-level component metrics** — the per-step distribution, semantic, sensitivity, and perturbation-response measurements the set is derived from
3. **Perturbation-field descriptors** — the run's behaviour as a *region* rather than a point

All distribution and semantic metrics are computed once per generation step over the top-K probability distribution. The `truncated=True` flag indicates the distribution covers only top-K tokens. On a `[P]` backend with a surrogate, the distributions are the surrogate's over the target model's output text.

---

## Part 1 — The Measurement Set

Twelve scalars, defined once in `hif/profile/signals.py::MEASUREMENTS` and extracted by `measurements(profile)`. The same function feeds the CLI table and the machine record, so a number shown in a terminal and a number in a JSONL line can never diverge. Each is a triple: observable × functional × resolution (aggregate).

| Key | Unit | Requires |
|-----|------|----------|
| `input_entropy_shift_bits` | bits | teacher forcing (or `--surrogate`) + perturbation variants |
| `perturbation_jsd_bits` | bits | perturbation variants + top-k logprobs |
| `io_correlation_r` | dimensionless | ≥ 2 perturbation variants with both sides measured |
| `io_cosine_similarity` | dimensionless | ≥ 1 perturbation variant + an embedding encoder |
| `prompt_surprisal_excess_bits` | bits | teacher forcing (or `--surrogate`) |
| `candidate_cluster_entropy_bits` | bits | top-k logprobs + an embedding encoder |
| `output_entropy_bits` | bits | top-k logprobs |
| `output_entropy_step_delta_bits` | bits | top-k logprobs, ≥ 2 steps |
| `semantic_centroid_veer_cosine` | cosine distance | `semantic_field.enabled` + an embedding encoder |
| `attention_entropy_output_bits` | bits | attention capture |
| `attention_entropy_input_bits` | bits | attention capture |
| `counterfactual_exposure_fraction` | fraction of steps | top-k logprobs + an embedding encoder |

---

### `input_entropy_shift_bits`

**Zone.** Input side, under perturbation.

**Definition.** Mean absolute difference, over perturbation variants, between the variant's mean input-token entropy and the baseline's.

```
input_entropy_shift_bits = mean_v | μ[H(Pᵢ)]_variant_v − μ[H(Pᵢ)]_baseline |

where H(Pᵢ) = −∑ₜ pₜ log₂ pₜ  (teacher-forced full-vocabulary entropy at prompt position i)
```

**Unit and range.** Bits. Unbounded above. `0` means the perturbations moved input-side entropy not at all.

**Not inverted, not normalised.** This replaces the former "Input Stability" (`1 − mean|Δ stability_score|`), which saturated at exactly 1.0 in the regime that mattered and divided by `log₂(vocab_size)`.

**Absent when** there are no perturbed input-side analyses — the backend cannot teacher-force and no surrogate was supplied. Computed in `hif/metrics/stability.py::compute_stability_metrics`.

---

### `perturbation_jsd_bits`

**Zone.** Perturbation.

**Definition.** Mean Jensen-Shannon divergence between the baseline output distribution and each paraphrase variant's, averaged across generators and generation steps.

```
perturbation_jsd_bits = mean_v [ mean_j [ JSD(P_baseline,j ‖ P_variant,j) ] ]

JSD(P ‖ Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),  M = ½(P + Q),  logs base 2
```

**Unit and range.** Bits. Genuinely bounded to `[0, 1]` by definition in log base 2 — that bound is a property of JSD, not a rescaling. It is reported as measured, no longer inverted into an "Output Stability" score.

**Perturbation generators.** Default `["synonym", "tone", "reorder"]` with `n_variants = 2` each. `substitution` and `ambiguity` are implemented and selectable; LLM-backed paraphrasing is opt-in and requires an explicit endpoint.

**Fallback.** When the aggregate is absent but per-perturbation `SensitivityMetrics` exist, `measurements()` averages `mean_js_divergence` over them — same quantity, same unit.

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

**Zone.** Selection, input side.

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

**Definition.** Mean over generation steps of the Shannon entropy of the probability mass distributed across the step's semantic clusters (see *Cluster Entropy* in Part 2).

**Unit and range.** Bits. Unbounded above in principle, bounded in practice by `log₂` of the cluster count. High: genuinely competing semantic directions. Low: one direction dominates even where alternatives exist.

**Absent when** no per-step semantic metrics were computed.

---

### `output_entropy_bits`

**Zone.** Output side.

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

**Unit and range.** Bits. Unbounded above.

**Absent when** fewer than two generation steps were recorded.

---

### `semantic_centroid_veer_cosine`

**Zone.** Output side, geometric (the Veer instrument, ◈).

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

**Zone.** Attention (an independent reader's, not the generating model's).

**Definition.** Mean Shannon entropy of the causal-prefix attention row, at each output position and at each input position respectively.

```
row i is restricted to columns 0..i, renormalised to a distribution, then
Hᵢ = −∑ⱼ≤ᵢ āᵢⱼ log₂ āᵢⱼ
```

**Unit and range.** Bits. **Grows with prefix length by construction** — read against the position axis, not as a fraction. It is deliberately *not* divided by `log₂(prefix length)`: that denominator is the sequence length, so it puts position metadata into a number labelled behaviour, the same mistake as the removed vocabulary-size normaliser.

**Source.** `profile.attention_capture`, a `TextAttentionAnalysis` produced by the DistilBERT reader in `hif/analysis/attention.py`, already aggregated across heads and layers. Extracted by `hif/viz/signals/_attention.py::row_entropy_trace`.

**Absent when** attention analysis did not run (`config.attention.enabled` is False by default).

---

### `counterfactual_exposure_fraction`

**Zone.** Output side, geometric (the Exposure instrument, ◇).

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

## Part 2 — Low-Level Component Metrics

These are the per-step measurements that feed into the aggregate metrics above.

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

Three fields, each computed independently from whatever evidence exists and set to `None` when that evidence does not exist — never a fake `0.0` and never a fake `1.0`:

| Field | Unit | Definition |
|-------|------|-----------|
| `input_entropy_shift_bits` | bits | `mean(|perturbed.mean_entropy − baseline.mean_entropy|)` |
| `perturbation_jsd_bits` | bits | `mean(mean_js_divergence)` over variants |
| `input_output_correlation` | dimensionless | Pearson `r` between the two per-variant series above |

`n_perturbations` records how many variants contributed. `temperature_robustness` and `prompt_order_robustness` exist on the model as optional fields and default to `None`; nothing in the pipeline populates them.

These are the three quantities that surface as `input_entropy_shift_bits`, `perturbation_jsd_bits`, and `io_correlation_r` in Part 1 — see there for ranges and absent-vs-zero semantics.

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

## Part 3 — Perturbation-Field Descriptors

Where Part 1 compresses a run to twelve scalars and Part 2 exposes the per-step components, Part 3 characterizes the model's behavior as a **region** rather than a point. All are derived scalars, computed from distributions/embeddings held only transiently (compute-and-discard): a top-k distribution *with token identity* is reconstructable content, so by default it never reaches an artifact. The one sanctioned exception is `config.traceability.enabled`, which persists the raw member traces on the profile so these descriptors can be recomputed without re-running models.

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

**Why the levels are gone.** `generate_findings()` used to bucket six measurements into low/medium/high against a threshold table and emit a one-sentence verdict. Assigning a level is an inference that requires a null distribution this project never established, and the decision rule built on those levels measured a **~43% false-positive rate on pairs of runs known to be identical**. What a run measured lives in `hif.profile.signals.measurements()`, in natural units; what it means is the reader's call.

The two surrogate names are reported rather than hidden because a measurement computed through a proxy describes the proxy reading the target's text — not the target's own computation. The CLI stars such measurements in its table, and `signals_record()` emits both names under `surrogate`.
