# Instrument Readings — Horizonal Interpretability Framework (HIF)

A run's scalar measurements (see [METRICS.md](METRICS.md)) compress a full generation into one number per quantity. The instrument readings restore what compression hides: which specific tokens drove a value, where entropy and attention converge or diverge step-by-step, and whether a flat mean reflects genuine uniformity or cancellation between extremes.

Seven readings are defined. Six of them have a chart in the viz registry (`hif/viz/registry.py`, `kind="reading"`); Veer has no chart, but its per-step trace is persisted on `profile.semantic_field` and its mean is reported as `semantic_centroid_veer_cosine`.

| Symbol | Name | Space measured | Access |
|--------|------|----------------|--------|
| ▲ | Wager | Surprisal excess over entropy, per prompt position | Teacher forcing (open-weight, or `--surrogate`) |
| ● | Entropy | Output distribution entropy, per generation step | All models with logprobs |
| ■ | Spread | Attention-row entropy over context positions, per generated token | Attention capture (`--diagnostics`) |
| ◆ | Shift | Step-to-step JSD between consecutive output distributions | All models with top-k data, ≥ 2 steps |
| ◈ | Veer | Step-to-step displacement of the candidate cloud's semantic centroid | Top-k probs + an embedding encoder (`--diagnostics`) |
| — | Horizon | Attention-row entropy per prompt position | Attention capture (`--diagnostics`) |
| ◇ | Exposure | Fraction of steps where an accessible alternative diverged in meaning | Top-k probs + an embedding encoder |

Horizon carries no glyph in the registry and is labelled *Input attention entropy* there; the ▼ symbol it once used is not in the code and is not used below.

Reading these traces simultaneously — across vocabulary space, embedding (semantic) space, and context-position space — is the multi-register practice the framework is designed to support. Shift (◆) and Veer (◈) are twins at the same resolution: Shift reads the step-to-step change in the distribution's *spread* (vocabulary space); Veer reads the step-to-step change in its *semantic location* (embedding space).

---

## ▲ Wager

**Formula.** `Wagerᵢ = max(0, sᵢ − H(Pᵢ))`

where `sᵢ = −log₂ p(tokᵢ | ctx<ᵢ)` is the surprisal of prompt token `i` and `H(Pᵢ)` is the Shannon entropy of the full-vocabulary distribution the model predicted at that position, both from the teacher-forced pass over the prompt.

**What it shows.** Surprisal excess per prompt position. Each bar shows how many bits the chosen token's surprisal exceeded the model's distributional entropy at that position — the residual cost of the actual token beyond general uncertainty. A tall bar at position `i` means the model had committed to a narrow distribution but the actual token was a long shot against that commitment.

The surprisal `sᵢ` is unweighted by probability — this is deliberate. Entropy is the probability-weighted average of surprisal: `∑ p(x)·(−log p(x))`. That weighting compresses the logarithmic spread: the product `p·(−log p)` is arch-shaped, peaking near `p = 1/e ≈ 0.37` and falling to zero at both extremes. High-probability tokens have small surprisals but large weights; low-probability tokens have large surprisals but small weights; the two effects pull toward each other. This compression makes entropy a stable average — but it dulls exactly the signal Wager is after. The unweighted surprisal lets the logarithmic spread do what it naturally does: make rare events stand out. A token selected with 1% probability produces a surprisal of ~6.6 bits; weighted by 0.01, it contributes only 0.066 — nearly silent, precisely when it should be loudest.

The interesting case is when `sᵢ` and `H(Pᵢ)` diverge: low `H(Pᵢ)` means the distribution is narrow and confident, but high `sᵢ` means the actual token is not what the model was confident about. `sᵢ − H(Pᵢ) > 0` is the excess — the model committed, and the token overrode that commitment. When the excess is zero, the model either selected its most likely token or was already broadly uncertain.

**Relation to `prompt_surprisal_excess_bits`.** The measurement runs `max(0, sᵢ − H(Pᵢ))` over all positions and averages to a single number; the Wager trace is that same quantity at full per-position resolution. The Surprise chart in the registry draws the same underlying series with the mean called out. They are one quantity at two resolutions, which is why the measurement set carries it only once — the historical `wager` aggregate was byte-for-byte identical to the `surprise` aggregate, and reporting one number twice under two names inflated the apparent dimensionality of the signal set.

**Expected range.** `[0, ∞)` bits. Most positions contribute zero. Large values at specific positions identify structurally surprising tokens — places where the model had committed and the actual token overrode that commitment.

**Access.** Requires teacher forcing: an open-weight backend (`hf`, `tlens`, `hf-vlm`), or a `--surrogate` proxy teacher-forced over the same prompt. A surrogate reading describes the surrogate, and is flagged as such via `findings.surrogate_model_name`.

---

## ● Entropy

**Formula.** `Entropyⱼ = H(Qⱼ) = −∑ᵥ Qⱼ(v) log₂ Qⱼ(v)`

Sum over the top-K candidates the backend returned at each generation step `j = 1…G`.

**What it shows.** Output distribution entropy per generation step, in bits. Peaks mark genuine decision moments where many tokens were competitive; troughs mark committed choices where the model narrowed sharply.

The chart draws two series: the **nucleus entropy** (95% mass, renormalised — comparable across backends regardless of how many logprobs each exposes) and the **raw top-K entropy** (the truncation lower bound). The trace is the full shape that a single mean compresses: it shows whether that mean conceals a flat plateau, a single tall spike, or alternating peaks and troughs — patterns that carry distinct interpretive weight.

**Relation to Breadth / ESS.** Entropy (●) and Effective Support Size are the same underlying signal in different units: `ESS = 2^H_nucleus`, an equivalent token count. The Breadth chart draws the ESS trace with its mean; the measurement set reports `output_entropy_bits` and not ESS, because a quantity appears once.

**Truncation.** Whenever the distribution is top-k truncated, the reported entropy is a **lower bound** on true full-vocabulary entropy, and values are not comparable across backends with different k. `DistributionMetrics.entropy_bits_upper` carries the uniform-tail upper bound where the vocabulary size is known.

**Expected range.** `[0, log₂ K]` bits — about 5.64 bits for K=50. The full-vocabulary ceiling `log₂|V|` (≈ 15.6 bits for a 50,257-token vocabulary) is not reachable from a truncated distribution.

**Access.** All models with output logprob data. On a selected-token-only backend the distribution degenerates unless a `--surrogate` recovers it.

---

## ■ Spread

**Formula.** `Spreadᵢ = H(āᵢ,₀:ᵢ)`

Row `i` of the stored continuation attention map is restricted to columns `0..i` (its causal prefix), renormalised to a probability distribution, and its Shannon entropy taken in bits (`hif/viz/signals/_attention.py::row_entropy_trace`).

**What it shows.** Attention spread over context positions. How evenly attention was distributed across prior context at each token. A value of `k` bits means approximately `2ᵏ` context positions received meaningful weight. High Spread: attention is diffuse across many positions. Low Spread: attention is concentrated on a few.

**Whose attention.** Not the generating model's. The attention map is produced by the bidirectional reader in `hif/analysis/attention.py` (DistilBERT by default) reading the generated continuation as a text, already aggregated across heads and layers by `AttentionConfig.aggregate_method` (default `mean_all_layers`; `last_layer` and `mean_upper_half` are the alternatives). There is no per-layer selection and no middle-layer isolation. This is a reading of the text's structure, not an inspection of how the text was produced.

Spread is measured in context-position space; Entropy (●) is measured in vocabulary space. These are orthogonal dimensions. A model can attend narrowly (low Spread) while remaining uncertain about which token to select next (high Entropy), or attend broadly while being highly confident. The two readings can and do move in opposite directions.

**Expected range.** `[0, log₂(i+1)]` bits — the ceiling grows with position as more prefix becomes available. The value is reported in raw bits and is deliberately **not** divided by `log₂(prefix length)`; read it against the position axis, not as a fraction.

**Access.** Requires attention capture, which runs only when `AttentionConfig.enabled` is set — `hif profile --diagnostics` does so. The corresponding measurement `attention_entropy_output_bits` is additionally gated by `hif/models/capabilities.py` to the `hf`, `tlens`, and `hf-vlm` backends.

---

## ◆ Shift

**Formula.** `Shiftⱼ = JSD(Qⱼ₋₁, Qⱼ)` — step-to-step Jensen-Shannon divergence of the output distribution, `j = 2…G`. The two steps' top-K distributions are aligned over the union of their token ids and renormalised before the divergence is taken.

**What it shows.** Step-to-step divergence within a single forward pass. Tall bars mark abrupt vocabulary pivots — the field of viable tokens reorganized sharply between steps `j−1` and `j`. Low bars mark smooth continuation, the distribution changing little as the model extends an established direction.

**Measurement caveat (real, not cosmetic).** JSD is computed only over the stored top-K candidates, not the full vocabulary. When two consecutive steps' top-K sets share little or no overlap, JSD saturates at exactly 1 bit regardless of how similar the true full-vocabulary distributions are — disjoint support alone is enough to hit the ceiling. A chart where most bars sit near 1 more often reflects narrow top-K supports failing to overlap than genuine maximal divergence. The chart therefore surfaces the top-K overlap fraction in the hover, and shows a banner when overlap is low, so this is not silently mistaken for "everything is maximally different".

**Distinction from other quantities.** Unlike the input entropy trace (prompt-side, before generation), and unlike Continuity / Trajectory (which compares independently sampled branches), Shift operates entirely within one forward pass. It is a within-run, step-local measure of distributional change. It is also distinct from `output_entropy_step_delta_bits`, which is the step-to-step change in the *amount* of uncertainty; Shift is the step-to-step change in *where the mass sits*.

**Expected range.** `[0, 1]` bits (JSD in log base 2 is bounded by definition).

**Access.** All models with top-k data and at least two generation steps. There is no attention-domain variant of Shift in this package.

---

## ◈ Veer

**Formula.** `Veerⱼ = 1 − cos( cⱼ , cⱼ₋₁ )`

where `cⱼ = Σᵥ pⱼ(v)·e(v) / Σᵥ pⱼ(v)` is the probability-weighted mean embedding — the *semantic centroid* — of the top-K candidate tokens at generation step `j`, `e(·)` is the embedding encoder, and `cos` is cosine similarity. Each candidate is embedded within a short window of left-context so the reading reflects the candidate in context, not the bare token.

**What it shows.** How far the *semantic center* of the model's candidate cloud moved between consecutive steps — the step-to-step velocity of the output's possibility field through embedding space. Low, steady Veer means coherent development around a stable topic; a tall Veer marks a semantic pivot, where the field of what the model is about to say relocates to a different region of meaning. A companion **deformation** trace, `|dispersionⱼ − dispersionⱼ₋₁|`, reads whether the field is widening or fragmenting between steps — its change in *shape*, separately from where its center moved.

**Distinction from Shift (◆).** Veer is the geometric twin of Shift. Shift measures the step-to-step change in the *spread* of the output distribution in vocabulary space (information-theoretic); Veer measures the step-to-step change in the *semantic location* of the output distribution in embedding space (geometric). They are independent: a model can hold a steady spread (low Shift) while the meaning of its candidates drifts (high Veer), or hold its meaning (low Veer) while the spread reshapes (high Shift).

**Distinction from `prompt_output_cosine_distance`.** That center diagnostic (formerly "Semantic Drift") reports a single cosine distance between the prompt embedding and the generated-text embedding — one endpoint number for the whole generation. Veer is the per-step trace of the same idea: it shows *where along the generation* the semantic center moved. Neither is evidence that a model drifted; both are distances between embeddings from a single run.

**Expected range.** `[0, 2]` (cosine distance); in practice `[0, 1]` for a sentence-embedding encoder. Most steps are small; spikes identify the steps at which the output's meaning pivoted.

**Access.** All models exposing top-K candidate probabilities, plus an embedding encoder — from the model's own top-K for open and truncated backends, and via a `--surrogate` for selected-only backends. Off by default (`SemanticFieldConfig.enabled = False`) because it re-embeds every step's candidate cloud; `hif profile --diagnostics` turns it on. Encoder-dependent: comparable only between profiles computed with the same encoder, which is recorded in `config.embedding.model_name`.

**Privacy.** Compute-and-discard. Candidate embeddings and the per-step centroids live only in the analyzer's stack frame; only the scalar traces (cosine distances) are returned and persisted.

---

## ◇ Exposure

**Formula.** `E = |{t : diffusion(t) ∧ d_t ≥ τ}| / G`

where `d_t = max_v dist(e(tok_t), e(v))` over candidates `v` in the top-K with `p_t(v) ≥ p_min` (default 0.01), `e(·)` is the embedding encoder, `τ` is the distance threshold (default 0.3), `diffusion(t)` marks steps whose candidate cloud is in the diffusion zone, and `G` is the number of generation steps analyzed.

**What it shows.** How often high-probability alternatives at a step would have pulled the response toward a different meaning — the response's sensitivity to sampling chance. A high-exposure step is one where the model could cheaply (probabilistically) have said something semantically different. This is a measure of exposure to alternative meanings, **not a factuality judgment about any output**.

**What it does not see.** Exposure is computed only over diffusion-zone steps. The convergence case — a model that is confident and narrow but aimed wrong — is excluded by construction. A confident response can still be wrong, and this reading does not see that case.

**Expected range.** `[0, 1]`.

**Access.** All models exposing top-K probabilities, plus an embedding encoder. Distances are embedding-space-dependent: values are comparable only between profiles computed with the same encoder (the encoder is recorded in the profile).

**Since.** HIF Signal Set v1.1. Implementation: `hif/analysis/exposure.py::ExposureAnalyzer`. Exposure is a measure of a response's exposure to a semantically divergent alternative under sampling chance — explicitly not a factuality judgment about any output.

---

## ▼ Horizon

**Formula.** `Horizon_i = H(ā_{i,0:i}) / log₂(seq_len)`

where `ā_{i,0:i}` is the mean-head, mean-layer attention row at prompt position `i` over its causal prefix, normalized to a probability distribution, and the entropy is normalized by `log₂(seq_len)` to bound the result to `[0, 1]`.

**What it shows.** Self-attention diffuseness per prompt position. Low Horizon: the position's attention is concentrated on a few prior tokens. High Horizon: attention is spread broadly across the prefix. This is an internal measure — it reads the generating model's own attention, the same way Spread (■) does, just averaged across all layers rather than restricted to the middle layer, and over prompt positions rather than generated ones.

**Expected range.** `[0, 1]`.

**Access.** Requires attention capture — open HuggingFace models only. Not available for API models.

**Cross-reader extension (▼ₓ, opt-in).** A second, independent Horizon reading is available via `run_cross_reader: true` on the `/analyze` request: `Horizon_cross_k = JSD(ê_k, â_{k,0:k})`, where `ê` is Llama's entropy landscape (normalized to a distribution over positions) and `â_{k,0:k}` is DistilBERT's own attention row at position `k`, restricted to its causal prefix. Unlike the default Horizon above — which reads the *same* model that produced the entropy trace, so it can't diverge from it in the cross-reader sense — DistilBERT is bidirectional and has no knowledge of how Llama generated the text. This makes it a genuine second opinion: low JSD means an independent reader's attention pattern lines up with where Llama found the text difficult; high JSD means they diverge.

Default reading (loads a second model); disable via `run_cross_reader: false` for a cheaper Llama-only configuration. Implementation: `hif/server.py::_run_horizon_cross_reader_analysis`, reusing the existing `AttentionAnalyzer` (`hif/analysis/attention.py`).

**Alignment caveat.** Llama (BPE) and DistilBERT (WordPiece) tokenize differently, so there's no exact position-for-position correspondence between the two traces. The cross-reader resamples Llama's entropy landscape onto DistilBERT's token count via linear interpolation over normalized position (`resample_to_length`) — an approximation, not an exact alignment. The resulting trace is indexed by DistilBERT tokens, not Llama tokens, and differs in length from the other four instruments.

---

## Reading the Instruments Together

The five instruments expose different dimensions of the same computational event. Useful combinations:

**Entropy (●) + Wager (▲).** Where Entropy is low (committed distribution) but Wager is high (the actual token was a long shot against that committed distribution), the model was confident and the selection overrode it. These are the positions where the model's commitments were most systematically violated.

**Entropy (●) + Spread (■).** These move in different spaces. Divergence between them — high Entropy with low Spread, or low Entropy with high Spread — indicates that vocabulary uncertainty and contextual attention are operating on different regions of the text. A model attending narrowly while remaining broadly uncertain about output is in a structurally interesting position.

**Shift (◆) + Entropy (●).** Large Shift at a step where Entropy is already low indicates an abrupt pivot in a committed distribution — the model changed direction sharply while staying confident. Large Shift at high-Entropy steps is more expected (uncertain distributions reorganize more freely).

**Horizon (▼) + Spread (■).** Both read the model's own attention, at different granularities: Horizon averages across all layers per prompt position, Spread isolates the middle layer per generated token. Divergence between the two — high Horizon (diffuse averaged-layer attention) alongside low Spread (focused middle-layer attention) — suggests the middle layer is doing more targeted work than the layer average implies.
