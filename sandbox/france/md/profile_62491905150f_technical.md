# BRI Technical Report — gpt2

**Schema version:** 0.11.0
**Created:** 2026-08-07T00:55:42.211919+00:00

## Model Identity

| Field | Value |
|---|---|
| Name | gpt2 |
| Backend | hf |
| Vocab size | 50257 |
| Context length | 1024 |
| Parameter count | unknown |

## Prompt

| Field | Value |
|---|---|
| Regime | ordinary_conversation |
| Token count | 7 |
| SHA-256 | `115049a298532be2...` |

```
What is the capital of France?
```

## Measurements — gpt2

All values are in natural units. There are no levels, thresholds, or
verdicts: assigning one needs a null distribution this instrument does
not have.

| Measurement | Value | Subject | Unit / definition |
|---|---|---|---|
| Input entropy shift (bits) | absent (not measurable on this run) | target-distribution | bits — mean \|mean input-token entropy(variant) − mean input-token entropy(baseline)\| over perturbation variants. Unbounded above. |
| Input entropy shift spread (bits) | absent (not measurable on this run) | target-distribution | bits — standard deviation (ddof=1) of the per-variant input entropy shifts. The spread of the model's entropy response across perturbations. Unbounded above; absent when fewer than two variants exist. |
| Perturbation JSD (bits) | absent (not measurable on this run) | target-distribution | bits — mean Jensen-Shannon divergence between the baseline output distribution and each perturbed variant's. Bounded to [0, 1] by definition in log base 2. Absent on a backend that returns only the selected token: two point masses have no distributional overlap to diverge over, so the computation would report a token-disagreement rate under a key that promises a divergence between distributions. |
| Input/output cosine similarity | absent (not measurable on this run) | target-output-text | dimensionless — cosine similarity between the input embedding and the output embedding. Bounded to [-1, 1] by definition. |
| Prompt surprisal excess (bits) | 0.4354 | target-distribution | bits — mean max(0, surprisal(token) − H(distribution)) over teacher-forced prompt positions. Unbounded above. |
| Output entropy (bits) | 2.75101 | target-distribution | bits — mean Shannon entropy of the per-step top-K output distribution. A lower bound on full-vocabulary entropy when the distribution is truncated. |
| Output nucleus entropy (bits) | absent (not measurable on this run) | target-distribution | bits — mean Shannon entropy of the smallest per-step prefix carrying --entropy-percentile of the output distribution's mass, renormalized to a proper distribution. Absent unless --entropy-percentile is passed, and absent on any run whose captured top-K does not reach that mass at every step — the entropy of a slice that does not contain the nucleus is a different quantity, not a smaller number. |

Similarity trend slope: +0 (OLS slope of per-step input/output cosine similarity).

## Center Diagnostics

| Metric | Value |
|---|---|
| Input mean entropy | 8.1536 |
| Output mean entropy | 3.4855 |
| Entropy ratio (output/input, both bits) | 0.4275 |
| Prompt/output cosine distance | 0.4200 |

## Input-Side Analysis

| Metric | Value |
|---|---|
| Mean surprisal (bits) | 5.1244 |
| Mean entropy (bits) | 8.1536 |
| Max entropy log2\|V\| (bits) | 15.6170 |

## Output-Side Analysis

| Metric | Value |
|---|---|
| Mean step entropy | 3.4708 |
| Generated tokens | 64 |
| Top-K | 50 |

## Trajectory Analysis

| Metric | Value |
|---|---|
| Start step | 71 |
| Branches | 0 |
| Rollout steps | 10 |
| Initial clusters | 0 |
| Persistence score | 0.0000 |
| Explosion score | 0.0000 |
| Convergence score | 0.0000 |

### Branches

| Cluster | Representative Token | Generated Text |
|---|---|---|

## Distribution Metrics (per output step)

| Step | Entropy (bits) | Logit margin | Top-K mass | Eff. support | Tail weight |
|---|---|---|---|---|---|
| 0 | 2.965 | 1.863 | 0.556 | 12.3 | 0.161 |
| 1 | 3.664 | 0.790 | 0.737 | 14.6 | 0.108 |
| 2 | 2.743 | 0.934 | 0.782 | 7.7 | 0.113 |
| 3 | 3.682 | 0.709 | 0.460 | 25.1 | 0.161 |
| 4 | 3.108 | 0.558 | 0.671 | 12.2 | 0.114 |
| 5 | 2.973 | 0.821 | 0.838 | 8.4 | 0.121 |
| 6 | 2.358 | 1.485 | 0.912 | 4.6 | 0.074 |
| 7 | 2.409 | 2.892 | 0.671 | 6.5 | 0.112 |
| 8 | 0.026 | 8.261 | 0.998 | 1.0 | 0.002 |
| 9 | 3.009 | 0.996 | 0.433 | 17.4 | 0.184 |
| ... | (54 more steps) | | | | |

## Semantic Metrics (per output step)

| Step | Clusters | Entropy | Mean pair dist | Max inter-cluster dist |
|---|---|---|---|---|

## Perturbation Response

| Metric | Value |
|---|---|
| Input entropy shift (bits) | n/a (not measurable) |
| Perturbation JSD (bits) | n/a (not measurable) |
| Input-output correlation (r) | n/a (not measurable) |
| N perturbations | 0 |

## Run Configuration

```json
{
  "model": {
    "name": "gpt2",
    "backend": "hf",
    "device": "auto",
    "dtype": "float32",
    "revision": null,
    "ollama_host": "http://localhost:11434",
    "ollama_timeout": 120.0,
    "api_key": null,
    "base_url": null,
    "temperature": null,
    "extra_body": null
  },
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "fallback_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "matryoshka_dim": null,
    "cache_dir": "/Users/michaelkalish/.cache/hif/embeddings"
  },
  "cluster": {
    "method": "hdbscan",
    "min_cluster_size": 2,
    "min_samples": 1
  },
  "generation": {
    "max_new_tokens": 64,
    "top_k": 50,
    "temperature": 1.0,
    "seed": 42,
    "entropy_percentile": null
  },
  "trajectory": {
    "n_branches": 0,
    "rollout_steps": 10
  },
  "perturbation": {
    "n_variants": 0,
    "generators": [],
    "use_llm_perturbation": false,
    "llm_base_url": null,
    "llm_api_key": null,
    "llm_model": null,
    "variants_file": null,
    "elicit_variant_outputs": true
  },
  "output": {
    "output_dir": "sandbox"
  },
  "attention": {
    "enabled": false,
    "model_name": "distilbert-base-uncased",
    "aggregate_method": "mean_all_layers",
    "max_seq_length": 512,
    "trajectory_interval": 4
  },
  "exposure": {
    "enabled": false,
    "min_prob": 0.01,
    "distance_threshold": 0.3
  },
  "semantic": {
    "enabled": false
  },
  "semantic_field": {
    "enabled": false,
    "context_window": 5
  },
  "traceability": {
    "enabled": false
  }
}
```
