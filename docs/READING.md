# Reading — what a number is about, and what is not a number

One question decides how much weight a value in a record can carry: **whose
behaviour does it describe?** A record answers it in two different ways, because
it holds two different kinds of thing.

**Measurements** are about the target model, and say so. Every row in
`measurements` declares a subject — `hif schema` prints it, and
[MEASUREMENTS.md § Subject](MEASUREMENTS.md) gives the enum and the rule that
keeps a quantity out of the set when the target's data never entered it. That
document owns the subject rule; this one does not restate it.

**Readings** are about a text. They are produced by a fixed local instrument
applied to the prompt and the continuation, they never touch the target's
internals, and they are **not measurements** — they live in their own blocks in
the record and are absent from `measurements` entirely.

This document is about the second kind.

## Hermeneutic attention is a reading

`hif profile --diagnostics` enables the analysis encoder in
`hif/analysis/attention.py` — DistilBERT by default, configurable via
`AttentionConfig.model_name`. It performs four readings:

1. **Prompt alone** — the internal attention structure of the input as its own
   text: which tokens carry structural weight, and how that shifts under
   perturbation.
2. **Continuation alone** — the same for the generated text, read without
   reference to how it was produced.
3. **Resonance** — an analytical comparison of the two independent readings.
   Which continuation tokens echo the load-bearing structure of the input, and
   which have moved away from its anchors.
4. **Joint trajectory** — the encoder run on `[prompt + continuation[:k]]` at
   intervals, tracking which prompt tokens hold or release cross-attention as
   the continuation grows.

Three statements about what this is not, all of which hold for every backend:

- **It is not the target's attention.** The encoder never receives the target's
  logits, distributions, or attention weights. It reads the texts the target
  produced, as texts.
- **It is not neutral.** Every reading is anchored in the encoder's own
  pre-training. What makes it useful is not neutrality but *consistency* — the
  same bias applied uniformly, which is what makes readings comparable across
  models on identical terms.
- **It is not evidence of causation.** Jain & Wallace (2019) showed that even a
  model's own attention weights are not faithful explanations of its
  predictions. An external encoder's attention is further still from one.

**This is enforced, not just asserted.** No attention row is in the measurement
set — `ATTENTION_METRICS` in `hif/models/capabilities.py` is derived from the
registry and is currently **empty**. `attention_entropy_input_bits` was the last
one, and hif-v4 cut it on exactly this ground: a fixed encoder reading the
prompt returned a bit-identical value across all fifteen corpus models, so it was
never about any of them. The zero-variance canary asserts the class stays empty.

The attention stage gates on the **analysis stage**, not the backend — nothing in
it reads the target's internals, so there is nothing for a backend to withhold.
An open-weight model and a closed API model get the same reading, of the same
kind, with the same standing.

## What this means on closed models

The access tiers in [the README](../README.md) are an inventory of how little a
closed surface exposes: `[F]` full distributions and teacher forcing, `[T-k]`
top-k logprobs only, `[P]` output text and nothing else. The tiers shrink what
can be **measured**. They do not change what can be **read**, because a reading
was never using the internals in the first place.

That is worth stating plainly in both directions:

- A reading of a closed model's output is a real observation about a real text.
  The text is the entire observable surface at `[P]`, and reading it is not the
  preferred method there — it is the only one that exists.
- A reading of a closed model's output is **not** a recovery of what the closed
  model did internally. It does not approximate the target's attention, stand in
  for its distributions, or license a claim about its mechanism. Nothing about
  the reading gets closer to the model because the model is closed; it gets no
  further away either. It was always a reading of the text.

The structural reason is not provider policy. It holds at three levels, and only
the third is anyone's choice:

1. **The forward pass is ephemeral.** At the moment a token is generated, the
   attention weights and logit vector that produced it are not persisted.
   Teacher forcing can reconstruct input-side distributions on a later run; it
   cannot replay the generative event.
2. **Backpropagation left no trail.** Gradient descent adjusts every parameter
   in response to every training example. The weights are the residue of that
   process, and there is no path from a parameter back to a meaning.
3. **Closed models withhold the rest.** What returns is generated text and, at
   `[T-k]`, a truncated logprob vector.

The first two apply to open weights too. This is why hif reads texts as texts
rather than treating a reading as a degraded measurement waiting for better
access — better access would not convert one into the other.

## The rule, in one line

A **measurement** answers "what did this model do", carries a subject, and is
absent when the target's data did not enter it. A **reading** answers "what does
this text look like to a fixed instrument", carries the instrument's name, and is
never in `measurements`. Neither is a weaker form of the other. Confusing them is
the failure both the `subject` field and the empty `ATTENTION_METRICS` set exist
to prevent.

Interpretation built on either belongs in the work that cites this tool, not in
the tool.

## References

Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *Proceedings of
NAACL-HLT 2019*, 3543–3556. https://aclanthology.org/N19-1357

Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation.
*Proceedings of EMNLP-IJCNLP 2019*, 11–20. https://aclanthology.org/D19-1002
