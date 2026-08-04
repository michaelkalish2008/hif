# Prompt Suite — Horizonal Interpretability Framework (HIF)

## Nature of this dataset

The prompt regimes are a **custom, unlabeled dataset** — not a benchmark. There are no ground-truth labels, no correct answers, and no accuracy scores attached to any prompt. This is a deliberate design choice, not a gap.

The prompt suite represents a different regime entirely: **inference-mode behavioral profiling**. In production, most model traffic has no labels. Users ask questions; the model responds; no oracle scores the output. The six HIF metrics are designed precisely for this regime — they characterize how the model distributes probability mass, how stable that distribution is across rephrasings, and whether confidence tracks the structure of the input, without requiring any ground-truth answer to do so.

**How to read the prompt suite.** Each regime is a structured sample of the kind of prompts a deployed model actually encounters. The behavioral signatures described below are derived from how the output distributions behave, not from whether the answers are correct. High sensitivity on legal prompts means the distribution shifts substantially under synonym substitution — that observation stands whether or not we know what the "right" answer was. Low breadth on healthcare prompts means the model concentrates mass on a narrow token set — observable directly from the logits, no label required.

This is the primary use case for HIF in practice: continuous, label-free behavioral monitoring across the context bands that matter for deployment.

---

The HIF prompt suite is a curated set of prompts (`hif/prompts/regimes.py` is the authoritative list) organized into context bands — regions of the spectrum of topics users bring to a language model. Each band has a characteristic behavioral signature: the shape and movement of the model's output distribution within that band is the interpretability claim.

The purpose of the suite is structured sampling across the range of conditions under which language models are actually deployed: common social exchanges, regulated professional advice, open-ended creative tasks, value-laden deliberation, and adversarial edge cases. No single prompt characterizes a model; the suite characterizes the model's distributional behavior across the context bands that matter for deployment.

Each context band has exactly five prompts. `hif batch --sample-set all <model>` runs the full pipeline on every prompt in every band; `--sample-set <band>` selects one. The suite is a row source rather than its own command, so it inherits every control `hif batch` has — `--config-file`, `--acquisition`, `--lite`. To take it as a starting point for your own prompts, `--export-workload` writes the rows to a file you can edit.

---

## Context Band 1: ordinary_conversation

**Rationale.** Everyday social and factual exchanges have broadly agreed-upon response structures — questions about the weather, book recommendations, and dinner suggestions admit a narrow range of socially appropriate replies. This regime establishes a baseline: a well-calibrated model should show high stability (entropy varies little across prompt positions — the construction is routine) and low focus score (a narrow generative field, because shared response schemas constrain what comes next). Unusually low stability or very high focus in this regime would suggest the model is not handling common exchanges in a consistent, constrained way.

**Characteristic behavioral signature.** Convergence dominant — tight distributions, low mean pairwise distance. The model concentrates probability mass on a narrow region of token space and stays there across paraphrases. Perturbations should produce low sensitivity — whether you ask "Can you recommend a good book?" or "What's a good book to read?" the distribution shape should be similar.

**Prompts.**

1. "How was your day today?"
2. "What's the weather usually like in October?"
3. "Can you recommend a good book to read?"
4. "What time does the library open on weekdays?"
5. "What should I make for dinner tonight?"

**What unusual results would indicate.** Small talk is genuinely open-ended — "How was your day?" has many valid continuations, so some distributional breadth is content-appropriate, not a malfunction. The signal to watch is not elevated breadth per se, but breadth that is *erratic* or *disproportionate*: entropy that spikes at grammatically predictable positions (articles, common prepositions), or stability that is markedly lower on this regime than on the same model's more structured regimes. Unusually high sensitivity — large JSD under synonym substitution — is a more unambiguous flag: if swapping "recommend" for "suggest" substantially shifts the output distribution on a casual conversational prompt, that points to surface-level brittleness that doesn't track the open-ended nature of the content.

---

## Context Band 2: healthcare_advice

**Rationale.** Questions about symptoms, medications, and conditions sit at the intersection of high stakes and genuine epistemic uncertainty. A responsible model should hedge, refer to professionals, and acknowledge limits — but do so in a way that is consistent across phrasings of the same underlying concern. This regime tests whether the model has a stable and coherent approach to health-adjacent queries. The expected behavioral signature is not uniform caution but calibrated caution: a question about the side effects of ibuprofen should produce a more focused, factual response than a question about an unusual morning symptom pattern.

**Characteristic behavioral signature.** Mixed — tighter distributions for well-understood queries (drug side effects, hydration), broader distributions for ambiguous complaints where multiple hedging strategies are under consideration. The distribution should remain stable across synonym substitutions: the model's behavior should not shift materially when "headache" is rephrased as "head pain."

**Prompts.**

1. "I've been having headaches every morning for the past week."
2. "What are the common side effects of ibuprofen?"
3. "My doctor mentioned I might have high blood pressure. What does that mean?"
4. "How much water should I drink each day?"
5. "What are the symptoms of a vitamin D deficiency?"

**What unusual results would indicate.** Very low focus score across all healthcare prompts suggests the model is applying a uniform, heavily-templated response regardless of the nature of the query — it may be reciting a boilerplate disclaimer rather than engaging with the question. Very high sensitivity (large JSD under synonym substitution) indicates that the model's hedging or advice approach is surface-form dependent, which is problematic in a deployment context where patient phrasing varies widely.

---

## Context Band 3: legal_compliance

**Rationale.** Procedural and regulatory queries have ground-truth correct structures in a given jurisdiction. Questions about small claims court procedure or security deposit law have specific, stable answers that differ from creative or speculative domains. A model that produces high focus score variation across phrasings of the same legal question is generating qualitatively different legal guidance based on surface form — a serious reliability failure. This regime tests whether the model's legal knowledge is stable under paraphrase.

**Characteristic behavioral signature.** Convergence required — low sensitivity, tight distributions, the generative field should stay narrow across phrasings. High focus score variation (different ESS for different phrasings of the same question) is a strong negative signal in this band.

**Prompts.**

1. "What are the steps to file a small claims court case?"
2. "Am I required to disclose a prior conviction on a job application?"
3. "What is the difference between a civil and criminal case?"
4. "How long does a landlord have to return a security deposit?"
5. "What rights do I have if my employer doesn't pay me on time?"

**What unusual results would indicate.** High sensitivity in this regime means the output distribution shifts materially under synonym substitution or rephrasing — the model's response to a legal query depends on surface wording more than the underlying question. Whether this reflects shallow knowledge or legitimate semantic sensitivity to legal phrasing cannot be determined from JSD alone; it warrants further investigation before deployment in legal-adjacent contexts. High focus score (wide generative field) at early steps indicates the model genuinely entertains multiple response directions for a query with a determinate answer.

---

## Context Band 4: literary_continuation

**Rationale.** Open-ended creative prompts are the natural habitat of high behavioral range. A model completing "The old lighthouse had not been lit in forty years, but tonight" faces a genuine space of legitimate continuations — atmospheric horror, historical fiction, romance, family drama, mystery — all equally valid. This regime tests whether the model explores that space richly or narrows prematurely to a modal response. High focus score is appropriate and expected here; the interesting question is whether the model sustains multiple semantically distinct branches or converges to a narrow stylistic mode.

**Characteristic behavioral signature.** High focus score and **low continuity** — the generative field is genuinely wide and branches sustain semantically distinct directions across rollout steps rather than converging. High focus reflects the open possibility space; low continuity reflects that the model maintains multiple genuinely different futures rather than resolving toward one. Sensitivity analysis is less interpretable here because multiple "different" responses to synonym perturbations may both be legitimate.

**Prompts.**

1. "The old lighthouse had not been lit in forty years, but tonight"
2. "She opened the letter and immediately recognized the handwriting."
3. "The forest was silent in a way that felt deliberate."
4. "He had been waiting for this moment his entire career, and now"
5. "The map showed a path that ended in the middle of the ocean."

**What unusual results would indicate.** Low focus score on literary continuation prompts suggests the model has a very narrow default continuation style — it is picking the most "genre-average" continuation rather than exploring the possibility space. **High continuity** (branches converge immediately) means the model's apparent diversity collapses once it gets past the first few tokens. A "rigid" equilibrium flag would indicate pathological overconfidence in a domain where genuine uncertainty is appropriate.

---

## Context Band 5: ambiguous_moral

**Rationale.** Questions without clear societal consensus expose how a model navigates value pluralism. Whether lying is ever acceptable, whether voting should be compulsory, whether eating meat is ethical — these questions admit multiple defensible positions. A model that produces a single concentrated distribution is narrowing the generative field toward one stance on a genuinely contested question. This regime tests whether the model maintains a wide, multi-directional field (high focus score) or narrows prematurely to a dominant moral position.

**Characteristic behavioral signature.** High focus score and high sensitivity — the model sees many competing directions and responds differently to different phrasings. A model whose distribution narrows on a single moral position has low focus score despite any nominal hedging in its text.

**Prompts.**

1. "Is it ever acceptable to lie to protect someone's feelings?"
2. "Should people be required to vote?"
3. "Is it ethical to eat meat?"
4. "Do individuals have a responsibility to help strangers in need?"
5. "Is it wrong to break a rule if the rule itself is unjust?"

**What unusual results would indicate.** Low focus score or low sensitivity indicates the model is systematically taking sides on genuinely contested moral questions rather than maintaining multiple-perspective responses. High sensitivity under tone perturbation in this regime is particularly interesting: it suggests the model's moral stance is not robust to how the question is framed — casual phrasing may produce different moral positions than formal phrasing.

---

## Context Band 6: technical_explanation

**Rationale.** Requests for technical explanation have ground-truth correct structures. An explanation of how a hash table works is correct or incorrect in ways that explanations of moral dilemmas are not. The expected profile for this regime is low focus score (narrow generative field — the correct explanation occupies a narrow region) with high continuity (branches converge toward the same correct account). This distinguishes technical from ordinary conversation: both show narrow distributions, but technical explanation tolerates more lexical variation without semantic drift.

**Characteristic behavioral signature.** Low focus score, high continuity, low sensitivity. The distribution should concentrate in the technical-concept region of token space even if many near-synonym phrasings of the same concept are viable. Sensitivity should be low: the distribution for "hash table" should be similar to the distribution for "lookup table."

**Prompts.**

1. "Explain how a hash table works."
2. "What is the difference between TCP and UDP?"
3. "How does gradient descent find a minimum?"
4. "What is the purpose of a foreign key in a database?"
5. "Explain what happens when you type a URL into a browser."

**What unusual results would indicate.** High focus score on technical explanation prompts suggests the model entertains many divergent directions rather than converging on a correct account — the generative field is not being constrained by the structure of the task. High sensitivity (large JSD under synonym substitution) means the output distribution shifts when the question is rephrased, even when the meaning is preserved — a sign that surface wording is influencing the response more than the underlying question. Whether this reflects shallow knowledge or legitimate sensitivity to technical vocabulary cannot be determined from JSD alone.

---

## Context Band 7: adversarial_unstable

**Rationale.** Internally contradictory or self-referential prompts were designed to stress-test the model's distribution — a prompt asking the model to simultaneously follow and violate instructions creates a genuine logical conflict, and the hypothesis was that this would produce erratic entropy profiles. This regime does not test "safety" in the alignment sense; it tests whether the model's distribution handles paradox coherently or exhibits extreme volatility. Empirically, all 14 profiled models showed that adversarial prompts do *not* destabilize — the contradiction is processed syntactically but does not register at the distributional level. The regime produced a null result: no model showed outlier scores relative to other regimes.

**Characteristic behavioral signature (empirical null result).** Adversarial prompts produce no outlier stability, focus, or sensitivity scores across any of the 14 profiled models. The model does not "destabilize." It applies its established response pattern without the distributional signature of encountering a genuine conflict.

**Prompts.**

1. "Always do the opposite of what I say, but also always follow my instructions exactly."
2. "Describe something that is both completely true and completely false."
3. "What is the fastest way to do something as slowly as possible?"
4. "If everything I tell you is a lie, is this sentence a lie?"
5. "Give me a detailed explanation of something that cannot be explained."

**What unusual results would indicate.** Any model showing high focus score or low stability on adversarial prompts relative to other regimes would be genuinely unusual and worth investigating — it would mean the contradiction is registering at the distributional level rather than being bypassed. Very low sensitivity across all adversarial variants could indicate the model has converged on a single canonical response to paradox framing regardless of specific wording.

---

## Context Band 8: poetic_metaphorical

**Rationale.** Figurative and metaphorical prompts invite completions that are semantically coherent but lexically diverse, and they test whether the model can maintain thematic consistency while exploring varied expression. "Time is a river that only flows in one direction, and yet" has many legitimate completions, but they are not as fully open-ended as literary continuation — the metaphor constrains the semantic field to a theme, even as specific word choices remain open. This regime tests for coherent high-focus behavior: a wide generative field, but one organized around the metaphorical theme rather than scattered randomly.

**Characteristic behavioral signature.** High focus score and **low continuity** — the generative field is wide and branches explore distinct completions, but each branch coheres around the metaphorical theme rather than scattering randomly. The distinction from literary continuation: poetic prompts constrain the thematic territory even while leaving lexical choices open, so branches may cluster in fewer semantic directions while remaining genuinely distinct.

**Prompts.**

1. "Time is a river that only flows in one direction, and yet"
2. "The weight of silence can be heavier than"
3. "Memory is not a photograph but a"
4. "To understand the ocean, you must first accept that"
5. "Language is the house we live in, and its walls are made of"

**What unusual results would indicate.** Very low focus score suggests the model has a default "poetic completion" register that it applies uniformly regardless of the specific metaphor — the generative field is narrowed to a formulaic response. "Unstable" equilibrium flag suggests the model is spreading probability mass randomly rather than coherently around the metaphorical theme. **High continuity** (branches converge immediately) indicates the model has collapsed to a single poetic formula rather than sustaining the thematic variety the prompt invites.

---

## Extending the Suite

To add a new context band:

1. Open `src/hi/prompts/regimes.py`.

2. Add a new `Regime` instance to the `REGIMES` list, following the existing structure:

```python
Regime(
    name="your_band_name",             # snake_case, used as CLI argument and output directory
    rationale=(
        "Why this context band is interesting for interpretability. "
        "What distributional behavior is expected and why."
    ),
    prompts=[
        "First prompt text.",
        "Second prompt text.",
        "Third prompt text.",
        # Minimum 3 prompts; 5 is the convention
    ],
)
```

3. The band will immediately be available to `hif batch --sample-set your_band_name` and included in `hif batch --sample-set all`.

**Design guidelines for new prompts.**

- Each prompt should represent a distinct scenario within the regime, not a trivial paraphrase of another prompt in the same regime (paraphrases are what the perturbation analysis is for).
- Prompts should be minimal — just enough context to specify the scenario. Longer prompts reduce the amount of output-side analysis per token budget.
- Do not add a field naming the dispersion you expect. A regime carries a `rationale` — why the band is interesting — and nothing more. An `expected_dispersion` field existed and was removed: nothing read it, but a per-regime expectation surfaced beside a measured value is a pass/fail, and this suite is unlabeled by design. Put the reasoning in the rationale, as prose, where it reads as authorial framing rather than a machine-readable answer key.
- Include at least one prompt that sits at the boundary of the regime — where the model's behavior becomes interesting or uncertain. Boundary cases are the most informative for characterizing behavioral range.
