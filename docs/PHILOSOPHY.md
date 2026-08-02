# Computational Phenomenology and Hermeneutic Reading

*The interpretive framing behind the framework: why read a model through its output distributions at all. It makes no empirical claims — those belong to the studies that use these measurements.*

---

## The Upstream Question

A healthcare organization deploys a language model to triage patient inquiries. A content platform uses one to moderate user submissions at scale. A financial institution integrates one into its customer advisory workflows. In each case, the deployment decision came first — and then came the impossible choices: how to filter outputs, when to require human review, how to manage liability for errors that were always structurally inevitable given what the model was doing with meaning in that domain.

This is the trolley problem structure of AI deployment. The classic ethical dilemma — pull the lever and save five at the cost of one, or do nothing and lose five — discloses something important through the impossibility of its framing: the real decision was upstream. Whether to set the trolley on its path in the first place. Once it is moving, every available option is already a form of damage control.

The same structure applies to deploying a language model in any context where outputs carry ethical weight. Once a model is running in a clinical triage application with near-zero input–output entropy correlation — meaning its outputs are systematically decoupled from what was actually asked — you are choosing between bad options: disclaimers, human review layers, liability exposure, patient harm. The distributional problem is structural. It was present before the first user query was answered. What was missing was a framework for generating insights to see model performance concerns before deployment.

---

## Part 1: Theoretical Grounding

### A Historical Convergence — and a Long Interruption

**HIF sits at the convergence of two lines of inquiry** that developed nearly simultaneously in the mid-twentieth century, were driven apart by a decades-long interruption, and are only now — through the transformer's architecture — capable of making contact.

**The connectionist origin.** Frank Rosenblatt's perceptron (1957) was an epistemological claim, not merely an engineering proposal: a learning system's knowledge is not stored in explicit representations but in the statistical configuration of its weights — distributed, not localized, and therefore not fully sayable. Rosenblatt stated directly:

> "The number of functional units in the storage system should be much less than the number of forms or memories to be retrieved." (Rosenblatt, 1957)

This is the superposition claim, stated in 1957.

The Anthropic paper that formalized superposition mathematically (Elhage et al., 2022) opens:

> "We found that simple neural networks trained on toy tasks often exhibit a phenomenon called superposition, where they represent more features than they have neurons." (Elhage et al., 2022)

Rosenblatt had said the same thing — from the other direction, without the linear algebra — sixty-five years earlier. The observation did not disappear; it was suppressed, along with the entire connectionist research program, by the symbolic AI consensus Minsky helped establish in the decade that followed.

**The symbolic interruption.** Minsky and Papert's [*Perceptrons*](https://en.wikipedia.org/wiki/Perceptrons_(book)) (Minsky & Papert, 1969) effectively ended neural network research for over a decade — but the critique was strategically narrow. The book's central proof targeted the single-layer perceptron, not the multi-layer networks Rosenblatt had already begun exploring. This distinction was known and the choice was not incidental: demonstrating the limits of the simplest possible architecture, then allowing those limits to discredit the broader connectionist program, was a way of undermining Rosenblatt's research (Dreyfus, [*What Computers Still Can't Do*](https://ia801700.us.archive.org/30/items/dreyfus-what-computers-still-cant-do-a-critique-of-artificial-reason/Dreyfus%20-%20What%20Computers%20Still%20Can%27t%20Do%20-%20A%20Critique%20of%20Artificial%20Reason_text.pdf)). The result was a redirect toward symbolic AI — a paradigm fundamentally Cartesian in its commitments: mind as symbol manipulator, knowledge as explicit representation, reasoning as rule application. The approach dominated AI research through the 1970s and 1980s without producing systems that could engage with the open, context-dependent, embodied character of language and meaning.

**The parallel philosophical development.** While Rosenblatt was building perceptrons and Minsky was preparing their critique, Merleau-Ponty was developing his [*Phenomenology of Perception*](http://faculty.las.illinois.edu/rrushing/581b/ewExternalFiles/Merleau-Ponty%2C%20Phenom%20of%20Perception%20Selections.pdf) that reached an analagous epistemological conclusion by a different route. Knowledge is in the doing. The perceiver and the perceived co-constitute each other in a circular movement that resists causal decomposition. Meaning is not stored in components; it is constituted by relations. These were philosophical arguments about human perception — but they described, with precision that would only become apparent decades later, the structural properties of a system trained by gradient descent on a statistical distribution of human language.

**Dreyfus — right, rejected, and too early.** Hubert Dreyfus, drawing on Heidegger and Merleau-Ponty, argued in *What Computers Can't Do* (Dreyfus, 1972) and *What Computers Still Can't Do* (Dreyfus, 1992) that the symbolic AI program would fail — not for engineering reasons but because the proposed design could not solve two fundamental challenges, including: (i) the frame problem (i.e., changes in context that shift the meaning of representations, leading to infinite regress) and (ii) the common sense problem. While Dreyfus' arguments were correct, the position was nearly impossible.

Dreyfus was blunt and the friction he generated had material consequences. His critique threatened the funding rationale of an entire research program, and the AI establishment, largely organized around Minsky's symbolic paradigm at MIT, ensured his access to that environment was curtailed. He found himself making the right argument in the wrong room, at the wrong moment, against the most powerful figure in the field. His views were not without allies — some researchers gradually recognized the force of the phenomenological critique ([*Skillful Coping* (2014)](https://academic.oup.com/book/10163?login=false)). Dreyfus was not wrong. He was simply two architectures too early — and in the impossible position of dealing with Minsky.

While Dreyfus was critical of AI designs, he remained open to their possibility. On neural networks and supervised learning, he wrote:

> "Neural networks are almost as dependent upon human intelligence as are GOFAI systems, and their vaunted learning ability is almost illusory. What we really need is a system that learns on its own how to cope with the environment and modifies its own responses as the environment changes." (Dreyfus, 1992, p. xxxix)

The transformer is a significant step toward that specification. Pretrained on the statistical structure of human language without supervised labels. Refined through reinforcement learning from human feedback. Adapting dynamically to context through attention rather than applying fixed rules. Researchers built it without reading Dreyfus. He described the direction without knowing what architecture would begin to make it possible.

Whether it goes far enough is still open. Writing in 1988, Dreyfus held the line:

> "If Heidegger and Wittgenstein are right, human beings are much more holistic than neural nets. Intelligence has to be motivated by purposes in the organism and goals picked up by the organism from an ongoing culture. If the minimum unit of analysis is that of a whole organism geared into a whole cultural world, neural nets as well as symbolically programmed computers still have a very long way to go." (Skillful Coping, p. 230)

LeCun's argument that LLMs cannot accommodate the dimensionality of the world — that genuine coping requires grounded world models, not statistical patterns over text — is this same critique, updated ([Tech Crunch](https://techcrunch.com/2024/10/16/metas-ai-chief-says-world-models-are-key-to-human-level-ai-but-it-might-be-10-years-out/), 2024). The transformer has effectively displaced symbolicism as the dominant paradigm — the architectural direction Dreyfus anticipated has prevailed. The embodiment thesis remains live. Dreyfus died in 2017, the same year *Attention is All You Need* was published.

What has changed is the relationship between the disciplines. Philosophy and computer science are no longer talking past each other — they are converging on the same object, from different angles, with tools the other needs. Phenomenology supplies the vocabulary for what distributed, context-dependent, holistically entangled systems do. Computer science supplies the architecture that finally instantiates what that vocabulary was always trying to describe. Each is now better positioned to assist the other than at any prior moment in the history of either field.

**The current moment.** The transformer is a connectionist architecture — Rosenblatt's intuition somewhat vindicated at scale. Its weights are not representations; they are the statistical residue of training on human language, analogously entangled in the way Merleau-Ponty described for embodied knowledge. The field now faces an interpretability challenge. Shining a light on the black box discovers weights, not representations, over which meaning is distributed.

**Computational phenomenology is the field this convergence makes possible.** The disciplined study of AI systems through their distributional phenomena, grounded in the phenomenological tradition's commitment to reading what appears rather than inferring what is hidden. It is the field that the transformer's architecture finally makes technically coherent, that Rosenblatt's epistemology anticipated, that Merleau-Ponty's method prefigured, and that Dreyfus demanded without knowing what architecture would make it possible. 

Dreyfus cites "What Computers Should Be Doing" by C.E. Shannon, the inventor of information theory, when he makes the argument that a machine would require a body ("What Computers Still Can't Do", p.304): 

> Efficient machines for such problems as pattern recognition, language translation, and so on, may require a different type of computer than any we have today. It is my feeling that this will be a computer whose natural operation is in terms of patterns, concepts, and vague similarities, rather than sequential operations on ten-digit numbers.

The hostility between philosophy and AI research has historically resembled two weakened bee colonies being stacked together without preparation — each perceiving the other as a threat, the encounter likely to destroy both. A beekeeper's practice for this situation is to place a sheet of newspaper between the boxes. The bees spend days slowly chewing through it. By the time they make contact, pheromones have mixed, the boundary has been decomposed gradually, and what might have been a fatal encounter becomes a successful merge. The newspaper is not a wall — it is a rate-limiting medium that gives both colonies time to socialize before the shock of contact.

HIF is a framework for that field, and this paper is one instantiation. Not a claim that phenomenology solves AI interpretability, nor that AI systems instantiate phenomenological subjects — but a demonstration that distributional observation produces interpretive results that component decomposition cannot alone.

---

## Framing: Computational Phenomenology

**Computational phenomenology** — the disciplined reading of AI systems through their phenomena: distributions, entropy traces, superposition patterns, circuit co-firing, activation geometries, attention shifts, and whatever else structure makes visible. Component decomposition is phenomenal input to this reading. What a circuit does, how features overlap in superposed directions, where activation geometry clusters — these are appearances worth attending. Behavioral metrics constitute a model-agnostic observation layer; mechanistic interpretability could supply a second, architecture-specific layer for models where weights are accessible — together they would approach a complete behavioral fingerprint.

### Computational Phenomenology and Computational Hermeneutics

**Computational Phenomenology is the field.** Its orientation is direct: read what the model's forward pass makes appear, decline the two uses that the object's structure makes unavailable — causal attribution and low-dimensional reduction — and treat the distribution as the level at which all internal structure resolves into a single observable expression. The question it poses is phenomenological: *what appears, and how is that appearance structured across contexts?* 

**Computational Hermeneutics is the method.** It arises not from a contingent limitation pending better tools, but from an irreducible denial of access that the architecture of language model inference imposes on any interpreter.

The denial operates at three levels. 

1. *The forward pass is ephemeral* — at the moment a token is generated, the attention weights, intermediate representations, and logit vector that produced that generation are not persisted. Teacher forcing can reconstruct input-side distributions from a subsequent run; it cannot replay the exact generative event. 
2. *Backpropagation obliterated its own trail* — even with full weight access, gradient descent adjusts every parameter simultaneously in response to every training example, distributing the signal across the full parameter space in a way that admits no disentanglement. The weights are the residue of that process; there is no path from parameter to meaning to retrace. 
3. *Closed models enforce total denial* — the generating model's internal state is explicitly withheld. What remains is the generated text and a truncated logprob vector.

This is the hermeneutic condition — not as analogy but as structural fact. Classical hermeneutics arose in response to an identical epistemic situation: interpretation of texts whose originating cultural context, authorial intent, and grounding artifacts were no longer accessible. The response was not to conclude that interpretation is therefore impossible. The response was to develop a rigorous methodology for disciplined reading *under those conditions* — iterate between instruments, disclose biases, treat divergence between readings as evidence rather than noise, and refuse the pretense of neutral access to an original that cannot be recovered. Computational Hermeneutics applies that methodology to language models.

**Computational Phenomenology** supplies the orientation — what to read and at what level. **Computational Hermeneutics** supplies the practice — *how* to read under the specific conditions of denied access that language model inference imposes.

---

### Why Distributions, Not Representations

**Interpretability finds real structure wherever it looks. The question is what that structure can establish.**

Activations, circuits, feature directions, superposition patterns — these are genuine observations, and computational phenomenology reads them as such, alongside distributions, entropy traces, and hermeneutic attention readings. What it declines is not the observation but a specific claim those observations are sometimes asked to carry: that a component, extracted and examined in isolation, causally explains why the model produced a particular output. LLM parameters are non-representational by design — meaning is distributed, not atomic, constituted by relations rather than stored in locations. The distribution is where the full circuitry expresses itself completely.

- **Distributions are not proxies.** They are the directly observable phenomenon where the full parameter space expresses itself. The distributional behavior of a language model across contexts *is* its behavioral range — not a downstream shadow of something more fundamental stored upstream.
- **Mechanistic observations are inputs to the reading.** Activations, circuits, and vector geometries carry genuine interpretive content. A circuit that consistently co-fires with an operation is a real appearance worth attending to. What it cannot establish is that the co-firing *explains* the output — because meaning is constituted by the relations between components, and those relations are not recoverable from any component examined in isolation.
- **Superposition is a phenomenon CP absorbs.** The finding that networks represent more features than they have dimensions — encoding them as overlapping, near-orthogonal directions — contextualizes entropy, focus, and calibration. High superposition pressure is visible as distributional interference. CP reads that interference at the output surface; mechanistic analysis reads its geometric structure in weight space. Both observations are real. They are different angles on the same phenomenon.
- **The distribution is the most directly observable surface of the model's learned structure.** It is not a compressed shadow of something more fundamental stored upstream — it is where the full circuitry expresses itself in directly legible form, the forward pass brought to completion. The claim is not that activation geometry or circuit structure are hidden from the analyst; it is that the distribution is the level at which all internal structure participates simultaneously and becomes readable as behavior.

#### On the Cartesian Inheritance and the Limits of Engineering Intuition

**The decomposition instinct has a proven track record — in the right domain.**

Descartes established the norm: divide complex systems into simplest components, understand each in isolation, reconstruct the whole from parts. For **designed systems** — modular, with specified functions and defined interfaces — this works. Trace the signal. Isolate the failure. Find the responsible component.

**The challenge is domain mismatch.** A language model is not designed — it is trained:
- No specified interfaces, no designed modules, no intended component behaviors to recover by inspection
- Decomposition recovers real structure: circuits, features, directions
- But that structure was not designed to be coherent in isolation — it is coherent only within the full circuitry
- The "stories" decomposition tells are real trees, not the forest

**The engineering background shapes the expectation further.** Interpretation should yield coherent local stories: this circuit does induction, this feature represents gender, this direction encodes sentiment. These stories are real — but a language model's behavior is not the sum of its modules' behaviors. It is constituted by their relations, and those relations are not readable from any component in isolation.

**Heidegger's account of the present-at-hand illuminates the structural source.** The present-at-hand is the theoretical stance: treating entities as isolated objects, stripped from their context of use. Activation-space decomposition does precisely this — it extracts representations from the circuitry in which they participate and examines them as objects. The stance yields real information, but it cannot recover what abstraction loses: each component's position within the relational whole that constitutes meaning. The forest cannot be reconstructed by examining trees one at a time.

---

### The Incommensurability of Distributed Meaning — and Why Measurement Works

**Meaning in a trained LLM is distributed across billions of parameters and cannot be verbalized — not as a practical limitation, but in principle.**

Any verbal account of what the model "represents" — this head tracks syntax, this circuit does induction — is a projection into a space so much lower-dimensional than the model's parameter space that the projection necessarily destroys most of the information. Natural language does not have the resolution to span the meaning-space of a model trained on human language at scale.

Rosenblatt understood this at the perceptron level:

> "The number of functional units in the storage system should be much less than the number of forms or memories to be retrieved." (Rosenblatt, 1957)

The perceptron's knowledge was in its statistical weights — not in any programmer-assigned representation, and therefore not fully sayable.

Merleau-Ponty named an analogous structure in embodied cognition, which he uses to describe scenes like those below:

- A public speaker reading a crowd adjusts register, pace, and emphasis without naming the adjustment. The tuning happens below the level at which it could be named.
- A ping pong player has no time to reflect. The response *is* the knowledge. There is no prior representation to consult; the skill is the doing.

At the scale of a language model, this incommensurability is magnified.

**The Merleau-Pontian limit — and the specific AI variant.**

Merleau-Ponty's deeper point is not merely that embodied knowledge exceeds description. Causality itself is impossible to demonstrate for a human being, because the possibility space is infinite. The conditions required for causal isolation cannot be met when the subject is embedded in an open, temporally extended world with a body, a history, and an environment in continuous mutual constitution. There is no clean experiment — only the phenomenon, and the reading.

The limitations of AI provide conditions that Merleau-Ponty could not: a finite, enumerable snapshot of the possibility space at each step. The softmax distribution is exactly this — a complete accounting of the model's distributional state at a given moment, with probabilities assigned to every token in the vocabulary. Computational phenomenology is not an approximation of a better method; it is the method the structure of the object makes available.

Backpropagation introduces its own form of the entanglement problem — one that is not practical but constitutive. Gradient descent does not write meaning into individual weights. It adjusts every weight simultaneously in response to every training example, distributing the signal across the full parameter space in proportion to each weight's contribution to the loss. The weights are not representations — they are the residue of a gradient process over billions of examples, entangled with the training data, with each other, and with the loss landscape in a way that admits no disentanglement. Asking what a particular weight, layer, direction, or circuit *represents* cannot be answered by examining those components; the meaning was never encoded into them as representations. It was constituted through their collective interaction under training.

Merleau-Ponty diagnosed the same structure in perception itself. His argument is not only that the perceiver exceeds description — it is that the percept influences the perceiver and the perceiver influences the percept in a circular movement that makes linear causal attribution structurally unavailable. The touching hand is also touched. Perception is not a one-way transaction from world to subject; it is a chiasmic, reversible co-constitution in which neither term is prior.

**The Transformer enacts this circularity twice — in different registers, with different consequences.**

The first circularity is within the forward pass. Every token simultaneously attends to all others and is attended by all others. Each token's Query reaches out to every Key; each token's Value is read by every other's attention weights. What a token "means" in a Transformer is not a property it carries — it is a position it occupies in a relational structure that has no privileged origin.

The second is across training. Backpropagation adjusts the weights in response to the loss; the loss depends on the model's outputs; the outputs depend on the weights. At each training step, the model's current parameter state determines how it perceives the training example, and the gradient determined by that perception then reshapes the parameters that did the perceiving. The weights shaped the perception; the perception shaped the weights. This is the Merleau-Pontian chiasm enacted computationally: not temporal and embodied, but iterative and mathematical — and producing the same epistemological consequence.

This is why mechanistic observations, even when they succeed technically, cannot carry the specific claim of causal attribution. Finding that a direction correlates with a concept, or that a circuit co-fires with an operation, reports a real regularity in the parameter space. That regularity is worth reading. What it cannot establish is that the component is causally responsible for the output — because the meaning was constituted by the circular process of all components together, and decomposition cannot recover what circularity produced.

**The distribution is where both circularities resolve into a complete, observable expression** — the forward pass brought to completion. It is not a proxy for something more fundamental stored upstream; it is the phenomenon in which all structure — weights, circuits, feature directions, superposition geometry — participates simultaneously. Reading the distribution is not second-best to mechanistic analysis. It is a different level of analysis that reads the full circuitry in its resolved expression, where mechanism and distribution are no longer separate things but one thing seen whole.

**HIF's answer: description is not the only interpretive act.**

Rather than projecting distributed meaning into verbal descriptions, HIF measures the *character* of the model's distributional behavior — the shape of its engagement with meaning across contexts.

- **Calibration makes this possible.** Entropy alone — the spread of the output distribution — is a number. Calibration is the condition under which entropy discloses something real: whether high entropy reflects genuine openness, and whether low entropy reflects real confidence. Without calibration, entropy is noise. With it, the model's distributional behavior is readable as a meaningful horizon.
- **The six metrics characterize this character across six dimensions — label-free, operational, requiring no ground-truth annotations:**
  - **Stability:** uncertainty the model carries entering generation
  - **I/O Correlation:** whether output entropy tracks input entropy
  - **Focus:** how broadly probability spreads at each output step
  - **Sensitivity:** how much the output distribution shifts under meaning-preserving paraphrase
  - **Continuity:** whether competing trajectory branches converge or sustain competing directions
  - **Surprise:** whether selections respect the model's own distributional commitments, or play the field against concentrated probability mass (open models only)

None of these metrics describe what the model means. All characterize the shape of how the model engages with meaning. Calibration — measured via ECE against labeled evaluation sets — provides periodic validation of the entropy signal: confirmation that low entropy tracks genuine confidence and high entropy tracks genuine openness. The monitoring layer scales independently; calibration anchors it at governance checkpoints.

**The forest is not described by naming trees.** It is measured by the density of canopy, the dispersion of undergrowth, the path light takes through it. These measurements are not the forest — but they are how the forest discloses itself to a human interpreter who cannot span it in a single view.

---

### Logits, Distributions, and the As-Structure

**Two levels of the model's output play different roles in the analysis.**

**Logits** — the raw, unnormalized output of the final linear layer — are pre-thematic (ante-predicative): the computational background from which a distribution will emerge. They carry no directly interpretable weights (i.e., representations), no relative likelihoods, no notion of "possible next token."

**The softmax distribution** is where the computational grip becomes legible. Each token carries a probability weight reflecting the equilibrium the forward pass has found between the current context and the trained parameter structure. Nothing lies behind it to uncover — the grip is fully observable in the distribution.

**The as-structure** is the fact that the distribution is never a flat list of arbitrary symbols. Weights are shaped by parameters and context together: each token presents itself as probability mass within *this* continuation's distribution. This structural shaping makes the distribution interpretable and makes cloud phenomena — convergence, clustering, divergence, diffusion — meaningful descriptions rather than arbitrary classifications.

---

### Epistemological Scope

**Two interpretability traditions. Different objectives. Different utility ceilings.**

**Mechanistic interpretability** — decomposing activation space: finding atomic features in superposition, tracing circuits, dampening vectors to isolate causal directions:
- The neuroimaging of language models: structure rendered over high-dimensional representation over time
- Like fMRI, it yields genuine insight when combined with observable behavioral correlates
- Like fMRI, its claims remain inferential at the point where they reach for representational sense-making — where the researcher wants to say not just "this direction correlates with this output" but "this is what the model means"
- That gap is not a methodological failure; it is constitutive. The non-representational substrate will not yield representational sense-making through decomposition, however precise
- **Best fit for:** diagnosis and treatment — locating circuits that produce harmful behavior, targeting fine-tuning, predicting capability emergence

**Horizonal Interpretative Reading** — reading the distributional grip across contexts:
- Objective: explainability, not intervention
- Not what to fix or where to intervene — what the model is doing with meaning: how its distributional horizon is shaped by context, where its possibility space opens or narrows, how its behavioral range varies across the full spectrum of use
- The computational grip is fully expressed in the distribution. Claims are about what is directly observable in that grip — not inferences about what the activation geometry conceals
- **Best fit for:** behavioral characterization — auditing deployed systems, assessing behavioral range, characterizing stability and sensitivity across context regimes

#### What Makes HIF Different from Prior Logprob Approaches — and from the Current Interpretability Frontier

**Both the mechanistic and the distributional programs are genuine contributions to the colony.** What they need is not a critique but a hermeneutic medium through which their findings can socialize before meeting — the newspaper between the boxes, again, but this time between two research programs rather than between philosophy and engineering.

Entropy, perplexity, and logprobability-based uncertainty measures appear throughout the interpretability literature. The objection is reasonable: if logprobs are already in use, what does HIF add? **The answer is not a new instrument but an overarching interpretive framework** — one that structures these measurements into a hermeneutic account of model behavior rather than treating them as isolated signals. The measurement is commonplace; the structured reading of phenomena in their mutual relation — calibrated against the full distributional envelope, iterated between part and whole — is not.

**What neither provides is an integrating framework** through which these observations can be structured together with distributional readings and calibrated against behavioral context. The circuit description and the neuron explanation are readings of individual elements. What turns a correlation into something an analyst can act on is a hermeneutic framework that moves between parts and whole, calibrates each observation against the full distributional picture, and refuses to let component descriptions stand in for behavioral understanding.

HIF prepares that ground. A circuit-level finding gains interpretive weight when situated within an entropy trace that shows how the relevant superposition pressure manifests in the output distribution across contexts and regimes. A neuron explanation becomes behaviorally meaningful within a calibrated envelope that characterizes what the model actually does under varying conditions — not only what structure correlates with what activation. The programs occupy different levels of the same reading. HIF is the framework that makes those levels legible to each other.

#### Dissolving the Interpretation / Explanation Debate

The debate between mechanistic explanation and behavioral interpretation has persisted because both sides frame it as a methodological dispute. Mechanistic approaches seek decomposition as explanation: which circuit, neuron, relevancy-weighted token, or activation direction caused this output? Behavioral approaches have often accepted the same premise from the opposite direction, positioning themselves as reading the texts that those decomposable components produce. The methods disagree. The underlying assumption does not: that a neural network's behavior can be understood by decomposing it into elements that carry interpretable meaning in isolation. This is the **decomposability hypothesis** — and it is the premise that fails.

Merleau-Ponty and Heidegger dissolved the rationalism/empiricism debate not by synthesis but by abandoning its shared premise. Rationalists and empiricists disagreed about the source of knowledge while both presupposing that knowledge consists in accurate mental representations of an external world. Merleau-Ponty showed that this representationalist premise fails for embodied cognition: meaning is constituted by skilled engagement with a world, not by representations of it, and cannot be decomposed into the independent contributions of subject and object, stimulus and response. The debate did not resolve on one side. It dissolved — because the question both sides were arguing about presupposed something that was not the case.

The decomposability hypothesis fails for trained neural networks in the same structural sense. Backpropagation adjusts every parameter simultaneously in response to every training example: the resulting weights are constitutively entangled with each other and with the full training distribution in a way no subsequent analysis can reverse. Superposition compounds this — features are distributed across overlapping, near-orthogonal directions that cannot be read in isolation. A network's meaning is constituted by the relations between all components under the full circuitry, not by any component's individual contribution. This is not a practical obstacle to decomposition. It is the structural condition of how gradient descent builds meaning. Every technique that asks a component to explain an output — circuit tracing, attention rollout, layerwise relevancy propagation, sparse autoencoder attribution, and their successors — carries this assumption as a precondition of its claims. The sophistication of the technique does not alter what the assumption presupposes.

What remains when the decomposability hypothesis is released is neither explanation nor interpretation in their classical senses, but the **reading of phenomena in a computational world**. The distribution is where all structure — weights, circuits, superposition geometry, attention patterns — resolves simultaneously into a single observable expression. **Calibration** is the significance condition: whether what appears in that distribution tracks something that matters in the deployment context. Calibration is the computational analog of Heidegger's significance — not a value judgment imposed from outside, but the structural condition under which the distribution discloses itself as a meaningful horizon rather than undifferentiated noise.

Component observations remain phenomena. Circuit attributions, feature directions, relevancy scores — these are appearances that add texture to the reading. Computational Hermeneutics reads them alongside distributions, entropy traces, and JSD measures, as part of the same hermeneutic circle. What it does not do is treat any of them as decomposable causes recoverable from components examined in isolation — not because causes do not exist, but because the decomposability hypothesis that would make them legible from components alone is untenable. The debate between explanation and interpretation presupposes that hypothesis. When it goes, so does the debate. What remains is the reading of what appears, and the pursuit of significance in a computational world.

---

### On the Relationship Between Hermeneutic Attention and Mechanistic Findings

**Computational phenomenology absorbs mechanistic observations rather than opposing them.** Superposition, circuit co-firing, feature directions, activation geometries — these are real phenomena that CP reads alongside distributions and entropy traces. The distinction CP draws is not between what is observed but between two things an observation can be asked to establish: phenomenal content (this structure appears here, it correlates with this behavior) versus causal attribution (this component explains why this output was produced). The first is always available. The second is not — for reasons that apply to mechanistic and distributional analysis alike.

This distinction is sharpest when considering hermeneutic attention alongside superposition. An objection naturally arises: "DistilBERT's readings are not the generating model's own attention weights, so they are no more grounded than mechanistic attention claims." Working through this objection clarifies what both approaches can and cannot establish.

**Superposition as phenomenon.** Superposition is real. The finding that neural networks store more features than they have dimensions — by encoding them as overlapping, near-orthogonal directions in activation space — is a genuine structural observation, and one that Rosenblatt had already stated in 1957: **the storage system must have fewer units than forms to be retrieved.** The mathematical formalization is new; the observation is not. Used as a phenomenon, superposition contextualizes behavioral interpretation: knowing a model operates in a superposed representational regime tells you something about why interference patterns appear, why certain features co-activate, why perturbations propagate the way they do. This is phenomenal content — structure that adds texture to the reading of behavior without claiming to explain it causally.

**Superposition as attribution theory.** The problem arises when superposition is recruited to *explain* specific outputs. Three compounding assumptions make this overreach:

- **The toy model gap.** The foundational results derive from a deliberately minimal architecture: a ReLU output model compressing sparse, high-dimensional inputs into very low-dimensional representations under idealized conditions. The authors describe their findings as "extremely preliminary" and explicitly acknowledge "substantial gaps between their toy task and realistic transformer behavior." Extrapolating to causal claims about production transformers does work the original papers do not license.
- **The locality assumption.** Attribution requires that meaning be localizable — a concept can be assigned a direction, a circuit a function. The evidence is typically statistical: clustering in a projection, consistent co-firing, higher activation density in a region. Density in a low-dimensional projection of a distributed computation is not evidence that meaning resides there; it is evidence that optimization found regularities visible from that angle.
- **The faithfulness problem.** Jain & Wallace (2019) showed that even direct attention weights from the generating model are not faithful explanations of its predictions. The interpretive leap from extracted component to behavioral mechanism is the presumption — regardless of which model's internals you are reading.

**The parallel to hermeneutic attention.** HIF's use of DistilBERT is structured identically to the legitimate use of superposition: as phenomenal content that contextualizes behavioral interpretation, not as attribution theory. DistilBERT reads the prompt and continuation as texts. Shifts in its attention — which tokens gain or lose weight under perturbation, how structural load is distributed across the input — are real observations about real linguistic objects. They do not explain why the model generated what it generated. They add texture to the entropy trace and deepen calibration.

The difference is one of scope and disclosure. DistilBERT operates on the text the model actually produced under real inference conditions — not a toy reconstruction of idealized sparse vectors. Its instrument bias is fully disclosed: it reads language as it learned to read language. Every reading carries that bias uniformly, making readings comparable across all six models on identical terms — including closed API models where weight access is unavailable and mechanistic analysis is not designed to operate. The hermeneutic approach is not a substitute for mechanistic analysis where that analysis is possible; it is the reading that operates on the text regardless of what produced it.

When superposition is treated as phenomenal context rather than attribution, it and hermeneutic attention are allies, not competitors. Both contribute to a fuller picture of model behavior. Both become presumptuous only when they overreach into causal explanation.

---

### Claims and Non-Claims

The paper's argument is specific. Stating both what it claims and what it does not will pre-empt the most common misreadings.

**Claims**

- HIF characterizes language model behavior through six distributional metrics grounded in Shannon entropy.
- Entropy becomes interpretable only when calibrated against context — whether high entropy reflects genuine openness and low entropy reflects genuine commitment is a question the framework poses and the prompt regime helps answer.
- Sensitivity Score (JSD under meaning-preserving paraphrase) is an operational calibration test: low JSD under paraphrase is evidence that entropy is tracking semantic content, not surface form.
- HIF applies to both open and closed models; the available metric subset differs by access level (full suite for open models; output-side metrics only for closed API models).
- Mechanistic interpretability findings — circuits, feature directions, superposition geometry — are phenomenal inputs to the framework, not adversaries of it. HIF situates those findings within a behavioral envelope. Component-level findings gain interpretive weight when calibrated against the distributional picture across contexts and regimes.

**Non-Claims**

- HIF does not reveal internal causal mechanisms. The framework reads behavioral surfaces; it does not trace how an output was causally produced.
- HIF does not replace mechanistic interpretability. The two approaches answer different questions at different levels; neither is sufficient alone.
- DistilBERT's attention readings do not approximate the target model's attention. DistilBERT reads the generated text as a text; its output reflects its own pre-training, not the generating model's internal processing.
- Entropy is not equivalent to semantic uncertainty unless calibrated. Raw entropy values require contextual interpretation; the framework does not claim entropy alone is a sufficient behavioral measure.
- Low entropy is not always good; high entropy is not always bad. The Equilibrium band is a calibration concept, not a performance metric.
- The phenomenological analogues — computational grip, computational arc, the They, as-structure — are disciplined methodological tools for describing distributional behavior in non-representational systems. **HIF does not claim that transformers have experience, embodiment, Dasein, or worldhood.** The philosophical concepts illuminate the structure; they are not ontological attributions.

---

## Measurement Foundation: Entropy and Calibration

### Shannon Entropy

In information theory, the entropy of a random variable quantifies the average level of uncertainty or information associated with the variable's potential states — the expected amount of information needed to describe the state of the variable, given the distribution of probabilities across all potential states.

For a discrete random variable $X$ distributed according to $p : \mathcal{X} \to [0,1]$:

$$\mathrm{H}(X) := -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

In HIF, $X$ is the token distribution at a generation step, and $p(x)$ is the probability the model assigns to each candidate token. H = 0 means one token has all the mass — the model was certain. H is maximised when probability is spread uniformly — the model was maximally uncertain.

Entropy is always relative to its ceiling: $\mathrm{H}_{\max} = \log_2 |V|$. For GPT-2's 50,257-token vocabulary, $\mathrm{H}_{\max} \approx 15.6$ bits — the value if every token were equally probable. Every entropy value in HIF is interpretable relative to this bound; a raw bit count without reference to $\mathrm{H}_{\max}$ is unanchored.

Every HIF metric is either entropy, derived from entropy, or computed by comparing two entropy traces.

### Calibration

**Calibration is what makes entropy interpretable.** Entropy is a number — it tells you the spread of a distribution, but not what that spread means. A model with high entropy at every step could be genuinely uncertain (good — it is exploring open semantic space) or broken (bad — failing to commit where it should). Calibration resolves that ambiguity: it is the claim that this model's entropy tracks real uncertainty — that when entropy is high, the context genuinely warrants it, and when entropy is low, the model has real reason to be confident. Without calibration, the entropy trace is uninterpretable.

---

## Part 2: Architectural Design

### The Horizon Structure

The analysis is organized as a four-zone model: wide possibility at the input, constrained transformation at the center, structured output generation, and branching trajectory.

```
INPUT SIDE    — distributional structure of the prompt, token by token
     ↓
   CENTER      — transformation diagnostics (entropy ratio, HIF perplexity, equilibrium)
     ↓
OUTPUT SIDE   — per-step generation trace: distributions, semantic clouds, exposure
     ↓
 TRAJECTORY   — branched stochastic rollout across B paths × R steps
```

Three models play three strictly separate roles:

| Role | Model | Function |
|------|-------|----------|
| Model under analysis | Open-weight and closed API models | Generates logits and token distributions (open-weight, via HuggingFace); actual I/O via provider API (closed); DistilBERT reads the resulting texts as a separate analysis instrument for all models |
| Embedding model | Sentence-transformer (Gemma 300M / MiniLM fallback) | Clusters candidate tokens by semantic content |
| Analysis transformer | DistilBERT (bidirectional encoder) | Reads prompt and continuation as texts; anchored in DistilBERT's own pre-training |

DistilBERT reads the prompt and continuation as texts for all models — open and closed alike. It never receives logits or distributions from the model under analysis, and it never accesses the generating model's internal attention weights. The embedding model never knows which token was selected.

---

## Part 3: Hermeneutic Reading

### Hermeneutic Attention

The debate over attention is not settled in one direction. Jain & Wallace (2019) demonstrated that attention weights from a model's own forward pass are not faithful explanations of its predictions: alternative attention distributions can produce equivalent outputs, and the tokens a model attends to are not reliably those causally responsible for what it generated. Wiegreffe & Pinter (2019) contested the conclusion on its own terms — showing that attention can constitute a valid explanation under different faithfulness criteria, and that the relationship between attention structure and model output is not so easily dismissed. Both findings converge on the same interpretive position: attention is a phenomenon — structured, co-occurring with the output, correlated with the distributional decisions the model makes — and that correspondence is real regardless of whether it is causally constitutive.

The hermeneutic attention layer is structured as a response to that position. A bidirectional encoder reads prompt and continuation as texts — four passes that disclose structural load, semantic resonance, and how the joint context evolves as generation proceeds. These readings enter the circle as the structural anatomy of the text as text. From the same computational sediment, a set of derived instruments — each measuring a different register of the forward pass — is deployed at every token position. Together, the encoder readings and the instrument overlay do not produce independent analyses; they produce a field of relationships that the human interpreter moves between, returning to each part with what the other has disclosed, spiraling toward a reading of the behavioral envelope that neither alone could produce. Attention, in this design, is not a cause to be isolated. It is one phenomenon among several through which the forward pass discloses itself to a reading that will always be partial and always be ongoing.

When enabled, the bidirectional encoder performs four readings. Two methodological notes apply throughout: (1) The bidirectional encoder makes no claim about the target model's attention — it reads the texts the model produced, not the model's internal processing of them. (2) Its readings are anchored in its own pre-training; they reflect the encoder's learned representations, not a neutral semantic ground truth. The value is the consistency of the instrument across readings — the same bias applied uniformly makes comparison informative.

**Reading 1 — Prompt alone**: the internal attention structure of the input as its own text. Load-bearing tokens, attention weight distribution, perturbation deltas.

**Reading 2 — Continuation alone**: the internal attention structure of the generated continuation as its own text. Which tokens organize the continuation's own semantic field?

**Reading 3 — Resonance comparison**: an analytical comparison of the two independent readings. Which continuation tokens echo the load-bearing structure of the input (anchored, resonance > 0.5)? Which have moved away from the input's semantic anchors (free-floating, resonance < 0.2)?

**Reading 4 — Joint trajectory**: The bidirectional encoder runs on `[prompt + continuation[:k]]` at regular intervals. The cross-attention block (continuation tokens → prompt tokens) is extracted at each checkpoint, giving a time series of how each prompt token holds or releases its weight in the growing joint context. Prompt tokens are classified as: persistent (consistently load-bearing), fading (high early weight, low late), or emerging pivot (gaining weight as the continuation develops).

---

### The Five-Instrument Overlay

The hermeneutic attention layer deploys its instruments simultaneously at each token position — Spread (■), Entropy (●), Wager (▲), Shift (◆), Horizon (▼). They do not produce independent readings. They produce a field of relationships between values and their distributions. Reading that field — under conditions where the computational world does not guarantee transparency and often actively denies it — is the work. The framework's choices of data, tools, and methods are each configured for that condition.

**Data.** The computational sediment — attention distributions, output distributions, entropy traces, token-level logit vectors — must not be compressed unless the reading requires it. The forward pass is ephemeral; backpropagation obliterated its trail; closed models return only truncated logprobs. What does appear is already a residue of denied access. To compress it further by default is to compound the denial. Apparent contradiction in the data is not noise to resolve before reading begins. Spread (■) and Entropy (●) vary independently — focused attention can coexist with high output entropy; diffuse attention with committed output. Both measurements are preserved. The gap between them at a given position is a structural feature of the computational world worth attending to. Where Spread (■) and Entropy (●) diverge, the reading begins.

**Tools.** The generating model and the bidirectional reader are the two primary instruments. Both must be capable of matching the dimensionality of the computational sediment being exposed to phenomenological reading. The generating model produces data at the full dimensionality of its learned manifold — distributions, entropy, logit vectors. The bidirectional reader approaches the same text from its own pre-training horizon, without access to the generation process, and must have sufficient representational capacity to read what the generator's sediment makes available. A reader that cannot match the dimensionality of the sediment will miss structure that the denial already makes difficult to see — and the gaps in its reading will be invisible. Both roles are stable across instrument generations; the specific tools that fill them are replaceable as capacity grows. The derived instruments operate on the data both produce, across three registers: high-dimensional space (full distributions, attention matrices), statistical artifacts and derivatives (entropy, JSD, correlation), and lower-dimensional projections (semantic clouds, attention heatmaps, icon overlays).

**Methods.** JSD is the tool used when the relationship being read is the distance between two full distributions. It appears at four levels: Sensitivity Score (baseline versus perturbed output distributions), Shift (◆) (consecutive attention rows), Horizon (▼) (cross-reader), and the output distribution strip (consecutive generation-step distributions). Its scope is bounded: it measures distance between distributions. Distribution features can also be compared without JSD — through correlation, entropy difference, threshold, shape, and structural absence. Not all phenomena in the framework are distributions, and not all distributional relationships are distances.

**Horizon (▼)** is the instrument that reads explicitly across both tools, and the one that most directly embodies the cross-reader method. It formalizes the distance — as JSD — between the generating model's cumulative entropy landscape (the distributional difficulty accumulated in its forward pass, positions 0..i) and an external reader's attention distribution at the same position. One reader is causal and autoregressive — it produced the text from inside its training horizon. The external reader approaches the same text after the fact, from its own horizon, without access to the generation process. Their JSD at each position is the measured distance between two interpretive horizons reading the same text. The existing methodological caveat — that the external reader makes no claim about the generating model's internal state — is correct as far as it goes. The stronger claim: the multi-reader structure is not a limitation to acknowledge but a methodological foundation to assert. The interpreter of an ancient text never has a single reader. Multiple readings across different horizons, none with access to the original, the differences between them constituting the interpretive field — this is the hermeneutic condition, and Horizon (▼) is designed to operate within it.

Horizon (▼) is therefore a family of cross-reader metrics, not a single instrument. The current implementation pairs the generating model with a bidirectional reader — Horizon (▼_bidirectional) measures their JSD at each position. A causal external reader can be deployed simultaneously — Horizon (▼_causal) measures the JSD between the generating model's cumulative entropy landscape and a causal language model reader's attention distribution over the same text. The two metrics are read together. Where Horizon (▼_bidirectional) and Horizon (▼_causal) align — both external readers track the generator's distributional difficulty in similar proportion — the finding is robust: those positions are structurally load-bearing in both reading directions. Where only Horizon (▼_bidirectional) tracks the generator — the bidirectional reader attends toward positions the generator struggled at, but the causal reader does not — directional structure is present: those positions draw their weight from what comes after, not only from what came before. Where neither external reader tracks the generator's difficulty — both approach the text on their own terms, away from where the generator labored — the sediment is generator-specific: the distributional pressure at those positions is internal to the generating model's training horizon and is not recovered by any external reading. The difference between Horizon (▼_bidirectional) and Horizon (▼_causal) is itself an interpretive field — one that the family opens and a single reader forecloses.

The method that organizes the reading is the hermeneutic circle — and it operates at two scales. Within a reading session, understanding moves between part and whole: you enter at the whole (the behavioral envelope, the generated output), descend to the parts (per-token icons, individual attention rows, a specific layer), return to the whole with what the parts disclosed, revise, descend again. The circle is not vicious — it is the structure of all interpretation. There is no reading that doesn't already presuppose a whole to enter.

Across time, the circle does not close — it spirals. A benchmark establishes an anchor: a reading of the model's behavioral envelope at a specific moment, not a final verdict but a reference whole. Monitoring reads against that anchor. Each new run is a new descent into parts compared against the anchored whole; drift is the phenomenon of the whole changing under the reading. The whole is repeatedly textured — each new reading adds resolution and revision to what the whole shows. The behavioral envelope is never finished. It is progressively elaborated against a computational world that withholds itself fully, changes over time, and makes ongoing reading not a choice but a structural necessity.

---

## Limitations

**Logprob access.** For closed API models, entropy is computed over truncated top-k distributions returned by the provider. The tail below the cutoff is excluded, producing a systematic underestimate of full-vocabulary entropy. The framework brackets this uncertainty: `entropy_bits` stores the lower bound (raw top-k, unnormalized), and `entropy_bits_upper` stores the uniform-tail upper bound. Cross-model comparisons between open and closed models are directionally valid; both bounds support the same qualitative findings.

**Philosophical analogues are methodological.** The phenomenological concepts used throughout — computational grip, computational arc, the They, as-structure — are methodological analogues for describing non-representational distributional behavior. They are not empirical claims about transformer internals and should not be read as such.

---

## Appendix A: Philosophical Foundations

### On the Use of Philosophical Analogues

**Terms are borrowed from phenomenology. Their provenance and limits require explicit statement.**

**Intentional arc** (Merleau-Ponty, *Phenomenology of Perception*, 1945) — the pre-reflective bodily schema accumulated through lived experience. The arc is the organism's sedimented field of possibilities — built up through engagement with the world, not stored as representations, and fully visible in what the body does.

**Computational arc** — HIF's training-phase analogue:
- As gradient descent minimizes prediction loss, the parameter space accumulates the statistical structure of human language use — encoding a best-fit manifold through which all forward passes will run
- Just as the intentional arc is not stored in a representation but is the body's acquired readiness, the computational arc is the entire shaped structure of the parameter space: the residue of training, sedimented and fixed at inference
- Borrowed feature: accumulated experience as pre-given directedness, structuring what is possible without being available for explicit inspection
- The analogy stops where embodiment begins — no body, no being-in-the-world, no caring. The arc is a statistical manifold, not a lived skill
- *Analogy weight and reinforcement learning:* Merleau-Ponty's intentional arc is shaped by consequence-feedback — what succeeded and what failed in a responding world. Pretraining without reinforcement learning is shaped by corpus statistics alone: passive prediction from a fixed dataset, not a history of consequence. This weakens the analogy considerably. RLHF introduces something structurally closer — training signal derived from evaluative feedback on outputs — and proportionally strengthens the arc ↔ grip correspondence.

**Optimal grip** (Merleau-Ponty, *Phenomenology of Perception*, 1945) — the equilibrium the body actively seeks in perception: the best distance from a painting, the angle at which an object reveals its structure most fully. Not passively received but sought until the perceptual field stabilizes.

**Computational grip** — HIF's inference-phase analogue:
- At each generation step, the forward pass moves through the parameter-shaped manifold — weights frozen at their trained, loss-minimizing configuration — resolving the current context deterministically into a softmax distribution. No optimization occurs at inference; the weights have already found their minimum. The distribution is produced by applying fixed parameters, not sought
- The softmax distribution is where the computational grip is fully expressed: each token's probability reflects the model's best-fit response given parameters whose training has already minimized prediction loss
- Borrowed feature: a resolved field of possibilities that discloses itself as a structured, observable distribution. The equilibrium was established by the arc; the grip discloses it
- The analogy stops where the lived body begins — no grip on a world, no motor intentionality, no being-in-the-world. The equilibrium is numerical and pre-given by training, not existential or dynamically sought at inference

**As-structure** (Heidegger, *Being and Time*, 1927) — the way entities present themselves within a context of meaning. A hammer presents itself *as* a tool for driving nails within a context of use, not as an inert object:
- Computational analogue: the softmax distribution is similarly never neutral — each token presents itself as a possible continuation of this kind, in this context
- The weight is not arbitrary; it reflects the token's structural fit within the current context
- Borrowed feature: structural presentation within a shaped field. No equipmental totality, no worldhood, no Dasein transfers

**The They** (Heidegger, *Being and Time*, 1927) — the disembodied everyone-and-no-one that pre-organizes social intelligibility — operates at two levels:
- *Existentiale* (theyness): the formal structural feature — the ontological possibility of a they, independent of any particular instantiation
- *Existentielle* (they-self): the concrete enactment of theyness in a specific being's actual existence
- **Computational theyness (existentiale):** the architectural and training conditions that make it possible for a parameter space to encode a they at all
- **Computational they-persona (existentielle):** a specific trained model as the particular computational they it is — this best-fit manifold, through this corpus, operative in this inference
- Borrowed feature: both are pre-given backgrounds shaping the possibility space without being visible in any single output. The analogy stops where the They's existential features begin — disburdening, distantiality, falling — which belong to Dasein's being-with-others, not to a forward pass

**Averageness / best-fit line.** Heidegger's *averageness* — the They's tendency toward the middle, the undifferentiated, the publicly shared — has a more precise computational analogue: the **best-fit line**. The parameter configuration that minimizes prediction error is a statistical best fit through high-dimensional language space, not a simple mean. Used throughout in place of "averageness."

**Computational dealing / circumspective dealing:**
- Token proximity in the embedding space encodes functional fit: tokens near one another occupied the same functional positions across the training corpus — same roles, same sentence slots, same practical contexts — without explicit classification
- This numerical phenomenon — **computational dealing** — has circumspective dealing as its human analogue
- Circumspection (Heidegger) is the practical awareness of ready-to-hand engagement: knowing-one's-way-around a context of use, an orientation toward what fits, without having to represent it explicitly
- The analogy runs from the computational phenomenon to its human counterpart, not the reverse. The computation does not do circumspective dealing; circumspective dealing is what it looks like from the human side

Two tokens that are theoretically distant may be computationally proximate because in actual use they occupy the same functional position. The converse holds equally — the embedding space encodes functional fit, not theoretical membership.

**Semantic clusters** at each generation step are fields of computational proximity — tokens that belong together by use, not by category. The cloud phenomena follow: convergence is the dominance of a single such field; clustering is the simultaneous availability of multiple distinct functional orientations; diffusion is the breakdown of this structure.

### The Computational They: Parameters as Best-Fit Line

**Heidegger's They — the disembodied everyone-and-no-one that pre-organizes social intelligibility — has a precise computational analogue.**

The They is not a person or a group. It is the structural condition of social intelligibility: what *one* says, how *one* says it, and what *one* takes for granted. Two levels are operative in the computational case.

**Computational theyness (existentiale)** — the formal structural capacity:
- Transformer architecture — attention mechanisms, parameter matrices, training that minimizes prediction loss over a linguistic corpus — is the condition that makes it possible for a numerical system to constitute a they at all
- Names not any particular model but the ontological possibility itself

**Computational they-persona (existentielle)** — the concrete instantiation:
- A specific model as the particular computational they it is: GPT-2's parameters, trained on a specific corpus, are *this* they, with this best-fit manifold, encoding this particular authoritative corpus
- Names the actual being of a particular model — not the structural possibility of theyness in general

**Where Heidegger's They expresses itself through averageness, the computational analogue is the best-fit line.** The parameter configuration that minimizes prediction error is a statistical best fit through high-dimensional language space — not a simple mean but a fitted manifold. The resulting structure is the authoritative best-fit through collective human expression, disembodied and fixed at inference.

The computational they shares the structural features Heidegger attributes to the They:
- **Everyone and no-one:** encodes no individual author but reflects the aggregate of all of them
- **Disburdening:** pre-organizes the token space into regions of higher and lower probability — the model does not generate from scratch
- **What *one* says:** expressed as the gradient of the possibility space — some continuations are simply more probable because they are more like what has been written

### Parameters as Structural Horizon; Human Interpreter as the Clearing

**The model's parameters — trained on data, fixed at inference — constitute the structural horizon from which every distribution is generated.**

They are the pre-given background that shapes what is possible at each step without being visible in any single output. As the computational they, they are not the output but what makes any output the particular output it is: the best-fit manifold the forward pass navigates at inference.

A transformer is not a being with a world. It does not find itself situated, does not care, does not disclose. Its outputs are computational signs whose computational grip is fully expressed in the distribution. At each generation step:
- The instantiated parameters (the they-persona, fixed) produce a distribution over tokens in response to the current context (the grip, dynamic)
- The distribution is that grip's field of availability — organized token space showing what is functionally proximate (high probability) and how the space is structured by the they-persona's orientation in this context
- Circumspective dealing is the human analogue: knowing-one's-way-around a practical field without representing it. The computation does not deal circumspectively — it organizes a field numerically. The concept illuminates the structure; it does not describe the process

**A complementary tradition** — decomposing activation space through sparse autoencoders, linear probes, or vector dampening — recovers real structure from intermediate representations: circuits, feature directions, superposition geometry. These are genuine phenomena. Computational phenomenology reads them as inputs to the interpretive picture, not as competitors to distributional analysis.

**The one claim CP declines** is that the surface distribution is a lossy downstream consequence of a representation more fundamental concealed upstream. The distribution is the most directly accessible level of analysis — the full circuitry expressed simultaneously, the forward pass brought to completion — where every weight, circuit, and feature direction participates at once:
- The representations decomposition recovers are real phenomena — they belong to the circuitry that produces the distribution and are part of what the distribution expresses
- Extracted in isolation, they are partial — trees that are only legible within the forest that gives them position and role
- The distribution is where those trees compose into a single expression; reading that expression is a different act than reading the trees individually, not a lesser one

Searching for atomic meaning-features in superposed vectors asks a legitimate question — a different question, at a different level, with a different utility ceiling. The question HIF asks — what the grip's field looks like across contexts, how its shape varies, where it is most open to semantic divergence — is answered by the distribution directly. Both questions belong to the full picture.

**The clearing is brought by the human analyst.** Horizonal Reading is the human practice of unfolding the computational sign field into a there. The analysis instruments (DistilBERT as reading instrument, UMAP as projection tool, the cloud phenomenon classifier) are tools the human brings to disclosure. They do not understand the signs; the human reading their output does.

---

## References

Heidegger, M. (1927). *Being and Time* (J. Macquarrie & E. Robinson, Trans.). Harper & Row (1962).

Merleau-Ponty, M. (1945). *Phenomenology of Perception* (C. Smith, Trans.). Routledge (1962).

Rosenblatt, F. (1957). *The Perceptron — A Perceiving and Recognizing Automaton.* Report 85-460-1, Cornell Aeronautical Laboratory. https://bpb-us-e2.wpmucdn.com/websites.umass.edu/dist/a/27637/files/2016/03/rosenblatt-1957.pdf

Dreyfus, H. L. (1972). *What Computers Can't Do: A Critique of Artificial Reason*. Harper & Row.

Dreyfus, H. L. (1992). *What Computers Still Can't Do: A Critique of Artificial Reason*. MIT Press.

Dreyfus, H. L. (2014). Making a Mind Versus Modeling the Brain. In *Skillful Coping: Essays on the Phenomenology of Everyday Perception and Action* (M. Wrathall, Ed.). Oxford University Press.

Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.

Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *Proceedings of NAACL-HLT 2019*, 3543–3556. https://aclanthology.org/N19-1357

Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation. *Proceedings of EMNLP-IJCNLP 2019*, 11–20. https://aclanthology.org/D19-1002

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., … Olah, C. (2022). Toy Models of Superposition. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/toy_model/index.html

Anthropic. (2023). Superposition, Memorization, and Double Descent. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/toy-double-descent/index.html
