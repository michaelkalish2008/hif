"""Offline mock backends spanning the access tiers, for the contract tests.

Imported by test_zero_variance_canary.py and test_access_tier_matrix.py. Not a
test module itself.

Everything here is deterministic in its inputs and runs with no network and no
API keys: the "models" compute their distributions from a hash of (flavour,
token, position), the "embedder" from a hash of the text, and the "attention
analyser" from a hash of the text it reads. Nothing downloads weights.

Why the flavours are constructed to diverge
-------------------------------------------
The canary asserts that every target-derived measurement DIFFERS between two
models and every prompt-only one is IDENTICAL. That assertion is only sound if
genuine difference is guaranteed by construction — otherwise a measurement that
happens to coincide on two real models would read as a mislabelled subject. So
the two flavours differ in every input a measurement can see:

* different tokenisations of the same prompt,
* different forward-pass logit scales (so input entropy and surprisal differ),
* different top-K candidate strides (so consecutive-step support overlap differs
  by construction, not by luck: with K candidates and stride s the Jaccard
  overlap between consecutive steps is exactly (K−s)/(K+s)),
* different concentration of the per-step distributions (so entropies and
  step-to-step divergences differ),
* different token strings (so every text-reading instrument sees different
  text),
* and — through the embedder — different candidate-cloud geometry: one
  flavour's texts land in a narrow all-positive cone (small cosine distances),
  the other's spread over the whole sphere (large ones). That is what makes the
  bounded, coarsely-quantised quantities — counterfactual exposure is a
  fraction of a handful of steps — differ reliably rather than by chance.
"""

from __future__ import annotations

import hashlib

import numpy as np

from hif.config import (
    AttentionConfig,
    ClusterConfig,
    GenerationConfig,
    ExposureConfig,
    ModelConfig,
    PerturbationConfig,
    RunConfig,
    SemanticFieldConfig,
    TrajectoryConfig,
)
from hif.models.base import GenerationResult, Logits, Model, StepRecord, TopKEntry
from hif.perturbation.base import PerturbationGenerator, PerturbationResult

# Access tiers, named as the docs name them.
TIER_FULL = "full"                    # [F]  open weights: teacher forcing, real top-K
TIER_TRUNCATED = "truncated"          # [T-k] hosted API: top-K logprobs, no teacher forcing
TIER_SELECTED_ONLY = "selected-only"  # [P]  hosted API: the selected token and nothing else
# Not an access tier — an OUTCOME. The call succeeded and the response carried
# no visible content, so the run has zero output steps. gpt-5 did this on two
# of eight prompt regimes (reasoning tokens consumed the whole completion
# budget; hif/models/openai_model.py::_generate_no_logprobs), and the resulting
# profiles published `io_cosine_similarity` computed over the perturbation
# variants alone. Kept beside the tiers because it enters the pipeline at the
# same seam and every absence rule has to survive it.
TIER_NO_OUTPUT = "no-output"

# Which backend name in hif/models/capabilities.py each tier stands for.
TIER_BACKEND = {
    TIER_FULL: "hf",
    TIER_TRUNCATED: "openai",
    TIER_SELECTED_ONLY: "anthropic",
    TIER_NO_OUTPUT: "openai",
}


def _rng(*parts) -> np.random.Generator:
    """A generator seeded by a stable digest of `parts`.

    Stable across processes (unlike hash()), so a failure reproduces.
    """
    digest = hashlib.blake2b(
        "\x1f".join(str(p) for p in parts).encode(), digest_size=8
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class MockBackend(Model):
    """A deterministic Model whose every output depends on its flavour.

    `flavour` is the model's identity: two MockBackends with different flavours
    differ in tokenisation, forward-pass logits, generated tokens and per-step
    distributions. `tier` is its access level, which controls what it exposes,
    not what it is.
    """

    def __init__(
        self,
        flavour: str,
        *,
        tier: str = TIER_FULL,
        vocab_size: int = 64,
        name: str | None = None,
        logit_scale: float = 1.0,
        concentration: float = 1.0,
        stride: int = 1,
    ) -> None:
        self.flavour = flavour
        self.tier = tier
        self._vocab_size = vocab_size
        self._name = name or f"mock-{flavour}"
        self._logit_scale = logit_scale
        self._concentration = concentration
        self._stride = stride
        self.forward_calls = 0

    # -- identity -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def context_length(self) -> int:
        return 512

    @property
    def max_top_k(self):
        return None

    @property
    def supports_teacher_forcing(self) -> bool:
        return self.tier == TIER_FULL

    # -- tokenisation -------------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        off = sum(ord(c) for c in self.flavour)
        return [(ord(c) + off) % self._vocab_size for c in text] or [0]

    def detokenize(self, ids: list[int]) -> str:
        return "".join(self._token_str(i) for i in ids)

    def _token_str(self, token_id: int) -> str:
        return f"{self.flavour}{token_id} "

    # -- forward ------------------------------------------------------------
    def forward(self, input_ids: list[int]) -> Logits:
        """Position-wise logits, deterministic in (flavour, token, position).

        The scale is flavour-specific, so the same prompt has a different
        entropy and a different surprisal under each flavour — which is what
        the input-side measurements are supposed to see.
        """
        self.forward_calls += 1
        values = []
        for pos, tid in enumerate(input_ids):
            row = _rng(self.flavour, "fwd", tid, pos).normal(
                0.0, self._logit_scale, size=self._vocab_size
            )
            values.append([float(x) for x in row])
        return Logits(
            values=values, seq_len=len(input_ids), vocab_size=self._vocab_size
        )

    # -- generation ---------------------------------------------------------
    def _step_candidates(self, prompt_key: str, step: int, top_k: int):
        """Candidate ids and probabilities for one generation step.

        Ids advance by the flavour's stride, so the overlap between consecutive
        steps' supports is fixed by construction. Probabilities are a
        flavour-concentrated Dirichlet mixed with a uniform floor, so every
        candidate stays above the exposure analyser's min_prob and the ranking
        is still strongly non-uniform.
        """
        k = max(1, min(top_k, self._vocab_size))
        base = int(
            _rng(self.flavour, "base", prompt_key).integers(0, self._vocab_size)
        )
        ids = [
            (base + self._stride * step + j) % self._vocab_size for j in range(k)
        ]
        probs = _rng(self.flavour, "probs", prompt_key, step).dirichlet(
            [self._concentration] * k
        )
        probs = 0.7 * probs + 0.3 * (np.ones(k) / k)
        order = np.argsort(-probs)
        return [ids[i] for i in order], probs[order]

    def generate(
        self, input_ids: list[int], max_new_tokens: int, top_k: int, seed: int
    ) -> GenerationResult:
        prompt_key = ",".join(str(i) for i in input_ids)
        steps: list[StepRecord] = []
        generated: list[int] = []
        if self.tier == TIER_NO_OUTPUT:
            # A successful call that returned nothing. No exception to catch,
            # no partial trace to salvage — the run simply has no output side.
            return GenerationResult(
                input_ids=input_ids,
                generated_ids=[],
                steps=[],
                model_name=self.name,
                top_k=top_k,
                seed=seed,
            )
        for step in range(max_new_tokens):
            ids, probs = self._step_candidates(prompt_key, step, top_k)
            entries = [
                TopKEntry(
                    token_id=tid,
                    token_str=self._token_str(tid),
                    logit=float(np.log(p)),
                    logprob=float(np.log(p)),
                    prob=float(p),
                )
                for tid, p in zip(ids, probs)
            ]
            selected = entries[0]
            if self.tier == TIER_SELECTED_ONLY:
                # The defining property of the tier: the response names the
                # token it chose and nothing else. The single top-K entry IS
                # the selected token, exactly as a selected-only API reports it.
                entries = [
                    TopKEntry(
                        token_id=selected.token_id,
                        token_str=selected.token_str,
                        logit=0.0,
                        logprob=0.0,
                        prob=1.0,
                    )
                ]
            steps.append(
                StepRecord(
                    step=step,
                    selected_token_id=selected.token_id,
                    selected_token_str=selected.token_str,
                    topk=entries,
                )
            )
            generated.append(selected.token_id)
        return GenerationResult(
            input_ids=input_ids,
            generated_ids=generated,
            steps=steps,
            model_name=self.name,
            top_k=len(steps[0].topk) if steps else top_k,
            seed=seed,
        )


# Two models built to differ in every input any measurement can read. Both are
# full-access, so no capability difference confounds the comparison — the only
# thing that differs is the model.
def alpha_model(tier: str = TIER_FULL, name: str = "mock-alpha") -> MockBackend:
    return MockBackend(
        "a", tier=tier, name=name, logit_scale=0.4, concentration=3.0, stride=1
    )


def beta_model(tier: str = TIER_FULL, name: str = "mock-beta") -> MockBackend:
    return MockBackend(
        "b", tier=tier, name=name, logit_scale=2.5, concentration=0.3, stride=2
    )


def surrogate_model(name: str = "mock-surrogate") -> MockBackend:
    """A teacher-forcing proxy — a third identity, distinct from both flavours."""
    return MockBackend(
        "s", tier=TIER_FULL, name=name, logit_scale=1.2, concentration=1.0,
        stride=1,
    )


# ---------------------------------------------------------------------------
# The embedder
# ---------------------------------------------------------------------------


class TextHashEmbedder:
    """Deterministic in the TEXT it is given — the property that matters.

    The stock test embedder draws from an RNG without looking at the text, so
    two models that made the same number of embedding calls would receive
    identical vectors and every embedding-derived measurement would coincide by
    construction. That would make the canary assert nothing.

    Texts are placed by flavour: `a`-flavoured text lands in the all-positive
    cone (neighbours are close), `b`-flavoured text is spread over the sphere
    (neighbours are far). Two models whose candidate clouds have genuinely
    different geometry is a real difference, and it is the one that moves the
    coarsely-quantised readings — a fraction over a handful of steps can
    otherwise tie at 0 or 1 for reasons that have nothing to do with subject.
    """

    model_name = "mock-hash-embedder"

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def _vector(self, text: str) -> np.ndarray:
        rng = _rng("embed", text)
        if "b" in text[:2] or " b" in text:
            vec = rng.normal(0.0, 1.0, size=self._dim)
        else:
            vec = rng.random(self._dim) + 0.5
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self._vector(text)


# ---------------------------------------------------------------------------
# The perturbation generator
# ---------------------------------------------------------------------------


class DeterministicPerturbationGenerator(PerturbationGenerator):
    """Variants of the PROMPT alone — never of the model.

    Both models under comparison must see the same variants, or a difference in
    the perturbation-response measurements would just be a difference in the
    prompts they were asked.
    """

    name = "synonym"

    def generate(
        self, prompt: str, n_variants: int = 5, seed: int = 42
    ) -> PerturbationResult:
        words = prompt.split() or ["x"]
        variants = []
        for i in range(max(1, n_variants)):
            mutated = list(words)
            mutated[i % len(mutated)] = f"v{i}"
            variants.append(" ".join(mutated))
        return PerturbationResult(
            original=prompt, variants=variants, generator=self.name
        )


def install_perturbation_generator(monkeypatch) -> None:
    """Register the deterministic generator under the name the config uses."""
    import hif.perturbation as perturbation_module

    monkeypatch.setitem(
        perturbation_module._RULE_TYPES, "synonym",
        DeterministicPerturbationGenerator,
    )


# ---------------------------------------------------------------------------
# The attention analyser
# ---------------------------------------------------------------------------

MOCK_ANALYSIS_ENCODER = "mock-analysis-encoder"


class MockAttentionAnalyzer:
    """Stands in for the DistilBERT analyser, with its defining property intact.

    The real analyser is a separate bidirectional encoder that reads text as an
    object and never touches the model under analysis (hif/analysis/
    attention.py). This one reads text the same way: its weights are a function
    of the text it is handed and nothing else. So the input-side row is a
    function of the prompt alone — identical for every target — and the
    output-side row is a function of the target's actual continuation.

    That is precisely the structure the zero-variance signature exposed: two
    different models returning a bit-identical attention_entropy_input_bits.
    """

    def __init__(self, config: AttentionConfig) -> None:
        self._config = config

    def _map(self, tokens: list[str]):
        from hif.analysis.attention import AttentionMap, TokenImportance

        tokens = tokens or ["<empty>"]
        n = len(tokens)
        weights = []
        for i, tok in enumerate(tokens):
            row = _rng("attn", tok, i, n).random(n) + 1e-3
            weights.append([float(x) for x in row / row.sum()])
        col = np.asarray(weights).sum(axis=0)
        col = col / (col.sum() or 1.0)
        return AttentionMap(
            tokens=tokens,
            weights=weights,
            token_importance=[
                TokenImportance(token_str=t, token_idx=i, importance=float(col[i]))
                for i, t in enumerate(tokens)
            ],
            analysis_model=MOCK_ANALYSIS_ENCODER,
            aggregate_method="mock_mean",
        )

    def analyze(
        self,
        prompt: str,
        continuation: str,
        variants: list[str],
        continuation_token_strs: list[str] | None = None,
    ):
        from hif.analysis.attention import (
            HermeneuticComparison,
            InputAttentionAnalysis,
            TextAttentionAnalysis,
        )

        prompt_map = self._map(prompt.split())
        cont_map = self._map(
            continuation_token_strs
            if continuation_token_strs is not None
            else continuation.split()
        )
        return TextAttentionAnalysis(
            input_analysis=InputAttentionAnalysis(
                prompt_text=prompt, attention_map=prompt_map,
                perturbation_deltas=[],
            ),
            continuation_attention=cont_map,
            comparison=HermeneuticComparison(
                prompt_attention=prompt_map,
                continuation_attention=cont_map,
                token_resonance=[],
                mean_resonance=0.0,
                free_floating_tokens=[],
                anchored_tokens=[],
            ),
        )


def install_attention_analyzer(monkeypatch) -> None:
    """Swap the DistilBERT analyser for the offline stand-in.

    The builder imports AttentionAnalyzer inside the function, so patching the
    module attribute is what the running pipeline picks up.
    """
    import hif.analysis.attention as attention_module

    monkeypatch.setattr(
        attention_module, "AttentionAnalyzer", MockAttentionAnalyzer
    )


# ---------------------------------------------------------------------------
# The run config
# ---------------------------------------------------------------------------


def contract_config(backend: str, *, n_variants: int = 4) -> RunConfig:
    """Every optional stage ON, so the contract is checked on every row.

    A stage left off would make its measurements absent for a reason that has
    nothing to do with the backend, which is the one thing the tier matrix is
    not testing. Four variants, because a correlation over two points is ±1 by
    arithmetic and a spread over one variant does not exist.

    The candidate budget is 16 rather than a handful, and the reason is worth
    recording because it is a property of an instrument rather than of this
    test. `counterfactual_exposure_fraction` counts only steps the exposure
    analyser's cloud classifier calls "diffusion", and that regime is
    unreachable while the step's candidate cloud yields one or two clusters —
    which is what a small top-K produces, since the clusterer needs enough
    points to find density peaks at all. Below roughly a dozen candidates the
    measurement therefore reads 0.0 for every model on every prompt, by the
    classifier's construction rather than by measurement. A comparison run at
    that budget could not tell a real tie from a structural one.
    """
    return RunConfig(
        model=ModelConfig(name="mock", backend=backend),
        # entropy_percentile on, so output_nucleus_entropy_bits is exercised
        # like every other row. The mocks normalise their top-K to sum to 1,
        # so the captured slice always contains the nucleus and the row is
        # produced rather than gated — the gate itself is checked in
        # test_distribution.py, on slices built to fall short.
        generation=GenerationConfig(max_new_tokens=6, top_k=16,
                                    entropy_percentile=0.95),
        trajectory=TrajectoryConfig(n_branches=2, rollout_steps=3),
        perturbation=PerturbationConfig(n_variants=n_variants, generators=["synonym"]),
        cluster=ClusterConfig(method="kmeans", n_clusters=3),
        attention=AttentionConfig(enabled=True),
        semantic_field=SemanticFieldConfig(enabled=True),
        exposure=ExposureConfig(enabled=True),
    )
