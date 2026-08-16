"""Shared synthetic data builders for profile unit tests.

Imported by test_profile_*.py files. Not a test module itself.
"""

from __future__ import annotations

import numpy as np

from hif.config import (
    ClusterConfig,
    EmbeddingConfig,
    GenerationConfig,
    ModelConfig,
    OutputConfig,
    PerturbationConfig,
    RunConfig,
    TrajectoryConfig,
)
from hif.hourglass.center import CenterDiagnostics
from hif.hourglass.input_side import InputSideAnalysis, PositionRecord
from hif.hourglass.output_side import OutputSideTrace
from hif.hourglass.trajectory import (
    Branch,
    BranchConvergence,
    TrajectoryAnalysis,
)
from hif.metrics.distribution import DistributionMetrics
from hif.metrics.semantic import SemanticMetrics
from hif.metrics.sensitivity import SensitivityMetrics, StepSensitivity
from hif.metrics.stability import PerturbationResponse
from hif.models.base import GenerationResult, Logits, Model, StepRecord, TopKEntry
from hif.profile.builder import generate_findings
from hif.profile.schema import (
    BehavioralRangeProfile,
    Findings,
    MetricBundle,
    ModelIdentity,
    PromptRecord,
)


def _make_topk(token_probs: dict[int, float]) -> list[TopKEntry]:
    entries = []
    for tid, prob in token_probs.items():
        logit = float(np.log(prob + 1e-12))
        entries.append(
            TopKEntry(
                token_id=tid,
                token_str=f"tok{tid}",
                logit=logit,
                logprob=logit,
                prob=prob,
            )
        )
    return entries


def _make_step(step: int, token_probs: dict[int, float]) -> StepRecord:
    selected_tid = max(token_probs, key=lambda t: token_probs[t])
    return StepRecord(
        step=step,
        selected_token_id=selected_tid,
        selected_token_str=f"tok{selected_tid}",
        topk=_make_topk(token_probs),
    )


def _make_input_analysis(
    mean_entropy: float = 5.0,
    max_entropy: float = 16.0,
    surprisal: float = 3.0,
    pos_entropy: float | None = None,
) -> InputSideAnalysis:
    # pos_entropy controls the per-position entropy used for the Surprise
    # excess max(0, surprisal - entropy); defaults to mean_entropy.
    pe = mean_entropy if pos_entropy is None else pos_entropy
    pos = PositionRecord(
        position=1,
        token_id=1,
        token_str="hello",
        surprisal=surprisal,
        entropy=pe,
        top_k_alternatives=[{"token_id": 2, "token_str": "world", "prob": 0.1}],
    )
    return InputSideAnalysis(
        positions=[pos],
        prompt_token_ids=[0, 1],
        prompt_text="hello world",
        mean_surprisal=surprisal,
        mean_entropy=mean_entropy,
        max_entropy=max_entropy,
    )


def _make_output_trace(n_steps: int = 3, mean_entropy: float = 3.0) -> OutputSideTrace:
    steps = []
    for i in range(n_steps):
        probs = {j: 1.0 / 5 for j in range(5)}
        steps.append(_make_step(i, probs))
    generated_ids = list(range(n_steps))
    return OutputSideTrace(
        steps=steps,
        input_ids=[0, 1],
        generated_ids=generated_ids,
        prompt_text="hello world",
        model_name="mock-model",
        top_k=5,
        max_new_tokens=n_steps,
        seed=42,
        mean_step_entropy=mean_entropy,
    )


def _make_center(prompt_output_cosine_distance: float = 0.2) -> CenterDiagnostics:
    return CenterDiagnostics(
        input_mean_entropy=5.0,
        output_mean_entropy=5.0,
        entropy_ratio=1.0,
        prompt_output_cosine_distance=prompt_output_cosine_distance,
    )


def _make_sensitivity(mean_js: float = 0.05) -> SensitivityMetrics:
    return SensitivityMetrics(
        perturbation_generator="synonym",
        perturbed_prompt="hi world",
        original_prompt="hello world",
        step_sensitivities=[
            StepSensitivity(
                step=0,
                js_divergence=mean_js,
                kl_divergence=0.1,
                entropy_delta=0.0,
                nucleus_overlap_p90=1.0,
            )
        ],
        mean_js_divergence=mean_js,
        mean_kl_divergence=0.1,
        mean_entropy_delta=0.0,
        output_entropy_delta=0.0,
        mean_nucleus_stability_p90=1.0,
    )


def _make_stability() -> PerturbationResponse:
    return PerturbationResponse(
        input_entropy_shift_bits=0.4,
        perturbation_jsd_bits=0.1,
        input_output_correlation=0.1,
        n_perturbations=1,
    )


def _make_distribution_metrics() -> DistributionMetrics:
    return DistributionMetrics(
        entropy_bits=3.0,
        logit_margin=2.0,
        topk_cumulative_mass=0.9,
        nucleus_effective_support_size=8.0,
        tail_weight=0.05,
        truncated=True,
        nucleus_fraction={"p90": 0.002, "p95": 0.004},
        nucleus_entropy_bits=2.5,
    )


def _make_semantic_metrics(cluster_entropy: float = 1.0) -> SemanticMetrics:
    return SemanticMetrics(
        cluster_count=3,
        cluster_entropy=cluster_entropy,
        mean_pairwise_distance=0.3,
        max_inter_cluster_distance=0.8,
        intra_cluster_density=0.7,
        topic_variance=0.1,
        n_candidates=5,
        truncated=True,
    )


def _make_trajectory() -> TrajectoryAnalysis:
    branch = Branch(
        cluster_id=0,
        representative_token_ids=[42],
        generated_ids=[1, 2, 3],
        steps=[_make_step(i, {j: 0.2 for j in range(5)}) for i in range(3)],
        final_text="foo bar baz",
    )
    return TrajectoryAnalysis(
        start_step=5,
        n_branches=2,
        rollout_steps=3,
        branches=[branch],
        convergence_profile=[
            BranchConvergence(step=i, n_remaining_clusters=2) for i in range(3)
        ],
        persistence_score=1.0,
        explosion_score=0.0,
        convergence_score=0.0,
        initial_n_clusters=2,
    )


def _make_run_config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(name="mock", backend="hf"),
        generation=GenerationConfig(max_new_tokens=3, top_k=5),
        trajectory=TrajectoryConfig(n_branches=2, rollout_steps=3),
        perturbation=PerturbationConfig(n_variants=1, generators=["synonym"]),
    )


def _make_profile(
    cluster_entropy: float = 1.0,
    mean_js: float = 0.05,
    perturbation_jsd_bits: float = 0.1,
    prompt_output_cosine_distance: float = 0.2,
) -> BehavioralRangeProfile:
    """Build a minimal but valid BehavioralRangeProfile with synthetic data."""
    from hif.profile.schema import PerturbationRecord

    input_analysis = _make_input_analysis()
    output_trace = _make_output_trace()
    center = _make_center(
        prompt_output_cosine_distance=prompt_output_cosine_distance
    )
    trajectory = _make_trajectory()
    sens = _make_sensitivity(mean_js=mean_js)
    stability = PerturbationResponse(
        input_entropy_shift_bits=0.4,
        perturbation_jsd_bits=perturbation_jsd_bits,
        input_output_correlation=0.0,
        n_perturbations=1,
    )
    metric_bundle = MetricBundle(
        distribution=[_make_distribution_metrics()],
        semantic=[_make_semantic_metrics(cluster_entropy=cluster_entropy)],
        sensitivity=[sens],
        stability=stability,
    )
    findings = generate_findings(input_analysis, output_trace, center, metric_bundle)

    return BehavioralRangeProfile(
        model=ModelIdentity(
            name="mock-model",
            backend="hf",
            vocab_size=50257,
            context_length=1024,
        ),
        prompt=PromptRecord.from_text("hello world", "factual", 2),
        input_side=input_analysis,
        output_side=output_trace,
        center=center,
        trajectory=trajectory,
        perturbations=[
            PerturbationRecord(
                generator="synonym",
                variants=["hi world"],
                sensitivity=[sens],
            )
        ],
        metrics=metric_bundle,
        findings=findings,
        config=_make_run_config(),
    )


class FakeModel(Model):
    """Implements the real Model ABC contract with deterministic logic — a
    genuine subclass, so Python's abstractmethod machinery enforces interface
    compliance the way MagicMock never could."""

    def __init__(
        self,
        vocab_size: int = 50,
        context_length: int = 512,
        name: str = "fake-model",
        supports_teacher_forcing: bool = True,
    ) -> None:
        self._vocab_size = vocab_size
        self._context_length = context_length
        self._name = name
        self._supports_teacher_forcing = supports_teacher_forcing
        self.forward_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def max_top_k(self):
        return None

    @property
    def supports_teacher_forcing(self) -> bool:
        return self._supports_teacher_forcing

    @supports_teacher_forcing.setter
    def supports_teacher_forcing(self, value: bool) -> None:
        self._supports_teacher_forcing = value

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) % self._vocab_size for c in text] or [0]

    def detokenize(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def forward(self, input_ids: list[int]) -> Logits:
        self.forward_calls += 1
        seq_len = len(input_ids)
        vals = [[0.0] * self._vocab_size for _ in range(seq_len)]
        return Logits(values=vals, seq_len=seq_len, vocab_size=self._vocab_size)

    def generate(self, input_ids: list[int], max_new_tokens: int, top_k: int, seed: int) -> GenerationResult:
        generated = list(range(max_new_tokens))
        steps = [
            StepRecord(
                step=i,
                selected_token_id=i,
                selected_token_str=str(i),
                topk=[
                    TopKEntry(
                        token_id=j, token_str=str(j), logit=0.0,
                        logprob=float(np.log(1.0 / top_k)), prob=1.0 / top_k,
                    )
                    for j in range(top_k)
                ],
            )
            for i in range(max_new_tokens)
        ]
        return GenerationResult(
            input_ids=input_ids, generated_ids=generated, steps=steps,
            model_name=self.name, top_k=top_k, seed=seed,
        )


class FakeEmbeddingModel:
    """Implements EmbeddingModel's real contract (embed(texts) -> np.ndarray
    and embed_single(text) -> np.ndarray) with deterministic unit vectors."""

    def __init__(self, dim: int = 16, seed: int = 1) -> None:
        self._dim = dim
        self._rng = np.random.default_rng(seed)

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = self._rng.random((len(texts), self._dim)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        return (arr / norms).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
