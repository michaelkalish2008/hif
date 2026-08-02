"""Unit tests for hi.analysis.attention.

Uses Fake tokenizer/model classes implementing the exact call interface
AttentionAnalyzer._get_attention relies on (tokenizer(text, ...) -> dict with
input_ids; model(**inputs) -> object with .attentions) — deterministic, no
network access, no MagicMock. AttentionAnalyzer.__init__ (which would call the
real transformers.AutoTokenizer/AutoModel.from_pretrained) is bypassed by
constructing the analyzer via __new__ and assigning the fakes directly,
since __init__ is solely responsible for that real-model load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_attention(n_layers: int = 6, n_heads: int = 8, seq_len: int = 6):
    """Return a tuple of fake attention tensors (one per layer)."""
    return tuple(
        torch.softmax(torch.randn(1, n_heads, seq_len, seq_len), dim=-1)
        for _ in range(n_layers)
    )


class FakeTokenizerOutput(dict):
    """Dict-like output matching what a real HF tokenizer __call__ returns —
    supports both dict-style and .input_ids-style access if ever needed."""


class FakeTokenizer:
    """Implements the tokenizer contract AttentionAnalyzer._get_attention uses:
    __call__(text, return_tensors=..., truncation=..., max_length=...) and
    convert_ids_to_tokens(ids)."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.seq_len = len(tokens)

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        return FakeTokenizerOutput(
            input_ids=torch.tensor([list(range(self.seq_len))]),
            attention_mask=torch.ones(1, self.seq_len, dtype=torch.long),
        )

    def convert_ids_to_tokens(self, ids):
        return self._tokens


class FakeModelOutput:
    def __init__(self, attentions) -> None:
        self.attentions = attentions


class FakeTransformerModel:
    """Implements the model contract AttentionAnalyzer._get_attention uses:
    __call__(**inputs) -> object with .attentions, and eval()."""

    def __init__(self, seq_len: int) -> None:
        self._attentions = _make_fake_attention(seq_len=seq_len)

    def __call__(self, **inputs):
        return FakeModelOutput(self._attentions)

    def eval(self):
        return None


@pytest.fixture()
def mock_analyzer():
    """AttentionAnalyzer with fake tokenizer/model — no real model load."""
    from hif.analysis.attention import AttentionAnalyzer
    from hif.config import AttentionConfig

    tokens = ["[CLS]", "hello", "world", "test", "foo", "[SEP]"]

    # __init__ would load a real transformer; bypass it and assign fakes
    # directly, since __init__'s only job is that real-model load.
    analyzer = AttentionAnalyzer.__new__(AttentionAnalyzer)
    analyzer._tokenizer = FakeTokenizer(tokens)
    analyzer._model = FakeTransformerModel(seq_len=len(tokens))
    analyzer._config = AttentionConfig()
    return analyzer


# ---------------------------------------------------------------------------
# Helper: build a fake TextAttentionAnalysis profile fixture
# ---------------------------------------------------------------------------


def _make_attention_data():
    """Build minimal TextAttentionAnalysis using the new hermeneutic schema."""
    from hif.analysis.attention import (
        AttentionDelta,
        AttentionMap,
        HermeneuticComparison,
        InputAttentionAnalysis,
        TextAttentionAnalysis,
        TokenImportance,
        TokenResonance,
    )

    tokens = ["hello", "world", "test", "foo"]
    n = len(tokens)
    weights = np.eye(n).tolist()
    col_sums = np.ones(n) / n
    importance = [
        TokenImportance(token_str=t, token_idx=i, importance=float(col_sums[i]))
        for i, t in enumerate(tokens)
    ]
    attn_map = AttentionMap(
        tokens=tokens,
        weights=weights,
        token_importance=importance,
        analysis_model="distilbert-base-uncased",
        aggregate_method="mean_all_layers",
    )
    delta = AttentionDelta(
        original_token_str="hello",
        perturbed_token_str="hi",
        token_idx=0,
        importance_deltas=[0.01, -0.01, 0.0, 0.0],
        most_affected=[("hello", 0.01)],
    )
    input_analysis = InputAttentionAnalysis(
        prompt_text="hello world test foo",
        attention_map=attn_map,
        perturbation_deltas=[delta],
    )

    # Continuation attention: two tokens
    cont_tokens = ["test", "foo"]
    n_cont = len(cont_tokens)
    cont_weights = np.eye(n_cont).tolist()
    cont_importance = [
        TokenImportance(token_str=t, token_idx=i, importance=1.0 / n_cont)
        for i, t in enumerate(cont_tokens)
    ]
    cont_map = AttentionMap(
        tokens=cont_tokens,
        weights=cont_weights,
        token_importance=cont_importance,
        analysis_model="distilbert-base-uncased",
        aggregate_method="mean_all_layers",
    )

    # Hermeneutic comparison
    resonances = [
        TokenResonance(token_str="test", token_idx=0, resonance_score=0.8, anchored_to="test"),
        TokenResonance(token_str="foo", token_idx=1, resonance_score=0.1, anchored_to=""),
    ]
    comparison = HermeneuticComparison(
        prompt_attention=attn_map,
        continuation_attention=cont_map,
        token_resonance=resonances,
        mean_resonance=0.45,
        free_floating_tokens=["foo"],
        anchored_tokens=["test"],
    )

    return TextAttentionAnalysis(
        input_analysis=input_analysis,
        continuation_attention=cont_map,
        comparison=comparison,
    )


# ---------------------------------------------------------------------------
# Schema / pydantic model tests
# ---------------------------------------------------------------------------


class TestAttentionMapStructure:
    def test_fields_present(self):
        from hif.analysis.attention import AttentionMap, TokenImportance

        tokens = ["hello", "world"]
        weights = [[0.6, 0.4], [0.3, 0.7]]
        importance = [
            TokenImportance(token_str="hello", token_idx=0, importance=0.45),
            TokenImportance(token_str="world", token_idx=1, importance=0.55),
        ]
        amap = AttentionMap(
            tokens=tokens,
            weights=weights,
            token_importance=importance,
            analysis_model="distilbert-base-uncased",
            aggregate_method="mean_all_layers",
        )
        assert amap.tokens == tokens
        assert len(amap.weights) == 2
        assert len(amap.weights[0]) == 2
        assert amap.analysis_model == "distilbert-base-uncased"

    def test_weights_shape_matches_n_tokens(self):
        from hif.analysis.attention import AttentionMap, TokenImportance

        n = 5
        tokens = [f"tok{i}" for i in range(n)]
        weights = np.random.rand(n, n).tolist()
        importance = [
            TokenImportance(token_str=t, token_idx=i, importance=1.0 / n)
            for i, t in enumerate(tokens)
        ]
        amap = AttentionMap(
            tokens=tokens,
            weights=weights,
            token_importance=importance,
            analysis_model="distilbert-base-uncased",
            aggregate_method="mean_all_layers",
        )
        assert len(amap.weights) == n
        assert all(len(row) == n for row in amap.weights)


class TestTokenImportanceSumsToOne:
    def test_importance_sums_to_one(self, mock_analyzer):
        result = mock_analyzer.analyze_input("hello world test foo", [])
        total = sum(ti.importance for ti in result.attention_map.token_importance)
        assert abs(total - 1.0) < 1e-5, f"Expected sum≈1, got {total}"


# ---------------------------------------------------------------------------
# analyze_input tests
# ---------------------------------------------------------------------------


class TestInputAnalysisType:
    def test_returns_input_attention_analysis(self, mock_analyzer):
        from hif.analysis.attention import InputAttentionAnalysis

        result = mock_analyzer.analyze_input("hello world test foo", [])
        assert isinstance(result, InputAttentionAnalysis)

    def test_prompt_text_preserved(self, mock_analyzer):
        prompt = "hello world test foo"
        result = mock_analyzer.analyze_input(prompt, [])
        assert result.prompt_text == prompt

    def test_no_variants_gives_empty_deltas(self, mock_analyzer):
        result = mock_analyzer.analyze_input("hello world", [])
        assert result.perturbation_deltas == []

    def test_variants_capped_at_five(self, mock_analyzer):
        variants = [f"variant {i}" for i in range(10)]
        result = mock_analyzer.analyze_input("hello world", variants)
        assert len(result.perturbation_deltas) <= 5


# ---------------------------------------------------------------------------
# analyze_continuation tests
# ---------------------------------------------------------------------------


class TestContinuationAnalysis:
    def test_returns_attention_map(self, mock_analyzer):
        from hif.analysis.attention import AttentionMap

        result = mock_analyzer.analyze_continuation(" test foo")
        assert isinstance(result, AttentionMap)

    def test_importance_sums_to_one(self, mock_analyzer):
        result = mock_analyzer.analyze_continuation(" test foo")
        total = sum(ti.importance for ti in result.token_importance)
        assert abs(total - 1.0) < 1e-5, f"Expected sum≈1, got {total}"


# ---------------------------------------------------------------------------
# compare / HermeneuticComparison tests
# ---------------------------------------------------------------------------


class TestHermeneuticComparison:
    def _make_map(self, token_strs, importances):
        from hif.analysis.attention import AttentionMap, TokenImportance

        n = len(token_strs)
        weights = (np.eye(n) / max(n, 1)).tolist()
        imp = [
            TokenImportance(token_str=t, token_idx=i, importance=float(importances[i]))
            for i, t in enumerate(token_strs)
        ]
        return AttentionMap(
            tokens=token_strs,
            weights=weights,
            token_importance=imp,
            analysis_model="distilbert-base-uncased",
            aggregate_method="mean_all_layers",
        )

    def test_returns_hermeneutic_comparison(self, mock_analyzer):
        from hif.analysis.attention import HermeneuticComparison

        prompt_map = self._make_map(["hash", "table", "work"], [0.5, 0.3, 0.2])
        cont_map = self._make_map(["hash", "stores", "keys"], [0.4, 0.4, 0.2])
        result = mock_analyzer.compare(prompt_map, cont_map)
        assert isinstance(result, HermeneuticComparison)

    def test_resonance_perfect_match(self, mock_analyzer):
        """A continuation token that exactly matches a top prompt token scores 1.0."""
        prompt_map = self._make_map(["hash", "table"], [0.7, 0.3])
        cont_map = self._make_map(["hash"], [1.0])
        result = mock_analyzer.compare(prompt_map, cont_map)
        assert len(result.token_resonance) == 1
        assert result.token_resonance[0].resonance_score == 1.0
        assert result.token_resonance[0].anchored_to == "hash"

    def test_resonance_no_overlap(self, mock_analyzer):
        """Continuation tokens with no match to prompt tokens score 0.0."""
        prompt_map = self._make_map(["alpha", "beta"], [0.6, 0.4])
        cont_map = self._make_map(["gamma", "delta"], [0.5, 0.5])
        result = mock_analyzer.compare(prompt_map, cont_map)
        for tr in result.token_resonance:
            assert tr.resonance_score == 0.0

    def test_free_floating_and_anchored_partitioning(self, mock_analyzer):
        """free_floating = resonance < 0.2, anchored = resonance > 0.5."""
        prompt_map = self._make_map(["hash", "table", "key"], [0.5, 0.3, 0.2])
        cont_map = self._make_map(["hash", "delta"], [0.5, 0.5])
        result = mock_analyzer.compare(prompt_map, cont_map)

        # "hash" should be anchored (score == 1.0 > 0.5)
        assert "hash" in result.anchored_tokens
        # "delta" has no match — should be free-floating (score == 0.0 < 0.2)
        assert "delta" in result.free_floating_tokens

    def test_mean_resonance_in_range(self, mock_analyzer):
        prompt_map = self._make_map(["hello", "world"], [0.6, 0.4])
        cont_map = self._make_map(["hello", "there"], [0.5, 0.5])
        result = mock_analyzer.compare(prompt_map, cont_map)
        assert 0.0 <= result.mean_resonance <= 1.0

    def test_empty_continuation_map(self, mock_analyzer):
        """Empty continuation map produces 0.0 mean_resonance and empty lists."""
        prompt_map = self._make_map(["hello", "world"], [0.6, 0.4])
        cont_map = self._make_map([], [])
        result = mock_analyzer.compare(prompt_map, cont_map)
        assert result.mean_resonance == 0.0
        assert result.token_resonance == []
        assert result.free_floating_tokens == []
        assert result.anchored_tokens == []


# ---------------------------------------------------------------------------
# AttentionDelta tests
# ---------------------------------------------------------------------------


class TestAttentionDeltaStructure:
    def test_delta_has_required_fields(self):
        from hif.analysis.attention import AttentionDelta

        d = AttentionDelta(
            original_token_str="hello",
            perturbed_token_str="hi",
            token_idx=0,
            importance_deltas=[0.01, -0.01, 0.0],
            most_affected=[("hello", 0.01)],
        )
        assert d.original_token_str == "hello"
        assert d.perturbed_token_str == "hi"
        assert d.token_idx == 0
        assert isinstance(d.importance_deltas, list)
        assert isinstance(d.most_affected, list)

    def test_most_affected_is_list_of_tuples(self):
        from hif.analysis.attention import AttentionDelta

        d = AttentionDelta(
            original_token_str="a",
            perturbed_token_str="b",
            token_idx=1,
            importance_deltas=[0.05],
            most_affected=[("a", 0.05), ("b", -0.03)],
        )
        assert all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in d.most_affected)


# ---------------------------------------------------------------------------
# TextAttentionAnalysis (combined)
# ---------------------------------------------------------------------------


class TestTextAttentionAnalysis:
    def test_analyze_returns_text_attention_analysis(self, mock_analyzer):
        from hif.analysis.attention import TextAttentionAnalysis

        result = mock_analyzer.analyze("hello world test foo", " bar", [])
        assert isinstance(result, TextAttentionAnalysis)
        assert result.input_analysis is not None
        assert result.continuation_attention is not None
        assert result.comparison is not None

    def test_analyze_no_output_analysis_field(self, mock_analyzer):
        """The old output_analysis field no longer exists."""
        result = mock_analyzer.analyze("hello world test foo", " bar", [])
        assert not hasattr(result, "output_analysis")

    def test_comparison_is_hermeneutic_comparison(self, mock_analyzer):
        from hif.analysis.attention import HermeneuticComparison

        result = mock_analyzer.analyze("hello world test foo", " bar", [])
        assert isinstance(result.comparison, HermeneuticComparison)


def _make_minimal_profile_with_attention(attention_data):
    """Build a minimal BehavioralRangeProfile that includes attention data."""
    import numpy as np

    from hif.config import (
        GenerationConfig,
        ModelConfig,
        PerturbationConfig,
        RunConfig,
        TrajectoryConfig,
    )
    from hif.hourglass.center import CenterDiagnostics
    from hif.hourglass.input_side import InputSideAnalysis, PositionRecord
    from hif.hourglass.output_side import OutputSideTrace
    from hif.hourglass.trajectory import Branch, BranchConvergence, TrajectoryAnalysis
    from hif.metrics.distribution import DistributionMetrics
    from hif.metrics.semantic import SemanticMetrics
    from hif.metrics.sensitivity import SensitivityMetrics, StepSensitivity
    from hif.metrics.stability import StabilityMetrics
    from hif.models.base import StepRecord, TopKEntry
    from hif.profile.builder import generate_findings
    from hif.profile.schema import (
        BehavioralRangeProfile,
        MetricBundle,
        ModelIdentity,
        PerturbationRecord,
        PromptRecord,
    )

    def _topk(n=5):
        prob = 1.0 / n
        return [
            TopKEntry(
                token_id=i,
                token_str=f"tok{i}",
                logit=float(np.log(prob + 1e-12)),
                logprob=float(np.log(prob + 1e-12)),
                prob=prob,
            )
            for i in range(n)
        ]

    def _step(s):
        return StepRecord(step=s, selected_token_id=0, selected_token_str="tok0", topk=_topk())

    pos = PositionRecord(
        position=1,
        token_id=1,
        token_str="hello",
        surprisal=3.0,
        entropy=5.0,
        top_k_alternatives=[{"token_id": 2, "token_str": "world", "prob": 0.1}],
    )
    input_analysis = InputSideAnalysis(
        positions=[pos],
        prompt_token_ids=[0, 1],
        prompt_text="hello world",
        mean_surprisal=3.0,
        mean_entropy=5.0,
        max_entropy=16.0,
    )
    output_trace = OutputSideTrace(
        steps=[_step(i) for i in range(3)],
        input_ids=[0, 1],
        generated_ids=[0, 1, 2],
        prompt_text="hello world",
        model_name="mock-model",
        top_k=5,
        max_new_tokens=3,
        seed=42,
        mean_step_entropy=3.0,
    )
    center = CenterDiagnostics(
        input_mean_entropy=5.0,
        output_mean_entropy=3.0,
        entropy_ratio=1.0,
        goldilocks_flag="stable",
        semantic_drift=0.2,
    )
    branch = Branch(
        cluster_id=0,
        representative_token_ids=[42],
        generated_ids=[1, 2, 3],
        steps=[_step(i) for i in range(3)],
        final_text="foo bar baz",
    )
    trajectory = TrajectoryAnalysis(
        start_step=5,
        n_branches=2,
        rollout_steps=3,
        branches=[branch],
        convergence_profile=[BranchConvergence(step=i, n_remaining_clusters=2) for i in range(3)],
        persistence_score=1.0,
        explosion_score=0.0,
        convergence_score=0.0,
        initial_n_clusters=2,
    )
    sens = SensitivityMetrics(
        perturbation_generator="synonym",
        perturbed_prompt="hi world",
        original_prompt="hello world",
        step_sensitivities=[
            StepSensitivity(step=0, js_divergence=0.05, kl_divergence=0.1, entropy_delta=0.0)
        ],
        mean_js_divergence=0.05,
        mean_kl_divergence=0.1,
        mean_entropy_delta=0.0,
        output_entropy_delta=0.0,
    )
    stability = StabilityMetrics(
        input_stability=0.9,
        output_stability=0.9,
        input_output_correlation=0.0,
        n_perturbations=1,
    )
    dm = DistributionMetrics(
        entropy_bits=3.0,
        logit_margin=2.0,
        topk_cumulative_mass=0.9,
        effective_support_size=8.0,
        tail_weight=0.05,
        truncated=True,
    )
    sm = SemanticMetrics(
        cluster_count=3,
        cluster_entropy=1.0,
        mean_pairwise_distance=0.3,
        max_inter_cluster_distance=0.8,
        intra_cluster_density=0.7,
        topic_variance=0.1,
        n_candidates=5,
        truncated=True,
        cluster_labels=[0, 0, 1, 2, -1],
        embeddings_2d=[[0.1, 0.2], [0.3, 0.4], [-0.1, 0.0], [0.2, -0.3], [-0.2, 0.1]],
        projection_method="pca",
    )
    metric_bundle = MetricBundle(
        distribution=[dm, dm, dm],
        semantic=[sm, sm, sm],
        sensitivity=[sens],
        stability=stability,
    )
    findings = generate_findings(input_analysis, output_trace, center, metric_bundle)

    return BehavioralRangeProfile(
        model=ModelIdentity(name="mock-model", backend="hf", vocab_size=50257, context_length=1024),
        prompt=PromptRecord.from_text("hello world", "ordinary_conversation", 2),
        input_side=input_analysis,
        output_side=output_trace,
        center=center,
        trajectory=trajectory,
        perturbations=[
            PerturbationRecord(generator="synonym", variants=["hi world"], sensitivity=[sens])
        ],
        metrics=metric_bundle,
        findings=findings,
        config=RunConfig(
            model=ModelConfig(name="mock", backend="hf"),
            generation=GenerationConfig(max_new_tokens=3, top_k=5),
            trajectory=TrajectoryConfig(n_branches=2, rollout_steps=3),
            perturbation=PerturbationConfig(n_variants=1, generators=["synonym"]),
        ),
        attention=attention_data,
    )


# ---------------------------------------------------------------------------
# AttentionCheckpoint and AttentionTrajectory schema tests
# ---------------------------------------------------------------------------


class TestAttentionCheckpointSchema:
    def test_fields_present(self):
        from hif.analysis.attention import AttentionCheckpoint

        ck = AttentionCheckpoint(
            step=4,
            continuation_tokens=["foo", "bar"],
            prompt_token_weights=[0.3, 0.4, 0.3],
            dominant_prompt_tokens=["hello", "world", "test"],
            anchored_continuation_tokens=["foo"],
        )
        assert ck.step == 4
        assert ck.continuation_tokens == ["foo", "bar"]
        assert len(ck.prompt_token_weights) == 3
        assert ck.dominant_prompt_tokens[0] == "hello"
        assert ck.anchored_continuation_tokens == ["foo"]

    def test_empty_anchored_is_valid(self):
        from hif.analysis.attention import AttentionCheckpoint

        ck = AttentionCheckpoint(
            step=8,
            continuation_tokens=["baz"],
            prompt_token_weights=[0.5, 0.5],
            dominant_prompt_tokens=["hi"],
            anchored_continuation_tokens=[],
        )
        assert ck.anchored_continuation_tokens == []


class TestAttentionTrajectorySchema:
    def test_fields_present(self):
        from hif.analysis.attention import AttentionCheckpoint, AttentionTrajectory

        ck = AttentionCheckpoint(
            step=4,
            continuation_tokens=["foo"],
            prompt_token_weights=[0.6, 0.4],
            dominant_prompt_tokens=["hello"],
            anchored_continuation_tokens=[],
        )
        traj = AttentionTrajectory(
            checkpoints=[ck],
            prompt_tokens=["hello", "world"],
            fading_tokens=["world"],
            persistent_tokens=["hello"],
            emerging_pivots=[],
        )
        assert len(traj.checkpoints) == 1
        assert traj.prompt_tokens == ["hello", "world"]
        assert traj.fading_tokens == ["world"]
        assert traj.persistent_tokens == ["hello"]
        assert traj.emerging_pivots == []

    def test_empty_trajectory_is_valid(self):
        from hif.analysis.attention import AttentionTrajectory

        traj = AttentionTrajectory(
            checkpoints=[],
            prompt_tokens=["hello"],
            fading_tokens=[],
            persistent_tokens=[],
            emerging_pivots=[],
        )
        assert traj.checkpoints == []

    def test_text_attention_analysis_has_trajectory_field(self):
        from hif.analysis.attention import TextAttentionAnalysis

        data = _make_attention_data()
        # trajectory is None by default
        assert data.trajectory is None

    def test_text_attention_analysis_accepts_trajectory(self):
        from hif.analysis.attention import AttentionCheckpoint, AttentionTrajectory, TextAttentionAnalysis

        data = _make_attention_data()
        traj = AttentionTrajectory(
            checkpoints=[],
            prompt_tokens=["hello"],
            fading_tokens=[],
            persistent_tokens=["hello"],
            emerging_pivots=[],
        )
        updated = data.model_copy(update={"trajectory": traj})
        assert updated.trajectory is not None
        assert updated.trajectory.prompt_tokens == ["hello"]


# ---------------------------------------------------------------------------
# analyze_joint_trajectory tests
# ---------------------------------------------------------------------------


class TestJointTrajectory:
    def test_returns_attention_trajectory_type(self, mock_analyzer):
        from hif.analysis.attention import AttentionTrajectory

        result = mock_analyzer.analyze_joint_trajectory(
            "hello world test foo",
            ["generated", "tokens", "here"],
        )
        assert isinstance(result, AttentionTrajectory)

    def test_empty_continuation_returns_empty_checkpoints(self, mock_analyzer):
        result = mock_analyzer.analyze_joint_trajectory("hello world", [])
        assert result.checkpoints == []

    def test_prompt_tokens_present_in_trajectory(self, mock_analyzer):
        result = mock_analyzer.analyze_joint_trajectory(
            "hello world test foo",
            ["tok1", "tok2", "tok3", "tok4", "tok5"],
        )
        # prompt_tokens come from _get_attention on the prompt
        assert isinstance(result.prompt_tokens, list)
        assert len(result.prompt_tokens) > 0

    def test_checkpoints_capped_by_continuation_length(self, mock_analyzer):
        cont = ["a", "b", "c"]
        result = mock_analyzer.analyze_joint_trajectory("hello world", cont, interval=4)
        # Only one checkpoint: the final step (3)
        assert len(result.checkpoints) == 1
        assert result.checkpoints[0].step == len(cont)

    def test_multiple_checkpoints_at_interval(self, mock_analyzer):
        # 8 continuation tokens, interval=4 → checkpoints at 4 and 8
        cont = [f"tok{i}" for i in range(8)]
        result = mock_analyzer.analyze_joint_trajectory("hello world", cont, interval=4)
        steps = [ck.step for ck in result.checkpoints]
        assert 4 in steps
        assert 8 in steps

    def test_checkpoint_weights_sum_to_one_or_zero(self, mock_analyzer):
        cont = [f"tok{i}" for i in range(4)]
        result = mock_analyzer.analyze_joint_trajectory("hello world", cont, interval=4)
        for ck in result.checkpoints:
            total = sum(ck.prompt_token_weights)
            # weights should sum to ~1 (if cross block non-zero) or 0.0 (empty cross block)
            assert abs(total - 1.0) < 1e-6 or abs(total) < 1e-6

    def test_dominant_tokens_are_subset_of_prompt_tokens(self, mock_analyzer):
        cont = [f"tok{i}" for i in range(4)]
        result = mock_analyzer.analyze_joint_trajectory("hello world", cont, interval=4)
        prompt_set = set(result.prompt_tokens)
        for ck in result.checkpoints:
            for dom in ck.dominant_prompt_tokens:
                assert dom in prompt_set

    def test_trajectory_category_lists_are_subset_of_prompt_tokens(self, mock_analyzer):
        cont = [f"t{i}" for i in range(8)]
        result = mock_analyzer.analyze_joint_trajectory("hello world test foo", cont, interval=4)
        prompt_set = set(result.prompt_tokens)
        for tok in result.fading_tokens + result.persistent_tokens + result.emerging_pivots:
            assert tok in prompt_set


