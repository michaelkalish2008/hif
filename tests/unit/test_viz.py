"""Unit tests for signal-visualization engine (hif/viz/)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hif.config import (
    ClusterConfig,
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
from hif.models.base import StepRecord, TopKEntry
from hif.profile.builder import generate_findings
from hif.profile.schema import (
    BehavioralRangeProfile,
    Findings,
    MetricBundle,
    ModelIdentity,
    PerturbationRecord,
    PromptRecord,
)
from hif.viz import (
    AGGREGATE_SIGNALS,
    READING_SIGNALS,
    SIGNALS,
    SIGNALS_BY_ID,
    generate_signal_plots,
)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_plots.py helpers)
# ---------------------------------------------------------------------------


def _make_topk(n: int = 5) -> list[TopKEntry]:
    entries = []
    for i in range(n):
        prob = 1.0 / n
        entries.append(
            TopKEntry(
                token_id=i,
                token_str=f"tok{i}",
                logit=float(np.log(prob + 1e-12)),
                logprob=float(np.log(prob + 1e-12)),
                prob=prob,
            )
        )
    return entries


def _make_step(step: int, n_tokens: int = 5) -> StepRecord:
    topk = _make_topk(n_tokens)
    return StepRecord(
        step=step,
        selected_token_id=0,
        selected_token_str="tok0",
        topk=topk,
    )


def _make_input_analysis() -> InputSideAnalysis:
    pos = PositionRecord(
        position=1,
        token_id=1,
        token_str="hello",
        surprisal=3.0,
        entropy=5.0,
        top_k_alternatives=[{"token_id": 2, "token_str": "world", "prob": 0.1}],
    )
    return InputSideAnalysis(
        positions=[pos, pos],  # 2 positions for visual interest
        prompt_token_ids=[0, 1],
        prompt_text="hello world",
        mean_surprisal=3.0,
        mean_entropy=5.0,
        max_entropy=16.0,
        volatility_score=0.5,
    )


def _make_output_trace(n_steps: int = 3) -> OutputSideTrace:
    steps = [_make_step(i) for i in range(n_steps)]
    return OutputSideTrace(
        steps=steps,
        input_ids=[0, 1],
        generated_ids=list(range(n_steps)),
        prompt_text="hello world",
        model_name="mock-model",
        top_k=5,
        max_new_tokens=n_steps,
        seed=42,
        mean_step_entropy=3.0,
    )


def _make_center() -> CenterDiagnostics:
    return CenterDiagnostics(
        input_mean_entropy=5.0,
        output_mean_entropy=3.0,
        entropy_ratio=1.0,
        prompt_output_cosine_distance=0.2,
    )


def _make_distribution_metrics() -> DistributionMetrics:
    return DistributionMetrics(
        entropy_bits=3.0,
        logit_margin=2.0,
        topk_cumulative_mass=0.9,
        effective_support_size=8.0,
        tail_weight=0.05,
        truncated=True,
        nucleus_fraction={"p90": 0.002, "p95": 0.004},
        nucleus_entropy_bits=2.5,
    )


def _make_semantic_metrics() -> SemanticMetrics:
    return SemanticMetrics(
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


def _make_sensitivity() -> SensitivityMetrics:
    return SensitivityMetrics(
        perturbation_generator="synonym",
        perturbed_prompt="hi world",
        original_prompt="hello world",
        step_sensitivities=[
            StepSensitivity(
                step=0,
                js_divergence=0.05,
                kl_divergence=0.1,
                entropy_delta=0.0,
                nucleus_overlap_p90=1.0,
            )
        ],
        mean_js_divergence=0.05,
        mean_kl_divergence=0.1,
        mean_entropy_delta=0.0,
        output_entropy_delta=0.0,
        mean_nucleus_stability_p90=1.0,
    )


def _make_trajectory() -> TrajectoryAnalysis:
    branch = Branch(
        cluster_id=0,
        representative_token_ids=[42],
        generated_ids=[1, 2, 3],
        steps=[_make_step(i) for i in range(3)],
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


@pytest.fixture()
def minimal_profile() -> BehavioralRangeProfile:
    """Return a minimal but valid BehavioralRangeProfile built from synthetic data.

    This profile has:
    - input_side positions (for the perturbation-response signals)
    - distribution, semantic, sensitivity, trajectory metrics
    - perturbation-response metrics
    - NO similarity (metrics.similarity is None)
    - NO exposure or attention metrics

    Available signals: stability, breadth, surprise, io_correlation, sensitivity,
    continuity, entropy, shift, wager
    Unavailable: similarity, spread, horizon, exposure
    """
    input_analysis = _make_input_analysis()
    output_trace = _make_output_trace()
    center = _make_center()
    trajectory = _make_trajectory()
    sens = _make_sensitivity()
    stability = PerturbationResponse(
        input_entropy_shift_bits=0.4,
        perturbation_jsd_bits=0.1,
        input_output_correlation=0.0,
        n_perturbations=1,
    )
    metric_bundle = MetricBundle(
        distribution=[_make_distribution_metrics(), _make_distribution_metrics(), _make_distribution_metrics()],
        semantic=[_make_semantic_metrics(), _make_semantic_metrics(), _make_semantic_metrics()],
        sensitivity=[sens],
        stability=stability,
        similarity=None,  # Explicitly no similarity data
    )
    findings = generate_findings(input_analysis, output_trace, center, metric_bundle)

    return BehavioralRangeProfile(
        model=ModelIdentity(
            name="mock-model",
            backend="hf",
            vocab_size=50257,
            context_length=1024,
        ),
        prompt=PromptRecord.from_text("hello world", "ordinary_conversation", 2),
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


# ---------------------------------------------------------------------------
# Signal engine tests
# ---------------------------------------------------------------------------


class TestSignalEngine:
    """Test the signal-visualization engine (hif.viz.generate_signal_plots)."""

    def test_every_registered_signal_renders_plus_index(self, minimal_profile, tmp_path):
        """Rendering is gated on data availability ONLY. Every signal in the
        registry is drawn on every run, plus the combined index."""
        results = generate_signal_plots(minimal_profile, tmp_path)

        assert set(results) == set(SIGNALS_BY_ID) | {"index"}
        assert len(results) == len(SIGNALS) + 1
        # Aggregates and readings partition the registry — no third bucket.
        assert set(AGGREGATE_SIGNALS) | set(READING_SIGNALS) == set(SIGNALS_BY_ID)
        assert not set(AGGREGATE_SIGNALS) & set(READING_SIGNALS)

    def test_generate_takes_no_tier_argument(self, minimal_profile, tmp_path):
        """The free/premium split was a product-tier concept, not a property of
        the instrument. generate_signal_plots must not accept a tier again."""
        import inspect

        params = inspect.signature(generate_signal_plots).parameters
        assert "tier" not in params
        with pytest.raises(TypeError):
            generate_signal_plots(minimal_profile, tmp_path, tier="free")

    def test_only_signal_renders_exactly_one_chart(self, minimal_profile, tmp_path):
        results = generate_signal_plots(minimal_profile, tmp_path, only_signal="breadth")
        assert set(results) == {"breadth"}  # one chart, no dashboard
        assert results["breadth"]["html"].exists()

    def test_only_signal_unknown_id_raises(self, minimal_profile, tmp_path):
        with pytest.raises(ValueError, match="Unknown signal"):
            generate_signal_plots(minimal_profile, tmp_path, only_signal="nope")

    def test_every_result_html_exists_and_nonempty(self, minimal_profile, tmp_path):
        """Every returned HTML file exists and has size > 0 (even unavailable signals)."""
        results = generate_signal_plots(minimal_profile, tmp_path)

        for sig_id, formats_dict in results.items():
            assert "html" in formats_dict, f"Signal '{sig_id}' missing 'html' key"
            html_path = formats_dict["html"]
            assert isinstance(html_path, Path), f"HTML path for '{sig_id}' is not a Path"
            assert html_path.exists(), f"HTML file for '{sig_id}' does not exist: {html_path}"
            assert html_path.stat().st_size > 0, f"HTML file for '{sig_id}' is empty: {html_path}"

    def test_unavailable_signals_still_render(self, minimal_profile, tmp_path):
        """Unavailable signals (similarity, spread, horizon, exposure) still generate valid HTML."""
        results = generate_signal_plots(minimal_profile, tmp_path)

        # These signals should be unavailable in minimal_profile (no similarity data, etc.)
        unavailable_ids = ["similarity", "spread", "horizon", "exposure"]

        for sig_id in unavailable_ids:
            # Signal should still be in results with a valid HTML file
            assert sig_id in results, f"Unavailable signal '{sig_id}' missing from results"
            assert "html" in results[sig_id], f"Signal '{sig_id}' missing 'html' key"
            html_path = results[sig_id]["html"]
            assert html_path.exists(), f"HTML for unavailable signal '{sig_id}' does not exist"
            assert html_path.stat().st_size > 0, f"HTML for unavailable signal '{sig_id}' is empty"

            # Verify the signal's available() method returns non-None for this profile
            sig_obj = next((s for s in SIGNALS if s.id == sig_id), None)
            assert sig_obj is not None, f"Signal '{sig_id}' not found in SIGNALS registry"
            reason = sig_obj.available(minimal_profile)
            assert reason is not None, f"Signal '{sig_id}' should be unavailable for minimal_profile"

    def test_available_signals_report_none(self, minimal_profile):
        """Available signals (breadth, entropy, shift, stability, sensitivity) report None from available()."""
        available_ids = ["breadth", "entropy", "shift", "stability", "sensitivity"]

        for sig_id in available_ids:
            sig_obj = next((s for s in SIGNALS if s.id == sig_id), None)
            assert sig_obj is not None, f"Signal '{sig_id}' not found in SIGNALS registry"
            reason = sig_obj.available(minimal_profile)
            assert reason is None, (
                f"Signal '{sig_id}' should be available for minimal_profile, "
                f"but got reason: {reason}"
            )

    def test_index_groups_sections_and_states_the_boundary(self, minimal_profile, tmp_path):
        """The index groups by kind and states what a single snapshot is not."""
        results = generate_signal_plots(minimal_profile, tmp_path)

        index_html = results["index"]["html"].read_text(encoding="utf-8")

        assert "Aggregate views" in index_html
        assert "Per-step views" in index_html
        assert "single behavioral snapshot" in index_html
        assert "not</strong> detect drift" in index_html
        # No thresholds, levels, or verdicts anywhere on the dashboard.
        assert "no thresholds, levels, or verdicts" in index_html

    def test_index_embeds_every_signal_and_sells_nothing(self, minimal_profile, tmp_path):
        """Every signal's chart is embedded — there are no locked cards and no
        upsell, because the free/premium split is gone."""
        results = generate_signal_plots(minimal_profile, tmp_path)

        index_html = results["index"]["html"].read_text(encoding="utf-8")

        for sig_id in SIGNALS_BY_ID:
            assert f'src="{sig_id}.html"' in index_html, (
                f"signal '{sig_id}' is not embedded in the index"
            )
        for upsell in ("Premium", "premium", "Upgrade", "locked", "See the readings live"):
            assert upsell not in index_html, f"upsell copy leaked into the index: {upsell!r}"

    def test_availability_badge_reflects_real_data(self, minimal_profile, tmp_path):
        """Availability, not entitlement, decides how a card is labeled."""
        results = generate_signal_plots(minimal_profile, tmp_path)

        index_html = results["index"]["html"].read_text(encoding="utf-8")
        # minimal_profile has real perturbation data but no similarity/attention,
        # so the dashboard must show both badge states honestly.
        assert "live data" in index_html
        assert "not available this run" in index_html
