"""Unit tests for generate_findings.

This module used to assert three levels (low/medium/high) for each of six
dimensions plus a one-sentence verdict. All of that is gone: assigning a level
is an inference requiring a null distribution this project never established,
and the decision rule built on those levels measured a ~43% false-positive rate
on pairs of runs known to be identical. What survives is provenance — the OLS
similarity-trend slope and the surrogate-model attributions — and the tests
below pin that surface shut so levels cannot creep back in.
"""

from __future__ import annotations

import pytest

from hif.metrics.similarity import SimilarityMetrics
from hif.metrics.stability import PerturbationResponse
from hif.profile.builder import generate_findings
from hif.profile.schema import Findings, MetricBundle

from profile_helpers import (
    _make_center,
    _make_distribution_metrics,
    _make_input_analysis,
    _make_output_trace,
    _make_semantic_metrics,
    _make_sensitivity,
)


def _findings(
    *,
    similarity: SimilarityMetrics | None = None,
    surrogate_model_name: str | None = None,
    output_distribution_surrogate_name: str | None = None,
    mean_js: float = 0.05,
) -> Findings:
    input_analysis = _make_input_analysis()
    output_trace = _make_output_trace()
    center = _make_center()
    sens = _make_sensitivity(mean_js=mean_js)
    metric_bundle = MetricBundle(
        distribution=[_make_distribution_metrics()],
        semantic=[_make_semantic_metrics()],
        sensitivity=[sens],
        stability=PerturbationResponse(
            input_entropy_shift_bits=0.4,
            perturbation_jsd_bits=mean_js,
            input_output_correlation=0.0,
            n_perturbations=1,
        ),
        similarity=similarity,
    )
    return generate_findings(
        input_analysis,
        output_trace,
        center,
        metric_bundle,
        surrogate_model_name=surrogate_model_name,
        output_distribution_surrogate_name=output_distribution_surrogate_name,
    )


class TestFindingsSurface:
    """Findings carries provenance and nothing else."""

    def test_exactly_three_fields(self):
        assert set(Findings.model_fields) == {
            "similarity_trend_slope",
            "surrogate_model_name",
            "output_distribution_surrogate_name",
        }

    @pytest.mark.parametrize(
        "removed",
        [
            "stability_level",
            "breadth_level",
            "sensitivity_level",
            "continuity_level",
            "io_correlation_level",
            "surprise_level",
            "equilibrium_flag",
            "summary",
            "notable_regimes",
        ],
    )
    def test_removed_level_fields_stay_removed(self, removed):
        assert removed not in Findings.model_fields
        assert not hasattr(_findings(), removed)


class TestSimilarityTrendSlope:
    def test_absent_similarity_gives_zero_slope(self):
        assert _findings(similarity=None).similarity_trend_slope == 0.0

    @pytest.mark.parametrize("trend", [-0.42, 0.0, 0.137])
    def test_slope_is_passed_through_signed_and_unrounded(self, trend):
        sim = SimilarityMetrics(
            input_sim=0.8,
            output_sim=0.7,
            io_sim=0.6,
            io_ratio=0.875,
            trend=trend,
            n_pairs=2,
        )
        f = _findings(similarity=sim)
        # Signed and exact — not bucketed, not clamped, not abs()'d.
        assert f.similarity_trend_slope == pytest.approx(trend, abs=0.0)


class TestSurrogateProvenance:
    def test_no_surrogate_by_default(self):
        f = _findings()
        assert f.surrogate_model_name is None
        assert f.output_distribution_surrogate_name is None

    def test_input_side_surrogate_recorded(self):
        f = _findings(surrogate_model_name="gpt2")
        assert f.surrogate_model_name == "gpt2"
        # The two surrogate attributions are independent.
        assert f.output_distribution_surrogate_name is None

    def test_output_distribution_surrogate_recorded(self):
        f = _findings(output_distribution_surrogate_name="gpt2")
        assert f.output_distribution_surrogate_name == "gpt2"
        assert f.surrogate_model_name is None

    def test_both_surrogates_recorded_independently(self):
        f = _findings(
            surrogate_model_name="gpt2",
            output_distribution_surrogate_name="distilgpt2",
        )
        assert f.surrogate_model_name == "gpt2"
        assert f.output_distribution_surrogate_name == "distilgpt2"
