"""Tests for the "exposure" instrument reading.

exposure = fraction of analyzed generation steps flagged high-divergence
(a probabilistically accessible alternative was semantically distant, in a
diffuse candidate cloud), in [0, 1]. It measures counterfactual semantic
exposure, not factuality, and is absent (never pinned to 0) when the
semantic-divergence analysis wasn't computed for a run.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import hif.cli as cli
from hif.analysis.exposure import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_MIN_PROB,
    ExposureAnalyzer,
    ExposureCandidate,
    ExposureProfile,
)
from hif.cli import app
from tests.unit.profile_helpers import _make_profile
from tests.unit.test_exposure import (
    FakeEmbeddingModel,
    _make_output_step,
    _make_output_trace,
    _make_semantic_metrics,
)

runner = CliRunner()


def _candidate(step: int, distance: float, phenomenon: str = "diffusion") -> ExposureCandidate:
    return ExposureCandidate(
        step=step, selected_token="a", selected_prob=0.5,
        divergent_token="b", divergent_prob=0.1,
        prob_rank=1, semantic_distance=distance,
        cloud_phenomenon=phenomenon, cloud_position_2d=[],
    )


def _synthetic_profile(n_candidates: int, exposed: list[int]) -> ExposureProfile:
    return ExposureProfile(
        candidates=[_candidate(i, 0.5) for i in range(n_candidates)],
        exposed_steps=exposed,
        mean_semantic_distance=0.5,
        diffusion_zone_ratio=1.0,
        exposure=len(exposed) / n_candidates,
        embedder="fake-embedder",
    )


# ---------------------------------------------------------------------------
# Scalar derivation
# ---------------------------------------------------------------------------


class TestExposureScalar:
    def test_defaults_are_documented_constants(self):
        assert DEFAULT_MIN_PROB == pytest.approx(0.01)
        assert DEFAULT_DISTANCE_THRESHOLD == pytest.approx(0.3)

    def test_old_profiles_default_to_zero_exposure_and_no_embedder(self):
        # Back-compat: profiles serialized before these fields existed.
        hp = ExposureProfile(
            candidates=[], exposed_steps=[],
            mean_semantic_distance=0.0, diffusion_zone_ratio=0.0,
        )
        assert hp.exposure == 0.0
        assert hp.embedder is None

    def test_analyzer_exposure_is_exposed_fraction_of_analyzed_steps(self):
        steps = [
            _make_output_step(i, f"tok{i}", [(f"tok{i}", 0.5), (f"alt{i}", 0.2)])
            for i in range(4)
        ]
        # All steps in a diffuse cloud so distance alone decides high-risk.
        sems = [_make_semantic_metrics(cluster_count=0) for _ in steps]
        analyzer = ExposureAnalyzer(FakeEmbeddingModel())
        profile = analyzer.analyze(_make_output_trace(steps), sems)

        assert profile.candidates, "analysis should produce candidates"
        assert 0.0 <= profile.exposure <= 1.0
        assert profile.exposure == pytest.approx(
            len(profile.exposed_steps) / len(profile.candidates)
        )

    def test_empty_analysis_yields_zero_exposure(self):
        analyzer = ExposureAnalyzer(FakeEmbeddingModel())
        profile = analyzer.analyze(_make_output_trace([]), [])
        assert profile.exposure == 0.0


# ---------------------------------------------------------------------------
# CLI: measurements() + --metric counterfactual_exposure_fraction
# ---------------------------------------------------------------------------

EXPOSURE_KEY = "counterfactual_exposure_fraction"


def _patch_pipeline(monkeypatch, profile):
    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_load_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_run_single_profile", lambda *a, **k: (profile, None))


class TestExposureCli:
    def test_measurement_absent_when_not_computed(self):
        p = _make_profile()
        p.exposure = None
        values = cli._measurements(p)
        assert EXPOSURE_KEY not in values  # absent, never pinned to 0

    def test_measurement_absent_when_no_candidates(self):
        p = _make_profile()
        p.exposure = ExposureProfile(
            candidates=[], exposed_steps=[],
            mean_semantic_distance=0.0, diffusion_zone_ratio=0.0,
        )
        values = cli._measurements(p)
        assert EXPOSURE_KEY not in values

    def test_populated_exposure_block_emits_no_measurement(self):
        """The block is artifact evidence; the measurement was cut in hif-v4.

        The fraction was defined by two embedded thresholds (min_prob,
        distance) inside a no-thresholds instrument. The analysis still runs
        and its block still ships — but nothing about it is claimed as a
        measurement, and this pins that a populated block does not leak one.
        """
        p = _make_profile()
        p.exposure = _synthetic_profile(4, [0, 2])
        values = cli._measurements(p)
        assert EXPOSURE_KEY not in values



    def test_metric_exposure_is_a_retired_key_and_says_so(self, monkeypatch, tmp_path):
        """--metric with the hif-v4-retired key is an unknown-metric error.

        Before the cut this asserted exit 1 (measurement absent for the run).
        The key is no longer in the set at all, so the correct answer is the
        unknown-metric exit (3) with the schema pointer — and still never the
        forbidden word: the analysis was renamed from "hallucination" because
        it never established one.
        """
        p = _make_profile()
        p.exposure = None
        _patch_pipeline(monkeypatch, p)
        result = runner.invoke(
            app, ["profile", "m", "p", "--output-dir", str(tmp_path),
                  "--metric", EXPOSURE_KEY],
        )
        assert result.exit_code == 3
        assert "hallucin" not in result.output.lower()

    def test_verbose_stats_show_high_divergence_line_when_computed(self, monkeypatch, tmp_path):
        p = _make_profile()
        p.exposure =_synthetic_profile(4, [0, 2])
        _patch_pipeline(monkeypatch, p)
        result = runner.invoke(
            app, ["profile", "m", "p", "--output-dir", str(tmp_path),
                  "--verbose"],
        )
        assert result.exit_code == 0
        assert "exposure" in result.output
        assert "hallucin" not in result.output.lower()

    def test_help_text_has_no_forbidden_word(self):
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0
        assert "hallucin" not in result.output.lower()


# ---------------------------------------------------------------------------
# surprise: absent (never a pinned 0.0) without teacher-forced input positions
# ---------------------------------------------------------------------------


SURPRISAL_KEY = "prompt_surprisal_excess_bits"


class TestSurprisalExcessAbsent:
    def test_absent_without_input_positions(self):
        # Backends with no teacher-forced positions must not emit a pinned
        # 0.0 — absent-not-pinned, same rule as exposure above.
        p = _make_profile()
        p.input_side.positions = []
        values = cli._measurements(p)
        assert SURPRISAL_KEY not in values

    def test_present_with_positions_as_surprisal_excess_in_bits(self):
        """The measurement is mean max(0, surprisal − entropy) per position,
        in bits — not the raw mean surprisal, and not normalised by anything."""
        p = _make_profile()
        for pos in p.input_side.positions:
            pos.surprisal = 6.0
            pos.entropy = 2.0
        values = cli._measurements(p)
        assert values[SURPRISAL_KEY] == pytest.approx(4.0)

    def test_zero_excess_is_measured_not_absent(self):
        p = _make_profile()
        for pos in p.input_side.positions:
            pos.surprisal = 1.0
            pos.entropy = 4.0  # excess clamps at 0
        values = cli._measurements(p)
        assert values[SURPRISAL_KEY] == pytest.approx(0.0)
