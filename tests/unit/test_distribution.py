"""Unit tests for distribution metrics using synthetic distributions."""

import numpy as np
import pytest

from hif.metrics.distribution import (
    percentile_entropy_bits,
    nucleus_entropy_bits,
    DistributionMetrics,
    compute_distribution_metrics,
    effective_support_size,
    entropy_bits,
    logit_margin,
    nucleus_fraction,
    tail_weight,
    topk_cumulative_mass,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_uniform(k: int) -> np.ndarray:
    return np.ones(k, dtype=np.float64) / k


def make_point_mass(size: int = 10, idx: int = 0) -> np.ndarray:
    p = np.zeros(size, dtype=np.float64)
    p[idx] = 1.0
    return p


def make_two_peak() -> np.ndarray:
    p = np.zeros(100, dtype=np.float64)
    p[0] = 0.6
    p[1] = 0.4
    return p


def make_long_tail(size: int = 100) -> np.ndarray:
    """One dominant token at 0.5, rest spread uniformly."""
    p = np.zeros(size, dtype=np.float64)
    p[0] = 0.5
    rest = (1.0 - 0.5) / (size - 1)
    p[1:] = rest
    return p


# ---------------------------------------------------------------------------
# entropy_bits
# ---------------------------------------------------------------------------

class TestEntropyBits:
    def test_uniform_k4(self):
        k = 4
        p = make_uniform(k)
        assert abs(entropy_bits(p) - np.log2(k)) < 1e-6

    def test_uniform_k8(self):
        k = 8
        p = make_uniform(k)
        assert abs(entropy_bits(p) - np.log2(k)) < 1e-6

    def test_point_mass_is_zero(self):
        p = make_point_mass(10, 0)
        assert entropy_bits(p) == pytest.approx(0.0, abs=1e-9)

    def test_two_peak(self):
        p = make_two_peak()
        expected = -(0.6 * np.log2(0.6) + 0.4 * np.log2(0.4))
        assert abs(entropy_bits(p) - expected) < 1e-6

    def test_all_zero_does_not_crash(self):
        p = np.zeros(10, dtype=np.float64)
        # All entries clipped to 1e-10 — result may not be 0, but must not raise
        result = entropy_bits(p)
        assert np.isfinite(result)

    def test_single_element(self):
        p = np.array([1.0])
        assert entropy_bits(p) == pytest.approx(0.0, abs=1e-9)

    def test_entropy_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            raw = rng.random(50)
            p = raw / raw.sum()
            assert entropy_bits(p) >= 0.0


# ---------------------------------------------------------------------------
# logit_margin
# ---------------------------------------------------------------------------

class TestLogitMargin:
    def test_known_margin(self):
        logits = np.array([3.0, 1.5, 0.0, -1.0])
        assert logit_margin(logits) == pytest.approx(1.5, abs=1e-9)

    def test_single_element_returns_zero(self):
        logits = np.array([5.0])
        assert logit_margin(logits) == pytest.approx(0.0, abs=1e-9)

    def test_empty_returns_zero(self):
        logits = np.array([], dtype=np.float64)
        assert logit_margin(logits) == pytest.approx(0.0, abs=1e-9)

    def test_negative_logits(self):
        logits = np.array([-1.0, -3.0, -5.0])
        assert logit_margin(logits) == pytest.approx(2.0, abs=1e-9)

    def test_order_independent(self):
        logits = np.array([1.0, 5.0, 3.0])
        assert logit_margin(logits) == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# topk_cumulative_mass
# ---------------------------------------------------------------------------

class TestTopkCumulativeMass:
    def test_top1_equals_max(self):
        p = make_two_peak()
        assert topk_cumulative_mass(p, k=1) == pytest.approx(0.6, abs=1e-9)

    def test_top2(self):
        p = make_two_peak()
        assert topk_cumulative_mass(p, k=2) == pytest.approx(1.0, abs=1e-9)

    def test_uniform(self):
        k = 4
        p = make_uniform(k)
        mass = topk_cumulative_mass(p, k=2)
        assert mass == pytest.approx(0.5, abs=1e-9)

    def test_k_larger_than_array(self):
        p = np.array([0.5, 0.3, 0.2])
        assert topk_cumulative_mass(p, k=10) == pytest.approx(1.0, abs=1e-9)

    def test_point_mass_top1(self):
        p = make_point_mass(10, 0)
        assert topk_cumulative_mass(p, k=1) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# effective_support_size
# ---------------------------------------------------------------------------

class TestEffectiveSupportSize:
    def test_point_mass_is_one(self):
        p = make_point_mass(10, 0)
        assert effective_support_size(p) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_k4(self):
        k = 4
        p = make_uniform(k)
        assert effective_support_size(p) == pytest.approx(float(k), abs=1e-6)

    def test_uniform_k16(self):
        k = 16
        p = make_uniform(k)
        assert effective_support_size(p) == pytest.approx(float(k), abs=1e-6)

    def test_two_peak_between_one_and_two(self):
        p = make_two_peak()
        ess = effective_support_size(p)
        assert 1.0 < ess < 2.0

    def test_always_positive(self):
        p = make_long_tail()
        assert effective_support_size(p) > 0.0


# ---------------------------------------------------------------------------
# tail_weight
# ---------------------------------------------------------------------------

class TestTailWeight:
    def test_point_mass_no_tail(self):
        p = make_point_mass(10, 0)
        # Only entry is 1.0, which is >= default threshold 0.01
        assert tail_weight(p) == pytest.approx(0.0, abs=1e-9)

    def test_uniform_below_threshold(self):
        k = 1000
        p = make_uniform(k)  # each prob = 0.001 < 0.01
        assert tail_weight(p, threshold=0.01) == pytest.approx(1.0, abs=1e-9)

    def test_two_peak_no_tail(self):
        p = make_two_peak()
        # Both entries are >= 0.01, zeros are not >= threshold but sum to 0
        assert tail_weight(p, threshold=0.01) == pytest.approx(0.0, abs=1e-9)

    def test_long_tail_has_tail(self):
        p = make_long_tail(100)  # entries 1..99 each = 0.5/99 ≈ 0.005 < 0.01
        tw = tail_weight(p, threshold=0.01)
        assert tw > 0.0
        assert tw < 1.0

    def test_custom_threshold(self):
        p = np.array([0.9, 0.05, 0.05])
        # With threshold=0.1: 0.05 and 0.05 are below → tail = 0.1
        assert tail_weight(p, threshold=0.1) == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_distribution_metrics
# ---------------------------------------------------------------------------

class TestComputeDistributionMetrics:
    def test_returns_distribution_metrics_instance(self):
        p = make_uniform(10)
        logits = np.log(p)
        result = compute_distribution_metrics(p, logits)
        assert isinstance(result, DistributionMetrics)

    def test_truncated_flag_false(self):
        p = make_uniform(10)
        logits = np.log(p)
        result = compute_distribution_metrics(p, logits, truncated=False)
        assert result.truncated is False

    def test_truncated_flag_true(self):
        p = make_uniform(10)
        logits = np.log(p)
        result = compute_distribution_metrics(p, logits, truncated=True)
        assert result.truncated is True

    def test_entropy_consistent(self):
        p = make_two_peak()
        logits = np.zeros(len(p))
        result = compute_distribution_metrics(p, logits)
        expected = entropy_bits(p)
        assert result.entropy_bits == pytest.approx(expected, abs=1e-9)

    def test_logit_margin_consistent(self):
        p = make_uniform(5)
        logits = np.array([5.0, 3.0, 1.0, 0.0, -1.0])
        result = compute_distribution_metrics(p, logits)
        assert result.logit_margin == pytest.approx(2.0, abs=1e-9)

    def test_topk_mass_consistent(self):
        p = make_two_peak()
        logits = np.zeros(len(p))
        result = compute_distribution_metrics(p, logits, top_k_for_mass=1)
        assert result.topk_cumulative_mass == pytest.approx(0.6, abs=1e-9)

    def test_effective_support_consistent(self):
        k = 8
        p = make_uniform(k)
        logits = np.log(p)
        result = compute_distribution_metrics(p, logits)
        assert result.effective_support_size == pytest.approx(float(k), abs=1e-6)

    def test_tail_weight_consistent(self):
        p = make_uniform(1000)  # each 0.001 < 0.01
        logits = np.log(p)
        result = compute_distribution_metrics(p, logits, tail_threshold=0.01)
        assert result.tail_weight == pytest.approx(1.0, abs=1e-9)

    def test_nucleus_fraction_present(self):
        p = make_uniform(10)
        logits = np.log(p)
        result = compute_distribution_metrics(p, logits)
        assert "p90" in result.nucleus_fraction
        assert "p95" in result.nucleus_fraction

    def test_nucleus_fraction_with_vocab_size(self):
        """With vocab_size provided, fraction should be relative to vocab_size."""
        p = make_two_peak()  # 0.6 + 0.4 → needs 2 tokens for p90
        logits = np.zeros(len(p))
        result = compute_distribution_metrics(p, logits, vocab_size=50_000)
        # 2 tokens out of 50_000
        assert result.nucleus_fraction["p90"] == pytest.approx(2 / 50_000, abs=1e-9)


# ---------------------------------------------------------------------------
# nucleus_fraction standalone
# ---------------------------------------------------------------------------


class TestNucleusFraction:
    def test_point_mass_single_token_fraction(self):
        """One token covers p90 → 1/len(probs)."""
        p = make_point_mass(100, 0)
        frac = nucleus_fraction(p, p=0.9)
        assert frac == pytest.approx(1 / 100, abs=1e-9)

    def test_uniform_needs_all_tokens(self):
        """Uniform distribution: need 90% of tokens to cover p90."""
        k = 100
        p = make_uniform(k)
        frac = nucleus_fraction(p, p=0.9)
        assert frac == pytest.approx(90 / 100, abs=0.01)

    def test_two_peak_p90_two_tokens(self):
        """0.6 + 0.4 = 1.0 ≥ 0.9, so 2 tokens needed."""
        p = make_two_peak()
        frac = nucleus_fraction(p, p=0.9)
        assert frac == pytest.approx(2 / 100, abs=1e-9)

    def test_vocab_size_scales_denominator(self):
        p = make_point_mass(10, 0)
        # 1 token needed; vocab is 50_257
        frac = nucleus_fraction(p, p=0.9, vocab_size=50_257)
        assert frac == pytest.approx(1 / 50_257, abs=1e-12)

    def test_result_in_zero_one(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            raw = rng.random(50)
            p = raw / raw.sum()
            frac = nucleus_fraction(p, p=0.9)
            assert 0.0 <= frac <= 1.0 + 1e-9

    def test_p95_geq_p90(self):
        """p95 nucleus must be at least as large as p90 nucleus."""
        p = make_long_tail(100)
        f90 = nucleus_fraction(p, p=0.90)
        f95 = nucleus_fraction(p, p=0.95)
        assert f95 >= f90 - 1e-9


# ---------------------------------------------------------------------------
# percentile_entropy_bits — the measurement-grade nucleus
# ---------------------------------------------------------------------------

class TestPercentileEntropyBits:
    """The one behaviour separating it from nucleus_entropy_bits: refusing."""

    def test_agrees_with_nucleus_when_the_slice_contains_the_nucleus(self):
        # A full distribution carries all the mass, so both read the same
        # quantity and must return the same number.
        probs = np.array([0.5, 0.25, 0.15, 0.10])
        assert percentile_entropy_bits(probs, 0.95) == pytest.approx(
            nucleus_entropy_bits(probs, p=0.95)
        )

    def test_absent_when_the_captured_slice_falls_short(self):
        # Top-k from a long-tailed vocabulary: 0.80 of the mass is visible and
        # the p95 nucleus extends into tokens this run never saw. Where
        # nucleus_entropy_bits degrades to "use what I have", the measurement
        # reports nothing.
        probs = np.array([0.4, 0.2, 0.1, 0.1])
        assert probs.sum() < 0.95
        assert percentile_entropy_bits(probs, 0.95) is None
        assert nucleus_entropy_bits(probs, p=0.95) > 0  # still draws a chart

    def test_exact_mass_is_enough(self):
        # A slice carrying exactly p contains the nucleus — the boundary is
        # inclusive, so a run is not refused for landing on it.
        probs = np.array([0.6, 0.35])
        assert probs.sum() == pytest.approx(0.95)
        assert percentile_entropy_bits(probs, 0.95) is not None

    def test_a_lower_percentile_reads_a_smaller_choice_set(self):
        # Monotonicity is the property that makes the flag meaningful: asking
        # for less mass must never report more uncertainty.
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        assert percentile_entropy_bits(probs, 0.50) < percentile_entropy_bits(
            probs, 0.99
        )

    def test_empty_is_absent_not_zero(self):
        assert percentile_entropy_bits(np.array([]), 0.95) is None
