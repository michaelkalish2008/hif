"""Unit tests for sensitivity and stability metrics using synthetic data."""

import math

import numpy as np
import pytest

from hif.hourglass.input_side import InputSideAnalysis, PositionRecord
from hif.hourglass.output_side import OutputSideTrace
from hif.metrics.sensitivity import (
    SensitivityMetrics,
    StepSensitivity,
    _nucleus_token_ids,
    compute_sensitivity_metrics,
    compute_step_sensitivity,
    js_divergence,
    kl_divergence,
)
from hif.metrics.stability import (
    PerturbationResponse,
    StabilityMetrics,
    compute_stability_metrics,
)
from hif.models.base import StepRecord, TopKEntry


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_topk(token_probs: dict[int, float]) -> list[TopKEntry]:
    entries = []
    for tid, prob in token_probs.items():
        entries.append(
            TopKEntry(
                token_id=tid,
                token_str=f"tok{tid}",
                logit=float(np.log(prob + 1e-12)),
                logprob=float(np.log(prob + 1e-12)),
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


def _make_trace(
    steps: list[StepRecord],
    prompt: str = "test prompt",
    model_name: str = "fake-model",
    mean_entropy: float = 1.0,
) -> OutputSideTrace:
    return OutputSideTrace(
        steps=steps,
        input_ids=[1, 2, 3],
        generated_ids=[s.selected_token_id for s in steps],
        prompt_text=prompt,
        model_name=model_name,
        top_k=10,
        max_new_tokens=len(steps),
        seed=42,
        mean_step_entropy=mean_entropy,
    )


def _make_position_record(pos: int, entropy: float) -> PositionRecord:
    return PositionRecord(
        position=pos,
        token_id=pos,
        token_str=f"tok{pos}",
        surprisal=1.0,
        entropy=entropy,
        top_k_alternatives=[],
    )


def _make_input_analysis(
    prompt: str = "test",
    mean_entropy: float = 2.0,
    max_entropy: float = 10.0,
) -> InputSideAnalysis:
    pos = _make_position_record(1, mean_entropy)
    return InputSideAnalysis(
        positions=[pos],
        prompt_token_ids=[1, 2],
        prompt_text=prompt,
        mean_surprisal=1.0,
        mean_entropy=mean_entropy,
        max_entropy=max_entropy,
    )


# ---------------------------------------------------------------------------
# JS divergence tests
# ---------------------------------------------------------------------------


class TestJsDivergence:
    def test_identical_distributions(self):
        p = np.array([0.1, 0.3, 0.4, 0.2])
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_identical_uniform(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_distributions(self):
        """JSD of disjoint distributions should equal 1.0 bit (log2 scale)."""
        p = np.array([0.5, 0.5, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.5, 0.5])
        jsd = js_divergence(p, q)
        # For completely disjoint distributions: JSD = 1.0 bit (log2 scale)
        assert jsd <= 1.0 + 1e-10
        assert jsd >= 0.99  # should be very close to 1.0

    def test_result_in_range(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.6, 0.3])
        jsd = js_divergence(p, q)
        assert 0.0 <= jsd <= 1.0 + 1e-10

    def test_symmetry(self):
        p = np.array([0.4, 0.3, 0.3])
        q = np.array([0.1, 0.5, 0.4])
        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-12)


# ---------------------------------------------------------------------------
# KL divergence tests
# ---------------------------------------------------------------------------


class TestKlDivergence:
    def test_identical_distributions(self):
        p = np.array([0.3, 0.3, 0.4])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_point_mass_vs_uniform_is_inf(self):
        """KL(uniform || point mass) is inf because point mass has zero where uniform > 0."""
        p = np.array([0.5, 0.5])  # uniform
        q = np.array([1.0, 0.0])  # point mass
        result = kl_divergence(p, q)
        assert math.isinf(result)

    def test_known_value(self):
        """KL(p||q) for known distributions."""
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        # KL(p||q) = 0.5*log2(0.5/0.25) + 0.5*log2(0.5/0.75)
        expected = 0.5 * math.log2(0.5 / 0.25) + 0.5 * math.log2(0.5 / 0.75)
        assert kl_divergence(p, q) == pytest.approx(expected, rel=1e-6)

    def test_non_negative(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.3, 0.7])
        assert kl_divergence(p, q) >= 0.0


# ---------------------------------------------------------------------------
# Compute step sensitivity
# ---------------------------------------------------------------------------


class TestComputeStepSensitivity:
    def test_identical_steps_zero_jsd(self):
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        step_a = _make_step(0, probs)
        step_b = _make_step(0, probs)
        ss = compute_step_sensitivity(step_a, step_b)
        assert ss.js_divergence == pytest.approx(0.0, abs=1e-10)
        assert ss.kl_divergence == pytest.approx(0.0, abs=1e-10)
        assert ss.entropy_delta == pytest.approx(0.0, abs=1e-10)

    def test_different_steps_positive_jsd(self):
        probs_a = {1: 0.9, 2: 0.1}
        probs_b = {3: 0.9, 4: 0.1}
        step_a = _make_step(0, probs_a)
        step_b = _make_step(0, probs_b)
        ss = compute_step_sensitivity(step_a, step_b)
        assert ss.js_divergence > 0.0
        assert ss.js_divergence <= 1.0 + 1e-10

    def test_returns_step_sensitivity_instance(self):
        probs = {1: 0.6, 2: 0.4}
        step = _make_step(5, probs)
        ss = compute_step_sensitivity(step, step)
        assert isinstance(ss, StepSensitivity)
        assert ss.step == 5


# ---------------------------------------------------------------------------
# Compute sensitivity metrics (structure)
# ---------------------------------------------------------------------------


class TestComputeSensitivityMetrics:
    def _make_uniform_trace(self, n_steps: int, prompt: str = "test") -> OutputSideTrace:
        probs = {i: 1.0 / 4 for i in range(4)}
        steps = [_make_step(i, probs) for i in range(n_steps)]
        return _make_trace(steps, prompt=prompt, mean_entropy=2.0)

    def test_returns_sensitivity_metrics(self):
        base = self._make_uniform_trace(3, "original prompt")
        pert = self._make_uniform_trace(3, "perturbed prompt")
        result = compute_sensitivity_metrics(base, pert, "perturbed prompt", "synonym")
        assert isinstance(result, SensitivityMetrics)

    def test_correct_field_values(self):
        base = self._make_uniform_trace(3, "original prompt")
        pert = self._make_uniform_trace(3, "perturbed prompt")
        result = compute_sensitivity_metrics(base, pert, "perturbed prompt", "synonym")
        assert result.perturbation_generator == "synonym"
        assert result.perturbed_prompt == "perturbed prompt"
        assert result.original_prompt == "original prompt"
        assert len(result.step_sensitivities) == 3

    def test_identical_traces_zero_jsd(self):
        probs = {1: 0.5, 2: 0.3, 3: 0.2}
        steps = [_make_step(i, probs) for i in range(4)]
        base = _make_trace(steps, prompt="p", mean_entropy=1.5)
        pert = _make_trace(steps, prompt="p2", mean_entropy=1.5)
        result = compute_sensitivity_metrics(base, pert, "p2", "word_order")
        assert result.mean_js_divergence == pytest.approx(0.0, abs=1e-10)

    def test_non_negative_mean_js(self):
        base = self._make_uniform_trace(5, "a")
        pert = self._make_uniform_trace(5, "b")
        result = compute_sensitivity_metrics(base, pert, "b", "tone")
        assert result.mean_js_divergence >= 0.0

    def test_mismatched_step_counts(self):
        """Should use min(len(base), len(pert)) steps."""
        probs = {1: 0.6, 2: 0.4}
        base = _make_trace([_make_step(i, probs) for i in range(5)], prompt="a")
        pert = _make_trace([_make_step(i, probs) for i in range(3)], prompt="b")
        result = compute_sensitivity_metrics(base, pert, "b", "substitution")
        assert len(result.step_sensitivities) == 3


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------


class TestComputeStabilityMetrics:
    def _make_sensitivity(self, mean_js: float) -> SensitivityMetrics:
        return SensitivityMetrics(
            perturbation_generator="synonym",
            perturbed_prompt="p",
            original_prompt="o",
            step_sensitivities=[
                StepSensitivity(
                    step=0,
                    js_divergence=mean_js,
                    kl_divergence=0.0,
                    entropy_delta=0.0,
                    nucleus_overlap_p90=1.0,
                )
            ],
            mean_js_divergence=mean_js,
            mean_kl_divergence=0.0,
            mean_entropy_delta=0.0,
            output_entropy_delta=0.0,
            mean_nucleus_stability_p90=1.0,
        )

    def test_identical_inputs_report_zero_response(self):
        """Baseline and perturbed identical → the measured response is 0, in
        both natural units. Zero here is a real measurement (there WERE
        perturbations, they just moved nothing), never the absent marker."""
        baseline = _make_input_analysis(mean_entropy=2.0)
        perturbed = [
            _make_input_analysis(mean_entropy=2.0) for _ in range(3)
        ]
        sensitivities = [self._make_sensitivity(0.0) for _ in range(3)]
        result = compute_stability_metrics(baseline, perturbed, sensitivities)
        assert result.perturbation_jsd_bits == pytest.approx(0.0, abs=1e-10)
        assert result.input_entropy_shift_bits == pytest.approx(0.0, abs=1e-10)

    def test_measurements_are_in_natural_units(self):
        """input_entropy_shift_bits is the mean |Δ mean_entropy| in bits —
        unbounded above and NOT inverted; perturbation_jsd_bits is base-2 JSD,
        genuinely bounded to [0, 1]."""
        baseline = _make_input_analysis(mean_entropy=2.0)
        perturbed = [
            _make_input_analysis(mean_entropy=3.0),   # Δ 1.0
            _make_input_analysis(mean_entropy=8.0),   # Δ 6.0
            _make_input_analysis(mean_entropy=1.0),   # Δ 1.0
        ]
        sensitivities = [
            self._make_sensitivity(0.1),
            self._make_sensitivity(0.2),
            self._make_sensitivity(0.05),
        ]
        result = compute_stability_metrics(baseline, perturbed, sensitivities)
        assert isinstance(result, PerturbationResponse)
        # mean(|1.0|, |6.0|, |1.0|) — a bits quantity well above 1.0, proving
        # nothing squashes it into [0, 1].
        assert result.input_entropy_shift_bits == pytest.approx(8.0 / 3.0)
        # std(|1.0|, |6.0|, |1.0|, ddof=1) — the spread of the same shifts,
        # also in raw bits and also above 1.0.
        assert result.input_entropy_std_bits == pytest.approx(
            float(np.std([1.0, 6.0, 1.0], ddof=1))
        )
        assert result.perturbation_jsd_bits == pytest.approx(0.35 / 3.0)
        assert 0.0 <= result.perturbation_jsd_bits <= 1.0

    def test_returns_perturbation_response_instance(self):
        baseline = _make_input_analysis()
        perturbed = [_make_input_analysis()]
        sensitivities = [self._make_sensitivity(0.0)]
        result = compute_stability_metrics(baseline, perturbed, sensitivities)
        assert isinstance(result, PerturbationResponse)
        # The historical name still resolves to the same class, so profile
        # JSON and downstream imports written against it keep working.
        assert StabilityMetrics is PerturbationResponse

    def test_n_perturbations_correct(self):
        baseline = _make_input_analysis()
        perturbed = [_make_input_analysis() for i in range(4)]
        sensitivities = [self._make_sensitivity(0.05 * i) for i in range(4)]
        result = compute_stability_metrics(baseline, perturbed, sensitivities)
        assert result.n_perturbations == 4

    def test_empty_perturbations_absent_not_pinned(self):
        """No evidence at all → every component ABSENT (None), never fake 1.0/0.0."""
        baseline = _make_input_analysis()
        result = compute_stability_metrics(baseline, [], [])
        assert result.input_entropy_shift_bits is None
        assert result.perturbation_jsd_bits is None
        assert result.input_output_correlation is None
        assert result.n_perturbations == 0

    def test_sensitivity_only_partial_access_absent_not_pinned(self):
        """Sensitivity results without perturbed input analyses (partial-access
        models; media perturbation on non-teacher-forcing backends): the output
        response is real, the input-side quantities are absent."""
        baseline = _make_input_analysis()
        sensitivities = [self._make_sensitivity(0.2), self._make_sensitivity(0.4)]
        result = compute_stability_metrics(baseline, [], sensitivities)
        assert result.input_entropy_shift_bits is None
        assert result.input_output_correlation is None
        assert result.perturbation_jsd_bits == pytest.approx(0.3, abs=1e-10)
        assert result.n_perturbations == 2

    def test_full_access_varying_inputs_real_values(self):
        """Aligned input + output series → all three quantities computed."""
        baseline = _make_input_analysis(mean_entropy=2.0)
        perturbed = [
            _make_input_analysis(mean_entropy=2.1),
            _make_input_analysis(mean_entropy=5.0),
            _make_input_analysis(mean_entropy=2.05),
        ]
        sensitivities = [
            self._make_sensitivity(0.05),
            self._make_sensitivity(0.5),
            self._make_sensitivity(0.02),
        ]
        result = compute_stability_metrics(baseline, perturbed, sensitivities)
        assert result.input_entropy_shift_bits is not None
        assert result.input_entropy_shift_bits > 0.0
        assert result.perturbation_jsd_bits is not None
        assert result.perturbation_jsd_bits > 0.0
        # Both series move together here, so r is a real, strongly positive
        # correlation — not the degenerate 0.0 fallback for a constant series.
        assert result.input_output_correlation is not None
        assert result.input_output_correlation > 0.9

    def test_higher_js_raises_reported_jsd(self):
        baseline = _make_input_analysis()
        perturbed = [_make_input_analysis()]
        # max JS in bits (log2) is 1.0; use 0.8 as "high"
        sensitivities_high = [self._make_sensitivity(0.8)]
        sensitivities_low = [self._make_sensitivity(0.0)]
        result_high = compute_stability_metrics(baseline, perturbed, sensitivities_high)
        result_low = compute_stability_metrics(baseline, perturbed, sensitivities_low)
        # Reported directly, not as 1 - x: more divergence is a BIGGER number.
        assert result_high.perturbation_jsd_bits > result_low.perturbation_jsd_bits
        assert result_high.perturbation_jsd_bits == pytest.approx(0.8)

    def test_optional_fields_none_by_default(self):
        baseline = _make_input_analysis()
        result = compute_stability_metrics(baseline, [], [])
        assert result.temperature_robustness is None
        assert result.prompt_order_robustness is None


# ---------------------------------------------------------------------------
# Nucleus token IDs helper
# ---------------------------------------------------------------------------


class TestNucleusTokenIds:
    def test_point_mass_single_token(self):
        """A single dominant token covers p90 immediately."""
        ids = [1, 2, 3]
        probs = [0.95, 0.03, 0.02]
        nucleus = _nucleus_token_ids(ids, probs, p=0.9)
        assert nucleus == {1}

    def test_two_tokens_needed(self):
        ids = [1, 2, 3]
        probs = [0.6, 0.4, 0.0]
        nucleus = _nucleus_token_ids(ids, probs, p=0.9)
        assert nucleus == {1, 2}

    def test_all_tokens_when_no_single_dominant(self):
        ids = [1, 2, 3]
        probs = [0.33, 0.33, 0.34]
        nucleus = _nucleus_token_ids(ids, probs, p=0.99)
        assert nucleus == {1, 2, 3}

    def test_empty_returns_empty_set(self):
        assert _nucleus_token_ids([], [], p=0.9) == set()


# ---------------------------------------------------------------------------
# Nucleus overlap in step sensitivity
# ---------------------------------------------------------------------------


class TestNucleusOverlapInStepSensitivity:
    def test_identical_steps_full_overlap(self):
        """Identical steps → nucleus overlap = 1.0."""
        probs = {1: 0.7, 2: 0.2, 3: 0.1}
        step = _make_step(0, probs)
        ss = compute_step_sensitivity(step, step)
        assert ss.nucleus_overlap_p90 == pytest.approx(1.0, abs=1e-10)

    def test_fully_disjoint_nuclei_zero_overlap(self):
        """Completely different top tokens → overlap = 0.0."""
        probs_a = {1: 0.95, 2: 0.05}
        probs_b = {3: 0.95, 4: 0.05}
        step_a = _make_step(0, probs_a)
        step_b = _make_step(0, probs_b)
        ss = compute_step_sensitivity(step_a, step_b)
        assert ss.nucleus_overlap_p90 == pytest.approx(0.0, abs=1e-10)

    def test_partial_overlap(self):
        """Shared token 1 in nucleus; token 2 swapped out → partial overlap."""
        probs_a = {1: 0.6, 2: 0.4}          # nucleus(p90) = {1, 2} (need both for 0.9)
        probs_b = {1: 0.6, 3: 0.4}          # nucleus(p90) = {1, 3}
        step_a = _make_step(0, probs_a)
        step_b = _make_step(0, probs_b)
        ss = compute_step_sensitivity(step_a, step_b)
        # baseline nucleus = {1, 2}, perturbed = {1, 3}, intersection = {1}
        # overlap = 1/2 = 0.5
        assert ss.nucleus_overlap_p90 == pytest.approx(0.5, abs=1e-10)

    def test_nucleus_overlap_in_range(self):
        probs_a = {1: 0.5, 2: 0.3, 3: 0.2}
        probs_b = {1: 0.4, 2: 0.4, 4: 0.2}
        ss = compute_step_sensitivity(_make_step(0, probs_a), _make_step(0, probs_b))
        assert 0.0 <= ss.nucleus_overlap_p90 <= 1.0


# ---------------------------------------------------------------------------
# Mean nucleus stability in sensitivity metrics
# ---------------------------------------------------------------------------


class TestMeanNucleusStabilityInSensitivityMetrics:
    def _make_uniform_trace(self, n_steps: int, prompt: str = "test") -> OutputSideTrace:
        probs = {i: 0.25 for i in range(4)}
        steps = [_make_step(i, probs) for i in range(n_steps)]
        return _make_trace(steps, prompt=prompt, mean_entropy=2.0)

    def test_identical_traces_stability_one(self):
        """Identical baseline and perturbed → mean_nucleus_stability_p90 = 1.0."""
        probs = {1: 0.6, 2: 0.3, 3: 0.1}
        steps = [_make_step(i, probs) for i in range(3)]
        base = _make_trace(steps, prompt="p", mean_entropy=1.5)
        pert = _make_trace(steps, prompt="p2", mean_entropy=1.5)
        result = compute_sensitivity_metrics(base, pert, "p2", "synonym")
        assert result.mean_nucleus_stability_p90 == pytest.approx(1.0, abs=1e-10)

    def test_stability_in_range(self):
        base = self._make_uniform_trace(4, "a")
        pert = self._make_uniform_trace(4, "b")
        result = compute_sensitivity_metrics(base, pert, "b", "tone")
        assert 0.0 <= result.mean_nucleus_stability_p90 <= 1.0

    def test_empty_steps_stability_defaults_to_one(self):
        """Zero steps → mean_nucleus_stability_p90 defaults to 1.0 (no instability observed)."""
        base = _make_trace([], prompt="p", mean_entropy=0.0)
        pert = _make_trace([], prompt="p2", mean_entropy=0.0)
        result = compute_sensitivity_metrics(base, pert, "p2", "synonym")
        assert result.mean_nucleus_stability_p90 == pytest.approx(1.0, abs=1e-10)
