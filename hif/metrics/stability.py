"""Perturbation-response metrics: how the input- and output-side measurements
move when the prompt is perturbed in content-preserving ways.

Everything here is reported in its natural unit. Nothing is normalised by a
vocabulary size, squashed into [0, 1], or inverted into a "score".

History (kept deliberately visible — see docs/METRICS.md § Natural units):
this module used to report ``input_stability = 1 - mean|delta volatility|``
and ``output_stability = 1 - mean JSD``. Both were wrong in the same three
ways: they saturated (pinning at exactly 1.0 and destroying resolution in the
regime that mattered), the input one divided by ``log2(vocab_size)`` so
tokenizer metadata leaked into a number presented as behaviour, and ``1 - x``
hid the measurement behind a score. They are replaced by the measured
quantities themselves.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel
from scipy.stats import pearsonr  # type: ignore[import]

from hif.hourglass.input_side import InputSideAnalysis
from hif.metrics.sensitivity import SensitivityMetrics


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PerturbationResponse(BaseModel):
    """Response of the measured quantities to content-preserving perturbation.

    None means the quantity is ABSENT — not measurable for this run (e.g. no
    perturbed input-side analyses because the backend cannot teacher-force).
    Absent is deliberately distinct from a measured value: consumers must omit
    absent quantities, never report them as zero or as perfect.
    """

    # Mean absolute difference, over perturbation variants, between the
    # variant's mean input-token entropy and the baseline's. Unit: bits.
    # Unbounded above; 0 means the perturbations moved input-side entropy not
    # at all. NOT inverted, NOT divided by log2(vocab_size).
    input_entropy_shift_bits: float | None
    # Standard deviation of those same per-variant entropy shifts. Unit: bits.
    # This is the natural-unit form of what used to be reported as
    # "Stability = 1 - normalised(variance)": the spread of the model's
    # entropy response across perturbations, unscaled, uninverted, uncapped.
    # None when fewer than two perturbation variants exist (no spread to
    # measure) — absent, never a fake 0.0.
    input_entropy_std_bits: float | None = None
    # Mean Jensen-Shannon divergence between the baseline output distribution
    # and each perturbed variant's. Unit: bits (log base 2), so genuinely
    # bounded to [0, 1] by definition — left alone, but no longer inverted.
    perturbation_jsd_bits: float | None
    # Pearson r between the per-variant |input entropy shift| and the
    # per-variant JSD. Dimensionless, genuinely bounded to [-1, 1]. Reported
    # signed and un-clamped: the sign is the interesting part.
    input_output_correlation: float | None
    n_perturbations: int
    temperature_robustness: float | None = None
    prompt_order_robustness: float | None = None


# Backwards-compatible alias for the historical class name.
StabilityMetrics = PerturbationResponse


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_stability_metrics(
    baseline_input: InputSideAnalysis,
    perturbed_inputs: list[InputSideAnalysis],
    sensitivity_results: list[SensitivityMetrics],
) -> PerturbationResponse:
    """Compute the perturbation-response measurements.

    Parameters
    ----------
    baseline_input:
        Input-side analysis of the original (unperturbed) prompt.
    perturbed_inputs:
        Input-side analyses of each perturbed prompt variant.
    sensitivity_results:
        Sensitivity metrics for each perturbation (same ordering as
        perturbed_inputs).

    Derivations
    -----------
    input_entropy_shift_bits:
        ``mean(|perturbed.mean_entropy - baseline.mean_entropy|)`` in bits.
        ``mean_entropy`` is the mean per-position Shannon entropy of the
        teacher-forced full-vocabulary distribution over the prompt.

    perturbation_jsd_bits:
        ``mean(mean_js_divergence)`` in bits (log base 2).

    input_output_correlation:
        Pearson r between the per-perturbation ``|input entropy shift|`` (bits)
        and ``mean_js_divergence`` (bits). 0.0 only when the correlation is
        computable (>= 2 aligned points) but degenerate (a constant series
        makes r undefined/NaN); None when there are fewer than 2 aligned
        points — a single perturbation variant has no correlation to report,
        and reporting one as a measured 0.0 would misrepresent "no evidence"
        as "measured zero correlation".

    Absent-not-pinned rule
    ----------------------
    Each quantity is computed independently from whatever evidence exists, and
    is ``None`` when the evidence for it does not exist. Never a fake 0.0 and
    never a fake 1.0.
    """
    n_in = len(perturbed_inputs)
    n_out = len(sensitivity_results)

    # Input-side response — only from real perturbed input-side analyses.
    input_entropy_shift_bits: float | None = None
    entropy_shifts: list[float] = []
    if n_in > 0:
        for p_input in perturbed_inputs:
            entropy_shifts.append(
                abs(p_input.mean_entropy - baseline_input.mean_entropy)
            )
        input_entropy_shift_bits = float(np.mean(entropy_shifts))
    # Spread of the entropy response. Needs >= 2 variants to mean anything.
    input_entropy_std_bits: float | None = None
    if n_in >= 2:
        input_entropy_std_bits = float(np.std(entropy_shifts, ddof=1))

    # Output-side response — only from real sensitivity results.
    perturbation_jsd_bits: float | None = None
    js_divergences: list[float] = [
        sr.mean_js_divergence for sr in sensitivity_results
    ]
    if n_out > 0:
        perturbation_jsd_bits = float(np.mean(js_divergences))

    # Pearson correlation between input entropy shift and output JS divergence.
    r: float | None = None
    if n_in >= 2 and n_out == n_in:
        corr_result = pearsonr(entropy_shifts, js_divergences)
        # scipy >= 1.9 returns a PearsonRResult object; older returns a tuple
        if hasattr(corr_result, "statistic"):
            r = float(corr_result.statistic)
        else:
            r = float(corr_result[0])
        # NaN can occur if one array is constant
        if math.isnan(r):
            r = 0.0
    # else: fewer than 2 aligned points (e.g. a single perturbation variant,
    # the --mode fast default) — correlation is undefined, not zero.

    return PerturbationResponse(
        input_entropy_shift_bits=input_entropy_shift_bits,
        input_entropy_std_bits=input_entropy_std_bits,
        perturbation_jsd_bits=perturbation_jsd_bits,
        input_output_correlation=r,
        n_perturbations=max(n_in, n_out),
    )


# ---------------------------------------------------------------------------
# Legacy shims
# ---------------------------------------------------------------------------


def intra_class_consistency(grouped_embeddings):  # type: ignore[no-untyped-def]
    """Deprecated: prefer compute_stability_metrics for trace-based analysis."""
    raise NotImplementedError(
        "intra_class_consistency is deprecated. Use compute_stability_metrics instead."
    )


def cluster_coherence(labels, embeddings):  # type: ignore[no-untyped-def]
    """Deprecated: prefer compute_stability_metrics for trace-based analysis."""
    raise NotImplementedError(
        "cluster_coherence is deprecated. Use compute_stability_metrics instead."
    )
