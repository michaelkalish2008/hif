"""Sensitivity metrics: output shift magnitude as a function of structured prompt perturbation.

Approximation note
------------------
All divergence computations operate on truncated top-K distributions.  We build a
distribution over the *union* of token IDs present in the two steps' top-K lists and
assign zero probability to tokens absent from a given step.  This under-estimates true
divergence (the tail mass is invisible), but it is the best we can do without full
vocabulary logits at generation time.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pydantic import BaseModel
from scipy.special import rel_entr  # type: ignore[import]

from hif.hourglass.output_side import OutputSideTrace
from hif.models.base import StepRecord
from hif.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class StepSensitivity(BaseModel):
    step: int
    js_divergence: float  # Jensen-Shannon divergence between baseline and perturbed top-K dists
    # KL(baseline || perturbed) in bits. None when the perturbed distribution
    # places no mass somewhere the baseline does — KL is genuinely infinite
    # there, which is a fact about the pair and not a large number.
    #
    # This was clamped to a 1e9 sentinel "so the value round-trips through
    # JSON". null round-trips through JSON perfectly well, and the clamp cost
    # more than it bought: `compute_sensitivity_metrics` filters the steps it
    # averages with `math.isfinite`, and 1e9 IS finite — so the guard written
    # to exclude undefined steps excluded none of them, and
    # `mean_kl_divergence` came out at 9.65e8 on 833 records across half the
    # published corpus, rendered into the report table as 965517241.3793.
    kl_divergence: Optional[float]
    entropy_delta: float  # perturbed_entropy - baseline_entropy
    # Set-based: |baseline_nucleus_p90 ∩ perturbed_nucleus_p90| / |baseline_nucleus_p90|.
    # None when the baseline nucleus is empty — there is no set to take a
    # fraction of. This was 1.0 ("define as stable"), the maximum of the range,
    # asserting perfect nucleus agreement from no nucleus at all.
    nucleus_overlap_p90: Optional[float]
    baseline_topk_probs: list[float] = []   # ranked probabilities from baseline (desc), no token strings
    perturbed_topk_probs: list[float] = []  # ranked probabilities from perturbed variant (desc)


class SensitivityMetrics(BaseModel):
    """One paraphrase variant's divergence from the baseline, step by step.

    Every mean here is over `step_sensitivities`, and every one of them is
    None when that list is empty. It used to be a set of constants — 0.0 for
    the divergences, 1.0 for nucleus stability — which is the fabricated-zero
    pattern `io_correlation_r` was retired for, one layer below where anyone
    was looking: `stability.py` declares "Never a fake 0.0 and never a fake
    1.0" in its own docstring and enforces it faithfully at its own level,
    while being handed fabrications from here.

    A variant can produce no aligned steps for an ordinary reason — the
    variant generated nothing, or the baseline did. Six variants of
    `gpt5/legal_compliance` aligned zero steps against a 603-step baseline and
    each contributed a measured 0.0 to the run's perturbation response.
    """

    perturbation_generator: str
    perturbed_prompt: str
    original_prompt: str
    step_sensitivities: list[StepSensitivity]
    # How many steps the two traces actually shared. `len(step_sensitivities)`
    # says the same thing, but only to a reader who knows to count — and the
    # count is the difference between a mean over 64 steps and a mean over 6,
    # which the aggregate weights identically. `PerturbationField` has carried
    # `n_steps_aligned` for the same alignment since 0.4.0; the record that
    # feeds the headline measurement did not.
    n_steps_aligned: int = 0
    # Steps whose KL was undefined (infinite) and therefore left out of
    # `mean_kl_divergence`. Kept so a mean over 3 of 58 steps is visibly that.
    n_undefined_kl_steps: int = 0
    mean_js_divergence: Optional[float]
    mean_kl_divergence: Optional[float]  # over the steps where KL is defined
    mean_entropy_delta: Optional[float]
    # Scalar: mean perturbed entropy - mean baseline entropy. None when either
    # side has no steps to average, which is the one case where a difference of
    # two means is not a difference of anything.
    output_entropy_delta: Optional[float]
    mean_nucleus_stability_p90: Optional[float]  # mean nucleus_overlap_p90 across steps


# ---------------------------------------------------------------------------
# Core divergence functions
# ---------------------------------------------------------------------------


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions (log base 2).

    JSD(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5*(p+q).

    Both *p* and *q* must be non-negative and normalised to sum to 1.
    Result is in [0, 1] when using log base 2.
    Uses scipy.special.rel_entr for numerical stability.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    # rel_entr(a, b) = a * log(a/b), 0 if a == 0, inf if b == 0 and a > 0
    kl_pm = np.sum(rel_entr(p, m)) / math.log(2)
    kl_qm = np.sum(rel_entr(q, m)) / math.log(2)
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence KL(p||q) in bits.

    Returns inf if q has zero mass where p > 0.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    result = np.sum(rel_entr(p, q)) / math.log(2)
    return float(result)


# ---------------------------------------------------------------------------
# Generalized (n-way) Jensen-Shannon divergence and centroid
# ---------------------------------------------------------------------------
#
# Perturbation-field helpers. These operate on a *set* of distributions that
# already share a common support (row-stochastic matrix), not on a privileged
# baseline<->variant pair. They are pure functions over in-memory arrays —
# nothing here persists a distribution. Compute-and-discard is the reason:
# callers derive scalars from the centroid and let the distributions fall out
# of scope. Never write the `dists` argument, or the centroid, to any
# artifact.


def js_centroid(dists: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Jensen-Shannon centroid (the mixture mean) of a set of distributions.

    ``M = Σ wᵢ Pᵢ`` — the distribution that minimises ``Σ wᵢ KL(Pᵢ ‖ M)``, i.e.
    the anchor the generalized JSD below is measured against. This is the
    standard cheap centroid; the exact iterative JS-centroid (Nielsen) is not
    needed — the mixture mean already gives the field its reference point.

    Parameters
    ----------
    dists:
        ``(n, k)`` array of ``n`` distributions over a shared ``k``-support,
        each row non-negative and summing to 1.
    weights:
        Optional length-``n`` non-negative weights (need not be normalised);
        uniform when omitted. Supports the "weighted perturbations" case.
    """
    d = np.asarray(dists, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("dists must be a 2-D (n, k) array")
    n = d.shape[0]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / total
    return w @ d


def generalized_js_divergence(
    dists: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Generalized (n-way) Jensen-Shannon divergence of a distribution set, bits.

    ``GJSD = H(Σ wᵢ Pᵢ) − Σ wᵢ H(Pᵢ)``

    Equivalently the weighted mean divergence to the centroid,
    ``Σ wᵢ KL(Pᵢ ‖ M)`` with ``M = js_centroid(dists, weights)`` — the two forms
    agree to numerical precision (see the benchmark's decomposition check). This
    is the baseline-free counterpart to the pairwise ``mean_js_divergence`` in
    :class:`SensitivityMetrics`: it measures the *internal dispersion of the
    whole perturbation family*, not the shift from one privileged prompt
    realisation.

    Range ``[0, log₂ n]`` for uniform weights; ``0`` iff all rows are identical.
    Complexity is ``O(n·k)`` — the same order as the current pairwise-to-baseline
    Sensitivity, so no scaling cliff (the cost of a field is step alignment, not
    this divergence).
    """
    d = np.asarray(dists, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("dists must be a 2-D (n, k) array")
    n = d.shape[0]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / total

    m = w @ d  # mixture centroid

    def _entropy(p: np.ndarray) -> float:
        pc = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(np.where(p > 0, p * np.log2(pc), 0.0)))

    h_mixture = _entropy(m)
    mean_component_entropy = float(sum(wi * _entropy(d[i]) for i, wi in enumerate(w)))
    # Clamp tiny negatives from floating-point noise; GJSD is non-negative.
    return max(0.0, h_mixture - mean_component_entropy)


# ---------------------------------------------------------------------------
# Per-step computation
# ---------------------------------------------------------------------------


def _normalize(probs: np.ndarray) -> np.ndarray:
    total = probs.sum()
    if total <= 0:
        n = len(probs)
        return np.full(n, 1.0 / n) if n > 0 else probs
    return probs / total


def _step_entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits of a (normalised) distribution."""
    p = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(p * np.log2(p)))


def _nucleus_token_ids(token_ids: list[int], probs: list[float], p: float = 0.9) -> set[int]:
    """Return the set of token IDs in the top-p nucleus.

    Pairs token_ids with probs, sorts descending by prob, and collects IDs
    until cumulative mass >= p.  If the top-K list doesn't reach p (truncated
    distribution), all K IDs are returned.
    """
    if not token_ids or not probs:
        return set()
    pairs = sorted(zip(probs, token_ids), reverse=True)
    nucleus: set[int] = set()
    cumulative = 0.0
    for prob, tid in pairs:
        nucleus.add(tid)
        cumulative += prob
        if cumulative >= p:
            break
    return nucleus


def compute_step_sensitivity(
    baseline_step: StepRecord,
    perturbed_step: StepRecord,
) -> StepSensitivity:
    """Compute per-step sensitivity between baseline and perturbed top-K distributions.

    We build distributions over the *union* of token IDs from both steps' top-K lists.
    Tokens absent from a step receive probability 0 before normalisation.

    This is an approximation: tail tokens outside both top-K sets are invisible.
    """
    # Build maps from token_id → raw prob
    base_map: dict[int, float] = {e.token_id: e.prob for e in baseline_step.topk}
    pert_map: dict[int, float] = {e.token_id: e.prob for e in perturbed_step.topk}

    all_ids = sorted(set(base_map) | set(pert_map))

    base_probs = np.array([base_map.get(tid, 0.0) for tid in all_ids], dtype=np.float64)
    pert_probs = np.array([pert_map.get(tid, 0.0) for tid in all_ids], dtype=np.float64)

    base_probs = _normalize(base_probs)
    pert_probs = _normalize(pert_probs)

    jsd = js_divergence(base_probs, pert_probs)
    kld = kl_divergence(base_probs, pert_probs)
    entropy_delta = _step_entropy(pert_probs) - _step_entropy(base_probs)

    # Undefined, not enormous. KL is infinite when the perturbed distribution
    # places no mass where the baseline does, which is a statement about the
    # pair's supports; null carries it and a number does not.
    kld_value: float | None = kld if math.isfinite(kld) else None

    # Nucleus stability: set-based complement to JSD.
    # JSD measures mass shift; nucleus overlap measures whether the *viable token set*
    # changed — even a small mass shuffle near the threshold can flip nucleus membership.
    base_ids = [e.token_id for e in baseline_step.topk]
    base_ps = [e.prob for e in baseline_step.topk]
    pert_ids = [e.token_id for e in perturbed_step.topk]
    pert_ps = [e.prob for e in perturbed_step.topk]

    base_nucleus = _nucleus_token_ids(base_ids, base_ps, p=0.90)
    pert_nucleus = _nucleus_token_ids(pert_ids, pert_ps, p=0.90)

    if base_nucleus:
        nucleus_overlap = float(len(base_nucleus & pert_nucleus) / len(base_nucleus))
    else:
        # No baseline nucleus, so no set to take a fraction of. This was 1.0
        # — "define as stable" — which is the top of the range and reads as
        # perfect agreement rather than as no comparison.
        nucleus_overlap = None

    # Ranked probabilities (descending) for visualization — no token strings needed
    baseline_topk_probs = sorted([e.prob for e in baseline_step.topk], reverse=True)
    perturbed_topk_probs = sorted([e.prob for e in perturbed_step.topk], reverse=True)

    return StepSensitivity(
        step=baseline_step.step,
        js_divergence=jsd,
        kl_divergence=kld_value,
        entropy_delta=entropy_delta,
        nucleus_overlap_p90=nucleus_overlap,
        baseline_topk_probs=baseline_topk_probs,
        perturbed_topk_probs=perturbed_topk_probs,
    )


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------


def compute_sensitivity_metrics(
    baseline_trace: OutputSideTrace,
    perturbed_trace: OutputSideTrace,
    perturbed_prompt: str,
    generator_name: str,
) -> SensitivityMetrics:
    """Compute sensitivity metrics between a baseline and perturbed output trace.

    Alignment is the shared prefix — `min(len(baseline), len(variant))` — and
    how long that prefix ran is recorded rather than left to be inferred. It is
    routinely shorter than the baseline: 215 variants in the published corpus
    aligned over fewer steps than their baseline ran, the worst covering 6 of
    64. Each still contributes one equally-weighted number to the run's
    perturbation response, which is a defensible reading of the measurement's
    own definition (`mean_v [ mean_j [ JSD ] ]` — the variant is the unit of
    observation) but only while a reader can see the difference.
    """
    n_steps = min(len(baseline_trace.steps), len(perturbed_trace.steps))

    step_sensitivities: list[StepSensitivity] = []
    for i in range(n_steps):
        ss = compute_step_sensitivity(baseline_trace.steps[i], perturbed_trace.steps[i])
        step_sensitivities.append(ss)

    baseline_len = len(baseline_trace.steps)
    if baseline_len and n_steps < baseline_len:
        logger.debug(
            "Variant %r aligned over %d of the baseline's %d steps (%.0f%%); its "
            "means describe that prefix.",
            perturbed_prompt, n_steps, baseline_len, 100 * n_steps / baseline_len,
        )

    def _mean(values: list) -> float | None:
        """Mean of the defined values, or None when there are none.

        The one rule this module got wrong. Every branch below used to fall
        back to a constant, so "no steps to average" and "averaged to zero"
        arrived downstream as the same number.
        """
        present = [v for v in values if v is not None]
        return float(np.mean(present)) if present else None

    kls = [s.kl_divergence for s in step_sensitivities]
    mean_js = _mean([s.js_divergence for s in step_sensitivities])
    mean_kl = _mean(kls)
    mean_entropy_delta = _mean([s.entropy_delta for s in step_sensitivities])
    mean_nucleus_stability = _mean([s.nucleus_overlap_p90 for s in step_sensitivities])

    base_h = baseline_trace.mean_step_entropy
    pert_h = perturbed_trace.mean_step_entropy
    output_entropy_delta = (
        float(pert_h - base_h) if base_h is not None and pert_h is not None else None
    )

    return SensitivityMetrics(
        perturbation_generator=generator_name,
        perturbed_prompt=perturbed_prompt,
        original_prompt=baseline_trace.prompt_text,
        step_sensitivities=step_sensitivities,
        n_steps_aligned=n_steps,
        n_undefined_kl_steps=sum(1 for k in kls if k is None),
        mean_js_divergence=mean_js,
        mean_kl_divergence=mean_kl,
        mean_entropy_delta=mean_entropy_delta,
        output_entropy_delta=output_entropy_delta,
        mean_nucleus_stability_p90=mean_nucleus_stability,
    )

