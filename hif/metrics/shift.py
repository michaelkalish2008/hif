"""Step-to-step output divergence, and the top-K overlap that bounds it.

CANONICAL IMPLEMENTATION. ``Shiftⱼ = JSD(Qⱼ₋₁, Qⱼ)`` — the Jensen-Shannon
divergence (bits, log base 2) between CONSECUTIVE generation steps' output
distributions, aligned by token id over the union of the two steps' stored
top-K candidates and renormalised. Within a single forward pass: a large value
marks an abrupt vocabulary pivot (the field of viable tokens reorganized
sharply), a small one marks smooth continuation.

This module owns the computation. ``hif/viz/signals/shift.py`` (the chart) and
``hif/profile/measure.py`` (the measurement) both import from here, so the
number a reader sees on a chart and the number in a machine record cannot
drift apart. Nothing else should reimplement it.

Resolution limit — why the overlap fraction ships beside the divergence
----------------------------------------------------------------------
JSD is computed over the STORED TOP-K supports, not the full vocabulary. Two
consecutive steps whose top-K sets are disjoint give exactly 1 bit no matter
how similar their true full-vocabulary distributions are: disjoint support
alone hits the ceiling. So the divergence cannot be read on its own. It is
reported alongside the mean top-K overlap between the same consecutive steps,
which says how much support the transitions actually shared and therefore how
much of the divergence the truncation could have manufactured. Low overlap
does not invalidate the number — a real gpt2 run shows per-step values ranging
from 0.10 to 1.00 at a median overlap near 0.08, so the quantity still
discriminates — but it does bound how much of the value is evidence.

This is the same shape as ``output_entropy_bits``, which is a lower bound under
truncation and is reported with that bound stated rather than withheld. What is
NOT reported is the case below, where truncation stops the computation being a
divergence between distributions at all.

Absence rule — point masses are not distributions
-------------------------------------------------
On a backend that returns only the selected token (Anthropic), every step's
top-K has length 1. JSD between two point masses is 0 when the tokens match and
1 bit when they differ: a token-disagreement indicator, not a divergence
between distributions. That is a DIFFERENT QUANTITY, so it is not reported
under this key — :func:`shift_summary` returns ``None`` and the measurement is
absent. The same rule governs ``perturbation_jsd_bits`` (see
``hif/profile/measure.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hif.hourglass.output_side import output_distribution_degenerate
from hif.metrics.sensitivity import js_divergence

LABEL = "Output step-to-step JSD (bits)"

# Reason strings, so the chart's "not available" panel and the measurement's
# absence agree on why.
NEEDS_TWO_STEPS = "Requires at least two generation steps with top-K distributions."
NEEDS_DISTRIBUTIONS = (
    "This backend returns only the selected token, so consecutive steps are "
    "point masses: the divergence between them is a token-disagreement "
    "indicator, not a divergence between distributions."
)


@dataclass(frozen=True)
class StepShift:
    """One transition, step j−1 → j."""

    step: int  # the step transitioned INTO
    jsd_bits: float
    topk_overlap_fraction: float


@dataclass(frozen=True)
class ShiftSummary:
    """Run-level summary of the per-transition trace.

    mean_jsd_bits          the output_step_jsd_bits measurement.
    mean_overlap_fraction  its companion — mean top-K overlap over the same
                           transitions, the resolution limit on the divergence.
    median_overlap_fraction
                           the robust form the chart banners on; kept here so
                           chart and record read one computation.
    n_transitions          how many step pairs the means are over.
    """

    mean_jsd_bits: float
    mean_overlap_fraction: float
    median_overlap_fraction: float
    n_transitions: int


def step_distributions(steps) -> list[dict[int, float]]:
    """Each step's stored top-K as a sparse ``{token_id: prob}`` map."""
    return [{e.token_id: e.prob for e in s.topk} for s in steps]


def shift_jsd_bits(p: dict[int, float], q: dict[int, float]) -> float:
    """JSD (bits) between two sparse top-K distributions.

    Aligned over the union of their token ids, renormalised (stored top-K mass
    sums to less than 1), and clamped to the [0, 1] bound the log-base-2 form
    guarantees. Returns 0.0 when either side carries no mass — no evidence of a
    shift, and the caller's absence rules handle the degenerate cases.
    """
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    ps, qs = pv.sum(), qv.sum()
    if ps <= 0 or qs <= 0:
        return 0.0
    return max(0.0, min(1.0, js_divergence(pv / ps, qv / qs)))


def topk_overlap_fraction(p: dict[int, float], q: dict[int, float]) -> float:
    """Jaccard overlap of two steps' top-K token-id sets, ``|∩| / |∪|``.

    The support-level fact that explains a saturated divergence: at 0 the two
    steps share no candidate at all and the JSD above is pinned at 1 bit by
    construction. Bounded to [0, 1]; 1.0 for two empty supports (nothing
    differs).
    """
    union = set(p) | set(q)
    if not union:
        return 1.0
    return len(set(p) & set(q)) / len(union)


def shift_trace(steps) -> list[StepShift]:
    """Per-transition (JSD, top-K overlap) for a run's output steps.

    Empty when the run has fewer than two steps or the steps carry no real
    distributions (see the module docstring's absence rule) — the caller
    reports absence rather than a fabricated series.
    """
    if len(steps) < 2 or output_distribution_degenerate(steps):
        return []
    dists = step_distributions(steps)
    return [
        StepShift(
            step=i,
            jsd_bits=shift_jsd_bits(dists[i - 1], dists[i]),
            topk_overlap_fraction=topk_overlap_fraction(dists[i - 1], dists[i]),
        )
        for i in range(1, len(dists))
    ]


def shift_summary(steps) -> ShiftSummary | None:
    """Run-level step JSD and its companion overlap, or ``None`` when absent.

    ``None`` — not 0.0 — whenever the run produced no evidence for the
    quantity: fewer than two steps, or a selected-only backend whose steps are
    point masses. A measured zero (two identical distributions) is a different
    statement and is returned as one.
    """
    trace = shift_trace(steps)
    if not trace:
        return None
    jsd = [t.jsd_bits for t in trace]
    overlap = [t.topk_overlap_fraction for t in trace]
    return ShiftSummary(
        mean_jsd_bits=float(np.mean(jsd)),
        mean_overlap_fraction=float(np.mean(overlap)),
        median_overlap_fraction=float(np.median(overlap)),
        n_transitions=len(trace),
    )


def unavailable_reason(steps) -> str | None:
    """Why Shift cannot be reported for these steps, or ``None`` when it can."""
    if len(steps) < 2:
        return NEEDS_TWO_STEPS
    if output_distribution_degenerate(steps):
        return NEEDS_DISTRIBUTIONS
    return None
