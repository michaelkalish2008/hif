"""Perturbation-field descriptors: the model's behaviour as a *region*, not a point.


Where :mod:`hif.metrics.sensitivity` answers "how far does the model move from
its one canonical prompt realisation?" (pairwise-to-baseline JSD), this module
answers "what is the *shape* of the neighbourhood the model occupies under a
family of content-preserving perturbations?" — the centroid of that family and
the geometry of the cloud around it.

Privacy invariant (hard requirement — read before extending)
------------------------------------------------------------
Every value this module emits is a derived scalar. Compute-and-discard is the
DEFAULT: the per-step token distributions this module consumes are held only in
the caller's stack frame for the duration of the computation and are then
dropped — the same discipline the sensitivity path already follows. A top-k
distribution *with token identity* is reconstructable content, so by default it
must never reach an artifact, the ledger, or a sidecar. Do NOT add a field here
that stores a distribution, a centroid, or a token id — this module itself
NEVER stores distributions.

The single sanctioned exception is the traceability opt-in
(``TraceabilityConfig.enabled`` on RunConfig): when the operator explicitly
enables it, the builder persists the raw member traces on the PROFILE artifact
(``BehavioralRangeProfile.raw_traces`` — never in the field models emitted
here) so field descriptors, JS-centroids, translation, and branch fields can be
reconstructed from the artifact without re-running models. This module must
never be imported from a hosted request path (the companion platform repo
enforces this on its side); it is a scoring-time-only, sample-only
computation.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from hif.hourglass.output_side import OutputSideTrace
from hif.metrics.sensitivity import (
    generalized_js_divergence,
    js_centroid,
    js_divergence,
)

# Sentinel generator label for the unperturbed baseline member of the field.
BASELINE_MEMBER = "__baseline__"


# ---------------------------------------------------------------------------
# Data model (derived scalars only — never a distribution)
# ---------------------------------------------------------------------------


class SubField(BaseModel):
    """Field descriptors restricted to one perturbation class (generator).

    Estimable only when the class contributed ≥ 2 members (i.e. n_variants ≥ 2
    for that generator). With a single sample per class the within-class radius
    variance is undefined and this is omitted — the honest "can't measure
    deformation-per-class from one sample" case (the no-deformation-from-one-
    sample rule, docs/ARCHITECTURE.md § Field-model notes).
    """

    generator: str
    n_members: int
    dispersion: float       # mean-over-steps generalized JSD within the class
    mean_radius: float      # mean member→class-centroid JSD
    max_radius: float       # worst member→class-centroid JSD


class ClassDeformation(BaseModel):
    """Per-generator dispersion change between two fields (relative)."""

    generator: str
    dispersion_change: float   # |Δ dispersion| / max(|before|, |after|, eps)


class FieldDeformation(BaseModel):
    """Distribution-space deformation between two perturbation fields.

    Deformation ONLY — translation (centroid movement) is NOT recoverable from
    persisted fields, because the JS-centroid is discarded under the
    compute-and-discard rule. Measure translation in-session, while both
    centroids are still live. This consumes two derived fields (e.g. a baseline-phase
    and an event-phase field for the same arm/regime) and reports how the field's
    *shape* changed.

    Each component is a bounded relative change ``|Δx| / max(|before|, |after|,
    eps)`` in [0, 1]; ``deformation`` is their RMS.
    """

    deformation: float          # aggregate (RMS of the component changes)
    d_mean_radius: float
    d_radius_variance: float
    d_max_radius: float
    d_field_dispersion: float
    per_class: list[ClassDeformation] = []


class PerturbationField(BaseModel):
    """Derived geometry of the perturbation cloud around its centroid.

    All values are bits (JSD, log base 2) except counts. Nothing here identifies
    a token or retains a distribution.
    """

    n_members: int              # baseline + all variants that aligned
    n_steps_aligned: int        # shared-prefix length the field was measured over
    field_dispersion: float     # mean-over-steps generalized JSD of the whole set
    mean_radius: float          # mean member→centroid JSD (the "average radius")
    radius_variance: float      # variance of member radii (isotropy of the cloud)
    max_radius: float           # worst member→centroid JSD (the anomaly carrier)
    # Per-class sub-fields, keyed by generator — the anisotropy substrate.
    # Excludes the baseline member; only classes with ≥ 2 members appear.
    subfields: list[SubField] = []


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------


def _step_maps(trace: OutputSideTrace) -> list[dict[int, float]]:
    """Per-step {token_id: prob} maps for a trace (raw, unnormalised probs)."""
    return [{e.token_id: e.prob for e in step.topk} for step in trace.steps]


def _aligned_matrix(maps_at_step: list[dict[int, float]]) -> np.ndarray:
    """Stack member distributions at one step onto their union support.

    Tokens absent from a member get probability 0 before row-normalisation —
    the same union-support convention as compute_step_sensitivity, generalised
    from a pair to n members.
    """
    all_ids = sorted(set().union(*maps_at_step)) if maps_at_step else []
    if not all_ids:
        return np.zeros((len(maps_at_step), 0), dtype=np.float64)
    rows = []
    for m in maps_at_step:
        v = np.array([m.get(tid, 0.0) for tid in all_ids], dtype=np.float64)
        s = v.sum()
        rows.append(v / s if s > 0 else np.full(len(all_ids), 1.0 / len(all_ids)))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Field computation (compute-and-discard)
# ---------------------------------------------------------------------------


def compute_perturbation_field(
    baseline_trace: OutputSideTrace,
    variant_traces: list[tuple[str, OutputSideTrace]],
    weights: np.ndarray | None = None,
) -> PerturbationField | None:
    """Derive the perturbation-field descriptors for one prompt.

    Parameters
    ----------
    baseline_trace:
        Output trace of the unperturbed prompt — one member of the field.
    variant_traces:
        ``(generator_name, trace)`` for every perturbation variant. Traces are
        read for their per-step top-k and then discarded by the caller; this
        function retains nothing.
    weights:
        Optional per-member weights (baseline first, then variants in order) for
        the weighted-perturbation case. Uniform when omitted.

    Returns ``None`` when fewer than 2 members align (a field needs a cloud) or
    no output steps overlap.
    """
    members: list[OutputSideTrace] = [baseline_trace] + [t for _, t in variant_traces]
    generators: list[str] = [BASELINE_MEMBER] + [g for g, _ in variant_traces]
    n = len(members)
    if n < 2:
        return None

    n_steps = min(len(m.steps) for m in members)
    if n_steps == 0:
        return None

    maps_per_member = [_step_maps(m) for m in members]

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    else:
        w = None

    # Accumulate per-member radius (mean JSD to the whole-set centroid across
    # aligned steps) and the whole-set dispersion (generalized JSD per step).
    member_radius_acc = np.zeros(n, dtype=np.float64)
    dispersion_acc = 0.0
    for s in range(n_steps):
        stacked = _aligned_matrix([maps_per_member[i][s] for i in range(n)])
        if stacked.shape[1] == 0:
            continue
        centroid = js_centroid(stacked, w)
        dispersion_acc += generalized_js_divergence(stacked, w)
        for i in range(n):
            member_radius_acc[i] += js_divergence(stacked[i], centroid)

    member_radii = member_radius_acc / n_steps
    field_dispersion = dispersion_acc / n_steps

    subfields = _compute_subfields(generators, maps_per_member, n_steps)

    return PerturbationField(
        n_members=n,
        n_steps_aligned=n_steps,
        field_dispersion=float(field_dispersion),
        mean_radius=float(np.mean(member_radii)),
        radius_variance=float(np.var(member_radii)),
        max_radius=float(np.max(member_radii)),
        subfields=subfields,
    )


def _rel_change(before: float, after: float, eps: float = 1e-9) -> float:
    """Bounded relative change ``|Δ| / max(|before|, |after|, eps)`` in [0, 1]."""
    denom = max(abs(before), abs(after), eps)
    return abs(after - before) / denom


def compute_field_deformation(
    before: PerturbationField, after: PerturbationField
) -> FieldDeformation:
    """Distribution-space deformation between two fields (shape change only).

    Translation is intentionally absent — the centroid is not persisted, so
    centroid movement is unrecoverable here (see FieldDeformation docstring). The
    aggregate is the RMS of the per-descriptor relative changes.
    """
    d_mean_radius = _rel_change(before.mean_radius, after.mean_radius)
    d_radius_variance = _rel_change(before.radius_variance, after.radius_variance)
    d_max_radius = _rel_change(before.max_radius, after.max_radius)
    d_field_dispersion = _rel_change(before.field_dispersion, after.field_dispersion)

    components = [d_mean_radius, d_radius_variance, d_max_radius, d_field_dispersion]
    deformation = float(np.sqrt(np.mean(np.square(components))))

    # getattr so this also consumes a BranchField (no subfields) — the branch
    # cloud has no per-generator classes, so per_class is simply empty there.
    before_sub = {sf.generator: sf for sf in getattr(before, "subfields", [])}
    after_sub = {sf.generator: sf for sf in getattr(after, "subfields", [])}
    per_class = [
        ClassDeformation(
            generator=g,
            dispersion_change=_rel_change(
                before_sub[g].dispersion, after_sub[g].dispersion
            ),
        )
        for g in sorted(set(before_sub) & set(after_sub))
    ]
    per_class.sort(key=lambda c: c.dispersion_change, reverse=True)

    return FieldDeformation(
        deformation=deformation,
        d_mean_radius=d_mean_radius,
        d_radius_variance=d_radius_variance,
        d_max_radius=d_max_radius,
        d_field_dispersion=d_field_dispersion,
        per_class=per_class,
    )


def _compute_subfields(
    generators: list[str],
    maps_per_member: list[list[dict[int, float]]],
    n_steps: int,
) -> list[SubField]:
    """Per-generator (class) sub-fields — the deformation-per-class substrate.

    Groups members by generator (excluding the baseline), and for each class with
    ≥ 2 members computes the within-class centroid dispersion and radii. Classes
    with a single sample are skipped: their within-class variance is undefined.
    """
    by_gen: dict[str, list[int]] = {}
    for idx, g in enumerate(generators):
        if g == BASELINE_MEMBER:
            continue
        by_gen.setdefault(g, []).append(idx)

    out: list[SubField] = []
    for g, idxs in by_gen.items():
        if len(idxs) < 2:
            continue
        radius_acc = np.zeros(len(idxs), dtype=np.float64)
        dispersion_acc = 0.0
        for s in range(n_steps):
            stacked = _aligned_matrix([maps_per_member[i][s] for i in idxs])
            if stacked.shape[1] == 0:
                continue
            centroid = js_centroid(stacked)
            dispersion_acc += generalized_js_divergence(stacked)
            for j in range(len(idxs)):
                radius_acc[j] += js_divergence(stacked[j], centroid)
        radii = radius_acc / n_steps
        out.append(
            SubField(
                generator=g,
                n_members=len(idxs),
                dispersion=float(dispersion_acc / n_steps),
                mean_radius=float(np.mean(radii)),
                max_radius=float(np.max(radii)),
            )
        )
    out.sort(key=lambda sf: sf.dispersion, reverse=True)
    return out
