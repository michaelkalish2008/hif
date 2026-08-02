"""Unit tests for the perturbation-field prototype (docs/ARCHITECTURE.md
§ Field-model notes).

Covers the generalized-JSD / centroid primitives and the trace-level field
descriptors. All inputs are constructed — no data is read or written (the field
is compute-and-discard, and these tests must not depend on any raw distribution
on disk).
"""

import math

import numpy as np
import pytest

from hif.hourglass.output_side import OutputSideTrace
from hif.metrics.field import (
    BASELINE_MEMBER,
    PerturbationField,
    SubField,
    compute_field_deformation,
    compute_perturbation_field,
)
from hif.metrics.sensitivity import (
    generalized_js_divergence,
    js_centroid,
    js_divergence,
)
from hif.models.base import StepRecord, TopKEntry


# ---------------------------------------------------------------------------
# Generalized JSD primitives
# ---------------------------------------------------------------------------


def _rand_dist(rng, k):
    v = rng.random(k) ** 3
    return v / v.sum()


def test_gjsd_decomposition_identity():
    """GJSD equals the weighted mean KL to the mixture centroid."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        n, k = int(rng.integers(2, 8)), 40
        P = np.vstack([_rand_dist(rng, k) for _ in range(n)])
        w = rng.random(n)
        w /= w.sum()
        gjsd = generalized_js_divergence(P, w)
        M = js_centroid(P, w)
        kl_form = sum(wi * js_divergence(P[i], M) * 0 for i, wi in enumerate(w))  # noqa
        # KL(Pi || M) in bits, matching generalized_js_divergence's own base.
        def _kl(a, b):
            a = np.clip(a, 1e-12, 1.0)
            b = np.clip(b, 1e-12, 1.0)
            return float(np.sum(a * np.log2(a / b)))
        kl_form = sum(wi * _kl(P[i], M) for i, wi in enumerate(w))
        assert gjsd == pytest.approx(kl_form, abs=1e-9)


def test_gjsd_zero_for_identical_set():
    d = _rand_dist(np.random.default_rng(2), 25)
    P = np.vstack([d] * 5)
    assert generalized_js_divergence(P) == pytest.approx(0.0, abs=1e-12)


def test_gjsd_reaches_log2_n_for_disjoint_point_masses():
    for n in (2, 4, 8):
        assert generalized_js_divergence(np.eye(n)) == pytest.approx(math.log2(n), abs=1e-9)


def test_gjsd_weighted_rejects_nonpositive_weights():
    P = np.eye(3)
    with pytest.raises(ValueError):
        generalized_js_divergence(P, weights=np.zeros(3))


# ---------------------------------------------------------------------------
# Trace-level field
# ---------------------------------------------------------------------------


def _trace(mode_ids, jitter, rng, n_steps=10):
    steps = []
    for s in range(n_steps):
        probs = np.array([0.6, 0.25, 0.1, 0.05]) + rng.normal(0, jitter, 4)
        probs = np.clip(probs, 1e-3, None)
        probs /= probs.sum()
        topk = [
            TopKEntry(
                token_id=int(t), token_str=f"t{t}", prob=float(p),
                logit=float(np.log(p)), logprob=float(np.log(p)),
            )
            for t, p in zip(mode_ids, probs)
        ]
        steps.append(
            StepRecord(step=s, selected_token_id=int(mode_ids[0]),
                       selected_token_str=f"t{mode_ids[0]}", topk=topk)
        )
    return OutputSideTrace(
        steps=steps, input_ids=[1], generated_ids=list(mode_ids) * n_steps,
        prompt_text="p", model_name="synthetic", top_k=4,
        max_new_tokens=n_steps, seed=0, mean_step_entropy=1.0,
    )


def test_field_none_when_fewer_than_two_members():
    rng = np.random.default_rng(3)
    base = _trace([10, 11, 12, 13], 0.02, rng)
    assert compute_perturbation_field(base, []) is None


def test_field_descriptors_and_anisotropic_subfields():
    rng = np.random.default_rng(4)
    base = _trace([10, 11, 12, 13], 0.02, rng)
    variants = [("synonym", _trace([10, 11, 12, 14], 0.03, rng)) for _ in range(4)]
    variants += [("tone", _trace([10, 20, 21, 22], 0.15, rng)) for _ in range(4)]

    field = compute_perturbation_field(base, variants)
    assert field is not None
    assert field.n_members == 9
    assert field.n_steps_aligned == 10
    # Radii are non-negative and max ≥ mean by construction.
    assert field.max_radius >= field.mean_radius >= 0.0
    assert field.radius_variance >= 0.0
    # The baseline sentinel never leaks into a sub-field.
    gens = {sf.generator for sf in field.subfields}
    assert BASELINE_MEMBER not in gens
    assert gens == {"synonym", "tone"}
    # The wide class must dominate dispersion — anisotropy a scalar would hide.
    by_gen = {sf.generator: sf for sf in field.subfields}
    assert by_gen["tone"].dispersion > by_gen["synonym"].dispersion


def test_single_sample_class_has_no_subfield():
    """A generator with one member has undefined within-class variance → omitted."""
    rng = np.random.default_rng(5)
    base = _trace([10, 11, 12, 13], 0.02, rng)
    variants = [("only_one", _trace([10, 11, 12, 14], 0.03, rng))]
    field = compute_perturbation_field(base, variants)
    assert field is not None
    assert field.n_members == 2
    assert field.subfields == []


# ---------------------------------------------------------------------------
# Field deformation (distribution-space, shape change only)
# ---------------------------------------------------------------------------


def _field(mean_r, var, max_r, disp, subs=None):
    return PerturbationField(
        n_members=5, n_steps_aligned=10, field_dispersion=disp,
        mean_radius=mean_r, radius_variance=var, max_radius=max_r,
        subfields=subs or [],
    )


def test_field_deformation_zero_for_identical_field():
    f = _field(0.1, 0.001, 0.2, 0.3)
    d = compute_field_deformation(f, f)
    assert d.deformation == pytest.approx(0.0, abs=1e-12)


def test_field_deformation_grows_with_shape_change_and_is_bounded():
    before = _field(0.10, 0.001, 0.20, 0.30)
    after = _field(0.30, 0.010, 0.50, 0.30)  # radii/variance widen; dispersion flat
    d = compute_field_deformation(before, after)
    assert 0.0 < d.deformation <= 1.0
    # The component that widened most (max_radius 0.2→0.5) shows a large change;
    # the unchanged dispersion contributes zero.
    assert d.d_field_dispersion == pytest.approx(0.0, abs=1e-12)
    assert d.d_max_radius > 0.5


def test_field_deformation_per_class_over_common_generators():
    before = _field(0.1, 0.001, 0.2, 0.3, subs=[
        SubField(generator="tone", n_members=4, dispersion=0.05, mean_radius=0.05, max_radius=0.06),
        SubField(generator="synonym", n_members=4, dispersion=0.01, mean_radius=0.01, max_radius=0.01),
    ])
    after = _field(0.1, 0.001, 0.2, 0.3, subs=[
        SubField(generator="tone", n_members=4, dispersion=0.20, mean_radius=0.2, max_radius=0.25),
        SubField(generator="synonym", n_members=4, dispersion=0.011, mean_radius=0.01, max_radius=0.01),
    ])
    d = compute_field_deformation(before, after)
    gens = [c.generator for c in d.per_class]
    assert gens == ["tone", "synonym"]  # sorted by change desc; tone deformed most
    assert d.per_class[0].dispersion_change > d.per_class[1].dispersion_change
