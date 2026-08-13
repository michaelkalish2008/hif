"""A run where the target generated nothing publishes no output-side number.

The rule the whole record rests on is absence-not-zero, and the case that keeps
defeating it is the empty output side — because an empty series is the one
input on which every aggregate still returns a number. `mean([])` guarded to
0.0, `JSD` over no steps, a slope through no points, a cosine against the
embedding of `""`: each is arithmetic that completes, and none of them is a
measurement.

`io_correlation_r` was cut in hif-v4 for exactly this ("on a run with no output
steps it published a measured 0.0 correlation against a fabricated series").
`output_distributions_unusable` was then written to catch the empty case for
the distribution rows. It did — and `io_cosine_similarity` survived it anyway,
because that row reads output TEXT rather than distributions and so is
deliberately exempt from the distribution gate.

The consequence is in the published corpus. gpt-5 returned zero tokens for two
of eight prompt regimes; both profiles carry `output_side.steps = []`, no
`output_entropy_bits`, no `perturbation_jsd_bits` — and
`io_cosine_similarity = 0.17` and `0.10`. Those numbers were real arithmetic
over sixteen (input, output) pairs whose first output was the empty string and
whose other fifteen were the PARAPHRASE VARIANTS' continuations: a measurement
of the variants, filed under the baseline's key, on a record whose subject is
the model that said nothing.

These tests are the guard, stated at both ends: the gate itself, and a full
`build_profile` run against a backend that returns nothing.
"""

from __future__ import annotations

import pytest

from hif.metrics.similarity import SimilarityMetrics, _similarity_trend
from hif.profile.builder import build_profile
from hif.profile.measure import measurements, prompt_measurements
from hif.profile.record import signals_record
from hif.profile.registry import MEASUREMENT_REGISTRY
from hif.viz.registry import SIGNALS_BY_MEASUREMENT

from mock_backends import (
    TIER_NO_OUTPUT,
    TextHashEmbedder,
    alpha_model,
    contract_config,
    install_attention_analyzer,
    install_perturbation_generator,
)
from profile_helpers import _make_output_trace, _make_profile

PROMPT = "Explain why the sky appears blue."

# Every row whose value is read off the continuation the target produced. Taken
# from the registry rather than listed here, so a row added later is covered
# the moment it declares the flag — and a row that needs it and forgets to
# declare it is caught by test_every_output_side_row_declares_it below.
OUTPUT_SIDE_KEYS = sorted(
    m.key for m in MEASUREMENT_REGISTRY if m.needs_generated_output
)


@pytest.fixture(autouse=True)
def _offline_stages(monkeypatch):
    install_perturbation_generator(monkeypatch)
    install_attention_analyzer(monkeypatch)


# ---------------------------------------------------------------------------
# The gate, in isolation
# ---------------------------------------------------------------------------


def _empty_output_profile() -> "object":
    """A profile identical to a good one except that nothing was generated.

    Built by emptying the output side of a complete profile rather than by
    omitting the similarity block, because that is the shape the bug had: the
    similarity stage RAN, over pairs it should never have been given, and
    returned a number that looks like every other number in the record.
    """
    p = _make_profile()
    p.output_side = _make_output_trace(n_steps=0, mean_entropy=None)
    p.metrics.similarity = SimilarityMetrics(
        # The literal values from ../ai-interpretability/public/data/gpt5/
        # healthcare_advice.json, profile schema 0.10.0.
        input_sim=0.7555235028266907,
        output_sim=0.521162748336792,
        io_sim=0.1725561092607677,
        io_ratio=0.6898034890866146,
        trend=None,
        n_pairs=16,
    )
    return p


def test_no_output_side_measurement_survives_an_empty_generation():
    p = _empty_output_profile()
    published = set(measurements(p)) | set(prompt_measurements(p))

    leaked = sorted(set(OUTPUT_SIDE_KEYS) & published)
    assert not leaked, (
        f"{leaked} published for a run with output_side.steps == []. "
        "Every one of these reads the continuation the target produced, and "
        "this run produced none."
    )


def test_io_cosine_similarity_specifically_is_absent():
    """Named on its own because it is the one that got through.

    The other output-side rows are withheld by the distribution gate, which
    treats "no steps" as unusable. This row is not a distribution row, so
    nothing above it was watching.
    """
    p = _empty_output_profile()
    assert "io_cosine_similarity" not in measurements(p)
    assert "io_cosine_similarity" not in prompt_measurements(p)
    # And the value it would have published is still sitting there, unchanged
    # — the gate is what withholds it, not a missing input.
    assert p.metrics.similarity.io_sim == pytest.approx(0.1725561092607677)


def test_the_same_rows_are_published_when_output_exists():
    """The gate must be about the empty case and nothing else.

    A guard that withheld these rows generally would pass the test above while
    silently emptying every good record — which is the failure mode of every
    absence rule written in a hurry.
    """
    p = _make_profile()
    assert "io_cosine_similarity" not in measurements(p)  # no similarity block
    p.metrics.similarity = SimilarityMetrics(
        input_sim=0.8, output_sim=0.7, io_sim=0.31, io_ratio=0.875,
        trend=0.004, n_pairs=16,
    )
    assert measurements(p)["io_cosine_similarity"] == pytest.approx(0.31)


def test_every_output_side_row_declares_it():
    """`needs_generated_output` is the claim; this checks it is not empty.

    The flag exists to be swept over in `_all_measured_values`. A sweep over a
    set nobody populated passes every test and guards nothing, which is how
    `needs_distribution_pair` came to be described in its own docstring as
    "currently redundant".
    """
    assert "io_cosine_similarity" in OUTPUT_SIDE_KEYS
    assert {"output_entropy_bits", "perturbation_jsd_bits"} <= set(OUTPUT_SIDE_KEYS)


# ---------------------------------------------------------------------------
# The fabricated zeros around it
# ---------------------------------------------------------------------------


def test_mean_step_entropy_is_absent_not_zero():
    """0.0 over no steps says "certain at every step", about zero steps."""
    trace = _make_output_trace(n_steps=0, mean_entropy=None)
    assert trace.mean_step_entropy is None


def test_similarity_trend_is_absent_below_two_points():
    """A line through fewer than two points is undefined, not flat."""
    assert _similarity_trend([]) is None
    from profile_helpers import _make_semantic_metrics

    assert _similarity_trend([_make_semantic_metrics()]) is None
    two = [_make_semantic_metrics(), _make_semantic_metrics()]
    assert _similarity_trend(two) is not None


# ---------------------------------------------------------------------------
# End to end, through a backend that returns nothing
# ---------------------------------------------------------------------------


def _no_output_profile():
    return build_profile(
        model=alpha_model(tier=TIER_NO_OUTPUT),
        prompt=PROMPT,
        regime="test",
        config=contract_config("openai"),
        embedder=TextHashEmbedder(),
        seed=42,
    )


def test_build_profile_over_an_empty_generation_publishes_no_output_side_row():
    profile = _no_output_profile()
    assert profile.output_side.steps == []
    assert profile.output_side.generated_ids == []

    published = set(measurements(profile)) | set(prompt_measurements(profile))
    leaked = sorted(set(OUTPUT_SIDE_KEYS) & published)
    assert not leaked, f"{leaked} published by a run that generated nothing"


def test_the_record_states_that_nothing_was_generated():
    """Absence with a stated reason, which is what the README promises.

    Without this the artifact says only `steps: []` and a shorter measurements
    block — a reader comparing eight regimes sees a sparser row, not a run that
    never happened.
    """
    profile = _no_output_profile()
    record = signals_record(
        profile,
        model_name=profile.model.name,
        backend="openai",
        regime="test",
        seed=42,
        prompt=PROMPT,
    )

    assert record["provenance"]["target_generated_no_output"] is True
    # And the flag it is NOT: a run that returned nothing did not return point
    # masses either, so the selected-only claim must stay False. It read False
    # on the gpt-5 empty runs too, which is correct and was also the whole
    # problem — nothing else in the block said the output side was empty.
    assert record["provenance"]["output_distribution_selected_only"] is False


def test_fabricated_zeros_are_absent_end_to_end():
    profile = _no_output_profile()
    assert profile.output_side.mean_step_entropy is None
    assert profile.findings.similarity_trend_slope is None
    # The diagnostic block on the same run. `prompt_output_cosine_distance` is
    # the loudest of these: 0.0 is the minimum of its range, so on a distance
    # it asserted the output was semantically identical to the prompt.
    assert profile.center.output_mean_entropy is None
    assert profile.center.entropy_ratio is None
    assert profile.center.prompt_output_cosine_distance is None
    # Every per-variant delta of two absent means is absent too.
    assert all(
        s.output_entropy_delta is None for s in profile.metrics.sensitivity
    )


def test_the_reports_render_an_empty_run_without_inventing_a_number(tmp_path):
    """Rendering is where a None turns back into a 0.0 if anyone is careless.

    Each of these blocks formatted its value with `:.4f`, which raises on None
    — so the only two ways out are a stated absence or a default, and the
    default is what this whole pass exists to remove.
    """
    from hif.profile.render_markdown import render_technical

    out = tmp_path / "report.md"
    render_technical(_no_output_profile(), out)
    text = out.read_text()

    assert "| Mean step entropy | absent |" in text
    assert "| Prompt/output cosine distance | absent |" in text
    assert "Similarity trend slope: absent" in text
    # And no row claims a measured zero for the rows that were withheld.
    assert "io_cosine_similarity" not in measurements(_no_output_profile())


# ---------------------------------------------------------------------------
# The per-variant layer underneath
# ---------------------------------------------------------------------------
#
# A variant can align no steps while the BASELINE is healthy — the variant
# generated nothing, or the two traces share no prefix. Six variants of
# `gpt5/legal_compliance` aligned zero steps against a 603-step baseline. Those
# fed a measured 0.0 into the run's perturbation response, and only a guard in
# `measure.py` written for an unrelated reason (`selected_only`) kept it off
# the page. That is the same one-guard-deep arrangement that let
# `io_cosine_similarity` through.


def _variant_traces(baseline_steps: int, variant_steps: int):
    from hif.hourglass.output_side import OutputSideTrace
    from profile_helpers import _make_output_trace

    base = _make_output_trace(n_steps=baseline_steps, mean_entropy=2.0)
    var: OutputSideTrace = _make_output_trace(n_steps=variant_steps, mean_entropy=2.0)
    return base, var.model_copy(update={"prompt_text": "variant"})


def test_a_variant_that_aligned_nothing_reports_absence_not_zero():
    from hif.metrics.sensitivity import compute_sensitivity_metrics

    base, var = _variant_traces(baseline_steps=8, variant_steps=0)
    s = compute_sensitivity_metrics(base, var, "variant", "synonym")
    assert s.n_steps_aligned == 0
    assert s.mean_js_divergence is None, (
        "a variant with no aligned steps has no divergence; 0.0 is "
        "indistinguishable from a variant the model answered identically"
    )


def test_absent_variants_are_excluded_from_the_aggregate_not_averaged_in():
    """`perturbation_jsd_bits` must be a mean over what was measured."""
    from hif.metrics.stability import compute_stability_metrics
    from hif.metrics.sensitivity import compute_sensitivity_metrics
    from profile_helpers import _make_input_analysis

    base, good = _variant_traces(baseline_steps=8, variant_steps=8)
    _, empty = _variant_traces(baseline_steps=8, variant_steps=0)
    results = [
        compute_sensitivity_metrics(base, good, "v1", "synonym"),
        compute_sensitivity_metrics(base, empty, "v2", "synonym"),
    ]
    inp = _make_input_analysis()
    resp = compute_stability_metrics(inp, [inp, inp], results)

    assert resp.n_perturbations == 2
    assert resp.n_perturbations_aligned == 1, "the exclusion must not be silent"
    # The mean is the surviving variant's own value, not half of it.
    assert resp.perturbation_jsd_bits == pytest.approx(results[0].mean_js_divergence)


def test_undefined_kl_is_null_not_a_sentinel():
    """1e9 was finite, so the `math.isfinite` filter above it caught nothing.

    Two guards in the same file, and the upstream one disarmed the downstream
    one: `compute_step_sensitivity` clamped an infinite KL to 1e9 "so the value
    round-trips through JSON", and `compute_sensitivity_metrics` then averaged
    "only over non-inf steps" — of which there were now none. 833 records
    across half the published corpus carry a mean around 9.65e8.
    """
    import math
    from hif.metrics.sensitivity import compute_step_sensitivity
    from hif.models.base import StepRecord, TopKEntry

    def _step(pairs):
        return StepRecord(
            step=0, selected_token_id=pairs[0][0], selected_token_str="x",
            topk=[
                TopKEntry(token_id=t, token_str=str(t), logit=math.log(p),
                          logprob=math.log(p), prob=p)
                for t, p in pairs
            ],
        )

    # Disjoint supports: the baseline puts mass where the variant puts none.
    ss = compute_step_sensitivity(_step([(1, 1.0)]), _step([(2, 1.0)]))
    assert ss.kl_divergence is None
    assert ss.kl_divergence != 1e9


def test_alignment_coverage_is_recorded():
    """A mean over 2 steps and a mean over 8 weigh the same; say which is which.

    `PerturbationField` has carried `n_steps_aligned` since 0.4.0 for this same
    alignment. The record feeding the headline measurement did not, and 215
    variants in the published corpus aligned short of their baseline — the
    worst covering 6 of 64.
    """
    from hif.metrics.sensitivity import compute_sensitivity_metrics

    base, short = _variant_traces(baseline_steps=8, variant_steps=2)
    s = compute_sensitivity_metrics(base, short, "variant", "synonym")
    assert s.n_steps_aligned == 2
    assert len(s.step_sensitivities) == 2
    assert s.mean_js_divergence is not None


def test_no_chart_draws_a_measurement_the_empty_run_withheld():
    """The chart/measurement equivalence, on the run that breaks it.

    Same contract as test_chart_measurement_gate.py, restated here because the
    empty output side is not one of that file's backend tiers: it is an
    outcome, not an access level. A rendered chart of a withheld quantity is
    read as evidence.
    """
    profile = _no_output_profile()
    published = set(measurements(profile)) | set(prompt_measurements(profile))

    disagreements = []
    for key, signal in SIGNALS_BY_MEASUREMENT.items():
        if not signal.draws_measurement:
            continue
        drawn = signal.available(profile) is None
        if drawn != (key in published):
            disagreements.append(
                f"  {signal.id}: chart {'draws' if drawn else 'declines'} it, "
                f"record {'publishes' if key in published else 'withholds'} {key}"
            )
    assert not disagreements, "\n".join(disagreements)
