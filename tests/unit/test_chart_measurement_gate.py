"""A chart declines exactly the runs its measurement declines.

This is the fidelity contract in `hif/viz/registry.py` stated as an
executable equivalence rather than a promise:

    chart.available(p) is None
        <=>  chart.measurement_key in measurements(p) | prompt_measurements(p)

Both blocks on the right, because the question the gate answers is "did this
run publish the value", and a row whose effective subject is prompt-only under
`--surrogate` leaves `measurements()` for `prompt_measurements()` carrying its
number. That is a change of subject, not an absence, and a chart that declined
it would contradict the record beside it.

The two sides are decided in different modules by different code.
`hif/profile/measure.py` withholds a measurement when the evidence for it
does not exist — `output_entropy_bits` behind a real, non-degenerate output
distribution, `perturbation_jsd_bits` behind a pair of them. Each chart
answers separately, in its own `available()`, from whatever block it happens
to read. Nothing but agreement between two hand-written answers keeps a
dashboard from drawing a full trace for a quantity the record deliberately
refuses to publish — which is the "existed only as a chart" gap inverted, and
worse than the original, because a rendered chart of a withheld quantity is
read as evidence.

There was a test of this class. It lived in `tests/unit/test_shift.py` and
was deleted in hif-v4 along with the measurement it was written against,
taking the only guard on the invariant with it. This is that guard,
generalised over every keyed chart instead of one.
"""

from __future__ import annotations

import pytest

from hif.profile.builder import build_profile
from hif.profile.measure import measurements, prompt_measurements
from hif.viz.registry import SIGNALS_BY_MEASUREMENT

from mock_backends import (
    TextHashEmbedder,
    TIER_SELECTED_ONLY,
    alpha_model,
    contract_config,
    install_attention_analyzer,
    install_perturbation_generator,
    surrogate_model,
)

PROMPT = "Explain why the sky appears blue."


@pytest.fixture(autouse=True)
def _offline_stages(monkeypatch):
    install_perturbation_generator(monkeypatch)
    install_attention_analyzer(monkeypatch)


def _profile(model, backend: str, surrogate=None):
    return build_profile(
        model=model,
        prompt=PROMPT,
        regime="test",
        config=contract_config(backend),
        embedder=TextHashEmbedder(),
        seed=42,
        surrogate_model=surrogate,
    )


# The access range, plus the case that changes a row's subject rather than its
# presence. `hf` has full logprob access and should publish everything;
# `anthropic` is selected-only, which exercises every absence rule at once;
# `anthropic+surrogate` is the documented use of --surrogate, where the input
# rows are recovered by a reference model and so move to `prompt_measurements`
# with their values intact. Without that third case the equivalence below
# passes vacuously for `wager` — the only keyed chart whose measurement
# declares `subject_under_surrogate=prompt-only`.
CASES = [
    ("hf", lambda: alpha_model(), None),
    ("anthropic", lambda: alpha_model(tier=TIER_SELECTED_ONLY), None),
    (
        "anthropic+surrogate",
        lambda: alpha_model(tier=TIER_SELECTED_ONLY),
        surrogate_model,
    ),
]


@pytest.mark.parametrize(
    "backend,make_model,make_surrogate", CASES, ids=[c[0] for c in CASES]
)
def test_every_keyed_chart_declines_the_runs_its_measurement_declines(
    backend, make_model, make_surrogate
):
    profile = _profile(
        make_model(),
        backend.split("+")[0],
        surrogate=make_surrogate() if make_surrogate else None,
    )
    # Both blocks: the gate asks whether the run published the value at all,
    # and a row that moved to `prompt_measurements` under --surrogate changed
    # subject, not existence. Reading `measurements()` alone would assert that
    # `wager` MUST decline a surrogate run whose record carries its number.
    published = set(measurements(profile)) | set(prompt_measurements(profile))

    disagreements = []
    for key, signal in SIGNALS_BY_MEASUREMENT.items():
        # `stability` carries a key for --metric resolution but plots a
        # different series than its measurement reduces, so its availability
        # is its own. See the `draws_measurement` note in hif/viz/registry.py.
        if not signal.draws_measurement:
            continue
        drawn = signal.available(profile) is None
        claimed = key in published
        if drawn != claimed:
            disagreements.append(
                f"  {signal.id}: chart {'draws' if drawn else 'declines'} it, "
                f"record {'publishes' if claimed else 'withholds'} {key}"
            )

    assert not disagreements, (
        f"On backend {backend!r}, chart availability and measurement absence "
        f"disagree:\n" + "\n".join(disagreements) + "\n"
        "A chart drawn for a withheld measurement is read as evidence for a "
        "quantity the record refused to claim."
    )


def test_the_bridge_only_names_live_measurements():
    """No chart may be keyed on a measurement that no longer exists.

    `test_viz_measurement_bridge.py` walks this relation from the registry
    side, so a chart keyed on a cut row is invisible to it. hif-v4 removed ten
    rows and their charts by hand; this makes the next such pass mandatory
    rather than careful.
    """
    from hif.profile.registry import MEASUREMENT_BY_KEY

    stale = sorted(set(SIGNALS_BY_MEASUREMENT) - set(MEASUREMENT_BY_KEY))
    assert not stale, f"charts keyed on measurements that do not exist: {stale}"


# ---------------------------------------------------------------------------
# The record survives its own artifact
# ---------------------------------------------------------------------------
def test_signals_record_survives_a_json_roundtrip_with_populated_blocks():
    """A record built from a re-loaded artifact equals one built in memory.

    `semantic_field`, `exposure` and `attention_capture` are typed
    `Optional[Any]`, so they come back from JSON as plain dicts while an
    in-memory profile carries models. `signals_record()` read them by
    attribute, which raised on every round-tripped profile that had one
    populated — directly contradicting the claim in `hif/profile/registry.py`
    that justified the hif-v4 cut: "the artifact is the evidence." Evidence
    you cannot read back is not evidence.

    This generalises the deleted `test_rehydrated_dict_blocks_still_yield_
    their_measurements`, whose specimen (`measure.py::_field`) was retired
    with the rows it served, to the module where the hazard actually lives.
    """
    from hif.profile.record import semantic_field_scalars

    class _Model:
        mean_veer = 0.25
        max_veer = 0.31
        mean_deformation = 0.12
        n_steps = 4

    class _InMemory:
        semantic_field = _Model()

    class _Rehydrated:
        semantic_field = {
            "mean_veer": 0.25, "max_veer": 0.31,
            "mean_deformation": 0.12, "n_steps": 4,
        }

    live = semantic_field_scalars(_InMemory())
    loaded = semantic_field_scalars(_Rehydrated())

    assert live == loaded, (
        "the same semantic-field block read two ways gave different scalars"
    )
    # And specifically not fabricated absence: every value is really there.
    assert all(v is not None for v in loaded.values()), loaded


def test_the_gate_places_a_placeholder_rather_than_drawing(tmp_path):
    """The withheld case must reach the FILE, not just the predicate.

    Each chart module's `generate()` re-asks its own module-level
    `available()` to decide whether to draw the not-available placeholder, so
    gating the predicate alone would leave the index page correctly marking a
    chart unavailable while the chart file beside it was drawn in full.
    """
    from hif.viz.registry import SIGNALS_BY_ID

    profile = _profile(alpha_model(tier=TIER_SELECTED_ONLY), "anthropic")
    entropy = SIGNALS_BY_ID["output_entropy_bits"]

    reason = entropy.available(profile)
    assert reason is not None and "did not publish" in reason, reason

    written = entropy.generate(profile, tmp_path / "output_entropy_bits", formats=["html"])
    html = list(written.values())[0].read_text()
    assert "did not publish" in html, "generate() drew the chart anyway"


def test_the_diagnostic_blocks_the_cut_relies_on_still_reach_the_artifact():
    """The compensating claim behind hif-v4, asserted rather than promised.

    `hif/profile/registry.py` justifies removing ten rows by saying the
    stages behind them still record their blocks as raw material — "the SET
    is the claims; the artifact is the evidence." Nothing tested it. A later
    cleanup that removed the analysis stages outright would have passed the
    whole suite while quietly falsifying the argument the cut rests on.
    """
    import json

    # contract_config already runs the exposure stage (ExposureConfig
    # enabled=True), which is the point: these blocks are produced by the
    # ordinary configured run, not by a special path.
    profile = _profile(alpha_model(), "hf")

    # Whatever the run populated in memory must survive into its own JSON.
    dumped = json.loads(profile.model_dump_json())
    populated = [
        b for b in ("exposure", "attention_capture", "semantic_field")
        if getattr(profile, b, None) is not None
    ]
    # Guard against the assertion below going vacuous: if a future change
    # stops the stages populating anything, this test must fail loudly rather
    # than silently checking nothing.
    assert populated, (
        "no diagnostic block was populated, so the artifact carries no "
        "evidence for the ten rows hif-v4 cut — the justification for the "
        "cut no longer holds"
    )
    for block in populated:
        assert dumped.get(block) is not None, (
            f"{block} was populated on the profile but is absent from the "
            f"artifact — the cut's claim that the evidence still ships "
            f"does not hold for this block"
        )
