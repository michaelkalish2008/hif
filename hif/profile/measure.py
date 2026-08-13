"""Taking a measurement — reading the registry's quantities off a profile.

Single source of truth for turning a profile object into the scalar
measurements hif reports. The CLI presentation layer (hif/cli.py) and the
SessionEngine record path (hif/engine.py) both build on these functions, so a
number shown in a terminal table and a number emitted in a machine record can
never diverge.

What a measurement *is* lives in hif/profile/registry.py; how the values are
assembled into the wire record lives in hif/profile/record.py. This module is
the middle step: one guard per registry key, deciding whether this run produced
evidence for that quantity, and splitting the result by subject.

Absence is the load-bearing rule here. A quantity the run produced no evidence
for is OMITTED, never pinned to 0.0 or 1.0, and never emitted under a caveat
flag when what it actually measured was something else.

Everything returned here is a derived scalar. Nothing in this module reads or
emits raw token distributions: a record is a set of readings, and the raw
material they were read from lives on the artifact (TraceabilityConfig /
--trace), written by the engine, never here.
"""

from __future__ import annotations

from typing import Optional

from hif.profile.registry import (
    MEASUREMENT_BY_KEY,
    MEASUREMENT_REGISTRY,
    SUBJECT_PROMPT_ONLY,
    run_subjects,
)


def _all_measured_values(p) -> dict[str, float]:
    """Every value this profile produced, before the subject split.

    Internal: callers want `measurements()` (about the target) or
    `prompt_measurements()` (about the prompt under a reference model). This
    function does not distinguish them, which is exactly the confusion the
    subject field exists to prevent.
    """
    from hif.hourglass.input_side import mean_surprisal_excess
    from hif.hourglass.output_side import output_distributions_unusable

    out: dict[str, float] = {}
    m = p.metrics
    st = m.stability

    steps = getattr(p.output_side, "steps", None) or []
    # Did the target generate anything at all? Separate from every question
    # about the QUALITY of what came back: a run with no continuation has no
    # output-side evidence of any kind, whatever the backend can normally do.
    no_generated_output = not steps

    # Did this backend return real distributions at generation time, or only
    # the selected token? The perturbation JSDs were computed from the RAW
    # baseline and variant traces (builder.py step 6, before any surrogate
    # recovery), and every one of those traces came off the same backend as
    # `output_side` — so this one check answers it for all of them.
    selected_only = output_distributions_unusable(steps)
    # Did the step-6b recovery rebuild a real candidate cloud from the target's
    # actual continuation? Without it, every "cloud" on a selected-only backend
    # is a single token, and every quantity read off the cloud answers a
    # question about a set of one rather than the question its key names.
    _f = getattr(p, "findings", None)
    point_mass_cloud = selected_only and not getattr(
        _f, "output_distribution_surrogate_name", None
    )

    # --- perturbation response (input side, output side, and their coupling)
    if st.input_entropy_shift_bits is not None:
        out["input_entropy_shift_bits"] = st.input_entropy_shift_bits
    if getattr(st, "input_entropy_std_bits", None) is not None:
        out["input_entropy_std_bits"] = st.input_entropy_std_bits
    # Absent, not degenerate: between two point masses the "divergence" is 0
    # when the selected tokens agree and 1 bit when they differ, which is a
    # token-disagreement rate and not what `perturbation_jsd_bits` promises.
    # See the registry row.
    if not selected_only:
        if st.perturbation_jsd_bits is not None:
            out["perturbation_jsd_bits"] = st.perturbation_jsd_bits
        elif m.sensitivity:
            # Fall back to the per-perturbation records when the aggregate is
            # absent — same quantity, same unit, and the same exclusion. A
            # variant that aligned no steps has no divergence to contribute;
            # averaging its absence in as a zero here would reintroduce
            # through the back door exactly what the aggregate now refuses.
            js = [
                s.mean_js_divergence
                for s in m.sensitivity
                if s.mean_js_divergence is not None
            ]
            if js:
                out["perturbation_jsd_bits"] = sum(js) / len(js)

    # --- input/output anchoring
    if m.similarity is not None:
        out["io_cosine_similarity"] = m.similarity.io_sim

    # --- prompt-side surprisal excess (the input-side instrument)
    mean_excess = mean_surprisal_excess(getattr(p.input_side, "positions", None) or [])
    if mean_excess is not None:
        out["prompt_surprisal_excess_bits"] = mean_excess


    # --- output distribution
    if m.distribution and not point_mass_cloud:
        ents = [d.entropy_bits for d in m.distribution]
        out["output_entropy_bits"] = sum(ents) / len(ents)

        # The percentile companion, present only when every step could answer.
        # A mean over the subset of steps whose top-K happened to reach the
        # threshold is an average of a self-selecting sample — peaked steps
        # reach it, flat ones do not — so it would report LOWER entropy the
        # more often the nucleus went unobserved. One missing step makes the
        # run's answer absent, which is the same rule the rest of the set
        # follows.
        nuc = [d.percentile_entropy_bits for d in m.distribution]
        if nuc and all(v is not None for v in nuc):
            out["output_nucleus_entropy_bits"] = sum(nuc) / len(nuc)

    # One enforcement of needs_distribution_pair, derived from the rows rather
    # than hand-written per quantity.
    #
    # One row declares it today, and the explicit gate above already covers
    # that row — so this sweep is currently redundant. It is kept because it is
    # the general form: the flag was once hand-enforced branch by branch, which
    # is how `io_correlation_r` came to publish a measured 0.0 correlation
    # between a real input series and a fabricated one before hif-v4 cut it.
    #
    # A row that says it needs a pair of real distributions is absent whenever
    # the run has none, and a second such row inherits that for free.
    if selected_only:
        for row in MEASUREMENT_REGISTRY:
            if row.needs_distribution_pair:
                out.pop(row.key, None)

    # The same enforcement for needs_generated_output, and the reason it is a
    # SECOND sweep rather than another clause of the first: the two flags
    # answer different questions, and one row proved it. `io_cosine_similarity`
    # reads output TEXT, so it survives a selected-only backend by design and
    # the sweep above deliberately leaves it alone — which meant nothing was
    # left to catch it on a run that produced no text at all. gpt-5 published
    # io_cosine_similarity = 0.17 for two prompt regimes it answered with zero
    # tokens: the mean ran over sixteen (input, output) pairs of which fifteen
    # were the perturbation variants' continuations, so the number was real
    # arithmetic about the paraphrases, filed under the baseline's key.
    #
    # `output_distributions_unusable` already treats "no steps" as unusable,
    # so the distribution rows were absent on those same runs. That is what
    # made the survivor hard to see: the record looked correctly sparse.
    if no_generated_output:
        for row in MEASUREMENT_REGISTRY:
            if row.needs_generated_output:
                out.pop(row.key, None)

    return out


def measurements(p) -> dict[str, float]:
    """The measurements OF THE TARGET MODEL this profile supports.

    Absent measurements are OMITTED, never pinned to 0.0 or 1.0: absent means
    the run produced no evidence for that quantity (a backend that cannot
    teacher-force, an analysis stage that did not run), which is a different
    statement from a measured zero.

    Absence extends to "measured something else". A quantity whose effective
    subject on this run is `prompt-only` never touched the target's data, so it
    is omitted here — not emitted with a surrogate flag — and reported in
    `prompt_measurements()` instead. A flag would say "a caveated number about
    this model"; only "this model produced no number" is true.
    """
    subjects = run_subjects(p)
    return {
        k: v
        for k, v in _all_measured_values(p).items()
        if subjects.get(k) != SUBJECT_PROMPT_ONLY
    }


def prompt_measurements(p) -> dict[str, float]:
    """The values whose subject on this run is the prompt, not the target.

    Useful and comparable in their own right — "how surprising is this prompt
    under a fixed reference model" is a real question, and the answer is
    comparable across targets precisely BECAUSE the target does not enter it.
    They are simply not measurements of the target, so they are reported
    separately rather than inside `measurements()`.
    """
    subjects = run_subjects(p)
    return {
        k: v
        for k, v in _all_measured_values(p).items()
        if subjects.get(k) == SUBJECT_PROMPT_ONLY
    }


def _text_analysis_encoder(p) -> Optional[str]:
    """Name of the encoder that read the prompt as text, if one ran.

    The analysis encoder is recorded on the attention map it produced, so this
    reports the model that actually ran rather than the configured default.
    """
    att = getattr(p, "attention_capture", None) or getattr(p, "attention", None)
    if att is None:
        return None
    try:
        if isinstance(att, dict):
            return (
                att.get("input_analysis", {})
                .get("attention_map", {})
                .get("analysis_model")
            )
        return att.input_analysis.attention_map.analysis_model
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return None


def _prompt_reference_model(key: str, p) -> Optional[str]:
    """Whose behaviour a prompt-only value describes."""
    m = MEASUREMENT_BY_KEY[key]
    if m.subject_under_surrogate == SUBJECT_PROMPT_ONLY:
        # Degraded to prompt-only because a teacher-forcing surrogate read the
        # prompt in the target's place.
        return getattr(getattr(p, "findings", None), "surrogate_model_name", None)
    # Statically prompt-only: a local encoder read the prompt text directly.
    return _text_analysis_encoder(p)


def prompt_measurement_block(p) -> Optional[dict]:
    """The record's `prompt_measurements` block, or None when it would be empty.

    Omitted rather than emitted empty, for the same reason an unmeasurable
    quantity is omitted from `measurements`: an empty block would assert that
    the run considered these quantities and found nothing, when in fact none
    was in play.
    """
    values = prompt_measurements(p)
    if not values:
        return None
    return {
        "subject": SUBJECT_PROMPT_ONLY,
        "about": (
            "These describe the PROMPT under the reference model named for "
            "each key, not the model named in this record. No data the target "
            "produced enters their computation, so they cannot vary with the "
            "target's behaviour. They are comparable across targets for "
            "exactly that reason."
        ),
        "reference_models": {
            k: _prompt_reference_model(k, p) for k in values
        },
        "values": values,
    }
