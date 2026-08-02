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

Privacy note: everything returned here is a derived scalar. Nothing in this
module reads or emits raw token distributions — records built from these
values are safe under the compute-and-discard default. Raw-artifact
persistence is a separate, explicit opt-in (TraceabilityConfig / --trace)
handled by the engine, never here.
"""

from __future__ import annotations

from typing import Optional

from hif.profile.registry import (
    MEASUREMENT_BY_KEY,
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
    from hif.hourglass.output_side import output_distribution_degenerate

    out: dict[str, float] = {}
    m = p.metrics
    st = m.stability

    # Did this backend return real distributions at generation time, or only
    # the selected token? The perturbation JSDs were computed from the RAW
    # baseline and variant traces (builder.py step 6, before any surrogate
    # recovery), and every one of those traces came off the same backend as
    # `output_side` — so this one check answers it for all of them.
    selected_only = output_distribution_degenerate(
        getattr(p.output_side, "steps", None) or []
    )
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
            # absent — same quantity, same unit.
            js = [s.mean_js_divergence for s in m.sensitivity]
            out["perturbation_jsd_bits"] = sum(js) / len(js)
    if st.input_output_correlation is not None:
        out["io_correlation_r"] = st.input_output_correlation

    # --- input/output anchoring
    if m.similarity is not None:
        out["io_cosine_similarity"] = m.similarity.io_sim

    # --- prompt-side surprisal excess (the input-side instrument)
    mean_excess = mean_surprisal_excess(getattr(p.input_side, "positions", None) or [])
    if mean_excess is not None:
        out["prompt_surprisal_excess_bits"] = mean_excess

    # --- candidate-cloud semantics
    if m.semantic and not point_mass_cloud:
        ces = [s.cluster_entropy for s in m.semantic]
        out["candidate_cluster_entropy_bits"] = sum(ces) / len(ces)

    # --- output distribution
    if m.distribution and not point_mass_cloud:
        ents = [d.entropy_bits for d in m.distribution]
        out["output_entropy_bits"] = sum(ents) / len(ents)
    if m.distribution and len(m.distribution) >= 2 and not point_mass_cloud:
        # Nucleus entropy (95% mass, renormalised) so the trace is comparable
        # across backends regardless of how many logprobs each exposes.
        nents = [d.nucleus_entropy_bits for d in m.distribution]
        deltas = [abs(nents[i] - nents[i - 1]) for i in range(1, len(nents))]
        out["output_entropy_step_delta_bits"] = sum(deltas) / len(deltas)

    # --- Shift ◆: step-to-step divergence of the output distribution, plus the
    # top-K overlap that bounds how much of it the truncation could have
    # manufactured. Both come from hif/metrics/shift.py, the same function the
    # Shift chart draws — one computation, so instrument and record agree.
    # shift_summary() returns None (never 0.0) on < 2 steps or a selected-only
    # backend, so both keys are simply omitted there.
    from hif.metrics.shift import shift_summary

    _shift = shift_summary(getattr(p.output_side, "steps", None) or [])
    if _shift is not None:
        out["output_step_jsd_bits"] = _shift.mean_jsd_bits
        out["output_step_topk_overlap_fraction"] = _shift.mean_overlap_fraction

    # --- semantic field (Veer): present only when semantic_field analysis ran.
    # Absent on a point-mass cloud: the "candidate cloud's centroid" would be
    # the selected token's own embedding, so the number would be the selected
    # token's path through embedding space — a different quantity under this
    # key's definition.
    sf = getattr(p, "semantic_field", None)
    if (
        sf is not None
        and getattr(sf, "mean_veer", None) is not None
        and not point_mass_cloud
    ):
        out["semantic_centroid_veer_cosine"] = sf.mean_veer

    # --- counterfactual exposure: present only when the divergence analysis
    # ran and found accessible alternatives. A point mass has none by
    # construction, which is absence of evidence, not a measured zero.
    exp = getattr(p, "exposure", None)
    if exp is not None and getattr(exp, "candidates", None) and not point_mass_cloud:
        out["counterfactual_exposure_fraction"] = exp.exposure

    # --- trajectory continuity, in its natural unit (mean pairwise cosine
    # between branch embeddings). Computed by the trajectory analysis; was
    # previously reduced to a score and never surfaced directly.
    traj = getattr(p, "trajectory", None)
    tc = getattr(traj, "trajectory_continuity", None) if traj is not None else None
    if tc is not None:
        out["branch_pairwise_cosine_similarity"] = float(tc)

    # --- attention-row entropy: present only when attention analysis ran.
    # Both sides in raw bits. The historical "Horizon" reading divided the
    # input-side row entropy by log2(prefix_len); that normaliser leaks
    # sequence-length metadata into a behavioural number, so it is gone.
    from hif.viz.signals._attention import get_attention_map, row_entropy_trace

    _, w_out = get_attention_map(p, "output")
    if w_out:
        tr = row_entropy_trace(w_out)
        if tr:
            out["attention_entropy_output_bits"] = sum(tr) / len(tr)
    _, w_in = get_attention_map(p, "input")
    if w_in:
        tr = row_entropy_trace(w_in)
        if tr:
            out["attention_entropy_input_bits"] = sum(tr) / len(tr)

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
