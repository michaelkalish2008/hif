"""Run provenance: what actually ran, checked against what the registry claims.

Every row in ``MEASUREMENT_REGISTRY`` declares a ``subject`` — whose behaviour
the number describes — and a ``surrogate_group`` naming the proxy that can
stand in. Those are assertions written by a contributor. This module records
the ground truth the pipeline already knows as it runs, and turns the
assertions into a contract that can be checked against it.

The four defects this exists to catch all had the same shape: a declaration
that the computation did not honour. Three ``surrogate_group`` values flagged a
proxy that never ran (or failed to flag one that did); a capability claim named
a backend feature the analyser never uses; a key kept its name on a backend
where the arithmetic had become a different quantity. Each was found by a human
reading code against prose, which does not scale past the people who already
know where the bodies are.

What is recorded
----------------
The actual model identity behind each ROLE in the run, plus the degradation
flags that drive the absence rules. Roles, not stages: "which model's
distributions are these numbers computed from" is the question a subject
declaration answers, and a role is the smallest thing that answers it.

This is deliberately not a parallel metadata system. It records what no other
part of the profile records — the *identity per role* — and nothing that can be
derived from the registry, the config, or the metric bundle.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from hif.models.capabilities import (
    ATTENTION_METRICS,
    NEEDS_DISTRIBUTION,
    NEEDS_TWO_DISTRIBUTIONS,
    TRAJECTORY_METRICS,
)
from hif.profile.measure import _all_measured_values
from hif.profile.registry import MEASUREMENT_BY_KEY, run_subjects


class ProvenanceMismatch(RuntimeError):
    """A measurement's declared subject contradicts what the run actually did.

    Raised from ``signals_record`` — see ``check_provenance`` for why this is a
    runtime error rather than a test-only assertion.
    """


class RunProvenance(BaseModel):
    """Which model filled each role in one run, and what degraded.

    Every field is an observation made while the pipeline ran, never an
    inference from the result. ``None`` means the role was never filled (the
    stage did not run), which is a different statement from "the target filled
    it".
    """

    # The model under analysis: the one that generated the output text. Every
    # other role is compared against this name to answer "did a proxy stand in
    # here?", so it is the anchor of the whole structure.
    generation_model: str
    # Who teacher-forced the PROMPT (builder step 2). The target when it can
    # teacher-force, the --surrogate otherwise, None when neither could and the
    # input-side stage produced nothing.
    input_teacher_forcing_model: Optional[str] = None
    # Whose per-step distributions the output-side measurements are computed
    # from (builder step 6b). The target's own generation trace, or the
    # --surrogate's when the recovery ran over the target's actual continuation.
    output_distribution_model: Optional[str] = None
    # Which encoder the attention analyser loaded (builder step 11c). None when
    # the stage did not run. This is never the target: the analyser is a
    # separate bidirectional encoder reading text as an object, which is why
    # the attention rows are available on every backend.
    attention_analysis_model: Optional[str] = None
    # The target backend returned only the selected token at generation time
    # (top-K of length 1 everywhere), so its raw per-step "distributions" are
    # point masses. Drives the absence rules for the divergence rows, which no
    # surrogate recovers.
    output_distribution_selected_only: bool = False
    # Trajectory branches were actually rolled out (builder step 5). False when
    # the stage was skipped, which is when branch quantities must be absent
    # rather than computed over an empty branch list.
    trajectory_analysis_ran: bool = False


def _role_model(measurement, prov: RunProvenance) -> Optional[str]:
    """Which model produced the data behind `measurement`, per provenance.

    ``None`` for a row whose ``surrogate_group`` is empty: no proxy can stand
    in there, so there is no role to look up and the declared subject cannot
    depend on one.
    """
    if measurement.surrogate_group == "input":
        return prov.input_teacher_forcing_model
    if measurement.surrogate_group == "output":
        return prov.output_distribution_model
    return None


def check_provenance(profile) -> list[str]:
    """Every emitted measurement's declared subject, checked against the run.

    Returns a list of violation messages — empty when the record's claims match
    what happened. A profile carrying no provenance (an artifact written before
    the block existed, or a hand-built test fixture) returns no violations:
    absent evidence is not a contradiction, and asserting one would be the same
    fabrication this module exists to prevent.

    Runtime, not test-only
    ----------------------
    ``signals_record`` raises on a non-empty result rather than logging it. The
    argument for the softer option — that a library which raises punishes a
    user for a maintainer's bug — does not apply here, because of what can
    actually trigger this check: the inputs are the registry's static
    declarations and the run's role identities. It cannot fire on an unusual
    prompt, an unlucky sample, or a slow backend. It fires when a registry row
    disagrees with the code, which is a defect in this package, present for
    every user on every run of that measurement, and deterministic.

    So the choice is not "raise on a rare edge case" versus "log it". It is
    "refuse to emit a record that misattributes a number" versus "emit it with
    a warning in a log the consumer of the published JSON will never see". The
    record is the product; a mislabelled subject is the failure mode with the
    highest cost to this project's credibility, and it is silent by nature. It
    is worth ending a run over.
    """
    prov = getattr(profile, "provenance", None)
    if prov is None:
        return []

    target = prov.generation_model
    subjects = run_subjects(profile)
    emitted = set(_all_measured_values(profile))
    problems: list[str] = []

    for key in sorted(emitted):
        m = MEASUREMENT_BY_KEY.get(key)
        if m is None:  # unregistered keys are a separate invariant's business
            continue
        producer = _role_model(m, prov)
        if m.surrogate_group and producer is None:
            problems.append(
                f"{key}: emitted, but no model filled the "
                f"{m.surrogate_group!r} role in this run — the value cannot "
                f"have a subject"
            )
            continue
        stood_in = bool(m.surrogate_group) and producer != target
        if stood_in and m.subject_under_surrogate is None:
            problems.append(
                f"{key}: {producer!r} stood in for {target!r} on the "
                f"{m.surrogate_group!r} side, but the row declares no "
                f"subject_under_surrogate — it claims {m.subject!r} for a "
                f"number the target did not produce"
            )
            continue
        expected = m.subject_under_surrogate if stood_in else m.subject
        declared = subjects.get(key)
        if declared != expected:
            problems.append(
                f"{key}: record declares subject {declared!r}, but provenance "
                f"says the {m.surrogate_group or 'own'} role was filled by "
                f"{producer or target!r} (target is {target!r}), which makes "
                f"it {expected!r}"
            )

    # Absence rules, checked against the same evidence. Each is a case where a
    # value is computable and is NOT the quantity the key names.
    if prov.output_distribution_selected_only:
        for key in sorted(emitted & NEEDS_TWO_DISTRIBUTIONS):
            problems.append(
                f"{key}: emitted on a selected-only run — between two point "
                f"masses this is a token-disagreement rate, not the divergence "
                f"the key names"
            )
        if prov.output_distribution_model == target:
            for key in sorted(emitted & NEEDS_DISTRIBUTION):
                problems.append(
                    f"{key}: emitted from the target's own point-mass steps "
                    f"with no surrogate recovery — the value is 0 by "
                    f"construction, not measured"
                )
    if not prov.trajectory_analysis_ran:
        for key in sorted(emitted & TRAJECTORY_METRICS):
            problems.append(
                f"{key}: emitted, but the trajectory stage never rolled out a "
                f"branch in this run"
            )
    if prov.attention_analysis_model is None:
        for key in sorted(emitted & ATTENTION_METRICS):
            problems.append(
                f"{key}: emitted, but no attention analysis encoder ran"
            )

    return problems
