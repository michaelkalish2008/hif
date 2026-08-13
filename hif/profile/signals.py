"""Measurement extraction from a BehavioralRangeProfile — public façade.

This module is the stable import path. The implementation lives in three
files, split by the question each answers:

* ``hif/profile/registry.py`` — *what a measurement is*. The ontology
  (resolution, functional, subject), the ``Measurement`` row type, the
  ``MEASUREMENT_REGISTRY`` rows, the derived views over them, and subject
  resolution. This is the single extension point: adding a measurement means
  adding one row there.
* ``hif/profile/measure.py`` — *how a measurement is taken*. One guard per
  registry key deciding whether a run produced evidence for that quantity, and
  the split of the result by subject into ``measurements()`` (about the target)
  and ``prompt_measurements()`` (about the prompt under a reference model).
* ``hif/profile/record.py`` — *how measurements are reported*. The record
  schema version, the run hash, the field-descriptor projections, and
  ``signals_record()``.

Everything those modules define is re-exported here unchanged, so
``from hif.profile.signals import ...`` keeps working for every name it ever
carried. New code should import from the module that owns the name.
"""

from __future__ import annotations

from hif.profile.measure import (
    _all_measured_values,
    _prompt_reference_model,
    _text_analysis_encoder,
    measurements,
    prompt_measurement_block,
    prompt_measurements,
)
from hif.profile.record import (
    RECORD_SCHEMA_VERSION,
    branch_field_scalars,
    field_scalars,
    profile_hash,
    semantic_field_scalars,
    signals_record,
    stage_budget,
)
from hif.profile.registry import (
    FUNCTIONALS,
    MEASUREMENT_BY_KEY,
    MEASUREMENT_KEYS,
    MEASUREMENT_REGISTRY,
    MEASUREMENT_UNITS,
    MEASUREMENTS,
    RESOLUTIONS,
    SIGNAL_SET_VERSION,
    SUBJECT_LEGEND,
    SUBJECT_MIXED,
    SUBJECT_PROMPT_ONLY,
    SUBJECT_TARGET_DISTRIBUTION,
    SUBJECT_TARGET_OUTPUT_TEXT,
    SUBJECTS,
    Measurement,
    effective_subject,
    run_subjects,
)

__all__ = [
    # registry — what a measurement is
    "FUNCTIONALS",
    "MEASUREMENT_BY_KEY",
    "MEASUREMENT_KEYS",
    "MEASUREMENT_REGISTRY",
    "MEASUREMENT_UNITS",
    "MEASUREMENTS",
    "Measurement",
    "RESOLUTIONS",
    "SIGNAL_SET_VERSION",
    "SUBJECTS",
    "SUBJECT_LEGEND",
    "SUBJECT_MIXED",
    "SUBJECT_PROMPT_ONLY",
    "SUBJECT_TARGET_DISTRIBUTION",
    "SUBJECT_TARGET_OUTPUT_TEXT",
    "effective_subject",
    "run_subjects",
    # measure — how a measurement is taken
    "_all_measured_values",
    "_prompt_reference_model",
    "_text_analysis_encoder",
    "measurements",
    "prompt_measurement_block",
    "prompt_measurements",
    # record — how measurements are reported
    "RECORD_SCHEMA_VERSION",
    "branch_field_scalars",
    "field_scalars",
    "profile_hash",
    "semantic_field_scalars",
    "signals_record",
    "stage_budget",
]
