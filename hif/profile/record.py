"""The canonical machine record — how measurements are reported.

Wire format only: the record schema version, the run hash, the field-descriptor
projections, and ``signals_record()``, which assembles one JSONL line. What a
measurement *is* lives in hif/profile/registry.py; how its value is *taken*
lives in hif/profile/measure.py.

Derived scalars only. Nothing here reads or emits raw token distributions —
records built from these values are safe under the compute-and-discard
default. Raw-artifact persistence is a separate, explicit opt-in
(TraceabilityConfig / --trace) handled by the engine, never here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from hif.config import public_config_dict
from hif.profile.measure import measurements, prompt_measurement_block
from hif.profile.registry import MEASUREMENT_UNITS, SIGNAL_SET_VERSION

# Version of the machine-record shape emitted by signals_record(). Bump on
# any breaking change to record structure (field renames/removals). Additive
# keys do not require a bump.
#
# record-v2: the `signals`/`readings` split, the `normalized` and `levels`
# blocks, and `findings_levels` are gone; a single flat `measurements` block
# in natural units replaces them.
# record-v3: the per-record `units` block is opt-in (`--units`)
# rather than always present — it is identical for every record of a given
# schema_version and `hif schema` prints it on demand. The field-descriptor
# blocks are renamed to the names docs/MEASUREMENTS.md Part 4 gives them:
# `field` -> `perturbation_field`, `branch_field` -> `trajectory_branch_field`.
# record-v4: every measurement declares a SUBJECT (whose behaviour
# the number describes), and quantities whose subject on the active backend is
# `prompt-only` are no longer emitted inside `measurements` with a surrogate
# flag — they move to a separate top-level `prompt_measurements` block naming
# the reference model that produced them. See the "Subject" section below.
# record-v5: a `provenance` block carries which model actually filled
# each role in the run (teacher forcing, output distributions, attention
# analysis) plus the degradation flags, so a published profile carries the
# evidence behind its subject declarations rather than only the claim. The
# record path cross-checks every emitted measurement against it and refuses to
# emit a record that contradicts it (hif/profile/provenance.py). Absent — like
# any other absent block — on a profile built before the block existed.
# record-v6 (current): a `run_config` block carries the RESOLVED run
# configuration — the same dict `hif config show` prints, from the same
# serializer (hif.config.public_config_dict). Three measurements are
# comparisons against runs the tool constructs (perturbation variants,
# trajectory branches, exposure thresholds); before this block, two records
# that differed only in [perturbation] generators or distance_threshold were
# identical in shape and different in value, with nothing in either to say
# why. That is the provenance failure this record format exists to prevent,
# applied to procedure instead of model identity. Secrets are redacted, not
# omitted ("<redacted>" vs null distinguishes "authenticated" from "no key").
# Absent on a profile built before the block existed.
# record-v7 (current): the `hash` covers the run's STAGE BUDGET as
# well as (model, prompt, seed). It did not, so a `--lite` run and a full run
# of the same prompt — six measurements vs two — shared an identifier, and the
# hash could neither dedupe a corpus nor answer "did this run actually do the
# perturbation stage?". The record shape is unchanged; the VALUE of `hash`
# changes for every run, which is why this is a version bump and not a silent
# fix: a consumer keying on the hash needs to know which function produced it.
# See profile_hash() / stage_budget() below for what is covered and what is
# deliberately not.
RECORD_SCHEMA_VERSION = "record-v7"


def stage_budget(config) -> Optional[dict]:
    """Which measurement-producing stages the RESOLVED config permits.

    The projection of a RunConfig that decides which measurements a run can
    emit at all — as opposed to what values they take. Two runs that agree
    here produce the same measurement KEYS; two that differ produce different
    ones, and are therefore different runs however identical their prompt.

    Read off the resolved config, never off the flags that set it. `--lite`
    and `--acquisition` have no independent existence in a run: they are
    ceilings applied last in `hif/cli/_run.py::_resolve_run_config`, and they
    act by switching these fields off. Hashing the flag names instead would be
    wrong in both directions — a `--config-file` that disables the same stages
    would collide with a full run (the bug this exists to close), and `--lite`
    would be distinguished from a config file that did exactly the same work
    (a difference that is not one).

    Returns None when the config is unknown, which is not the same as a config
    with everything defaulted — see profile_hash().

    Deliberately NOT covered: knobs that move the numbers without changing
    which numbers exist (`top_k`, `max_new_tokens`, `temperature`,
    `rollout_steps`, `distance_threshold`, the embedder, the backend). Those
    are carried in full by the record's `run_config` block, and folding them in
    here would churn every identifier on a knob that answers a different
    question.
    """
    if config is None:
        return None
    p = config.perturbation
    return {
        # The perturbation stage feeds four of the six measurements. Zero
        # variants, no generators, and no authored file each mean it did not
        # run; elicit_variant_outputs splits it in half (the two input-side
        # readings survive, the two that read a variant continuation do not).
        "perturbation": {
            "n_variants": p.n_variants,
            "generators": list(p.generators),
            "elicit_variant_outputs": p.elicit_variant_outputs,
            "variants_file": (
                str(p.variants_file) if p.variants_file is not None else None
            ),
        },
        "trajectory_branches": config.trajectory.n_branches,
        "semantic": config.semantic.enabled,
        "exposure": config.exposure.enabled,
        "semantic_field": config.semantic_field.enabled,
        "attention": config.attention.enabled,
        # Gates output_nucleus_entropy_bits: absent entirely when None.
        "entropy_percentile": config.generation.entropy_percentile,
    }


def profile_hash(model_name: str, prompt: str, seed: int, config) -> str:
    """The run identifier: (model, prompt, seed, stage budget).

    `config` is REQUIRED and takes the resolved RunConfig — pass None only
    where the budget is genuinely unknown (a profile built before it was
    carried). It is not optional-with-a-default on purpose: a call site that
    forgot it would silently reproduce the collision this signature exists to
    fix, and a TypeError is the cheaper failure.

    A None config hashes the legacy key, so a budget-less profile keeps the
    identifier it always had. That is a distinct claim from "the budget was
    the defaults", and hashes to a distinct value.
    """
    key = f"{model_name}|{prompt}|{seed}"
    budget = stage_budget(config)
    if budget is not None:
        key += "|" + json.dumps(budget, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Field / branch / semantic-field descriptor scalars
# ---------------------------------------------------------------------------


def field_scalars(profile) -> Optional[dict]:
    """Per-profile perturbation-field descriptors, or None when absent.

    Derived scalars only — the raw distributions the field was computed from
    were discarded inside build_profile (unless the run opted into
    traceability, in which case the raw traces live on the profile artifact,
    not here)."""
    f = getattr(profile.metrics, "field", None)
    return f.model_dump() if f is not None else None


def branch_field_scalars(profile) -> Optional[dict]:
    """Per-profile trajectory branch-field descriptors, or None when absent
    (< 2 branches, or a skipped/degenerate trajectory path)."""
    bf = getattr(profile.trajectory, "branch_field", None)
    return bf.model_dump() if bf is not None else None


def _attr(block, name):
    """Read one field off a block that may be a model OR a plain dict.

    `semantic_field`, `exposure` and `attention_capture` are typed
    `Optional[Any]` in the schema, so a profile built in memory carries a
    model there while the same profile loaded back from its own JSON carries
    a dict. Attribute access answers on the first and raises on the second,
    which made `signals_record()` crash on any round-tripped artifact with a
    populated field — and the earlier getattr-with-default form was worse
    still: it answered None and fabricated an ABSENCE, which is the one thing
    the absence rules exist to prevent. Read both shapes explicitly.
    """
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def semantic_field_scalars(profile) -> Optional[dict]:
    """Per-profile within-generation semantic-field (Veer) summary scalars,
    or None when absent (< 2 generation steps, or the instrument disabled)."""
    sf = getattr(profile, "semantic_field", None)
    if sf is None:
        return None
    return {"mean_veer": _attr(sf, "mean_veer"), "max_veer": _attr(sf, "max_veer"),
            "mean_deformation": _attr(sf, "mean_deformation"),
            "n_steps": _attr(sf, "n_steps")}


# ---------------------------------------------------------------------------
# The canonical machine record
# ---------------------------------------------------------------------------


def signals_record(
    profile,
    *,
    model_name: str,
    backend: str,
    regime: str,
    seed: int,
    prompt: str,
    latency: Optional[dict] = None,
    trace_path: Optional[str] = None,
    extras: Optional[dict] = None,
    include_units: bool = False,
) -> dict:
    """Build the canonical measurement record for one profiled prompt.

    This is what `--json` prints for a single profile run and what each JSONL
    line of `hif batch` contains. Derived values only — no raw distributions,
    no token alternatives: a record is a line in a stream, read row by row
    across a workload, and per-step distributions would bury the readings it
    exists to carry. Those live on the artifact, which `trace_path` points at.
    The model's generated output TEXT is included — it's the response the
    caller already has.

    Round-trip rule: every value here is the same number the terminal table
    displays, sourced from the same function.

    Raises
    ------
    ProvenanceMismatch
        When a measurement's declared subject contradicts what the run
        actually did. See hif/profile/provenance.py::check_provenance for why
        this ends the run rather than warning.
    """
    from hif.profile.provenance import ProvenanceMismatch, check_provenance

    f = profile.findings

    output_text = "".join(s.selected_token_str for s in profile.output_side.steps)

    # The contract check: every emitted measurement's declared subject against
    # what the run actually did. A mismatch means this record would attribute a
    # number to the wrong model, so no record is produced at all.
    violations = check_provenance(profile)
    if violations:
        raise ProvenanceMismatch(
            "measurement subjects contradict what the run recorded:\n  - "
            + "\n  - ".join(violations)
        )
    provenance = getattr(profile, "provenance", None)

    # Quantities whose subject on this backend is the prompt rather than the
    # target are reported in their own block, never inside `measurements`.
    # The block is omitted when nothing falls into it, so a record from a
    # backend where every quantity is target-side is byte-identical in shape
    # to one with no such block at all.
    prompt_block = prompt_measurement_block(profile)

    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "signal_set_version": SIGNAL_SET_VERSION,
        # Covers the stage budget as well as (model, prompt, seed), so two
        # runs whose measurement sets differ cannot share it. The budget comes
        # off the profile's own resolved config — the same one `run_config`
        # below serializes — not off any flag the caller remembers passing.
        "hash": profile_hash(
            model_name, prompt, seed, getattr(profile, "config", None)
        ),
        "model": model_name,
        "backend": backend,
        "regime": regime,
        "seed": seed,
        # Every measurement OF THIS MODEL in its natural unit. Absent
        # measurements are omitted. See MEASUREMENT_REGISTRY for what each
        # quantity and unit means, and which subject it has.
        "measurements": measurements(profile),
        **({"prompt_measurements": prompt_block} if prompt_block else {}),

        # The resolved configuration this run executed — same dict, same
        # serializer as `hif config show`, so what was confirmed before the
        # run is what the record attests after it. Three measurements are
        # comparisons against constructed runs; without this block their
        # numbers are not reproducible from the record alone. Omitted (never
        # guessed) when the profile predates the block.
        **(
            {"run_config": public_config_dict(profile.config)}
            if getattr(profile, "config", None) is not None
            else {}
        ),

        # Part 4 of docs/MEASUREMENTS.md — behaviour as a region rather than a
        # point. Named as the docs name them.
        "perturbation_field": field_scalars(profile),
        "trajectory_branch_field": branch_field_scalars(profile),
        "semantic_field": semantic_field_scalars(profile),
        "surrogate": {
            "input_side": f.surrogate_model_name,
            "output_distribution": getattr(
                f, "output_distribution_surrogate_name", None
            ),
        },
        # What actually ran, per role — the evidence behind every subject
        # above. Omitted (never emitted empty or guessed) on a profile built
        # before the block existed: an unchecked record must not look like a
        # checked one.
        **(
            {"provenance": provenance.model_dump()}
            if provenance is not None
            else {}
        ),
        "output_text": output_text,
        "output_tokens": len(profile.output_side.generated_ids),
        "input_tokens": len(profile.input_side.prompt_token_ids),
    }
    # Units are constant per signal_set_version and identical on every record,
    # so they are opt-in (`--units`) rather than repeated on every JSONL line.
    # `hif schema` prints them for every measurement without running a model.
    if include_units:
        keyed = list(record["measurements"]) + list(
            (prompt_block or {}).get("values", {})
        )
        record["units"] = {
            k: MEASUREMENT_UNITS[k] for k in keyed if k in MEASUREMENT_UNITS
        }
    if latency:
        record["latency"] = {k: round(v, 6) for k, v in latency.items()}
    if trace_path:
        record["trace_path"] = trace_path
    if extras:
        record.update(extras)
    return record
