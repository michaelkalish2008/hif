"""Measurement extraction from a BehavioralRangeProfile.

Single source of truth for turning a profile object into the scalar
measurements hif reports. The CLI presentation layer (hif/cli.py) and the
SessionEngine record path (hif/engine.py) both build on these functions, so a
number shown in a terminal table and a number emitted in a machine record can
never diverge.

Natural units only
------------------
Every measurement is reported in the unit it is measured in — bits for
entropies and surprisals, dimensionless for correlations, cosine distance for
embedding displacement, a fraction for a count of steps. Key names carry the
unit.

Nothing here normalises an unbounded quantity into [0, 1], and nothing reports
``1 - x``. Three things were removed, and this docstring records why so they
do not come back:

* ``normalized`` block — unbounded quantities were divided by
  ``log2(vocab_size)``. That normaliser then surfaced as the strongest
  apparent "behavioural" feature in the study corpus (r = 0.980, constant
  within a model): tokenizer metadata masquerading as behaviour. Bounded
  scales also saturate, and bits are self-interpreting (4.9 bits is about the
  uncertainty of a uniform choice among ~30 tokens; "0.0178" is not
  checkable).
* ``levels`` (low/medium/high) — assigning a level is an inference requiring a
  null this project never established. The decision rule built on them
  measured a ~43% false-positive rate on pairs of runs known to be identical.
* Duplicate names — ``continuity`` was ``1 - sensitivity`` computed from the
  same JS divergences, and the ``wager`` reading was byte-for-byte the same
  computation as the ``surprise`` aggregate. Reporting one measurement twice
  under two names inflates the apparent dimensionality of the signal set.
  Each quantity now appears exactly once.

Privacy note: everything returned here is a derived scalar. Nothing in this
module reads or emits raw token distributions — records built from these
values are safe under the compute-and-discard default. Raw-artifact
persistence is a separate, explicit opt-in (TraceabilityConfig / --trace)
handled by the engine, never here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

# Version of the machine-record shape emitted by signals_record(). Bump on
# any breaking change to record structure (field renames/removals). Additive
# keys do not require a bump.
#
# record-v2: the `signals`/`readings` split, the `normalized` and `levels`
# blocks, and `findings_levels` are gone; a single flat `measurements` block
# in natural units replaces them.
# record-v3 (current): the per-record `units` block is opt-in (`--units`)
# rather than always present — it is identical for every record of a given
# schema_version and `hif schema` prints it on demand. The field-descriptor
# blocks are renamed to the names docs/METRICS.md Part 3 gives them:
# `field` -> `perturbation_field`, `branch_field` -> `trajectory_branch_field`.
RECORD_SCHEMA_VERSION = "record-v3"

# Version of the measurement set this package computes. Minor bumps within a
# major family are additive supersets (see cli._signal_set_family), so
# artifacts across them still compare over the intersection.
#
# hif-v2.1: added input_entropy_std_bits and
# branch_pairwise_cosine_similarity — the natural-unit forms of the Stability
# and Continuity aggregates, which were computed but never surfaced.
SIGNAL_SET_VERSION = "hif-v2.1"


def profile_hash(model_name: str, prompt: str, seed: int) -> str:
    key = f"{model_name}|{prompt}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# The measurement set
# ---------------------------------------------------------------------------

# (key, human label, what it needs, surrogate group)
#
# `surrogate group` names which surrogate caveat applies when the target model
# could not produce the quantity itself: "input" readings inherit
# findings.surrogate_model_name's proxy caveat, "output" readings inherit
# output_distribution_surrogate_name's.
MEASUREMENTS: list[tuple[str, str, str]] = [
    ("input_entropy_shift_bits", "Input entropy shift (bits)", "input"),
    ("input_entropy_std_bits", "Input entropy shift spread (bits)", "input"),
    ("perturbation_jsd_bits", "Perturbation JSD (bits)", "output"),
    ("io_correlation_r", "Input/output correlation (r)", "input"),
    ("io_cosine_similarity", "Input/output cosine similarity", ""),
    ("prompt_surprisal_excess_bits", "Prompt surprisal excess (bits)", "input"),
    ("candidate_cluster_entropy_bits", "Candidate cluster entropy (bits)", "output"),
    ("output_entropy_bits", "Output entropy (bits)", "output"),
    ("output_entropy_step_delta_bits", "Output entropy step delta (bits)", "output"),
    ("semantic_centroid_veer_cosine", "Semantic centroid veer (cosine distance)", ""),
    ("attention_entropy_output_bits", "Output attention-row entropy (bits)", ""),
    ("attention_entropy_input_bits", "Input attention-row entropy (bits)", ""),
    ("counterfactual_exposure_fraction", "Counterfactual exposure (fraction of steps)", ""),
    ("branch_pairwise_cosine_similarity", "Branch pairwise cosine similarity", ""),
]

MEASUREMENT_KEYS: tuple[str, ...] = tuple(k for k, _l, _s in MEASUREMENTS)

# What each measurement is, in one line. Emitted by `hif schema` and used for
# the CLI table's unit column, so the unit is never ambiguous at the point of
# use.
MEASUREMENT_UNITS: dict[str, str] = {
    "input_entropy_shift_bits":
        "bits — mean |mean input-token entropy(variant) − mean input-token entropy(baseline)| "
        "over perturbation variants. Unbounded above.",
    "input_entropy_std_bits":
        "bits — standard deviation (ddof=1) of the per-variant input entropy shifts. The "
        "spread of the model's entropy response across perturbations. Unbounded above; "
        "absent when fewer than two variants exist.",
    "perturbation_jsd_bits":
        "bits — mean Jensen-Shannon divergence between the baseline output distribution and "
        "each perturbed variant's. Bounded to [0, 1] by definition in log base 2.",
    "io_correlation_r":
        "dimensionless — Pearson r between per-variant input entropy shift and per-variant "
        "JSD. Bounded to [-1, 1] by definition; reported signed.",
    "io_cosine_similarity":
        "dimensionless — cosine similarity between the input embedding and the output "
        "embedding. Bounded to [-1, 1] by definition.",
    "prompt_surprisal_excess_bits":
        "bits — mean max(0, surprisal(token) − H(distribution)) over teacher-forced prompt "
        "positions. Unbounded above.",
    "candidate_cluster_entropy_bits":
        "bits — mean Shannon entropy of the semantic-cluster mass distribution over each "
        "generation step's top-K candidate cloud. Unbounded above (bounded in practice by "
        "log2 of the cluster count).",
    "output_entropy_bits":
        "bits — mean Shannon entropy of the per-step top-K output distribution. A lower "
        "bound on full-vocabulary entropy when the distribution is truncated.",
    "output_entropy_step_delta_bits":
        "bits — mean |H(step i) − H(step i−1)| over the nucleus (95% mass, renormalised) "
        "entropy trace. Unbounded above.",
    "semantic_centroid_veer_cosine":
        "cosine distance — mean step-to-step displacement of the candidate cloud's semantic "
        "centroid in embedding space. Bounded to [0, 2] by definition.",
    "attention_entropy_output_bits":
        "bits — mean Shannon entropy of the causal-prefix attention row at each output "
        "position. Grows with prefix length; not divided by log2(prefix length).",
    "attention_entropy_input_bits":
        "bits — mean Shannon entropy of the causal-prefix attention row at each input "
        "position. Grows with prefix length; not divided by log2(prefix length).",
    "branch_pairwise_cosine_similarity":
        "dimensionless — mean pairwise cosine similarity between trajectory branch "
        "embeddings. High = branches converge semantically; low = they scatter. Bounded "
        "to [-1, 1] by definition. This is the natural-unit form of the Continuity "
        "aggregate, reported directly rather than as a derived score.",
    "counterfactual_exposure_fraction":
        "fraction of analysed generation steps — steps where a probabilistically accessible "
        "alternative token would have pulled the response toward a different meaning. "
        "A proportion, bounded to [0, 1] by construction.",
}


def measurements(p) -> dict[str, float]:
    """Every measurement this profile actually supports, in natural units.

    Absent measurements are OMITTED, never pinned to 0.0 or 1.0: absent means
    the run produced no evidence for that quantity (a backend that cannot
    teacher-force, an analysis stage that did not run), which is a different
    statement from a measured zero.
    """
    from hif.hourglass.input_side import mean_surprisal_excess

    out: dict[str, float] = {}
    m = p.metrics
    st = m.stability

    # --- perturbation response (input side, output side, and their coupling)
    if st.input_entropy_shift_bits is not None:
        out["input_entropy_shift_bits"] = st.input_entropy_shift_bits
    if getattr(st, "input_entropy_std_bits", None) is not None:
        out["input_entropy_std_bits"] = st.input_entropy_std_bits
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
    if m.semantic:
        ces = [s.cluster_entropy for s in m.semantic]
        out["candidate_cluster_entropy_bits"] = sum(ces) / len(ces)

    # --- output distribution
    if m.distribution:
        ents = [d.entropy_bits for d in m.distribution]
        out["output_entropy_bits"] = sum(ents) / len(ents)
    if m.distribution and len(m.distribution) >= 2:
        # Nucleus entropy (95% mass, renormalised) so the trace is comparable
        # across backends regardless of how many logprobs each exposes.
        nents = [d.nucleus_entropy_bits for d in m.distribution]
        deltas = [abs(nents[i] - nents[i - 1]) for i in range(1, len(nents))]
        out["output_entropy_step_delta_bits"] = sum(deltas) / len(deltas)

    # --- semantic field (Veer): present only when semantic_field analysis ran
    sf = getattr(p, "semantic_field", None)
    if sf is not None and getattr(sf, "mean_veer", None) is not None:
        out["semantic_centroid_veer_cosine"] = sf.mean_veer

    # --- counterfactual exposure: present only when the divergence analysis ran
    exp = getattr(p, "exposure", None) or getattr(p, "hallucination", None)
    if exp is not None and getattr(exp, "candidates", None):
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


# Historical names kept as thin aliases so external callers and tests that
# import them keep working. Both return the same flat measurement dict.
def profile_scores(profile) -> dict[str, float]:
    return measurements(profile)


def extended_signal_values(p) -> dict[str, float]:
    return measurements(p)


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


def semantic_field_scalars(profile) -> Optional[dict]:
    """Per-profile within-generation semantic-field (Veer) summary scalars,
    or None when absent (< 2 generation steps, or the instrument disabled)."""
    sf = getattr(profile, "semantic_field", None)
    if sf is None:
        return None
    return {"mean_veer": sf.mean_veer, "max_veer": sf.max_veer,
            "mean_deformation": sf.mean_deformation, "n_steps": sf.n_steps}


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
    no token alternatives. The model's generated output TEXT is included (it's
    the response the caller already has); reconstructable distribution data is
    not, unless the run traced (then trace_path points at the artifact).

    Round-trip rule: every value here is the same number the terminal table
    displays, sourced from the same function.
    """
    f = profile.findings

    output_text = "".join(s.selected_token_str for s in profile.output_side.steps)

    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "signal_set_version": SIGNAL_SET_VERSION,
        "hash": profile_hash(model_name, prompt, seed),
        "model": model_name,
        "backend": backend,
        "regime": regime,
        "seed": seed,
        "modality": getattr(profile.prompt, "modality", "text") or "text",
        # Every measurement in its natural unit. Absent measurements are
        # omitted. See MEASUREMENT_UNITS for what each unit means.
        "measurements": measurements(profile),

        # Part 3 of docs/METRICS.md — behaviour as a region rather than a
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
        "output_text": output_text,
        "output_tokens": len(profile.output_side.generated_ids),
        "input_tokens": len(profile.input_side.prompt_token_ids),
    }
    # Multimodal provenance — InputPartRecord is hash + dims ONLY (never
    # pixels/base64; see schema.py), so it is safe in a derived record.
    # Units are constant per signal_set_version and identical on every record,
    # so they are opt-in (`--units`) rather than repeated on every JSONL line.
    # `hif schema` prints them for every measurement without running a model.
    if include_units:
        record["units"] = {
            k: MEASUREMENT_UNITS[k] for k in record["measurements"]
            if k in MEASUREMENT_UNITS
        }
    input_parts = getattr(profile.prompt, "input_parts", None)
    if input_parts:
        record["input_parts"] = [part.model_dump() for part in input_parts]
    rs = getattr(profile, "region_sensitivity", None)
    if rs is not None:
        # Derived per-cell JSD grid (multimodal runs) — scalars only.
        record["region_sensitivity"] = rs.model_dump()
    if latency:
        record["latency"] = {k: round(v, 6) for k, v in latency.items()}
    if trace_path:
        record["trace_path"] = trace_path
    if extras:
        record.update(extras)
    return record
