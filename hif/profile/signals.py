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
# blocks are renamed to the names docs/MEASUREMENTS.md Part 4 gives them:
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
# The measurement registry
# ---------------------------------------------------------------------------

# Every measurement is a triple — observable × functional × resolution — plus
# the contract details a consumer needs (key, unit, definition) and the
# surrogate caveat. This registry is the single source of truth: the CLI
# table, `hif schema`, the Markdown renderers, and the record path all derive
# from it. Adding a measurement means adding one row here (see
# CONTRIBUTING.md).

# Resolution — the granularity of the underlying series a scalar summarises
# within one run. The record always carries the run-level scalar; resolution
# says what that scalar is a summary OF, and therefore whether a token-level
# trace exists behind the number (docs/MEASUREMENTS.md Part 2):
#
#   "per-position"  one sample per prompt/context position — the trace behind
#                   the scalar is indexed by token position.
#   "per-step"      one sample per generation step — the trace is indexed by
#                   output step.
#   "aggregate"     the quantity exists only at whole-run level (across
#                   perturbation variants, trajectory branches, or the run's
#                   endpoints); there is no per-token trace to restore.
RESOLUTIONS: tuple[str, ...] = ("aggregate", "per-step", "per-position")

# Functional — which family of functional produces the number.
# "information-theoretic" reads the shape of a distribution (entropy,
# surprisal, JSD, trace correlation); "geometric" reads where the mass sits
# in embedding space (distance, cluster structure).
FUNCTIONALS: tuple[str, ...] = ("information-theoretic", "geometric")


@dataclass(frozen=True)
class Measurement:
    """One registry row: a measurement's triple and its record contract.

    key             record/measurement key — descriptive and unit-suffixed;
                    this is the stable machine name and never changes.
    name            human-readable quantity name, shown in CLI and report
                    tables.
    unit            the natural unit ("bits", "dimensionless", "cosine
                    distance", "fraction of analysed generation steps").
    definition      what the quantity is, in one or two sentences, including
                    its bound (or that it is unbounded) and any absence rule.
    observable      what the quantity is computed from — what the forward
                    pass (or the encoder over its text) exposes.
    functional      one of FUNCTIONALS.
    resolution      one of RESOLUTIONS (see the comment above).
    label           optional canonical shorthand from the docs ("Wager ▲",
                    "Continuity"). None when no established shorthand exists —
                    a made-up name would be worse than none.
    surrogate_group which surrogate caveat applies when the target model could
                    not produce the quantity itself: "input" measurements
                    inherit findings.surrogate_model_name's proxy caveat,
                    "output" measurements inherit
                    output_distribution_surrogate_name's, "" inherits neither.
    """

    key: str
    name: str
    unit: str
    definition: str
    observable: str
    functional: str
    resolution: str
    label: Optional[str] = None
    surrogate_group: str = ""


MEASUREMENT_REGISTRY: tuple[Measurement, ...] = (
    Measurement(
        key="input_entropy_shift_bits",
        name="Input entropy shift (bits)",
        unit="bits",
        definition=(
            "mean |mean input-token entropy(variant) − mean input-token "
            "entropy(baseline)| over perturbation variants. Unbounded above."
        ),
        observable="input distribution",
        functional="information-theoretic",
        resolution="aggregate",
        label=None,  # the historical "Input Stability" named the removed
        # inverted score (1 − x), not this quantity — carrying that name
        # forward would re-attach the score reading.
        surrogate_group="input",
    ),
    Measurement(
        key="input_entropy_std_bits",
        name="Input entropy shift spread (bits)",
        unit="bits",
        definition=(
            "standard deviation (ddof=1) of the per-variant input entropy "
            "shifts. The spread of the model's entropy response across "
            "perturbations. Unbounded above; absent when fewer than two "
            "variants exist."
        ),
        observable="input distribution",
        functional="information-theoretic",
        resolution="aggregate",
        label="Stability",  # the natural-unit form of the Stability
        # aggregate (see SIGNAL_SET_VERSION history above).
        surrogate_group="input",
    ),
    Measurement(
        key="perturbation_jsd_bits",
        name="Perturbation JSD (bits)",
        unit="bits",
        definition=(
            "mean Jensen-Shannon divergence between the baseline output "
            "distribution and each perturbed variant's. Bounded to [0, 1] by "
            "definition in log base 2."
        ),
        observable="output distribution",
        functional="information-theoretic",
        resolution="aggregate",
        label="Sensitivity",  # the quantity the historical `sensitivity`
        # score was computed from (mean JS divergence per variant).
        surrogate_group="output",
    ),
    Measurement(
        key="io_correlation_r",
        name="Input/output correlation (r)",
        unit="dimensionless",
        definition=(
            "Pearson r between per-variant input entropy shift and "
            "per-variant JSD. Bounded to [-1, 1] by definition; reported "
            "signed."
        ),
        observable="input × output distributions",
        functional="information-theoretic",
        resolution="aggregate",
        label=None,  # no established shorthand — the docs name it only by
        # its zone ("Center").
        surrogate_group="input",
    ),
    Measurement(
        key="io_cosine_similarity",
        name="Input/output cosine similarity",
        unit="dimensionless",
        definition=(
            "cosine similarity between the input embedding and the output "
            "embedding. Bounded to [-1, 1] by definition."
        ),
        observable="input/output text embeddings",
        functional="geometric",
        resolution="aggregate",
        label=None,  # no established shorthand (`io_sim` is a field name,
        # not a doc name).
    ),
    Measurement(
        key="prompt_surprisal_excess_bits",
        name="Prompt surprisal excess (bits)",
        unit="bits",
        definition=(
            "mean max(0, surprisal(token) − H(distribution)) over "
            "teacher-forced prompt positions. Unbounded above."
        ),
        observable="input distribution",
        functional="information-theoretic",
        resolution="per-position",
        label="Wager ▲",
        surrogate_group="input",
    ),
    Measurement(
        key="candidate_cluster_entropy_bits",
        name="Candidate cluster entropy (bits)",
        unit="bits",
        definition=(
            "mean Shannon entropy of the semantic-cluster mass distribution "
            "over each generation step's top-K candidate cloud. Unbounded "
            "above (bounded in practice by log2 of the cluster count)."
        ),
        observable="output distribution",
        functional="geometric",
        resolution="per-step",
        label=None,  # "Cluster Entropy" in the docs names the per-step
        # component, not a canonical instrument shorthand.
        surrogate_group="output",
    ),
    Measurement(
        key="output_entropy_bits",
        name="Output entropy (bits)",
        unit="bits",
        definition=(
            "mean Shannon entropy of the per-step top-K output distribution. "
            "A lower bound on full-vocabulary entropy when the distribution "
            "is truncated."
        ),
        observable="output distribution",
        functional="information-theoretic",
        resolution="per-step",
        label="Entropy ●",
        surrogate_group="output",
    ),
    Measurement(
        key="output_entropy_step_delta_bits",
        name="Output entropy step delta (bits)",
        unit="bits",
        definition=(
            "mean |H(step i) − H(step i−1)| over the nucleus (95% mass, "
            "renormalised) entropy trace. Unbounded above."
        ),
        observable="output distribution",
        functional="information-theoretic",
        resolution="per-step",
        label=None,  # deliberately NOT "Shift ◆": Shift is the step-to-step
        # JSD (where the mass sits); this is the step-to-step change in the
        # amount of uncertainty. Confusing the two is the exact mistake the
        # docs warn against.
        surrogate_group="output",
    ),
    Measurement(
        key="semantic_centroid_veer_cosine",
        name="Semantic centroid veer (cosine distance)",
        unit="cosine distance",
        definition=(
            "mean step-to-step displacement of the candidate cloud's "
            "semantic centroid in embedding space. Bounded to [0, 2] by "
            "definition."
        ),
        observable="output distribution",
        functional="geometric",
        resolution="per-step",
        label="Veer ◈",
    ),
    Measurement(
        key="attention_entropy_output_bits",
        name="Output attention-row entropy (bits)",
        unit="bits",
        definition=(
            "mean Shannon entropy of the causal-prefix attention row at each "
            "output position. Grows with prefix length; not divided by "
            "log2(prefix length)."
        ),
        observable="attention row",
        functional="information-theoretic",
        resolution="per-position",
        label="Spread ■",
    ),
    Measurement(
        key="attention_entropy_input_bits",
        name="Input attention-row entropy (bits)",
        unit="bits",
        definition=(
            "mean Shannon entropy of the causal-prefix attention row at each "
            "input position. Grows with prefix length; not divided by "
            "log2(prefix length)."
        ),
        observable="attention row",
        functional="information-theoretic",
        resolution="per-position",
        label="Horizon",  # no glyph: the ▼ symbol is not used in the code.
    ),
    Measurement(
        key="counterfactual_exposure_fraction",
        name="Counterfactual exposure (fraction of steps)",
        unit="fraction of analysed generation steps",
        definition=(
            "steps where a probabilistically accessible alternative token "
            "would have pulled the response toward a different meaning. A "
            "proportion, bounded to [0, 1] by construction."
        ),
        observable="output distribution",
        functional="geometric",
        resolution="per-step",
        label="Exposure ◇",
    ),
    Measurement(
        key="branch_pairwise_cosine_similarity",
        name="Branch pairwise cosine similarity",
        unit="dimensionless",
        definition=(
            "mean pairwise cosine similarity between trajectory branch "
            "embeddings. High = branches converge semantically; low = they "
            "scatter. Bounded to [-1, 1] by definition. This is the "
            "natural-unit form of the Continuity aggregate, reported "
            "directly rather than as a derived score."
        ),
        observable="trajectory branch embeddings",
        functional="geometric",
        resolution="aggregate",
        label="Continuity",
    ),
)


# ---------------------------------------------------------------------------
# Derived views — thin projections of the registry, kept for callers that
# only need one facet. The registry above is the source of truth.
# ---------------------------------------------------------------------------

# (key, name, surrogate group) — the historical tuple shape.
MEASUREMENTS: list[tuple[str, str, str]] = [
    (m.key, m.name, m.surrogate_group) for m in MEASUREMENT_REGISTRY
]

MEASUREMENT_KEYS: tuple[str, ...] = tuple(m.key for m in MEASUREMENT_REGISTRY)

# "unit — definition" per key, the historical one-line form.
MEASUREMENT_UNITS: dict[str, str] = {
    m.key: f"{m.unit} — {m.definition}" for m in MEASUREMENT_REGISTRY
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
        # omitted. See MEASUREMENT_REGISTRY for what each quantity and unit
        # means.
        "measurements": measurements(profile),

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
