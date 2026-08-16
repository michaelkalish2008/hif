"""Markdown rendering for BehavioralRangeProfile artifacts.

One report, not two. There was also a `render_public()` — a plain-English
subset of the same measurement table — written alongside the technical report
on every `--output-dir` run. It was dropped: a second rendering of the same
numbers is a second place for them to disagree, and the reader who wants the
values without the diagnostics is better served by the JSON artifact every run
writes, which is the whole profile rather than a lossy excerpt of it.
"""

from __future__ import annotations

from pathlib import Path

from hif.profile.measure import (
    _prompt_reference_model,
    measurements,
    prompt_measurements,
)
from hif.profile.registry import (
    MEASUREMENT_REGISTRY,
    SUBJECT_PROMPT_ONLY,
    run_subjects,
)
from hif.profile.schema import BehavioralRangeProfile


def _absent_reason(key: str, profile) -> str:
    """Why this row is absent: never requested, or requested and unobtainable."""
    from hif.cli._output import OPT_IN_FLAGS

    flag = OPT_IN_FLAGS.get(key)
    if flag is not None and getattr(
        getattr(getattr(profile, "config", None), "generation", None),
        "entropy_percentile", None,
    ) is None:
        return f"absent (not requested — pass {flag})"
    return "absent (not measurable on this run)"


def _num(value, spec: str = ".4f") -> str:
    """A diagnostic cell: the number, or the word for its absence.

    The diagnostic blocks below carry several quantities that are None on a
    run that generated nothing (center entropies, the prompt/output distance,
    the mean step entropy). Formatting None with `:.4f` raises, and defaulting
    it to 0.0 is the fabrication this whole pass removes — so the report says
    absent, the same word the measurement table uses.
    """
    return "absent" if value is None else format(value, spec)


def _cell(text: str) -> str:
    """Escape the table delimiter in text destined for a Markdown table cell.

    Registry prose is written in maths, not Markdown: `input_entropy_shift_bits`
    defines itself as `mean |...|`, and a bare `|` in a cell is a column break,
    so that one row rendered as five columns against a four-column header. Same
    escape tools/gen_flags_doc.py applies to help text, for the same reason.
    """
    return (text or "").replace("|", "\\|")


def _step_tokens(profile) -> "list[str] | None":
    """The token each per-step metric row is about — or None when unknowable.

    The per-step tables are computed over builder.py step 6b's basis: the
    target's own trace normally, but the SURROGATE's steps when the target
    exposed no usable distribution and --surrogate recovered one by
    teacher-forcing the proxy over the continuation. That recovery is "in the
    surrogate's own tokenization" (hif/hourglass/output_side.py) — differently
    segmented, so row *i* is not the target's token *i*, and the artifact
    stores only `output_side` (the target's trace), never the recovered basis.
    Labelling those rows from `output_side.steps` would put the wrong token
    beside every number, so we say we cannot label them instead.
    """
    if profile.findings.output_distribution_surrogate_name is not None:
        return None
    return [s.selected_token_str for s in profile.output_side.steps]


def _token_cell(tokens: "list[str] | None", i: int) -> str:
    """One token as a table cell: repr'd, so whitespace and newlines are visible.

    A generated token is routinely " the" or "\\n"; printed bare, the column
    reads as empty or breaks the row. repr() makes the leading space and the
    escape visible, and _cell() handles a token that is literally "|".
    """
    if tokens is None or i >= len(tokens):
        return ""
    return f"`{_cell(repr(tokens[i]))}`"


# ---------------------------------------------------------------------------
# Technical report (full)
# ---------------------------------------------------------------------------


def render_technical(profile: BehavioralRangeProfile, output_path: Path) -> None:
    """Write a full technical Markdown report including all metric values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    a = lines.append

    a(f"# BRI Technical Report — {profile.model.name}")
    a(f"")
    a(f"**Schema version:** {profile.schema_version}")
    a(f"**Created:** {profile.created_at.isoformat()}")
    a(f"")

    # Model identity
    a("## Model Identity")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    a(f"| Name | {profile.model.name} |")
    a(f"| Backend | {profile.model.backend} |")
    a(f"| Vocab size | {profile.model.vocab_size} |")
    a(f"| Context length | {profile.model.context_length} |")
    a(f"| Parameter count | {profile.model.parameter_count or 'unknown'} |")
    a("")

    # Prompt
    a("## Prompt")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    a(f"| Regime | {profile.prompt.regime} |")
    a(f"| Token count | {profile.prompt.token_count} |")
    a(f"| SHA-256 | `{profile.prompt.prompt_hash[:16]}...` |")
    a("")
    a("```")
    a(profile.prompt.text)
    a("```")
    a("")

    # Measurements — natural units, no levels
    a(f"## Measurements — {profile.model.name}")
    a("")
    a("All values are in natural units. There are no levels, thresholds, or")
    a("verdicts: assigning one needs a null distribution this instrument does")
    a("not have.")
    a("")
    vals = measurements(profile)
    subjects = run_subjects(profile)
    star = " *" if profile.findings.surrogate_model_name is not None else ""
    a("| Measurement | Value | Subject | Unit / definition |")
    a("|---|---|---|---|")
    for m in MEASUREMENT_REGISTRY:
        if subjects.get(m.key) == SUBJECT_PROMPT_ONLY:
            continue
        v = vals.get(m.key)
        mark = star if m.surrogate_group else ""
        # Same two absences the terminal distinguishes (hif/cli/_output.py):
        # a report that says "not measurable" about an opt-in row nobody asked
        # for sends the reader to check their backend over a missing flag.
        shown = f"{v:.6g}" if v is not None else _absent_reason(m.key, profile)
        a(
            f"| {_cell(m.name)}{mark} | {shown} | {subjects.get(m.key, '')} | "
            f"{_cell(m.unit)} — {_cell(m.definition)} |"
        )
    a("")
    _slope = profile.findings.similarity_trend_slope
    a("Similarity trend slope: "
      + (f"{_slope:+.6g}" if _slope is not None
         else "absent (fewer than two output steps)")
      + " (OLS slope of the per-step candidate-cloud similarity).")
    a("")
    if profile.findings.surrogate_model_name is not None:
        a(f"\\* computed via surrogate model `{profile.findings.surrogate_model_name}` "
          "(teacher-forcing proxy) — a measurement of the surrogate over the "
          "target's text, not of the target model.")
        a("")

    # Prompt-only quantities — real measurements, but not of this model, so
    # they get their own section rather than a footnote in the table above.
    prompt_vals = prompt_measurements(profile)
    if prompt_vals:
        a("## Prompt measurements — not about this model")
        a("")
        a("Subject: `prompt-only`. Computed from the prompt text under the")
        a("reference model named below, with no input from")
        a(f"`{profile.model.name}`. Comparable across targets for exactly that")
        a("reason, and reported separately rather than as caveated")
        a("measurements of the model. See docs/MEASUREMENTS.md § Subject.")
        a("")
        a("| Measurement | Value | Reference model | Unit |")
        a("|---|---|---|---|")
        for m in MEASUREMENT_REGISTRY:
            if m.key not in prompt_vals:
                continue
            ref = _prompt_reference_model(m.key, profile) or "unknown"
            a(f"| {_cell(m.name)} | {prompt_vals[m.key]:.6g} | `{ref}` | {_cell(m.unit)} |")
        a("")

    # Center diagnostics
    a("## Center Diagnostics")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Input mean entropy | {profile.center.input_mean_entropy:.4f} |")
    a(f"| Output mean entropy | {_num(profile.center.output_mean_entropy)} |")
    a(f"| Entropy ratio (output/input, both bits) | {_num(profile.center.entropy_ratio)} |")
    a(f"| Prompt/output cosine distance | {_num(profile.center.prompt_output_cosine_distance)} |")
    a("")

    # Input-side summary
    a("## Input-Side Analysis")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Mean surprisal (bits) | {profile.input_side.mean_surprisal:.4f} |")
    a(f"| Mean entropy (bits) | {profile.input_side.mean_entropy:.4f} |")
    a(f"| Max entropy log2\\|V\\| (bits) | {profile.input_side.max_entropy:.4f} |")
    a("")

    # Output-side summary
    a("## Output-Side Analysis")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Mean step entropy | {_num(profile.output_side.mean_step_entropy)} |")
    a(f"| Generated tokens | {len(profile.output_side.generated_ids)} |")
    a(f"| Top-K | {profile.output_side.top_k} |")
    a("")

    # Trajectory summary
    a("## Trajectory Analysis")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Start step | {profile.trajectory.start_step} |")
    a(f"| Branches | {profile.trajectory.n_branches} |")
    a(f"| Rollout steps | {profile.trajectory.rollout_steps} |")
    a(f"| Initial clusters | {profile.trajectory.initial_n_clusters} |")
    a(f"| Persistence score | {profile.trajectory.persistence_score:.4f} |")
    a(f"| Explosion score | {profile.trajectory.explosion_score:.4f} |")
    a(f"| Convergence score | {profile.trajectory.convergence_score:.4f} |")
    a("")
    a("### Branches")
    a("")
    a("| Cluster | Representative Token | Generated Text |")
    a("|---|---|---|")
    for branch in profile.trajectory.branches:
        rep_ids_str = str(branch.representative_token_ids)
        final_text_escaped = branch.final_text.replace("|", "\\|").replace("\n", " ")
        a(f"| {branch.cluster_id} | `{rep_ids_str}` | {final_text_escaped[:80]} |")
    a("")

    # Distribution metrics table (per step, truncated to 10 steps for readability)
    a("## Distribution Metrics (per output step)")
    a("")
    tokens = _step_tokens(profile)
    a("| Token | Step | Entropy (bits) | Logit margin | Top-K mass | Nucleus eff. support | Tail weight |")
    a("|---|---|---|---|---|---|---|")
    for i, dm in enumerate(profile.metrics.distribution[:10]):
        a(f"| {_token_cell(tokens, i)} | {i} | {dm.entropy_bits:.3f} | {dm.logit_margin:.3f} | {dm.topk_cumulative_mass:.3f} | {dm.nucleus_effective_support_size:.1f} | {dm.tail_weight:.3f} |")
    if len(profile.metrics.distribution) > 10:
        a(f"| | ... | ({len(profile.metrics.distribution) - 10} more steps) | | | | |")
    a("")
    if tokens is None:
        a(f"Tokens are not shown: these rows were read off `"
          f"{profile.findings.output_distribution_surrogate_name}` teacher-forced "
          "over the target's continuation, in the surrogate's own tokenization. "
          "Row *i* is the surrogate's position *i*, which is not "
          f"`{profile.model.name}`'s token *i*, and the artifact does not carry "
          "the surrogate's segmentation to label it with.")
        a("")

    # Semantic metrics table
    a("## Semantic Metrics (per output step)")
    a("")
    a("| Step | Clusters | Entropy | Mean pair dist | Max inter-cluster dist |")
    a("|---|---|---|---|---|")
    for i, sm in enumerate(profile.metrics.semantic[:10]):
        a(f"| {i} | {sm.cluster_count} | {sm.cluster_entropy:.3f} | {sm.mean_pairwise_distance:.3f} | {sm.max_inter_cluster_distance:.3f} |")
    if len(profile.metrics.semantic) > 10:
        a(f"| ... | ({len(profile.metrics.semantic) - 10} more steps) | | | |")
    a("")

    # Stability
    a("## Perturbation Response")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    stab = profile.metrics.stability

    def _fmt(v: "float | None") -> str:
        return "n/a (not measurable)" if v is None else f"{v:.4f}"

    a(f"| Input entropy shift (bits) | {_fmt(stab.input_entropy_shift_bits)} |")
    a(f"| Perturbation JSD (bits) | {_fmt(stab.perturbation_jsd_bits)} |")
    a(f"| Input-output correlation (r) | {_fmt(stab.input_output_correlation)} |")
    a(f"| N perturbations | {stab.n_perturbations} |")
    a("")

    # Perturbation records
    if profile.perturbations:
        a("## Perturbation Records")
        a("")
        for rec in profile.perturbations:
            a(f"### Generator: `{rec.generator}`")
            a("")
            a("**Variants:**")
            for v in rec.variants:
                a(f"- {v}")
            a("")
            if rec.sensitivity:
                # `Steps` is the shared-prefix length each row's means were
                # taken over. Without it every row reads as equally weighted,
                # which is how a mean over 6 steps sat beside a mean over 64
                # with nothing to tell them apart.
                a("| Variant | Steps | Mean JS | Mean KL | Entropy delta |")
                a("|---|---|---|---|---|")
                for i, sens in enumerate(rec.sensitivity):
                    kl = _num(sens.mean_kl_divergence)
                    if sens.n_undefined_kl_steps:
                        kl += f" ({sens.n_undefined_kl_steps} undefined)"
                    a(
                        f"| {i} | {sens.n_steps_aligned} | "
                        f"{_num(sens.mean_js_divergence)} | {kl} | "
                        f"{_num(sens.mean_entropy_delta)} |"
                    )
                a("")

    # Config
    a("## Run Configuration")
    a("")
    a("```json")
    a(profile.config.model_dump_json(indent=2))
    a("```")
    a("")

    if profile.notes:
        a("## Notes")
        a("")
        a(profile.notes)
        a("")

    # The one paragraph the dropped public summary was carrying that this
    # report was not. It is the scope of every number above, so it outlives
    # the file it used to live in.
    a("## What this is not")
    a("")
    a("These numbers describe what the model did on one prompt at one moment.")
    a("They are not a drift detection, an attack detection, or a quality score,")
    a("and none of them carries a threshold above or below which something is")
    a("wrong.")
    a("")

    output_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Legacy shim
# ---------------------------------------------------------------------------


def render_markdown(profile, path: Path) -> None:
    """Render a BehavioralRangeProfile to a structured Markdown report.

    Kept as the historical entry point. It took a `public` flag selecting the
    plain-English variant; there is one report now, so it takes none.
    """
    render_technical(profile, path)
