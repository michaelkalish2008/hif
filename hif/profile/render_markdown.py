"""Markdown rendering for BehavioralRangeProfile artifacts (full and public-facing variants)."""

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


def _cell(text: str) -> str:
    """Escape the table delimiter in text destined for a Markdown table cell.

    Registry prose is written in maths, not Markdown: `input_entropy_shift_bits`
    defines itself as `mean |...|`, and a bare `|` in a cell is a column break,
    so that one row rendered as five columns against a four-column header. Same
    escape tools/gen_flags_doc.py applies to help text, for the same reason.
    """
    return (text or "").replace("|", "\\|")


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
    a(f"Similarity trend slope: {profile.findings.similarity_trend_slope:+.6g} "
      "(OLS slope of per-step input/output cosine similarity).")
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
    a(f"| Output mean entropy | {profile.center.output_mean_entropy:.4f} |")
    a(f"| Entropy ratio (output/input, both bits) | {profile.center.entropy_ratio:.4f} |")
    a(f"| Prompt/output cosine distance | {profile.center.prompt_output_cosine_distance:.4f} |")
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
    a(f"| Mean step entropy | {profile.output_side.mean_step_entropy:.4f} |")
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
    a("| Step | Entropy (bits) | Logit margin | Top-K mass | Eff. support | Tail weight |")
    a("|---|---|---|---|---|---|")
    for i, dm in enumerate(profile.metrics.distribution[:10]):
        a(f"| {i} | {dm.entropy_bits:.3f} | {dm.logit_margin:.3f} | {dm.topk_cumulative_mass:.3f} | {dm.effective_support_size:.1f} | {dm.tail_weight:.3f} |")
    if len(profile.metrics.distribution) > 10:
        a(f"| ... | ({len(profile.metrics.distribution) - 10} more steps) | | | | |")
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
                a("| Variant | Mean JS | Mean KL | Entropy delta |")
                a("|---|---|---|---|")
                for i, sens in enumerate(rec.sensitivity):
                    a(f"| {i} | {sens.mean_js_divergence:.4f} | {sens.mean_kl_divergence:.4f} | {sens.mean_entropy_delta:.4f} |")
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

    output_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Public summary (non-technical)
# ---------------------------------------------------------------------------


def render_public(profile: BehavioralRangeProfile, output_path: Path) -> None:
    """Write a plain-English summary of what was measured.

    Deliberately contains no level, no adjective, and no interpretation of
    whether a value is good or bad — the previous version mapped each metric to
    a low/medium/high paragraph, which read as a judgement the instrument
    cannot support.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f = profile.findings
    vals = measurements(profile)
    lines: list[str] = []
    a = lines.append

    a(f"# Behavioural measurements — {profile.model.name}")
    a("")
    a(f"**Prompt:** {profile.prompt.text}")
    a(f"**Regime:** {profile.prompt.regime}")
    a("")
    a("---")
    a("")
    a("## What was measured")
    a("")
    a("| Measurement | Value | Unit |")
    a("|---|---|---|")
    for m in MEASUREMENT_REGISTRY:
        v = vals.get(m.key)
        shown = "absent" if v is None else f"{v:.6g}"
        a(f"| {_cell(m.name)} | {shown} | {_cell(m.unit)} |")
    a("")
    a("Absent means this run produced no evidence for that quantity — the")
    a("backend could not teacher-force, or an optional analysis stage did not")
    a("run. It does not mean zero.")
    a("")
    if f.surrogate_model_name is not None:
        a(f"Input-side measurements were computed by teacher-forcing the "
          f"surrogate model `{f.surrogate_model_name}` over the prompt. They "
          f"describe the surrogate reading {profile.model.name}'s text, not "
          f"{profile.model.name} itself.")
        a("")
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


def render_markdown(profile, path: Path, public: bool = False) -> None:
    """Render a BehavioralRangeProfile to a structured Markdown report.

    Dispatches to render_public() or render_technical() based on the `public` flag.
    """
    if public:
        render_public(profile, path)
    else:
        render_technical(profile, path)
