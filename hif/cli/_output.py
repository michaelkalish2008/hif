"""Terminal presentation of a finished profile.

Tables and prose only — every number printed here is read from
hif/profile/measure.py, never recomputed, so a value in a terminal table and
the same key in a `--json` record cannot diverge.

There is no Level column and no Normalized column anywhere in this module, by
design; see hif/profile/registry.py for why those were removed. Absent
measurements are named and left absent rather than rendered as a number.
"""

from __future__ import annotations

from rich.table import Table

from hif.cli._app import console
from hif.profile.measure import (
    measurements as _measurements,
    prompt_measurements as _prompt_measurements,
)
from hif.profile.registry import (
    MEASUREMENT_REGISTRY,
    SUBJECT_MIXED,
    SUBJECT_PROMPT_ONLY,
    run_subjects as _run_subjects,
)


# Absent-signal rendering.
#
# StabilityMetrics components are Optional: None means the signal is ABSENT —
# not measurable for this run (e.g. partial-access API backends have no
# input-side series), which is deliberately distinct from a pinned/degenerate
# value (an early defect, since fixed by the absent-not-pinned
# rule in hif/metrics/stability.py). The CLI renders absent signals as
# "n/a", never as numbers; non-None values print normally on text and
# Absent means "this run produced no evidence for that quantity" — a backend
# that cannot teacher-force, an analysis stage that did not run. It is
# deliberately distinct from a measured value and is never rendered as a
# number.
ABSENT_TEXT = "absent (not measurable on this backend/run)"

# ...except where the run never asked. Every measurement used to be attempted
# on every run, so "absent" and "not measurable" were the same statement and
# one string could serve both. `output_nucleus_entropy_bits` is the first row
# that is opt-in, and ABSENT_TEXT libels the run for it: printed against gpt2
# on hf — one of exactly two backends that CAN produce it — "not measurable on
# this backend" is false, and it points a reader at their backend when the
# answer is a flag they did not pass.
NOT_REQUESTED_TEXT = "absent (not requested — pass {flag})"

# key -> the flag that asks for it. A row here is absent-by-default; a row not
# here is absent only when the run could not produce it.
OPT_IN_FLAGS: dict[str, str] = {
    "output_nucleus_entropy_bits": "--entropy-percentile",
}


def _absent_text(key: str, profile) -> str:
    """Which absence this is: not asked for, or not obtainable."""
    flag = OPT_IN_FLAGS.get(key)
    if flag is None:
        return ABSENT_TEXT
    # Asked for and still absent is a real absence — the run tried and the
    # data did not support it (see percentile_entropy_bits). Only an unasked
    # measurement gets the softer, accurate line.
    requested = getattr(
        getattr(getattr(profile, "config", None), "generation", None),
        "entropy_percentile", None,
    )
    return ABSENT_TEXT if requested is not None else NOT_REQUESTED_TEXT.format(flag=flag)


def _print_measurements(p) -> None:
    """The measurement set, one row per quantity, in natural units.

    There is no Level column and no Normalized column, by design. A level is
    an inference requiring a null this project never established; the
    normaliser (log2 of the vocabulary size) put tokenizer metadata into a
    column labelled behaviour. Absent measurements are named and left absent.

    Quantities whose subject on this run is the prompt rather than the model
    get their own table below, for the same reason the record puts them in
    their own block: they are not measurements of this model.
    """
    vals = _measurements(p)
    subjects = _run_subjects(p)
    input_surrogate = p.findings.surrogate_model_name
    output_surrogate = p.findings.output_distribution_surrogate_name

    # No Subject column: the whole table is one subject — this model. Rows
    # that would have said something else are not in it. The one exception a
    # reader must not miss is `mixed`, which is marked.
    table = Table(title=f"Measurements — {p.model.name}", show_header=True)
    table.add_column("Quantity", style="bold", no_wrap=True)
    table.add_column("Value", justify="right")
    table.add_column("Unit / definition")

    any_mixed = False
    for m in MEASUREMENT_REGISTRY:
        subject = subjects.get(m.key)
        if subject == SUBJECT_PROMPT_ONLY:
            continue
        starred = (m.surrogate_group == "input" and input_surrogate) or (
            m.surrogate_group == "output" and output_surrogate
        )
        marks = " *" if starred else ""
        if subject == SUBJECT_MIXED:
            marks += " †"
            any_mixed = True
        v = vals.get(m.key)
        table.add_row(
            f"{m.name}{marks}",
            _absent_text(m.key, p) if v is None else f"{v:.6g}",
            m.unit,
        )
    console.print(table)

    surrogate_names = sorted({n for n in (input_surrogate, output_surrogate) if n})
    if surrogate_names:
        names = ", ".join(repr(n) for n in surrogate_names)
        console.print(
            f"[dim]* computed via surrogate model {names} (teacher-forcing "
            f"proxy) — a measurement of the surrogate over the target's text, "
            f"not of the target model.[/dim]"
        )
    if any_mixed:
        console.print(
            "[dim]† subject 'mixed': couples a series derived from the target "
            "with one derived from the surrogate, so it is a claim about the "
            "pair rather than about the target alone.[/dim]"
        )
    _slope = p.findings.similarity_trend_slope
    console.print(
        "[dim]Similarity trend slope: "
        + (f"{_slope:+.6g}" if _slope is not None else "absent (fewer than two output steps)")
        + " (OLS slope of the per-step candidate-cloud similarity).[/dim]"
    )
    console.print(
        "[dim]No thresholds, levels, or verdicts: this instrument describes "
        "behaviour, it does not decide anything. Run `hif schema` for the full "
        "unit definitions.[/dim]"
    )
    console.print()
    _print_prompt_measurements(p, subjects)


def _print_prompt_measurements(p, subjects: dict) -> None:
    """Prompt-only quantities, kept out of the model's measurement table.

    Printed only when the run produced any. These are real measurements — of
    the prompt, under the reference model named beside each one. They are not
    caveated facts about the model under test; nothing the model did enters
    them, so they cannot vary with its behaviour.
    """
    from hif.profile.measure import _prompt_reference_model

    vals = _prompt_measurements(p)
    if not vals:
        return

    table = Table(title="Prompt measurements — not about this model", show_header=True)
    table.add_column("Quantity", style="bold", no_wrap=True)
    table.add_column("Value", justify="right")
    table.add_column("Reference model", no_wrap=True)
    table.add_column("Unit")

    for m in MEASUREMENT_REGISTRY:
        if m.key not in vals:
            continue
        ref = _prompt_reference_model(m.key, p)
        table.add_row(m.name, f"{vals[m.key]:.6g}", ref or "unknown", m.unit)
    console.print(table)
    console.print(
        "[dim]Subject: prompt-only. Computed from the prompt text under the "
        "reference model shown, with no input from "
        f"{p.model.name} — comparable across targets for exactly that reason, "
        "and reported here rather than as a caveated measurement of the "
        "model. See docs/MEASUREMENTS.md § Subject.[/dim]"
    )
    console.print()


def _print_output_text(p) -> None:
    """Always shown (not just --verbose): the model's full generated output text,
    so a Hash/Findings result can be associated with what the model actually said."""
    output_text = "".join(s.selected_token_str for s in p.output_side.steps)
    console.print(f"[bold]Output[/bold] ({len(p.output_side.generated_ids)} tokens)")
    console.print(f"[dim]{output_text}[/dim]")
    console.print()


def _print_verbose_io(p) -> None:
    """--verbose: additionally show the model's input text and perturbation variants
    (the output text itself is always shown — see _print_output_text)."""
    console.print("[bold]Input[/bold]")
    console.print(f"[dim]{p.prompt.text}[/dim]")
    console.print()

    if p.perturbations:
        console.print("[bold]Perturbation variants[/bold]")
        for rec in p.perturbations:
            for i, variant in enumerate(rec.variants):
                js = rec.sensitivity[i].mean_js_divergence if i < len(rec.sensitivity) else None
                js_str = f"  [cyan]JSD={js:.4f}[/cyan]" if js is not None else ""
                console.print(f"  ({rec.generator} #{i + 1}){js_str}")
                console.print(f"  [dim]{variant[:200]}[/dim]")
        console.print()


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as M:SS.mmm (e.g. 0:02.417, 1:07.031)."""
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:06.3f}"


def _print_latency(timings: dict[str, float]) -> None:
    """--verbose: wall-clock timings for each pipeline stage."""
    table = Table(title="Latency", show_header=True)
    table.add_column("Stage", style="bold")
    table.add_column("Duration", justify="right")
    for stage, secs in timings.items():
        table.add_row(stage, _fmt_duration(secs))
    console.print(table)
    console.print()


def _print_verbose_stats(p) -> None:
    """--verbose: the raw numbers behind the measurement table, plus the
    effective-config notes that change what those numbers mean."""
    table = Table(title="Stats", show_header=True)
    table.add_column("Stat", style="bold")
    table.add_column("Value", justify="right")

    st = p.metrics.stability

    def _fmt_optional(v) -> str:
        # None = absent — never a number.
        return ABSENT_TEXT if v is None else f"{v:.6g}"

    table.add_row("n_perturbation_variants", str(st.n_perturbations))
    table.add_row("input_mean_entropy (bits)", f"{p.input_side.mean_entropy:.6g}")
    table.add_row("input_mean_surprisal (bits)", f"{p.input_side.mean_surprisal:.6g}")
    table.add_row("max_entropy log2|V| (bits)", f"{p.input_side.max_entropy:.6g}")
    if p.metrics.distribution:
        ents = [d.entropy_bits for d in p.metrics.distribution]
        table.add_row("mean_output_entropy (bits)", f"{sum(ents) / len(ents):.6g}")
        table.add_row("max_output_entropy (bits)", f"{max(ents):.6g}")
        table.add_row("min_output_entropy (bits)", f"{min(ents):.6g}")
    exp = getattr(p, "exposure", None)
    if exp is not None and getattr(exp, "candidates", None):
        table.add_row(
            "exposure (divergent steps / analysed)",
            f"{len(exp.exposed_steps)}/{len(exp.candidates)}",
        )
    table.add_row("center entropy_ratio (out/in)",
                  _fmt_optional(p.center.entropy_ratio))
    table.add_row("prompt/output cosine distance",
                  _fmt_optional(p.center.prompt_output_cosine_distance))
    table.add_row("input_tokens", str(len(p.input_side.prompt_token_ids)))
    table.add_row("output_tokens", str(len(p.output_side.generated_ids)))

    # Effective-config notes: adjustments that change what the numbers mean.
    requested_k = p.config.generation.top_k
    effective_k = p.output_side.top_k
    if effective_k != requested_k:
        table.add_row("top-K", f"{effective_k} (backend max; {requested_k} requested)")
    else:
        table.add_row("top-K", str(effective_k))
    table.add_row("embedder", p.config.embedding.model_name)

    console.print(table)
    console.print()


def _print_subject_degradation(info) -> None:
    """Which measurements stop being about the target on this backend.

    Two separate statements, and the difference matters. Some quantities are
    prompt-only on every backend — no access tier can make them about a model.
    Others are the target's own when the backend teacher-forces, and become
    prompt-only when `--surrogate` reads the prompt in the target's place; on
    those backends they leave `measurements` for `prompt_measurements`.
    """
    from hif.profile.registry import SUBJECT_PROMPT_ONLY as _PO

    always = [
        m.key for m in MEASUREMENT_REGISTRY
        if m.subject == _PO and m.subject_under_surrogate is None
    ]
    if always:
        console.print(
            f"  [yellow]⊘ never about the target:[/yellow] {', '.join(always)} "
            "[dim](prompt-only on every backend)[/dim]"
        )
    if info.teacher_forcing:
        return
    degrades = [
        m.key for m in MEASUREMENT_REGISTRY
        if m.subject_under_surrogate == _PO
    ]
    if degrades:
        console.print(
            f"  [yellow]⊘ prompt-only under --surrogate:[/yellow] "
            f"{', '.join(degrades)}"
        )
        console.print(
            "  [dim]This backend cannot teacher-force, so --surrogate reads "
            "the prompt with a small local model instead. Those numbers "
            "describe the prompt under that reference model, not this "
            "backend's model, and are reported in `prompt_measurements` "
            "rather than `measurements`.[/dim]"
        )
