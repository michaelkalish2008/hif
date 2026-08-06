"""`hif profile` — the full pipeline on one (model, prompt) pair."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from hif.cli._app import (
    CHARTS_HELP,
    TRACE_DIR_HELP,
    UNITS_HELP,
    app,
    console,
    err_console,
)
from hif.cli._config import (
    _check_acquisition,
    _check_mode,
    _explicit_generation_params,
    _load_config_file,
)
from hif.cli._load import (
    _resolve_backend,
)
# Called through their modules, not bound by name at import: a test that
# patches `hif.cli._load._load_model` must affect every caller, and
# `from ... import _load_model` would bind a copy this module keeps using.
from hif.cli import _load
from hif.cli import _run
from hif.cli._output import (
    ABSENT_TEXT,
    _print_latency,
    _print_measurements,
    _print_output_text,
    _print_verbose_io,
    _print_verbose_stats,
)

# Canonical measurement extraction lives in hif/profile/measure.py so the CLI
# tables and the SessionEngine record path report identical numbers; the
# registry rows they are keyed on live in hif/profile/registry.py and the wire
# record in hif/profile/record.py.
from hif.profile.measure import (
    measurements as _measurements,
    prompt_measurements as _prompt_measurements,
)
from hif.profile.record import (
    profile_hash as _profile_hash,
    signals_record as _signals_record,
)
from hif.profile.registry import (
    MEASUREMENT_KEYS,
    MEASUREMENT_UNITS,
    SUBJECT_PROMPT_ONLY,
    run_subjects as _run_subjects,
)



@app.command()
def profile(
    ctx: typer.Context,
    model_name: str = typer.Argument(..., help="Model name (e.g. gpt2)"),
    prompt: str = typer.Argument(..., help="Prompt text"),
    regime: str = typer.Option("ordinary_conversation", help="Prompt regime"),
    backend: str = typer.Option(
        "hf",
        help="Model backend: hf | tlens | ollama | openai | anthropic | gemini",
    ),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Write derived reports (technical + public markdown, --charts "
        "plots) here. Default: nothing is written to disk — results print to "
        "the terminal only (privacy-first compute-and-discard).",
    ),
    max_new_tokens: int = typer.Option(64, help="Maximum new tokens to generate"),
    top_k: int = typer.Option(50, help="Top-K candidates per step"),
    config_file: Optional[Path] = typer.Option(
        None,
        help="TOML run config (tables mirror RunConfig: [generation], "
        "[perturbation], [trajectory], [attention], [semantic_field], ...). "
        "CLI flags you pass explicitly override the file.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Opt-in traceability: persist the full profile artifact (raw "
        "per-step top-K distributions — reconstructable content) so signals "
        "can be recomputed or audited later without re-running the model. "
        "Default off: compute-and-discard.",
    ),
    trace_dir: Optional[Path] = typer.Option(
        None, "--trace-dir", help=TRACE_DIR_HELP
    ),
    charts: bool = typer.Option(False, "--charts", help=CHARTS_HELP),
    diagnostics: bool = typer.Option(
        False,
        "--diagnostics",
        help="Also run the two optional analysis stages — attention capture "
        "and the semantic field. Neither produces a measurement in hif-v4; "
        "their blocks ship in the --trace artifact as evidence. Off by "
        "default because both cost extra compute.",
    ),
    application: Optional[str] = typer.Option(
        None,
        help="Application archetype (support-chatbot, rag-qa, coding-assistant, summarization, extraction, classification, agent-tool-use, document-understanding). "
        "Labels the run and supplies the default --analysis-window; both are "
        "recorded in the JSON record. It does not change how anything is "
        "measured.",
    ),
    mode: str = typer.Option(
        "fast",
        help="fast: fewer perturbation variants. "
        "audit: full perturbation set. "
        "Input is always passed in full regardless of mode.",
    ),
    variant_io: bool = typer.Option(
        False,
        "--variant-io",
        help="Include a `variant_io` block in the --json record: each "
        "perturbation variant's input text and the continuation it elicited "
        "(null where none was — synthesized-input tier, or a failure). "
        "Opt-in because it adds model-generated content to every record; "
        "outputs live in records, inputs stay immutable.",
    ),
    acquisition: str = typer.Option(
        "elicited-output",
        "--acquisition",
        help="Ceiling on what this run may bring into existence. "
        "observational: read the prompt as given and the one continuation the "
        "run produces — nothing else is sent to the model and no new model "
        "output exists afterwards. synthesized-input: additionally author "
        "paraphrased prompts and teacher-force over them (the model still does "
        "not generate). elicited-output (default): additionally let the model "
        "generate variant continuations and trajectory branches. "
        "Measurements above the ceiling are absent, not zero. "
        "Run `hif schema` to see each measurement's acquisition tier.",
    ),
    lite: bool = typer.Option(
        False,
        "--lite",
        help="Skip every stage that costs an extra generation pass or an "
        "embedding sweep: perturbation variants, trajectory branches, and "
        "per-step candidate geometry. The entropy-side measurements are "
        "unchanged; the ones those stages feed are omitted, not zeroed. "
        "Overrides --mode and --config-file for the stages it disables.",
    ),
    analysis_window: Optional[str] = typer.Option(
        None,
        help="Maximum output tokens to analyze (does not truncate inference). "
        "Integer or 'adaptive' (default: adaptive = analyze all output).",
    ),
    metric: Optional[str] = typer.Option(
        None,
        help="Print ONE measurement, in its natural unit, and exit. "
        "Run `hif schema` for the full list with unit definitions.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show model input/output text, perturbation variants, full numeric stats, "
        "effective-config notes, and full internal logging (pipeline + HTTP chatter)",
    ),
    output_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON profile"),
    units: bool = typer.Option(False, "--units", help=UNITS_HELP),
    truncate: Optional[int] = typer.Option(
        None,
        "--truncate",
        help="Truncate input to N tokens before analysis. Results reflect truncated context only.",
    ),
    surrogate: bool = typer.Option(
        False,
        "--surrogate",
        help="Recover the input-side measurements (input_entropy_shift_bits, "
        "input_entropy_std_bits, prompt_surprisal_excess_bits) on backends "
        "that cannot teacher-force (ollama, openai, gemini, "
        "anthropic) by teacher-forcing a small local proxy model over the "
        "prompt+output — the same technique the study harness uses. Ignored when the "
        "target backend already teacher-forces (hf/tlens). Implied by "
        "--surrogate-model, so passing that alone is enough.",
    ),
    surrogate_model: Optional[str] = typer.Option(
        None,
        "--surrogate-model",
        help="Open-weight HF model id used for --surrogate (default: Llama 3.2 1B, "
        "ungated mirror). Passing this flag implies --surrogate — you don't need both.",
    ),
) -> None:
    """Run the full hif pipeline on a single (model, prompt) pair."""
    # --surrogate-model implies --surrogate: passing a model id but forgetting
    # the boolean flag used to silently do nothing (input-side signals still
    # zeroed with a warning).
    if surrogate_model is not None:
        surrogate = True
    effective_surrogate_model = surrogate_model or "unsloth/Llama-3.2-1B"

    if verbose:
        from hif.utils.logging import configure_logging

        configure_logging(verbose=True)

    base_config = _load_config_file(config_file) if config_file is not None else None
    # Which options the user passed explicitly (vs. their defaults) — these
    # override --config-file values in _make_run_config.
    explicit = _explicit_generation_params(ctx)

    # A --config-file [generation] seed wins over the CLI *default* (an
    # explicit --seed still beats the file) — the seed used for the run,
    # hashing, and labels must match config.generation.seed.
    if base_config is not None and "seed" not in explicit:
        seed = base_config.generation.seed

    # --trace-dir implies --trace (passing a destination but forgetting the
    # boolean should not silently discard the artifacts you asked for).
    if trace_dir is not None:
        trace = True

    if charts and output_dir is None:
        err_console.print(
            "[red]--charts writes plot files — pass --output-dir to say "
            "where.[/red]"
        )
        raise typer.Exit(3)

    _check_mode(mode)
    _check_acquisition(acquisition)

    if metric is not None and metric not in MEASUREMENT_KEYS:
        err_console.print(
            f"[red]--metric must be one of: {', '.join(MEASUREMENT_KEYS)} — "
            f"got {metric!r}[/red]"
        )
        raise typer.Exit(3)

    # Chart guard: fail fast when no chart draws the requested measurement,
    # BEFORE loading the model — the same message generate_signal_plots would
    # raise at the end of the run, minus the wasted pipeline.
    if metric is not None and charts:
        from hif.viz.registry import NEAREST_CHART, SIGNALS_BY_ID, resolve_signal

        if resolve_signal(metric) is None:
            near = SIGNALS_BY_ID[NEAREST_CHART[metric]]
            err_console.print(
                f"[red]No chart draws the measurement {metric!r} directly.[/red]\n"
                f"[yellow]The nearest chart is {near.id!r} ({near.label}) — it "
                f"shows the series or companion quantity behind it. Drop "
                f"--charts to print the number alone, or use --charts without "
                f"--metric for the full dashboard.[/yellow]"
            )
            raise typer.Exit(3)

    # Capability guard: fail fast when the requested metric can't be produced by
    # the chosen backend, BEFORE loading the model or running the pipeline.
    # (This is what would have caught `--metric stability --backend ollama`.)
    if metric is not None:
        # The guard has to ask about the backend the run will really use, but
        # must not print the auto-route notice a second time — _load_model
        # prints it when it performs the route.
        _effective_backend = _resolve_backend(model_name, backend, warn=False)
        from hif.models.capabilities import metric_support
        # The attention rows are gated on the analysis STAGE, not the backend
        # (nothing reads the target's attention). --diagnostics turns it on;
        # so can a --config file, which is why both are consulted here.
        _attention_on = bool(diagnostics) or bool(
            base_config is not None and base_config.attention.enabled
        )
        # A teacher-forcing surrogate changes what a restricted backend can
        # produce (input-side rows, and the candidate-cloud rows it rebuilds
        # from the target's actual continuation), so the guard is asked the
        # question the run will actually face. Which surrogate recovery applies
        # to which measurement is capabilities.py's to know, not the CLI's —
        # this used to be an ad-hoc exemption here for the input-side set only,
        # which silently refused `--metric output_entropy_bits --surrogate` on
        # a selected-only backend that produces it perfectly well.
        _reason = metric_support(
            metric, _effective_backend,
            attention_enabled=_attention_on, surrogate=bool(surrogate),
        )
        if _reason is not None:
            err_console.print(
                f"[red]Cannot compute --metric {metric} on --backend "
                f"{_effective_backend}.[/red]\n[yellow]{_reason}[/yellow]\n"
                "[dim]Run `hif models` to see which measurements each backend "
                "supports.[/dim]"
            )
            raise typer.Exit(3)

    # mode affects perturbation count only; input is always full unless --truncate is set
    n_variants = 2 if mode == "fast" else 5

    # Input truncation — explicit user choice only, never a silent default
    input_truncated = False
    if truncate is not None:
        if truncate <= 0:
            err_console.print("[red]--truncate must be a positive integer[/red]")
            raise typer.Exit(3)
        # Approximate token truncation by whitespace-split words (no tokenizer dep here)
        words = prompt.split()
        if len(words) > truncate:
            prompt = " ".join(words[:truncate])
            input_truncated = True
            err_console.print(
                f"[yellow]Warning: Input truncated to {truncate} tokens — "
                "results reflect truncated context only.[/yellow]"
            )

    # Validate --application against the archetype registry and apply defaults
    archetype_def = None
    if application:
        from hif.archetypes import UnknownArchetypeError, load_archetype

        try:
            archetype_def = load_archetype(application)
        except UnknownArchetypeError as exc:
            err_console.print(
                f"[red]Unknown archetype {application!r}. "
                f"Valid: {', '.join(exc.valid_ids)}[/red]"
            )
            raise typer.Exit(3)
        # Apply the archetype's default analysis window when not set by the user
        if analysis_window is None:
            analysis_window = str(archetype_def.default_analysis_window)

    # Validate analysis_window's form. The value itself is recorded in the
    # run's extras (and nothing else): no pipeline stage consumes it, so no
    # parsed copy is kept — a local integer sat here for a while, assigned and
    # never read, which made the flag look wired into analysis when it is a
    # recorded label.
    if analysis_window and analysis_window != "adaptive":
        try:
            int(analysis_window)
        except ValueError:
            err_console.print(f"[red]--analysis-window must be an integer or 'adaptive', got {analysis_window!r}[/red]")
            raise typer.Exit(3)

    if not output_json:
        console.print(f"[bold]hif Profile[/bold]")
        if application:
            console.print(f"  Application: {application}")
        console.print(f"  Model:   {model_name} ({backend})")
        console.print(f"  Prompt:  {prompt[:80]!r}")
        # In single-profile mode the regime is a category label only — it tags
        # the run (run_id) but does not change any metric computation
        # and nothing is compared against it. Say so, so the number isn't
        # mistaken for a regime-relative score.
        console.print(f"  Regime:  {regime} [dim](label only — not a comparison)[/dim]")
        console.print(f"  Seed:    {seed}")
        console.print(f"  Mode:    {mode}")
        if analysis_window:
            console.print(f"  Analysis window: {analysis_window}")
        console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        timings: dict[str, float] = {}
        t_total = time.perf_counter()
        try:
            t0 = time.perf_counter()
            model = _load._load_model(model_name, backend)
            timings["model_load"] = time.perf_counter() - t0
            progress.update(task, description="Loading embedder...")
            t0 = time.perf_counter()
            embedder = _load._load_embedder()
            timings["embedder_load"] = time.perf_counter() - t0
            progress.update(task, description="Running pipeline...")
            t0 = time.perf_counter()
            variant_output_sink: Optional[dict] = {} if variant_io else None
            p, trace_path = _run._run_single_profile(
                model_name=model_name,
                prompt=prompt,
                regime=regime,
                backend=backend,
                seed=seed,
                output_dir=output_dir,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                charts=charts,
                model=model,
                embedder=embedder,
                n_perturbation_variants=n_variants,
                diagnostics=diagnostics,
                metric=metric,
                surrogate=surrogate,
                surrogate_model_id=effective_surrogate_model,
                trace=trace,
                trace_dir=trace_dir,
                base_config=base_config,
                explicit=explicit,
                lite=lite,
                acquisition=acquisition,
                variant_output_sink=variant_output_sink,
            )
            timings["pipeline"] = time.perf_counter() - t0
            progress.update(task, description="Done.")
        except Exception as exc:
            err_console.print(f"[red]Pipeline error: {exc}[/red]")
            raise typer.Exit(1)
        timings["total"] = time.perf_counter() - t_total

    h = _profile_hash(model_name, prompt, seed)

    # --metric: print one measurement, in its natural unit, and exit.
    if metric is not None:
        from hif.profile.measure import _prompt_reference_model

        subject = _run_subjects(p).get(metric)
        prompt_only = subject == SUBJECT_PROMPT_ONLY
        vals = _prompt_measurements(p) if prompt_only else _measurements(p)
        value = vals.get(metric)
        if value is None:
            err_console.print(
                f"[red]--metric {metric}: {ABSENT_TEXT}.[/red]\n"
                f"[dim]{MEASUREMENT_UNITS[metric]}[/dim]"
            )
            raise typer.Exit(1)
        # A single value has no block around it to say who it is about, so the
        # subject travels with it — a bare number is exactly how a prompt-only
        # quantity gets mistaken for a fact about the model.
        reference = _prompt_reference_model(metric, p) if prompt_only else None
        if output_json:
            payload = {"metric": metric, "value": value,
                       "unit": MEASUREMENT_UNITS[metric], "subject": subject}
            if prompt_only:
                payload["reference_model"] = reference
                payload["about"] = "the prompt, not the model named in --model"
            print(json.dumps(payload))
        else:
            console.print(f"{metric} = {value:.6g}")
            console.print(f"[dim]{MEASUREMENT_UNITS[metric]}[/dim]")
            console.print(f"[dim]subject: {subject}[/dim]")
            if prompt_only:
                console.print(
                    f"[dim]This describes the prompt under reference model "
                    f"{reference or 'unknown'}, not {model_name}.[/dim]"
                )
        return

    if output_json:
        # Derived signals only — the raw profile (per-step distributions) is
        # never emitted here. Full artifacts are the --trace opt-in, and the
        # record links to them via trace_path.
        extras: dict = {}
        if application:
            extras["application"] = application
        if analysis_window:
            extras["analysis_window"] = analysis_window
        if input_truncated:
            extras["input_truncated"] = True
            extras["input_truncate_tokens"] = truncate
        if variant_io:
            from hif.perturbation.authored import variant_io_block

            # What was sent per variant and what came back — outputs live in
            # records, so the record is the review surface for elicited text.
            extras["variant_io"] = variant_io_block(p, variant_output_sink or {})
        record = _signals_record(
            p,
            model_name=model_name,
            backend=backend,
            regime=regime,
            seed=seed,
            prompt=prompt,
            latency=timings,
            trace_path=str(trace_path) if trace_path is not None else None,
            include_units=units,
            extras=extras,
        )
        print(json.dumps(record, indent=2))
        return

    console.print(f"[green]Profile complete.[/green] Hash: {h}")
    console.print()

    _print_output_text(p)

    if verbose:
        _print_verbose_io(p)

    _print_measurements(p)

    if verbose:
        _print_verbose_stats(p)
        _print_latency(timings)

    if trace_path is not None:
        console.print(f"Trace artifact: [cyan]{trace_path}[/cyan]")
    if output_dir is not None:
        console.print(f"Outputs written to: [cyan]{output_dir}/[/cyan]")



