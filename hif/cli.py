"""HI command-line interface — the command surface.

Every `hif` subcommand is defined here: its flags, its help text, the checks it
runs before touching a model, and the order it does things in. The concerns the
commands share live in siblings, so this file is the list of things a user can
ask for and nothing else:

    hif/cli_base.py    the typer app, the two consoles, shared option help
    hif/cli_config.py  --config-file / CLI precedence -> RunConfig
    hif/cli_load.py    backend resolution, model / embedder / surrogate loads
    hif/cli_render.py  terminal presentation of a finished profile
    hif/cli_compat.py  whether two artifacts may be compared at all

`pyproject.toml` names `hif.cli:app` as the entry point, so this module stays
the one that assembles the app.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from hif.cli_base import (
    CHARTS_HELP,
    TRACE_DIR_HELP,
    UNITS_HELP,
    _emit_json_line,
    app,
    console,
    err_console,
)
from hif.cli_compat import (
    _artifact_signal_set_version,
    _modality_mismatch_exit,
    _profile_modality,
    _signal_set_family,
    _signal_set_mismatch_exit,
)
from hif.cli_config import (
    _check_mode,
    _explicit_generation_params,
    _load_config_file,
    _make_run_config,
)
from hif.cli_load import (
    _build_multimodal_input,
    _check_surrogate_candidates,
    _live_models_for_backend,
    _load_embedder,
    _load_model,
    _load_surrogate,
    _resolve_backend,
    _resolve_validation_corpus,
)
from hif.cli_render import (
    ABSENT_TEXT,
    _print_latency,
    _print_measurements,
    _print_output_text,
    _print_subject_degradation,
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
    RECORD_SCHEMA_VERSION as SIGNAL_RECORD_VERSION,
    profile_hash as _profile_hash,
    signals_record as _signals_record,
)
from hif.profile.registry import (
    MEASUREMENT_KEYS,
    MEASUREMENT_REGISTRY,
    MEASUREMENT_UNITS,
    SIGNAL_SET_VERSION,
    SUBJECT_LEGEND,
    SUBJECT_PROMPT_ONLY,
    run_subjects as _run_subjects,
)


# ---------------------------------------------------------------------------
# The shared pipeline call
# ---------------------------------------------------------------------------


def _run_single_profile(
    model_name: str,
    prompt: str,
    regime: str,
    backend: str,
    seed: int,
    output_dir: Optional[Path],
    max_new_tokens: int,
    top_k: int,
    charts: bool = False,
    model=None,
    embedder=None,
    n_perturbation_variants: int = 2,
    mm_input=None,
    diagnostics: bool = False,
    metric: Optional[str] = None,
    surrogate: bool = False,
    surrogate_model_id: str = "unsloth/Llama-3.2-1B",
    trace: bool = False,
    trace_dir: Optional[Path] = None,
    base_config=None,
    explicit: frozenset = frozenset(),
) -> "tuple[BehavioralRangeProfile, Optional[Path]]":
    """Core pipeline: build the profile in memory, return (profile, trace_path).

    Privacy-first default: NOTHING is written to disk. `output_dir` opts into
    the derived reports (markdown, charts); `trace` opts into persisting the
    full profile artifact (raw per-step top-K distributions) for
    traceability/reconstruction — trace_path is None unless it did.

    When mm_input is provided (a MultimodalInput with media parts), it is
    passed to build_profile in place of the text prompt; `prompt` still names
    the text portion for hashing/labels.
    """
    from hif.engine import SessionEngine
    from hif.profile.render_markdown import render_public, render_technical

    config = _make_run_config(model_name, backend, max_new_tokens, top_k, seed, output_dir,
                               diagnostics=diagnostics, base=base_config,
                               explicit=explicit)
    # Apply perturbation variant count from --mode — unless a --config-file
    # set its own perturbation budget and the user didn't pass --mode.
    if base_config is None or "mode" in explicit:
        config.perturbation.n_variants = n_perturbation_variants
    config.traceability.enabled = trace

    if model is None:
        model = _load_model(model_name, backend)
    if embedder is None:
        embedder = _load_embedder()

    surrogate_model = None
    if surrogate:
        if model.supports_teacher_forcing:
            console.print(
                "  [dim]--surrogate ignored: this backend already teacher-forces "
                "(input-side signals come from the target itself).[/dim]"
            )
        else:
            surrogate_model = _load_surrogate(surrogate_model_id)

    engine = SessionEngine(config, model, embedder, surrogate_model)
    profile = engine.profile_one(mm_input if mm_input is not None else prompt,
                                 regime=regime, seed=seed)

    h = _profile_hash(model_name, prompt, seed)

    trace_path: Optional[Path] = None
    if trace:
        resolved_trace_dir = trace_dir or (
            (output_dir / "traces") if output_dir else Path("traces")
        )
        trace_path = engine.write_trace(profile, prompt=prompt, seed=seed,
                                        trace_dir=resolved_trace_dir)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        render_technical(profile, output_dir / f"profile_{h}_technical.md")
        render_public(profile, output_dir / f"profile_{h}_public.md")

    if charts and output_dir is not None:
        from hif.viz import generate_signal_plots
        plots_dir = output_dir / "plots" / h
        if metric is not None:
            # Single metric → a single chart that gives the printed number meaning.
            res = generate_signal_plots(profile, plots_dir, only_signal=metric)
            chart = res.get(metric, {}).get("html")
            if chart is not None:
                console.print(f"  [dim]Chart: {chart}[/dim]")
        else:
            res = generate_signal_plots(profile, plots_dir)
            index = res.get("index", {}).get("html")
            if index is not None:
                console.print(f"  [dim]Dashboard: {index}[/dim]")

    return profile, trace_path


# ---------------------------------------------------------------------------
# Commands — Local plane
# ---------------------------------------------------------------------------


@app.command()
def profile(
    ctx: typer.Context,
    model_name: str = typer.Argument(..., help="Model name (e.g. gpt2)"),
    prompt: str = typer.Argument(..., help="Prompt text"),
    regime: str = typer.Option("ordinary_conversation", help="Prompt regime"),
    backend: str = typer.Option(
        "hf",
        help="Model backend: hf | tlens | ollama | openai | anthropic | gemini | "
        "hf-vlm | openai-vlm. For image inputs (--input) use an explicit VLM "
        "backend: hf-vlm (local AutoModelForImageTextToText checkpoints, e.g. "
        "SmolVLM/Gemma 3 multimodal) or openai-vlm (hosted vision API, e.g. gpt-4o).",
    ),
    input_files: list[Path] = typer.Option(
        [],
        "--input",
        help="Image file (PNG/JPEG) to include as model input; repeatable. "
        "Images are presented before the prompt text. Requires --backend "
        "hf-vlm or openai-vlm.",
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
        "(the two attention-row entropy measurements) and the semantic field "
        "(centroid veer). Both cost extra compute, so they are off by default; "
        "the measurements they produce are simply absent from the record when "
        "they have not run.",
    ),
    application: Optional[str] = typer.Option(
        None,
        help="Application archetype (support-chatbot, rag-qa, coding-assistant, summarization, extraction, classification, agent-tool-use, multimodal-qa, document-understanding). "
        "Labels the run and supplies the default --analysis-window; both are "
        "recorded in the JSON record. It does not change how anything is "
        "measured.",
    ),
    mode: str = typer.Option(
        "fast",
        help="fast: fewer perturbation variants. "
        "audit: full perturbation set (multimodal: exhaustive grid sweep). "
        "Input is always passed in full regardless of mode.",
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
        help="Recover the input-side signals (stability, surprise, io_correlation, "
        "wager) on backends that cannot teacher-force (ollama, openai, gemini, "
        "anthropic) by teacher-forcing a small local proxy model over the "
        "prompt+output — the same technique the study harness uses. Ignored when the "
        "target backend already teacher-forces (hf/tlens/hf-vlm). Implied by "
        "--surrogate-model, so passing that alone is enough.",
    ),
    surrogate_model: Optional[str] = typer.Option(
        None,
        "--surrogate-model",
        help="Open-weight HF model id used for --surrogate (default: Llama 3.2 1B, "
        "ungated mirror). Passing this flag implies --surrogate — you don't need both.",
    ),
) -> None:
    """Run the full HI pipeline on a single (model, prompt) pair."""
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

    # Multimodal input: validate image files and assemble the input up front,
    # before any model work (exit 3 on bad files).
    mm_input = None
    run_modality = "text"
    if input_files:
        if backend not in ("hf-vlm", "openai-vlm"):
            err_console.print(
                "[red]--input requires a multimodal backend: use --backend "
                "hf-vlm (local VLM) or --backend openai-vlm (hosted vision API).[/red]"
            )
            raise typer.Exit(3)
        mm_input = _build_multimodal_input(list(input_files), prompt)
        run_modality = mm_input.modality

    # mode affects perturbation count only; input is always full unless --truncate is set
    if mm_input is not None:
        # Multimodal: audit = exhaustive grid-mask sweep (n_variants<=0 means
        # every cell, 16 on the default 4x4 grid); fast = default variant budget.
        n_variants = 0 if mode == "audit" else 2
    else:
        n_variants = 2 if mode == "fast" else 5

    # Input truncation — explicit user choice only, never a silent default
    input_truncated = False
    if truncate is not None:
        if mm_input is not None:
            err_console.print(
                "[red]--truncate is not supported with --input (image runs "
                "always send the full input — media inputs are the "
                "experimental condition).[/red]"
            )
            raise typer.Exit(3)
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
        console.print(f"[bold]HI Profile[/bold]")
        if application:
            console.print(f"  Application: {application}")
        console.print(f"  Model:   {model_name} ({backend})")
        console.print(f"  Prompt:  {prompt[:80]!r}")
        if input_files:
            for f in input_files:
                console.print(f"  Input:   {f}")
            console.print(f"  Modality: {run_modality}")
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
            model = _load_model(model_name, backend)
            timings["model_load"] = time.perf_counter() - t0
            progress.update(task, description="Loading embedder...")
            t0 = time.perf_counter()
            embedder = _load_embedder()
            timings["embedder_load"] = time.perf_counter() - t0
            progress.update(task, description="Running pipeline...")
            t0 = time.perf_counter()
            p, trace_path = _run_single_profile(
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
                mm_input=mm_input,
                diagnostics=diagnostics,
                metric=metric,
                surrogate=surrogate,
                surrogate_model_id=effective_surrogate_model,
                trace=trace,
                trace_dir=trace_dir,
                base_config=base_config,
                explicit=explicit,
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

    # Region sensitivity (perturbation-JSD heatmap; multimodal runs only)
    rs = getattr(p, "region_sensitivity", None)
    if rs is not None:
        console.print("[bold]Region sensitivity[/bold]")
        console.print(
            "[dim]Cells that materially affected the model's response behavior[/dim]"
        )
        console.print(rs.to_text_grid())
        console.print()

    if verbose:
        _print_verbose_stats(p)
        _print_latency(timings)

    if trace_path is not None:
        console.print(f"Trace artifact: [cyan]{trace_path}[/cyan]")
    if output_dir is not None:
        console.print(f"Outputs written to: [cyan]{output_dir}/[/cyan]")



@app.command()
def models(
    backend: Optional[str] = typer.Option(
        None, help="Show only this backend (hf, tlens, ollama, openai, anthropic, gemini, hf-vlm, openai-vlm)."
    ),
    list_live: bool = typer.Option(
        False, "--list",
        help="Query each backend's actual model catalog right now (needs the provider's API key, or a "
             "running Ollama server) instead of showing static examples — use this when an example model "
             "from the docs turns out to be retired/unavailable.",
    ),
    surrogates: bool = typer.Option(
        False, "--surrogates",
        help="List recommended --surrogate-model choices (small open-weight models for recovering "
             "input-side signals on closed/Ollama backends via --surrogate) and check each is currently "
             "reachable and ungated on the Hugging Face Hub.",
    ),
) -> None:
    """List the backends you can profile, example models, and which signals each supports.

    Use this to answer "what can I test, and what will I get?" before running a
    profile. Signal availability depends on the backend: input-side signals
    (Stability, Surprise, I/O Correlation, Wager) need teacher forcing — open
    models only. The attention readings (Spread, Horizon) do NOT depend on the
    backend: they come from a separate analysis encoder reading text, so every
    backend can produce them once --diagnostics runs that stage.
    Pass --list to check live model availability instead of static examples.
    Pass --surrogates to check --surrogate-model candidates instead.
    """
    from rich.markup import escape
    from hif.models.capabilities import BACKENDS, signals_available

    if surrogates:
        console.print(
            "\n[bold]--surrogate-model candidates[/bold]  [dim](small open-weight models for "
            "--surrogate on backends that can't teacher-force)[/dim]\n"
        )
        for model_id, status in _check_surrogate_candidates():
            marker = "[green]✓ ok[/green]" if status == "ok" else f"[yellow]{status}[/yellow]"
            default_tag = "  [dim](default)[/dim]" if model_id == "unsloth/Llama-3.2-1B" else ""
            console.print(f"  {model_id:<28} {marker}{default_tag}")
        console.print()
        return

    infos = [BACKENDS[backend]] if backend and backend in BACKENDS else list(BACKENDS.values())
    if backend and backend not in BACKENDS:
        err_console.print(f"[red]Unknown backend {backend!r}. Known: {', '.join(BACKENDS)}[/red]")
        raise typer.Exit(1)

    for info in infos:
        tf = "[green]yes[/green]" if info.teacher_forcing else "[dim]no[/dim]"
        console.print(
            f"\n[bold]{info.name}[/bold]  [dim]({info.kind})[/dim]  "
            f"teacher-forcing: {tf}  ·  logprobs: {info.logprobs}"
        )
        console.print(f"  [dim]deps:[/dim]  {escape(info.deps)}")
        console.print(f"  [dim]setup:[/dim] {escape(info.setup)}")
        if list_live:
            live, note = _live_models_for_backend(info.name)
            if live is not None:
                console.print(f"  [green]models (live):[/green] {', '.join(live) if live else '[dim]none[/dim]'}")
            else:
                console.print(f"  [dim]models:[/dim] {', '.join(info.example_models)}  [yellow]({note})[/yellow]")
        else:
            console.print(f"  [dim]models:[/dim] {', '.join(info.example_models)}")
        if info.notes:
            console.print(f"  [dim]{info.notes}[/dim]")
        avail = signals_available(info.name)
        ok = [m for m, v in avail.items() if v]
        no = [m for m, v in avail.items() if not v]
        console.print(f"  [green]✓ signals:[/green] {', '.join(ok)}")
        if no:
            console.print(f"  [yellow]✗ unavailable:[/yellow] {', '.join(no)}")
        console.print(
            "  [dim]attention rows need --diagnostics, not a backend: they are "
            "an analysis encoder's attention over text, so they are available "
            "here as on every backend.[/dim]"
        )
        _print_subject_degradation(info)
    console.print()


@app.command()
def doctor() -> None:
    """Preflight check: dependencies, running services, credentials, and per-backend readiness.

    Run this first. It probes each backend's optional dependencies, checks for
    API keys / a running Ollama server, and reports which backends are ready to
    profile right now — so you don't discover a missing dep or unpulled model
    mid-pipeline.
    """
    import importlib.util
    import os

    from rich.markup import escape
    from hif.models.capabilities import BACKENDS

    def _has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    console.print("\n[bold]HIF doctor[/bold] — environment & backend readiness\n")

    # Core deps
    core_ok = _has("plotly") and _has("numpy")
    console.print(f"  core (numpy, plotly): {'[green]ok[/green]' if core_ok else '[red]missing[/red]'}")
    console.print(f"  embedder (sentence-transformers): "
                  f"{'[green]ok[/green]' if _has('sentence_transformers') else '[yellow]missing — run: pip install sentence-transformers[/yellow]'}")

    # Ollama server reachability
    ollama_up = False
    try:
        import httpx  # type: ignore
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        pulled = []
        try:
            resp = httpx.get(f"{host}/api/tags", timeout=1.5)
            ollama_up = resp.status_code == 200
            if ollama_up:
                pulled = [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:  # noqa: BLE001
            ollama_up = False
        if ollama_up:
            plist = ", ".join(pulled) if pulled else "[dim]none pulled — run `ollama pull llama3.2`[/dim]"
            console.print(f"  ollama server ({host}): [green]running[/green] · pulled: {plist}")
        else:
            console.print(f"  ollama server ({host}): [yellow]not reachable — run `ollama serve`[/yellow]")
    except Exception:  # noqa: BLE001
        console.print("  ollama client (httpx): [dim]not installed — pip install 'hif[ollama]'[/dim]")

    # Credentials
    console.print("\n[bold]credentials[/bold]")
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "HF_TOKEN"):
        console.print(f"  {env}: {'[green]set[/green]' if os.environ.get(env) else '[dim]unset[/dim]'}")

    # Per-backend readiness
    console.print("\n[bold]backends[/bold]")
    dep_probe = {
        "hf": "transformers", "tlens": "transformer_lens", "hf-vlm": "transformers",
        "ollama": "httpx", "openai": "openai", "anthropic": "anthropic",
        "gemini": "google.genai", "openai-vlm": "openai",
    }
    cred_probe = {
        "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY", "openai-vlm": "OPENAI_API_KEY",
    }
    for name, info in BACKENDS.items():
        dep = dep_probe.get(name, "")
        dep_ok = _has(dep.split(".")[0]) if dep else True
        issues = []
        if not dep_ok:
            issues.append(f"missing dep ({escape(info.deps)})")
        cred = cred_probe.get(name)
        if cred and not os.environ.get(cred):
            issues.append(f"{cred} unset")
        if name == "ollama" and not ollama_up:
            issues.append("server not running")
        status = "[green]ready[/green]" if not issues else f"[yellow]{'; '.join(issues)}[/yellow]"
        console.print(f"  {name:11s} {status}")

    console.print("\n[dim]The full measurement set needs an open-weight backend: "
                  "`hif profile gpt2 \"hello\" --backend hf`. "
                  "Run `hif models` for the full capability matrix.[/dim]\n")


@app.command()
def suite(
    model_name: str = typer.Argument(..., help="Model name (e.g. gpt2)"),
    regime: Optional[str] = typer.Option(None, help="Single regime; None = all eight"),
    backend: str = typer.Option("hf", help="Model backend: hf | tlens | ollama"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Path = typer.Option(Path("outputs"), help="Output directory"),
    max_new_tokens: int = typer.Option(64, help="Maximum new tokens to generate"),
    top_k: int = typer.Option(50, help="Top-K candidates per step"),
    charts: bool = typer.Option(False, "--charts", help=CHARTS_HELP),
    units: bool = typer.Option(False, "--units", help=UNITS_HELP),
) -> None:
    """Run the full HI pipeline over the prompt suite (every regime, or one)."""
    from hif.prompts.suite import REGIMES, get_regime

    if regime is not None:
        try:
            selected_regimes = [get_regime(regime)]
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        selected_regimes = list(REGIMES)

    console.print(f"[bold]HI Suite[/bold]")
    console.print(f"  Model:   {model_name} ({backend})")
    console.print(f"  Regimes: {len(selected_regimes)}")
    console.print(f"  Seed:    {seed}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Loading model...", total=None)
        model = _load_model(model_name, backend)
        progress.update(t, description="Loading embedder...")
        embedder = _load_embedder()

    n_ok = 0
    n_failed = 0

    for reg in selected_regimes:
        console.print(f"[bold]Regime:[/bold] {reg.name}")
        for prompt_text in reg.prompts:
            console.print(f"  [dim]{prompt_text[:70]!r}[/dim]")
            try:
                p, _ = _run_single_profile(
                    model_name=model_name,
                    prompt=prompt_text,
                    regime=reg.name,
                    backend=backend,
                    seed=seed,
                    output_dir=output_dir / reg.name,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    charts=charts,
                    model=model,
                    embedder=embedder,
                )
            except Exception as exc:  # noqa: BLE001 — per-prompt isolation
                console.print(f"    [red]error: {exc}[/red]")
                _emit_json_line({
                    "schema_version": SIGNAL_RECORD_VERSION,
                    "regime": reg.name,
                    "prompt": prompt_text,
                    "error": str(exc) or exc.__class__.__name__,
                })
                n_failed += 1
                continue
            _emit_json_line(_signals_record(
                p,
                model_name=model_name,
                backend=backend,
                regime=reg.name,
                seed=seed,
                prompt=prompt_text,
                include_units=units,
            ))
            n_ok += 1

    console.print(
        f"suite complete: {n_ok} ok, {n_failed} failed. "
        f"Reports written to {output_dir}/"
    )
    if n_ok == 0:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# hif compare
# ---------------------------------------------------------------------------
#
# Compare reports the difference between two profiles, measurement by
# measurement, in natural units. It deliberately does NOT emit a verdict.
#
# The previous version bucketed each delta as consistent/moderate/high against
# fixed thresholds and printed "SIGNATURE HELD / SHIFTED / MOVED", exiting 2 on
# MOVED so it could be wired into a release gate. That decision rule was
# audited against pairs of runs known to be identical and produced a ~43%
# false-positive rate. Thresholds and decisions are out of scope for this
# instrument; a delta and its unit are what it can honestly report.


@app.command()
def compare(
    profile_a: Path = typer.Argument(..., help="Path to the first profile JSON"),
    profile_b: Path = typer.Argument(..., help="Path to the second profile JSON"),
    output: Optional[Path] = typer.Option(None, help="Optional output Markdown file"),
    output_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Report the per-measurement difference between two profiles.

    Every row is (A, B, B − A) in the measurement's own natural unit. There is
    no verdict, no band, and no non-zero exit on divergence: deciding whether a
    difference matters needs a null distribution for your workload, which this
    tool does not have and does not fabricate.

    Exit codes: 0 normally, 1 for an unreadable file, 2 for a comparison that
    cannot be made at all (different modalities, or no shared measurements).
    """
    from hif.profile.schema import BehavioralRangeProfile

    for path in (profile_a, profile_b):
        if not path.exists():
            err_console.print(f"[red]File not found: {path}[/red]")
            raise typer.Exit(1)

    raw_a = json.loads(profile_a.read_text())
    raw_b = json.loads(profile_b.read_text())
    pa = BehavioralRangeProfile.model_validate(raw_a)
    pb = BehavioralRangeProfile.model_validate(raw_b)

    # Cross-modality comparison is a different experimental condition — hard
    # error, never a warning. Missing modality on an older profile reads "text".
    modality_a = _profile_modality(pa)
    modality_b = _profile_modality(pb)
    if modality_a != modality_b:
        _modality_mismatch_exit(modality_a, modality_b)

    version_a = _artifact_signal_set_version(raw_a)
    version_b = _artifact_signal_set_version(raw_b)
    if _signal_set_family(version_a) != _signal_set_family(version_b):
        _signal_set_mismatch_exit(version_a, version_b)

    vals_a = _measurements(pa)
    vals_b = _measurements(pb)
    shared = [k for k in MEASUREMENT_KEYS if k in vals_a and k in vals_b]
    if not shared:
        err_console.print(
            "[red]These profiles share no measurements — the comparison would "
            "be empty.[/red]"
        )
        raise typer.Exit(2)

    only_a = [k for k in MEASUREMENT_KEYS if k in vals_a and k not in vals_b]
    only_b = [k for k in MEASUREMENT_KEYS if k in vals_b and k not in vals_a]
    for key in only_a:
        err_console.print(
            f"[yellow]excluded: {key} — absent from {profile_b.name}[/yellow]"
        )
    for key in only_b:
        err_console.print(
            f"[yellow]excluded: {key} — absent from {profile_a.name}[/yellow]"
        )

    if output_json:
        result = {
            "profile_a": str(profile_a),
            "profile_b": str(profile_b),
            "model_a": pa.model.name,
            "model_b": pb.model.name,
            "measurements_a": vals_a,
            "measurements_b": vals_b,
            "delta": {k: vals_b[k] - vals_a[k] for k in shared},
            "units": {k: MEASUREMENT_UNITS[k] for k in shared},
            "excluded": {"only_in_a": only_a, "only_in_b": only_b},
        }
        print(json.dumps(result, indent=2))
        return

    table = Table(title="Profile comparison", show_header=True)
    table.add_column("Measurement", style="bold", no_wrap=True)
    table.add_column(pa.model.name, justify="right")
    table.add_column(pb.model.name, justify="right")
    table.add_column("B − A", justify="right")
    table.add_column("Unit")
    for key in shared:
        table.add_row(
            key,
            f"{vals_a[key]:.6g}",
            f"{vals_b[key]:.6g}",
            f"{vals_b[key] - vals_a[key]:+.6g}",
            MEASUREMENT_UNITS[key].split(" — ")[0],
        )
    console.print(table)
    console.print(
        "[dim]Differences only. Whether a difference matters depends on a null "
        "distribution for your workload, which this tool does not have.[/dim]"
    )

    if output is not None:
        lines = [
            "# hif profile comparison",
            "",
            f"- A: `{profile_a}` ({pa.model.name})",
            f"- B: `{profile_b}` ({pb.model.name})",
            "",
            "| Measurement | A | B | B − A | Unit |",
            "|---|---|---|---|---|",
        ]
        for key in shared:
            lines.append(
                f"| {key} | {vals_a[key]:.6g} | {vals_b[key]:.6g} | "
                f"{vals_b[key] - vals_a[key]:+.6g} | "
                f"{MEASUREMENT_UNITS[key].split(' — ')[0]} |"
            )
        lines += [
            "",
            "Differences only. Whether a difference matters depends on a null "
            "distribution for your workload, which this tool does not have.",
            "",
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines))
        console.print(f"\nComparison written to: [cyan]{output}[/cyan]")


@app.command("validate-model")
def validate_model(
    model_name: str = typer.Argument(..., help="Model name (e.g. a HF VLM checkpoint or gpt-4o)"),
    backend: str = typer.Option(..., "--backend", help="Model backend: hf-vlm | openai-vlm"),
    grid: Optional[str] = typer.Option(
        None, "--grid",
        help="Mask grid as ROWSxCOLS (default: 4x4; 2x2 with --pilot).",
    ),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus",
        help="Directory containing a corpus.jsonl known-answer suite "
        "(default: built-in suite, generated to ~/.hif/validation-corpus/ on first use).",
    ),
    pilot: bool = typer.Option(
        False, "--pilot",
        help="Fast smoke run: 4 images on a 2x2 grid instead of 10 images on 4x4.",
    ),
    seed: int = typer.Option(20260703, help="Corpus-generation / run seed"),
    yes: bool = typer.Option(False, "--yes", help="Skip the full-run confirmation prompt"),
    output_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output"),
) -> None:
    """Validate region-sensitivity measurement for a model against HIF's known-answer suite.

    Runs the model on synthetic tasks where the task-relevant region of each
    image is known by construction, and checks that region sensitivity locates
    it. PASS means the instrument is validated against ground-truth synthetic
    tasks for this model — it is not an accuracy claim about your workload.

    Exit codes: 0 pass, 2 fail, 3 usage error.
    """
    if backend not in ("hf-vlm", "openai-vlm"):
        err_console.print("[red]--backend must be 'hf-vlm' or 'openai-vlm'[/red]")
        raise typer.Exit(3)


    if grid is None:
        grid_tuple = (2, 2) if pilot else (4, 4)
    else:
        try:
            r, c = grid.lower().split("x")
            grid_tuple = (int(r), int(c))
            if grid_tuple[0] <= 0 or grid_tuple[1] <= 0:
                raise ValueError
        except ValueError:
            err_console.print(f"[red]--grid must look like 4x4, got {grid!r}[/red]")
            raise typer.Exit(3)

    corpus_dir = _resolve_validation_corpus(corpus, seed, quiet or output_json)

    if not pilot and not yes:
        n_cells = grid_tuple[0] * grid_tuple[1]
        console.print(
            f"[yellow]Full validation runs 10 images x 3 question variants x "
            f"({n_cells} masked cells + 1 baseline) = {10 * 3 * (n_cells + 1)} "
            "inference runs. This can take a while on local models and incurs "
            "API cost on hosted backends. Use --pilot for a fast smoke run.[/yellow]"
        )
        if not typer.confirm("Proceed?"):
            raise typer.Exit(3)

    from hif.validation.harness import validate_region_sensitivity

    if not quiet and not output_json:
        console.print(
            f"[dim]Validating {model_name} ({backend}) on "
            f"{grid_tuple[0]}x{grid_tuple[1]} grid"
            f"{' (pilot subset)' if pilot else ''}...[/dim]"
        )

    try:
        model = _load_model(model_name, backend)
        result = validate_region_sensitivity(
            model, corpus_dir=corpus_dir, grid=grid_tuple, pilot=pilot, seed=seed,
        )
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3)
    except typer.Exit:
        raise
    except Exception as exc:
        err_console.print(f"[red]Validation run failed: {exc}[/red]")
        raise typer.Exit(1)

    if output_json:
        payload = {
            "model": result.model_id,
            "backend": backend,
            "grid": f"{result.grid[0]}x{result.grid[1]}",
            "threshold": result.threshold,
            "top2_rate": round(result.top2_rate, 4),
            "passed": result.passed,
            "per_image": [
                {
                    "image_id": rec.image_id,
                    "variant": rec.variant,
                    "answer_cell": rec.answer_cell,
                    "answer_cell_rank": rec.answer_cell_rank,
                    "in_top2": rec.in_top2,
                    "cell_jsd": {k: round(v, 6) for k, v in rec.cell_jsd.items()},
                }
                for rec in result.per_image
            ],
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        table = Table(
            title=f"Region-sensitivity validation — {result.model_id}",
            show_header=True,
        )
        table.add_column("Image", style="bold")
        table.add_column("Variant")
        table.add_column("Answer cell")
        table.add_column("Rank", justify="right")
        table.add_column("Top-2")
        for rec in result.per_image:
            table.add_row(
                rec.image_id,
                str(rec.variant),
                f"({rec.answer_cell['row']},{rec.answer_cell['col']})",
                str(rec.answer_cell_rank),
                "[green]yes[/green]" if rec.in_top2 else "[red]no[/red]",
            )
        console.print(table)
        console.print()
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(
            f"{status} — answer cell in top 2 for {result.top2_rate:.0%} of "
            f"runs (threshold {result.threshold:.0%}, "
            f"grid {result.grid[0]}x{result.grid[1]}, n={len(result.per_image)})."
        )
        console.print(
            "[dim]Validated against ground-truth synthetic tasks — a statement "
            "about the measurement instrument on this model, not about your "
            "workload.[/dim]"
        )

    if not result.passed:
        raise typer.Exit(2)


@app.command()
def render(
    profile_json: Path = typer.Argument(..., help="Path to profile JSON"),
    public: bool = typer.Option(False, help="Produce public-facing summary instead of technical"),
    output: Optional[Path] = typer.Option(None, help="Output path (default: alongside JSON)"),
) -> None:
    """Load an existing profile from JSON and re-render Markdown."""
    from hif.profile.render_markdown import render_public, render_technical
    from hif.profile.schema import BehavioralRangeProfile

    if not profile_json.exists():
        console.print(f"[red]File not found: {profile_json}[/red]")
        raise typer.Exit(1)

    p = BehavioralRangeProfile.model_validate_json(profile_json.read_text())

    suffix = "_public.md" if public else "_technical.md"
    if output is None:
        output = profile_json.with_suffix("").with_name(
            profile_json.stem + suffix
        )

    if public:
        render_public(p, output)
    else:
        render_technical(p, output)

    console.print(f"[green]Rendered to:[/green] {output}")


@app.command()
def schema(
    output_json: bool = typer.Option(
        True, "--json/--text",
        help="Emit the machine-readable schema document (default) or a human table.",
    ),
) -> None:
    """Print the measurement registry: every key with its full row.

    Each measurement is emitted as its complete registry row — key, name,
    unit, definition, and its triple (observable, functional, resolution).
    A row carries one name and no coined shorthand; see the
    SIGNAL_SET_VERSION history in hif/profile/registry.py (hif-v3.3).
    This is the contract for `hif profile --json`, `hif suite`
    and `hif batch` records, and the machine-readable mirror of
    docs/MEASUREMENTS.md. Every measurement is in natural units; there is no
    normalised variant and no level.
    """
    if output_json:
        print(json.dumps({
            "schema_version": SIGNAL_RECORD_VERSION,
            "signal_set_version": SIGNAL_SET_VERSION,
            "stdout_format": {
                "hif profile --json": "a single JSON document",
                "prompt_measurements": "present only when the run produced "
                        "prompt-only quantities (see \"subjects\"); they are "
                        "never inside \"measurements\", which carries "
                        "measurements of the model named in the record and "
                        "nothing else.",
                "hif suite": "JSONL, one record per prompt",
                "hif batch": "JSONL, one record per workload row",
                "hif compare --json": "a single JSON document",
                "note": "stdout carries JSON only; progress, warnings and errors "
                        "go to stderr. A failed row is still a record, carrying "
                        "an \"error\" key instead of \"measurements\".",
            },
            # What the resolution field means: the granularity of the
            # underlying series the run-level scalar summarises. Rows with a
            # per-step / per-position resolution have a token-level trace
            # behind the number; "aggregate" rows exist only at run level.
            "resolutions": {
                "aggregate": "the quantity exists only at whole-run level "
                             "(across perturbation variants, trajectory "
                             "branches, or the run's endpoints)",
                "per-step": "one sample per generation step; the scalar "
                            "summarises a per-step trace",
                "per-position": "one sample per prompt/context position; the "
                                "scalar summarises a per-position trace",
            },
            # Whose behaviour each measurement describes. `subject` is the
            # answer when the target's own machinery produced the quantity;
            # `subject_under_surrogate`, when present, is what it becomes once
            # the surrogate named by `surrogate_group` stands in.
            "subjects": dict(SUBJECT_LEGEND),
            "measurements": {
                m.key: {
                    "name": m.name,
                    "unit": m.unit,
                    "definition": m.definition,
                    "observable": m.observable,
                    "functional": m.functional,
                    "resolution": m.resolution,
                    "subject": m.subject,
                    "subject_under_surrogate": m.subject_under_surrogate,
                    "surrogate_group": m.surrogate_group or None,
                }
                for m in MEASUREMENT_REGISTRY
            },
        }, indent=2))
        return

    table = Table(title="hif measurement set", show_header=True)
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Name")
    table.add_column("Unit")
    table.add_column("Resolution")
    table.add_column("Subject")
    table.add_column("Definition")
    for m in MEASUREMENT_REGISTRY:
        subject = m.subject
        if m.subject_under_surrogate is not None:
            subject = f"{m.subject} → {m.subject_under_surrogate} (surrogate)"
        table.add_row(
            m.key, m.name, m.unit, m.resolution, subject, m.definition
        )
    console.print(table)
    console.print("\n[bold]Subject[/bold] — whose behaviour the number describes:")
    for value, gloss in SUBJECT_LEGEND.items():
        console.print(f"  [bold]{value}[/bold] — {gloss}")


# ---------------------------------------------------------------------------
# Commands — Batch (workload runner)
# ---------------------------------------------------------------------------


def _open_records_file(output_dir: Optional[Path]):
    """Open <output-dir>/records.jsonl for the batch stream mirror (or None)."""
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir / "records.jsonl").open("w")


@app.command()
def batch(
    ctx: typer.Context,
    workload: Path = typer.Argument(
        ..., help="Workload JSONL file: one {\"query_id\", \"text\"[, \"image\", \"regime\"]} row per line."
    ),
    model_name: str = typer.Argument(..., help="Model name (e.g. gpt2)"),
    backend: str = typer.Option(
        "hf",
        help="Model backend: hf | tlens | ollama | openai | anthropic | gemini | "
        "hf-vlm | openai-vlm. Workloads containing image rows require hf-vlm "
        "or openai-vlm.",
    ),
    regime: str = typer.Option(
        "batch", help="Default prompt regime (a per-row \"regime\" key overrides it)."
    ),
    seed: int = typer.Option(42, help="Random seed"),
    max_new_tokens: int = typer.Option(64, help="Maximum new tokens to generate"),
    top_k: int = typer.Option(50, help="Top-K candidates per step"),
    config_file: Optional[Path] = typer.Option(
        None,
        help="TOML run config (tables mirror RunConfig). CLI flags you pass "
        "explicitly override the file.",
    ),
    mode: str = typer.Option(
        "fast",
        help="fast: fewer perturbation variants. audit: full perturbation set.",
    ),
    surrogate: bool = typer.Option(
        False,
        "--surrogate",
        help="Recover input-side signals on backends that cannot teacher-force "
        "by teacher-forcing a small local proxy model (see `hif profile "
        "--surrogate`). Implied by --surrogate-model.",
    ),
    surrogate_model: Optional[str] = typer.Option(
        None,
        "--surrogate-model",
        help="Open-weight HF model id used for --surrogate (default: Llama 3.2 "
        "1B, ungated mirror). Passing this implies --surrogate.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Opt-in traceability: persist each row's full profile artifact "
        "(raw per-step top-K distributions). Default off: compute-and-discard.",
    ),
    trace_dir: Optional[Path] = typer.Option(
        None, "--trace-dir", help=TRACE_DIR_HELP
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Profile only the first N workload rows."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Also mirror the stdout record stream to <output-dir>/records.jsonl. "
        "Default: records stream to stdout only.",
    ),
    units: bool = typer.Option(False, "--units", help=UNITS_HELP),
) -> None:
    """Profile every prompt in a workload file (model loaded once).

    Streams one compact JSON record per row to stdout (pipe-safe; all
    progress/logs go to stderr). Row failures emit an error record and the
    run continues.
    """
    from hif import batch as batch_mod


    # Backend validation FIRST — cheap, and an unknown backend should fail
    # fast (exit 3) before any model load.
    backend = _resolve_backend(model_name, backend)
    from hif.models.factory import KNOWN_BACKENDS
    if backend not in KNOWN_BACKENDS:
        err_console.print(
            f"[red]Unknown --backend {backend!r}. "
            f"Use one of: {', '.join(KNOWN_BACKENDS)}.[/red]"
        )
        raise typer.Exit(3)

    _check_mode(mode)

    # --surrogate-model implies --surrogate; --trace-dir implies --trace
    # (same conventions as `profile`).
    if surrogate_model is not None:
        surrogate = True
    if trace_dir is not None:
        trace = True

    # Validate the whole workload up front — before any model load.
    try:
        rows = batch_mod.load_workload(workload, limit=limit)
    except batch_mod.WorkloadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3)

    if not rows:
        err_console.print(
            f"[red]Workload {workload} has no rows to profile"
            f"{' after --limit' if limit is not None else ''} — nothing to do.[/red]"
        )
        raise typer.Exit(3)

    if batch_mod.has_image_rows(rows) and backend not in batch_mod.VLM_BACKENDS:
        err_console.print(
            "[red]This workload contains image rows — use --backend hf-vlm "
            "(local VLM) or --backend openai-vlm (hosted vision API).[/red]"
        )
        raise typer.Exit(3)

    explicit = _explicit_generation_params(ctx)
    base_config = _load_config_file(config_file) if config_file is not None else None
    config = _make_run_config(
        model_name, backend, max_new_tokens, top_k, seed, output_dir,
        base=base_config, explicit=explicit,
    )
    if base_config is None or "mode" in explicit:
        config.perturbation.n_variants = 2 if mode == "fast" else 5
    config.traceability.enabled = trace
    # A --config-file [generation] seed wins over the CLI *default* (an
    # explicit --seed still beats the file) — the seed passed to the run must
    # match config.generation.seed or the record lies about reproducibility.
    if "seed" not in explicit:
        seed = config.generation.seed

    resolved_trace_dir = trace_dir or (
        (output_dir / "traces") if output_dir else Path("traces")
    )

    records_fh = _open_records_file(output_dir)

    def _emit(record: dict) -> None:
        _emit_json_line(record)
        if records_fh is not None:
            records_fh.write(json.dumps(record) + "\n")
            records_fh.flush()

    def _log(msg: str) -> None:
        err_console.print(msg, markup=False, highlight=False)

    try:
        n_ok, n_failed = batch_mod.run_batch(
            config,
            rows,
            default_regime=regime,
            seed=seed,
            surrogate=surrogate,
            surrogate_model_id=surrogate_model,
            trace=trace,
            trace_dir=resolved_trace_dir,
            emit=_emit,
            include_units=units,
            log=_log,
        )
    finally:
        if records_fh is not None:
            records_fh.close()

    _log(f"batch complete: {n_ok} ok, {n_failed} failed of {len(rows)} row(s)")
    if rows and n_ok == 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
