"""`hif profile` — the full pipeline on one (model, prompt) pair.

The option help below says what each flag gets you, in one sentence, and
stops. The reasoning behind the defaults lives here instead, where there is
room for it: a help column forty characters wide, scanned by someone looking
for one answer, is the worst place in the tool to defend a design decision.

WHY THE EXPENSIVE THINGS ARE OFF BY DEFAULT

A run that was not asked for files leaves none, so `--output-dir` is the opt-in
that puts anything on disk: the technical report, the `--charts` plots, and the
profile JSON — which goes to the trace dir (`<output-dir>/traces`, or
`--trace-dir`), the artifact's one home whether or not `--trace` was passed.
`--trace` is a second axis, not a stricter version of the first: it decides
what is IN that JSON, adding the raw perturbation-variant and trajectory-branch
traces, which is what makes field descriptors recomputable without re-running
the models, at a cost in size that scales with the variant count. (The BASELINE
per-step top-K is on `output_side.steps` and therefore in the JSON either way;
`--trace` has never been the line between distributions on disk and not.)

`--variant-io` is the same decision one level down — it adds model-generated
text to every record, so the record becomes the review surface for elicited
output; inputs stay immutable, outputs live in records. `--diagnostics` runs
two stages (attention capture, the semantic field) that publish no
measurement in hif-v4 and cost real compute; their blocks ship in the trace
as evidence. `--charts` needs `--output-dir` because plots are files.

`--entropy-percentile` is off by default so `output_entropy_bits` keeps its
full-vocabulary basis: the nucleus entropy is an ADDITIONAL row
(`output_nucleus_entropy_bits`), never a redefinition of the existing one. It
is also the one place `--top-k` stops being a capture detail — the nucleus
has to fall inside the captured slice, so a run that asks for it and gets an
absence is usually a run that needed a larger `--top-k`, which is what the
CLI says when it happens.

`--units` is off because the block is constant per signal_set_version and
identical on every record — `hif schema` prints it without running a model.

THE THREE "HOW MUCH WORK" KNOBS, AND WHY THEY ARE NOT ONE KNOB

`--lite` is speed. It drops the stages that cost an extra generation pass or
an embedding sweep — paraphrase variants, trajectory branches, and the
per-step candidate geometry the exposure and semantic-field stages read. The
entropy-side measurements are unchanged; the ones those stages feed come back
ABSENT rather than zero, and it is applied after `--config-file` so a run
asking for less never silently does more.

`--mode` is the perturbation budget alone: two paraphrase variants at `fast`,
five at `audit`. Nothing else in the pipeline changes.

`--acquisition` is not a speed knob and does not belong on the same axis. It
is a ceiling on what the run may bring into existence — `observational` sends
nothing beyond the call the caller asked for and leaves no model output that
did not already exist; `synthesized-input` lets the tool author prompt text
and teacher-force over it, with the model still not generating;
`elicited-output` lets the model generate variant continuations and
trajectory branches, which is the tier that costs tokens and produces
unreviewed output. It applies before `--lite` because the two compose: this
one says what the run is permitted to produce, `--lite` says how much work to
do within that permission. Measurements above the ceiling are absent, not
zero, and each measurement's tier is in `hif schema`.

LABELS THAT LOOK LIKE CONTROLS

`--regime`, `--application` and `--analysis-window` are recorded with the run
and read by nothing. `--application` additionally supplies the default
`--analysis-window`, and that is the whole of its effect on the run.
`--analysis-window` in particular is validated (an integer or `adaptive`),
written into the record's extras, and consumed by no stage — a cap nothing
enforces, kept because the record is where an intended window is declared.

`--truncate` is the opposite case, and reads like a recording detail when it
is not one: it cuts the prompt before anything runs — by whitespace-split
words rather than tokenizer tokens, the CLI having no tokenizer at that
point — so every number afterwards describes the truncated prompt and not
the one that was typed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from hif.cli._app import (
    CHARTS_HELP,
    PANEL_FILES,
    PANEL_LABELS,
    PANEL_MODEL,
    PANEL_REPORT,
    PANEL_SCOPE,
    PANEL_SURROGATE,
    PanelledCommand,
    examples,
    REGIME_LABEL_HELP,
    TRACE_DIR_HELP,
    UNITS_HELP,
    app,
    console,
    err_console,
)
from hif.cli._config import (
    _check_acquisition,
    _check_entropy_percentile,
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



@app.command(cls=PanelledCommand)
@examples(
    'hif profile Qwen/Qwen3-0.6B-Base "Why is the sky blue?"',
    "measure one prompt; prints to the terminal and writes nothing",

    'hif profile Qwen/Qwen3-0.6B-Base "Why is the sky blue?" --output-dir out --charts',
    "same run, plus Markdown reports and one Plotly chart per signal under out/",

    'hif profile Qwen/Qwen3-0.6B-Base "Why is the sky blue?" --metric output_entropy_bits',
    "print one number and exit — the form to use inside a script",

    'hif profile Qwen/Qwen3-0.6B-Base "Why is the sky blue?" --lite --json',
    "the fast subset, as a JSON record; skipped stages come back absent, not zero",

    'hif profile Qwen/Qwen3-0.6B-Base "Why is the sky blue?" --entropy-percentile 95 --top-k 2000 --lite',
    "add output_nucleus_entropy_bits; the wide --top-k is what it needs, not the --lite",
)
def profile(
    ctx: typer.Context,
    model_name: str = typer.Argument(..., help="Model name (e.g. Qwen/Qwen3-0.6B-Base)"),
    prompt: str = typer.Argument(..., help="Prompt text"),
    # -- Model and generation: what you are running, and how it generates. ---
    backend: str = typer.Option(
        "hf",
        rich_help_panel=PANEL_MODEL,
        help="Model backend: hf | tlens | ollama | openai | anthropic | gemini. "
        "Run `hif models` for what each one can measure.",
    ),
    max_new_tokens: int = typer.Option(
        64, rich_help_panel=PANEL_MODEL,
        help="Maximum new tokens to generate.",
    ),
    top_k: int = typer.Option(
        50, rich_help_panel=PANEL_MODEL,
        help="How many candidates to record at each step.",
    ),
    seed: int = typer.Option(
        42, rich_help_panel=PANEL_MODEL,
        help="Random seed, recorded with the run.",
    ),
    truncate: Optional[int] = typer.Option(
        None,
        "--truncate",
        rich_help_panel=PANEL_MODEL,
        # Says "whitespace-split" because that is what the implementation
        # below does: it splits on whitespace and keeps N words, with no
        # tokenizer involved. On a prompt of ordinary prose the two are close;
        # the flag still must not promise a unit it does not count in.
        help="Cut the prompt to its first N whitespace-split tokens before "
        "the run. Results then reflect truncated context only.",
    ),
    # -- Scope of the run: how much work it does, and what it may create. ----
    #
    # The three knobs a reader confuses are adjacent and in this order on
    # purpose, and each one's first clause names the job only it has: --lite
    # is speed, --mode is the perturbation budget, --acquisition is a ceiling
    # on provenance and is not a speed control at all. Read together they
    # answer "I want it faster — which do I pass?" in the first line of each.
    lite: bool = typer.Option(
        False,
        "--lite",
        rich_help_panel=PANEL_SCOPE,
        help="Speed: skip every stage that costs an extra generation pass or "
        "an embedding sweep. Their measurements come back absent, not zero.",
    ),
    mode: str = typer.Option(
        "fast",
        rich_help_panel=PANEL_SCOPE,
        help="Perturbation budget: fast = 2 paraphrase variants, audit = 5. "
        "The prompt itself is always passed in full.",
    ),
    acquisition: str = typer.Option(
        "elicited-output",
        "--acquisition",
        rich_help_panel=PANEL_SCOPE,
        help="Ceiling on what the run may bring into existence — provenance, "
        "not speed. observational: only the one call you asked for. "
        "synthesized-input: + authored prompts, teacher-forced. "
        "elicited-output: + model-generated variants and branches. Above the "
        "ceiling, measurements are absent; `hif schema` gives each one's tier.",
    ),
    diagnostics: bool = typer.Option(
        False,
        "--diagnostics",
        rich_help_panel=PANEL_SCOPE,
        help="Also run attention capture and the semantic field. Neither "
        "produces a measurement; their blocks ship in the --trace artifact.",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        rich_help_panel=PANEL_SCOPE,
        # `\[` is Rich's escape for a literal bracket: help text is Rich
        # markup, and a bare [generation] is read as a style tag and swallowed.
        # tools/gen_flags_doc.py drops the backslash for docs/FLAGS.md.
        help="TOML run config; its tables mirror RunConfig (\\[generation], "
        "\\[perturbation], \\[trajectory], \\[attention], \\[semantic_field], "
        "...). Flags you pass explicitly win. Confirm with `hif config show`.",
    ),
    # -- What is reported: what comes back on stdout, and in what shape. -----
    output_json: bool = typer.Option(
        False, "--json",
        rich_help_panel=PANEL_REPORT,
        help="Print the record as JSON: derived measurements only, never the "
        "raw per-step distributions.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        rich_help_panel=PANEL_REPORT,
        help="Print ONE measurement in its natural unit, then exit. Names and "
        "units: `hif schema`.",
    ),
    units: bool = typer.Option(
        False, "--units", rich_help_panel=PANEL_REPORT, help=UNITS_HELP
    ),
    variant_io: bool = typer.Option(
        False,
        "--variant-io",
        rich_help_panel=PANEL_REPORT,
        help="Add each perturbation variant's input text and the continuation "
        "it elicited to the --json record (null where none was elicited).",
    ),
    entropy_percentile: Optional[float] = typer.Option(
        None,
        rich_help_panel=PANEL_REPORT,
        help="Also report output_nucleus_entropy_bits: the entropy of the "
        "smallest per-step prefix carrying this percent of the output "
        "distribution's mass (e.g. 95), renormalized. Needs a full-logprob "
        "backend (`hif models`).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        rich_help_panel=PANEL_REPORT,
        help="Also show model input/output text, perturbation variants, full "
        "numeric stats, and internal logging.",
    ),
    # -- Files written: nothing reaches the disk unless one of these is set. -
    output_dir: Optional[Path] = typer.Option(
        None,
        rich_help_panel=PANEL_FILES,
        help="Write the run's files here: the technical Markdown report, the "
        "--charts plots, and the profile JSON (in <output-dir>/traces).",
    ),
    charts: bool = typer.Option(
        False, "--charts", rich_help_panel=PANEL_FILES, help=CHARTS_HELP
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        rich_help_panel=PANEL_FILES,
        help="Add the raw perturbation-variant and trajectory-branch traces "
        "to the profile artifact, so field descriptors can be recomputed "
        "later without re-running the model.",
    ),
    trace_dir: Optional[Path] = typer.Option(
        None, "--trace-dir", rich_help_panel=PANEL_FILES, help=TRACE_DIR_HELP
    ),
    # -- Labels: recorded with the run, read by nothing. ---------------------
    regime: str = typer.Option(
        "ordinary_conversation",
        rich_help_panel=PANEL_LABELS,
        help=REGIME_LABEL_HELP,
    ),
    application: Optional[str] = typer.Option(
        None,
        rich_help_panel=PANEL_LABELS,
        help="A free-form label for what this run is for, recorded with the "
        "run — any string, changing no measurement.",
    ),
    # -- Input-side recovery: the expert path, last. -------------------------
    surrogate: bool = typer.Option(
        False,
        "--surrogate",
        rich_help_panel=PANEL_SURROGATE,
        help="Recover the input-side measurements on backends that cannot "
        "teacher-force — score text they did not generate (ollama, openai, "
        "anthropic, gemini; see `hif models`). A small local proxy model is "
        "teacher-forced instead, so those numbers describe the proxy, not "
        "your model. Ignored on hf/tlens.",
    ),
    surrogate_model: Optional[str] = typer.Option(
        None,
        "--surrogate-model",
        rich_help_panel=PANEL_SURROGATE,
        help="Open-weight HF model id to use as that proxy (default: Llama "
        "3.2 1B, ungated mirror). Passing it implies --surrogate; "
        "`hif models --surrogates` lists candidates.",
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
    # Validated before the model loads: a bad percentile or an incapable
    # backend should cost a message, not a pipeline.
    entropy_pct = _check_entropy_percentile(entropy_percentile, backend)

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
    # (This is what would have caught `--metric input_entropy_std_bits
    #  --backend ollama`.)
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
                f"[yellow]Warning: Input truncated to {truncate} words "
                f"({len(words)} before) — results reflect truncated context "
                "only. Whitespace-split, not tokenizer tokens: the model will "
                "see a different count.[/yellow]"
            )

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
                entropy_percentile=entropy_pct,
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

    # The profile's own resolved config, so the hash printed here is the one
    # the record carries and the one the artifacts were named with.
    h = _profile_hash(model_name, prompt, seed, getattr(p, "config", None))

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



