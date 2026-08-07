"""`hif batch` — profile many prompts against one loaded model.

Every option `profile` has and `batch` also has means the same thing here and
carries the same one-sentence help, so the reasoning behind it is in
hif/cli/profile.py's module docstring rather than restated: why the expensive
stages are opt-in, why `--lite`, `--mode` and `--acquisition` are three knobs
and not one, and why the labels are labels. Two commands whose shared flags
are explained twice are two explanations that will disagree.

What is particular to `batch` is where the rows come from. A workload JSONL
file and `--sample-set` are two sources for the same thing, so they are
mutually exclusive and both flow through the same run: the built-in suite
produces ROWS rather than running a pipeline of its own, which is how it
inherits every control here instead of drifting from them. That suite is a
FIXED stimulus set — its value is that two models profiled on it were
profiled on identical strings, which is the condition for a cross-model
comparison being a comparison at all. It is not a benchmark: the prompts are
unlabeled and nothing scores anything (docs/PROMPT_SUITE.md).

`--export-workload` is the fork: it resolves rows from either source, writes
them, and exits without loading a model, so the suite can be edited, given
per-row `variants`, and run back through the same command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from hif.cli._app import (
    PANEL_FILES,
    PANEL_MODEL,
    PANEL_REPORT,
    PANEL_ROWS,
    PANEL_SCOPE,
    PANEL_SURROGATE,
    PanelledCommand,
    examples,
    REGIME_LABEL_HELP,
    TRACE_DIR_HELP,
    UNITS_HELP,
    _emit_json_line,
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
from hif.cli._run import _resolve_run_config




def _sample_set_rows(selector: str, *, limit: Optional[int] = None) -> list:
    """The built-in prompt suite as workload rows.

    The suite is a FIXED stimulus set — 8 regimes x 5 prompts — and that is
    its whole value: two models profiled on it were profiled on identical
    strings, which is the condition for a cross-model comparison being a
    comparison at all. It is not a benchmark; the prompts are unlabeled and
    nothing here scores anything (docs/PROMPT_SUITE.md).

    Producing ROWS rather than running its own pipeline is the point: the
    sample set is one source of workload rows among others, so it inherits
    every control `hif batch` has instead of needing its own command that
    drifts from it.
    """
    from hif.batch import BatchRow
    from hif.prompts.suite import REGIMES, get_regime

    if selector == "all":
        selected = list(REGIMES)
    else:
        try:
            selected = [get_regime(selector)]
        except ValueError:
            names = ", ".join(r.name for r in REGIMES)
            err_console.print(
                f"[red]--sample-set must be 'all' or a regime name — got "
                f"{selector!r}.[/red]\n[dim]Regimes: {names}[/dim]"
            )
            raise typer.Exit(3)

    rows = [
        BatchRow(
            query_id=f"{reg.name}_{i:02d}",
            text=prompt_text,
            regime=reg.name,
        )
        for reg in selected
        for i, prompt_text in enumerate(reg.prompts, 1)
    ]
    if limit is not None:
        rows = rows[: max(limit, 0)]
    return rows


def _open_records_file(output_dir: Optional[Path]):
    """Open <output-dir>/records.jsonl for the batch stream mirror (or None)."""
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir / "records.jsonl").open("w")


@app.command(cls=PanelledCommand)
@examples(
    "hif batch --sample-set all gpt2",
    "the built-in suite: 8 regimes x 5 prompts, one record per row on stdout",

    "hif batch workload.jsonl gpt2 --output-dir out",
    "your own rows; records stream to stdout and mirror to out/records.jsonl",

    "hif batch --sample-set all --export-workload suite.jsonl gpt2",
    "write the suite's rows as a file to edit and run back — no model is loaded",

    "hif batch workload.jsonl gpt2 --lite --limit 5",
    "a quick shape-check of a new workload before committing to the full run",
)
def batch(
    ctx: typer.Context,
    workload: Optional[Path] = typer.Argument(
        None,
        help="Workload JSONL file: one {\"query_id\", \"text\"[, \"regime\", "
        "\"variants\"]} row per line. Omit it when using --sample-set.",
    ),
    model_name: Optional[str] = typer.Argument(None, help="Model name (e.g. gpt2)"),
    # -- Rows to profile: where the workload comes from. --------------------
    sample_set: Optional[str] = typer.Option(
        None,
        "--sample-set",
        rich_help_panel=PANEL_ROWS,
        help="Profile the built-in prompt suite instead of a workload file: "
        "`all` (8 regimes x 5 prompts) or one regime name. A fixed stimulus "
        "set, identical for every model — not a benchmark, and nothing is "
        "scored.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", rich_help_panel=PANEL_ROWS,
        help="Profile only the first N rows.",
    ),
    export_workload: Optional[Path] = typer.Option(
        None,
        "--export-workload",
        rich_help_panel=PANEL_ROWS,
        help="Write the resolved rows to a workload JSONL and exit; no model "
        "loads. This is how you fork --sample-set: edit the rows, add per-row "
        "`variants`, run it back.",
    ),
    regime: str = typer.Option(
        "batch",
        rich_help_panel=PANEL_ROWS,
        help="Regime for rows with no \"regime\" key of their own. "
        + REGIME_LABEL_HELP,
    ),
    # -- Model and generation ------------------------------------------------
    backend: str = typer.Option(
        "hf",
        rich_help_panel=PANEL_MODEL,
        help="Model backend: hf | tlens | ollama | openai | anthropic | gemini. "
        "Run `hif models` for what each one can measure.",
    ),
    max_new_tokens: int = typer.Option(
        64, rich_help_panel=PANEL_MODEL,
        help="Maximum new tokens to generate, per row.",
    ),
    top_k: int = typer.Option(
        50, rich_help_panel=PANEL_MODEL,
        help="How many candidates to record at each step.",
    ),
    seed: int = typer.Option(
        42, rich_help_panel=PANEL_MODEL,
        help="Random seed, recorded with every record.",
    ),
    # -- Scope of the run (same three knobs as `hif profile`) ----------------
    lite: bool = typer.Option(
        False,
        "--lite",
        rich_help_panel=PANEL_SCOPE,
        help="Speed: skip every stage that costs an extra generation pass or "
        "an embedding sweep, on every row. Their measurements come back "
        "absent, not zero.",
    ),
    mode: str = typer.Option(
        "fast",
        rich_help_panel=PANEL_SCOPE,
        help="Perturbation budget per row: fast = 2 paraphrase variants, "
        "audit = 5.",
    ),
    acquisition: str = typer.Option(
        "elicited-output",
        "--acquisition",
        rich_help_panel=PANEL_SCOPE,
        help="Ceiling on what each row may bring into existence — provenance, "
        "not speed: observational | synthesized-input | elicited-output. Same "
        "meaning as `hif profile --acquisition`; `hif schema` gives each "
        "measurement's tier.",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        rich_help_panel=PANEL_SCOPE,
        help="TOML run config; its tables mirror RunConfig. Flags you pass "
        "explicitly win. Confirm with `hif config show`.",
    ),
    # -- What is reported ----------------------------------------------------
    units: bool = typer.Option(
        False, "--units", rich_help_panel=PANEL_REPORT, help=UNITS_HELP
    ),
    variant_io: bool = typer.Option(
        False,
        "--variant-io",
        rich_help_panel=PANEL_REPORT,
        help="Add each perturbation variant's input text and the continuation "
        "it elicited to every record.",
    ),
    entropy_percentile: Optional[float] = typer.Option(
        None,
        rich_help_panel=PANEL_REPORT,
        help="Also report output_nucleus_entropy_bits: the entropy of the "
        "smallest per-step prefix carrying this percent of the output "
        "distribution's mass (e.g. 95), renormalized. Needs a full-logprob "
        "backend (`hif models`).",
    ),
    # -- Files written -------------------------------------------------------
    output_dir: Optional[Path] = typer.Option(
        None,
        rich_help_panel=PANEL_FILES,
        help="Also mirror the stdout record stream to "
        "<output-dir>/records.jsonl.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        rich_help_panel=PANEL_FILES,
        help="Persist each row's full profile artifact — raw per-step top-K "
        "distributions, reconstructable content — for later recomputation.",
    ),
    trace_dir: Optional[Path] = typer.Option(
        None, "--trace-dir", rich_help_panel=PANEL_FILES, help=TRACE_DIR_HELP
    ),
    # -- Input-side recovery: the expert path, last. -------------------------
    surrogate: bool = typer.Option(
        False,
        "--surrogate",
        rich_help_panel=PANEL_SURROGATE,
        help="Recover the input-side measurements on backends that cannot "
        "teacher-force by teacher-forcing a small local proxy model instead, "
        "so those numbers describe the proxy, not your model (see `hif "
        "profile --surrogate`).",
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
    """Profile many prompts against one loaded model.

    Rows come from a workload JSONL file, or from the built-in prompt suite
    via --sample-set. Streams one compact JSON record per row to stdout
    (pipe-safe; all progress/logs go to stderr). Row failures emit an error
    record and the run continues.

        hif batch workload.jsonl gpt2
        hif batch --sample-set all gpt2
        hif batch --sample-set all --export-workload suite.jsonl   # fork it
    """
    from hif import batch as batch_mod

    # With --sample-set there is no workload path, so the single positional
    # the user typed is the MODEL. Click binds positionals left to right and
    # would call it `workload`; shift it back rather than making the model a
    # named option, which would break `hif batch <file> <model>`.
    if sample_set is not None and model_name is None and workload is not None:
        model_name, workload = str(workload), None

    if sample_set is not None and workload is not None:
        err_console.print(
            "[red]Pass either a workload file or --sample-set, not both — "
            "they are two sources for the same rows.[/red]"
        )
        raise typer.Exit(3)
    if sample_set is None and workload is None:
        err_console.print(
            "[red]Nothing to profile: pass a workload JSONL file, or "
            "--sample-set all to use the built-in prompt suite.[/red]"
        )
        raise typer.Exit(3)
    # --export-workload writes rows and exits; everything else needs a model.
    if model_name is None and export_workload is None:
        err_console.print("[red]Missing argument: model name (e.g. gpt2).[/red]")
        raise typer.Exit(3)

    # Backend validation FIRST — cheap, and an unknown backend should fail
    # fast (exit 3) before any model load.
    backend = _resolve_backend(model_name or "gpt2", backend)
    from hif.models.factory import KNOWN_BACKENDS
    if backend not in KNOWN_BACKENDS:
        err_console.print(
            f"[red]Unknown --backend {backend!r}. "
            f"Use one of: {', '.join(KNOWN_BACKENDS)}.[/red]"
        )
        raise typer.Exit(3)

    _check_mode(mode)
    _check_acquisition(acquisition)

    # --surrogate-model implies --surrogate; --trace-dir implies --trace
    # (same conventions as `profile`).
    if surrogate_model is not None:
        surrogate = True
    if trace_dir is not None:
        trace = True

    # Resolve rows up front — before any model load — whichever source.
    if sample_set is not None:
        rows = _sample_set_rows(sample_set, limit=limit)
    else:
        try:
            rows = batch_mod.load_workload(workload, limit=limit)
        except batch_mod.WorkloadError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(3)

    # --export-workload: write the rows and stop. No model, no inference.
    if export_workload is not None:
        export_workload.write_text(
            "\n".join(
                json.dumps(
                    {"query_id": r.query_id, "text": r.text}
                    | ({"regime": r.regime} if r.regime else {})
                    | ({"variants": r.variants} if r.variants else {})
                )
                for r in rows
            )
            + "\n"
        )
        console.print(
            f"[green]Wrote:[/green] {export_workload} ({len(rows)} rows)\n"
            f"[dim]Edit it, add per-row \"variants\", then: "
            f"hif batch {export_workload} <model>[/dim]"
        )
        return

    if not rows:
        err_console.print(
            f"[red]Workload {workload} has no rows to profile"
            f"{' after --limit' if limit is not None else ''} — nothing to do.[/red]"
        )
        raise typer.Exit(3)

    explicit = _explicit_generation_params(ctx)
    base_config = _load_config_file(config_file) if config_file is not None else None
    # Same single resolution path `profile` and `config show` use, so the
    # ceilings mean the same thing at every scale.
    config = _resolve_run_config(
        model_name, backend, max_new_tokens, top_k, seed, output_dir,
        base_config=base_config, explicit=explicit,
        n_perturbation_variants=(2 if mode == "fast" else 5),
        trace=trace, lite=lite, acquisition=acquisition,
        entropy_percentile=_check_entropy_percentile(entropy_percentile, backend),
    )
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
            variant_io=variant_io,
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
