"""`hif batch` — profile many prompts against one loaded model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from hif.cli._app import (
    TRACE_DIR_HELP,
    UNITS_HELP,
    _emit_json_line,
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
from hif.cli._run import _resolve_run_config

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
    run_subjects as _run_subjects,
)



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


@app.command()
def batch(
    ctx: typer.Context,
    workload: Optional[Path] = typer.Argument(
        None,
        help="Workload JSONL file: one {\"query_id\", \"text\"[, \"regime\", "
        "\"variants\"]} row per line. Omit it when using --sample-set.",
    ),
    model_name: Optional[str] = typer.Argument(None, help="Model name (e.g. gpt2)"),
    backend: str = typer.Option(
        "hf",
        help="Model backend: hf | tlens | ollama | openai | anthropic | gemini",
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
    acquisition: str = typer.Option(
        "elicited-output",
        "--acquisition",
        help="Ceiling on what this run may bring into existence, applied to "
        "every row. observational | synthesized-input | elicited-output. "
        "Same meaning as `hif profile --acquisition`; run `hif schema` for "
        "each measurement's tier.",
    ),
    lite: bool = typer.Option(
        False,
        "--lite",
        help="Skip perturbation variants, trajectory branches, and per-step "
        "candidate geometry on every row (see `hif profile --lite`).",
    ),
    variant_io: bool = typer.Option(
        False,
        "--variant-io",
        help="Include a `variant_io` block in each record: every perturbation "
        "variant's input text and the continuation it elicited.",
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
    sample_set: Optional[str] = typer.Option(
        None,
        "--sample-set",
        help="Use the built-in prompt suite instead of a workload file: "
        "`all` (8 regimes x 5 prompts) or a single regime name. A FIXED "
        "stimulus set — identical prompts for every model, which is the "
        "condition for a cross-model comparison being a comparison. It is "
        "not a benchmark: the prompts are unlabeled and nothing is scored. "
        "Pair with --export-workload to fork it.",
    ),
    export_workload: Optional[Path] = typer.Option(
        None,
        "--export-workload",
        help="Write the resolved rows as a workload JSONL and exit — no model "
        "is loaded. With --sample-set, this is how you fork the built-in "
        "suite: edit the rows, add per-row `variants`, then run it back.",
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
