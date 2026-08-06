"""`hif models` — backend discovery: what each can measure, and what it cannot."""

from __future__ import annotations

import json
from typing import Optional

import typer

from hif.cli._app import (
    app,
    console,
    err_console,
)
from hif.cli._load import (
    _check_surrogate_candidates,
    _live_models_for_backend,
)
from hif.cli._output import (
    _print_subject_degradation,
)

from hif.profile.registry import SIGNAL_SET_VERSION



@app.command()
def models(
    backend: Optional[str] = typer.Option(
        None, help="Show only this backend (hf, tlens, ollama, openai, anthropic, gemini)."
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
    output_json: bool = typer.Option(
        False, "--json",
        help="Emit the catalogue as a single JSON document on stdout instead of the human table, "
             "so the model list can be piped, scripted, or fed to a picker. Composes with --backend, "
             "--list and --surrogates.",
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
    Pass --json for the same catalogue on stdout, machine-readable.
    """
    from rich.markup import escape
    from hif.engine import DEFAULT_SURROGATE_MODEL_ID
    from hif.models.capabilities import BACKENDS, signals_available

    if surrogates and output_json:
        print(json.dumps({
            "signal_set_version": SIGNAL_SET_VERSION,
            "surrogate_candidates": [
                {"model": model_id, "status": status,
                 "default": model_id == DEFAULT_SURROGATE_MODEL_ID}
                for model_id, status in _check_surrogate_candidates()
            ],
        }, indent=2))
        return

    if surrogates:
        console.print(
            "\n[bold]--surrogate-model candidates[/bold]  [dim](small open-weight models for "
            "--surrogate on backends that can't teacher-force)[/dim]\n"
        )
        for model_id, status in _check_surrogate_candidates():
            marker = "[green]✓ ok[/green]" if status == "ok" else f"[yellow]{status}[/yellow]"
            default_tag = "  [dim](default)[/dim]" if model_id == DEFAULT_SURROGATE_MODEL_ID else ""
            console.print(f"  {model_id:<28} {marker}{default_tag}")
        console.print()
        return

    infos = [BACKENDS[backend]] if backend and backend in BACKENDS else list(BACKENDS.values())
    if backend and backend not in BACKENDS:
        err_console.print(f"[red]Unknown backend {backend!r}. Known: {', '.join(BACKENDS)}[/red]")
        raise typer.Exit(1)

    if output_json:
        # Built from BACKENDS and signals_available, the same two sources the
        # text rendering below reads, so the machine-readable catalogue cannot
        # come to disagree with the printed one.
        document = {"signal_set_version": SIGNAL_SET_VERSION, "backends": []}
        for info in infos:
            models_note = None
            if list_live:
                live, note = _live_models_for_backend(info.name)
                # A backend that could not be reached reports its examples and
                # says so, rather than an empty list a caller would read as
                # "this backend has no models".
                catalogue = live if live is not None else info.example_models
                source = "live" if live is not None else "examples"
                models_note = note
            else:
                catalogue, source = info.example_models, "examples"
            avail = signals_available(info.name)
            document["backends"].append({
                "name": info.name,
                "kind": info.kind,
                "deps": info.deps,
                "setup": info.setup,
                "teacher_forcing": info.teacher_forcing,
                "logprobs": info.logprobs,
                "models": catalogue,
                "models_source": source,
                "models_note": models_note,
                "notes": info.notes,
                "signals": {
                    "available": [m for m, v in avail.items() if v],
                    "unavailable": [m for m, v in avail.items() if not v],
                },
            })
        print(json.dumps(document, indent=2))
        return

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
        _print_subject_degradation(info)
    console.print()


