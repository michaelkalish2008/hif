"""`hif render` — re-render Markdown from a stored profile artifact."""

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
    app,
    console,
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
    run_subjects as _run_subjects,
)



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


# ---------------------------------------------------------------------------
# Commands — Config (resolution without a run)
# ---------------------------------------------------------------------------

