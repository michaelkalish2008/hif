"""`hif render` — re-render Markdown from a stored profile artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from hif.cli._app import (
    app,
    console,
)




@app.command()
def render(
    profile_json: Path = typer.Argument(..., help="Path to profile JSON"),
    output: Optional[Path] = typer.Option(None, help="Output path (default: alongside JSON)"),
) -> None:
    """Load an existing profile from JSON and re-render Markdown."""
    from hif.profile.render_markdown import render_technical
    from hif.profile.schema import BehavioralRangeProfile

    if not profile_json.exists():
        console.print(f"[red]File not found: {profile_json}[/red]")
        raise typer.Exit(1)

    p = BehavioralRangeProfile.model_validate_json(profile_json.read_text())

    if output is None:
        output = profile_json.with_suffix("").with_name(
            profile_json.stem + "_technical.md"
        )

    render_technical(p, output)

    console.print(f"[green]Rendered to:[/green] {output}")


# ---------------------------------------------------------------------------
# Commands — Config (resolution without a run)
# ---------------------------------------------------------------------------

