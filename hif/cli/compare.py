"""`hif compare` — the per-measurement difference between two profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from hif.cli._app import (
    PanelledCommand,
    app,
    examples,
    console,
    err_console,
)
from hif.cli._compat import (
    _artifact_signal_set_version,
    _signal_set_family,
    _signal_set_mismatch_exit,
)

# Canonical measurement extraction lives in hif/profile/measure.py so the CLI
# tables and the SessionEngine record path report identical numbers; the
# registry rows they are keyed on live in hif/profile/registry.py and the wire
# record in hif/profile/record.py.
from hif.profile.measure import measurements as _measurements
from hif.profile.registry import MEASUREMENT_KEYS, MEASUREMENT_UNITS



@app.command(cls=PanelledCommand)
@examples(
    'hif profile Qwen/Qwen3-0.6B-Base "..." --trace --trace-dir tr',
    "first make the artifacts: compare reads --trace profiles, NOT --json records",

    "hif compare tr/profile_<a>.json tr/profile_<b>.json",
    "per-measurement difference between the two, as a table",

    "hif compare tr/profile_<a>.json tr/profile_<b>.json --json",
    "the same comparison as a record, for a script",
)
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


