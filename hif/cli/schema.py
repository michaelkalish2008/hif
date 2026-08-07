"""`hif schema` — print the measurement registry, every key with its full row."""

from __future__ import annotations

import json

import typer
from rich.table import Table

from hif.cli._app import (
    PanelledCommand,
    app,
    examples,
    console,
)

# Canonical measurement extraction lives in hif/profile/measure.py so the CLI
# tables and the SessionEngine record path report identical numbers; the
# registry rows they are keyed on live in hif/profile/registry.py and the wire
# record in hif/profile/record.py.
from hif.profile.record import RECORD_SCHEMA_VERSION as SIGNAL_RECORD_VERSION
from hif.profile.registry import (
    ACQUISITION_LEGEND,
    MEASUREMENT_REGISTRY,
    SIGNAL_SET_VERSION,
    SUBJECT_LEGEND,
)



@app.command(cls=PanelledCommand)
@examples(
    "hif schema",
    "the measurement registry as JSON: every key, unit, subject and definition",

    "hif schema | jq -r '.measurements | keys[]'",
    "just the measurement names — the valid values for --metric",
)
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
    This is the contract for `hif profile --json` and `hif batch` records, and the machine-readable mirror of
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
                "hif batch": "JSONL, one record per workload row (or per prompt of the built-in suite under --sample-set)",
                "hif compare --json": "a single JSON document",
                "hif models --json": "a single JSON document: the backend "
                        "catalogue, each backend's model options, and the "
                        "measurements it can and cannot produce",
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
            # What taking each measurement had to bring into existence.
            # `subject` says whose behaviour the number is about; this says
            # whether getting it required authoring prompt text or eliciting
            # model output the caller never asked for. `--acquisition` caps a
            # run at one of these tiers.
            "acquisitions": dict(ACQUISITION_LEGEND),
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
                    "acquisition": m.acquisition,
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
    table.add_column("Acquisition")
    table.add_column("Definition")
    for m in MEASUREMENT_REGISTRY:
        subject = m.subject
        if m.subject_under_surrogate is not None:
            subject = f"{m.subject} → {m.subject_under_surrogate} (surrogate)"
        table.add_row(
            m.key, m.name, m.unit, m.resolution, subject, m.acquisition, m.definition
        )
    console.print(table)
    console.print("\n[bold]Subject[/bold] — whose behaviour the number describes:")
    for value, gloss in SUBJECT_LEGEND.items():
        console.print(f"  [bold]{value}[/bold] — {gloss}")
    console.print(
        "\n[bold]Acquisition[/bold] — what taking the measurement brings into "
        "existence (cap a run with --acquisition):"
    )
    for value, gloss in ACQUISITION_LEGEND.items():
        console.print(f"  [bold]{value}[/bold] — {gloss}")


# ---------------------------------------------------------------------------
# Commands — Batch (workload runner)
# ---------------------------------------------------------------------------


