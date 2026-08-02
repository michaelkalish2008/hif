"""The CLI's shared foundation: the typer app, the two consoles, and the
option help strings more than one command uses.

Every other `hif/cli_*.py` module imports from here and nothing here imports
from them, so the command modules can be split without an import cycle.
"""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console

app = typer.Typer(
    name="hif",
    help="Horizonal Interpretability — using the horizon of the possibility space to "
    "describe model behaviour. Every measurement is reported in its natural unit "
    "(bits, cosine distance, Pearson r, a fraction of steps); nothing is normalised, "
    "inverted, or thresholded. Run `hif schema` for the full measurement set.",
)
# stdout is reserved for data. Every human-facing line — progress, warnings,
# tables, errors — goes to stderr so `hif <cmd> ... | jq .` always parses.
console = Console(stderr=True)
err_console = console


def _emit_json_line(record: dict) -> None:
    """Write one JSONL record to stdout and flush.

    stdout carries JSON and nothing else. Every data-producing command uses
    this (or a single json.dumps for the one-document commands), so
    `hif <cmd> ... 2>/dev/null | jq .` always parses.
    """
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


@app.callback()
def _main() -> None:
    """hif — Horizonal Interpretability CLI."""
    from hif.utils.logging import configure_logging

    # Default: results only. Commands that accept --verbose re-call
    # configure_logging(verbose=True) to restore full internal chatter.
    configure_logging(verbose=False)


# ---------------------------------------------------------------------------
# Option help shared by more than one command
# ---------------------------------------------------------------------------
#
# These strings were identical copies on two or three commands. A help string
# that describes one behaviour has to say the same thing everywhere, and three
# copies is three chances for it not to.

UNITS_HELP = (
    "Include a per-measurement units block in each record. Constant per "
    "signal_set_version and identical on every record, so off by default; "
    "`hif schema` prints the same information without running a model."
)

TRACE_DIR_HELP = (
    "Where --trace artifacts are written (default: <output-dir>/traces, "
    "or ./traces when no --output-dir). Passing this implies --trace."
)

CHARTS_HELP = (
    "Generate plots + the combined dashboard locally (off by default)."
)
