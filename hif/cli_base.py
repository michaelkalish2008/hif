"""The CLI's shared foundation: the typer app, the two consoles, and the
option help strings more than one command uses.

Every other `hif/cli_*.py` module imports from here and nothing here imports
from them, so the command modules can be split without an import cycle.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

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


def load_env_file(path: Path) -> int:
    """Read `KEY=value` lines into os.environ. Returns how many were set.

    Sourcing a dotenv in a shell does NOT do this. `source .env` on bare
    `KEY=value` lines creates SHELL variables, and a child process inherits
    only the ENVIRONMENT — so the keys are visibly present to the shell and
    invisible to hif, which reads os.environ. The usual symptom is `hif doctor`
    reporting every credential unset immediately after apparently exporting
    them. (`set -a` before sourcing is the shell-side fix; this is the one that
    does not require knowing that.)

    Values already in the environment win: an explicit `KEY=… hif …` or a real
    export is a deliberate override and must not be silently replaced by a file.
    """
    n = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        # Strip one matched pair of surrounding quotes, the way a shell would.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value
        n += 1
    return n


@app.callback()
def _main(
    env_file: Optional[Path] = typer.Option(
        None,
        "--env-file",
        envvar="HIF_ENV_FILE",
        help="Read credentials from a dotenv file before running. Applies to "
        "every command. Values already set in the environment are left alone. "
        "Nothing is auto-discovered — hosted backends bill per token, so the "
        "file that pays is named on the command line or in HIF_ENV_FILE.",
    ),
) -> None:
    """hif — Horizonal Interpretability CLI."""
    from hif.utils.logging import configure_logging

    # Default: results only. Commands that accept --verbose re-call
    # configure_logging(verbose=True) to restore full internal chatter.
    configure_logging(verbose=False)

    if env_file is not None:
        if not env_file.is_file():
            console.print(f"[red]--env-file: no such file: {env_file}[/red]")
            raise typer.Exit(2)
        n = load_env_file(env_file)
        # Names only. The whole point of the file is that the values do not
        # get printed.
        console.print(f"[dim]Loaded {n} variable(s) from {env_file}[/dim]")


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
