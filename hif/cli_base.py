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
    help="""Horizonal Interpretability — using the horizon of the possibility space to describe model behaviour.

Every measurement is reported in its natural unit (bits, cosine distance, Pearson r, a fraction of steps); nothing is normalised, inverted, or thresholded. A measurement missing from a record was NOT TAKEN — absence is never zero.

SCALE — same configuration, same ceilings at both:
  hif profile <model> <prompt>       one case
  hif batch <workload.jsonl> <model> many cases, model loaded once
  hif batch --sample-set all <model> the built-in fixed stimulus set

CONFIGURE — three measurements are comparisons against runs the tool constructs, so the configuration is part of the measurement:
  hif config init                a run.toml with every key at its default
  hif config show --diff         what will run, before running it
  --config-file run.toml         apply it (a mistyped key exits 3)

CONTROL what a run may bring into existence (--acquisition):
  observational       the prompt as given, and the one continuation
  synthesized-input   + authored paraphrases, teacher-forced only
  elicited-output     + generated variants and branches (default)
  --lite              skip the expensive stages (speed, not policy)

SEE — nothing is written to disk unless you ask:
  --output-dir DIR    Markdown reports
  --charts            one interactive Plotly HTML per signal, plus an
                      index.html dashboard (needs --output-dir)
  --trace             the full profile artifact, for recomputation

INSPECT: `hif schema` (every measurement: unit, definition, subject, acquisition), `hif models` (what each backend can produce), `hif doctor` (what is installed and reachable, including chart support).""",
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


# Which dotenv file supplied each variable this process loaded. `doctor`
# reports it, because "set" on its own is exactly what makes a stale, shadowed,
# or never-read dotenv so hard to diagnose — the state you have to distinguish
# is not set/unset but *which file won*.
ENV_SOURCES: dict[str, Path] = {}

# The credentials `doctor` reports and the backends check. One list, so a new
# provider cannot be added to one and forgotten in the other.
CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "HF_TOKEN",
)

USER_ENV_FILE = Path.home() / ".config" / "hif" / ".env"


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
    Because a value already present is never replaced, calling this over a list
    of files in order gives first-file-wins precedence for free.
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
        ENV_SOURCES[key] = path
        n += 1
    return n


def discover_env_files() -> list[Path]:
    """Dotenv files to read when none was named, nearest first.

    The nearest `.env` at or above the working directory, then the user-level
    `~/.config/hif/.env`. The first covers the ordinary case — a project
    directory with its own keys — and the second covers the installed user who
    has no project directory at all and wants one file to serve every run.

    Only the nearest project `.env` is read, not every one up the tree: two
    dotenvs silently merging is worse than the one the user is standing in.
    """
    found: list[Path] = []
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            found.append(candidate)
            break
    if USER_ENV_FILE.is_file() and USER_ENV_FILE not in found:
        found.append(USER_ENV_FILE)
    return found


def env_origin(key: str) -> str:
    """Where `key` came from, for display: a file path, or the inherited
    environment. Empty string when the variable is not set at all."""
    if not os.environ.get(key):
        return ""
    path = ENV_SOURCES.get(key)
    return _display_path(path) if path else "environment"


def _display_path(path: Path) -> str:
    """Shortest unambiguous spelling of `path` — relative to the working
    directory when it is below it, else `~`-relative, else absolute."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        pass
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


@app.callback()
def _main(
    env_file: Optional[Path] = typer.Option(
        None,
        "--env-file",
        envvar="HIF_ENV_FILE",
        help="Read credentials from this dotenv file, ahead of any discovered "
        "one. Applies to every command. Values already set in the real "
        "environment are left alone.",
    ),
) -> None:
    """hif — Horizonal Interpretability CLI."""
    from hif.utils.logging import configure_logging

    # Default: results only. Commands that accept --verbose re-call
    # configure_logging(verbose=True) to restore full internal chatter.
    configure_logging(verbose=False)

    # Credentials are resolved here and nowhere else, so every command sees the
    # same environment. `doctor` predicting a run that a later, separate load
    # would have changed is worse than no preflight at all.
    #
    # Precedence, first to set a name wins: the real environment, then
    # --env-file / HIF_ENV_FILE, then discovery. A value that is already
    # exported is a deliberate override and survives all of this.
    if env_file is not None:
        if not env_file.is_file():
            console.print(f"[red]--env-file: no such file: {env_file}[/red]")
            raise typer.Exit(2)
        n = load_env_file(env_file)
        # Names only. The whole point of the file is that the values do not
        # get printed.
        console.print(f"[dim]Loaded {n} variable(s) from {env_file}[/dim]")
    for discovered in discover_env_files():
        # Silent: this runs on every command, and a line of chatter per
        # invocation is how a tool teaches people to stop reading stderr.
        # `doctor` is where the resolved picture gets reported.
        load_env_file(discovered)


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
