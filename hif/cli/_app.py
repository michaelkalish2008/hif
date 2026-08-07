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
from typer.core import TyperCommand

app = typer.Typer(
    name="hif",
    help="""Horizonal Interpretability — using the horizon of the possibility space to describe model behaviour.

Every measurement is reported in its natural unit (bits, cosine distance, Pearson r, a fraction of steps); nothing is normalised, inverted, or thresholded. A measurement missing from a record was NOT TAKEN — absence is never zero.

SCALE — same configuration, same ceilings at both:
  hif profile <model> <prompt>       one case
  hif batch <workload.jsonl> <model> many cases, model loaded once
  hif batch --sample-set all <model> the built-in fixed stimulus set

CONFIGURE — several measurements are comparisons against runs the tool constructs, so the configuration is part of the measurement:
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


def _is_help_invocation() -> bool:
    """Whether this process was invoked to print help, not to run anything.

    The callback fires before the subcommand parses its own --help, so without
    this check the dotenv line is the first line of every help screen. Click
    has not seen the subcommand's arguments yet at callback time, so the
    command line itself is consulted — everything before a literal `--`, which
    is where an option spelled `--help` stops being one.
    """
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[: argv.index("--")]
    return any(arg in ("--help", "-h") for arg in argv)


@app.callback()
def _main(
    ctx: typer.Context,
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

    # A help invocation runs no model and needs no credentials, so nothing is
    # loaded at all — not merely printed quietly. Loading while help renders
    # would put a stale HIF_ENV_FILE one `--help` away from silently entering
    # the environment of a shell that only asked what a flag means.
    if _is_help_invocation():
        return

    # Credentials are resolved here and nowhere else, so every command sees the
    # same environment. `doctor` predicting a run that a later, separate load
    # would have changed is worse than no preflight at all.
    #
    # Precedence, first to set a name wins: the real environment, then
    # --env-file / HIF_ENV_FILE, then discovery. A value that is already
    # exported is a deliberate override and survives all of this.
    if env_file is not None:
        # Name the trigger. An --env-file typed on this command line and an
        # HIF_ENV_FILE inherited from the shell are corrected in different
        # places, and the line exists for the person wondering why it fired.
        #
        # Compared by member NAME, not enum identity: typer vendors click
        # (typer._click), so the ParameterSource this context returns is a
        # different enum class from the standalone `click` package's — equal
        # names, disjoint members, and an `==` against the importable one is
        # always False.
        source = ctx.get_parameter_source("env_file")
        trigger = (
            "HIF_ENV_FILE"
            if source is not None and source.name == "ENVIRONMENT"
            else "--env-file"
        )
        if not env_file.is_file():
            console.print(f"[red]{trigger}: no such file: {env_file}[/red]")
            raise typer.Exit(2)
        n = load_env_file(env_file)
        # Names only. The whole point of the file is that the values do not
        # get printed.
        console.print(
            f"[dim]Loaded {n} variable(s) from {env_file} ({trigger})[/dim]"
        )
    for discovered in discover_env_files():
        # Silent: this runs on every command, and a line of chatter per
        # invocation is how a tool teaches people to stop reading stderr.
        # `doctor` is where the resolved picture gets reported.
        load_env_file(discovered)


# ---------------------------------------------------------------------------
# Help panels
# ---------------------------------------------------------------------------
#
# `profile` takes twenty-five options and `batch` twenty. In one flat list,
# ordered by whatever order the parameters happened to be declared in, the
# reader has to hold all twenty-five in their head to find the two that answer
# their question — and an expert knob like --surrogate reads as no more
# specialised than --json, because nothing on the page says otherwise.
#
# Rich groups options into panels, one per `rich_help_panel` value, in the
# order the panels are first mentioned. So these constants are both the titles
# and the running order, and the DECLARATION ORDER of the parameters in
# profile.py and batch.py is now a reading order: what you are running, how
# much it does, what comes back, what lands on disk, what is merely labelled,
# and last the expert recovery path most runs never touch.
#
# The names are shared rather than typed per command, because two commands
# that group the same flags under titles differing by a word teach the reader
# that the grouping is decorative.
PANEL_MODEL = "Model and generation"
PANEL_SCOPE = "Scope of the run"
PANEL_REPORT = "What is reported"
PANEL_FILES = "Files written (nothing by default)"
PANEL_LABELS = "Labels recorded with the run"
PANEL_SURROGATE = "Input-side recovery (expert)"
PANEL_ROWS = "Rows to profile"
PANEL_HELP = "Help"


def examples(*lines: str):
    """Attach worked examples to a command, rendered as a final panel.

    Not `epilog=`. Typer's own epilog rendering is
    (typer/rich_utils.py, "Epilogue if we have it"):

        lines = obj.epilog.split("\\n\\n")
        epilogue = "\\n".join([x.replace("\\n", " ").strip() for x in lines])

    — single newlines are destroyed and blank lines collapse to one. A command
    line and the sentence explaining it therefore cannot occupy two lines, and
    an example you cannot copy off the screen intact is not an example. This
    keeps the text exactly as written and renders it in a panel like the
    option groups above it, so the page has one visual grammar.

    Pass alternating command / description strings.
    """
    def decorate(fn):
        fn.__hif_examples__ = tuple(lines)
        return fn
    return decorate


class PanelledCommand(TyperCommand):
    """A command whose `--help` line does not head its own help page.

    Rich prints the DEFAULT panel first — the one titled "Options", which is
    where every parameter with no `rich_help_panel` lands. Put every real
    option in a panel and click's own `--help` is the only thing left in it,
    so the page opens with a box containing one line about itself, and the
    reader meets the tool's least interesting flag first.

    The help option is click's, built on demand rather than declared, so the
    panel is stamped on it where it is handed over. Any title other than
    "Options" is enough to move it: non-default panels render in the order
    their options appear, and click appends `--help` last.
    """

    def get_help_option(self, ctx):
        option = super().get_help_option(ctx)
        if option is not None and getattr(option, "rich_help_panel", None) is None:
            option.rich_help_panel = PANEL_HELP
        return option

    def format_help(self, ctx, formatter):
        super().format_help(ctx, formatter)
        pairs = getattr(self.callback, "__hif_examples__", None)
        if not pairs:
            return
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        body = Text()
        for n, (cmd, note) in enumerate(zip(pairs[::2], pairs[1::2])):
            if n:
                body.append("\n")
            body.append(cmd + "\n", style="bold cyan")
            body.append("    " + note + "\n", style="dim")
        Console().print(
            Panel(body, title="Examples", title_align="left", border_style="dim")
        )


# ---------------------------------------------------------------------------
# Option help shared by more than one command
# ---------------------------------------------------------------------------
#
# These strings were identical copies on two or three commands. A help string
# that describes one behaviour has to say the same thing everywhere, and three
# copies is three chances for it not to.
#
# Every one of them leads with what you get for passing the flag. The reason a
# default is the default, the design pressure behind an opt-in, and the history
# of a name are all real — and they belong in the module docstring or the
# generated reference, where a reader has room to read them, not in a column
# forty characters wide that someone is scanning for one answer.

UNITS_HELP = (
    "Add a per-measurement units block to each record. Constant per "
    "signal_set_version, so off by default; `hif schema` prints the same "
    "information without running a model."
)

TRACE_DIR_HELP = (
    "Where --trace artifacts are written (default: <output-dir>/traces, "
    "or ./traces when no --output-dir). Passing this implies --trace."
)

CHARTS_HELP = (
    "One interactive Plotly HTML per signal, plus an index.html dashboard. "
    "Needs --output-dir."
)

# --regime is a recorded label, not a switch: nothing validates it and no
# measurement changes because of it. The built-in suite's vocabulary is worth
# listing anyway — it is the only set of regime names anything else in the
# tool knows (--sample-set, docs/PROMPT_SUITE.md) — so it is read from the
# module that defines it. A hand-typed copy here is exactly the drift a
# generated flag reference exists to prevent.
from hif.prompts.regimes import REGIMES as _REGIMES  # noqa: E402

REGIME_LABEL_HELP = (
    "A free-form label recorded with the run — any string, compared against "
    "nothing, changing no measurement. The built-in suite's regimes: "
    + ", ".join(r.name for r in _REGIMES) + "."
)
