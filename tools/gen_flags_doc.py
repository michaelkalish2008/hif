"""Generate docs/FLAGS.md — every command, argument and flag, from the CLI itself.

Run: python tools/gen_flags_doc.py

The flag reference is GENERATED, never hand-written. A hand-maintained list of
options is the same failure as a hand-maintained measurement reference: it
agrees with the tool on the day it is written and drifts silently afterwards,
and a flag documented but removed is worse than a flag never documented. This
introspects the live typer app, so the doc cannot claim a flag the CLI does not
have.

`hif doctor` takes no flags — it is listed for completeness, with the checks it
performs, because a reader looking for "what does doctor tell me" should not
have to run it to find out.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import typer

from hif.cli_base import app
import hif.cli  # noqa: F401 — importing registers every command on `app`

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "FLAGS.md"

# What `hif doctor` reports. Kept beside the doctor implementation's own order;
# it takes no options, so there is nothing to introspect.
DOCTOR_CHECKS = [
    ("core (numpy, plotly)", "the two hard dependencies"),
    ("embedder (sentence-transformers)", "needed by every geometric measurement"),
    ("charts (--charts)", "HTML always; PNG only with kaleido installed"),
    ("ollama server", "reachability and which models are pulled"),
    ("per-backend readiness", "optional deps and credentials, one row per backend"),
]


def _clean(text: str) -> str:
    """Collapse help text to one line and escape the table delimiter."""
    return re.sub(r"\s+", " ", (text or "").strip()).replace("|", "\\|")


def _rows(command: click.Command) -> tuple[list[str], list[str]]:
    args, opts = [], []
    for p in command.params:
        kind = getattr(p, "param_type_name", None)
        if kind == "argument":
            args.append(f"| `{p.name}` | {_clean(getattr(p, 'help', '') or '')} |")
        elif kind == "option":
            if "--help" in p.opts:
                continue
            names = ", ".join(f"`{o}`" for o in p.opts)
            default = p.default
            shown = (
                "" if default is None or default is False
                else "" if callable(default)
                else f" *(default: `{default}`)*"
            )
            opts.append(f"| {names} | {_clean(p.help)}{shown} |")
    return args, opts


def _section(path: str, command) -> str:
    args, opts = _rows(command)
    out = [f"## `hif {path}`", ""]
    summary = _clean((command.help or "").split("\n\n")[0])
    if summary:
        out += [summary, ""]
    if args:
        out += ["| argument | meaning |", "| --- | --- |", *args, ""]
    if opts:
        out += ["| flag | meaning |", "| --- | --- |", *opts, ""]
    if not args and not opts:
        out += ["Takes no arguments or flags.", ""]
    return "\n".join(out)


def build() -> str:
    root = typer.main.get_command(app)
    parts = [
        "# Flags — Horizonal Interpretability Framework (HIF)",
        "",
        "Every command, argument and flag, generated from the CLI by "
        "`tools/gen_flags_doc.py`. Do not edit by hand: regenerate it, so this "
        "file cannot claim a flag `hif` does not have.",
        "",
        "Run `hif <command> --help` for the same text in the terminal.",
        "",
        "---",
        "",
    ]
    for name, sub in sorted(root.commands.items()):
        if getattr(sub, "commands", None):
            for sub_name, leaf in sorted(sub.commands.items()):
                parts.append(_section(f"{name} {sub_name}", leaf))
            continue
        parts.append(_section(name, sub))

    parts += [
        "## What `hif doctor` checks",
        "",
        "`doctor` takes no flags. It reports, in order:",
        "",
        "| check | what it tells you |",
        "| --- | --- |",
        *[f"| {name} | {desc} |" for name, desc in DOCTOR_CHECKS],
        "",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT.relative_to(ROOT)}")
