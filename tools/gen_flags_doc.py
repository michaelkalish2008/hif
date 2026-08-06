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

`--backend` is the one flag whose help is a list. In a terminal `hf | tlens |
ollama | ...` is fine; in a table cell every one of those pipes has to be
escaped, and the sentence that follows them wraps into a column of its own.
Three decisions follow, and the order matters:

1. The cell KEEPS the values, rewritten with commas instead of pipes. A flag
   cell that does not say what you may type has stopped being a flag
   reference. "See the table below" is not an answer to "what can I type?",
   and it is no answer at all to someone deep-linked to one command.
2. The capabilities go in ONE shared `## Backends` section, emitted once and
   linked by anchor. Four commands take `--backend`; repeating the table under
   each of them was six rows of identical prose four times over.
3. That table is one table, not a "full fidelity" / "output-side only" pair.
   The registry has three output tiers (full, top-k, selected-only) crossed
   with a binary input axis, and any two-bucket split has to put `openai`
   (top-k, entropy works) next to `anthropic` (selected-only, entropy
   degenerates) and call them the same thing. Bucket headings would also be
   the one hand-written claim in a generated file — invisible to
   introspection, and a lie the first time a backend lands that does not fit
   them. Columns carry the same facts and cannot drift: each maps to a field
   on the row.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import typer

from hif.cli import app
import hif.cli  # noqa: F401 — importing registers every command on `app`
from hif.models.capabilities import BACKENDS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "FLAGS.md"

# The shared section, and the anchor every `--backend` cell points at. Both
# are derived from this one name so a link can never outlive its target.
BACKENDS_HEADING = "Backends"
BACKENDS_ANCHOR = "#" + BACKENDS_HEADING.lower()
BACKENDS_LINK = f"see [{BACKENDS_HEADING}]({BACKENDS_ANCHOR})"

# How each BackendInfo.kind reads in a table. The registry's own values are
# identifiers; these are the words for a reader. `.get(v, v)` everywhere it is
# used, so an unmapped kind degrades to the raw identifier — accurate but
# unpolished — rather than vanishing.
KIND_LABELS = {
    "local-open": "local, open weights",
    "local-service": "local service",
    "hosted-api": "hosted API",
}

# Worst-to-best is the reading order that makes the table's point, so rank
# best-first and sort. `logprobs` values not listed here sort last and are
# rendered verbatim: a new tier shows up as an unfamiliar word in a column,
# which is a reader noticing something new, not the doc claiming something
# false.
LOGPROB_RANK = {"full": 0, "top-k": 1, "selected-only": 2}

# Backend names longest-first, so the alternation cannot match `hf` inside a
# longer name (`hf-vlm`, while that existed) and leave the rest stranded in
# the middle of an enumeration.
_ALT = "|".join(re.escape(n) for n in sorted(BACKENDS, key=len, reverse=True))
# `hf | tlens | ollama | ...` — two or more names joined by the table
# delimiter. Two is the floor deliberately: a help text that names one backend
# in passing ("Ignored when the target backend already teacher-forces") is
# prose, not an enumeration, and must survive untouched.
_PIPE_ENUM = re.compile(rf"(?:{_ALT})(?:\s*\|\s*(?:{_ALT}))+")

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


def _offered_backends(help_text: str) -> list[str]:
    """Which backends this `--backend` accepts, in registry (best-first) order.

    Read off the help text rather than assumed: a command restricted to a
    subset says so in its help, and enumerating the whole registry at it would
    document backends it rejects. A help text that names fewer than two is
    enumerating nothing — `hif config show --backend` says only "Model
    backend" — so it falls back to the whole registry.
    """
    named = [
        n for n in BACKENDS
        # `(?![\w-])` keeps `hf` from matching inside a longer name such as
        # `hf-vlm`; without it, mentioning a compound backend would count as
        # mentioning the one it is named after.
        if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", help_text)
    ]
    return named if len(named) >= 2 else list(BACKENDS)


def _relist(help_text: str, names: list[str]) -> str:
    """Rewrite the inline enumeration with commas, and point at the section.

    The values stay in the cell — the pipes are what had to go, not the list.
    Rewriting from the registry rather than reflowing the help string also
    means the cell cannot enumerate a backend the registry does not have.

    No default is marked here. `_rows` already appends `*(default: hf)*` from
    the parameter itself, and a cell that says it twice is the duplication
    this change exists to remove.
    """
    listed = ", ".join(f"`{n}`" for n in names)
    # `.sub` is a no-op when there is no enumeration — `hif config show
    # --backend` is just "Model backend". Nothing is invented into the cell;
    # the link is appended either way and the values stay one jump away.
    out = _PIPE_ENUM.sub(listed, help_text).rstrip()
    # A help string that already ends in a sentence gets the link as its own
    # sentence; a fragment gets it as a trailing clause. The distinction is
    # not fussiness — the help strings genuinely differ, and one rule for both
    # produces either ". — see" or a dangling capital mid-sentence.
    if out.endswith((".", "!", "?")):
        return f"{out} See [{BACKENDS_HEADING}]({BACKENDS_ANCHOR})."
    return f"{out} — {BACKENDS_LINK}"


def _backend_default(root) -> str | None:
    """The `--backend` default, asserted to be the same for every command.

    The shared section marks one backend as *the* default. That is a claim
    about all four commands at once, and nothing else would notice if one of
    them drifted — so the generator refuses rather than publishing a table
    that is quietly wrong for one command. `None` defaults are skipped:
    `hif models --backend` is a filter, not a selection, and has no default
    to disagree with.
    """
    found: dict[str, str] = {}
    for name, cmd in _walk(root):
        for p in cmd.params:
            if "--backend" in getattr(p, "opts", []) and p.default is not None:
                found[name] = str(p.default)
    distinct = set(found.values())
    if len(distinct) > 1:
        raise RuntimeError(
            "--backend defaults disagree across commands, so no single "
            f"default can be marked in the {BACKENDS_HEADING} table: {found}"
        )
    return distinct.pop() if distinct else None


def _backends_section(default: str | None) -> str:
    """The one shared capability table, every column a field on the row."""
    ordered = sorted(
        BACKENDS.values(),
        key=lambda b: (LOGPROB_RANK.get(b.logprobs, len(LOGPROB_RANK)), b.name),
    )
    rows = [
        "| `{name}`{mark} | {access} | {tf} | {lp} | {notes} |".format(
            name=b.name,
            mark=" (default)" if b.name == default else "",
            access=KIND_LABELS.get(b.kind, b.kind),
            tf="yes" if b.teacher_forcing else "no",
            lp=b.logprobs,
            notes=_clean(b.notes),
        )
        for b in ordered
    ]
    return "\n".join([
        f"## {BACKENDS_HEADING}",
        "",
        "What every `--backend` accepts, and what each one lets you measure. "
        "Generated from `hif/models/capabilities.py`, the same registry "
        "`hif models` and `hif doctor` read.",
        "",
        "| backend | access | input-side signals | output logprobs | notes |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
    ])


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
            dflt = p.default
            shown = (
                "" if dflt is None or dflt is False
                else "" if callable(dflt)
                else f" *(default: `{dflt}`)*"
            )
            help_text = p.help or ""
            if "--backend" in p.opts:
                help_text = _relist(help_text, _offered_backends(help_text))
            # The link is markdown and must survive `_clean`, which escapes
            # pipes — it has none — so cleaning first and appending would be
            # equivalent. Cleaning the whole cell keeps one code path.
            opts.append(f"| {names} | {_clean(help_text)}{shown} |")
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


def _walk(root):
    """Every leaf command as (display path, command), groups flattened."""
    for name, sub in sorted(root.commands.items()):
        if getattr(sub, "commands", None):
            for sub_name, leaf in sorted(sub.commands.items()):
                yield f"{name} {sub_name}", leaf
            continue
        yield name, sub


def build() -> str:
    root = typer.main.get_command(app)
    # Resolved before anything is rendered so the run fails on a disagreement
    # rather than publishing a table that is quietly wrong for one command.
    default = _backend_default(root)
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
    parts += [_section(path, cmd) for path, cmd in _walk(root)]

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
    # Last, and once. Every `--backend` cell links here by anchor, so the
    # section's position is a reading-order choice rather than a constraint —
    # and a reference table belongs at the back of a reference document.
    parts.append(_backends_section(default))
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT.relative_to(ROOT)}")
