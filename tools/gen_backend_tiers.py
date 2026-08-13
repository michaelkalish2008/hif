"""Generate the README's backend access-tier table, from BACKENDS.

Run: python3 tools/gen_backend_tiers.py            # write the block
     python3 tools/gen_backend_tiers.py --check    # exit 1 if it has drifted
     python3 tools/regen_docs.py                   # this, the rest, and the sync

The table lives between markers in README.md and is REPLACED wholesale, for
the same reason docs/FLAGS.md § Backends is generated from this same registry:
a hand-kept list of backends agrees with `hif/models/capabilities.py` on the
day it is typed and drifts silently afterwards. This one had drifted three
ways at once — it listed `deepseek`, which is not a backend and never was; it
omitted `ollama`, which is; and it credited the `[F]` tier with "attention",
which no backend has ever been asked to expose (`hif/analysis/attention.py`
runs its own encoder over text, and hif-v4 cut both attention rows besides).

The two tables are not redundant, and neither can drift from the other because
neither is typed. FLAGS.md's is per-backend and answers "what do I get from
`openai`"; this one is per-tier and answers "what class of thing am I giving
up by not running local weights".

One thing the table deliberately does NOT carry is gemini's endpoint split —
Vertex AI returns logprobs and the developer API degenerates, so the same
model is `[T-k]` on one and `[P]` on the other. That is real, and it is not a
field on the row: `BackendInfo` has one `logprobs` value per backend. Rather
than invent a second gemini row the registry cannot justify, the README states
it as prose under the table and points at `hif models`, which reports what a
given set of credentials actually reaches.

What is derived and what is not
-------------------------------
MEMBERSHIP is derived, and is the whole point. The three access tiers are not
a taxonomy laid over the registry — they are exactly the `logprobs` field:
`full` -> `[F]`, `top-k` -> `[T-k]`, `selected-only` -> `[P]`. A backend
cannot be in the wrong row here because nothing chooses its row. The
teacher-forcing column is read off `BackendInfo.teacher_forcing` the same way,
and says `varies` rather than picking a side if a tier's backends ever
disagree.

The tier LABELS and the one-line descriptions are written here, once. They are
the readable name for a `logprobs` value and a sentence about what that value
costs you — neither is on the row, and neither can be introspected out of one.
Keeping them in a generated file is the risk `tools/gen_flags_doc.py` names:
a hand-written claim inside generated output is invisible to the generator and
becomes a lie the first time a backend lands that does not fit it. So it is
made visible — an unmapped `logprobs` value raises rather than dropping its
backends out of a table that still reads as complete. A new tier stops the
build; it does not quietly stop being documented.

The `[F]`/`[T-k]`/`[P]` vocabulary is used across docs/MEASUREMENTS.md,
AGENTS.md and tests/unit/mock_backends.py, so it is spelled the same way here
rather than coined fresh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# Importable without an editable install: the pre-commit hook runs this on a
# checkout that may never have had `pip install -e .` run in it, and a drift
# check that fails because of its own import is indistinguishable, at the
# terminal, from one that fails because the doc is wrong.
sys.path.insert(0, str(ROOT))

from hif.models.capabilities import BACKENDS, BackendInfo  # noqa: E402

BEGIN = "<!-- generated: backend access tiers — tools/gen_backend_tiers.py -->"
END = "<!-- /generated: backend access tiers -->"

# (logprobs value on the row, tier tag, tier name, what that tier costs you).
# The first field is the derivation key; the rest are the words for a reader.
TIERS: list[tuple[str, str, str, str]] = [
    (
        "full", "[F]", "full",
        "full-vocabulary distributions — every measurement",
    ),
    (
        "top-k", "[T-k]", "truncated",
        "top-k logprobs only; output entropy is a lower bound",
    ),
    (
        "selected-only", "[P]", "proxy",
        # Phrased in the singular-safe form on purpose: how MANY rows are
        # divergences is a registry fact, and CONTRIBUTING forbids writing a
        # count of measurements into prose. "a divergence between
        # distributions" stays true whether the registry holds one or five.
        "the selected token only; the entropy-shaped measurements degenerate "
        "and a divergence between distributions is absent outright",
    ),
]


def _teacher_forcing(rows: list[BackendInfo]) -> str:
    """`yes` / `no` for a tier whose backends agree, `varies` when they do not.

    Teacher forcing is a separate field from `logprobs`, and today the two
    happen to move together. If a backend ever lands where they do not, the
    honest cell is the one that says so and sends the reader to the per-backend
    list — not a majority vote that reads as a fact about every row.
    """
    flags = {r.teacher_forcing for r in rows}
    if flags == {True}:
        return "yes"
    if flags == {False}:
        return "no"
    return "varies — see `hif models`"


def tier_members() -> dict[str, list[str]]:
    """Tier tag -> backend names, in the registry's own best-first order."""
    unknown = sorted(
        {i.logprobs for i in BACKENDS.values()} - {t[0] for t in TIERS}
    )
    if unknown:
        raise SystemExit(
            f"gen_backend_tiers: no access tier for logprobs={unknown!r}.\n"
            "  A backend whose logprobs value is not in TIERS would be missing "
            "from a table that still reads as the complete list. Add the tier "
            "here (tag, name, and what it costs the reader) and regenerate."
        )
    return {
        tag: [n for n, i in BACKENDS.items() if i.logprobs == logprobs]
        for logprobs, tag, _name, _desc in TIERS
    }


def build() -> str:
    members = tier_members()
    lines = [
        BEGIN,
        "",
        "| access | backends | teacher forcing | what you get |",
        "|---|---|---|---|",
    ]
    for _logprobs, tag, name, desc in TIERS:
        names = members[tag]
        if not names:
            continue
        rows = [BACKENDS[n] for n in names]
        listed = ", ".join(f"`{n}`" for n in names)
        lines.append(
            f"| `{tag}` {name} | {listed} | {_teacher_forcing(rows)} | {desc} |"
        )
    lines += ["", END]
    return "\n".join(lines)


def _replace(text: str, block: str) -> str:
    start, end = text.find(BEGIN), text.find(END)
    if start == -1 or end == -1:
        raise SystemExit(
            f"gen_backend_tiers: {README.name} has no generated block.\n"
            f"  Expected the marker pair:\n    {BEGIN}\n    {END}"
        )
    return text[:start] + block + text[end + len(END):]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Report drift and exit 1 instead of writing. For hooks and CI.",
    )
    args = parser.parse_args()

    current = README.read_text()
    updated = _replace(current, build())
    if args.check:
        if updated != current:
            print(
                f"{README.relative_to(ROOT)}: the backend access-tier table has "
                "drifted from hif/models/capabilities.py.\n"
                "  Run: python3 tools/gen_backend_tiers.py",
                file=sys.stderr,
            )
            return 1
        return 0
    if updated != current:
        README.write_text(updated)
        print(f"wrote the backend access tiers into {README.relative_to(ROOT)}")
    else:
        print(f"{README.relative_to(ROOT)} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
