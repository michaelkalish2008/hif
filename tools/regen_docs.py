#!/usr/bin/env python3
"""Run every generator the pre-commit hook checks, in dependency order.

    python3 tools/regen_docs.py             # regenerate, then sync to the site
    python3 tools/regen_docs.py --check     # exit 1 if anything has drifted
    python3 tools/regen_docs.py --site DIR  # a site checkout somewhere else

The hook guards several derived surfaces, and until now told you about them one
at a time. Regenerating docs/FLAGS.md changed a file the site publishes, which
tripped the sync check on the next attempt, which was a second failed commit
to learn a command you could have run first. The order is not a matter of
taste — sync_docs copies FLAGS.md, so it has to run after the thing that
writes it — so it belongs in one place rather than in whatever order you
happened to hit the errors.

Each step is a real command you can still run alone; this only sequences them.
Steps that need the site are skipped, not failed, when it is not checked out,
because working on the CLI alone must not require the website present.

That skip used to fire in every git worktree, where the site is checked out
and the sibling path the tools looked at is another worktree. It reported
"no site checkout" and exited 0, so the guard against publishing stale docs
was off for anyone working on a branch. The site is now located through
`git rev-parse --git-common-dir` (tools/site_paths.py), resolved once here and
passed to each child, so the skip means what it says.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from site_paths import site_repo

ROOT = Path(__file__).resolve().parent.parent

# (script, how it takes the site path — None when it needs no site at all).
# Order is dependency order: the generators write files in this repo, and
# sync_docs copies them onward. gen_backend_tiers writes into README.md, which
# sync_docs publishes, so it has the same before-the-sync constraint
# gen_flags_doc has.
#
# The two site-aware tools disagree about both the flag and what it points at
# — `--site` is the site repo, `--site-docs` is a directory inside it. That is
# recorded here rather than normalised: renaming either flag to suit this
# script would give one option two names, and both are commands a person runs
# directly.
STEPS: list[tuple[str, tuple[str, str] | None]] = [
    ("gen_flags_doc.py", None),
    ("gen_backend_tiers.py", None),
    ("gen_site_measurements.py", ("--site", ".")),
    ("sync_docs.py", ("--site-docs", "public/docs")),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Report drift and exit 1 instead of writing. For hooks and CI.",
    )
    parser.add_argument(
        "--site", type=Path, default=None, metavar="DIR",
        help="The ai-interpretability checkout (default: the sibling of the "
             "main worktree, which is not the same directory as the sibling "
             "of a linked one).",
    )
    args = parser.parse_args()

    # Resolved once here and passed down, rather than left to each child to
    # work out for itself. Four tools computing the same path independently is
    # how they came to be wrong in a worktree independently.
    site = (args.site or site_repo()).resolve()
    have_site = site.is_dir()
    failed: list[str] = []
    for script, site_spec in STEPS:
        if site_spec is not None and not have_site:
            print(f"regen_docs: {script} skipped — no site checkout at {site}")
            continue
        cmd = [sys.executable, str(ROOT / "tools" / script)]
        if site_spec is not None:
            flag, rel = site_spec
            cmd += [flag, str(site if rel == "." else site / rel)]
        if args.check:
            cmd.append("--check")
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failed.append(script)
            # Keep going. One run that names every drifted surface is the
            # whole point; stopping at the first would reproduce the
            # one-at-a-time discovery this script exists to end.

    if not failed:
        return 0
    print(
        f"\nregen_docs: {', '.join(failed)} reported drift."
        + ("" if args.check else " Re-run to see whether it settles."),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
