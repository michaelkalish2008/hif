#!/usr/bin/env python3
"""Run every generator the pre-commit hook checks, in dependency order.

    python3 tools/regen_docs.py            # regenerate, then sync to the site
    python3 tools/regen_docs.py --check    # exit 1 if anything has drifted

The hook guards three derived surfaces, and until now told you about them one
at a time. Regenerating docs/FLAGS.md changed a file the site publishes, which
tripped the sync check on the next attempt, which was a second failed commit
to learn a command you could have run first. The order is not a matter of
taste — sync_docs copies FLAGS.md, so it has to run after the thing that
writes it — so it belongs in one place rather than in whatever order you
happened to hit the errors.

Each step is a real command you can still run alone; this only sequences them.
Steps that need the site are skipped, not failed, when it is not checked out,
because working on the CLI alone must not require the website present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_DOCS = ROOT.parent / "ai-interpretability" / "public" / "docs"

# (script, needs the site checked out). Order is dependency order: the two
# generators write files in this repo, and sync_docs copies them onward.
STEPS: list[tuple[str, bool]] = [
    ("gen_flags_doc.py", False),
    ("gen_site_measurements.py", True),
    ("sync_docs.py", True),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Report drift and exit 1 instead of writing. For hooks and CI.",
    )
    args = parser.parse_args()

    site = SITE_DOCS.is_dir()
    failed: list[str] = []
    for script, needs_site in STEPS:
        if needs_site and not site:
            print(f"regen_docs: {script} skipped — no site checkout at {SITE_DOCS}")
            continue
        cmd = [sys.executable, str(ROOT / "tools" / script)]
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
