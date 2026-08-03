#!/usr/bin/env python3
"""Copy this repo's documentation into the website that renders it.

The ai-interpretability site does not author a description of hif. Its home
page and Docs tab render `public/docs/*.md`, which are copies of the files
here — the arrangement exists because the site once held four rival copies of
the measurement reference, no two of which agreed on a formula.

A copy is only safe while it is actually a copy. This makes that a command
rather than something to remember:

    python3 tools/sync_docs.py            # write the copies
    python3 tools/sync_docs.py --check    # exit 1 if any has drifted

`--check` is what tools/hooks/pre-commit runs, so a commit that edits a doc
cannot silently leave the published version behind.

The site is a separate repository and this only writes its working tree —
reviewing and committing that change is still a deliberate act over there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SITE_DOCS = REPO.parent / "ai-interpretability" / "public" / "docs"

# source (relative to this repo) → destination filename
DOCS = {
    Path("README.md"): "README.md",
    Path("docs/ARCHITECTURE.md"): "ARCHITECTURE.md",
    Path("docs/MEASUREMENTS.md"): "MEASUREMENTS.md",
    Path("docs/PHILOSOPHY.md"): "PHILOSOPHY.md",
    Path("docs/PROMPT_SUITE.md"): "PROMPT_SUITE.md",
}

# Repository-relative links have no meaning once the file is served by the
# site, so each one is rewritten to the route that renders the same document.
# Applied to every file, not just the README: a link added to any of them
# later should not need this table to be found and extended by hand.
LINK_REWRITES = {
    "](docs/MEASUREMENTS.md)": "](/writing#measurements)",
    "](docs/PROMPT_SUITE.md)": "](/writing#prompt-suite)",
    "](docs/ARCHITECTURE.md)": "](/writing#architecture)",
    "](docs/PHILOSOPHY.md)": "](/writing#philosophy)",
    "](CONTRIBUTING.md)": "](https://github.com/michaelkalish2008/hif/blob/main/CONTRIBUTING.md)",
}


def rendered(source: Path) -> str:
    text = source.read_text()
    for repo_link, site_link in LINK_REWRITES.items():
        text = text.replace(repo_link, site_link)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Report drift and exit 1 instead of writing. For hooks and CI.",
    )
    parser.add_argument(
        "--site-docs", type=Path, default=DEFAULT_SITE_DOCS,
        help=f"The site's docs directory (default: {DEFAULT_SITE_DOCS}).",
    )
    args = parser.parse_args()

    # Not having the site checked out is the ordinary case for anyone who only
    # works on the CLI. That is not a failure, and a hook must not block on it.
    if not args.site_docs.is_dir():
        print(f"sync_docs: {args.site_docs} not present — nothing to sync.")
        return 0

    stale: list[str] = []
    for source_rel, dest_name in DOCS.items():
        source = REPO / source_rel
        if not source.is_file():
            print(f"sync_docs: missing source {source_rel}", file=sys.stderr)
            return 2
        dest = args.site_docs / dest_name
        text = rendered(source)
        if dest.is_file() and dest.read_text() == text:
            continue
        if args.check:
            stale.append(f"{source_rel} → {dest}")
            continue
        dest.write_text(text)
        print(f"sync_docs: wrote {dest}")

    if stale:
        print("sync_docs: the site's copies have drifted:", file=sys.stderr)
        for entry in stale:
            print(f"  {entry}", file=sys.stderr)
        print("\nRun: python3 tools/sync_docs.py", file=sys.stderr)
        return 1

    if args.check:
        print("sync_docs: site copies are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
