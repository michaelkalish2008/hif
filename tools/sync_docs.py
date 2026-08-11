#!/usr/bin/env python3
"""Copy this repo's documentation into the website that renders it.

The ai-interpretability site does not author a description of hif. Its CLI page
renders `public/docs/README.md` and `public/docs/FLAGS.md`, which are copies of
the files here — the arrangement exists because the site once held four rival
copies of the measurement reference, no two of which agreed on a formula.

The reference docs are deliberately NOT among them. Publishing them put ~20k
words on a page nobody read, duplicated the site's own Methodology narrative,
and gave every fact two homes to drift between — which is how the site came to
claim an embedder the tool has never actually used. They live in `docs/`, where
anyone running the CLI already is, and the README links out to them.

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
    Path("docs/FLAGS.md"): "FLAGS.md",
}

# Repository-relative links have no meaning once the file is served by the
# site, so each one is rewritten to the route that renders the same document.
# Applied to every file, not just the README: a link added to any of them
# later should not need this table to be found and extended by hand.
_BLOB = "https://github.com/michaelkalish2008/hif/blob/main/"

LINK_REWRITES = {
    "](CONTRIBUTING.md)": f"]({_BLOB}CONTRIBUTING.md)",
    "](AGENTS.md)": f"]({_BLOB}AGENTS.md)",
    "](../CONTRIBUTING.md)": f"]({_BLOB}CONTRIBUTING.md)",
    # The reference docs are no longer published by the site — they live in the
    # repo, where anyone using the CLI already is. The README still links to
    # them, so those links have to leave the site rather than 404 inside it.
    "](docs/MEASUREMENTS.md)": f"]({_BLOB}docs/MEASUREMENTS.md)",
    "](docs/CONFIG.md)": f"]({_BLOB}docs/CONFIG.md)",
    "](docs/READING.md)": f"]({_BLOB}docs/READING.md)",
    "](MEASUREMENTS.md)": f"]({_BLOB}docs/MEASUREMENTS.md)",
    "](CONFIG.md)": f"]({_BLOB}docs/CONFIG.md)",
    "](READING.md)": f"]({_BLOB}docs/READING.md)",
    # docs/ files linking up to the repo README, which the site renders at /cli.
    "](../README.md)": "](/cli)",
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
