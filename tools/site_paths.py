"""Where the ai-interpretability checkout is — resolved once, for every tool.

Four tools need it (sync_docs, gen_site_measurements, check_corpus,
regen_docs) and each used to compute `Path(__file__).parent.parent.parent /
"ai-interpretability"` for itself. That expression is right in a normal
checkout and wrong in every git worktree, where the repo root is
`<repo>/.claude/worktrees/<name>` and its parent holds sibling worktrees
rather than sibling repositories.

The failure was silent, which is what made it worth a module. Each tool treats
a missing site as "not checked out, nothing to do" and passes quietly, so that
the CLI can be worked on alone. In a worktree that branch was taken for the
wrong reason: the site WAS checked out, the tool looked in a directory that
has never existed, and the pre-commit guard against publishing stale docs
stopped firing for exactly the people doing branch work. A guard that no-ops
where the work happens is worse than no guard, because the passing exit code
reads as evidence.

`git rev-parse --git-common-dir` is the fix: in a linked worktree it names the
MAIN checkout's `.git`, so its parent is the directory the sibling assumption
was always about. In an ordinary checkout it names our own, and the answer is
unchanged — this is not a special case bolted on, it is the general form.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE_NAME = "ai-interpretability"


def main_worktree(repo: Path = REPO) -> Path:
    """The main checkout's root — `repo` itself unless it is a linked worktree.

    Falls back to `repo` when git cannot answer: no git on PATH, a source
    tarball with no history, a sandbox that blocks subprocesses. The fallback
    is the behaviour every caller had before this module existed, so a tool
    that cannot run git is no worse off than it was.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=repo, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return repo
    if out.returncode != 0 or not out.stdout.strip():
        return repo
    # `--git-common-dir` is `<main>/.git` for both a linked worktree and an
    # ordinary one. A bare repo would give a directory whose parent is not a
    # checkout at all; callers only ever probe the result with `is_dir()`, so
    # a wrong guess there degrades to the same quiet skip as an absent site.
    return Path(out.stdout.strip()).parent


def site_repo(repo: Path = REPO) -> Path:
    """The ai-interpretability checkout, as a sibling of the MAIN worktree."""
    return main_worktree(repo).parent / SITE_NAME


def site_docs(repo: Path = REPO) -> Path:
    """The site's published-docs directory (`public/docs`)."""
    return site_repo(repo) / "public" / "docs"


def site_data(repo: Path = REPO) -> Path:
    """The site's published-corpus directory (`public/data`)."""
    return site_repo(repo) / "public" / "data"
