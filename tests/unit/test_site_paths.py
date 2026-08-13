"""Locating the site must not depend on which checkout you are standing in.

The published-docs guard is allowed to pass quietly when the site is not
checked out, so that the CLI can be worked on alone. That escape hatch used to
open in every git worktree: the tools looked for the site beside the repo
root, which in a worktree is `<repo>/.claude/worktrees/`, so they reported
"no site checkout" and exited 0 while the site sat one directory further up.
The guard was off for precisely the people working on a branch, and said
nothing about it.

These build a real worktree rather than mocking git, because the bug was in
what git actually reports, and a mock would have agreed with the wrong answer.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="needs a real git to build a worktree"
)


def _module():
    """Import tools/site_paths.py — a script directory, not a package."""
    spec = importlib.util.spec_from_file_location(
        "site_paths", ROOT / "tools" / "site_paths.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A throwaway repo with one commit — enough to add a worktree to."""
    repo = tmp_path / "hif"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "t@example.invalid", cwd=repo)
    _git("config", "user.name", "test", cwd=repo)
    (repo / "README.md").write_text("x\n")
    _git("add", "README.md", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    return repo


def test_ordinary_checkout_is_its_own_main_worktree(checkout: Path):
    assert _module().main_worktree(checkout) == checkout


def test_linked_worktree_resolves_to_the_main_checkout(checkout: Path):
    """The regression: a worktree must not answer with its own root."""
    linked = checkout / ".claude" / "worktrees" / "wt"
    _git("worktree", "add", "-q", "-b", "wt", str(linked), cwd=checkout)

    site_paths = _module()
    assert linked.is_dir()
    assert site_paths.main_worktree(linked) == checkout
    # The whole point: same answer from either checkout, so the site is found
    # once rather than per-worktree.
    assert site_paths.site_repo(linked) == site_paths.site_repo(checkout)
    assert site_paths.site_repo(linked) == checkout.parent / "ai-interpretability"
    # And specifically NOT the sibling-of-the-worktree path, which is a
    # directory of other worktrees.
    assert site_paths.site_repo(linked) != linked.parent / "ai-interpretability"


def test_docs_and_data_hang_off_the_same_site_root(checkout: Path):
    site_paths = _module()
    site = site_paths.site_repo(checkout)
    assert site_paths.site_docs(checkout) == site / "public" / "docs"
    assert site_paths.site_data(checkout) == site / "public" / "data"


def test_falls_back_to_the_repo_when_git_cannot_answer(checkout, monkeypatch):
    """No git, no history, no subprocesses — degrade, do not raise.

    The fallback is the behaviour every caller had before this module, so a
    tool that cannot run git is no worse off than it was.
    """
    site_paths = _module()

    def refuse(*_args, **_kwargs):
        raise OSError("no git here")

    monkeypatch.setattr(site_paths.subprocess, "run", refuse)
    assert site_paths.main_worktree(checkout) == checkout


def test_falls_back_when_git_exits_nonzero(tmp_path: Path):
    """A directory that is not a repository at all."""
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    assert _module().main_worktree(outside) == outside
