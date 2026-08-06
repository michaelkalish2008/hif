"""Credential resolution: one place, one precedence order, and a `doctor`
that reports what the next command will actually get.

The bug these cover: `doctor` read os.environ while `_load_model` separately
auto-loaded a dotenv, so a preflight could report a credential unset that the
very next run would find. Resolution now happens once, in the `hif` callback,
and every command inherits the same environment.
"""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hif.cli import app
from hif.cli._app import (
    ENV_SOURCES,
    discover_env_files,
    env_origin,
    load_env_file,
)

runner = CliRunner()

KEY = "HIF_TEST_CREDENTIAL"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with the probe key unset and no recorded origins."""
    monkeypatch.delenv(KEY, raising=False)
    saved = dict(ENV_SOURCES)
    ENV_SOURCES.clear()
    yield
    ENV_SOURCES.clear()
    ENV_SOURCES.update(saved)


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_reads_bare_assignments(tmp_path):
    env = _write(tmp_path / ".env", f"{KEY}=value\n")
    assert load_env_file(env) == 1
    assert os.environ[KEY] == "value"


def test_tolerates_export_prefix_and_quotes(tmp_path):
    """A dotenv people also `source` carries `export` and quotes; both are the
    shell's syntax, not part of the value."""
    env = _write(tmp_path / ".env", f'export {KEY}="quoted value"\n')
    load_env_file(env)
    assert os.environ[KEY] == "quoted value"


def test_skips_comments_and_blank_lines(tmp_path):
    env = _write(tmp_path / ".env", f"# a comment\n\n{KEY}=value\n")
    assert load_env_file(env) == 1


def test_ignores_lines_without_an_assignment(tmp_path):
    env = _write(tmp_path / ".env", f"not an assignment\n{KEY}=value\n")
    assert load_env_file(env) == 1
    assert os.environ[KEY] == "value"


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


def test_a_real_export_is_never_overridden(tmp_path, monkeypatch):
    """The whole safety property: an explicit `KEY=… hif …` or a CI environment
    is a deliberate choice, and a file found on disk must not replace it."""
    monkeypatch.setenv(KEY, "from-environment")
    env = _write(tmp_path / ".env", f"{KEY}=from-file\n")
    assert load_env_file(env) == 0
    assert os.environ[KEY] == "from-environment"


def test_first_file_read_wins(tmp_path):
    """Precedence across files falls out of never replacing a set value, so
    callers get it right by reading in order."""
    first = _write(tmp_path / "first.env", f"{KEY}=first\n")
    second = _write(tmp_path / "second.env", f"{KEY}=second\n")
    load_env_file(first)
    load_env_file(second)
    assert os.environ[KEY] == "first"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovers_dotenv_in_the_working_directory(tmp_path, monkeypatch):
    _write(tmp_path / ".env", f"{KEY}=here\n")
    monkeypatch.chdir(tmp_path)
    assert discover_env_files()[0] == (tmp_path / ".env").resolve()


def test_discovers_dotenv_above_the_working_directory(tmp_path, monkeypatch):
    """The ordinary case: run from a subdirectory of a project that keeps its
    keys at the root."""
    _write(tmp_path / ".env", f"{KEY}=root\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert discover_env_files()[0] == (tmp_path / ".env").resolve()


def test_stops_at_the_nearest_dotenv(tmp_path, monkeypatch):
    """Two dotenvs silently merging is worse than the one you are standing in."""
    _write(tmp_path / ".env", f"{KEY}=far\n")
    near = tmp_path / "a"
    near.mkdir()
    _write(near / ".env", f"{KEY}=near\n")
    monkeypatch.chdir(near)
    found = discover_env_files()
    assert found[0] == (near / ".env").resolve()
    assert (tmp_path / ".env").resolve() not in found


def test_user_config_file_is_read_after_the_project_one(tmp_path, monkeypatch):
    """The installed user with no project directory still has one place to put
    keys — and a project that has its own keeps priority."""
    home = tmp_path / "home"
    user_env = home / ".config" / "hif"
    user_env.mkdir(parents=True)
    _write(user_env / ".env", f"{KEY}=user\n")
    monkeypatch.setattr("hif.cli._app.USER_ENV_FILE", user_env / ".env")

    project = tmp_path / "project"
    project.mkdir()
    _write(project / ".env", f"{KEY}=project\n")
    monkeypatch.chdir(project)

    found = discover_env_files()
    assert found == [(project / ".env").resolve(), user_env / ".env"]
    for path in found:
        load_env_file(path)
    assert os.environ[KEY] == "project"


# ---------------------------------------------------------------------------
# Origin reporting
# ---------------------------------------------------------------------------


def test_origin_names_the_file_that_supplied_the_value(tmp_path):
    env = _write(tmp_path / ".env", f"{KEY}=value\n")
    load_env_file(env)
    assert str(env) in env_origin(KEY) or env_origin(KEY).endswith(".env")


def test_origin_of_an_inherited_value_is_the_environment(monkeypatch):
    monkeypatch.setenv(KEY, "value")
    assert env_origin(KEY) == "environment"


def test_origin_of_an_unset_variable_is_empty():
    assert env_origin(KEY) == ""


# ---------------------------------------------------------------------------
# End to end through the CLI
# ---------------------------------------------------------------------------


def test_doctor_reports_a_discovered_dotenv(tmp_path, monkeypatch):
    """The regression this whole change exists for: `doctor` used to report
    unset for a file the run would go on to read."""
    _write(tmp_path / ".env", "OPENAI_API_KEY=sk-test\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("hif.cli._app.USER_ENV_FILE", tmp_path / "absent")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "OPENAI_API_KEY: set" in result.output


def test_env_file_option_beats_discovery(tmp_path, monkeypatch):
    _write(tmp_path / ".env", "OPENAI_API_KEY=from-discovered\n")
    named = _write(tmp_path / "named.env", "OPENAI_API_KEY=from-named\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("hif.cli._app.USER_ENV_FILE", tmp_path / "absent")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["--env-file", str(named), "doctor"])
    assert result.exit_code == 0
    assert os.environ["OPENAI_API_KEY"] == "from-named"


def test_missing_env_file_is_an_error_not_a_silent_fallback(tmp_path):
    """Naming a file that is not there is a typo, and continuing on a
    discovered file instead would spend tokens against the wrong account."""
    result = runner.invoke(app, ["--env-file", str(tmp_path / "absent"), "doctor"])
    assert result.exit_code == 2
