"""Input truncation — the only truncation this package exposes.

`hif profile --truncate N` shortens the prompt to N whitespace-separated
words BEFORE any analysis, and stamps the record so a consumer can tell a
truncated-context run apart from a full-context one. Truncation is never a
silent default: it happens only when the flag is passed, and it always warns
on stderr.

(This module previously tested `_truncate_output`/`_trunc_label`/
`_parse_truncation_tokens`/`_parse_variants` from the Modal study runner —
an output-side, env-var-driven sweep harness that is not part of this
package. Those functions are unreachable here, so their tests are gone and
these cover the truncation behaviour that IS reachable.)
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import hif.cli as cli
from hif.cli import app
from tests.unit.profile_helpers import _make_profile

runner = CliRunner()

LONG_PROMPT = "one two three four five six seven eight nine ten"


@pytest.fixture()
def captured(monkeypatch):
    """Patch the pipeline and capture the prompt it actually received."""
    seen: dict = {}

    def fake_run(*args, **kwargs):
        seen["prompt"] = kwargs.get("prompt", args[1] if len(args) > 1 else None)
        return _make_profile(), None

    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_load_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_run_single_profile", fake_run)
    return seen


def _invoke(tmp_path, *extra):
    return runner.invoke(
        app, ["profile", "m", LONG_PROMPT, "--output-dir", str(tmp_path), *extra]
    )


# ---------------------------------------------------------------------------
# The flag is required — truncation is never silent
# ---------------------------------------------------------------------------


def test_no_flag_means_full_prompt(captured, tmp_path):
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == LONG_PROMPT
    assert "truncated" not in result.output.lower()


def test_truncate_shortens_prompt_and_warns(captured, tmp_path):
    result = _invoke(tmp_path, "--truncate", "4")
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "one two three four"
    assert "Input truncated to 4 tokens" in " ".join(result.output.split())


def test_truncate_longer_than_prompt_is_a_no_op(captured, tmp_path):
    result = _invoke(tmp_path, "--truncate", "500")
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == LONG_PROMPT
    assert "Input truncated" not in result.output


def test_truncate_exactly_at_length_is_a_no_op(captured, tmp_path):
    result = _invoke(tmp_path, "--truncate", str(len(LONG_PROMPT.split())))
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == LONG_PROMPT
    assert "Input truncated" not in result.output


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_non_positive_truncate_exits_3(captured, tmp_path, bad):
    result = _invoke(tmp_path, "--truncate", bad)
    assert result.exit_code == 3
    assert "positive integer" in result.output
    assert "prompt" not in captured  # rejected before the pipeline ran


# ---------------------------------------------------------------------------
# Record provenance: a truncated run is labeled as one
# ---------------------------------------------------------------------------


def test_record_records_truncation(captured, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.console, "width", 100_000)  # no soft-wrap in JSON
    result = runner.invoke(app, [
        "profile", "m", LONG_PROMPT, "--output-dir", str(tmp_path),
        "--truncate", "3", "--json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["input_truncated"] is True
    assert data["input_truncate_tokens"] == 3
    # The record's hash is over the prompt that was actually analysed, not the
    # original — otherwise the hash would not describe the run. (The prompt
    # text itself is never in the record: derived values only.)
    from hif.profile.signals import profile_hash

    assert data["hash"] == profile_hash("m", "one two three", data["seed"])
    assert data["hash"] != profile_hash("m", LONG_PROMPT, data["seed"])


def test_untruncated_record_carries_no_truncation_keys(captured, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.console, "width", 100_000)
    result = runner.invoke(app, [
        "profile", "m", LONG_PROMPT, "--output-dir", str(tmp_path), "--json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert "input_truncated" not in data
    assert "input_truncate_tokens" not in data
