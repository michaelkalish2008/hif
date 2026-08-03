"""`hif models --json`: the model catalogue as data on stdout.

`models` used to render only to stderr, so the one command that answers "what
can I pass to profile?" put zero bytes on stdout and could not be piped,
scripted, or read by a picker — while the README promised stdout carries JSON.
These cover the document's shape and, most of all, the stdout/stderr split that
makes `hif models --json | jq` work.
"""

import json

from typer.testing import CliRunner

from hif.cli import app
from hif.engine import DEFAULT_SURROGATE_MODEL_ID
from hif.models.capabilities import BACKENDS

# Click 8.2+ keeps the two streams apart on its own, which is what lets these
# assert the split rather than take it on trust.
runner = CliRunner()


def _document(*args) -> dict:
    result = runner.invoke(app, ["models", "--json", *args])
    assert result.exit_code == 0, result.stderr
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# The stdout contract
# ---------------------------------------------------------------------------


def test_stdout_is_parseable_json():
    """The whole point: `hif models --json | jq` has to work."""
    assert _document()["backends"]


def test_human_rendering_stays_off_stdout():
    """Without --json, stdout carries nothing — the table is stderr, so a
    pipeline that expects data gets an empty stream rather than rich markup."""
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert result.stdout == ""


def test_unknown_backend_writes_nothing_to_stdout():
    """A failure must not emit a half-document a caller would parse."""
    result = runner.invoke(app, ["models", "--json", "--backend", "bogus"])
    assert result.exit_code == 1
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_every_backend_is_present():
    names = [b["name"] for b in _document()["backends"]]
    assert names == list(BACKENDS)


def test_backend_filter_narrows_the_document():
    doc = _document("--backend", "hf")
    assert [b["name"] for b in doc["backends"]] == ["hf"]


def test_each_backend_carries_its_model_options():
    """The question the command exists to answer."""
    for entry in _document()["backends"]:
        assert entry["models"] == BACKENDS[entry["name"]].example_models


def test_static_examples_are_labelled_as_examples():
    """A caller has to be able to tell a worked example from a live catalogue,
    or it will present five models as though they were the whole of `hf`."""
    for entry in _document()["backends"]:
        assert entry["models_source"] == "examples"
        assert entry["models_note"] is None


def test_signals_are_split_into_available_and_unavailable():
    for entry in _document()["backends"]:
        available = set(entry["signals"]["available"])
        unavailable = set(entry["signals"]["unavailable"])
        assert available and not (available & unavailable)


def test_teacher_forcing_matches_the_capability_table():
    """The document is a view of BACKENDS, not a second copy of it."""
    for entry in _document()["backends"]:
        assert entry["teacher_forcing"] == BACKENDS[entry["name"]].teacher_forcing


def test_open_backends_report_more_signals_than_hosted_ones():
    """The substantive claim a user reads this document to check: teacher
    forcing buys the input-side measurements."""
    by_name = {b["name"]: b for b in _document()["backends"]}
    assert len(by_name["hf"]["signals"]["available"]) > len(
        by_name["anthropic"]["signals"]["available"]
    )


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------


def test_surrogates_json_marks_exactly_one_default(monkeypatch):
    # cli.py binds the name at import, so this is the binding that runs — and
    # patching it keeps the test off the Hugging Face Hub.
    monkeypatch.setattr(
        "hif.cli._check_surrogate_candidates",
        lambda: [(DEFAULT_SURROGATE_MODEL_ID, "ok"), ("gpt2", "ok")],
    )
    doc = _document("--surrogates")
    defaults = [c for c in doc["surrogate_candidates"] if c["default"]]
    assert [c["model"] for c in defaults] == [DEFAULT_SURROGATE_MODEL_ID]
