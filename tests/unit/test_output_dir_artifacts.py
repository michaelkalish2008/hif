"""What a profile run puts on disk.

Every run writes its profile JSON; `--output-dir` adds the DERIVED files (the
report, the charts) that are regenerable from it. The JSON used to be withheld
unless `--trace`, on the reasoning that per-step top-K with token identity
should not reach disk by default. That reasoning belonged to the hosted
deployment, where the text being profiled was not the operator's; it is
archived, and a researcher profiling their own prompts on their own machine is
not disclosing anything to themselves.

Where it lands: `--trace-dir`, else a `traces/` directory — under the output
dir when there is one, in the working directory when there is not. `--trace`
decides the artifact's CONTENT, never its existence or its location.
"""

from __future__ import annotations

import json
from pathlib import Path

import hif.engine
from hif.cli import _load, _run
from tests.unit.profile_helpers import _make_profile


class _FakeEngine:
    """Stands in for SessionEngine: no model, same two methods the CLI calls."""

    def __init__(self, config, *args, **kwargs):
        self.config = config

    def profile_one(self, prompt, **kwargs):
        return _make_profile()

    # The real one: hash-addressed naming and real serialization are exactly
    # what these tests are about, and only the model loading needs faking.
    write_trace = hif.engine.SessionEngine.write_trace


def _patch(monkeypatch):
    monkeypatch.setattr(hif.engine, "SessionEngine", _FakeEngine)
    monkeypatch.setattr(_load, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(_load, "_load_embedder", lambda *a, **k: object())


def _run_with(monkeypatch, tmp_path, **kwargs):
    _patch(monkeypatch)
    return _run._run_single_profile(
        model_name="mock-model",
        prompt="hello world",
        regime="factual",
        backend="hf",
        seed=42,
        output_dir=tmp_path,
        max_new_tokens=4,
        top_k=5,
        **kwargs,
    )


def _artifacts(root: Path) -> list[str]:
    """Every file a run leaves: the report, and the JSON in the trace dir."""
    return sorted(
        [p.name for p in root.glob("*_technical.md")]
        + [p.name for p in (root / "traces").glob("profile_*.json")]
    )


def test_output_dir_writes_report_and_json(monkeypatch, tmp_path):
    _, trace_path = _run_with(monkeypatch, tmp_path)

    assert len(list(tmp_path.glob("*_technical.md"))) == 1
    assert trace_path is not None and trace_path.exists()
    assert trace_path.parent == tmp_path / "traces"

    # The JSON is the whole profile, not a summary of it: it round-trips.
    data = json.loads(trace_path.read_text())
    assert data["model"]["name"] == "mock-model"
    assert data["output_side"]["steps"]


def test_trace_changes_the_artifact_not_its_location(monkeypatch, tmp_path):
    """--trace decides what is IN the JSON, not where it lands.

    A file that moves between directories depending on an unrelated flag is a
    file nobody can write a path to, so both runs put it in the trace dir.
    """
    _, without = _run_with(monkeypatch, tmp_path)
    _, with_trace = _run_with(monkeypatch, tmp_path, trace=True)

    assert without.parent == with_trace.parent == tmp_path / "traces"
    assert list(tmp_path.glob("profile_*.json")) == []


def test_trace_dir_overrides_the_default(monkeypatch, tmp_path):
    elsewhere = tmp_path / "somewhere" / "else"
    _, trace_path = _run_with(monkeypatch, tmp_path, trace_dir=elsewhere)

    assert trace_path.parent == elsewhere


def test_lite_and_full_runs_do_not_overwrite_each_others_artifacts(
    monkeypatch, tmp_path
):
    """The artifact names are hash-addressed, and the two runs are not the
    same run: a lite run reports two measurements where a full run reports
    six. While the hash covered only (model, prompt, seed), the second
    invocation silently replaced the first one's report and JSON in place.
    """
    _run_with(monkeypatch, tmp_path)
    full = _artifacts(tmp_path)

    _run_with(monkeypatch, tmp_path, lite=True)
    both = _artifacts(tmp_path)

    assert len(full) == 2, full
    assert len(both) == 4, both
    assert set(full) < set(both)


def _run_bare(monkeypatch, tmp_path):
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    return _run._run_single_profile(
        model_name="mock-model",
        prompt="hello world",
        regime="factual",
        backend="hf",
        seed=42,
        output_dir=None,
        max_new_tokens=4,
        top_k=5,
    )


def test_bare_run_creates_traces_in_the_working_directory(monkeypatch, tmp_path):
    """No --output-dir: the JSON still lands, in ./traces, made if absent."""
    _, trace_path = _run_bare(monkeypatch, tmp_path)

    assert trace_path.parent == Path("traces")
    assert (tmp_path / "traces").is_dir()
    assert len(list((tmp_path / "traces").glob("profile_*.json"))) == 1
    # Only the trace dir — the derived files are what --output-dir buys.
    assert [p.name for p in tmp_path.iterdir()] == ["traces"]


def test_bare_run_reuses_an_existing_traces_directory(monkeypatch, tmp_path):
    """An existing traces/ is where the artifact goes, not a sibling of it."""
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "profile_earlier.json").write_text("{}")

    _run_bare(monkeypatch, tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == ["traces"]
    assert len(list((tmp_path / "traces").glob("*.json"))) == 2
