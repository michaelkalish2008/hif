"""What `--output-dir` puts on disk.

The JSON used to be withheld unless `--trace`, on the reasoning that per-step
top-K with token identity should not reach disk by default. That reasoning
belonged to the hosted deployment, where the text being profiled was not the
operator's; it is archived, and a researcher profiling their own prompts on
their own machine is not disclosing anything to themselves. So the artifact
the report is an excerpt of — and the only form `hif render` reads back — now
ships with the report.
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

    def write_trace(self, profile, *, prompt, seed, trace_dir):
        path = Path(trace_dir) / "profile_trace.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path


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


def test_output_dir_writes_report_and_json(monkeypatch, tmp_path):
    _run_with(monkeypatch, tmp_path)

    reports = list(tmp_path.glob("*_technical.md"))
    jsons = [p for p in tmp_path.glob("profile_*.json")]
    assert len(reports) == 1, sorted(p.name for p in tmp_path.iterdir())
    assert len(jsons) == 1, sorted(p.name for p in tmp_path.iterdir())

    # The JSON is the whole profile, not a summary of it: it round-trips.
    data = json.loads(jsons[0].read_text())
    assert data["model"]["name"] == "mock-model"
    assert data["output_side"]["steps"]


def test_json_not_duplicated_when_trace_already_wrote_it(monkeypatch, tmp_path):
    """--trace writes the same bytes to the trace dir. One artifact, not two."""
    _, trace_path = _run_with(monkeypatch, tmp_path, trace=True)

    assert trace_path is not None and trace_path.exists()
    assert list(tmp_path.glob("profile_*.json")) == []
    assert len(list(tmp_path.glob("*_technical.md"))) == 1


def test_lite_and_full_runs_do_not_overwrite_each_others_artifacts(
    monkeypatch, tmp_path
):
    """The artifact names are hash-addressed, and the two runs are not the
    same run: a lite run reports two measurements where a full run reports
    six. While the hash covered only (model, prompt, seed), the second
    invocation silently replaced the first one's report and JSON in place.
    """
    _run_with(monkeypatch, tmp_path)
    full = sorted(p.name for p in tmp_path.iterdir())

    _run_with(monkeypatch, tmp_path, lite=True)
    both = sorted(p.name for p in tmp_path.iterdir())

    assert len(full) == 2, full
    assert len(both) == 4, both
    assert set(full) < set(both)


def test_no_output_dir_writes_nothing(monkeypatch, tmp_path):
    """A run that was not asked for files leaves none."""
    _patch(monkeypatch)
    _run._run_single_profile(
        model_name="mock-model",
        prompt="hello world",
        regime="factual",
        backend="hf",
        seed=42,
        output_dir=None,
        max_new_tokens=4,
        top_k=5,
    )
    assert list(tmp_path.iterdir()) == []
