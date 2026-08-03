"""Contract tests for `hif batch` (the workload runner).

Follows the test_cli_logging.py style: stub the heavy internals
(SessionEngine) on hif.batch and drive the command through typer's
CliRunner. The contract under test:

- stdout is a pure JSONL stream: one compact record per row, query_ids
  preserved; progress/log lines go to stderr only.
- every record on the stream — successes and row errors alike — carries the
  same schema_version.
- a row failure emits an error record and the stream continues.
- malformed workload / bad path / mixed image rows without a VLM backend
  exit 3 before any engine is created.
- --limit truncates; --output-dir mirrors the stream to records.jsonl.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import hif.batch as batch_mod
from hif.cli import app
from hif.profile.signals import RECORD_SCHEMA_VERSION

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEngine:
    """Stands in for SessionEngine: profile_one returns a token; record_for
    builds a canned record-shaped dict echoing prompt/regime/extras. It stamps
    the real RECORD_SCHEMA_VERSION so a drift between the success path and
    batch's own error path shows up as a test failure."""

    created = 0

    def __init__(self, config):
        self.config = config

    @classmethod
    def create(cls, config, **kwargs):
        cls.created += 1
        cls.last_create_kwargs = kwargs
        return cls(config)

    def profile_one(self, prompt, *, regime, seed, authored_variants=None,
                    variant_output_sink=None):
        if isinstance(prompt, str) and "BOOM" in prompt:
            raise RuntimeError("pipeline exploded")
        return {"profile_for": prompt, "authored_variants": authored_variants}

    def record_for(self, profile, *, prompt, regime, seed, latency=None,
                   trace_path=None, extras=None, include_units=False):
        rec = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "prompt": prompt,
            "regime": regime,
            "seed": seed,
            "latency": latency,
            "trace_path": trace_path,
        }
        if extras:
            rec.update(extras)
        return rec


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch):
    FakeEngine.created = 0
    monkeypatch.setattr(batch_mod, "SessionEngine", FakeEngine)


def _write_workload(tmp_path, rows, name="workload.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _stdout_records(result):
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


ROWS = [
    {"query_id": "wt_01", "text": "first prompt"},
    {"query_id": "wt_02", "text": "second prompt"},
    {"query_id": "wt_03", "text": "third prompt", "regime": "custom_regime"},
]


# ---------------------------------------------------------------------------
# happy path: JSONL stream on stdout
# ---------------------------------------------------------------------------


def test_batch_streams_one_json_record_per_row(tmp_path):
    wl = _write_workload(tmp_path, ROWS)
    result = runner.invoke(app, ["batch", str(wl), "m"])
    assert result.exit_code == 0, result.output
    records = _stdout_records(result)
    assert len(records) == 3
    assert [r["query_id"] for r in records] == ["wt_01", "wt_02", "wt_03"]
    for r in records:
        assert r["schema_version"] == RECORD_SCHEMA_VERSION
        assert "pipeline" in r["latency"]
    # per-row regime override beats the default
    assert records[0]["regime"] == "batch"
    assert records[2]["regime"] == "custom_regime"
    # model loaded exactly once for the whole workload
    assert FakeEngine.created == 1


def test_batch_stdout_lines_are_compact(tmp_path):
    wl = _write_workload(tmp_path, ROWS[:1])
    result = runner.invoke(app, ["batch", str(wl), "m"])
    assert result.exit_code == 0, result.output
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == 1  # compact: no indented multi-line JSON
    json.loads(lines[0])


def test_stdout_is_json_only_and_progress_goes_to_stderr(tmp_path):
    """The stream is machine-consumable: EVERY stdout line parses as JSON, and
    the human-readable progress lines appear on stderr instead."""
    wl = _write_workload(tmp_path, ROWS)
    result = runner.invoke(app, ["batch", str(wl), "m"])
    assert result.exit_code == 0, result.output
    for line in result.stdout.splitlines():
        if line.strip():
            json.loads(line)  # raises if any chatter leaked onto stdout
    assert "wt_01 ok" in result.stderr
    assert "batch complete: 3 ok, 0 failed" in result.stderr


def test_records_carry_no_normalized_or_levels_block(tmp_path):
    """Natural units only: the removed normalisation and level blocks must not
    reappear on the record stream."""
    wl = _write_workload(tmp_path, ROWS)
    result = runner.invoke(app, ["batch", str(wl), "m"])
    assert result.exit_code == 0, result.output
    for r in _stdout_records(result):
        assert "normalized" not in r
        assert "levels" not in r
        assert "findings_levels" not in r


# ---------------------------------------------------------------------------
# row errors never kill the stream
# ---------------------------------------------------------------------------


def test_row_error_emits_error_record_and_continues(tmp_path):
    wl = _write_workload(tmp_path, [
        {"query_id": "ok_1", "text": "fine"},
        {"query_id": "bad", "text": "BOOM"},
        {"query_id": "ok_2", "text": "also fine"},
    ])
    result = runner.invoke(app, ["batch", str(wl), "m"])
    assert result.exit_code == 0, result.output  # some rows succeeded
    records = _stdout_records(result)
    assert [r["query_id"] for r in records] == ["ok_1", "bad", "ok_2"]
    assert "pipeline exploded" in records[1]["error"]
    assert "error" not in records[0] and "error" not in records[2]
    # One stream, one schema: the error record is stamped with the SAME
    # version as the successes around it.
    assert {r["schema_version"] for r in records} == {RECORD_SCHEMA_VERSION}


def test_all_rows_failing_exits_1(tmp_path):
    wl = _write_workload(tmp_path, [
        {"query_id": "b1", "text": "BOOM one"},
        {"query_id": "b2", "text": "BOOM two"},
    ])
    result = runner.invoke(app, ["batch", str(wl), "m"])
    assert result.exit_code == 1
    records = _stdout_records(result)
    assert len(records) == 2
    assert all("error" in r for r in records)


# ---------------------------------------------------------------------------
# workload validation (exit 3 before any engine)
# ---------------------------------------------------------------------------


def test_missing_workload_path_exits_3(tmp_path):
    result = runner.invoke(app, ["batch", str(tmp_path / "nope.jsonl"), "m"])
    assert result.exit_code == 3
    assert FakeEngine.created == 0


def test_malformed_jsonl_exits_3(tmp_path):
    path = tmp_path / "workload.jsonl"
    path.write_text('{"query_id": "a", "text": "fine"}\nnot json at all\n')
    result = runner.invoke(app, ["batch", str(path), "m"])
    assert result.exit_code == 3
    assert result.stdout == ""
    assert FakeEngine.created == 0


def test_missing_required_keys_exits_3(tmp_path):
    wl = _write_workload(tmp_path, [{"query_id": "a"}])  # no "text"
    result = runner.invoke(app, ["batch", str(wl), "m"])
    assert result.exit_code == 3
    assert FakeEngine.created == 0


def test_image_rows_without_vlm_backend_exit_3(tmp_path):
    (tmp_path / "form.png").write_bytes(b"png-bytes")  # exists; content not checked here
    wl = _write_workload(tmp_path, [
        {"query_id": "t1", "text": "plain"},
        {"query_id": "m1", "text": "what is this?", "image": "form.png"},
    ])
    result = runner.invoke(app, ["batch", str(wl), "m"])
    assert result.exit_code == 3
    assert "hf-vlm" in result.stderr
    assert FakeEngine.created == 0  # rejected before loading models


# ---------------------------------------------------------------------------
# --limit
# ---------------------------------------------------------------------------


def test_limit_respected(tmp_path):
    wl = _write_workload(tmp_path, ROWS)
    result = runner.invoke(app, ["batch", str(wl), "m", "--limit", "2"])
    assert result.exit_code == 0, result.output
    records = _stdout_records(result)
    assert [r["query_id"] for r in records] == ["wt_01", "wt_02"]


# ---------------------------------------------------------------------------
# --output-dir mirrors the stream
# ---------------------------------------------------------------------------


def test_output_dir_writes_records_jsonl_mirroring_stdout(tmp_path):
    wl = _write_workload(tmp_path, ROWS)
    out = tmp_path / "out"
    result = runner.invoke(app, ["batch", str(wl), "m", "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    mirror = (out / "records.jsonl").read_text()
    stdout_lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert mirror.splitlines() == stdout_lines


# ---------------------------------------------------------------------------
# --backend validation & colon auto-route (before gate / engine creation)
# ---------------------------------------------------------------------------


def test_unknown_backend_exits_3_before_engine(tmp_path):
    wl = _write_workload(tmp_path, ROWS)
    result = runner.invoke(app, ["batch", str(wl), "m", "--backend", "bogus"])
    assert result.exit_code == 3
    assert result.stdout == ""
    assert "bogus" in result.stderr
    assert "ollama" in result.stderr  # lists the valid backends
    assert FakeEngine.created == 0


def test_colon_model_name_autoroutes_hf_to_ollama(tmp_path):
    wl = _write_workload(tmp_path, ROWS[:1])
    result = runner.invoke(app, ["batch", str(wl), "gemma3:4b"])
    assert result.exit_code == 0, result.output
    assert "Ollama model tag" in result.stderr
    assert len(_stdout_records(result)) == 1


# ---------------------------------------------------------------------------
# --config-file: seed precedence & unknown-key rejection
# ---------------------------------------------------------------------------


def test_config_file_seed_wins_over_cli_default(tmp_path):
    wl = _write_workload(tmp_path, ROWS[:1])
    cfg = tmp_path / "run.toml"
    cfg.write_text("[generation]\nseed = 7\n")
    result = runner.invoke(app, ["batch", str(wl), "m", "--config-file", str(cfg)])
    assert result.exit_code == 0, result.output
    assert _stdout_records(result)[0]["seed"] == 7


def test_explicit_seed_beats_config_file(tmp_path):
    wl = _write_workload(tmp_path, ROWS[:1])
    cfg = tmp_path / "run.toml"
    cfg.write_text("[generation]\nseed = 7\n")
    result = runner.invoke(
        app, ["batch", str(wl), "m", "--config-file", str(cfg), "--seed", "99"]
    )
    assert result.exit_code == 0, result.output
    assert _stdout_records(result)[0]["seed"] == 99


def test_config_file_unknown_table_exits_3(tmp_path):
    wl = _write_workload(tmp_path, ROWS[:1])
    cfg = tmp_path / "run.toml"
    cfg.write_text("[perturbaton]\nn_variants = 5\n")  # typo'd table
    result = runner.invoke(app, ["batch", str(wl), "m", "--config-file", str(cfg)])
    assert result.exit_code == 3
    assert "perturbaton" in result.stderr
    assert FakeEngine.created == 0


# ---------------------------------------------------------------------------
# image rows: missing file fails fast; unreadable file mid-run isolates the row
# ---------------------------------------------------------------------------


def test_missing_image_file_exits_3_before_engine(tmp_path):
    wl = _write_workload(tmp_path, [
        {"query_id": "m1", "text": "what is this?", "image": "gone.png"},
    ])
    result = runner.invoke(app, ["batch", str(wl), "m", "--backend", "hf-vlm"])
    assert result.exit_code == 3
    assert "gone.png" in result.stderr
    assert FakeEngine.created == 0


def test_unreadable_image_mid_run_emits_error_record(tmp_path):
    # Exists at validation time but is not a decodable image — the row must
    # fail into an {"error": ...} record, not abort the stream.
    (tmp_path / "junk.png").write_bytes(b"not really a png")
    wl = _write_workload(tmp_path, [
        {"query_id": "ok_1", "text": "plain row"},
        {"query_id": "bad_img", "text": "describe", "image": "junk.png"},
    ])
    result = runner.invoke(app, ["batch", str(wl), "m", "--backend", "hf-vlm"])
    assert result.exit_code == 0, result.output
    records = _stdout_records(result)
    by_id = {r["query_id"]: r for r in records}
    assert "error" not in by_id["ok_1"]
    assert by_id["bad_img"]["error"]  # non-empty, meaningful message
    assert "junk.png" in by_id["bad_img"]["error"]


# ---------------------------------------------------------------------------
# empty effective workload
# ---------------------------------------------------------------------------


def test_all_blank_workload_exits_3(tmp_path):
    path = tmp_path / "workload.jsonl"
    path.write_text("\n\n   \n")
    result = runner.invoke(app, ["batch", str(path), "m"])
    assert result.exit_code == 3
    assert FakeEngine.created == 0


def test_limit_zero_exits_3(tmp_path):
    wl = _write_workload(tmp_path, ROWS)
    result = runner.invoke(app, ["batch", str(wl), "m", "--limit", "0"])
    assert result.exit_code == 3
    assert FakeEngine.created == 0
