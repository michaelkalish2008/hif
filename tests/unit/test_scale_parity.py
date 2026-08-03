"""The control surface is the same at every scale.

`profile`, `batch`, and `suite` are three scales of one operation. A ceiling
that means something different at one of them — or is missing there — makes a
corpus incomparable with the single runs it is supposed to aggregate. These
tests hold the three together at the CLI boundary.
"""

from __future__ import annotations

import inspect

import pytest
from typer.testing import CliRunner

from hif.cli_base import app
import hif.cli as cli_mod  # noqa: F401 — registers commands

runner = CliRunner()

SCALE_COMMANDS = ["profile", "batch", "suite"]

# Every scale must accept these. They are the measurement-defining controls;
# a scale missing one cannot express a condition the others can.
SHARED_CONTROLS = ["config_file", "mode", "acquisition", "lite"]


def _params(command_name: str) -> set[str]:
    fn = getattr(cli_mod, command_name)
    return set(inspect.signature(fn).parameters)


class TestParameterParity:
    @pytest.mark.parametrize("command", SCALE_COMMANDS)
    @pytest.mark.parametrize("control", SHARED_CONTROLS)
    def test_every_scale_accepts_every_shared_control(self, command, control):
        assert control in _params(command), (
            f"`hif {command}` cannot express --{control.replace('_', '-')}, "
            f"which the other scales can"
        )

    def test_variant_io_available_wherever_perturbation_can_be_elicited(self):
        """suite is exempt: it writes reports, not a record stream."""
        for command in ("profile", "batch"):
            assert "variant_io" in _params(command)


class TestValidationParity:
    @pytest.mark.parametrize("command", SCALE_COMMANDS)
    def test_bad_acquisition_rejected_with_exit_3(self, command, tmp_path):
        args = {
            "profile": ["profile", "gpt2", "hello"],
            "batch": ["batch", str(tmp_path / "w.jsonl"), "gpt2"],
            "suite": ["suite", "gpt2"],
        }[command]
        (tmp_path / "w.jsonl").write_text('{"query_id":"a","text":"hello"}\n')
        result = runner.invoke(app, args + ["--acquisition", "bogus"])
        assert result.exit_code == 3, result.output

    @pytest.mark.parametrize("command", SCALE_COMMANDS)
    def test_typo_in_config_file_rejected_at_every_scale(self, command, tmp_path):
        cfg = tmp_path / "run.toml"
        cfg.write_text('[perturbation]\ngeneratorz = ["synonym"]\n')
        (tmp_path / "w.jsonl").write_text('{"query_id":"a","text":"hello"}\n')
        args = {
            "profile": ["profile", "gpt2", "hello"],
            "batch": ["batch", str(tmp_path / "w.jsonl"), "gpt2"],
            "suite": ["suite", "gpt2"],
        }[command]
        result = runner.invoke(app, args + ["--config-file", str(cfg)])
        assert result.exit_code == 3, result.output


class TestSuiteExport:
    def test_export_writes_workload_rows_without_loading_a_model(self, tmp_path):
        out = tmp_path / "suite.jsonl"
        result = runner.invoke(app, ["suite", "gpt2", "--export-workload", str(out)])
        assert result.exit_code == 0, result.output
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 40  # 8 regimes x 5 prompts

    def test_exported_rows_load_as_a_workload(self, tmp_path):
        """The export is only useful if `hif batch` accepts it verbatim."""
        import json

        from hif.batch import load_workload

        out = tmp_path / "suite.jsonl"
        runner.invoke(app, ["suite", "gpt2", "--export-workload", str(out)])
        rows = load_workload(out)
        assert len(rows) == 40
        assert all(r.query_id and r.text and r.regime for r in rows)
        assert all(r.variants is None for r in rows)
        # query_ids must be unique — they name trace artifacts.
        assert len({r.query_id for r in rows}) == len(rows)
        json.loads(out.read_text().splitlines()[0])

    def test_single_regime_export_is_that_regime_only(self, tmp_path):
        from hif.batch import load_workload

        out = tmp_path / "one.jsonl"
        result = runner.invoke(
            app, ["suite", "gpt2", "--regime", "ordinary_conversation",
                  "--export-workload", str(out)]
        )
        assert result.exit_code == 0, result.output
        rows = load_workload(out)
        assert {r.regime for r in rows} == {"ordinary_conversation"}
