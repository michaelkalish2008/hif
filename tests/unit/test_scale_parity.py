"""The control surface is the same at every scale.

`profile` (one case) and `batch` (many, model loaded once) are two scales of
one operation. A ceiling that means something different at one of them — or is
missing there — makes a corpus incomparable with the single runs it is supposed
to aggregate.

The built-in prompt suite is deliberately NOT a third command. It is a source
of workload rows (`--sample-set`), so it inherits every control `batch` has
instead of needing its own entry point that drifts from it — which is exactly
what happened to the old `hif suite`: no --config-file, no ceilings.
"""

from __future__ import annotations

import inspect
import json

import pytest
from typer.testing import CliRunner

from hif.cli_base import app
import hif.cli as cli_mod  # noqa: F401 — registers commands

runner = CliRunner()

SCALE_COMMANDS = ["profile", "batch"]

# Every scale must accept these. They are the measurement-defining controls;
# a scale missing one cannot express a condition the other can.
SHARED_CONTROLS = ["config_file", "mode", "acquisition", "lite", "variant_io"]


def _params(command_name: str) -> set[str]:
    fn = getattr(cli_mod, command_name)
    return set(inspect.signature(fn).parameters)


def _args_for(command: str, tmp_path) -> list[str]:
    (tmp_path / "w.jsonl").write_text('{"query_id":"a","text":"hello"}\n')
    return {
        "profile": ["profile", "gpt2", "hello"],
        "batch": ["batch", str(tmp_path / "w.jsonl"), "gpt2"],
    }[command]


class TestParameterParity:
    @pytest.mark.parametrize("command", SCALE_COMMANDS)
    @pytest.mark.parametrize("control", SHARED_CONTROLS)
    def test_every_scale_accepts_every_shared_control(self, command, control):
        assert control in _params(command), (
            f"`hif {command}` cannot express --{control.replace('_', '-')}, "
            f"which the other scale can"
        )

    def test_suite_is_not_a_separate_command(self):
        """It was one, and it drifted: no --config-file, no ceilings. The
        sample set is a row source now, so it cannot drift again."""
        assert not hasattr(cli_mod, "suite")
        assert "sample_set" in _params("batch")


class TestValidationParity:
    @pytest.mark.parametrize("command", SCALE_COMMANDS)
    def test_bad_acquisition_rejected_with_exit_3(self, command, tmp_path):
        result = runner.invoke(
            app, _args_for(command, tmp_path) + ["--acquisition", "bogus"]
        )
        assert result.exit_code == 3, result.output

    @pytest.mark.parametrize("command", SCALE_COMMANDS)
    def test_typo_in_config_file_rejected_at_every_scale(self, command, tmp_path):
        cfg = tmp_path / "run.toml"
        cfg.write_text('[perturbation]\ngeneratorz = ["synonym"]\n')
        result = runner.invoke(
            app, _args_for(command, tmp_path) + ["--config-file", str(cfg)]
        )
        assert result.exit_code == 3, result.output


class TestSampleSetRowSource:
    def test_export_writes_rows_without_loading_a_model(self, tmp_path):
        out = tmp_path / "suite.jsonl"
        result = runner.invoke(
            app, ["batch", "--sample-set", "all", "--export-workload", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert len(out.read_text().strip().splitlines()) == 40  # 8 regimes x 5

    def test_exported_rows_load_back_as_a_workload(self, tmp_path):
        """The export is only useful if `hif batch` accepts it verbatim."""
        from hif.batch import load_workload

        out = tmp_path / "suite.jsonl"
        runner.invoke(app, ["batch", "--sample-set", "all", "--export-workload", str(out)])
        rows = load_workload(out)
        assert len(rows) == 40
        assert all(r.query_id and r.text and r.regime for r in rows)
        assert all(r.variants is None for r in rows)
        # query_ids name trace artifacts — they must be unique.
        assert len({r.query_id for r in rows}) == len(rows)
        json.loads(out.read_text().splitlines()[0])

    def test_authored_variants_survive_a_round_trip(self, tmp_path):
        """Fork the suite, add variants, run it back — the loop the sample set
        exists to start."""
        from hif.batch import load_workload

        out = tmp_path / "suite.jsonl"
        runner.invoke(app, ["batch", "--sample-set", "ordinary_conversation",
                            "--export-workload", str(out)])
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        rows[0]["variants"] = ["a paraphrase the researcher wrote"]
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        assert load_workload(out)[0].variants == ["a paraphrase the researcher wrote"]

    def test_single_regime_selects_that_regime_only(self, tmp_path):
        from hif.batch import load_workload

        out = tmp_path / "one.jsonl"
        result = runner.invoke(
            app, ["batch", "--sample-set", "ordinary_conversation",
                  "--export-workload", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert {r.regime for r in load_workload(out)} == {"ordinary_conversation"}

    def test_unknown_selector_names_the_valid_regimes(self, tmp_path):
        result = runner.invoke(
            app, ["batch", "--sample-set", "nope",
                  "--export-workload", str(tmp_path / "x.jsonl")]
        )
        assert result.exit_code == 3
        assert "ordinary_conversation" in result.output


class TestRowSourceIsExclusive:
    def test_both_sources_rejected(self, tmp_path):
        wl = tmp_path / "w.jsonl"
        wl.write_text('{"query_id":"a","text":"hello"}\n')
        result = runner.invoke(app, ["batch", str(wl), "gpt2", "--sample-set", "all"])
        assert result.exit_code == 3, result.output

    def test_neither_source_rejected(self):
        result = runner.invoke(app, ["batch"])
        assert result.exit_code == 3, result.output

    def test_sample_set_shifts_the_lone_positional_to_the_model(self):
        """`hif batch --sample-set all gpt2` — click binds `gpt2` to the
        workload slot; it must land on the model instead."""
        result = runner.invoke(
            app, ["batch", "--sample-set", "all", "definitely-not-a-real-model"]
        )
        # Reaches model resolution, rather than the two arity errors.
        assert "Nothing to profile" not in result.output
        assert "Missing argument" not in result.output

    def test_export_needs_no_model(self, tmp_path):
        out = tmp_path / "s.jsonl"
        result = runner.invoke(
            app, ["batch", "--sample-set", "all", "--export-workload", str(out)]
        )
        assert result.exit_code == 0, result.output
