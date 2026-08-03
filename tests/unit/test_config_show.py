"""`hif config show` / `hif config init`, strict TOML validation, record-v6.

The three legs of config trust:
  1. a mistyped key anywhere in the file is rejected (never silently defaulted),
  2. the resolved config is printable without a run, through the SAME
     resolution path the run uses,
  3. the record carries the resolved config, with secrets redacted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hif.cli_base import app
from hif.config import RunConfig, public_config_dict

runner = CliRunner()


def _cli(*args: str):
    import hif.cli  # noqa: F401 — registers commands on the shared app

    return runner.invoke(app, list(args))


class TestStrictTomlValidation:
    def test_unknown_table_exits_3(self, tmp_path: Path):
        f = tmp_path / "run.toml"
        f.write_text("[perturbaton]\nn_variants = 4\n")
        result = _cli("config", "show", "--config-file", str(f))
        assert result.exit_code == 3

    def test_unknown_inner_key_exits_3_and_names_it(self, tmp_path: Path):
        """The silent-default failure this guard exists for: before it, this
        file validated cleanly and the run measured with the default
        generators."""
        f = tmp_path / "run.toml"
        f.write_text('[perturbation]\ngeneratorz = ["substitution"]\n')
        result = _cli("config", "show", "--config-file", str(f))
        assert result.exit_code == 3
        assert "generatorz" in result.output

    def test_alias_table_still_loads(self, tmp_path: Path):
        """[hallucination] is the pre-rename alias for [exposure]; archived
        configs must keep loading."""
        f = tmp_path / "run.toml"
        f.write_text("[hallucination]\nmin_prob = 0.05\n")
        result = _cli("config", "show", "--config-file", str(f), "--diff")
        assert result.exit_code == 0
        assert "min_prob = 0.05" in result.output

    def test_extra_body_inner_keys_are_exempt(self, tmp_path: Path):
        """extra_body's keys are the provider's vocabulary, not ours."""
        f = tmp_path / "run.toml"
        f.write_text('[model]\nextra_body = { thinking = { type = "disabled" } }\n')
        result = _cli("config", "show", "--config-file", str(f))
        assert result.exit_code == 0


class TestConfigShow:
    def test_diff_shows_only_departures(self, tmp_path: Path):
        f = tmp_path / "run.toml"
        f.write_text("[perturbation]\nn_variants = 4\n")
        result = _cli("config", "show", "--config-file", str(f), "--diff")
        assert result.exit_code == 0
        assert "n_variants = 4" in result.output
        # A key nobody touched must not appear in a diff.
        assert "rollout_steps" not in result.output

    def test_acquisition_cap_visible_before_any_run(self, tmp_path: Path):
        f = tmp_path / "run.toml"
        f.write_text("[trajectory]\nn_branches = 8\n")
        result = _cli(
            "config", "show", "--config-file", str(f),
            "--acquisition", "observational", "--diff",
        )
        assert result.exit_code == 0
        # The ceiling wins over the file, and the researcher can see that
        # WITHOUT running: branches forced to zero, not eight.
        assert "n_branches = 0" in result.output

    def test_output_round_trips_as_config_file(self, tmp_path: Path):
        f = tmp_path / "run.toml"
        f.write_text('[perturbation]\ngenerators = ["synonym"]\nn_variants = 3\n')
        shown = _cli("config", "show", "--config-file", str(f))
        assert shown.exit_code == 0
        body = "\n".join(
            line for line in shown.output.splitlines() if not line.startswith("#")
        )
        round_trip = tmp_path / "round.toml"
        round_trip.write_text(body)
        again = _cli("config", "show", "--config-file", str(round_trip), "--diff")
        assert again.exit_code == 0
        assert "n_variants = 3" in again.output

    def test_api_key_redacted_never_printed(self, tmp_path: Path):
        f = tmp_path / "run.toml"
        f.write_text('[model]\napi_key = "sk-SECRET-VALUE"\n')
        result = _cli("config", "show", "--config-file", str(f))
        assert result.exit_code == 0
        assert "sk-SECRET-VALUE" not in result.output
        assert "<redacted>" in result.output


class TestConfigInit:
    def test_template_is_neutral(self, tmp_path: Path):
        """Loading the untouched template must change nothing — including the
        temperature mirror, which fires on explicitly-set defaults."""
        out = tmp_path / "run.toml"
        assert _cli("config", "init", "-o", str(out)).exit_code == 0
        result = _cli("config", "show", "--config-file", str(out), "--diff")
        assert result.exit_code == 0
        assert "no departures from defaults" in result.output

    def test_refuses_to_overwrite_without_force(self, tmp_path: Path):
        out = tmp_path / "run.toml"
        out.write_text("[generation]\nseed = 7\n")
        assert _cli("config", "init", "-o", str(out)).exit_code == 3
        assert "seed = 7" in out.read_text()


class TestPublicConfigDict:
    def test_secrets_redacted_not_omitted(self):
        config = RunConfig()
        config.model.api_key = "sk-live-abc"
        config.perturbation.llm_api_key = "sk-llm-xyz"
        data = public_config_dict(config)
        assert data["model"]["api_key"] == "<redacted>"
        assert data["perturbation"]["llm_api_key"] == "<redacted>"

    def test_unset_secret_stays_none(self):
        """None vs "<redacted>" is the difference between "no key" and
        "authenticated" — provenance the record should keep."""
        data = public_config_dict(RunConfig())
        assert data["model"]["api_key"] is None

    def test_json_serialisable(self):
        import json

        json.dumps(public_config_dict(RunConfig()))
