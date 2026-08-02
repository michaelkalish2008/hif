"""--surrogate / --surrogate-model CLI wiring.

Regression for a footgun: --surrogate-model is a value option and --surrogate
is a separate boolean flag. Passing only the former used to silently do
nothing (input-side signals stayed zeroed) because the pipeline only loaded a
surrogate when the boolean was explicitly set. --surrogate-model must imply
--surrogate.
"""

from typer.testing import CliRunner

import hif.cli as cli
from hif.cli import app
from tests.unit.profile_helpers import _make_profile

runner = CliRunner()


def _patch_pipeline(monkeypatch, captured, profile=None):
    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_load_embedder", lambda *a, **k: object())

    def fake_run_single_profile(*args, **kwargs):
        captured.update(kwargs)
        return (profile if profile is not None else _make_profile()), None

    monkeypatch.setattr(cli, "_run_single_profile", fake_run_single_profile)


def test_surrogate_model_alone_implies_surrogate(monkeypatch, tmp_path):
    captured = {}
    _patch_pipeline(monkeypatch, captured)

    result = runner.invoke(app, [
        "profile", "claude-haiku-4-5-20251001", "hi",
        "--backend", "anthropic",
        "--surrogate-model", "gpt2",
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert captured["surrogate"] is True
    assert captured["surrogate_model_id"] == "gpt2"


def test_surrogate_flag_alone_uses_default_model(monkeypatch, tmp_path):
    captured = {}
    _patch_pipeline(monkeypatch, captured)

    result = runner.invoke(app, [
        "profile", "claude-haiku-4-5-20251001", "hi",
        "--backend", "anthropic",
        "--surrogate",
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert captured["surrogate"] is True
    assert captured["surrogate_model_id"] == "unsloth/Llama-3.2-1B"


def test_no_surrogate_flags_defaults_off(monkeypatch, tmp_path):
    captured = {}
    _patch_pipeline(monkeypatch, captured)

    result = runner.invoke(app, [
        "profile", "claude-haiku-4-5-20251001", "hi",
        "--backend", "anthropic",
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert captured["surrogate"] is False
    assert captured["surrogate_model_id"] == "unsloth/Llama-3.2-1B"


def test_measurement_table_flags_surrogate_derived_measurements(monkeypatch, tmp_path):
    profile = _make_profile()
    profile.findings.surrogate_model_name = "unsloth/Llama-3.2-1B"

    captured = {}
    _patch_pipeline(monkeypatch, captured, profile=profile)

    result = runner.invoke(app, [
        "profile", "claude-haiku-4-5-20251001", "hi",
        "--backend", "anthropic",
        "--surrogate",
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    # Exactly the input-side measurements are attributed to the surrogate.
    assert "Input entropy shift (bits) *" in flat
    assert "Input/output correlation (r) *" in flat
    assert "Prompt surprisal excess (bits) *" in flat
    # Output-side measurements are the target model's own — never starred here.
    assert "Output entropy (bits) *" not in flat
    assert "computed via surrogate model 'unsloth/Llama-3.2-1B'" in flat


def test_measurement_table_no_asterisk_without_surrogate(monkeypatch, tmp_path):
    captured = {}
    _patch_pipeline(monkeypatch, captured)  # default profile: surrogate_model_name=None

    result = runner.invoke(app, [
        "profile", "gpt2", "hi",
        "--backend", "hf",
        "--output-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert "*" not in result.output
