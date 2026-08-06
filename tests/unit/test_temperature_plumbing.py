"""[generation] temperature must actually reach the model.

The sampling adapters consume ModelConfig.temperature (hif/models/hf.py,
openai_model.py) — GenerationConfig.temperature is read by nothing
at inference time. `_make_run_config` therefore mirrors a TOML-set
[generation] temperature onto cfg.model.temperature, with an explicitly-set
[model] temperature winning, and None (backend default) when neither is set.
GenerationConfig.temperature defaults to 1.0, so the mirror must key off
model_fields_set, never the default value.
"""

from __future__ import annotations

import pytest

from hif.cli import _load_config_file, _make_run_config


def _config_from_toml(tmp_path, toml_text: str, model="gpt2", backend="hf"):
    path = tmp_path / "config.toml"
    path.write_text(toml_text)
    base = _load_config_file(path)
    return _make_run_config(model, backend, 64, 50, 42, None, base=base)


def test_generation_temperature_mirrored_onto_model(tmp_path):
    cfg = _config_from_toml(tmp_path, "[generation]\ntemperature = 0.7\n")
    assert cfg.generation.temperature == pytest.approx(0.7)
    assert cfg.model.temperature == pytest.approx(0.7)


def test_explicit_model_temperature_wins_over_generation(tmp_path):
    cfg = _config_from_toml(
        tmp_path,
        "[model]\ntemperature = 0.3\n\n[generation]\ntemperature = 0.7\n")
    assert cfg.model.temperature == pytest.approx(0.3)
    assert cfg.generation.temperature == pytest.approx(0.7)


def test_model_temperature_survives_cli_model_rebuild(tmp_path):
    # _make_run_config rebuilds cfg.model from the CLI name/backend — an
    # explicit [model] temperature must survive that rebuild.
    cfg = _config_from_toml(tmp_path, "[model]\ntemperature = 0.25\n",
                            model="other-model", backend="ollama")
    assert cfg.model.name == "other-model"
    assert cfg.model.backend == "ollama"
    assert cfg.model.temperature == pytest.approx(0.25)


def test_no_temperature_in_toml_leaves_backend_default(tmp_path):
    # generation.temperature's 1.0 DEFAULT must not be mirrored: for API
    # backends None means "backend default" (0 for OpenAI) and silently
    # pinning 1.0 would change every config-file run's sampling.
    cfg = _config_from_toml(tmp_path, "[generation]\nmax_new_tokens = 8\n")
    assert cfg.generation.temperature == pytest.approx(1.0)
    assert cfg.model.temperature is None


def test_no_config_file_leaves_backend_default():
    cfg = _make_run_config("gpt2", "hf", 64, 50, 42, None)
    assert cfg.model.temperature is None
