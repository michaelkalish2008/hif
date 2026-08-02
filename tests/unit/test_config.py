"""Smoke tests for config model instantiation and defaults."""

from hif.config import RunConfig


def test_run_config_defaults() -> None:
    cfg = RunConfig()
    assert cfg.model.name == "gpt2"
    assert cfg.model.backend == "hf"
    assert cfg.generation.seed == 42
    assert cfg.perturbation.n_variants == 2
    assert "synonym" in cfg.perturbation.generators
