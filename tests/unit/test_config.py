"""Smoke tests for config model instantiation and defaults."""

from hif.config import RunConfig


def test_run_config_defaults() -> None:
    cfg = RunConfig()
    # A -Base checkpoint on purpose: hif continues the prompt as a raw causal
    # LM and applies no chat template, so an instruct-tuned default would
    # answer a different question than the one the caller typed.
    assert cfg.model.name == "Qwen/Qwen3-0.6B-Base"
    assert cfg.model.backend == "hf"
    assert cfg.generation.seed == 42
    assert cfg.perturbation.n_variants == 2
    assert "synonym" in cfg.perturbation.generators
