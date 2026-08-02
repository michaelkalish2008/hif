"""Backend dispatch: construct a Model from a ModelConfig.

Single home for the backend→class mapping so the CLI, the SessionEngine, and
any future daemon construct models identically. Raises ValueError on an
unknown backend; presentation-layer concerns (the Ollama colon auto-route
warning, typer exits) stay in hif/cli.py.
"""

from __future__ import annotations

from hif.config import ModelConfig

KNOWN_BACKENDS = (
    "hf", "tlens", "ollama", "openai", "anthropic", "gemini",
    "hf-vlm", "openai-vlm",
)


def load_model(config: ModelConfig):
    """Instantiate the backend model for `config`. Imports lazily so optional
    provider deps are only required for the backend actually used."""
    backend = config.backend
    if backend == "hf":
        from hif.models.hf import HFModel
        return HFModel(config)
    elif backend == "tlens":
        from hif.models.tlens import TLensModel
        return TLensModel(config)
    elif backend == "ollama":
        from hif.models.ollama import OllamaModel
        return OllamaModel(config)
    elif backend == "openai":
        from hif.models.openai_model import OpenAIModel
        return OpenAIModel(config)
    elif backend == "anthropic":
        from hif.models.anthropic_model import AnthropicModel
        return AnthropicModel(config)
    elif backend == "gemini":
        from hif.models.gemini_model import GeminiModel
        return GeminiModel(config)
    elif backend == "hf-vlm":
        from hif.models.hf_vlm import HFVLMModel
        return HFVLMModel(config)
    elif backend == "openai-vlm":
        from hif.models.openai_vlm import OpenAIVLMModel
        return OpenAIVLMModel(config)
    raise ValueError(
        f"Unknown backend: {backend!r}. Use one of: {', '.join(KNOWN_BACKENDS)}."
    )
