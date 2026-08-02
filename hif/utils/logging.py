"""Structured logging helpers using Rich for human-readable console output.

Output contract (CLI) — two modes only:
- default: results only. No internal chatter — no encoder/model-loading lines,
  no fallback notices, no top_k adjustments, no HTTP request lines, no
  per-stage INFO. Genuine WARNINGs (unexpected conditions) and errors still
  show.
- --verbose: everything — the IO/stats/latency tables (rendered by the CLI)
  plus the full internal chatter: hif loggers at DEBUG, root at INFO,
  third-party HTTP/ML libraries (httpx, transformers, sentence_transformers,
  ...) at DEBUG, and HuggingFace progress bars / advisory warnings
  re-enabled.
"""

import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler

# Always stderr: stdout is reserved for machine-readable data so every
# data-producing command is pipe-safe (`hif ... | jq .`).
console = Console(stderr=True)

# Third-party loggers that are noisy at INFO and propagate to root.
_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "urllib3",
    "sentence_transformers",
    "transformers",
    "ollama",
    "openai",
    "anthropic",
)

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    # hif's own loggers: WARNING by default — INFO pipeline chatter
    # (model/tokenizer loading, stage progress) is internal detail and must not
    # print on a plain run. --verbose restores it via configure_logging.
    logging.getLogger("hif").setLevel(logging.WARNING)
    _set_third_party(logging.WARNING)
    _quiet_hf_noise()
    _configured = True


def _set_third_party(level: int) -> None:
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(level)


def _quiet_hf_noise() -> None:
    """Suppress HuggingFace progress bars and advisory warnings (stderr noise
    like tqdm 'Loading weights' bars and '`torch_dtype` is deprecated')."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    _apply_transformers_verbosity(verbose=False)


def _restore_hf_noise() -> None:
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    os.environ.pop("TRANSFORMERS_NO_ADVISORY_WARNINGS", None)
    _apply_transformers_verbosity(verbose=True)


def _apply_transformers_verbosity(verbose: bool) -> None:
    """If transformers is already imported, env vars won't be re-read — apply
    the setting through its API directly."""
    tf_logging = sys.modules.get("transformers.utils.logging")
    if tf_logging is None and "transformers" in sys.modules:
        tf_logging = getattr(sys.modules["transformers"], "utils", None)
        tf_logging = getattr(tf_logging, "logging", None)
    if tf_logging is not None:
        try:
            if verbose:
                tf_logging.set_verbosity_info()
                tf_logging.enable_progress_bar()
            else:
                tf_logging.set_verbosity_error()
                tf_logging.disable_progress_bar()
        except Exception:  # pragma: no cover - best-effort quieting
            pass


def configure_logging(verbose: bool = False) -> None:
    """Explicit logging setup, called from the CLI entrypoint and re-called by
    commands that accept --verbose.

    verbose=False (default): root at WARNING, hif loggers at WARNING
    (results only — internal INFO chatter hidden), third-party libraries
    (httpx, huggingface_hub, transformers, ollama, etc.) silenced
    to WARNING, HF progress bars/advisory warnings suppressed.

    verbose=True: full verbosity — hif loggers at DEBUG, root at INFO,
    third-party loggers at DEBUG, HF progress bars restored.
    """
    _configure_root()
    # HF verbosity first: transformers' set_verbosity_* mutates the
    # "transformers" logger level, so _set_third_party must run after it.
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("hif").setLevel(logging.DEBUG)
        _restore_hf_noise()
        _set_third_party(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("hif").setLevel(logging.WARNING)
        _quiet_hf_noise()
        _set_third_party(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
