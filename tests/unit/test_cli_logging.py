"""Output-contract tests for CLI logging.

Two modes only:
- default: clean results — no encoder/fallback/top_k/HTTP/loading chatter.
- --verbose: results + stats/latency tables AND full internal chatter
  (hif INFO/DEBUG, third-party HTTP logs, effective-config notes).
- --json: stdout is pure JSON regardless of chatter.
"""

from __future__ import annotations

import json
import logging
import re

import pytest
from typer.testing import CliRunner

import hif.cli as cli
from hif.cli import app
from hif.utils import logging as hif_logging
from hif.utils.logging import configure_logging
from tests.unit.profile_helpers import _make_profile

runner = CliRunner()

# `top_k=` (not bare `top_k`) targets the clamp-log chatter
# ("Requested top_k=50 exceeds…") without false-matching the legitimate
# authenticated ceiling label "(log2 top_k)", which is labeled output, not chatter.
NOISE_RE = re.compile(r"encoder|fallback|top_k=|HTTP|Loading", re.IGNORECASE)


def _emit_chatter() -> None:
    """Emit the exact classes of log lines the founder saw leak."""
    logging.getLogger("hif.models.hf").info("Loading tokenizer: Qwen/Qwen3-1.7B")
    logging.getLogger("hif.models.hf").info("Loading model: Qwen/Qwen3-1.7B | device=mps")
    logging.getLogger("hif.clustering.embed").debug(
        "Failed to load primary embedding model 'x'. Falling back to 'y'."
    )
    logging.getLogger("hif.hourglass.output_side").debug(
        "Requested top_k=50 exceeds model's max_top_k=20. Reducing top_k to 20."
    )
    logging.getLogger("httpx").info(
        "HTTP Request: POST https://internal-proxy.example/api 'HTTP/1.1 200 OK'"
    )


def _patch_pipeline(monkeypatch, profile):
    def fake_run(*a, **k):
        _emit_chatter()
        return profile, None

    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_load_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_run_single_profile", fake_run)


@pytest.fixture(autouse=True)
def _reset_logging_state():
    # Under pytest, root already has handlers so basicConfig in
    # hif.utils.logging is a no-op — attach the Rich handler explicitly to
    # reproduce the production console pipeline.
    from rich.logging import RichHandler

    handler = RichHandler(console=hif_logging.console)
    logging.getLogger().addHandler(handler)
    yield
    logging.getLogger().removeHandler(handler)
    # Undo per-test mutations of module-global logging state.
    hif_logging.console.file = None
    configure_logging(verbose=False)


def _invoke(args):
    return runner.invoke(app, ["profile", "m", "p", *args])


# ---------------------------------------------------------------------------
# default: clean results
# ---------------------------------------------------------------------------


def test_default_profile_output_is_clean(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, _make_profile())
    result = _invoke(["--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Measurements" in result.output  # results still render
    for line in result.output.splitlines():
        assert not NOISE_RE.search(line), f"internal chatter leaked: {line!r}"


def test_default_metric_output_is_clean(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, _make_profile())
    result = _invoke(
        ["--output-dir", str(tmp_path), "--metric", "perturbation_jsd_bits"]
    )
    assert result.exit_code == 0, result.output
    lines = [l for l in result.output.splitlines() if l.strip()]
    for line in lines:
        assert not NOISE_RE.search(line), f"internal chatter leaked: {line!r}"
    assert any(l.startswith("perturbation_jsd_bits = ") for l in lines)


def test_genuine_warnings_still_show_by_default(monkeypatch, tmp_path):
    profile = _make_profile()

    def fake_run(*a, **k):
        logging.getLogger("hif.profile.builder").warning(
            "Perturbation generator 'x' failed: boom"
        )
        return profile, None

    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_load_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_run_single_profile", fake_run)
    result = _invoke(["--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    # Rich wraps the message around its columns — assert on tokens, not the
    # exact phrase.
    assert "WARNING" in result.output
    assert "Perturbation" in result.output
    assert "boom" in result.output


# ---------------------------------------------------------------------------
# --verbose: everything
# ---------------------------------------------------------------------------


def test_verbose_shows_tables_and_chatter(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, _make_profile())
    result = _invoke(["--output-dir", str(tmp_path), "--verbose"])
    assert result.exit_code == 0, result.output
    # Stats / latency tables
    assert "Stats" in result.output
    assert "Latency" in result.output
    # Effective-config notes
    assert "top-K" in result.output
    assert "embedder" in result.output
    # Full internal chatter: pipeline INFO, DEBUG adjustments, third-party HTTP
    normalized = " ".join(result.output.split())
    assert "Loading tokenizer" in normalized
    assert "Reducing top_k to 20" in normalized
    assert "Falling back to" in normalized
    assert "HTTP Request" in normalized


# ---------------------------------------------------------------------------
# --json: pure JSON on stdout
# ---------------------------------------------------------------------------


def test_json_stdout_is_pure_json_with_default_logging(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, _make_profile())
    result = _invoke(["--output-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["schema_version"]
    assert "latency" in data


# ---------------------------------------------------------------------------
# logger-level contract (shared setup path — applies to every command)
# ---------------------------------------------------------------------------


def test_default_levels():
    configure_logging(verbose=False)
    assert logging.getLogger("hif").level == logging.WARNING
    # Every declared third-party logger, so adding one cannot silently escape
    # the quiet default.
    assert hif_logging._THIRD_PARTY_LOGGERS  # non-empty
    for name in hif_logging._THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_verbose_levels():
    configure_logging(verbose=True)
    assert logging.getLogger("hif").level == logging.DEBUG
    for name in hif_logging._THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).level == logging.DEBUG


def test_top_k_clamp_logs_at_debug_only(caplog):
    from hif.hourglass import output_side

    class FakeModel:
        name = "fake-clamp-model"
        max_top_k = 20

    output_side._warned_top_k_combos.clear()
    with caplog.at_level(logging.DEBUG, logger="hif.hourglass.output_side"):
        output_side._warn_top_k_once(FakeModel(), 50)
    records = [r for r in caplog.records if "Reducing top_k" in r.message]
    assert records and all(r.levelno == logging.DEBUG for r in records)


# ---------------------------------------------------------------------------
# artifact metadata: effective values are recorded even though default is quiet
# ---------------------------------------------------------------------------


def test_effective_embedder_recorded_in_profile_config():
    from hif.config import RunConfig
    from hif.profile.builder import _record_effective_embedder

    class FakeEmbedder:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    config = RunConfig()
    config.embedding.model_name = "google/embeddinggemma-300m"
    out = _record_effective_embedder(config, FakeEmbedder())
    assert out.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    # original config not mutated
    assert config.embedding.model_name == "google/embeddinggemma-300m"


def test_embedding_defaults_are_minilm_primary():
    from hif.config import EmbeddingConfig

    cfg = EmbeddingConfig()
    assert cfg.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.fallback_model_name == cfg.model_name
    assert cfg.matryoshka_dim is None
