"""Unit tests for the OllamaModel HF-tokenizer fallback when /api/tokenize 404s."""

import pytest

from hif.config import ModelConfig
from hif.models.ollama import OllamaModel, _OllamaHTTPError


def _make_model(monkeypatch, name: str = "llama3.2:latest") -> OllamaModel:
    """Build an OllamaModel without touching the network (metadata load stubbed)."""
    monkeypatch.setattr(OllamaModel, "_load_metadata", lambda self: None)
    return OllamaModel(ModelConfig(name=name, backend="ollama"))


def test_tokenize_falls_back_on_404_and_caches(monkeypatch) -> None:
    model = _make_model(monkeypatch, name="definitely-unknown-model:latest")

    calls = {"tokenize": 0}

    def fake_post(self, url, payload):
        if url.endswith("/api/tokenize"):
            calls["tokenize"] += 1
            raise _OllamaHTTPError(404, "Ollama API returned 404: page not found")
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(OllamaModel, "_post_with_retry", fake_post)

    try:
        ids = model.tokenize("Hello, world!")
    except NotImplementedError as exc:
        pytest.skip(f"HF fallback tokenizer not available locally: {exc}")

    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    # 404 is cached: a second tokenize call must not re-hit the endpoint
    model.tokenize("Second call")
    assert calls["tokenize"] == 1
    assert model._api_tokenize_available is False

    # detokenize now works via the fallback tokenizer
    text = model.detokenize(ids)
    assert "Hello" in text


def test_tokenize_uses_api_when_available(monkeypatch) -> None:
    model = _make_model(monkeypatch)

    def fake_post(self, url, payload):
        assert url.endswith("/api/tokenize")
        return {"tokens": [1, 2, 3]}

    monkeypatch.setattr(OllamaModel, "_post_with_retry", fake_post)
    assert model.tokenize("hi") == [1, 2, 3]
    assert model._api_tokenize_available is True
    # No fallback tokenizer was loaded
    assert model._fallback_tokenizer is None


def _stub_generate_response(monkeypatch, model: OllamaModel, response: dict) -> None:
    def fake_post(self, url, payload):
        assert url.endswith("/api/generate")
        # Current Ollama API: logprobs/top_logprobs are top-level fields
        assert payload["logprobs"] is True
        assert isinstance(payload["top_logprobs"], int)
        return response

    monkeypatch.setattr(OllamaModel, "_post_with_retry", fake_post)
    monkeypatch.setattr(OllamaModel, "_ids_to_text", lambda self, ids: "2+2=")


def test_generate_parses_current_dict_logprobs_format(monkeypatch) -> None:
    """Ollama >= 0.12 returns per-step dicts with nested top_logprobs."""
    model = _make_model(monkeypatch)
    response = {
        "response": "4!",
        "logprobs": [
            {
                "token": "4",
                "logprob": -0.01,
                "bytes": [52],
                "top_logprobs": [
                    {"token": "4", "logprob": -0.01, "bytes": [52]},
                    {"token": "2", "logprob": -5.0, "bytes": [50]},
                ],
            },
            {
                "token": "!",
                "logprob": -0.5,
                "bytes": [33],
                "top_logprobs": [
                    {"token": "!", "logprob": -0.5, "bytes": [33]},
                    {"token": ".", "logprob": -1.5, "bytes": [46]},
                ],
            },
        ],
    }
    _stub_generate_response(monkeypatch, model, response)

    result = model.generate([1, 2, 3], max_new_tokens=5, top_k=5, seed=0)
    assert len(result.steps) == 2
    assert result.steps[0].selected_token_str == "4"
    assert result.steps[0].topk[0].logprob == -0.01
    assert result.steps[0].topk[1].token_str == "2"
    assert result.steps[1].selected_token_str == "!"
    assert len(result.generated_ids) == 2


def test_generate_parses_old_list_logprobs_format(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    response = {
        "response": "4",
        "eval_logprobs": [
            [
                {"token": "4", "logprob": -0.02},
                {"token": "2", "logprob": -4.0},
            ]
        ],
    }
    _stub_generate_response(monkeypatch, model, response)

    result = model.generate([1], max_new_tokens=3, top_k=5, seed=0)
    assert len(result.steps) == 1
    assert result.steps[0].selected_token_str == "4"


def test_generate_all_steps_unusable_raises(monkeypatch) -> None:
    """If every step is skipped, raise instead of computing metrics from zero steps."""
    model = _make_model(monkeypatch)
    response = {"response": "4", "logprobs": [-0.1, -0.2, -0.3]}  # flat floats
    _stub_generate_response(monkeypatch, model, response)

    with pytest.raises(RuntimeError, match="no usable logprobs"):
        model.generate([1], max_new_tokens=3, top_k=5, seed=0)


def test_generate_missing_logprobs_raises(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    _stub_generate_response(monkeypatch, model, {"response": "4"})

    with pytest.raises(RuntimeError):
        model.generate([1], max_new_tokens=3, top_k=5, seed=0)


def test_non_404_error_still_raises_not_implemented(monkeypatch) -> None:
    model = _make_model(monkeypatch)

    def fake_post(self, url, payload):
        raise _OllamaHTTPError(500, "Ollama API returned 500: boom")

    monkeypatch.setattr(OllamaModel, "_post_with_retry", fake_post)
    with pytest.raises(NotImplementedError):
        model.tokenize("hi")
