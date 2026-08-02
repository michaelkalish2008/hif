"""Unit tests for HFModel using GPT-2 small on CPU."""

import numpy as np
import pytest
import torch

from hif.config import ModelConfig
from hif.models.hf import HFModel


@pytest.fixture(scope="module")
def gpt2_model() -> HFModel:
    try:
        config = ModelConfig(name="gpt2", backend="hf", device="cpu")
        return HFModel(config)
    except OSError:
        pytest.skip("GPT-2 model files not in local cache — connect to HuggingFace to download")


def test_load_gpt2(gpt2_model: HFModel) -> None:
    assert gpt2_model.name == "gpt2"
    assert gpt2_model.vocab_size == 50257
    assert gpt2_model.context_length == 1024


def test_tokenize_detokenize_roundtrip(gpt2_model: HFModel) -> None:
    text = "Hello, world!"
    ids = gpt2_model.tokenize(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    recovered = gpt2_model.detokenize(ids)
    # GPT-2 tokenizer may add a leading space; strip both sides for comparison
    assert text in recovered or recovered.strip() == text.strip()


def test_forward_shape(gpt2_model: HFModel) -> None:
    input_ids = [50256, 50256]  # two EOS tokens
    result = gpt2_model.forward(input_ids)
    assert result.seq_len == 2
    assert result.vocab_size == 50257
    arr = result.to_numpy()
    assert arr.shape == (2, 50257)


def test_forward_probabilities_sum_to_one(gpt2_model: HFModel) -> None:
    input_ids = [50256, 50256]
    result = gpt2_model.forward(input_ids)
    logits_pos0 = torch.tensor(result.values[0])
    probs = torch.softmax(logits_pos0, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-4


def test_generate_returns_correct_structure(gpt2_model: HFModel) -> None:
    prompt = "The quick brown fox"
    input_ids = gpt2_model.tokenize(prompt)
    top_k = 5
    result = gpt2_model.generate(input_ids=input_ids, max_new_tokens=5, top_k=top_k, seed=42)

    assert len(result.steps) == 5
    for step in result.steps:
        assert len(step.topk) == top_k
        total_prob = sum(entry.prob for entry in step.topk)
        assert all(entry.prob >= 0 for entry in step.topk)
        # Top-K probs are a slice of the full distribution, so sum <= 1.0
        assert total_prob <= 1.0 + 1e-5


def test_generate_deterministic(gpt2_model: HFModel) -> None:
    prompt = "Once upon a time"
    input_ids = gpt2_model.tokenize(prompt)
    result1 = gpt2_model.generate(input_ids=input_ids, max_new_tokens=5, top_k=10, seed=7)
    result2 = gpt2_model.generate(input_ids=input_ids, max_new_tokens=5, top_k=10, seed=7)
    assert result1.generated_ids == result2.generated_ids
    assert [s.selected_token_id for s in result1.steps] == [
        s.selected_token_id for s in result2.steps
    ]


def test_max_top_k_is_none(gpt2_model: HFModel) -> None:
    assert gpt2_model.max_top_k is None


def test_supports_teacher_forcing(gpt2_model: HFModel) -> None:
    assert gpt2_model.supports_teacher_forcing is True
