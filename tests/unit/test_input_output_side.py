"""Unit tests for hourglass input_side and output_side modules using GPT-2 small (CPU)."""

import math

import pytest

from hif.config import ModelConfig
from hif.hourglass.input_side import InputSideAnalysis, analyze_input_side
from hif.hourglass.output_side import OutputSideTrace, collect_output_trace
from hif.models.hf import HFModel

from profile_helpers import FakeModel


# ---------------------------------------------------------------------------
# Module-scoped GPT-2 fixture (loaded once for all tests in this file)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_model() -> HFModel:
    try:
        config = ModelConfig(name="gpt2", backend="hf", device="cpu")
        return HFModel(config)
    except OSError:
        pytest.skip("GPT-2 model files not in local cache — connect to HuggingFace to download")


# ---------------------------------------------------------------------------
# Input-side tests
# ---------------------------------------------------------------------------

class TestAnalyzeInputSide:
    def test_analyze_input_side_returns_correct_structure(self, gpt2_model: HFModel):
        prompt = "The cat sat on the mat"
        result = analyze_input_side(gpt2_model, prompt, top_k=50)

        assert isinstance(result, InputSideAnalysis)

        token_ids = gpt2_model.tokenize(prompt)
        # positions should be 1..N-1 (one per token after the first)
        assert len(result.positions) == len(token_ids) - 1
        assert result.prompt_token_ids == token_ids
        assert result.prompt_text == prompt

        vocab_size = gpt2_model.vocab_size
        max_entropy = math.log2(vocab_size)

        for pos in result.positions:
            assert pos.surprisal > 0.0, f"surprisal should be positive, got {pos.surprisal}"
            assert 0.0 <= pos.entropy <= max_entropy + 1e-6, (
                f"entropy {pos.entropy} out of range [0, {max_entropy}]"
            )
            assert len(pos.top_k_alternatives) <= 50

    def test_volatility_score_in_range(self, gpt2_model: HFModel):
        prompt = "Once upon a time in a land far away"
        result = analyze_input_side(gpt2_model, prompt, top_k=20)
        assert 0.0 <= result.volatility_score <= 1.0

    def test_teacher_forcing_guard(self):
        """Model with supports_teacher_forcing=False must raise NotImplementedError."""
        fake_model = FakeModel(supports_teacher_forcing=False, name="fake-no-tf")

        with pytest.raises(NotImplementedError):
            analyze_input_side(fake_model, "hello world")

    def test_mean_surprisal_is_positive(self, gpt2_model: HFModel):
        prompt = "Hello world"
        result = analyze_input_side(gpt2_model, prompt)
        assert result.mean_surprisal > 0.0

    def test_mean_entropy_positive(self, gpt2_model: HFModel):
        prompt = "Hello world"
        result = analyze_input_side(gpt2_model, prompt)
        assert result.mean_entropy > 0.0

    def test_max_entropy_equals_log2_vocab(self, gpt2_model: HFModel):
        prompt = "Hello world"
        result = analyze_input_side(gpt2_model, prompt)
        expected = math.log2(gpt2_model.vocab_size)
        assert result.max_entropy == pytest.approx(expected, abs=1e-6)

    def test_top_k_alternatives_count(self, gpt2_model: HFModel):
        prompt = "Hello world"
        result = analyze_input_side(gpt2_model, prompt, top_k=10)
        for pos in result.positions:
            assert len(pos.top_k_alternatives) == 10

    def test_top_k_alternative_fields(self, gpt2_model: HFModel):
        prompt = "Hello world"
        result = analyze_input_side(gpt2_model, prompt, top_k=5)
        for pos in result.positions:
            for alt in pos.top_k_alternatives:
                assert "token_id" in alt
                assert "token_str" in alt
                assert "prob" in alt
                assert 0.0 <= alt["prob"] <= 1.0


# ---------------------------------------------------------------------------
# Output-side tests
# ---------------------------------------------------------------------------

class TestCollectOutputTrace:
    def test_collect_output_trace_structure(self, gpt2_model: HFModel):
        prompt = "The quick brown fox"
        max_new_tokens = 5
        top_k = 10
        result = collect_output_trace(
            gpt2_model, prompt, max_new_tokens=max_new_tokens, top_k=top_k, seed=42
        )

        assert isinstance(result, OutputSideTrace)
        # Generation may stop early on EOS, but here GPT-2 rarely hits EOS in 5 tokens
        assert len(result.steps) == max_new_tokens
        for step in result.steps:
            assert len(step.topk) == top_k

    def test_mean_step_entropy_positive(self, gpt2_model: HFModel):
        prompt = "The quick brown fox"
        result = collect_output_trace(gpt2_model, prompt, max_new_tokens=5, top_k=50, seed=42)
        assert result.mean_step_entropy > 0.0

    def test_output_trace_metadata(self, gpt2_model: HFModel):
        prompt = "Hello world"
        result = collect_output_trace(
            gpt2_model, prompt, max_new_tokens=3, top_k=20, seed=99
        )
        assert result.prompt_text == prompt
        assert result.model_name == gpt2_model.name
        assert result.top_k == 20
        assert result.max_new_tokens == 3
        assert result.seed == 99

    def test_input_ids_match_tokenization(self, gpt2_model: HFModel):
        prompt = "Hello world"
        result = collect_output_trace(gpt2_model, prompt, max_new_tokens=3, seed=0)
        expected_ids = gpt2_model.tokenize(prompt)
        assert result.input_ids == expected_ids

    def test_generated_ids_not_empty(self, gpt2_model: HFModel):
        prompt = "Hello world"
        result = collect_output_trace(gpt2_model, prompt, max_new_tokens=3, seed=0)
        assert len(result.generated_ids) > 0

    def test_deterministic_with_same_seed(self, gpt2_model: HFModel):
        prompt = "Once upon a time"
        result1 = collect_output_trace(gpt2_model, prompt, max_new_tokens=5, seed=7)
        result2 = collect_output_trace(gpt2_model, prompt, max_new_tokens=5, seed=7)
        assert result1.generated_ids == result2.generated_ids

