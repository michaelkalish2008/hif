"""Unit tests for OpenAIVLMModel (GPT-4o vision, partial-access tier).

All API traffic is mocked — no network. Covers:
- prepare(): messages payload (base64 data URL + text), geometry-free part map
- forward_prepared(): NotImplementedError (Ollama pattern)
- generate_prepared(): logprobs → GenerationResult step mapping
- region-sensitivity assembly works without part-map grid geometry
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("tiktoken", reason="openai extras not installed")
pytest.importorskip("openai", reason="openai extras not installed")

from hif.config import ModelConfig
from hif.models.mm import InputPart, MultimodalInput
from hif.models.openai_vlm import OpenAIVLMModel


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 200, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _mm_input() -> MultimodalInput:
    return MultimodalInput(
        parts=[
            InputPart.from_image_bytes(_png_bytes()),
            InputPart.from_text("Question: What color? Answer:"),
        ]
    )


@pytest.fixture
def vlm() -> OpenAIVLMModel:
    model = OpenAIVLMModel(
        ModelConfig(name="gpt-4o", backend="openai", api_key="test-key")
    )
    model._client = MagicMock()
    return model


class TestPrepare:
    def test_part_map_has_no_patch_geometry(self, vlm):
        prepared = vlm.prepare(_mm_input())
        image_spans = [s for s in prepared.part_map.spans if s.kind == "image"]
        assert len(image_spans) == 1
        span = image_spans[0]
        assert span.grid_rows is None and span.grid_cols is None
        assert span.pos_start == span.pos_end  # zero-length: no positions claimed

    def test_text_span_covers_text_tokens(self, vlm):
        prepared = vlm.prepare(_mm_input())
        text_spans = [s for s in prepared.part_map.spans if s.kind == "text"]
        assert len(text_spans) == 1
        span = text_spans[0]
        assert span.pos_end - span.pos_start == len(prepared.input_ids)
        assert prepared.part_map.seq_len == len(prepared.input_ids)
        assert prepared.part_map.text_positions() == list(range(len(prepared.input_ids)))

    def test_messages_payload_image_data_url_and_text(self, vlm):
        prepared = vlm.prepare(_mm_input())
        messages = prepared.backend_state["messages"]
        assert len(messages) == 1 and messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1] == {"type": "text", "text": "Question: What color? Answer:"}

    def test_backend_state_excluded_from_serialization(self, vlm):
        prepared = vlm.prepare(_mm_input())
        assert "backend_state" not in prepared.model_dump()

    def test_supports_flags(self, vlm):
        assert vlm.supports_multimodal_input is True
        assert vlm.supports_teacher_forcing is False
        assert vlm.max_top_k == 20


class TestForwardPrepared:
    def test_raises_not_implemented(self, vlm):
        prepared = vlm.prepare(_mm_input())
        with pytest.raises(NotImplementedError):
            vlm.forward_prepared(prepared)


def _mock_completion(tokens: list[tuple[str, float]]):
    """Chat completion with logprobs.content, 2 alternatives per step."""
    content = []
    for tok, lp in tokens:
        content.append(
            SimpleNamespace(
                token=tok,
                logprob=lp,
                top_logprobs=[
                    SimpleNamespace(token=tok, logprob=lp),
                    SimpleNamespace(token=tok + "_alt", logprob=lp - 1.5),
                ],
            )
        )
    choice = SimpleNamespace(
        logprobs=SimpleNamespace(content=content),
        message=SimpleNamespace(content="".join(t for t, _ in tokens)),
    )
    return SimpleNamespace(choices=[choice])


class TestGeneratePrepared:
    def test_steps_have_distributions(self, vlm):
        vlm._client.chat.completions.create.return_value = _mock_completion(
            [(" Green", -0.05), (".", -0.2)]
        )
        prepared = vlm.prepare(_mm_input())
        result = vlm.generate_prepared(prepared, max_new_tokens=8, top_k=20, seed=0)

        assert len(result.steps) == 2
        assert result.input_ids == prepared.input_ids
        for step in result.steps:
            assert len(step.topk) == 2
            probs = [e.prob for e in step.topk]
            assert all(0.0 < p <= 1.0 for p in probs)
            assert abs(sum(probs) - 1.0) < 1e-9  # renormalised over top-K
        assert result.steps[0].selected_token_str == " Green"

        # The API call carried the multimodal messages payload with logprobs.
        kwargs = vlm._client.chat.completions.create.call_args.kwargs
        assert kwargs["logprobs"] is True
        assert kwargs["top_logprobs"] == 20
        assert kwargs["messages"] == prepared.backend_state["messages"]

    def test_region_sensitivity_without_grid_geometry(self, vlm):
        """The sweep's grid comes from ImageGridMaskFamily traces, not the
        part map — assembly must work with a geometry-free part map."""
        from hif.analysis.region_sensitivity import assemble_region_sensitivity
        from hif.metrics.sensitivity import compute_sensitivity_metrics  # noqa: F401
        from hif.perturbation.base import PerturbationTrace
        from hif.metrics.sensitivity import SensitivityMetrics

        pairs = []
        for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            trace = PerturbationTrace(
                family="image_grid_mask",
                part_index=0,
                regions=[{"row": r, "col": c}],
                params={"grid_rows": 2, "grid_cols": 2, "fill": "mean"},
            )
            sens = SensitivityMetrics(
                perturbation_generator="image_grid_mask",
                perturbed_prompt=f"cell {r},{c}",
                original_prompt="baseline",
                step_sensitivities=[],
                mean_js_divergence=0.1 * (i + 1),
                mean_kl_divergence=0.0,
                mean_entropy_delta=0.0,
                output_entropy_delta=0.0,
                mean_nucleus_stability_p90=1.0,
            )
            pairs.append((trace, sens))

        rs = assemble_region_sensitivity(pairs)
        assert rs is not None
        assert rs.grid_rows == 2 and rs.grid_cols == 2
        assert len(rs.cells) == 4
        assert rs.max_cell.row == 1 and rs.max_cell.col == 1
