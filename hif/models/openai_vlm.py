"""OpenAI vision backend (partial access) implementing MultimodalModel.

GPT-4o (and other OpenAI vision chat models) over the chat completions API:
images travel as base64 data URLs inside the messages payload; the API
returns output logprobs only (top_logprobs<=20), so this adapter can produce
the output-side measurements but not the input-side ones:

- supports_teacher_forcing = False; forward_prepared raises
  NotImplementedError (same pattern as the Ollama backend).
- InputPartMap has NO patch geometry: image parts contribute zero-length
  spans with grid_rows/grid_cols=None. Region sensitivity still works because
  masking happens on OUR side before sending — the sweep's grid geometry
  comes from ImageGridMaskFamily's perturbation traces, never the part map.
- prepare() tokenizes text parts via tiktoken (approximation used only for
  span bookkeeping and length accounting; the API sees the messages payload
  in backend_state, which is excluded from every serialization path).
"""

from __future__ import annotations

import base64
from typing import Any

from hif.models.base import GenerationResult, Logits
from hif.models.mm import (
    InputPartMap,
    MultimodalInput,
    MultimodalModel,
    PartSpan,
    PreparedInput,
)
from hif.models.openai_model import OpenAIModel
from hif.utils.logging import get_logger

logger = get_logger(__name__)


def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _image_data_url(part) -> str:
    if part.image_path is not None:
        with open(part.image_path, "rb") as f:
            data = f.read()
    else:
        data = part.image_bytes
    mime = _sniff_mime(data)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class OpenAIVLMModel(OpenAIModel, MultimodalModel):
    """OpenAI chat-completions vision backend (e.g. gpt-4o)."""

    # supports_teacher_forcing=False and max_top_k=20 inherited from OpenAIModel.

    # --- Multimodal interface ---

    def prepare(self, mm_input: MultimodalInput) -> PreparedInput:
        """Build the messages payload and a geometry-free part map.

        input_ids are the tiktoken encoding of the concatenated text parts
        (in part order); text spans cover each part's own tokens. Image parts
        get zero-length spans (no patch geometry exists for an API model).
        """
        content: list[dict[str, Any]] = []
        input_ids: list[int] = []
        spans: list[PartSpan] = []

        for part_index, part in enumerate(mm_input.parts):
            if part.kind == "text":
                piece = list(self._enc.encode(part.text or ""))
                spans.append(
                    PartSpan(
                        part_index=part_index,
                        kind="text",
                        pos_start=len(input_ids),
                        pos_end=len(input_ids) + len(piece),
                    )
                )
                input_ids.extend(piece)
                content.append({"type": "text", "text": part.text or ""})
            else:
                # Zero-length span at the current position: API models expose
                # no patch geometry (grid_rows/grid_cols=None by default).
                spans.append(
                    PartSpan(
                        part_index=part_index,
                        kind="image",
                        pos_start=len(input_ids),
                        pos_end=len(input_ids),
                    )
                )
                content.append(
                    {"type": "image_url", "image_url": {"url": _image_data_url(part)}}
                )

        messages = [{"role": "user", "content": content}]
        return PreparedInput(
            input_ids=input_ids,
            part_map=InputPartMap(spans=spans, seq_len=len(input_ids)),
            backend_state={"messages": messages},
        )

    def forward_prepared(self, prepared: PreparedInput) -> Logits:
        raise NotImplementedError(
            "OpenAIVLMModel does not support teacher-forced forward passes "
            "(API returns generated-token logprobs only)."
        )

    def generate_prepared(
        self,
        prepared: PreparedInput,
        max_new_tokens: int,
        top_k: int,
        seed: int,
    ) -> GenerationResult:
        messages = prepared.backend_state["messages"]
        return self._generate_from_messages(
            messages=messages,
            input_ids=list(prepared.input_ids),
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            seed=seed,
        )
