"""Multimodal input representation and the MultimodalModel capability ABC.

Implements docs/confidential/specs/MULTIMODAL.md § Design §1-2 (M1: image+text →
text). Text-only models and call sites are untouched: `tokenize/detokenize/
forward/generate` keep their exact signatures on `Model`; multimodality enters
only via `MultimodalModel.prepare()/forward_prepared()/generate_prepared()`.

Privacy invariant: `InputPart.image_bytes` must never reach disk or the API.
Profiles store only `InputPartRecord` (hash + dims) — see
hif/profile/schema.py and the raw-trace guard rules in the spec.
"""

from __future__ import annotations

import hashlib
from abc import abstractmethod
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hif.models.base import GenerationResult, Logits, Model


class InputPart(BaseModel):
    """One ordered part of a multimodal input.

    M1 kinds: "text" | "image". M2/M3 add "document_page", "frame",
    "audio_chunk". For images, exactly one of image_path/image_bytes is set;
    image_bytes holds in-memory perturbed variants and is never persisted.
    """

    kind: Literal["text", "image"]
    text: Optional[str] = None
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    content_hash: str  # sha256 of text-utf8 or media bytes
    width: Optional[int] = None
    height: Optional[int] = None

    @model_validator(mode="after")
    def _check_kind_payload(self) -> "InputPart":
        if self.kind == "text":
            if self.text is None:
                raise ValueError("text part requires text")
            if self.image_path is not None or self.image_bytes is not None:
                raise ValueError("text part must not carry image payload")
        elif self.kind == "image":
            if (self.image_path is None) == (self.image_bytes is None):
                raise ValueError(
                    "image part requires exactly one of image_path/image_bytes"
                )
        return self

    @classmethod
    def from_text(cls, text: str) -> "InputPart":
        return cls(
            kind="text",
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )

    @classmethod
    def from_image_path(cls, image_path: str) -> "InputPart":
        from PIL import Image

        with open(image_path, "rb") as f:
            data = f.read()
        with Image.open(image_path) as img:
            width, height = img.size
        return cls(
            kind="image",
            image_path=image_path,
            content_hash=hashlib.sha256(data).hexdigest(),
            width=width,
            height=height,
        )

    @classmethod
    def from_image_bytes(cls, image_bytes: bytes) -> "InputPart":
        import io

        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
        return cls(
            kind="image",
            image_bytes=image_bytes,
            content_hash=hashlib.sha256(image_bytes).hexdigest(),
            width=width,
            height=height,
        )


class MultimodalInput(BaseModel):
    """Ordered list of input parts, as presented to the model."""

    parts: list[InputPart]

    @classmethod
    def from_text(cls, text: str) -> "MultimodalInput":
        return cls(parts=[InputPart.from_text(text)])

    @property
    def modality(self) -> str:
        """Deterministic modality string (closed enum per release).

        Derivation: sorted unique media (non-text) kinds joined with "+",
        with a "+text" suffix; no media parts ⇒ "text".
        M1 values: "text" | "image+text".
        """
        media_kinds = sorted({p.kind for p in self.parts if p.kind != "text"})
        if not media_kinds:
            return "text"
        return "+".join(media_kinds) + "+text"

    @property
    def text_concat(self) -> str:
        """Concatenation of text parts (embeddings, hashing, similarity)."""
        return "".join(p.text for p in self.parts if p.kind == "text" and p.text)


class PartSpan(BaseModel):
    """Positions in the prepared sequence covered by one input part.

    For image parts, position p in the span maps to patch
    (row, col) = divmod(p - pos_start, grid_cols) — row-major, guaranteed.
    """

    part_index: int  # index into MultimodalInput.parts
    kind: str
    pos_start: int  # inclusive
    pos_end: int  # exclusive
    grid_rows: Optional[int] = None
    grid_cols: Optional[int] = None


class InputPartMap(BaseModel):
    """Single artifact mapping prepared-sequence positions to input parts.

    Spans cover [0, seq_len) minus special/structural tokens. Geometry only —
    never pixels.
    """

    spans: list[PartSpan]
    seq_len: int

    def text_positions(self) -> list[int]:
        """All positions inside text-kind spans, ascending."""
        positions: list[int] = []
        for span in self.spans:
            if span.kind == "text":
                positions.extend(range(span.pos_start, span.pos_end))
        return sorted(positions)


class PreparedInput(BaseModel):
    """Output of MultimodalModel.prepare(): the processed sequence plus map.

    backend_state is opaque backend data (e.g. pixel_values tensors) and is
    excluded from every serialization path — it must never reach disk.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: list[int]  # processed sequence (placeholder ids for patches)
    part_map: InputPartMap
    backend_state: Any = Field(default=None, exclude=True)


class MultimodalModel(Model):
    """Capability ABC for models that accept media parts.

    Mirrors the `supports_teacher_forcing` capability pattern. `tokenize/
    forward/generate` remain the text-only interface inherited from Model;
    media enters only via prepare()/*_prepared().
    """

    @property
    def supports_multimodal_input(self) -> bool:
        return True

    @abstractmethod
    def prepare(self, mm_input: MultimodalInput) -> PreparedInput: ...

    @abstractmethod
    def forward_prepared(self, prepared: PreparedInput) -> Logits:
        """Teacher-forced forward pass over the prepared sequence.

        Raises NotImplementedError on backends without full teacher forcing.
        """

    @abstractmethod
    def generate_prepared(
        self,
        prepared: PreparedInput,
        max_new_tokens: int,
        top_k: int,
        seed: int,
    ) -> GenerationResult:
        """Generate from a prepared multimodal input, collecting top-K steps.

        GenerationResult.input_ids holds the full processed sequence ids
        (image placeholder ids included) so len(input_ids) stays meaningful;
        generated_ids/steps are unchanged in meaning.
        """
