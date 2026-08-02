"""Unit tests for the multimodal M1 contracts (MULTIMODAL.md § Design).

Covers: MultimodalInput modality derivation, InputPartMap.text_positions and
row-major patch math, HFVLMModel.prepare with a mocked processor, the
media+non-multimodal ValueError, 0.2.0 profile compatibility, and the
no-pixels-in-JSON invariant (Risk rule 2).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hif.models.mm import (
    InputPart,
    InputPartMap,
    MultimodalInput,
    PartSpan,
    PreparedInput,
)
from hif.profile.schema import BehavioralRangeProfile, InputPartRecord

from tests.unit.profile_helpers import (
    FakeEmbeddingModel,
    FakeModel,
    _make_profile,
    _make_run_config,
)

# 1x1 PNG (red) — a real, tiny image payload for byte-based parts.
def _make_png_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


_PNG_BYTES = _make_png_bytes()


def _mm_image_text() -> MultimodalInput:
    return MultimodalInput(
        parts=[
            InputPart.from_image_bytes(_PNG_BYTES),
            InputPart.from_text("What shape is in this image?"),
        ]
    )


# ---------------------------------------------------------------------------
# MultimodalInput
# ---------------------------------------------------------------------------


class TestMultimodalInput:
    def test_modality_text_only(self):
        assert MultimodalInput.from_text("hello").modality == "text"

    def test_modality_image_text(self):
        assert _mm_image_text().modality == "image+text"

    def test_modality_image_only_is_image_text(self):
        mm = MultimodalInput(parts=[InputPart.from_image_bytes(_PNG_BYTES)])
        assert mm.modality == "image+text"

    def test_modality_deterministic_regardless_of_order(self):
        a = MultimodalInput(
            parts=[
                InputPart.from_text("a"),
                InputPart.from_image_bytes(_PNG_BYTES),
            ]
        )
        b = MultimodalInput(
            parts=[
                InputPart.from_image_bytes(_PNG_BYTES),
                InputPart.from_text("a"),
            ]
        )
        assert a.modality == b.modality == "image+text"

    def test_from_text_single_part_and_hash(self):
        mm = MultimodalInput.from_text("hello")
        assert len(mm.parts) == 1
        assert mm.parts[0].kind == "text"
        assert mm.parts[0].content_hash == hashlib.sha256(b"hello").hexdigest()

    def test_text_concat_skips_media(self):
        mm = MultimodalInput(
            parts=[
                InputPart.from_text("foo "),
                InputPart.from_image_bytes(_PNG_BYTES),
                InputPart.from_text("bar"),
            ]
        )
        assert mm.text_concat == "foo bar"

    def test_image_part_dims_and_hash(self):
        part = InputPart.from_image_bytes(_PNG_BYTES)
        assert (part.width, part.height) == (1, 1)
        assert part.content_hash == hashlib.sha256(_PNG_BYTES).hexdigest()

    def test_image_part_requires_exactly_one_payload(self):
        with pytest.raises(ValueError):
            InputPart(kind="image", content_hash="x")


# ---------------------------------------------------------------------------
# InputPartMap / PartSpan
# ---------------------------------------------------------------------------


class TestInputPartMap:
    def test_text_positions_only_text_spans(self):
        pm = InputPartMap(
            spans=[
                PartSpan(part_index=0, kind="image", pos_start=2, pos_end=6,
                         grid_rows=2, grid_cols=2),
                PartSpan(part_index=1, kind="text", pos_start=7, pos_end=10),
            ],
            seq_len=11,
        )
        assert pm.text_positions() == [7, 8, 9]

    def test_row_major_patch_math(self):
        span = PartSpan(part_index=0, kind="image", pos_start=10, pos_end=26,
                        grid_rows=4, grid_cols=4)
        # Position p maps to (row, col) = divmod(p - pos_start, grid_cols)
        assert divmod(10 - span.pos_start, span.grid_cols) == (0, 0)
        assert divmod(13 - span.pos_start, span.grid_cols) == (0, 3)
        assert divmod(14 - span.pos_start, span.grid_cols) == (1, 0)
        assert divmod(25 - span.pos_start, span.grid_cols) == (3, 3)
        assert span.grid_rows * span.grid_cols == span.pos_end - span.pos_start

    def test_prepared_input_backend_state_never_serialized(self):
        prepared = PreparedInput(
            input_ids=[1, 2, 3],
            part_map=InputPartMap(spans=[], seq_len=3),
            backend_state={"pixel_values": object()},
        )
        dumped = prepared.model_dump()
        assert "backend_state" not in dumped
        assert "pixel" not in json.dumps(dumped).lower()


# ---------------------------------------------------------------------------
# HFVLMModel.prepare with a mocked processor
# ---------------------------------------------------------------------------


IMAGE_TOKEN_ID = 7


class _FakeTokenizer:
    vocab_size = 100
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        # "What shape is in this image?" → fixed ids
        return [20, 21, 22]

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


class _FakeProcessorOutput(dict):
    pass


class _FakeProcessor:
    def __call__(self, text=None, images=None, return_tensors=None):
        import torch

        # [BOS, wrap, img, img, img, img, wrap, t20, t21, t22]
        ids = [1, 90, 7, 7, 7, 7, 90, 20, 21, 22]
        return _FakeProcessorOutput(
            input_ids=torch.tensor([ids]),
            pixel_values=torch.zeros(1, 3, 4, 4),
        )


class _FakeVisionConfig:
    image_size = 4
    patch_size = 2


class _FakeModelConfig:
    image_token_id = IMAGE_TOKEN_ID
    vision_config = _FakeVisionConfig()


class _FakeTorchModel:
    config = _FakeModelConfig()


def _make_vlm_with_mocks():
    from hif.models.hf_vlm import HFVLMModel

    vlm = object.__new__(HFVLMModel)
    vlm._processor = _FakeProcessor()
    vlm._tokenizer = _FakeTokenizer()
    vlm._model = _FakeTorchModel()
    vlm._image_token_id = IMAGE_TOKEN_ID
    vlm._image_placeholder = "<image>"
    vlm._device = "cpu"
    return vlm


class TestHFVLMPrepare:
    def test_prepare_builds_part_map(self):
        vlm = _make_vlm_with_mocks()
        prepared = vlm.prepare(_mm_image_text())

        assert prepared.input_ids == [1, 90, 7, 7, 7, 7, 90, 20, 21, 22]
        assert prepared.part_map.seq_len == 10

        spans = {s.kind: s for s in prepared.part_map.spans}
        img = spans["image"]
        assert (img.part_index, img.pos_start, img.pos_end) == (0, 2, 6)
        # Grid derived from vision config: (4/2) = 2 → 2x2 == 4 patches
        assert (img.grid_rows, img.grid_cols) == (2, 2)

        txt = spans["text"]
        assert (txt.part_index, txt.pos_start, txt.pos_end) == (1, 7, 10)

        # Structural/special tokens (BOS at 0, wraps at 1 and 6) are in no span.
        assert prepared.part_map.text_positions() == [7, 8, 9]

        # backend_state carries pixel tensors but is excluded from dumps.
        assert "pixel_values" in prepared.backend_state
        assert "pixel" not in json.dumps(prepared.model_dump()).lower()

    def test_supports_multimodal_input_flags(self):
        vlm = _make_vlm_with_mocks()
        assert vlm.supports_multimodal_input is True
        assert FakeModel().supports_multimodal_input is False


# ---------------------------------------------------------------------------
# Builder guards
# ---------------------------------------------------------------------------


class TestBuilderGuards:
    def test_media_with_non_multimodal_model_raises_before_inference(self):
        from hif.profile.builder import build_profile

        model = FakeModel()
        with pytest.raises(ValueError, match="multimodal"):
            build_profile(
                model,
                _mm_image_text(),
                regime="factual",
                config=_make_run_config(),
                embedder=FakeEmbeddingModel(),
            )
        assert model.forward_calls == 0  # before any inference

    def test_text_generators_rejected_on_multimodal_path(self):
        from hif.profile.builder import _build_profile_mm

        vlm = _make_vlm_with_mocks()
        config = _make_run_config()  # has generators=["synonym"]
        with pytest.raises(ValueError, match="out of scope"):
            _build_profile_mm(
                vlm,
                _mm_image_text(),
                regime="factual",
                config=config,
                embedder=FakeEmbeddingModel(),
            )

    def test_default_text_generators_ignored_with_warning(self, caplog):
        """Ambiguity #1 resolution: an untouched default generators list is
        ignored (warning), not a hard error — only EXPLICIT text generator
        names raise. Proven by getting past the guard to prepare()."""
        import logging

        from hif.config import PerturbationConfig
        from hif.profile.builder import _build_profile_mm

        vlm = _make_vlm_with_mocks()

        class _Sentinel(Exception):
            pass

        def _boom(mm_input):
            raise _Sentinel("reached prepare — guard passed")

        vlm.prepare = _boom
        config = _make_run_config()
        config.perturbation = PerturbationConfig(n_variants=1)  # default generators
        with caplog.at_level(logging.WARNING):
            with pytest.raises(_Sentinel):
                _build_profile_mm(
                    vlm,
                    _mm_image_text(),
                    regime="factual",
                    config=config,
                    embedder=FakeEmbeddingModel(),
                )
        assert "Ignoring default text perturbation generators" in caplog.text

    def test_text_only_multimodal_input_takes_text_path(self):
        from hif.profile.builder import build_profile

        model = FakeModel()
        config = _make_run_config()
        profile_str = build_profile(
            model, "hello world", "factual", config, FakeEmbeddingModel()
        )
        profile_mm = build_profile(
            FakeModel(),
            MultimodalInput.from_text("hello world"),
            "factual",
            config,
            FakeEmbeddingModel(),
        )
        assert profile_mm.prompt.modality == "text"
        assert profile_mm.prompt.prompt_hash == profile_str.prompt.prompt_hash
        assert profile_mm.input_side == profile_str.input_side
        assert profile_mm.output_side == profile_str.output_side


# ---------------------------------------------------------------------------
# Schema compatibility (0.2.0 → 0.3.0) and privacy invariants
# ---------------------------------------------------------------------------


class TestSchemaCompat:
    def test_existing_0_2_0_profile_still_validates(self):
        profile = _make_profile()
        data = profile.model_dump(mode="json")
        # Simulate a 0.2.0 profile on disk: no new fields, old version string.
        data["schema_version"] = "0.2.0"
        data["prompt"].pop("modality")
        data["prompt"].pop("input_parts")
        data.pop("input_part_map")
        data.pop("region_sensitivity")

        loaded = BehavioralRangeProfile.model_validate(data)
        assert loaded.prompt.modality == "text"  # missing modality reads as text
        assert loaded.prompt.input_parts == []
        assert loaded.input_part_map is None
        assert loaded.region_sensitivity is None

    def test_new_default_schema_version_is_0_7_0(self):
        assert _make_profile().schema_version == "0.7.0"

    def test_serialized_profile_contains_no_pixels(self):
        profile = _make_profile()
        profile.prompt.modality = "image+text"
        profile.prompt.input_parts = [
            InputPartRecord(
                kind="image",
                content_hash=hashlib.sha256(_PNG_BYTES).hexdigest(),
                width=1,
                height=1,
                byte_len=len(_PNG_BYTES),
            ),
            InputPartRecord(
                kind="text",
                content_hash=hashlib.sha256(b"what shape?").hexdigest(),
                byte_len=11,
            ),
        ]
        profile.input_part_map = InputPartMap(
            spans=[
                PartSpan(part_index=0, kind="image", pos_start=2, pos_end=6,
                         grid_rows=2, grid_cols=2),
                PartSpan(part_index=1, kind="text", pos_start=7, pos_end=10),
            ],
            seq_len=11,
        )
        raw = profile.model_dump_json().lower()
        for forbidden in ("image_bytes", "base64", "pixel"):
            assert forbidden not in raw, f"{forbidden!r} leaked into profile JSON"
        # Round-trips under the bumped schema.
        loaded = BehavioralRangeProfile.model_validate_json(profile.model_dump_json())
        assert loaded.prompt.modality == "image+text"
        assert loaded.input_part_map.seq_len == 11


# ---------------------------------------------------------------------------
# Surrogate-recovered input-side signals on non-teacher-forcing mm backends
# (the gpt_4o_mm pipeline: API vision model + --surrogate proxy)
# ---------------------------------------------------------------------------


def _png_bytes_8x8() -> bytes:
    import io

    from PIL import Image

    img = Image.new("RGB", (8, 8), (200, 0, 0))
    img.paste((0, 200, 0), (4, 0, 8, 8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mm_image_text_8x8() -> MultimodalInput:
    return MultimodalInput(
        parts=[
            InputPart.from_image_bytes(_png_bytes_8x8()),
            InputPart.from_text("What shape is in this image?"),
        ]
    )


from hif.models.mm import MultimodalModel as _MultimodalModelABC


class _FakeAPIVLM(_MultimodalModelABC):
    """Minimal non-teacher-forcing MultimodalModel (API tier, mirrors
    OpenAIVLMModel's capability surface: generation top-k only, no
    teacher-forced forward, zero-length image spans). A real subclass so the
    builder's isinstance(model, MultimodalModel) contract holds."""

    def __init__(self, vocab_size: int = 50) -> None:
        self._vocab = vocab_size

    @property
    def name(self) -> str:
        return "fake-api-vlm"

    @property
    def vocab_size(self) -> int:
        return self._vocab

    @property
    def context_length(self) -> int:
        return 512

    @property
    def max_top_k(self):
        return 20

    @property
    def supports_teacher_forcing(self) -> bool:
        return False

    @property
    def supports_multimodal_input(self) -> bool:
        return True

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) % self._vocab for c in text] or [0]

    def detokenize(self, ids) -> str:
        return " ".join(str(i) for i in ids)

    def forward(self, input_ids):
        raise NotImplementedError("no teacher forcing (API tier)")

    def generate(self, input_ids, max_new_tokens, top_k, seed):
        raise NotImplementedError("mm path uses generate_prepared")

    def prepare(self, mm_input: MultimodalInput) -> PreparedInput:
        ids: list[int] = []
        spans: list[PartSpan] = []
        for idx, part in enumerate(mm_input.parts):
            if part.kind == "text":
                piece = self.tokenize(part.text or "")
                spans.append(PartSpan(part_index=idx, kind="text",
                                      pos_start=len(ids),
                                      pos_end=len(ids) + len(piece)))
                ids.extend(piece)
            else:
                spans.append(PartSpan(part_index=idx, kind="image",
                                      pos_start=len(ids), pos_end=len(ids)))
        return PreparedInput(
            input_ids=ids,
            part_map=InputPartMap(spans=spans, seq_len=len(ids)),
            backend_state={},
        )

    def forward_prepared(self, prepared):
        raise NotImplementedError("no teacher forcing (API tier)")

    def generate_prepared(self, prepared, max_new_tokens, top_k, seed):
        import numpy as np

        from hif.models.base import GenerationResult, StepRecord, TopKEntry

        steps = [
            StepRecord(
                step=i,
                selected_token_id=i,
                selected_token_str=str(i),
                topk=[
                    TopKEntry(token_id=j, token_str=str(j), logit=0.0,
                              logprob=float(np.log(1.0 / top_k)),
                              prob=1.0 / top_k)
                    for j in range(top_k)
                ],
            )
            for i in range(max_new_tokens)
        ]
        return GenerationResult(
            input_ids=list(prepared.input_ids),
            generated_ids=list(range(max_new_tokens)),
            steps=steps, model_name=self.name, top_k=top_k, seed=seed,
        )


def _mm_run_config():
    from hif.config import (
        GenerationConfig,
        ModelConfig,
        PerturbationConfig,
        RunConfig,
        TrajectoryConfig,
    )

    return RunConfig(
        model=ModelConfig(name="fake-api-vlm", backend="hf"),
        generation=GenerationConfig(max_new_tokens=3, top_k=5),
        trajectory=TrajectoryConfig(n_branches=2, rollout_steps=3),
        # generators left at default (ignored-with-warning on the mm path);
        # 2 media variants so the stability/io_correlation series align.
        perturbation=PerturbationConfig(n_variants=2, image_grid_rows=2,
                                        image_grid_cols=2),
    )


class TestMMSurrogateInputSide:
    def test_surrogate_recovers_input_side_measurements(self):
        """A surrogate recovers the input-side quantities — as PROMPT
        measurements, not as measurements of the target.

        The surrogate teacher-forces the prompt text; nothing the target
        produced enters those numbers, so they are reported under
        prompt_measurements() and are absent from measurements(). The
        quantities the target does participate in stay where they were.
        """
        from hif.profile.builder import build_profile
        from hif.profile.signals import measurements, prompt_measurements

        surrogate = FakeModel(vocab_size=50)
        profile = build_profile(
            _FakeAPIVLM(), _mm_image_text_8x8(), "factual",
            _mm_run_config(), FakeEmbeddingModel(), seed=7,
            surrogate_model=surrogate,
        )
        st = profile.metrics.stability
        assert st.input_entropy_shift_bits is not None  # was absent pre-fix
        assert st.input_output_correlation is not None
        assert profile.input_side.positions          # proxy-read text positions

        vals = measurements(profile)
        prompt_vals = prompt_measurements(profile)
        # Prompt-only: computed from the prompt under the surrogate.
        for key in ("input_entropy_shift_bits", "prompt_surprisal_excess_bits"):
            assert key in prompt_vals, f"{key} missing from prompt_measurements()"
            assert key not in vals, f"{key} reported as a measurement of the target"
        # The target's data is in these, so they stay in the measurement set:
        # perturbation_jsd_bits is the target's own output response, and
        # io_correlation_r couples that response with the surrogate's reading.
        for key in ("io_correlation_r", "perturbation_jsd_bits"):
            assert key in vals, f"{key} missing from measurements()"
        # Provenance: the record must say input-side came from the proxy.
        assert profile.findings.surrogate_model_name == surrogate.name

    def test_without_surrogate_input_side_stays_absent(self):
        from hif.profile.builder import build_profile
        from hif.profile.signals import measurements

        profile = build_profile(
            _FakeAPIVLM(), _mm_image_text_8x8(), "factual",
            _mm_run_config(), FakeEmbeddingModel(), seed=7,
        )
        st = profile.metrics.stability
        assert st.input_entropy_shift_bits is None   # absent, never pinned
        assert st.input_output_correlation is None
        vals = measurements(profile)
        # Absent quantities are OMITTED from the record, not reported as 0.
        assert "input_entropy_shift_bits" not in vals
        assert "io_correlation_r" not in vals
        assert profile.findings.surrogate_model_name is None

    def test_full_access_mm_path_unchanged_ignores_surrogate(self):
        """A teacher-forcing mm backend must keep using ITS OWN input side —
        the surrogate is only a fallback for closed backends."""
        from hif.profile.builder import _build_profile_mm

        vlm = _make_vlm_with_mocks()

        class _Sentinel(Exception):
            pass

        def _boom(mm_input):
            raise _Sentinel("reached prepare")

        vlm.prepare = _boom
        from hif.config import PerturbationConfig
        config = _make_run_config()
        config.perturbation = PerturbationConfig(n_variants=1)

        class _NeverCalledSurrogate:
            @property
            def name(self):
                raise AssertionError("surrogate must not be touched")

        # Reaching prepare() proves routing is identical with a surrogate
        # present (the guard/order before inference did not change).
        with pytest.raises(_Sentinel):
            _build_profile_mm(
                vlm, _mm_image_text(), regime="factual", config=config,
                embedder=FakeEmbeddingModel(),
                surrogate_model=_NeverCalledSurrogate(),
            )
