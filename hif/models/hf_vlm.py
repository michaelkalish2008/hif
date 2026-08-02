"""HuggingFace vision-language backend implementing MultimodalModel (M1).

Generic, processor-driven adapter: the reference model is a config choice
(ModelConfig.name), not a code fork. Verified against
HuggingFaceTB/SmolVLM-256M-Instruct; designed for any AutoProcessor +
AutoModelForImageTextToText checkpoint (e.g. Gemma 3 multimodal).

Mirrors HFModel's logits-extraction and top-K sampling patterns
(hif/models/hf.py); text-only tokenize/forward/generate keep the exact
Model signatures and never see media.
"""

from __future__ import annotations

import io
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from hif.config import ModelConfig
from hif.models.base import GenerationResult, Logits, StepRecord, TopKEntry
from hif.models.hf import _DTYPE_MAP, _resolve_device
from hif.models.mm import (
    InputPartMap,
    MultimodalInput,
    MultimodalModel,
    PartSpan,
    PreparedInput,
)
from hif.utils.logging import get_logger

logger = get_logger(__name__)


def _find_subsequence(haystack: list[int], needle: list[int], start: int) -> int:
    """Return the first index >= start where needle occurs in haystack, else -1."""
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    for i in range(start, n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1


class HFVLMModel(MultimodalModel):
    """HF backend wrapping AutoProcessor + AutoModelForImageTextToText."""

    def __init__(self, config: ModelConfig) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._config = config
        self._device = _resolve_device(config.device)
        self._dtype = _DTYPE_MAP.get(config.dtype, torch.float32)

        logger.info(f"Loading processor: {config.name}"
                    + (f" @ {config.revision}" if config.revision else ""))
        self._processor = AutoProcessor.from_pretrained(
            config.name, revision=config.revision
        )
        self._tokenizer = self._processor.tokenizer
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Disable tiling/splitting where supported so each image maps to one
        # contiguous placeholder run with a single deterministic patch grid.
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = False

        logger.info(
            f"Loading VLM: {config.name} | device={self._device} | dtype={config.dtype}"
            + (f" | revision={config.revision}" if config.revision else "")
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            config.name,
            revision=config.revision,
            torch_dtype=self._dtype,
        )
        self._model.eval()
        self._model.to(self._device)

        self._image_token_id = self._resolve_image_token_id()
        placeholder = getattr(self._processor, "image_token", None)
        if placeholder is None and self._image_token_id is not None:
            placeholder = self._tokenizer.convert_ids_to_tokens(self._image_token_id)
        self._image_placeholder = str(placeholder) if placeholder is not None else "<image>"
        logger.info(f"VLM ready: {config.name}")

    def _resolve_image_token_id(self) -> Optional[int]:
        cfg = self._model.config
        for attr in ("image_token_id", "image_token_index"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int):
                return val
        token = getattr(self._processor, "image_token", None)
        if token is not None:
            tid = self._tokenizer.convert_tokens_to_ids(str(token))
            if isinstance(tid, int) and tid >= 0:
                return tid
        return None

    # --- Properties ---

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def context_length(self) -> int:
        cfg = self._model.config
        text_cfg = getattr(cfg, "text_config", None)
        for candidate in (text_cfg, cfg):
            if candidate is not None:
                mpe = getattr(candidate, "max_position_embeddings", None)
                if isinstance(mpe, int):
                    return mpe
        return int(self._tokenizer.model_max_length)

    @property
    def max_top_k(self) -> Optional[int]:
        return None

    @property
    def supports_teacher_forcing(self) -> bool:
        return True

    # --- Text-only interface (Model signatures, never called with media) ---

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=True)

    def detokenize(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def forward(self, input_ids: list[int]) -> Logits:
        return self._forward_ids(input_ids, backend_state=None)

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        top_k: int,
        seed: int,
    ) -> GenerationResult:
        return self._generate_loop(
            input_ids, backend_state=None,
            max_new_tokens=max_new_tokens, top_k=top_k, seed=seed,
        )

    # --- Multimodal interface ---

    def _build_prompt_text(self, mm_input: MultimodalInput) -> str:
        """Chat-formatted prompt via the processor's chat template.

        The adapter owns prompt construction: parts become one user turn with
        interleaved image/text content, rendered by
        processor.apply_chat_template (add_generation_prompt=True) so the
        image placeholder lands where the template puts it. Falls back to raw
        concatenation (pilot behavior) when no chat template is available.

        Template/structural tokens are excluded from part spans automatically:
        text spans are located by positive subsequence match of each part's
        own tokens, image spans by placeholder-id runs (same rule as before).
        """
        has_template = getattr(self._tokenizer, "chat_template", None) or getattr(
            self._processor, "chat_template", None
        )
        if hasattr(self._processor, "apply_chat_template") and has_template:
            content: list[dict[str, Any]] = []
            for part in mm_input.parts:
                if part.kind == "text":
                    content.append({"type": "text", "text": part.text or ""})
                else:
                    content.append({"type": "image"})
            messages = [{"role": "user", "content": content}]
            return self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        # Fallback: raw concatenation with explicit placeholders.
        chunks = [
            (p.text or "") if p.kind == "text" else self._image_placeholder
            for p in mm_input.parts
        ]
        return "".join(chunks)

    def prepare(self, mm_input: MultimodalInput) -> PreparedInput:
        """Run the processor on interleaved parts and build the part map.

        Text spans are located positively (exact token-subsequence match per
        text part); positions that cannot be attributed with certainty are
        left out of every span (MULTIMODAL.md Risk rule 3: if unsure whether
        a position is text, exclude it). Structural/chat-template tokens are
        therefore never inside any part span.
        """
        from PIL import Image

        images: list[Any] = []
        for part in mm_input.parts:
            if part.kind == "image":
                if part.image_path is not None:
                    img = Image.open(part.image_path).convert("RGB")
                else:
                    img = Image.open(io.BytesIO(part.image_bytes)).convert("RGB")
                images.append(img)

        prompt_text = self._build_prompt_text(mm_input)
        if images:
            inputs = self._processor(
                text=prompt_text, images=images, return_tensors="pt"
            )
        else:
            inputs = self._processor(text=prompt_text, return_tensors="pt")

        input_ids: list[int] = inputs["input_ids"][0].tolist()
        seq_len = len(input_ids)

        backend_state = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask")
        }

        spans: list[PartSpan] = []

        # Image spans: contiguous runs of the image placeholder id, assigned
        # to image parts in order.
        image_part_indices = [
            i for i, p in enumerate(mm_input.parts) if p.kind == "image"
        ]
        image_runs: list[tuple[int, int]] = []
        if self._image_token_id is not None:
            run_start: Optional[int] = None
            for pos, tid in enumerate(input_ids + [None]):  # sentinel flush
                if tid == self._image_token_id:
                    if run_start is None:
                        run_start = pos
                elif run_start is not None:
                    image_runs.append((run_start, pos))
                    run_start = None
        if len(image_runs) != len(image_part_indices):
            raise ValueError(
                f"Image span mismatch: found {len(image_runs)} placeholder runs "
                f"for {len(image_part_indices)} image parts in {self.name}."
            )
        for (pos_start, pos_end), part_index in zip(image_runs, image_part_indices):
            n_patches = pos_end - pos_start
            grid_rows, grid_cols = self._derive_grid(n_patches)
            spans.append(
                PartSpan(
                    part_index=part_index,
                    kind="image",
                    pos_start=pos_start,
                    pos_end=pos_end,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                )
            )

        # Text spans: exact subsequence match, in part order, left to right.
        cursor = 0
        for part_index, part in enumerate(mm_input.parts):
            if part.kind != "text":
                continue
            piece_ids = self._tokenizer.encode(part.text, add_special_tokens=False)
            found = -1
            needle: list[int] = []
            # Leading-whitespace/merge variants: retry without the first token.
            for candidate in (piece_ids, piece_ids[1:]):
                if candidate:
                    found = _find_subsequence(input_ids, candidate, cursor)
                    if found >= 0:
                        needle = candidate
                        break
            if found < 0:
                logger.warning(
                    "Could not locate text part %d in the processed sequence; "
                    "excluding its positions from the part map (rule: if "
                    "unsure, exclude).",
                    part_index,
                )
                continue
            spans.append(
                PartSpan(
                    part_index=part_index,
                    kind="text",
                    pos_start=found,
                    pos_end=found + len(needle),
                )
            )
            cursor = found + len(needle)

        spans.sort(key=lambda s: s.pos_start)
        # Verify span arithmetic against the processed sequence.
        prev_end = 0
        for span in spans:
            if span.pos_start < prev_end or span.pos_end > seq_len:
                raise ValueError(
                    f"Invalid part span [{span.pos_start},{span.pos_end}) for "
                    f"seq_len={seq_len} in {self.name}."
                )
            if span.kind == "image" and (
                span.grid_rows * span.grid_cols != span.pos_end - span.pos_start
            ):
                raise ValueError(
                    f"Patch grid {span.grid_rows}x{span.grid_cols} does not "
                    f"match span length {span.pos_end - span.pos_start}."
                )
            prev_end = span.pos_end

        return PreparedInput(
            input_ids=input_ids,
            part_map=InputPartMap(spans=spans, seq_len=seq_len),
            backend_state=backend_state,
        )

    def _derive_grid(self, n_patches: int) -> tuple[int, int]:
        """Derive (grid_rows, grid_cols) for a placeholder run of n_patches.

        Row-major order is guaranteed by HF vision towers (patches are
        flattened row-major). Derivation: side = (image_size / patch_size) /
        pixel-shuffle scale factor from the model config; verified against
        n_patches, with a square-root fallback.
        """
        cfg = self._model.config
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None:
            image_size = getattr(vision_cfg, "image_size", None)
            patch_size = getattr(vision_cfg, "patch_size", None)
            if isinstance(image_size, int) and isinstance(patch_size, int) and patch_size:
                side = image_size // patch_size
                scale = getattr(cfg, "scale_factor", None)
                if isinstance(scale, int) and scale > 0:
                    side = side // scale
                if side > 0 and side * side == n_patches:
                    return side, side
        root = math.isqrt(n_patches)
        if root * root == n_patches:
            return root, root
        logger.warning(
            "Could not derive a square patch grid for %d placeholder tokens; "
            "falling back to 1x%d.", n_patches, n_patches,
        )
        return 1, n_patches

    def forward_prepared(self, prepared: PreparedInput) -> Logits:
        return self._forward_ids(prepared.input_ids, prepared.backend_state)

    def generate_prepared(
        self,
        prepared: PreparedInput,
        max_new_tokens: int,
        top_k: int,
        seed: int,
    ) -> GenerationResult:
        return self._generate_loop(
            prepared.input_ids,
            prepared.backend_state,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            seed=seed,
        )

    # --- Shared internals (mirror HFModel's logits extraction) ---

    def _model_kwargs(self, backend_state: Any) -> dict[str, Any]:
        if not backend_state:
            return {}
        kwargs = {}
        for k, v in backend_state.items():
            kwargs[k] = v.to(self._device) if hasattr(v, "to") else v
        return kwargs

    def _forward_ids(self, input_ids: list[int], backend_state: Any) -> Logits:
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                **self._model_kwargs(backend_state),
            )
        logits = outputs.logits[0]
        seq_len, vocab_size = logits.shape
        return Logits(
            values=logits.float().cpu().tolist(),
            seq_len=seq_len,
            vocab_size=vocab_size,
        )

    def _generate_loop(
        self,
        input_ids: list[int],
        backend_state: Any,
        max_new_tokens: int,
        top_k: int,
        seed: int,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Sampling loop with KV caching.

        With use_cache=True (default) the full sequence (plus media state) is
        forwarded once; each subsequent step forwards only the newly sampled
        token against past_key_values. use_cache=False re-forwards the whole
        sequence every step (the pre-cache path, kept for equivalence tests).
        """
        eos_token_id = self._tokenizer.eos_token_id
        running_ids: list[int] = list(input_ids)
        steps: list[StepRecord] = []
        model_kwargs = self._model_kwargs(backend_state)
        past_key_values = None

        for step in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                # Incremental step: only the last sampled token; media state
                # (pixel_values etc.) was consumed by the first forward.
                input_tensor = torch.tensor(
                    [running_ids[-1:]], dtype=torch.long, device=self._device
                )
                step_kwargs: dict[str, Any] = {}
            else:
                input_tensor = torch.tensor(
                    [running_ids], dtype=torch.long, device=self._device
                )
                step_kwargs = model_kwargs
            attention_mask = torch.ones(
                (1, len(running_ids)), dtype=torch.long, device=self._device
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **step_kwargs,
                )
            if use_cache:
                past_key_values = outputs.past_key_values

            last_logits = outputs.logits[0, -1, :].float()
            logprobs = F.log_softmax(last_logits, dim=-1)
            probs = torch.exp(logprobs)

            k = min(top_k, last_logits.shape[0])
            topk_logits, topk_ids = torch.topk(last_logits, k)

            topk_entries = [
                TopKEntry(
                    token_id=int(topk_ids[i].item()),
                    token_str=self._tokenizer.decode([int(topk_ids[i].item())]),
                    logit=float(topk_logits[i].item()),
                    logprob=float(logprobs[topk_ids[i]].item()),
                    prob=float(probs[topk_ids[i]].item()),
                )
                for i in range(k)
            ]

            # Temperature applies to SAMPLING only — recorded logprobs/topk stay
            # raw (hosted-API semantics). None/1.0 = unchanged behavior.
            temp = self._config.temperature
            if temp is not None and temp != 1.0 and temp > 0:
                sample_probs = F.softmax(last_logits / temp, dim=-1)
            else:
                sample_probs = probs
            step_seed = seed + step
            generator = torch.Generator(device=self._device)
            generator.manual_seed(step_seed)
            selected_idx = torch.multinomial(sample_probs, num_samples=1, generator=generator)
            selected_token_id = int(selected_idx.item())
            selected_token_str = self._tokenizer.decode([selected_token_id])

            steps.append(
                StepRecord(
                    step=step,
                    selected_token_id=selected_token_id,
                    selected_token_str=selected_token_str,
                    topk=topk_entries,
                )
            )

            running_ids.append(selected_token_id)

            if eos_token_id is not None and selected_token_id == eos_token_id:
                break

        generated_ids = running_ids[len(input_ids):]

        return GenerationResult(
            input_ids=list(input_ids),
            generated_ids=generated_ids,
            steps=steps,
            model_name=self.name,
            top_k=top_k,
            seed=seed,
        )
