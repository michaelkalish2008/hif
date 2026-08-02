"""Media perturbation families for image parts (MULTIMODAL.md § Design §6).

ImageGridMaskFamily — masks one grid cell per variant with the image's mean
color; the region-sensitivity artifact is assembled from the resulting
(trace.regions, SensitivityMetrics) pairs.

ImageBrightnessFamily — a benign whole-image control (small brightness shift,
regions=[]) useful as a baseline against which region-masked variants are
compared.

Privacy invariant: perturbed images live only as in-memory `image_bytes`
InputParts — never written to disk, never serialized (profiles store
InputPartRecord hash+dims only).
"""

from __future__ import annotations

import io
import random

from hif.models.mm import InputPart, MultimodalInput
from hif.perturbation.base import (
    MultimodalVariant,
    PerturbationFamily,
    PerturbationTrace,
)


def _replace_part(
    mm_input: MultimodalInput, part_index: int, new_part: InputPart
) -> MultimodalInput:
    """Return a NEW MultimodalInput with one part swapped; original untouched."""
    parts = list(mm_input.parts)
    parts[part_index] = new_part
    return MultimodalInput(parts=parts)


def _load_image(part: InputPart):
    from PIL import Image

    if part.image_path is not None:
        return Image.open(part.image_path).convert("RGB")
    return Image.open(io.BytesIO(part.image_bytes)).convert("RGB")


def _to_png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class ImageGridMaskFamily(PerturbationFamily):
    """Mask one cell of an r x c grid per variant, mean-color fill.

    Deterministic cell order derived from the seed. Exhaustive sweep (audit
    mode) when n_variants <= 0 or n_variants >= n_cells; otherwise the first
    n_variants cells of the shuffled order (fast mode).
    """

    name = "image_grid_mask"
    supported_kinds = {"image"}

    def __init__(self, grid_rows: int = 4, grid_cols: int = 4) -> None:
        if grid_rows < 1 or grid_cols < 1:
            raise ValueError("grid_rows and grid_cols must be >= 1")
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def perturb(
        self, mm_input: MultimodalInput, n_variants: int, seed: int
    ) -> list[MultimodalVariant]:
        from PIL import ImageStat

        variants: list[MultimodalVariant] = []
        for part_index, part in enumerate(mm_input.parts):
            if part.kind not in self.supported_kinds:
                continue
            img = _load_image(part)
            width, height = img.size
            mean_color = tuple(
                int(round(v)) for v in ImageStat.Stat(img).mean[:3]
            )

            cells = [
                (r, c)
                for r in range(self.grid_rows)
                for c in range(self.grid_cols)
            ]
            rng = random.Random(seed + part_index)
            rng.shuffle(cells)
            n_cells = len(cells)
            if n_variants <= 0 or n_variants >= n_cells:
                chosen = cells  # audit: full sweep
            else:
                chosen = cells[:n_variants]  # fast: subset

            for row, col in chosen:
                x0 = col * width // self.grid_cols
                x1 = (col + 1) * width // self.grid_cols
                y0 = row * height // self.grid_rows
                y1 = (row + 1) * height // self.grid_rows
                masked = img.copy()
                masked.paste(mean_color, (x0, y0, x1, y1))
                new_part = InputPart.from_image_bytes(_to_png_bytes(masked))
                variants.append(
                    MultimodalVariant(
                        input=_replace_part(mm_input, part_index, new_part),
                        trace=PerturbationTrace(
                            family=self.name,
                            part_index=part_index,
                            regions=[{"row": row, "col": col}],
                            params={
                                "grid_rows": self.grid_rows,
                                "grid_cols": self.grid_cols,
                                "fill": "mean",
                            },
                        ),
                    )
                )
        return variants


class ImageBrightnessFamily(PerturbationFamily):
    """Whole-image +/- brightness control variants (regions=[])."""

    name = "image_brightness"
    supported_kinds = {"image"}

    def __init__(self, delta: float = 0.10) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.delta = delta

    def perturb(
        self, mm_input: MultimodalInput, n_variants: int, seed: int
    ) -> list[MultimodalVariant]:
        from PIL import ImageEnhance

        variants: list[MultimodalVariant] = []
        for part_index, part in enumerate(mm_input.parts):
            if part.kind not in self.supported_kinds:
                continue
            img = _load_image(part)
            deltas = [+self.delta, -self.delta]
            if n_variants > 0:
                deltas = deltas[:n_variants]
            for d in deltas:
                adjusted = ImageEnhance.Brightness(img).enhance(1.0 + d)
                new_part = InputPart.from_image_bytes(_to_png_bytes(adjusted))
                variants.append(
                    MultimodalVariant(
                        input=_replace_part(mm_input, part_index, new_part),
                        trace=PerturbationTrace(
                            family=self.name,
                            part_index=part_index,
                            regions=[],
                            params={"delta": d},
                        ),
                    )
                )
        return variants
