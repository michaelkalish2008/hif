"""Unit tests for media perturbation families and the region-sensitivity artifact.

Covers: family determinism, original-input immutability, trace recording,
audit/fast sweep semantics, the get_family registry, PerturbationRecord.traces
back-compat, region artifact assembly, and the no-pixels-in-JSON invariant
with traces present (Risk rule 2, docs/ARCHITECTURE.md § Multimodal notes).
"""

from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

from hif.analysis.region_sensitivity import (
    RegionSensitivityResult,
    assemble_region_sensitivity,
)
from hif.models.mm import InputPart, MultimodalInput
from hif.perturbation import (
    ImageBrightnessFamily,
    ImageGridMaskFamily,
    get_family,
)
from hif.perturbation.base import PerturbationTrace
from hif.profile.schema import PerturbationRecord

from tests.unit.profile_helpers import _make_profile, _make_sensitivity


def _png_bytes(size: int = 8) -> bytes:
    # Non-uniform image (left red, right green): mean-color fill and small
    # brightness shifts are guaranteed to actually change pixel bytes.
    img = Image.new("RGB", (size, size), (200, 0, 0))
    img.paste((0, 200, 0), (size // 2, 0, size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mm() -> MultimodalInput:
    return MultimodalInput(
        parts=[
            InputPart.from_image_bytes(_png_bytes()),
            InputPart.from_text("What shape is in this image?"),
        ]
    )


# ---------------------------------------------------------------------------
# ImageGridMaskFamily
# ---------------------------------------------------------------------------


class TestImageGridMaskFamily:
    def test_deterministic_same_seed_same_cells(self):
        fam = ImageGridMaskFamily(grid_rows=4, grid_cols=4)
        a = fam.perturb(_mm(), n_variants=3, seed=42)
        b = fam.perturb(_mm(), n_variants=3, seed=42)
        assert [v.trace.regions for v in a] == [v.trace.regions for v in b]
        assert [v.input.parts[0].content_hash for v in a] == [
            v.input.parts[0].content_hash for v in b
        ]

    def test_different_seed_different_order(self):
        fam = ImageGridMaskFamily(grid_rows=4, grid_cols=4)
        a = fam.perturb(_mm(), n_variants=16, seed=1)
        b = fam.perturb(_mm(), n_variants=16, seed=2)
        assert [v.trace.regions for v in a] != [v.trace.regions for v in b]

    def test_original_never_mutated(self):
        mm = _mm()
        original_hashes = [p.content_hash for p in mm.parts]
        original_bytes = mm.parts[0].image_bytes
        fam = ImageGridMaskFamily(grid_rows=2, grid_cols=2)
        variants = fam.perturb(mm, n_variants=0, seed=42)
        assert [p.content_hash for p in mm.parts] == original_hashes
        assert mm.parts[0].image_bytes == original_bytes
        for v in variants:
            assert v.input is not mm
            assert v.input.parts[0].content_hash != original_hashes[0]
            # Text part is shared unchanged.
            assert v.input.parts[1].content_hash == original_hashes[1]

    def test_audit_sweep_covers_all_cells(self):
        fam = ImageGridMaskFamily(grid_rows=2, grid_cols=2)
        variants = fam.perturb(_mm(), n_variants=0, seed=42)
        cells = {(v.trace.regions[0]["row"], v.trace.regions[0]["col"]) for v in variants}
        assert cells == {(0, 0), (0, 1), (1, 0), (1, 1)}
        # n_variants >= n_cells also sweeps everything.
        variants2 = fam.perturb(_mm(), n_variants=99, seed=42)
        assert len(variants2) == 4

    def test_fast_mode_subset(self):
        fam = ImageGridMaskFamily(grid_rows=4, grid_cols=4)
        variants = fam.perturb(_mm(), n_variants=3, seed=42)
        assert len(variants) == 3

    def test_trace_records_params_and_single_region(self):
        fam = ImageGridMaskFamily(grid_rows=3, grid_cols=5)
        variants = fam.perturb(_mm(), n_variants=2, seed=7)
        for v in variants:
            assert v.trace.family == "image_grid_mask"
            assert v.trace.part_index == 0
            assert len(v.trace.regions) == 1
            assert v.trace.params == {"grid_rows": 3, "grid_cols": 5, "fill": "mean"}

    def test_variants_are_in_memory_bytes_parts(self):
        fam = ImageGridMaskFamily(grid_rows=2, grid_cols=2)
        for v in fam.perturb(_mm(), n_variants=1, seed=42):
            part = v.input.parts[0]
            assert part.image_bytes is not None
            assert part.image_path is None

    def test_mask_changes_only_target_cell(self):
        # Half-red, half-blue 8x8 image; 1x2 grid → masking one cell must
        # leave the other half's pixels untouched.
        img = Image.new("RGB", (8, 8), (255, 0, 0))
        img.paste((0, 0, 255), (4, 0, 8, 8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        mm = MultimodalInput(parts=[InputPart.from_image_bytes(buf.getvalue())])
        fam = ImageGridMaskFamily(grid_rows=1, grid_cols=2)
        variants = fam.perturb(mm, n_variants=0, seed=42)
        by_col = {v.trace.regions[0]["col"]: v for v in variants}
        masked_left = Image.open(io.BytesIO(by_col[0].input.parts[0].image_bytes))
        assert masked_left.getpixel((6, 4)) == (0, 0, 255)  # right half intact
        assert masked_left.getpixel((1, 4)) != (255, 0, 0)  # left half filled

    def test_skips_text_only_input(self):
        fam = ImageGridMaskFamily()
        assert fam.perturb(MultimodalInput.from_text("hi"), 4, 42) == []


class TestImageBrightnessFamily:
    def test_control_variants_no_regions(self):
        fam = ImageBrightnessFamily(delta=0.10)
        variants = fam.perturb(_mm(), n_variants=2, seed=42)
        assert len(variants) == 2
        deltas = sorted(v.trace.params["delta"] for v in variants)
        assert deltas == [-0.10, 0.10]
        for v in variants:
            assert v.trace.regions == []
            assert v.trace.family == "image_brightness"
            assert v.input.parts[0].content_hash != _mm().parts[0].content_hash


class TestFamilyRegistry:
    def test_get_family_resolves(self):
        fam = get_family("image_grid_mask", grid_rows=2, grid_cols=2)
        assert isinstance(fam, ImageGridMaskFamily)
        assert (fam.grid_rows, fam.grid_cols) == (2, 2)
        assert isinstance(get_family("image_brightness"), ImageBrightnessFamily)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown perturbation family"):
            get_family("synonym")  # text generator names never resolve here


# ---------------------------------------------------------------------------
# Schema: PerturbationRecord.traces
# ---------------------------------------------------------------------------


class TestPerturbationRecordTraces:
    def test_traces_default_keeps_text_profiles_valid(self):
        # A record without the field (pre-session-2 JSON) still validates.
        data = {
            "generator": "synonym",
            "variants": ["v1"],
            "sensitivity": [_make_sensitivity().model_dump(mode="json")],
        }
        rec = PerturbationRecord.model_validate(data)
        assert rec.traces == []

    def test_traces_round_trip(self):
        rec = PerturbationRecord(
            generator="image_grid_mask",
            variants=["image_grid_mask[part=0, ...]"],
            sensitivity=[_make_sensitivity()],
            traces=[
                PerturbationTrace(
                    family="image_grid_mask",
                    part_index=0,
                    regions=[{"row": 1, "col": 2}],
                    params={"grid_rows": 4, "grid_cols": 4, "fill": "mean"},
                )
            ],
        )
        loaded = PerturbationRecord.model_validate_json(rec.model_dump_json())
        assert loaded.traces[0].regions == [{"row": 1, "col": 2}]

    def test_profile_json_with_traces_contains_no_pixels(self):
        profile = _make_profile()
        profile.perturbations = [
            PerturbationRecord(
                generator="image_grid_mask",
                variants=["image_grid_mask[part=0, regions=[{'row': 0, 'col': 0}]]"],
                sensitivity=[_make_sensitivity()],
                traces=[
                    PerturbationTrace(
                        family="image_grid_mask",
                        part_index=0,
                        regions=[{"row": 0, "col": 0}],
                        params={"grid_rows": 2, "grid_cols": 2, "fill": "mean"},
                    )
                ],
            )
        ]
        raw = profile.model_dump_json().lower()
        for forbidden in ("image_bytes", "base64", "pixel"):
            assert forbidden not in raw, f"{forbidden!r} leaked into profile JSON"


# ---------------------------------------------------------------------------
# Region-sensitivity artifact
# ---------------------------------------------------------------------------


def _grid_pair(row: int, col: int, jsd: float):
    trace = PerturbationTrace(
        family="image_grid_mask",
        part_index=0,
        regions=[{"row": row, "col": col}],
        params={"grid_rows": 2, "grid_cols": 2, "fill": "mean"},
    )
    return trace, _make_sensitivity(mean_js=jsd)


class TestRegionSensitivity:
    def test_assembly_from_synthetic_pairs(self):
        pairs = [
            _grid_pair(0, 0, 0.02),
            _grid_pair(0, 1, 0.40),
            _grid_pair(1, 0, 0.05),
            _grid_pair(1, 1, 0.10),
        ]
        result = assemble_region_sensitivity(pairs)
        assert isinstance(result, RegionSensitivityResult)
        assert (result.grid_rows, result.grid_cols) == (2, 2)
        assert result.part_index == 0
        assert len(result.cells) == 4
        assert (result.max_cell.row, result.max_cell.col) == (0, 1)
        assert result.max_cell.jsd == pytest.approx(0.40)
        assert result.mean_jsd == pytest.approx((0.02 + 0.40 + 0.05 + 0.10) / 4)

    def test_repeated_cells_averaged(self):
        result = assemble_region_sensitivity(
            [_grid_pair(0, 0, 0.1), _grid_pair(0, 0, 0.3)]
        )
        assert len(result.cells) == 1
        assert result.cells[0].jsd == pytest.approx(0.2)

    def test_none_when_no_grid_pairs(self):
        assert assemble_region_sensitivity([]) is None
        # Brightness (regions=[]) alone contributes nothing.
        trace = PerturbationTrace(
            family="image_brightness", part_index=0, regions=[], params={"delta": 0.1}
        )
        assert assemble_region_sensitivity([(trace, _make_sensitivity())]) is None

    def test_to_text_grid_renders_and_marks_unmeasured(self):
        result = assemble_region_sensitivity(
            [_grid_pair(0, 0, 0.02), _grid_pair(1, 1, 0.40)]
        )
        text = result.to_text_grid()
        assert "2x2 grid" in text
        assert "0.020" in text and "0.400" in text
        assert "." in text  # unmeasured cells
        assert "materially affected the model's response behavior" in text
        # Copy rules: no causal/attention language.
        lowered = text.lower()
        for forbidden in ("attention", "caused", "because", "looks at"):
            assert forbidden not in lowered

    def test_result_serializes_without_pixels(self):
        result = assemble_region_sensitivity([_grid_pair(0, 0, 0.02)])
        raw = result.model_dump_json().lower()
        for forbidden in ("image_bytes", "base64", "pixel"):
            assert forbidden not in raw
