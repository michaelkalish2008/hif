"""CLI multimodal tests: --input validation, absent-measurement rendering,
modality-mismatch hard errors, and `hif validate-model` rendering.

Absent-measurement rule: PerturbationResponse components are Optional — None
means the quantity is not measurable on this backend/run (the backend cannot
teacher-force), and the CLI must say so in words, never print a number.
Measured values print normally on text and multimodal runs alike.
"""

import json

import pytest
from typer.testing import CliRunner

import hif.cli as cli
from hif.cli import ABSENT_TEXT, app
from hif.profile.signals import measurements
from tests.unit.profile_helpers import _make_profile

runner = CliRunner()


@pytest.fixture()
def png(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.new("RGB", (32, 32), "white").save(path)
    return path


def _mm_profile(with_region_sensitivity=True, absent_stability=False):
    from hif.analysis.region_sensitivity import (
        RegionCell,
        RegionSensitivityResult,
    )
    from hif.profile.schema import InputPartRecord

    p = _make_profile()
    p.prompt.modality = "image+text"
    p.prompt.input_parts = [
        InputPartRecord(kind="image", content_hash="a" * 64, width=32, height=32,
                        byte_len=128),
        InputPartRecord(kind="text", content_hash="b" * 64),
    ]
    if absent_stability:
        # Backend that cannot teacher-force: the perturbation-response
        # components are ABSENT (None), never pinned values.
        p.metrics.stability.input_entropy_shift_bits = None
        p.metrics.stability.perturbation_jsd_bits = None
        p.metrics.stability.input_output_correlation = None
        # measurements() falls back to the per-perturbation sensitivity records
        # for the JSD; clear them too so the quantity is genuinely absent.
        p.metrics.sensitivity = []
    if with_region_sensitivity:
        cells = [RegionCell(row=r, col=c, jsd=0.1 * (r + c))
                 for r in range(2) for c in range(2)]
        p.region_sensitivity = RegionSensitivityResult(
            part_index=0, grid_rows=2, grid_cols=2, cells=cells,
            max_cell=max(cells, key=lambda c: c.jsd), mean_jsd=0.2,
        )
    return p


def _patch_pipeline(monkeypatch, profile):
    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_load_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_run_single_profile", lambda *a, **k: (profile, None))


# ---------------------------------------------------------------------------
# --input validation
# ---------------------------------------------------------------------------


def test_input_requires_vlm_backend(png):
    result = runner.invoke(app, ["profile", "m", "p", "--input", str(png)])
    assert result.exit_code == 3
    assert "hf-vlm" in result.output


def test_input_missing_file(tmp_path):
    result = runner.invoke(
        app, ["profile", "m", "p", "--backend", "hf-vlm",
              "--input", str(tmp_path / "nope.png")],
    )
    assert result.exit_code == 3
    assert "not found" in result.output


def test_input_not_an_image(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not pixels")
    result = runner.invoke(
        app, ["profile", "m", "p", "--backend", "hf-vlm", "--input", str(bad)],
    )
    assert result.exit_code == 3
    # normalize whitespace — rich wraps output at terminal width in CI
    assert "not a readable image" in " ".join(result.output.split())


def test_input_rejects_truncate(png):
    result = runner.invoke(
        app, ["profile", "m", "p", "--backend", "hf-vlm",
              "--input", str(png), "--truncate", "5"],
    )
    assert result.exit_code == 3
    assert "--truncate is not supported with --input" in result.output


# ---------------------------------------------------------------------------
# Absent-measurement (None) rendering
# ---------------------------------------------------------------------------

ABSENT_KEYS = [
    "input_entropy_shift_bits",
    "perturbation_jsd_bits",
]


@pytest.mark.parametrize("metric", ABSENT_KEYS)
def test_metric_absent_measurement_exits_1(monkeypatch, png, tmp_path, metric):
    _patch_pipeline(monkeypatch, _mm_profile(absent_stability=True))
    result = runner.invoke(
        app, ["profile", "m", "p", "--backend", "hf-vlm", "--input", str(png),
              "--output-dir", str(tmp_path), "--metric", metric],
    )
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert ABSENT_TEXT in flat
    # Absent is stated, never rendered as a number.
    assert "= 0" not in flat


def test_metric_present_on_multimodal_prints_value(monkeypatch, png, tmp_path):
    _patch_pipeline(monkeypatch, _mm_profile())  # real measured values
    result = runner.invoke(
        app, ["profile", "m", "p", "--backend", "hf-vlm", "--input", str(png),
              "--output-dir", str(tmp_path),
              "--metric", "input_entropy_shift_bits"],
    )
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "input_entropy_shift_bits = 0.4" in flat
    assert "bits" in flat  # the unit always travels with the value


def test_measurement_table_names_absent_quantities(monkeypatch, png, tmp_path):
    _patch_pipeline(monkeypatch, _mm_profile(absent_stability=True))
    result = runner.invoke(
        app, ["profile", "m", "p", "--backend", "hf-vlm", "--input", str(png),
              "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert flat.count("absent") >= 3  # each absent quantity is named, not hidden
    # Region-sensitivity section with the mandated copy line + heatmap
    assert "Region sensitivity" in result.output
    assert "materially affected the model's response behavior" in flat
    assert "0.200" in result.output  # a cell value from the ASCII grid


def test_measurement_table_prints_values_when_measured(monkeypatch, png, tmp_path):
    _patch_pipeline(monkeypatch, _mm_profile())  # real measured values
    result = runner.invoke(
        app, ["profile", "m", "p", "--backend", "hf-vlm", "--input", str(png),
              "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "0.4" in flat  # input_entropy_shift_bits rendered as a number
    # No level column ever: the table reports quantity/value/unit only.
    for banned in ("LOW", "MEDIUM", "HIGH"):
        assert banned not in result.output


def test_measurements_omit_absent_quantities():
    """Absent is OMITTED from the measurement dict, never reported as 0."""
    vals = measurements(_mm_profile(absent_stability=True))
    for key in ABSENT_KEYS:
        assert key not in vals
    # Profiles with measured values keep them (text and multimodal alike)
    assert "input_entropy_shift_bits" in measurements(_make_profile())
    assert "input_entropy_shift_bits" in measurements(_mm_profile())


def test_json_output_has_modality_parts_and_no_pixels(monkeypatch, png, tmp_path):
    _patch_pipeline(monkeypatch, _mm_profile())
    monkeypatch.setattr(cli.console, "width", 100_000)  # no soft-wrap in JSON
    result = runner.invoke(
        app, ["profile", "m", "p", "--backend", "hf-vlm", "--input", str(png),
              "--json", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["modality"] == "image+text"
    assert [p["kind"] for p in data["input_parts"]] == ["image", "text"]
    assert data["region_sensitivity"]["cells"]
    raw = json.dumps(data)
    assert "image_bytes" not in raw
    assert "base64" not in raw
    assert "pixel" not in raw
    # Derived-signals contract: raw per-step distributions never in --json.
    assert "metrics" not in data
    assert "top_k_alternatives" not in raw


# ---------------------------------------------------------------------------
# Modality mismatch (compare + --prior) — exit 2, exact copy
# ---------------------------------------------------------------------------

MISMATCH_COPY = "is a different experimental condition than a"


def test_compare_modality_mismatch_exits_2(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(_make_profile().model_dump_json())
    b.write_text(_mm_profile(with_region_sensitivity=False).model_dump_json())
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 2
    assert (
        "A text profile is a different experimental condition than a "
        "image+text profile. Re-profile both under the same modality to "
        "compare." in " ".join(result.output.split())
    )


def test_compare_missing_modality_backfills_text(tmp_path):
    # Pre-0.3.0-style profiles (no explicit modality) read as "text" == "text"
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    raw = json.loads(_make_profile().model_dump_json())
    raw["prompt"].pop("modality", None)
    a.write_text(json.dumps(raw))
    b.write_text(json.dumps(raw))
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert MISMATCH_COPY not in result.output


# ---------------------------------------------------------------------------
# hif validate-model
# ---------------------------------------------------------------------------


def _fake_validation_result(rate=1.0):
    from hif.validation.harness import ImageValidationRecord, ValidationResult

    recs = [
        ImageValidationRecord(
            image_id="form_01", variant=0, question="q?",
            answer_cell={"row": 1, "col": 1},
            cell_jsd={"0,0": 0.01, "0,1": 0.02, "1,0": 0.03, "1,1": 0.5},
            answer_cell_rank=1, in_top2=True,
        ),
        ImageValidationRecord(
            image_id="chart_01", variant=0, question="q?",
            answer_cell={"row": 0, "col": 0},
            cell_jsd={"0,0": 0.2, "0,1": 0.3, "1,0": 0.03, "1,1": 0.01},
            answer_cell_rank=2, in_top2=True,
        ),
    ]
    return ValidationResult(
        model_id="mock-vlm", grid=(2, 2), top2_rate=rate, per_image=recs,
    )


def test_validate_model_renders_rank_and_separation(monkeypatch, tmp_path):
    import hif.validation.harness as harness

    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        harness, "validate_region_sensitivity",
        lambda *a, **k: _fake_validation_result(),
    )
    result = runner.invoke(app, [
        "validate-model", "mock-vlm", "--backend", "hf-vlm",
        "--pilot", "--corpus", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "form_01" in result.output and "chart_01" in result.output
    assert "Separation" in result.output
    assert "ground-truth synthetic tasks" in result.output.lower()
    # No verdict, and no threshold to compare one against.
    assert "PASS" not in result.output and "FAIL" not in result.output


def test_validate_model_never_exits_nonzero_on_a_low_rate(monkeypatch, tmp_path):
    """A low top-2 rate is a number, not a failure.

    This used to exit 2 below a 70% threshold. Three consecutive gpt-4o pilot
    runs on identical inputs scored 75%, 92% and 100% against that line, every
    difference a rank shuffle on one image whose cells sit within 0.007 bits of
    each other — so the exit code was reporting hosted-API noise as a verdict.
    """
    import hif.validation.harness as harness

    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        harness, "validate_region_sensitivity",
        lambda *a, **k: _fake_validation_result(rate=0.10),
    )
    result = runner.invoke(app, [
        "validate-model", "mock-vlm", "--backend", "hf-vlm",
        "--pilot", "--corpus", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "FAIL" not in result.output


def test_validate_model_json(monkeypatch, tmp_path):
    import hif.validation.harness as harness

    monkeypatch.setattr(cli, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        harness, "validate_region_sensitivity",
        lambda *a, **k: _fake_validation_result(),
    )
    result = runner.invoke(app, [
        "validate-model", "mock-vlm", "--backend", "hf-vlm",
        "--pilot", "--corpus", str(tmp_path), "--json",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["grid"] == "2x2"
    assert "passed" not in data and "threshold" not in data
    assert data["rank1_count"] + data["top2_count"] >= 0
    assert "separation" in data["per_image"][0]
    assert data["per_image"][0]["answer_cell_rank"] == 1


def test_validate_model_bad_backend_exits_3():
    result = runner.invoke(app, ["validate-model", "m", "--backend", "hf"])
    assert result.exit_code == 3


def test_validate_model_bad_grid_exits_3(tmp_path):
    result = runner.invoke(app, [
        "validate-model", "m", "--backend", "hf-vlm", "--grid", "banana",
        "--pilot", "--corpus", str(tmp_path),
    ])
    assert result.exit_code == 3
