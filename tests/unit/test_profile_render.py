"""Unit tests for JSON and Markdown renderers."""

from __future__ import annotations

import json

from hif.profile.render_json import render_json
from hif.profile.render_markdown import render_markdown, render_public, render_technical
from hif.profile.schema import BehavioralRangeProfile

from profile_helpers import _make_profile


class TestRenderJsonRoundtrip:
    def test_roundtrip(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "profile.json"
        render_json(profile, out)

        assert out.exists()
        raw = out.read_text()
        data = json.loads(raw)

        assert data["schema_version"] == "0.10.0"
        assert data["model"]["name"] == "mock-model"
        assert data["prompt"]["text"] == "hello world"

        reparsed = BehavioralRangeProfile.model_validate_json(raw)
        assert reparsed.model.name == profile.model.name
        assert reparsed.prompt.text == profile.prompt.text
        assert (
            reparsed.findings.similarity_trend_slope
            == profile.findings.similarity_trend_slope
        )
        assert (
            reparsed.findings.surrogate_model_name
            == profile.findings.surrogate_model_name
        )
        # Findings carries provenance only — no level, no verdict, no summary.
        assert "stability_level" not in data["findings"]
        assert "summary" not in data["findings"]

    def test_json_carries_no_normalized_or_levels_block(self, tmp_path):
        """The removed normalisation and level blocks must not reappear
        anywhere in a rendered artifact."""
        profile = _make_profile()
        out = tmp_path / "profile.json"
        render_json(profile, out)
        raw = out.read_text()
        assert '"normalized"' not in raw
        assert '"levels"' not in raw
        assert '"findings_levels"' not in raw

    def test_creates_parent_dirs(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "a" / "b" / "c" / "profile.json"
        render_json(profile, out)
        assert out.exists()

    def test_json_is_valid(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "profile.json"
        render_json(profile, out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)


class TestRenderMarkdownCreatesFiles:
    def test_technical_creates_nonempty_file(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "report.md"
        render_technical(profile, out)
        assert out.exists()
        assert len(out.read_text()) > 100

    def test_public_creates_nonempty_file(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "summary.md"
        render_public(profile, out)
        assert out.exists()
        assert len(out.read_text()) > 50

    def test_technical_contains_model_name(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "report.md"
        render_technical(profile, out)
        assert "mock-model" in out.read_text()

    def test_public_reports_measurements_with_units(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "summary.md"
        render_public(profile, out)
        content = out.read_text()
        assert profile.model.name in content
        assert "What was measured" in content
        # Every measurement row carries its unit — no bare numbers.
        assert "| Measurement | Value | Unit |" in content
        assert "bits" in content

    def test_public_renders_no_levels_or_verdict(self, tmp_path):
        """The public summary used to map each metric to a low/medium/high
        paragraph. That is a judgement the instrument cannot support, and it
        must not come back."""
        profile = _make_profile()
        out = tmp_path / "summary.md"
        render_public(profile, out)
        content = out.read_text()
        for banned in ("LOW", "MEDIUM", "HIGH"):
            assert banned not in content
        assert "not a drift detection" in content

    def test_render_markdown_shim_technical(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "tech.md"
        render_markdown(profile, out, public=False)
        assert out.exists()

    def test_render_markdown_shim_public(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "pub.md"
        render_markdown(profile, out, public=True)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "nested" / "dir" / "report.md"
        render_technical(profile, out)
        assert out.exists()
