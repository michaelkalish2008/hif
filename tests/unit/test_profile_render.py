"""Unit tests for JSON and Markdown renderers."""

from __future__ import annotations

import json
import re

from hif.profile.render_json import render_json
from hif.profile.render_markdown import render_markdown, render_technical
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

        assert data["schema_version"] == "0.15.0"
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


def _table_rows_are_rectangular(content: str) -> list[str]:
    """Rows whose unescaped-pipe count disagrees with their header's.

    A cell cannot contain a bare `|`. Registry prose is written in maths —
    `input_entropy_shift_bits` defines itself as `mean |...|` — so an
    unescaped interpolation turns one row into five columns under a
    four-column header, and every renderer of that Markdown mis-parses it.
    """
    bad: list[str] = []
    in_code = False
    header: int | None = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            header = None
            continue
        if in_code or not stripped.startswith("|"):
            header = None
            continue
        if re.fullmatch(r"\|[\s:|-]+\|", stripped) and "-" in stripped:
            continue
        n = len(re.split(r"(?<!\\)\|", stripped)) - 2  # drop leading/trailing
        if header is None:
            header = n
        elif n != header:
            bad.append(stripped)
    return bad


class TestRenderMarkdownCreatesFiles:
    def test_technical_creates_nonempty_file(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "report.md"
        render_technical(profile, out)
        assert out.exists()
        assert len(out.read_text()) > 100

    def test_technical_contains_model_name(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "report.md"
        render_technical(profile, out)
        assert "mock-model" in out.read_text()

    def test_technical_tables_are_rectangular(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "report.md"
        render_technical(profile, out)
        assert _table_rows_are_rectangular(out.read_text()) == []

    def test_technical_renders_no_levels_or_verdict(self, tmp_path):
        """The dropped public summary used to map each metric to a
        low/medium/high paragraph. That is a judgement the instrument cannot
        support, and it must not come back — nor may the scope paragraph it
        was carrying be lost with it."""
        profile = _make_profile()
        out = tmp_path / "report.md"
        render_technical(profile, out)
        content = out.read_text()
        for banned in ("LOW", "MEDIUM", "HIGH"):
            assert banned not in content
        assert "not a drift detection" in content

    def test_render_markdown_shim_technical(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "tech.md"
        render_markdown(profile, out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "nested" / "dir" / "report.md"
        render_technical(profile, out)
        assert out.exists()


class TestDistributionTableTokenColumn:
    """The per-step numbers are about a token; the table has to name it."""

    def test_token_column_labels_each_step(self, tmp_path):
        profile = _make_profile()
        out = tmp_path / "report.md"
        render_technical(profile, out)
        content = out.read_text()
        assert (
            "| Token | Step | Entropy (bits) | Logit margin | Top-K mass | "
            "Nucleus eff. support | Tail weight |"
        ) in content
        tok = profile.output_side.steps[0].selected_token_str
        assert f"| `{tok!r}` | 0 |" in content

    def test_token_withheld_when_rows_are_the_surrogates_segmentation(self, tmp_path):
        """Under output-distribution recovery the rows are the surrogate's
        positions in the surrogate's own tokenization, so the target's token
        *i* is not row *i*'s token. Better an empty column than a wrong one."""
        profile = _make_profile()
        profile.findings.output_distribution_surrogate_name = "gpt2"
        out = tmp_path / "report.md"
        render_technical(profile, out)
        content = out.read_text()
        tok = profile.output_side.steps[0].selected_token_str
        assert f"`{tok!r}`" not in content
        assert "Tokens are not shown" in content

    def test_token_cell_survives_a_pipe_token(self, tmp_path):
        """A model that emits `|` must not turn one row into two columns."""
        profile = _make_profile()
        profile.output_side.steps[0].selected_token_str = "|"
        out = tmp_path / "report.md"
        render_technical(profile, out)
        assert _table_rows_are_rectangular(out.read_text()) == []
